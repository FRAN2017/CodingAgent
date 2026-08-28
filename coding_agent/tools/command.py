"""Controlled local command execution for workspace verification."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from coding_agent.tools.base import ToolSpec
from coding_agent.tools.workspace import (
    is_ignored_name,
    resolve_workspace_path,
    workspace_relative_path,
)

MAX_COMMAND_ARGS = 128
MAX_COMMAND_TEXT_CHARS = 32_000
MAX_COMMAND_TIMEOUT_SECONDS = 120
DEFAULT_COMMAND_OUTPUT_CHARS = 12_000
MAX_COMMAND_OUTPUT_CHARS = 40_000
DANGEROUS_EXECUTABLES = {
    "bash",
    "cmd",
    "del",
    "diskpart",
    "doas",
    "erase",
    "fdisk",
    "fish",
    "format",
    "halt",
    "mkfs",
    "parted",
    "poweroff",
    "powershell",
    "pwsh",
    "reboot",
    "reg",
    "regedit",
    "rm",
    "rmdir",
    "runas",
    "sh",
    "shutdown",
    "su",
    "sudo",
    "wsl",
    "zsh",
}
DANGEROUS_GIT_SUBCOMMANDS = {
    "checkout",
    "clean",
    "reset",
    "restore",
    "switch",
}
SENSITIVE_ENV_SUFFIXES = (
    "_API_KEY",
    "_CREDENTIAL",
    "_PASSWORD",
    "_SECRET",
    "_TOKEN",
)


class RunCommandInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    argv: list[str] = Field(
        min_length=1,
        max_length=MAX_COMMAND_ARGS,
        description=(
            "Command and arguments as an array, for example "
            "['python', 'hello.py']. Shell syntax is not supported."
        ),
    )
    cwd: str = Field(
        default=".",
        min_length=1,
        description="Workspace-relative working directory",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=MAX_COMMAND_TIMEOUT_SECONDS,
        description="Command timeout in seconds",
    )
    max_output_chars: int = Field(
        default=DEFAULT_COMMAND_OUTPUT_CHARS,
        ge=200,
        le=MAX_COMMAND_OUTPUT_CHARS,
        description="Maximum characters retained for each output stream",
    )

    @model_validator(mode="after")
    def validate_argv(self) -> RunCommandInput:
        if not self.argv[0].strip():
            raise ValueError("argv[0] must contain an executable name")
        if any("\x00" in argument for argument in self.argv):
            raise ValueError("Command arguments must not contain null bytes")
        if sum(len(argument) for argument in self.argv) > MAX_COMMAND_TEXT_CHARS:
            raise ValueError(
                f"Combined command arguments exceed {MAX_COMMAND_TEXT_CHARS} characters"
            )
        return self


def _executable_name(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    for suffix in (".exe", ".com", ".bat", ".cmd", ".ps1"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _command_rejection_reason(argv: list[str]) -> str | None:
    executable = _executable_name(argv[0])
    if executable in DANGEROUS_EXECUTABLES:
        return f"Executable is blocked by command policy: {argv[0]}"
    if executable == "git" and len(argv) > 1:
        subcommand = argv[1].casefold()
        if subcommand in DANGEROUS_GIT_SUBCOMMANDS:
            return f"Destructive git subcommand is blocked: git {argv[1]}"
    return None


def _sanitized_subprocess_environment() -> dict[str, str]:
    environment = {}
    for name, value in os.environ.items():
        upper_name = name.upper()
        if upper_name.endswith(SENSITIVE_ENV_SUFFIXES):
            continue
        environment[name] = value
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    interpreter_directory = str(Path(sys.executable).resolve().parent)
    current_path = environment.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if interpreter_directory.casefold() not in {
        entry.casefold() for entry in path_entries
    }:
        environment["PATH"] = (
            interpreter_directory
            if not current_path
            else interpreter_directory + os.pathsep + current_path
        )
    return environment


def _normalize_process_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _truncate_process_output(output: str, limit: int) -> tuple[str, bool]:
    if len(output) <= limit:
        return output, False

    marker = "\n... output truncated ...\n"
    remaining = limit - len(marker)
    head_length = remaining // 2
    tail_length = remaining - head_length
    return output[:head_length] + marker + output[-tail_length:], True


def run_command(workspace: Path, arguments: RunCommandInput) -> dict[str, Any]:
    workspace = workspace.resolve()
    cwd = resolve_workspace_path(workspace, arguments.cwd)
    relative_cwd = cwd.relative_to(workspace)

    if any(is_ignored_name(part) for part in relative_cwd.parts):
        return {
            "ok": False,
            "error_type": "invalid_working_directory",
            "error": f"Running commands in ignored paths is not allowed: {arguments.cwd}",
        }
    if not cwd.exists():
        return {
            "ok": False,
            "error_type": "invalid_working_directory",
            "error": f"Working directory does not exist: {arguments.cwd}",
        }
    if not cwd.is_dir():
        return {
            "ok": False,
            "error_type": "invalid_working_directory",
            "error": f"Working directory is not a directory: {arguments.cwd}",
        }

    rejection_reason = _command_rejection_reason(arguments.argv)
    if rejection_reason is not None:
        return {
            "ok": False,
            "error_type": "command_rejected",
            "error": rejection_reason,
            "argv": arguments.argv,
            "cwd": workspace_relative_path(workspace, cwd),
        }

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            arguments.argv,
            cwd=cwd,
            env=_sanitized_subprocess_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=arguments.timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return {
            "ok": False,
            "error_type": "command_not_found",
            "error": f"Command was not found: {arguments.argv[0]}",
            "argv": arguments.argv,
            "cwd": workspace_relative_path(workspace, cwd),
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "truncated": False,
            "duration_ms": round((time.perf_counter() - started) * 1000),
        }
    except PermissionError:
        return {
            "ok": False,
            "error_type": "permission_denied",
            "error": f"Permission denied while starting: {arguments.argv[0]}",
            "argv": arguments.argv,
            "cwd": workspace_relative_path(workspace, cwd),
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "truncated": False,
            "duration_ms": round((time.perf_counter() - started) * 1000),
        }
    except subprocess.TimeoutExpired as exc:
        stdout, stdout_truncated = _truncate_process_output(
            _normalize_process_output(exc.stdout), arguments.max_output_chars
        )
        stderr, stderr_truncated = _truncate_process_output(
            _normalize_process_output(exc.stderr), arguments.max_output_chars
        )
        return {
            "ok": False,
            "error_type": "timeout",
            "error": f"Command exceeded {arguments.timeout_seconds} seconds",
            "argv": arguments.argv,
            "cwd": workspace_relative_path(workspace, cwd),
            "exit_code": None,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": True,
            "truncated": stdout_truncated or stderr_truncated,
            "duration_ms": round((time.perf_counter() - started) * 1000),
        }

    stdout, stdout_truncated = _truncate_process_output(
        completed.stdout, arguments.max_output_chars
    )
    stderr, stderr_truncated = _truncate_process_output(
        completed.stderr, arguments.max_output_chars
    )
    succeeded = completed.returncode == 0
    result = {
        "ok": succeeded,
        "argv": arguments.argv,
        "cwd": workspace_relative_path(workspace, cwd),
        "exit_code": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": False,
        "truncated": stdout_truncated or stderr_truncated,
        "duration_ms": round((time.perf_counter() - started) * 1000),
    }
    if not succeeded:
        result["error_type"] = "nonzero_exit"
        result["error"] = f"Command exited with code {completed.returncode}"
    return result


RUN_COMMAND_TOOL = ToolSpec(
    name="run_command",
    description=(
        "Run a local command inside the workspace without a shell. Use it "
        "to execute programs, tests, compilers, linters, and formatters. "
        "Returns the exit code plus captured stdout and stderr."
    ),
    input_model=RunCommandInput,
    handler=run_command,
)
