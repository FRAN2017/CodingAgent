"""Local tools exposed to the model."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

MAX_FILE_BYTES = 256 * 1024
MAX_WRITE_BYTES = 256 * 1024
MAX_RETURN_CHARS = 40_000
BLOCKED_NAMES = {".env"}
IGNORED_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "venv",
}
MAX_LIST_DEPTH = 10
MAX_LIST_ENTRIES = 1_000


class ReadFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="Workspace-relative file path")
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_line_range(self) -> ReadFileInput:
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ListFilesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        default=".",
        min_length=1,
        description="Workspace-relative directory path",
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=MAX_LIST_DEPTH,
        description="Maximum number of directory levels to include",
    )
    limit: int = Field(
        default=200,
        ge=1,
        le=MAX_LIST_ENTRIES,
        description="Maximum number of entries to return",
    )


class WriteFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="Workspace-relative file path")
    content: str = Field(description="Complete UTF-8 text content to write")
    overwrite: bool = Field(
        default=False,
        description="Must be true to replace an existing file",
    )
    create_parent_dirs: bool = Field(
        default=False,
        description="Create missing parent directories when true",
    )


def read_file_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read selected lines from a UTF-8 text file inside the workspace. "
                "Line numbers are included in the result."
            ),
            "parameters": ReadFileInput.model_json_schema(),
        },
    }


def list_files_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and directories inside the workspace in stable path "
                "order. Generated, dependency, cache, version-control, and secret "
                "paths are omitted."
            ),
            "parameters": ListFilesInput.model_json_schema(),
        },
    }


def write_file_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a UTF-8 text file inside the workspace or replace an "
                "existing file when overwrite is explicitly true. The content "
                "argument must contain the complete desired file contents."
            ),
            "parameters": WriteFileInput.model_json_schema(),
        },
    }


def resolve_workspace_path(workspace: Path, relative_path: str) -> Path:
    workspace = workspace.resolve()
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("Absolute paths are not allowed")

    target = (workspace / candidate).resolve()
    if not target.is_relative_to(workspace):
        raise ValueError("Path escapes the workspace")
    if any(part.lower() in BLOCKED_NAMES for part in target.parts):
        raise ValueError("Access to secret configuration files is blocked")
    return target


def _is_ignored_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in BLOCKED_NAMES
        or lowered in IGNORED_NAMES
        or lowered.endswith(".egg-info")
    )


def _workspace_relative_path(workspace: Path, path: Path) -> str:
    relative = path.relative_to(workspace)
    return "." if relative == Path(".") else relative.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_file(workspace: Path, arguments: ReadFileInput) -> dict[str, Any]:
    path = resolve_workspace_path(workspace, arguments.path)
    if not path.exists():
        return {"ok": False, "error": f"File does not exist: {arguments.path}"}
    if not path.is_file():
        return {"ok": False, "error": f"Not a file: {arguments.path}"}
    if path.stat().st_size > MAX_FILE_BYTES:
        return {
            "ok": False,
            "error": f"File exceeds {MAX_FILE_BYTES} bytes: {arguments.path}",
        }

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return {"ok": False, "error": f"File is not UTF-8 text: {arguments.path}"}

    end_line = arguments.end_line or len(lines)
    selected = lines[arguments.start_line - 1 : end_line]
    numbered = "\n".join(
        f"{number:>6} | {line}"
        for number, line in enumerate(selected, start=arguments.start_line)
    )
    truncated = len(numbered) > MAX_RETURN_CHARS
    if truncated:
        numbered = numbered[:MAX_RETURN_CHARS] + "\n... output truncated ..."

    return {
        "ok": True,
        "path": arguments.path,
        "start_line": arguments.start_line,
        "end_line": min(end_line, len(lines)),
        "total_lines": len(lines),
        "truncated": truncated,
        "content": numbered,
    }


def list_files(workspace: Path, arguments: ListFilesInput) -> dict[str, Any]:
    workspace = workspace.resolve()
    root = resolve_workspace_path(workspace, arguments.path)

    relative_root = root.relative_to(workspace)
    if any(_is_ignored_name(part) for part in relative_root.parts):
        return {
            "ok": False,
            "error": f"Listing ignored path is not allowed: {arguments.path}",
        }
    if not root.exists():
        return {"ok": False, "error": f"Directory does not exist: {arguments.path}"}
    if not root.is_dir():
        return {"ok": False, "error": f"Not a directory: {arguments.path}"}

    entries: list[dict[str, Any]] = []
    truncated = False

    def visit(directory: Path, depth: int) -> None:
        nonlocal truncated

        children = sorted(
            directory.iterdir(),
            key=lambda child: (child.name.casefold(), child.name),
        )
        for child in children:
            if _is_ignored_name(child.name):
                continue

            if len(entries) >= arguments.limit:
                truncated = True
                return

            relative_path = _workspace_relative_path(workspace, child)
            if child.is_symlink():
                entries.append({"path": relative_path, "type": "symlink"})
                continue

            resolved_child = child.resolve()
            if not resolved_child.is_relative_to(workspace):
                continue

            if child.is_dir():
                entries.append({"path": relative_path, "type": "directory"})
                if depth < arguments.max_depth:
                    visit(child, depth + 1)
                    if truncated:
                        return
                continue

            if child.is_file():
                entries.append(
                    {
                        "path": relative_path,
                        "type": "file",
                        "size": child.stat().st_size,
                    }
                )

    visit(root, 1)
    return {
        "ok": True,
        "path": _workspace_relative_path(workspace, root),
        "max_depth": arguments.max_depth,
        "limit": arguments.limit,
        "count": len(entries),
        "truncated": truncated,
        "entries": entries,
    }


def write_file(workspace: Path, arguments: WriteFileInput) -> dict[str, Any]:
    workspace = workspace.resolve()
    path = resolve_workspace_path(workspace, arguments.path)
    relative_path = path.relative_to(workspace)

    if relative_path == Path("."):
        return {"ok": False, "error": "Cannot replace the workspace directory"}
    if any(_is_ignored_name(part) for part in relative_path.parts):
        return {
            "ok": False,
            "error": f"Writing ignored or sensitive path is not allowed: {arguments.path}",
        }

    lexical_path = workspace / Path(arguments.path)
    if lexical_path.is_symlink():
        return {
            "ok": False,
            "error": f"Writing through a symbolic link is not allowed: {arguments.path}",
        }

    encoded_content = arguments.content.encode("utf-8")
    if len(encoded_content) > MAX_WRITE_BYTES:
        return {
            "ok": False,
            "error": f"Content exceeds {MAX_WRITE_BYTES} UTF-8 bytes",
        }

    existed = path.exists()
    if existed and not path.is_file():
        return {"ok": False, "error": f"Not a file: {arguments.path}"}
    if existed and not arguments.overwrite:
        return {
            "ok": False,
            "error": (
                f"File already exists: {arguments.path}. "
                "Set overwrite=true to replace it."
            ),
        }

    parent = path.parent
    parent_dirs_created = False
    if not parent.exists():
        if not arguments.create_parent_dirs:
            return {
                "ok": False,
                "error": (
                    f"Parent directory does not exist: "
                    f"{_workspace_relative_path(workspace, parent)}"
                ),
            }
        parent.mkdir(parents=True)
        parent_dirs_created = True
    elif not parent.is_dir():
        return {
            "ok": False,
            "error": (
                f"Parent path is not a directory: "
                f"{_workspace_relative_path(workspace, parent)}"
            ),
        }

    previous_sha256 = _sha256_file(path) if existed else None
    previous_mode = path.stat().st_mode if existed else None
    new_sha256 = hashlib.sha256(encoded_content).hexdigest()
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(encoded_content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)

        if previous_mode is not None:
            os.chmod(temporary_path, previous_mode)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "ok": True,
        "path": _workspace_relative_path(workspace, path),
        "action": "overwritten" if existed else "created",
        "bytes_written": len(encoded_content),
        "sha256": new_sha256,
        "previous_sha256": previous_sha256,
        "changed": previous_sha256 != new_sha256,
        "parent_dirs_created": parent_dirs_created,
    }


class ToolRegistry:
    """Validates model arguments and dispatches local tools."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [read_file_schema(), list_files_schema(), write_file_schema()]

    def execute(self, name: str, raw_arguments: str) -> dict[str, Any]:
        if name not in {"read_file", "list_files", "write_file"}:
            return {"ok": False, "error": f"Unknown tool: {name}"}

        try:
            decoded = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"Invalid JSON arguments: {exc.msg}"}
        if not isinstance(decoded, dict):
            return {"ok": False, "error": "Tool arguments must be a JSON object"}

        try:
            if name == "read_file":
                read_arguments = ReadFileInput.model_validate(decoded)
                return read_file(self.workspace, read_arguments)
            if name == "list_files":
                list_arguments = ListFilesInput.model_validate(decoded)
                return list_files(self.workspace, list_arguments)

            write_arguments = WriteFileInput.model_validate(decoded)
            return write_file(self.workspace, write_arguments)
        except (ValidationError, ValueError, OSError) as exc:
            return {"ok": False, "error": str(exc)}
