"""Precise, context-based patching for existing workspace files."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from coding_agent.tools.base import ToolSpec
from coding_agent.tools.workspace import (
    is_ignored_name,
    path_uses_symlink,
    resolve_workspace_path,
    workspace_relative_path,
)

MAX_PATCH_FILE_BYTES = 256 * 1024
MAX_PATCH_BYTES = 256 * 1024

_HUNK_HEADER_RE = re.compile(
    r"^@@\s+-(?P<old_start>\d+)(?:,(?P<old_count>\d+))?\s+"
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))?\s+@@(?P<trailing>.*)$"
)


class PatchError(ValueError):
    """Raised when a patch cannot be parsed or applied safely."""


class ApplyPatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="Workspace-relative file to patch")
    patch: str = Field(min_length=1, description="Patch text containing one or more hunks")

    @model_validator(mode="after")
    def validate_patch_text(self) -> ApplyPatchInput:
        if "\x00" in self.patch:
            raise ValueError("patch must not contain null bytes")
        return self


@dataclass(slots=True)
class PatchHunk:
    number: int
    old_start: int | None
    old_count: int | None
    new_start: int | None
    new_count: int | None
    old_lines: list[str]
    new_lines: list[str]
    removed_count: int
    added_count: int


def _normalize_patch_path(value: str) -> str:
    value = value.strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _validate_update_file_path(line: str, expected_path: str) -> None:
    if ":" not in line:
        raise PatchError(f"Malformed Update File wrapper: {line!r}")
    path_part = line.split(":", 1)[1].strip()
    if len(path_part) >= 2 and path_part[0] == path_part[-1] and path_part[0] in {"'", '"'}:
        path_part = path_part[1:-1]
    if _normalize_patch_path(path_part) != _normalize_patch_path(expected_path):
        raise PatchError(
            f"Patch wrapper path {path_part!r} does not match tool path "
            f"{expected_path!r}"
        )


def _parse_hunk_header(line: str, line_number: int) -> tuple[int | None, int | None, int | None, int | None]:
    stripped = line.strip()
    if stripped == "@@":
        return None, None, None, None

    match = _HUNK_HEADER_RE.match(stripped)
    if match is None:
        raise PatchError(
            f"Malformed hunk header on line {line_number}: {line!r}. "
            "Use '@@' or '@@ -start,count +start,count @@'."
        )

    old_start = int(match.group("old_start"))
    old_count = int(match.group("old_count")) if match.group("old_count") else None
    new_start = int(match.group("new_start"))
    new_count = int(match.group("new_count")) if match.group("new_count") else None
    return old_start, old_count, new_start, new_count


def _finalize_hunk(block_lines: list[str], hunk_number: int) -> PatchHunk:
    while block_lines and block_lines[-1] == "":
        block_lines.pop()

    header_line = block_lines[0]
    old_start, old_count, new_start, new_count = _parse_hunk_header(
        header_line, hunk_number
    )

    old_lines: list[str] = []
    new_lines: list[str] = []
    removed_count = 0
    added_count = 0

    for body_line in block_lines[1:]:
        if body_line.startswith(" "):
            old_lines.append(body_line[1:])
            new_lines.append(body_line[1:])
        elif body_line.startswith("-"):
            old_lines.append(body_line[1:])
            removed_count += 1
        elif body_line.startswith("+"):
            new_lines.append(body_line[1:])
            added_count += 1
        elif body_line == "":
            old_lines.append("")
            new_lines.append("")
        else:
            raise PatchError(
                f"Hunk {hunk_number} contains an invalid line: {body_line!r}. "
                "Patch lines must start with ' ', '-', or '+'."
            )

    if not old_lines:
        raise PatchError(
            f"Hunk {hunk_number} has no context or '-' lines. "
            "Insertion-only hunks are not supported; include at least one "
            "surrounding context line."
        )
    if removed_count == 0 and added_count == 0:
        raise PatchError(f"Hunk {hunk_number} contains no '+' or '-' lines.")
    if old_count is not None and old_count != len(old_lines):
        raise PatchError(
            f"Hunk {hunk_number} header old_count={old_count} does not match "
            f"{len(old_lines)} context/removal lines."
        )
    if new_count is not None and new_count != len(new_lines):
        raise PatchError(
            f"Hunk {hunk_number} header new_count={new_count} does not match "
            f"{len(new_lines)} context/addition lines."
        )
    if old_start is not None and old_start < 1:
        raise PatchError(
            f"Hunk {hunk_number} has invalid old_start={old_start}; "
            "old_start must be at least 1."
        )

    return PatchHunk(
        number=hunk_number,
        old_start=old_start,
        old_count=old_count,
        new_start=new_start,
        new_count=new_count,
        old_lines=old_lines,
        new_lines=new_lines,
        removed_count=removed_count,
        added_count=added_count,
    )


def _parse_patch_hunks(patch: str, expected_path: str) -> list[PatchHunk]:
    hunks: list[PatchHunk] = []
    block_lines: list[str] | None = None

    for raw_line_number, raw_line in enumerate(patch.splitlines(), start=1):
        if raw_line == "*** Begin Patch":
            if block_lines is not None:
                raise PatchError(
                    f"Unexpected Begin Patch wrapper inside hunk on line "
                    f"{raw_line_number}."
                )
            continue
        if raw_line == "*** End Patch":
            if block_lines is not None:
                hunks.append(_finalize_hunk(block_lines, len(hunks) + 1))
                block_lines = None
            continue

        if raw_line.startswith("@@"):
            if block_lines is not None:
                hunks.append(_finalize_hunk(block_lines, len(hunks) + 1))
            block_lines = [raw_line]
            continue

        if block_lines is None:
            stripped = raw_line.strip()
            if stripped == "":
                continue
            if stripped.startswith("*** Update File:"):
                _validate_update_file_path(stripped, expected_path)
                continue
            if raw_line.startswith(("--- ", "+++ ")):
                continue
            raise PatchError(
                f"Expected a hunk header starting with '@@' before line "
                f"{raw_line_number}: {raw_line!r}"
            )
        else:
            block_lines.append(raw_line)

    if block_lines is not None:
        hunks.append(_finalize_hunk(block_lines, len(hunks) + 1))
    if not hunks:
        raise PatchError("Patch contains no hunks; expected at least one '@@' hunk.")

    return hunks


def _find_hunk_match(lines: list[str], hunk: PatchHunk) -> tuple[int, int]:
    old_lines = hunk.old_lines
    matches = [
        index
        for index in range(len(lines) - len(old_lines) + 1)
        if lines[index : index + len(old_lines)] == old_lines
    ]

    if hunk.old_start is not None:
        anchor = hunk.old_start - 1
        if (
            0 <= anchor <= len(lines) - len(old_lines)
            and lines[anchor : anchor + len(old_lines)] == old_lines
        ):
            return anchor, anchor + len(old_lines)

    if not matches:
        header_note = (
            f" (header old_start={hunk.old_start})" if hunk.old_start is not None else ""
        )
        raise PatchError(
            f"Hunk {hunk.number} did not match any lines in the file{header_note}. "
            "Read the file again or add more surrounding context."
        )
    if len(matches) > 1:
        raise PatchError(
            f"Hunk {hunk.number} matched {len(matches)} locations in the file. "
            "Add more surrounding context or use a line-numbered hunk header."
        )

    return matches[0], matches[0] + len(old_lines)


def _check_hunk_overlaps(matches: list[tuple[PatchHunk, int, int]]) -> None:
    ordered = sorted(matches, key=lambda item: (item[1], item[2], item[0].number))
    for index, (_, start, end) in enumerate(ordered):
        for other, other_start, other_end in ordered[index + 1 :]:
            if start < other_end and other_start < end:
                raise PatchError(
                    f"Hunk {other.number} overlaps hunk {ordered[index][0].number}. "
                    "Hunks must modify non-overlapping regions."
                )


def _detect_newline(text: str) -> str:
    for index, char in enumerate(text):
        if char == "\r":
            if index + 1 < len(text) and text[index + 1] == "\n":
                return "\r\n"
            return "\r"
        if char == "\n":
            return "\n"
    return "\n"


def _render_text(lines: list[str], newline: str, had_trailing_newline: bool) -> str:
    if not lines:
        return ""
    rendered = newline.join(lines)
    if had_trailing_newline:
        rendered += newline
    return rendered


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_patch(workspace: Path, arguments: ApplyPatchInput) -> dict[str, Any]:
    workspace = workspace.resolve()
    path = resolve_workspace_path(workspace, arguments.path)
    relative_path = path.relative_to(workspace)

    if relative_path == Path("."):
        return {"ok": False, "error": "Cannot patch the workspace directory"}
    if any(is_ignored_name(part) for part in relative_path.parts):
        return {
            "ok": False,
            "error": f"Patching ignored or sensitive path is not allowed: {arguments.path}",
        }

    if path_uses_symlink(workspace, arguments.path):
        return {
            "ok": False,
            "error": f"Patching through a symbolic link is not allowed: {arguments.path}",
        }
    if not path.exists():
        return {"ok": False, "error": f"File does not exist: {arguments.path}"}
    if not path.is_file():
        return {"ok": False, "error": f"Not a file: {arguments.path}"}
    if path.stat().st_size > MAX_PATCH_FILE_BYTES:
        return {
            "ok": False,
            "error": f"File exceeds {MAX_PATCH_FILE_BYTES} bytes: {arguments.path}",
        }

    patch_bytes = arguments.patch.encode("utf-8")
    if len(patch_bytes) > MAX_PATCH_BYTES:
        return {
            "ok": False,
            "error": f"Patch exceeds {MAX_PATCH_BYTES} UTF-8 bytes",
        }

    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            original_text = file.read()
    except UnicodeDecodeError:
        return {"ok": False, "error": f"File is not UTF-8 text: {arguments.path}"}

    original_lines = original_text.splitlines()
    original_newline = _detect_newline(original_text)
    had_trailing_newline = original_text.endswith(("\r", "\n"))

    hunks = _parse_patch_hunks(arguments.patch, arguments.path)
    matches: list[tuple[PatchHunk, int, int]] = []
    for hunk in hunks:
        start, end = _find_hunk_match(original_lines, hunk)
        matches.append((hunk, start, end))
    _check_hunk_overlaps(matches)

    patched_lines = original_lines.copy()
    lines_added = 0
    lines_removed = 0
    match_details: list[dict[str, int]] = []

    for hunk, start, end in sorted(matches, key=lambda item: item[1], reverse=True):
        patched_lines[start:end] = hunk.new_lines
        lines_added += hunk.added_count
        lines_removed += hunk.removed_count
        match_details.append(
            {
                "hunk": hunk.number,
                "line": start + 1,
                "removed": hunk.removed_count,
                "added": hunk.added_count,
            }
        )

    new_text = _render_text(patched_lines, original_newline, had_trailing_newline)
    encoded_content = new_text.encode("utf-8")
    if len(encoded_content) > MAX_PATCH_FILE_BYTES:
        return {
            "ok": False,
            "error": (
                f"Patched content exceeds {MAX_PATCH_FILE_BYTES} UTF-8 bytes"
            ),
        }

    previous_sha256 = _sha256_file(path)
    previous_mode = path.stat().st_mode
    new_sha256 = hashlib.sha256(encoded_content).hexdigest()
    changed = new_text != original_text

    parent = path.parent
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

        os.chmod(temporary_path, previous_mode)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "ok": True,
        "path": workspace_relative_path(workspace, path),
        "action": "patched",
        "hunks_applied": len(matches),
        "changed": changed,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "bytes_written": len(encoded_content),
        "sha256": new_sha256,
        "previous_sha256": previous_sha256,
        "matches": sorted(match_details, key=lambda detail: detail["line"]),
    }


APPLY_PATCH_TOOL = ToolSpec(
    name="apply_patch",
    description=(
        "Apply one or more precise, context-based hunks to an existing UTF-8 text "
        "file inside the workspace. Unlike write_file, this tool does not require "
        "the complete file content. The patch text contains one or more hunks. "
        "Each hunk starts with a line containing '@@'; the header may be exactly "
        "'@@' or the unified-diff form '@@ -start,count +start,count @@'. "
        "Following hunk lines must be prefixed with ' ' for unchanged context, "
        "'-' for lines to remove, or '+' for lines to add. Every hunk must include "
        "at least one context or '-' line and at least one '+' or '-' line."
    ),
    input_model=ApplyPatchInput,
    handler=apply_patch,
)
