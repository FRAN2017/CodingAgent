"""Literal text search across workspace files."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from coding_agent.tools.base import ToolSpec
from coding_agent.tools.workspace import (
    is_ignored_name,
    resolve_workspace_path,
    workspace_relative_path,
)

MAX_QUERY_CHARS = 1_000
MAX_FILE_PATTERN_CHARS = 200
MAX_SEARCH_RESULTS = 1_000
MAX_SEARCH_FILES = 10_000
MAX_SEARCH_FILE_BYTES = 256 * 1024
MAX_MATCH_LINE_CHARS = 1_000
MAX_SEARCH_RETURN_CHARS = 40_000


class SearchTextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="Literal text to find; regular expressions are not supported",
    )
    path: str = Field(
        default=".",
        min_length=1,
        description="Workspace-relative file or directory to search",
    )
    case_sensitive: bool = Field(
        default=True,
        description="Match letter case when true",
    )
    file_pattern: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_FILE_PATTERN_CHARS,
        description="Optional file glob such as '*.py' or 'src/*.py'",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        le=MAX_SEARCH_RESULTS,
        description="Maximum number of matching lines to return",
    )
    max_files: int = Field(
        default=1_000,
        ge=1,
        le=MAX_SEARCH_FILES,
        description="Maximum number of candidate files to inspect",
    )

    @model_validator(mode="after")
    def validate_search_text(self) -> SearchTextInput:
        if "\x00" in self.query:
            raise ValueError("query must not contain null bytes")
        if "\n" in self.query or "\r" in self.query:
            raise ValueError("query must contain a single line of text")
        if self.file_pattern is not None:
            if "\x00" in self.file_pattern:
                raise ValueError("file_pattern must not contain null bytes")
            if "\n" in self.file_pattern or "\r" in self.file_pattern:
                raise ValueError("file_pattern must be a single-line glob")
        return self


def _matches_file_pattern(
    relative_path: str,
    file_name: str,
    pattern: str | None,
) -> bool:
    if pattern is None:
        return True
    normalized_pattern = pattern.replace("\\", "/")
    return fnmatch.fnmatchcase(relative_path, normalized_pattern) or fnmatch.fnmatchcase(
        file_name, normalized_pattern
    )


def _candidate_files(
    workspace: Path,
    root: Path,
    pattern: str | None,
):
    if root.is_file():
        relative_path = workspace_relative_path(workspace, root)
        if _matches_file_pattern(relative_path, root.name, pattern):
            yield root
        return

    def visit(directory: Path):
        children = sorted(
            directory.iterdir(),
            key=lambda child: (child.name.casefold(), child.name),
        )
        for child in children:
            if is_ignored_name(child.name) or child.is_symlink():
                continue
            resolved_child = child.resolve()
            if not resolved_child.is_relative_to(workspace):
                continue
            if child.is_dir():
                yield from visit(child)
                continue
            if child.is_file():
                relative_path = workspace_relative_path(workspace, child)
                if _matches_file_pattern(relative_path, child.name, pattern):
                    yield child

    yield from visit(root)


def _find_column(line: str, query: str, case_sensitive: bool) -> int | None:
    if case_sensitive:
        index = line.find(query)
        return None if index < 0 else index + 1
    match = re.search(re.escape(query), line, flags=re.IGNORECASE)
    return None if match is None else match.start() + 1


def _render_match_line(line: str, column: int) -> tuple[str, bool]:
    if len(line) <= MAX_MATCH_LINE_CHARS:
        return line, False

    marker = "..."
    match_index = column - 1
    half_window = (MAX_MATCH_LINE_CHARS - 2 * len(marker)) // 2
    start = max(0, match_index - half_window)
    end = min(len(line), start + MAX_MATCH_LINE_CHARS - 2 * len(marker))
    start = max(0, end - (MAX_MATCH_LINE_CHARS - 2 * len(marker)))
    prefix = marker if start > 0 else ""
    suffix = marker if end < len(line) else ""
    return prefix + line[start:end] + suffix, True


def search_text(workspace: Path, arguments: SearchTextInput) -> dict[str, Any]:
    workspace = workspace.resolve()
    root = resolve_workspace_path(workspace, arguments.path)
    relative_root = root.relative_to(workspace)

    if any(is_ignored_name(part) for part in relative_root.parts):
        return {
            "ok": False,
            "error": f"Searching ignored path is not allowed: {arguments.path}",
        }
    lexical_root = workspace / Path(arguments.path)
    if lexical_root.is_symlink():
        return {
            "ok": False,
            "error": f"Searching a symbolic link is not allowed: {arguments.path}",
        }
    if not root.exists():
        return {"ok": False, "error": f"Search path does not exist: {arguments.path}"}
    if not root.is_file() and not root.is_dir():
        return {"ok": False, "error": f"Not a file or directory: {arguments.path}"}

    matches: list[dict[str, Any]] = []
    files_considered = 0
    files_searched = 0
    skipped_too_large = 0
    skipped_non_text = 0
    skipped_unreadable = 0
    returned_chars = 0
    truncated = False

    for file_path in _candidate_files(workspace, root, arguments.file_pattern):
        if files_considered >= arguments.max_files:
            truncated = True
            break
        files_considered += 1

        try:
            if file_path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                skipped_too_large += 1
                continue
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped_non_text += 1
            continue
        except OSError:
            skipped_unreadable += 1
            continue

        if "\x00" in content:
            skipped_non_text += 1
            continue

        files_searched += 1
        relative_path = workspace_relative_path(workspace, file_path)
        for line_number, line in enumerate(content.splitlines(), start=1):
            column = _find_column(line, arguments.query, arguments.case_sensitive)
            if column is None:
                continue

            rendered_line, line_truncated = _render_match_line(line, column)
            estimated_chars = len(relative_path) + len(rendered_line) + 80
            if (
                len(matches) >= arguments.max_results
                or returned_chars + estimated_chars > MAX_SEARCH_RETURN_CHARS
            ):
                truncated = True
                break
            matches.append(
                {
                    "path": relative_path,
                    "line": line_number,
                    "column": column,
                    "text": rendered_line,
                    "line_truncated": line_truncated,
                }
            )
            returned_chars += estimated_chars

        if truncated:
            break

    return {
        "ok": True,
        "query": arguments.query,
        "path": workspace_relative_path(workspace, root),
        "case_sensitive": arguments.case_sensitive,
        "file_pattern": arguments.file_pattern,
        "max_results": arguments.max_results,
        "max_files": arguments.max_files,
        "match_count": len(matches),
        "files_considered": files_considered,
        "files_searched": files_searched,
        "truncated": truncated,
        "skipped": {
            "too_large": skipped_too_large,
            "non_text": skipped_non_text,
            "unreadable": skipped_unreadable,
        },
        "matches": matches,
    }


SEARCH_TEXT_TOOL = ToolSpec(
    name="search_text",
    description=(
        "Search for literal text in a workspace file or recursively in a directory. "
        "Returns matching file paths, line numbers, columns, and line text."
    ),
    input_model=SearchTextInput,
    handler=search_text,
)
