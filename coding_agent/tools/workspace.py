"""Workspace path validation shared by local tools."""

from __future__ import annotations

from pathlib import Path

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


def resolve_workspace_path(workspace: Path, relative_path: str) -> Path:
    """Resolve a relative path and reject access outside the workspace."""
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


def is_ignored_name(name: str) -> bool:
    """Return whether a path component is hidden from model-facing tools."""
    lowered = name.lower()
    return (
        lowered in BLOCKED_NAMES
        or lowered in IGNORED_NAMES
        or lowered.endswith(".egg-info")
    )


def path_uses_symlink(workspace: Path, relative_path: str) -> bool:
    """Return whether any existing component in a relative path is a symlink."""
    current = workspace
    for part in Path(relative_path).parts:
        if part in {"", "."}:
            continue
        current = current.parent if part == ".." else current / part
        if current.is_symlink():
            return True
    return False


def workspace_relative_path(workspace: Path, path: Path) -> str:
    """Render a path relative to the workspace using portable separators."""
    relative = path.relative_to(workspace)
    return "." if relative == Path(".") else relative.as_posix()
