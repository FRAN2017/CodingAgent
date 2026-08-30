"""Deterministic workspace scanning for complete local checkpoints."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path, PurePosixPath

from coding_agent.checkpoints.models import CheckpointError, CheckpointFile
from coding_agent.tools.workspace import is_ignored_name

MAX_CHECKPOINT_FILES = 20_000
MAX_CHECKPOINT_FILE_BYTES = 32 * 1024 * 1024
MAX_CHECKPOINT_TOTAL_BYTES = 256 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024


def validate_checkpoint_path(path: str) -> str:
    """Validate and normalize a portable checkpoint-relative path."""
    if not isinstance(path, str) or not path or "\\" in path:
        raise CheckpointError("Checkpoint paths must be non-empty POSIX paths")
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise CheckpointError(f"Unsafe checkpoint path: {path!r}")
    if candidate.as_posix() != path:
        raise CheckpointError(f"Checkpoint path is not normalized: {path!r}")
    if any(is_ignored_name(part) for part in candidate.parts):
        raise CheckpointError(f"Checkpoint path is protected or ignored: {path!r}")
    return candidate.as_posix()


def scan_workspace(workspace: Path) -> tuple[CheckpointFile, ...]:
    """Hash every supported file without following links or ignored paths."""
    workspace = workspace.resolve()
    if not workspace.is_dir():
        raise CheckpointError(f"Workspace is not a directory: {workspace}")

    files: list[CheckpointFile] = []
    total_bytes = 0
    visited_directories: set[Path] = set()

    def visit(directory: Path) -> None:
        nonlocal total_bytes
        resolved_directory = directory.resolve()
        if not resolved_directory.is_relative_to(workspace):
            raise CheckpointError(
                f"Directory resolves outside workspace: {directory}"
            )
        if resolved_directory in visited_directories:
            raise CheckpointError(f"Directory cycle detected: {directory}")
        visited_directories.add(resolved_directory)

        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name.casefold())
        except OSError as exc:
            raise CheckpointError(f"Cannot scan directory {directory}: {exc}") from exc

        for child in children:
            if is_ignored_name(child.name):
                continue
            relative_path = child.relative_to(workspace).as_posix()
            if child.is_symlink():
                raise CheckpointError(
                    f"Workspace checkpoints do not support symbolic links: "
                    f"{relative_path}"
                )
            try:
                if child.is_dir():
                    visit(child)
                    continue
                if not child.is_file():
                    raise CheckpointError(
                        f"Unsupported workspace entry: {relative_path}"
                    )
                file_stat = child.stat()
            except OSError as exc:
                raise CheckpointError(
                    f"Cannot inspect workspace entry {relative_path}: {exc}"
                ) from exc

            if file_stat.st_size > MAX_CHECKPOINT_FILE_BYTES:
                raise CheckpointError(
                    f"File exceeds checkpoint limit of "
                    f"{MAX_CHECKPOINT_FILE_BYTES} bytes: {relative_path}"
                )
            total_bytes += file_stat.st_size
            if total_bytes > MAX_CHECKPOINT_TOTAL_BYTES:
                raise CheckpointError(
                    "Workspace exceeds checkpoint total-size limit of "
                    f"{MAX_CHECKPOINT_TOTAL_BYTES} bytes"
                )
            if len(files) >= MAX_CHECKPOINT_FILES:
                raise CheckpointError(
                    f"Workspace exceeds checkpoint limit of {MAX_CHECKPOINT_FILES} files"
                )

            digest = hashlib.sha256()
            bytes_read = 0
            try:
                with child.open("rb") as source:
                    while chunk := source.read(READ_CHUNK_BYTES):
                        digest.update(chunk)
                        bytes_read += len(chunk)
            except OSError as exc:
                raise CheckpointError(
                    f"Cannot read workspace file {relative_path}: {exc}"
                ) from exc
            if bytes_read != file_stat.st_size:
                raise CheckpointError(
                    f"Workspace file changed while scanning: {relative_path}"
                )
            files.append(
                CheckpointFile(
                    path=validate_checkpoint_path(relative_path),
                    sha256=digest.hexdigest(),
                    size=bytes_read,
                    mode=stat.S_IMODE(file_stat.st_mode),
                )
            )

    visit(workspace)
    return tuple(sorted(files, key=lambda item: item.path))
