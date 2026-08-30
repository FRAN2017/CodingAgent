"""Validated documents and result types for workspace checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

CHECKPOINT_FORMAT_VERSION = 1
CHANGE_STATUSES = {"added", "modified", "deleted", "renamed"}


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be created, compared, or restored."""


def _required_text(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value:
        raise CheckpointError(
            f"Checkpoint field {name!r} must be a non-empty string"
        )
    return value


@dataclass(frozen=True, slots=True)
class CheckpointFile:
    path: str
    sha256: str
    size: int
    mode: int

    @classmethod
    def from_dict(cls, data: Any, *, position: int) -> CheckpointFile:
        if not isinstance(data, dict):
            raise CheckpointError(f"Checkpoint file {position} must be an object")
        if set(data) != {"path", "sha256", "size", "mode"}:
            raise CheckpointError(
                f"Checkpoint file {position} has unsupported fields"
            )
        path = _required_text(data, "path")
        sha256 = _required_text(data, "sha256")
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise CheckpointError(
                f"Checkpoint file {position} has an invalid SHA-256"
            )
        size = data.get("size")
        mode = data.get("mode")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise CheckpointError(
                f"Checkpoint file {position} size must be a non-negative integer"
            )
        if not isinstance(mode, int) or isinstance(mode, bool) or mode < 0:
            raise CheckpointError(
                f"Checkpoint file {position} mode must be a non-negative integer"
            )
        return cls(path=path, sha256=sha256, size=size, mode=mode)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class CheckpointDocument:
    checkpoint_id: str
    workspace: str
    created_at: str
    task: str
    kind: str
    session_id: str | None
    files: tuple[CheckpointFile, ...]
    format_version: int = CHECKPOINT_FORMAT_VERSION

    @classmethod
    def from_dict(cls, data: Any) -> CheckpointDocument:
        if not isinstance(data, dict):
            raise CheckpointError("Checkpoint document must be a JSON object")
        version = data.get("format_version")
        if version != CHECKPOINT_FORMAT_VERSION:
            raise CheckpointError(
                f"Unsupported checkpoint format version: {version!r}"
            )
        created_at = _required_text(data, "created_at")
        try:
            datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise CheckpointError(
                "Checkpoint field 'created_at' must be an ISO-8601 timestamp"
            ) from exc
        raw_files = data.get("files")
        if not isinstance(raw_files, list):
            raise CheckpointError("Checkpoint field 'files' must be a list")
        files = tuple(
            CheckpointFile.from_dict(item, position=position)
            for position, item in enumerate(raw_files)
        )
        paths = [item.path for item in files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise CheckpointError(
                "Checkpoint file paths must be unique and sorted"
            )
        kind = _required_text(data, "kind")
        if kind not in {"task", "pre_undo", "manual"}:
            raise CheckpointError(f"Unsupported checkpoint kind: {kind!r}")
        session_id = data.get("session_id")
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id
        ):
            raise CheckpointError(
                "Checkpoint field 'session_id' must be text or null"
            )
        return cls(
            format_version=version,
            checkpoint_id=_required_text(data, "checkpoint_id"),
            workspace=_required_text(data, "workspace"),
            created_at=created_at,
            task=_required_text(data, "task"),
            kind=kind,
            session_id=session_id,
            files=files,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "checkpoint_id": self.checkpoint_id,
            "workspace": self.workspace,
            "created_at": self.created_at,
            "task": self.task,
            "kind": self.kind,
            "session_id": self.session_id,
            "files": [item.to_dict() for item in self.files],
        }


@dataclass(frozen=True, slots=True)
class FileChange:
    status: str
    path: str
    old_path: str | None = None
    old_size: int | None = None
    new_size: int | None = None
    patch: str | None = None

    def __post_init__(self) -> None:
        if self.status not in CHANGE_STATUSES:
            raise ValueError(f"Unsupported change status: {self.status!r}")


@dataclass(frozen=True, slots=True)
class ChangeSet:
    checkpoint_id: str
    changes: tuple[FileChange, ...]


@dataclass(frozen=True, slots=True)
class RestoreResult:
    checkpoint_id: str
    safety_checkpoint_id: str
    restored_files: int
    removed_files: int

