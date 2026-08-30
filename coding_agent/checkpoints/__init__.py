"""Workspace checkpoint, diff, and restore support."""

from coding_agent.checkpoints.manager import CheckpointManager
from coding_agent.checkpoints.models import (
    ChangeSet,
    CheckpointDocument,
    CheckpointError,
    CheckpointFile,
    FileChange,
    RestoreResult,
)

__all__ = [
    "ChangeSet",
    "CheckpointDocument",
    "CheckpointError",
    "CheckpointFile",
    "CheckpointManager",
    "FileChange",
    "RestoreResult",
]
