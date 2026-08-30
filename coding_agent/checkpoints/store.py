"""Atomic manifest and content-addressed object persistence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from coding_agent.checkpoints.models import (
    CheckpointDocument,
    CheckpointError,
    CheckpointFile,
)
from coding_agent.checkpoints.scanner import READ_CHUNK_BYTES, validate_checkpoint_path
from coding_agent.tools.workspace import path_uses_symlink

CHECKPOINT_ID_PATTERN = re.compile(r"^cp-[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def validate_checkpoint_id(checkpoint_id: str) -> str:
    checkpoint_id = checkpoint_id.strip()
    if not CHECKPOINT_ID_PATTERN.fullmatch(checkpoint_id):
        raise CheckpointError(
            "Checkpoint id must start with 'cp-' and contain only letters, "
            "numbers, underscores, or hyphens"
        )
    return checkpoint_id


class CheckpointStore:
    """Store immutable file objects and versioned checkpoint manifests."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.state_directory = self.workspace / ".coding-agent"
        self.checkpoints_directory = self.state_directory / "checkpoints"
        self.manifests_directory = self.checkpoints_directory / "manifests"
        self.objects_directory = self.checkpoints_directory / "objects"

    def manifest_path(self, checkpoint_id: str) -> Path:
        valid_id = validate_checkpoint_id(checkpoint_id)
        return self.manifests_directory / f"{valid_id}.json"

    def save_snapshot(self, document: CheckpointDocument) -> Path:
        document = CheckpointDocument.from_dict(document.to_dict())
        self._validate_document(document)
        self._prepare_directories()
        for entry in document.files:
            self._store_object(entry)
        path = self.manifest_path(document.checkpoint_id)
        if path.exists():
            raise CheckpointError(
                f"Checkpoint already exists: {document.checkpoint_id}"
            )
        encoded = (
            json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        self._atomic_write(path, encoded, replace_existing=False)
        return path

    def load(self, checkpoint_id: str) -> CheckpointDocument:
        path = self.manifest_path(checkpoint_id)
        self._reject_symlink_state(path)
        if not path.is_file():
            raise CheckpointError(f"Checkpoint does not exist: {checkpoint_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise CheckpointError(f"Cannot read checkpoint {checkpoint_id}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise CheckpointError(
                f"Checkpoint contains invalid JSON at line {exc.lineno}, "
                f"column {exc.colno}: {checkpoint_id}"
            ) from exc
        document = CheckpointDocument.from_dict(data)
        if document.checkpoint_id != validate_checkpoint_id(checkpoint_id):
            raise CheckpointError(
                "Checkpoint id inside the document does not match its file"
            )
        self._validate_document(document)
        return document

    def list(self) -> list[CheckpointDocument]:
        self._reject_symlink_state(self.manifests_directory)
        if not self.manifests_directory.exists():
            return []
        documents: list[CheckpointDocument] = []
        for path in sorted(self.manifests_directory.glob("cp-*.json")):
            checkpoint_id = path.stem
            documents.append(self.load(checkpoint_id))
        return sorted(documents, key=lambda item: item.created_at, reverse=True)

    def read_object(self, entry: CheckpointFile) -> bytes:
        path = self._object_path(entry.sha256)
        self._reject_symlink_state(path)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise CheckpointError(
                f"Cannot read checkpoint object for {entry.path}: {exc}"
            ) from exc
        if len(content) != entry.size or hashlib.sha256(content).hexdigest() != entry.sha256:
            raise CheckpointError(
                f"Checkpoint object is missing or corrupt for {entry.path}"
            )
        return content

    def _store_object(self, entry: CheckpointFile) -> None:
        relative_path = validate_checkpoint_path(entry.path)
        if path_uses_symlink(self.workspace, relative_path):
            raise CheckpointError(f"Unsafe checkpoint source: {relative_path}")
        source = (self.workspace / Path(relative_path)).resolve()
        if not source.is_relative_to(self.workspace):
            raise CheckpointError(f"Unsafe checkpoint source: {relative_path}")
        try:
            content = source.read_bytes()
        except OSError as exc:
            raise CheckpointError(
                f"Cannot snapshot workspace file {relative_path}: {exc}"
            ) from exc
        if len(content) != entry.size or hashlib.sha256(content).hexdigest() != entry.sha256:
            raise CheckpointError(
                f"Workspace file changed while creating checkpoint: {relative_path}"
            )
        destination = self._object_path(entry.sha256)
        if destination.exists():
            existing = destination.read_bytes()
            if hashlib.sha256(existing).hexdigest() != entry.sha256:
                raise CheckpointError(
                    f"Checkpoint object is corrupt: {entry.sha256}"
                )
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(destination, content, replace_existing=False)

    def _object_path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise CheckpointError("Invalid checkpoint object hash")
        return self.objects_directory / sha256[:2] / sha256[2:]

    def _validate_document(self, document: CheckpointDocument) -> None:
        validate_checkpoint_id(document.checkpoint_id)
        if Path(document.workspace).resolve() != self.workspace:
            raise CheckpointError(
                f"Checkpoint belongs to a different workspace: {document.workspace}"
            )
        for entry in document.files:
            validate_checkpoint_path(entry.path)

    def _prepare_directories(self) -> None:
        self._reject_symlink_state(self.manifests_directory)
        self._reject_symlink_state(self.objects_directory)
        self.manifests_directory.mkdir(parents=True, exist_ok=True)
        self.objects_directory.mkdir(parents=True, exist_ok=True)
        self._reject_symlink_state(self.manifests_directory)
        self._reject_symlink_state(self.objects_directory)

    def _reject_symlink_state(self, target: Path) -> None:
        current = self.workspace
        for part in target.relative_to(self.workspace).parts:
            current /= part
            if current.is_symlink():
                raise CheckpointError(
                    f"Checkpoint storage must not use symbolic links: {current}"
                )

    @staticmethod
    def _atomic_write(path: Path, content: bytes, *, replace_existing: bool) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                for offset in range(0, len(content), READ_CHUNK_BYTES):
                    temporary_file.write(content[offset : offset + READ_CHUNK_BYTES])
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)
            if not replace_existing and path.exists():
                raise CheckpointError(f"Checkpoint state already exists: {path.name}")
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
