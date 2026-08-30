"""High-level checkpoint creation, diffing, and safe workspace restore."""

from __future__ import annotations

import difflib
import hashlib
import os
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from coding_agent.checkpoints.models import (
    ChangeSet,
    CheckpointDocument,
    CheckpointError,
    CheckpointFile,
    FileChange,
    RestoreResult,
)
from coding_agent.checkpoints.scanner import (
    READ_CHUNK_BYTES,
    scan_workspace,
    validate_checkpoint_path,
)
from coding_agent.checkpoints.store import CheckpointStore
from coding_agent.tools.workspace import path_uses_symlink

MAX_DIFF_FILE_BYTES = 256 * 1024
MAX_DIFF_CHARS = 40_000


class CheckpointManager:
    """Coordinate immutable snapshots, readable diffs, and reversible restore."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.store = CheckpointStore(self.workspace)
        self.last_checkpoint_id: str | None = None

    def create(
        self,
        task: str,
        *,
        session_id: str | None = None,
        kind: str = "task",
    ) -> CheckpointDocument:
        task = task.strip()
        if not task:
            raise CheckpointError("Checkpoint task must not be empty")
        now = datetime.now(UTC)
        checkpoint_id = (
            f"cp-{now.strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{uuid4().hex[:8]}"
        )
        document = CheckpointDocument(
            checkpoint_id=checkpoint_id,
            workspace=str(self.workspace),
            created_at=now.isoformat(),
            task=task,
            kind=kind,
            session_id=session_id,
            files=scan_workspace(self.workspace),
        )
        self.store.save_snapshot(document)
        self.last_checkpoint_id = checkpoint_id
        return document

    def list(self) -> list[CheckpointDocument]:
        return self.store.list()

    def get(self, checkpoint_id: str) -> CheckpointDocument:
        return self.store.load(checkpoint_id)

    def latest(self) -> CheckpointDocument:
        if self.last_checkpoint_id is not None:
            return self.store.load(self.last_checkpoint_id)
        documents = self.list()
        if not documents:
            raise CheckpointError("No checkpoints are available")
        self.last_checkpoint_id = documents[0].checkpoint_id
        return documents[0]

    def diff(self, checkpoint_id: str | None = None) -> ChangeSet:
        checkpoint = (
            self.store.load(checkpoint_id)
            if checkpoint_id is not None
            else self.latest()
        )
        current_files = scan_workspace(self.workspace)
        old_by_path = {entry.path: entry for entry in checkpoint.files}
        new_by_path = {entry.path: entry for entry in current_files}

        changes: list[FileChange] = []
        deleted = {
            path: entry
            for path, entry in old_by_path.items()
            if path not in new_by_path
        }
        added = {
            path: entry
            for path, entry in new_by_path.items()
            if path not in old_by_path
        }
        self._extract_renames(deleted, added, changes)

        for path in sorted(old_by_path.keys() & new_by_path.keys()):
            old_entry = old_by_path[path]
            new_entry = new_by_path[path]
            if (
                old_entry.sha256 == new_entry.sha256
                and old_entry.mode == new_entry.mode
            ):
                continue
            changes.append(
                FileChange(
                    status="modified",
                    path=path,
                    old_size=old_entry.size,
                    new_size=new_entry.size,
                    patch=self._build_patch(
                        path,
                        self.store.read_object(old_entry),
                        self._read_current(new_entry),
                    ),
                )
            )

        for path, entry in sorted(deleted.items()):
            changes.append(
                FileChange(
                    status="deleted",
                    path=path,
                    old_size=entry.size,
                    new_size=None,
                    patch=self._build_patch(
                        path,
                        self.store.read_object(entry),
                        b"",
                    ),
                )
            )
        for path, entry in sorted(added.items()):
            changes.append(
                FileChange(
                    status="added",
                    path=path,
                    old_size=None,
                    new_size=entry.size,
                    patch=self._build_patch(
                        path,
                        b"",
                        self._read_current(entry),
                    ),
                )
            )

        changes.sort(key=lambda change: (change.path, change.status))
        return ChangeSet(checkpoint.checkpoint_id, tuple(changes))

    def restore(self, checkpoint_id: str | None = None) -> RestoreResult:
        target = (
            self.store.load(checkpoint_id)
            if checkpoint_id is not None
            else self.latest()
        )
        target_content = {
            entry.path: self.store.read_object(entry) for entry in target.files
        }
        safety = self.create(
            f"State before restoring {target.checkpoint_id}",
            session_id=target.session_id,
            kind="pre_undo",
        )
        current_files = scan_workspace(self.workspace)
        current_by_path = {entry.path: entry for entry in current_files}
        target_by_path = {entry.path: entry for entry in target.files}
        self._preflight_restore(target_by_path, current_by_path)

        restored_files = 0
        removed_files = 0
        try:
            for path, entry in target_by_path.items():
                current = current_by_path.get(path)
                if (
                    current is not None
                    and current.sha256 == entry.sha256
                    and current.mode == entry.mode
                ):
                    continue
                self._restore_file(path, target_content[path], entry.mode)
                restored_files += 1

            for path in sorted(current_by_path.keys() - target_by_path.keys()):
                destination = self._workspace_file(path)
                destination.unlink()
                removed_files += 1
        except OSError as exc:
            raise CheckpointError(
                f"Restore failed: {exc}. Safety checkpoint: "
                f"{safety.checkpoint_id}"
            ) from exc

        restored_state = scan_workspace(self.workspace)
        if restored_state != target.files:
            raise CheckpointError(
                "Workspace verification failed after restore. Safety checkpoint: "
                f"{safety.checkpoint_id}"
            )
        self.last_checkpoint_id = safety.checkpoint_id
        return RestoreResult(
            checkpoint_id=target.checkpoint_id,
            safety_checkpoint_id=safety.checkpoint_id,
            restored_files=restored_files,
            removed_files=removed_files,
        )

    def _extract_renames(
        self,
        deleted: dict[str, CheckpointFile],
        added: dict[str, CheckpointFile],
        changes: list[FileChange],
    ) -> None:
        deleted_by_hash: dict[str, list[str]] = defaultdict(list)
        added_by_hash: dict[str, list[str]] = defaultdict(list)
        for path, entry in deleted.items():
            deleted_by_hash[entry.sha256].append(path)
        for path, entry in added.items():
            added_by_hash[entry.sha256].append(path)

        for sha256 in sorted(deleted_by_hash.keys() & added_by_hash.keys()):
            old_paths = sorted(deleted_by_hash[sha256])
            new_paths = sorted(added_by_hash[sha256])
            for old_path, new_path in zip(old_paths, new_paths, strict=False):
                old_entry = deleted.pop(old_path)
                new_entry = added.pop(new_path)
                changes.append(
                    FileChange(
                        status="renamed",
                        path=new_path,
                        old_path=old_path,
                        old_size=old_entry.size,
                        new_size=new_entry.size,
                    )
                )

    def _build_patch(self, path: str, old: bytes, new: bytes) -> str | None:
        if max(len(old), len(new)) > MAX_DIFF_FILE_BYTES:
            return None
        if b"\x00" in old or b"\x00" in new:
            return None
        try:
            old_text = old.decode("utf-8")
            new_text = new.decode("utf-8")
        except UnicodeDecodeError:
            return None
        old_label = f"a/{path}" if old else "/dev/null"
        new_label = f"b/{path}" if new else "/dev/null"
        patch = "".join(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=old_label,
                tofile=new_label,
            )
        )
        if len(patch) <= MAX_DIFF_CHARS:
            return patch
        return patch[:MAX_DIFF_CHARS] + "\n... diff truncated ...\n"

    def _read_current(self, entry: CheckpointFile) -> bytes:
        path = self._workspace_file(entry.path)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise CheckpointError(
                f"Cannot read current workspace file {entry.path}: {exc}"
            ) from exc
        if len(content) != entry.size or hashlib.sha256(content).hexdigest() != entry.sha256:
            raise CheckpointError(
                f"Workspace file changed while building diff: {entry.path}"
            )
        return content

    def _preflight_restore(
        self,
        target: dict[str, CheckpointFile],
        current: dict[str, CheckpointFile],
    ) -> None:
        for path in target:
            destination = self.workspace / Path(path)
            if path_uses_symlink(self.workspace, path):
                raise CheckpointError(f"Cannot restore through a symbolic link: {path}")
            if destination.exists() and not destination.is_file():
                raise CheckpointError(
                    f"Cannot restore file over a non-file entry: {path}"
                )
        for path in current.keys() - target.keys():
            destination = self._workspace_file(path)
            if not destination.is_file() or destination.is_symlink():
                raise CheckpointError(f"Cannot safely remove workspace entry: {path}")

    def _restore_file(self, path: str, content: bytes, mode: int) -> None:
        destination = self._workspace_file(path, require_existing=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                for offset in range(0, len(content), READ_CHUNK_BYTES):
                    temporary_file.write(content[offset : offset + READ_CHUNK_BYTES])
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)
            os.chmod(temporary_path, mode)
            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _workspace_file(
        self,
        path: str,
        *,
        require_existing: bool = True,
    ) -> Path:
        valid_path = validate_checkpoint_path(path)
        if path_uses_symlink(self.workspace, valid_path):
            raise CheckpointError(f"Checkpoint path uses a symbolic link: {path}")
        destination = (self.workspace / Path(valid_path)).resolve()
        if not destination.is_relative_to(self.workspace):
            raise CheckpointError(f"Checkpoint path escapes workspace: {path}")
        if require_existing and not destination.is_file():
            raise CheckpointError(f"Workspace file does not exist: {path}")
        return destination
