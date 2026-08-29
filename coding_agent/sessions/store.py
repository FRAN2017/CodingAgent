"""Atomic JSON persistence for complete workspace conversation histories."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from coding_agent.context import ConversationHistory, HistoryError
from coding_agent.sessions.models import SessionDocument, SessionError

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SESSION_STATE_DIRECTORY = ".coding-agent"
MAX_SESSION_BYTES = 32 * 1024 * 1024


def validate_session_id(session_id: str) -> str:
    session_id = session_id.strip()
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise SessionError(
            "Session id must contain 1-64 letters, numbers, underscores, or "
            "hyphens, and must start with a letter or number"
        )
    return session_id


class JsonSessionStore:
    """Load and atomically replace versioned JSON session documents."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.state_directory = self.workspace / SESSION_STATE_DIRECTORY
        self.sessions_directory = self.state_directory / "sessions"

    def session_path(self, session_id: str) -> Path:
        valid_id = validate_session_id(session_id)
        return self.sessions_directory / f"session-{valid_id}.json"

    def load(self, session_id: str) -> SessionDocument | None:
        path = self.session_path(session_id)
        self._reject_symlink_state(path)
        if not path.exists():
            return None
        if not path.is_file():
            raise SessionError(f"Session path is not a file: {path}")
        if path.stat().st_size > MAX_SESSION_BYTES:
            raise SessionError(
                f"Session file exceeds {MAX_SESSION_BYTES} bytes: {path.name}"
            )

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise SessionError(f"Session file is not UTF-8: {path.name}") from exc
        except json.JSONDecodeError as exc:
            raise SessionError(
                f"Session file contains invalid JSON at line {exc.lineno}, "
                f"column {exc.colno}: {path.name}"
            ) from exc

        document = SessionDocument.from_dict(data)
        if document.session_id != validate_session_id(session_id):
            raise SessionError("Session id inside the document does not match its file")
        if Path(document.workspace).resolve() != self.workspace:
            raise SessionError(
                "Session belongs to a different workspace: "
                f"{document.workspace}"
            )
        try:
            ConversationHistory.from_messages(document.messages)
        except HistoryError as exc:
            raise SessionError(f"Session message history is invalid: {exc}") from exc
        return document

    def save(self, document: SessionDocument) -> Path:
        document = SessionDocument.from_dict(document.to_dict())
        path = self.session_path(document.session_id)
        if Path(document.workspace).resolve() != self.workspace:
            raise SessionError("Cannot save a session for a different workspace")
        try:
            validated = ConversationHistory.from_messages(document.messages)
        except HistoryError as exc:
            raise SessionError(f"Cannot save invalid session history: {exc}") from exc

        document = document.with_messages(validated.messages, model=document.model)
        encoded = (
            json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        if len(encoded) > MAX_SESSION_BYTES:
            raise SessionError(
                f"Session content exceeds {MAX_SESSION_BYTES} UTF-8 bytes"
            )

        self._prepare_directories()
        self._reject_symlink_state(path)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.sessions_directory,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_file.write(encoded)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)

            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def _prepare_directories(self) -> None:
        self._reject_symlink_state(self.sessions_directory)
        self.sessions_directory.mkdir(parents=True, exist_ok=True)
        self._reject_symlink_state(self.sessions_directory)

    def _reject_symlink_state(self, target: Path) -> None:
        current = self.workspace
        relative = target.relative_to(self.workspace)
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise SessionError(
                    f"Session storage must not use symbolic links: {current}"
                )
