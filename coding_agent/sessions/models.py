"""Validated, versioned documents for file-backed conversation sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

SESSION_FORMAT_VERSION = 1


class SessionError(RuntimeError):
    """Raised when a persistent session cannot be loaded or saved safely."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require_text(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value:
        raise SessionError(f"Session field {name!r} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class SessionDocument:
    """Complete conversation history plus metadata needed for safe resumption."""

    session_id: str
    workspace: str
    provider: str
    model: str
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]]
    format_version: int = SESSION_FORMAT_VERSION

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        workspace: str,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
    ) -> SessionDocument:
        now = _utc_now()
        return cls(
            session_id=session_id,
            workspace=workspace,
            provider=provider,
            model=model,
            created_at=now,
            updated_at=now,
            messages=deepcopy(messages),
        )

    @classmethod
    def from_dict(cls, data: Any) -> SessionDocument:
        if not isinstance(data, dict):
            raise SessionError("Session document must be a JSON object")
        version = data.get("format_version")
        if version != SESSION_FORMAT_VERSION:
            raise SessionError(
                "Unsupported session format version: "
                f"{version!r}; expected {SESSION_FORMAT_VERSION}"
            )
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise SessionError("Session field 'messages' must be a list")

        created_at = _require_text(data, "created_at")
        updated_at = _require_text(data, "updated_at")
        for name, value in (("created_at", created_at), ("updated_at", updated_at)):
            try:
                datetime.fromisoformat(value)
            except ValueError as exc:
                raise SessionError(
                    f"Session field {name!r} must be an ISO-8601 timestamp"
                ) from exc

        return cls(
            format_version=version,
            session_id=_require_text(data, "session_id"),
            workspace=_require_text(data, "workspace"),
            provider=_require_text(data, "provider"),
            model=_require_text(data, "model"),
            created_at=created_at,
            updated_at=updated_at,
            messages=deepcopy(messages),
        )

    def with_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
    ) -> SessionDocument:
        return replace(
            self,
            model=model,
            updated_at=_utc_now(),
            messages=deepcopy(messages),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "session_id": self.session_id,
            "workspace": self.workspace,
            "provider": self.provider,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": deepcopy(self.messages),
        }
