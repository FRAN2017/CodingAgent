"""Validated, versioned documents for file-backed conversation sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

SESSION_FORMAT_VERSION = 2
LEGACY_SESSION_FORMAT_VERSION = 1


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
class ProviderSegment:
    """A half-open history segment produced while using one provider."""

    start_index: int
    provider: str

    @classmethod
    def from_dict(cls, data: Any, *, position: int) -> ProviderSegment:
        if not isinstance(data, dict):
            raise SessionError(f"Provider segment {position} must be an object")
        if set(data) != {"start_index", "provider"}:
            raise SessionError(
                f"Provider segment {position} must contain only "
                "'start_index' and 'provider'"
            )
        start_index = data.get("start_index")
        if not isinstance(start_index, int) or isinstance(start_index, bool):
            raise SessionError(
                f"Provider segment {position} start_index must be an integer"
            )
        provider = data.get("provider")
        if not isinstance(provider, str) or not provider:
            raise SessionError(
                f"Provider segment {position} provider must be non-empty"
            )
        return cls(start_index=start_index, provider=provider)

    def to_dict(self) -> dict[str, Any]:
        return {"start_index": self.start_index, "provider": self.provider}


def _validate_provider_segments(
    segments: tuple[ProviderSegment, ...],
    *,
    message_count: int,
    current_provider: str,
) -> None:
    if not segments:
        raise SessionError("Session must contain at least one provider segment")
    if segments[0].start_index != 0:
        raise SessionError("The first provider segment must start at message 0")

    previous_index = -1
    for position, segment in enumerate(segments):
        if segment.start_index <= previous_index:
            raise SessionError("Provider segment start indexes must be increasing")
        if segment.start_index >= message_count:
            raise SessionError(
                f"Provider segment {position} starts outside the message history"
            )
        previous_index = segment.start_index

    if segments[-1].provider != current_provider:
        raise SessionError(
            "The session provider must match the final provider segment"
        )


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
    provider_segments: tuple[ProviderSegment, ...]
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
            provider_segments=(ProviderSegment(0, provider),),
        )

    @classmethod
    def from_dict(cls, data: Any) -> SessionDocument:
        if not isinstance(data, dict):
            raise SessionError("Session document must be a JSON object")
        version = data.get("format_version")
        if version not in {LEGACY_SESSION_FORMAT_VERSION, SESSION_FORMAT_VERSION}:
            raise SessionError(
                "Unsupported session format version: "
                f"{version!r}; expected {LEGACY_SESSION_FORMAT_VERSION} or "
                f"{SESSION_FORMAT_VERSION}"
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

        provider = _require_text(data, "provider")
        if version == LEGACY_SESSION_FORMAT_VERSION:
            provider_segments = (ProviderSegment(0, provider),)
        else:
            raw_segments = data.get("provider_segments")
            if not isinstance(raw_segments, list):
                raise SessionError("Session field 'provider_segments' must be a list")
            provider_segments = tuple(
                ProviderSegment.from_dict(segment, position=position)
                for position, segment in enumerate(raw_segments)
            )
        _validate_provider_segments(
            provider_segments,
            message_count=len(messages),
            current_provider=provider,
        )

        return cls(
            format_version=SESSION_FORMAT_VERSION,
            session_id=_require_text(data, "session_id"),
            workspace=_require_text(data, "workspace"),
            provider=provider,
            model=_require_text(data, "model"),
            created_at=created_at,
            updated_at=updated_at,
            messages=deepcopy(messages),
            provider_segments=provider_segments,
        )

    def with_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        provider: str | None = None,
        model: str,
    ) -> SessionDocument:
        current_provider = provider or self.provider
        if not current_provider:
            raise SessionError("Session provider must be a non-empty string")
        provider_segments = self.provider_segments
        if current_provider != self.provider:
            if len(messages) <= len(self.messages):
                raise SessionError(
                    "Switching provider requires at least one appended message"
                )
            provider_segments += (
                ProviderSegment(len(self.messages), current_provider),
            )
        _validate_provider_segments(
            provider_segments,
            message_count=len(messages),
            current_provider=current_provider,
        )
        return replace(
            self,
            provider=current_provider,
            model=model,
            updated_at=_utc_now(),
            messages=deepcopy(messages),
            provider_segments=provider_segments,
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
            "provider_segments": [
                segment.to_dict() for segment in self.provider_segments
            ],
        }
