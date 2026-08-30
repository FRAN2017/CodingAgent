"""Validated, versioned documents for file-backed conversation sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

SESSION_FORMAT_VERSION = 3
LEGACY_SESSION_FORMAT_VERSIONS = {1, 2}


class SessionError(RuntimeError):
    """Raised when a persistent session cannot be loaded or saved safely."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require_text(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value:
        raise SessionError(f"Session field {name!r} must be a non-empty string")
    return value


def _require_checkpoint_id(data: dict[str, Any], name: str) -> str:
    value = _require_text(data, name)
    if not value.startswith("cp-") or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in value
    ):
        raise SessionError(
            f"Session field {name!r} must be a valid checkpoint id"
        )
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


@dataclass(frozen=True, slots=True)
class WorkspaceEvent:
    """Trusted local event stored outside provider conversation messages."""

    event_type: str
    created_at: str
    message_index: int
    checkpoint_id: str
    safety_checkpoint_id: str
    restored_files: int
    removed_files: int

    @classmethod
    def checkpoint_restored(
        cls,
        *,
        message_index: int,
        checkpoint_id: str,
        safety_checkpoint_id: str,
        restored_files: int,
        removed_files: int,
    ) -> WorkspaceEvent:
        event = cls(
            event_type="checkpoint_restored",
            created_at=_utc_now(),
            message_index=message_index,
            checkpoint_id=checkpoint_id,
            safety_checkpoint_id=safety_checkpoint_id,
            restored_files=restored_files,
            removed_files=removed_files,
        )
        return cls.from_dict(event.to_dict(), position=0)

    @classmethod
    def from_dict(cls, data: Any, *, position: int) -> WorkspaceEvent:
        expected_fields = {
            "event_type",
            "created_at",
            "message_index",
            "checkpoint_id",
            "safety_checkpoint_id",
            "restored_files",
            "removed_files",
        }
        if not isinstance(data, dict):
            raise SessionError(f"Workspace event {position} must be an object")
        if set(data) != expected_fields:
            raise SessionError(f"Workspace event {position} has unsupported fields")
        event_type = _require_text(data, "event_type")
        if event_type != "checkpoint_restored":
            raise SessionError(
                f"Workspace event {position} has unsupported type: {event_type!r}"
            )
        created_at = _require_text(data, "created_at")
        try:
            datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise SessionError(
                f"Workspace event {position} created_at must be ISO-8601"
            ) from exc
        integer_fields: dict[str, int] = {}
        for name in ("message_index", "restored_files", "removed_files"):
            value = data.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SessionError(
                    f"Workspace event {position} field {name!r} must be a "
                    "non-negative integer"
                )
            integer_fields[name] = value
        return cls(
            event_type=event_type,
            created_at=created_at,
            message_index=integer_fields["message_index"],
            checkpoint_id=_require_checkpoint_id(data, "checkpoint_id"),
            safety_checkpoint_id=_require_checkpoint_id(
                data,
                "safety_checkpoint_id",
            ),
            restored_files=integer_fields["restored_files"],
            removed_files=integer_fields["removed_files"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "created_at": self.created_at,
            "message_index": self.message_index,
            "checkpoint_id": self.checkpoint_id,
            "safety_checkpoint_id": self.safety_checkpoint_id,
            "restored_files": self.restored_files,
            "removed_files": self.removed_files,
        }

    def as_context_line(self) -> str:
        return (
            f"{self.created_at}: the user restored workspace checkpoint "
            f"{self.checkpoint_id} (restored_files={self.restored_files}, "
            f"removed_files={self.removed_files}). Earlier discussion of file "
            "contents may be stale; re-read relevant files before acting."
        )


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
    workspace_events: tuple[WorkspaceEvent, ...]
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
            workspace_events=(),
        )

    @classmethod
    def from_dict(cls, data: Any) -> SessionDocument:
        if not isinstance(data, dict):
            raise SessionError("Session document must be a JSON object")
        version = data.get("format_version")
        supported_versions = LEGACY_SESSION_FORMAT_VERSIONS | {
            SESSION_FORMAT_VERSION
        }
        if version not in supported_versions:
            raise SessionError(
                "Unsupported session format version: "
                f"{version!r}; expected one of {sorted(supported_versions)}"
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
        if version == 1:
            provider_segments = (ProviderSegment(0, provider),)
        else:
            raw_segments = data.get("provider_segments")
            if not isinstance(raw_segments, list):
                raise SessionError("Session field 'provider_segments' must be a list")
            provider_segments = tuple(
                ProviderSegment.from_dict(segment, position=position)
                for position, segment in enumerate(raw_segments)
            )
        if version < SESSION_FORMAT_VERSION:
            workspace_events: tuple[WorkspaceEvent, ...] = ()
        else:
            raw_events = data.get("workspace_events")
            if not isinstance(raw_events, list):
                raise SessionError("Session field 'workspace_events' must be a list")
            workspace_events = tuple(
                WorkspaceEvent.from_dict(event, position=position)
                for position, event in enumerate(raw_events)
            )
            if any(event.message_index > len(messages) for event in workspace_events):
                raise SessionError(
                    "Workspace event message_index exceeds session history"
                )
            if [event.message_index for event in workspace_events] != sorted(
                event.message_index for event in workspace_events
            ):
                raise SessionError(
                    "Workspace event message indexes must be non-decreasing"
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
            workspace_events=workspace_events,
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

    def with_workspace_restore(
        self,
        *,
        checkpoint_id: str,
        safety_checkpoint_id: str,
        restored_files: int,
        removed_files: int,
    ) -> SessionDocument:
        event = WorkspaceEvent.checkpoint_restored(
            message_index=len(self.messages),
            checkpoint_id=checkpoint_id,
            safety_checkpoint_id=safety_checkpoint_id,
            restored_files=restored_files,
            removed_files=removed_files,
        )
        return replace(
            self,
            updated_at=event.created_at,
            workspace_events=self.workspace_events + (event,),
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
            "workspace_events": [
                event.to_dict() for event in self.workspace_events
            ],
        }
