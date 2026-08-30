"""File-backed persistent sessions for coding-agent."""

from coding_agent.sessions.adapter import adapt_messages_for_provider
from coding_agent.sessions.models import (
    ProviderSegment,
    SessionDocument,
    SessionError,
    WorkspaceEvent,
)
from coding_agent.sessions.store import JsonSessionStore, validate_session_id

__all__ = [
    "JsonSessionStore",
    "ProviderSegment",
    "SessionDocument",
    "SessionError",
    "WorkspaceEvent",
    "adapt_messages_for_provider",
    "validate_session_id",
]
