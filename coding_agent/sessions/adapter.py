"""Build provider-compatible request copies from raw session history."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from coding_agent.sessions.models import ProviderSegment, SessionError


def adapt_messages_for_provider(
    messages: list[dict[str, Any]],
    provider_segments: tuple[ProviderSegment, ...],
    *,
    target_provider: str,
) -> list[dict[str, Any]]:
    """Return a request-only copy with foreign reasoning fields removed."""
    if not target_provider:
        raise SessionError("Target provider must be a non-empty string")
    if not provider_segments or provider_segments[0].start_index != 0:
        raise SessionError("Provider segments must begin at message 0")

    adapted = deepcopy(messages)
    segment_index = 0
    for message_index, message in enumerate(adapted):
        while (
            segment_index + 1 < len(provider_segments)
            and provider_segments[segment_index + 1].start_index <= message_index
        ):
            segment_index += 1
        source_provider = provider_segments[segment_index].provider
        if message.get("role") == "assistant" and source_provider != target_provider:
            message.pop("reasoning_content", None)
    return adapted
