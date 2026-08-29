"""Configuration for the self-managed request context window."""

from __future__ import annotations

import os
from dataclasses import dataclass


class ContextConfigurationError(ValueError):
    """Raised when context budget settings are invalid."""


def _read_int(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ContextConfigurationError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ContextConfigurationError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Token budget and deterministic compaction settings."""

    max_context_tokens: int = 65_536
    reserved_output_tokens: int = 8_192
    safety_margin_tokens: int = 2_048
    recent_blocks: int = 6
    summary_max_chars: int = 8_000

    def __post_init__(self) -> None:
        if self.max_context_tokens < 1:
            raise ContextConfigurationError("max_context_tokens must be positive")
        if self.reserved_output_tokens < 0:
            raise ContextConfigurationError(
                "reserved_output_tokens must not be negative"
            )
        if self.safety_margin_tokens < 0:
            raise ContextConfigurationError(
                "safety_margin_tokens must not be negative"
            )
        if self.recent_blocks < 1:
            raise ContextConfigurationError("recent_blocks must be at least 1")
        if self.summary_max_chars < 256:
            raise ContextConfigurationError(
                "summary_max_chars must be at least 256"
            )
        if self.input_token_budget < 1:
            raise ContextConfigurationError(
                "reserved output tokens and safety margin consume the entire "
                "context window"
            )

    @property
    def input_token_budget(self) -> int:
        return (
            self.max_context_tokens
            - self.reserved_output_tokens
            - self.safety_margin_tokens
        )

    @classmethod
    def from_env(cls) -> ContextConfig:
        return cls(
            max_context_tokens=_read_int(
                "CODING_AGENT_CONTEXT_TOKENS", 65_536, minimum=1
            ),
            reserved_output_tokens=_read_int(
                "CODING_AGENT_OUTPUT_RESERVE", 8_192, minimum=0
            ),
            safety_margin_tokens=_read_int(
                "CODING_AGENT_CONTEXT_SAFETY_MARGIN", 2_048, minimum=0
            ),
            recent_blocks=_read_int(
                "CODING_AGENT_RECENT_BLOCKS", 6, minimum=1
            ),
            summary_max_chars=_read_int(
                "CODING_AGENT_SUMMARY_MAX_CHARS", 8_000, minimum=256
            ),
        )
