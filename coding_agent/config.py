"""Environment-only configuration for DeepSeek."""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Config:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    thinking_enabled: bool = True
    reasoning_effort: str = "medium"
    request_timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> Config:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError(
                "DEEPSEEK_API_KEY is not set. Export it in your shell; "
                "never commit a real key to the repository."
            )

        thinking_value = os.environ.get("DEEPSEEK_THINKING", "enabled")
        thinking_enabled = thinking_value.strip().lower() not in {
            "0",
            "false",
            "disabled",
            "off",
        }

        return cls(
            api_key=api_key,
            base_url=os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            thinking_enabled=thinking_enabled,
            reasoning_effort=os.environ.get(
                "DEEPSEEK_REASONING_EFFORT", "high"
            ),
        )
