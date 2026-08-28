"""Environment-only configuration for supported model providers."""

from __future__ import annotations

import os
from dataclasses import dataclass

QIANWEN_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class DeepseekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    thinking_enabled: bool = True
    reasoning_effort: str = "medium"
    request_timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> DeepseekConfig:
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
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            thinking_enabled=thinking_enabled,
            reasoning_effort=os.environ.get(
                "DEEPSEEK_REASONING_EFFORT", "high"
            ),
        )


@dataclass(frozen=True, slots=True)
class QianwenConfig:
    api_key: str
    base_url: str = QIANWEN_DEFAULT_BASE_URL
    model: str = "qwen3.8-max"
    thinking_enabled: bool = True
    reasoning_effort: str = "medium"
    request_timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> QianwenConfig:
        api_key = os.environ.get("QIANWEN_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError(
                "QIANWEN_API_KEY is not set. Export it in your shell; "
                "never commit a real key to the repository."
            )

        thinking_value = os.environ.get("QIANWEN_THINKING", "enabled")
        thinking_enabled = thinking_value.strip().lower() not in {
            "0",
            "false",
            "disabled",
            "off",
        }

        return cls(
            api_key=api_key,
            base_url=os.environ.get(
                "QIANWEN_BASE_URL", QIANWEN_DEFAULT_BASE_URL
            ).rstrip("/"),
            model=os.environ.get("QIANWEN_MODEL", "qwen3.8-max"),
            thinking_enabled=thinking_enabled,
            reasoning_effort=os.environ.get(
                "QIANWEN_REASONING_EFFORT", "high"
            ),
        )


# Backward-compatible name used by the original DeepSeek-only public API.
Config = DeepseekConfig
