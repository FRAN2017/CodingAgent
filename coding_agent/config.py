"""Environment-only configuration for supported model providers."""

from __future__ import annotations

import os
from dataclasses import dataclass

QIANWEN_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


def _read_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(
            f"{name} must be between {minimum:g} and {maximum:g}"
        )
    return value


def _read_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _validate_request_settings(timeout_seconds: float, max_retries: int) -> None:
    if not isinstance(timeout_seconds, (int, float)) or not 1 <= timeout_seconds <= 3_600:
        raise ConfigurationError(
            "request_timeout_seconds must be between 1 and 3600"
        )
    if not isinstance(max_retries, int) or not 0 <= max_retries <= 10:
        raise ConfigurationError("max_retries must be between 0 and 10")


@dataclass(frozen=True, slots=True)
class DeepseekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    thinking_enabled: bool = True
    reasoning_effort: str = "medium"
    request_timeout_seconds: float = 180.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        _validate_request_settings(
            self.request_timeout_seconds,
            self.max_retries,
        )

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
            request_timeout_seconds=_read_float(
                "DEEPSEEK_REQUEST_TIMEOUT_SECONDS",
                180.0,
                minimum=1.0,
                maximum=3_600.0,
            ),
            max_retries=_read_int(
                "DEEPSEEK_MAX_RETRIES",
                2,
                minimum=0,
                maximum=10,
            ),
        )


@dataclass(frozen=True, slots=True)
class QianwenConfig:
    api_key: str
    base_url: str = QIANWEN_DEFAULT_BASE_URL
    model: str = "qwen3.8-max"
    thinking_enabled: bool = True
    reasoning_effort: str = "medium"
    request_timeout_seconds: float = 180.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        _validate_request_settings(
            self.request_timeout_seconds,
            self.max_retries,
        )

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
            request_timeout_seconds=_read_float(
                "QIANWEN_REQUEST_TIMEOUT_SECONDS",
                180.0,
                minimum=1.0,
                maximum=3_600.0,
            ),
            max_retries=_read_int(
                "QIANWEN_MAX_RETRIES",
                2,
                minimum=0,
                maximum=10,
            ),
        )


# Backward-compatible name used by the original DeepSeek-only public API.
Config = DeepseekConfig
