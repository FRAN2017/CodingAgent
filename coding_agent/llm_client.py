"""Thin OpenAI-compatible adapters. Agent behavior lives elsewhere."""

from __future__ import annotations

from typing import Any, NoReturn

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

from coding_agent.config import DeepseekConfig, QianwenConfig
from coding_agent.protocol import ModelClientError, ModelTurn, ToolCall

MAX_ERROR_DETAIL_CHARS = 400


def _error_detail(exc: Exception) -> str:
    detail = " ".join(str(exc).split())
    if len(detail) > MAX_ERROR_DETAIL_CHARS:
        return detail[: MAX_ERROR_DETAIL_CHARS - 3] + "..."
    return detail


def _request_id_suffix(exc: Exception) -> str:
    request_id = getattr(exc, "request_id", None)
    return f" (request_id={request_id})" if request_id else ""


def _raise_model_client_error(
    exc: OpenAIError,
    *,
    provider: str,
    api_key_name: str,
    base_url: str,
    timeout_seconds: float,
    max_retries: int,
) -> NoReturn:
    request_suffix = _request_id_suffix(exc)
    if isinstance(exc, APITimeoutError):
        raise ModelClientError(
            f"{provider} request timed out "
            f"(per-attempt timeout={timeout_seconds:g}s, retries={max_retries}). "
            "Check the network or proxy, increase the timeout, or disable thinking.",
            category="timeout",
            retryable=True,
        ) from exc
    if isinstance(exc, APIConnectionError):
        raise ModelClientError(
            f"Could not connect to {provider} API at {base_url}. "
            "Check the network, proxy, and base URL.",
            category="connection",
            retryable=True,
        ) from exc
    if isinstance(exc, AuthenticationError):
        raise ModelClientError(
            f"{provider} authentication failed. Check {api_key_name}.",
            category="authentication",
            retryable=False,
            status_code=exc.status_code,
        ) from exc
    if isinstance(exc, PermissionDeniedError):
        raise ModelClientError(
            f"{provider} denied access to the requested model or operation"
            f"{request_suffix}.",
            category="permission",
            retryable=False,
            status_code=exc.status_code,
        ) from exc
    if isinstance(exc, RateLimitError):
        raise ModelClientError(
            f"{provider} rate limit or quota was exceeded{request_suffix}. "
            "Wait before retrying or check the account quota.",
            category="rate_limit",
            retryable=True,
            status_code=exc.status_code,
        ) from exc
    if isinstance(exc, BadRequestError):
        detail = _error_detail(exc)
        raise ModelClientError(
            f"{provider} rejected the request (HTTP {exc.status_code})"
            f"{request_suffix}: {detail}",
            category="bad_request",
            retryable=False,
            status_code=exc.status_code,
        ) from exc
    if isinstance(exc, APIStatusError):
        retryable = exc.status_code >= 500 or exc.status_code in {408, 409}
        detail = _error_detail(exc)
        advice = " Retry later." if retryable else " Check the request configuration."
        raise ModelClientError(
            f"{provider} API returned HTTP {exc.status_code}{request_suffix}: "
            f"{detail}.{advice}",
            category="server" if exc.status_code >= 500 else "api_status",
            retryable=retryable,
            status_code=exc.status_code,
        ) from exc
    raise ModelClientError(
        f"{provider} client failed: {_error_detail(exc)}",
        category="client",
        retryable=False,
    ) from exc


def _parse_response(response: Any, *, provider: str) -> ModelTurn:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        raise ModelClientError(
            f"{provider} returned an invalid response: choices is empty or missing",
            category="invalid_response",
            retryable=False,
        )
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        raise ModelClientError(
            f"{provider} returned an invalid response: message is missing",
            category="invalid_response",
            retryable=False,
        )

    content = getattr(message, "content", None)
    if content is not None and not isinstance(content, str):
        raise ModelClientError(
            f"{provider} returned an invalid response: content must be text or null",
            category="invalid_response",
            retryable=False,
        )
    raw_calls = getattr(message, "tool_calls", None) or []
    if not isinstance(raw_calls, (list, tuple)):
        raise ModelClientError(
            f"{provider} returned an invalid response: tool_calls must be a list",
            category="invalid_response",
            retryable=False,
        )

    calls: list[ToolCall] = []
    for index, call in enumerate(raw_calls):
        call_id = getattr(call, "id", None)
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        if not isinstance(call_id, str) or not call_id:
            raise ModelClientError(
                f"{provider} returned an invalid tool call at index {index}: "
                "id is missing",
                category="invalid_response",
                retryable=False,
            )
        if not isinstance(name, str) or not name:
            raise ModelClientError(
                f"{provider} returned an invalid tool call at index {index}: "
                "function name is missing",
                category="invalid_response",
                retryable=False,
            )
        if not isinstance(arguments, str):
            raise ModelClientError(
                f"{provider} returned an invalid tool call at index {index}: "
                "arguments must be a JSON string",
                category="invalid_response",
                retryable=False,
            )
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments))

    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ModelClientError(
            f"{provider} returned an invalid response: finish_reason must be text",
            category="invalid_response",
            retryable=False,
        )
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise ModelClientError(
            f"{provider} returned an invalid response: reasoning_content must be text",
            category="invalid_response",
            retryable=False,
        )
    return ModelTurn(
        content=content,
        tool_calls=calls,
        finish_reason=finish_reason,
        reasoning_content=reasoning_content,
    )


def _complete_request(
    client: OpenAI,
    request: dict[str, Any],
    *,
    provider: str,
    api_key_name: str,
    base_url: str,
    timeout_seconds: float,
    max_retries: int,
) -> ModelTurn:
    try:
        response = client.chat.completions.create(**request)
    except OpenAIError as exc:
        _raise_model_client_error(
            exc,
            provider=provider,
            api_key_name=api_key_name,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    return _parse_response(response, provider=provider)


class DeepSeekClient:
    def __init__(self, config: DeepseekConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout_seconds,
            max_retries=config.max_retries,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": False,
        }
        extra_body: dict[str, Any] = {}
        if self.config.thinking_enabled:
            extra_body["reasoning_effort"] = self.config.reasoning_effort
            extra_body["thinking"] = {"type": "enabled"}
        else:
            extra_body["thinking"] = {"type": "disabled"}
        request["extra_body"] = extra_body

        return _complete_request(
            self.client,
            request,
            provider="DeepSeek",
            api_key_name="DEEPSEEK_API_KEY",
            base_url=self.config.base_url,
            timeout_seconds=self.config.request_timeout_seconds,
            max_retries=self.config.max_retries,
        )


class QianwenClient:
    def __init__(self, config: QianwenConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout_seconds,
            max_retries=config.max_retries,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": False,
        }
        extra_body: dict[str, Any] = {
            "enable_thinking": self.config.thinking_enabled,
        }
        if self.config.thinking_enabled:
            extra_body["reasoning_effort"] = self.config.reasoning_effort
        request["extra_body"] = extra_body

        return _complete_request(
            self.client,
            request,
            provider="Qianwen",
            api_key_name="QIANWEN_API_KEY",
            base_url=self.config.base_url,
            timeout_seconds=self.config.request_timeout_seconds,
            max_retries=self.config.max_retries,
        )
