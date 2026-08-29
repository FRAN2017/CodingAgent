from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

from coding_agent.config import Config, QianwenConfig
from coding_agent.llm_client import DeepSeekClient, QianwenClient
from coding_agent.protocol import ModelClientError


def model_response(*, tool_call=True):
    calls = None
    finish_reason = "stop"
    content = "done"
    if tool_call:
        calls = [
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="read_file",
                    arguments='{"path":"README.md"}',
                ),
            )
        ]
        finish_reason = "tool_calls"
        content = None
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    reasoning_content="inspect the requested file",
                    tool_calls=calls,
                ),
            )
        ]
    )


def test_deepseek_adapter_builds_request_and_parses_tool_call():
    adapter = DeepSeekClient(Config(api_key="test-key", model="test-model"))
    create = Mock(return_value=model_response())
    adapter.client.chat.completions.create = create

    turn = adapter.complete(
        messages=[{"role": "user", "content": "Read README.md"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )

    request = create.call_args.kwargs
    assert request["model"] == "test-model"
    assert request["tool_choice"] == "auto"
    assert "reasoning_effort" not in request
    assert request["extra_body"] == {
        "reasoning_effort": "medium",
        "thinking": {"type": "enabled"},
    }
    assert turn.finish_reason == "tool_calls"
    assert turn.tool_calls[0].name == "read_file"
    assert turn.tool_calls[0].arguments == '{"path":"README.md"}'


def test_qianwen_adapter_uses_provider_specific_thinking_parameters():
    adapter = QianwenClient(
        QianwenConfig(
            api_key="test-key",
            model="qwen-test-model",
            reasoning_effort="low",
        )
    )
    create = Mock(return_value=model_response())
    adapter.client.chat.completions.create = create

    adapter.complete(
        messages=[{"role": "user", "content": "Read README.md"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )

    request = create.call_args.kwargs
    assert request["model"] == "qwen-test-model"
    assert "reasoning_effort" not in request
    assert request["extra_body"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
    }


def test_qianwen_adapter_can_disable_thinking():
    adapter = QianwenClient(
        QianwenConfig(api_key="test-key", thinking_enabled=False)
    )
    create = Mock(return_value=model_response(tool_call=False))
    adapter.client.chat.completions.create = create

    turn = adapter.complete(
        messages=[{"role": "user", "content": "Say hello"}],
        tools=[],
    )

    request = create.call_args.kwargs
    assert request["extra_body"] == {"enable_thinking": False}
    assert turn.content == "done"


def _status_error(error_type, status_code, message):
    response = Mock(status_code=status_code, headers={}, request=Mock())
    return error_type(message, response=response, body={"error": message})


def test_adapter_converts_timeout_to_safe_model_error():
    adapter = QianwenClient(
        QianwenConfig(
            api_key="test-key",
            request_timeout_seconds=180,
            max_retries=3,
        )
    )
    adapter.client.chat.completions.create = Mock(
        side_effect=APITimeoutError(request=Mock())
    )

    with pytest.raises(ModelClientError, match="timed out") as error:
        adapter.complete([], [])

    assert error.value.category == "timeout"
    assert error.value.retryable is True
    assert "timeout=180s" in str(error.value)
    assert "retries=3" in str(error.value)


def test_adapter_converts_connection_error_without_traceback_details():
    adapter = DeepSeekClient(Config(api_key="test-key"))
    adapter.client.chat.completions.create = Mock(
        side_effect=APIConnectionError(message="socket failed", request=Mock())
    )

    with pytest.raises(ModelClientError, match="Could not connect") as error:
        adapter.complete([], [])

    assert error.value.category == "connection"
    assert error.value.retryable is True


def test_adapter_converts_authentication_and_rate_limit_errors():
    adapter = QianwenClient(QianwenConfig(api_key="test-key"))
    authentication = _status_error(AuthenticationError, 401, "invalid token")
    adapter.client.chat.completions.create = Mock(side_effect=authentication)

    with pytest.raises(ModelClientError, match="QIANWEN_API_KEY") as auth_error:
        adapter.complete([], [])

    assert auth_error.value.category == "authentication"
    assert auth_error.value.status_code == 401

    rate_limit = _status_error(RateLimitError, 429, "quota exhausted")
    adapter.client.chat.completions.create = Mock(side_effect=rate_limit)
    with pytest.raises(ModelClientError, match="quota") as rate_error:
        adapter.complete([], [])

    assert rate_error.value.category == "rate_limit"
    assert rate_error.value.retryable is True


def test_adapter_classifies_server_error_as_retryable():
    adapter = DeepSeekClient(Config(api_key="test-key"))
    server_error = _status_error(InternalServerError, 503, "temporarily unavailable")
    adapter.client.chat.completions.create = Mock(side_effect=server_error)

    with pytest.raises(ModelClientError, match="HTTP 503") as error:
        adapter.complete([], [])

    assert error.value.category == "server"
    assert error.value.retryable is True
    assert error.value.status_code == 503


def test_adapter_rejects_empty_choices_as_invalid_response():
    adapter = DeepSeekClient(Config(api_key="test-key"))
    adapter.client.chat.completions.create = Mock(
        return_value=SimpleNamespace(choices=[])
    )

    with pytest.raises(ModelClientError, match="choices is empty") as error:
        adapter.complete([], [])

    assert error.value.category == "invalid_response"
    assert error.value.retryable is False


def test_adapter_rejects_malformed_tool_call():
    adapter = QianwenClient(QianwenConfig(api_key="test-key"))
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    reasoning_content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="read_file", arguments=None),
                        )
                    ],
                ),
            )
        ]
    )
    adapter.client.chat.completions.create = Mock(return_value=response)

    with pytest.raises(ModelClientError, match="arguments must be a JSON string"):
        adapter.complete([], [])
