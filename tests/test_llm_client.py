from types import SimpleNamespace
from unittest.mock import Mock

from coding_agent.config import Config, QianwenConfig
from coding_agent.llm_client import DeepSeekClient, QianwenClient


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
