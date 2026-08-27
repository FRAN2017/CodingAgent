from types import SimpleNamespace
from unittest.mock import Mock

from coding_agent.config import Config
from coding_agent.llm_client import DeepSeekClient


def test_deepseek_adapter_builds_request_and_parses_tool_call():
    adapter = DeepSeekClient(Config(api_key="test-key", model="test-model"))
    create = Mock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content=None,
                        reasoning_content="inspect the requested file",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="read_file",
                                    arguments='{"path":"README.md"}',
                                ),
                            )
                        ],
                    ),
                )
            ]
        )
    )
    adapter.client.chat.completions.create = create

    turn = adapter.complete(
        messages=[{"role": "user", "content": "Read README.md"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )

    request = create.call_args.kwargs
    assert request["model"] == "test-model"
    assert request["tool_choice"] == "auto"
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert turn.finish_reason == "tool_calls"
    assert turn.tool_calls[0].name == "read_file"
    assert turn.tool_calls[0].arguments == '{"path":"README.md"}'
