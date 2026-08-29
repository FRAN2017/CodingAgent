import json

import pytest

from coding_agent.context import (
    ContextBudgetError,
    ContextConfig,
    ContextConfigurationError,
    ContextManager,
    ConversationHistory,
    HistoryError,
    TokenCounter,
)


class CharacterCounter:
    """Deterministic counter used to exercise budget branches."""

    def count_messages(self, messages):
        return len(json.dumps(messages, ensure_ascii=False))

    def count_tools(self, tools):
        return len(json.dumps(tools, ensure_ascii=False))


def _append_tool_turn(history, call_id, name, arguments, result):
    history.append_assistant(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }
    )
    history.append_tool(call_id, json.dumps(result))


def test_request_is_unchanged_when_history_fits_budget():
    history = ConversationHistory.for_task("system", "task")
    history.append_assistant({"role": "assistant", "content": "done"})
    original = history.messages
    manager = ContextManager(
        ContextConfig(
            max_context_tokens=2_000,
            reserved_output_tokens=0,
            safety_margin_tokens=0,
        ),
        token_counter=CharacterCounter(),
    )

    request = manager.build_request(history, [])

    assert request == original
    request[0]["content"] = "mutated"
    assert history.messages[0]["content"] == "system"


def test_old_tool_events_are_compacted_and_recent_block_stays_atomic():
    history = ConversationHistory.for_task("system", "task")
    _append_tool_turn(
        history,
        "old_read",
        "read_file",
        {"path": "old.py"},
        {
            "ok": True,
            "path": "old.py",
            "start_line": 1,
            "end_line": 50,
            "content": "x" * 2_000,
        },
    )
    _append_tool_turn(
        history,
        "latest_run",
        "run_command",
        {"argv": ["python", "check.py"]},
        {
            "ok": True,
            "argv": ["python", "check.py"],
            "cwd": ".",
            "exit_code": 0,
            "stdout": "passed",
            "stderr": "",
            "timed_out": False,
        },
    )
    manager = ContextManager(
        ContextConfig(
            max_context_tokens=1_100,
            reserved_output_tokens=0,
            safety_margin_tokens=0,
            recent_blocks=1,
            summary_max_chars=512,
        ),
        token_counter=CharacterCounter(),
    )

    request = manager.build_request(history, [])

    assert len(history.messages) == 6
    assert request[0] == {"role": "system", "content": "system"}
    assert request[1]["role"] == "system"
    assert "untrusted data" in request[1]["content"]
    assert "read_file path=old.py" in request[1]["content"]
    assert request[2] == {"role": "user", "content": "task"}
    assert [message["role"] for message in request[-2:]] == ["assistant", "tool"]
    assert request[-1]["tool_call_id"] == "latest_run"
    assert "passed" in request[-1]["content"]


def test_multiple_tool_results_form_one_atomic_block():
    history = ConversationHistory.for_task("system", "task")
    history.append_assistant(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "one",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                },
                {
                    "id": "two",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": "{}"},
                },
            ],
        }
    )
    history.append_tool("two", '{"ok": true}')
    history.append_tool("one", '{"ok": true}')

    block = history.blocks()[-1]

    assert block.kind == "tool_interaction"
    assert len(block.messages) == 3
    assert {message.get("tool_call_id") for message in block.messages[1:]} == {
        "one",
        "two",
    }


def test_history_rejects_orphan_and_incomplete_tool_results():
    history = ConversationHistory.for_task("system", "task")
    with pytest.raises(HistoryError, match="pending tool call"):
        history.append_tool("orphan", "{}")

    history.append_assistant(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "pending",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        }
    )
    with pytest.raises(HistoryError, match="Missing tool results"):
        history.blocks()


def test_history_can_be_rebuilt_for_a_later_user_turn():
    original = ConversationHistory.for_task("system", "first task")
    original.append_assistant({"role": "assistant", "content": "first answer"})
    original.append_user("second task")

    restored = ConversationHistory.from_messages(original.messages)

    assert restored.messages == original.messages


def test_multi_turn_compaction_keeps_latest_user_and_summarizes_earlier_turn():
    history = ConversationHistory.for_task("system", "first task " + "x" * 1_500)
    history.append_assistant(
        {"role": "assistant", "content": "first answer " + "y" * 1_500}
    )
    history.append_user("latest task")
    _append_tool_turn(
        history,
        "latest_read",
        "read_file",
        {"path": "latest.py"},
        {
            "ok": True,
            "path": "latest.py",
            "start_line": 1,
            "end_line": 1,
            "content": "current",
        },
    )
    manager = ContextManager(
        ContextConfig(
            max_context_tokens=1_800,
            reserved_output_tokens=0,
            safety_margin_tokens=0,
            recent_blocks=1,
            summary_max_chars=1_000,
        ),
        token_counter=CharacterCounter(),
    )

    request = manager.build_request(history, [])

    assert request[0] == {"role": "system", "content": "system"}
    assert request[1]["role"] == "system"
    assert "- user: first task" in request[1]["content"]
    assert "- assistant: first answer" in request[1]["content"]
    assert request[2] == {"role": "user", "content": "latest task"}
    assert request[-1]["tool_call_id"] == "latest_read"


def test_tool_schemas_are_included_in_the_budget():
    history = ConversationHistory.for_task("system", "task")
    manager = ContextManager(
        ContextConfig(
            max_context_tokens=300,
            reserved_output_tokens=0,
            safety_margin_tokens=0,
        ),
        token_counter=CharacterCounter(),
    )

    with pytest.raises(ContextBudgetError, match="Tool schemas consume"):
        manager.build_request(history, [{"schema": "x" * 400}])


def test_required_messages_that_cannot_fit_raise_clear_error():
    history = ConversationHistory.for_task("s" * 400, "task")
    manager = ContextManager(
        ContextConfig(
            max_context_tokens=300,
            reserved_output_tokens=0,
            safety_margin_tokens=0,
        ),
        token_counter=CharacterCounter(),
    )

    with pytest.raises(ContextBudgetError, match="Required system"):
        manager.build_request(history, [])


def test_context_config_reads_environment(monkeypatch):
    monkeypatch.setenv("CODING_AGENT_CONTEXT_TOKENS", "10000")
    monkeypatch.setenv("CODING_AGENT_OUTPUT_RESERVE", "1000")
    monkeypatch.setenv("CODING_AGENT_CONTEXT_SAFETY_MARGIN", "500")
    monkeypatch.setenv("CODING_AGENT_RECENT_BLOCKS", "3")
    monkeypatch.setenv("CODING_AGENT_SUMMARY_MAX_CHARS", "1200")

    config = ContextConfig.from_env()

    assert config.input_token_budget == 8_500
    assert config.recent_blocks == 3
    assert config.summary_max_chars == 1_200


def test_context_config_rejects_invalid_environment(monkeypatch):
    monkeypatch.setenv("CODING_AGENT_CONTEXT_TOKENS", "many")

    with pytest.raises(ContextConfigurationError, match="must be an integer"):
        ContextConfig.from_env()


def test_default_token_counter_handles_chinese_and_tool_schemas():
    counter = TokenCounter()

    assert counter.count_text("中文") > 0
    assert counter.count_messages([{"role": "user", "content": "你好"}]) > 0
    assert counter.count_tools([{"type": "function"}]) > 0
