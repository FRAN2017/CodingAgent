import json
import sys

import pytest

from coding_agent.agent import Agent, AgentError
from coding_agent.protocol import ModelTurn, ToolCall


class FakeClient:
    """Deterministic model replacement for testing the complete agent loop."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, messages, tools):
        self.call_count += 1
        assert tools[0]["function"]["name"] == "read_file"

        if self.call_count == 1:
            return ModelTurn(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_readme",
                        name="read_file",
                        arguments=json.dumps({"path": "README.md"}),
                    )
                ],
            )

        tool_message = messages[-1]
        assert tool_message["role"] == "tool"
        result = json.loads(tool_message["content"])
        assert result["ok"] is True
        assert "calculator.py" in result["content"]
        return ModelTurn(
            content="The implementation is in calculator.py.",
            finish_reason="stop",
        )


class EndlessToolClient:
    def complete(self, messages, tools):
        return ModelTurn(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id=f"call_{len(messages)}",
                    name="read_file",
                    arguments=json.dumps({"path": "README.md"}),
                )
            ],
        )


class WriteThenReadClient:
    """Fake model that creates a file and reads it back before finishing."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, messages, tools):
        self.call_count += 1
        tool_names = [schema["function"]["name"] for schema in tools]
        assert tool_names == [
            "read_file",
            "list_files",
            "write_file",
            "rename_file",
            "search_text",
            "run_command",
            "apply_patch",
        ]

        if self.call_count == 1:
            return ModelTurn(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments=json.dumps(
                            {"path": "hello.py", "content": "print('Hello, Agent!')\n"}
                        ),
                    )
                ],
            )

        tool_result = json.loads(messages[-1]["content"])
        assert tool_result["ok"] is True
        if self.call_count == 2:
            assert tool_result["action"] == "created"
            return ModelTurn(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_read",
                        name="read_file",
                        arguments=json.dumps({"path": "hello.py"}),
                    )
                ],
            )

        assert "Hello, Agent!" in tool_result["content"]
        return ModelTurn(
            content="Created hello.py and verified its contents.",
            finish_reason="stop",
        )


class WriteThenRunClient:
    """Fake model that creates and executes a program before finishing."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, messages, tools):
        self.call_count += 1
        tool_names = [schema["function"]["name"] for schema in tools]
        assert tool_names == [
            "read_file",
            "list_files",
            "write_file",
            "rename_file",
            "search_text",
            "run_command",
            "apply_patch",
        ]

        if self.call_count == 1:
            return ModelTurn(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write_program",
                        name="write_file",
                        arguments=json.dumps(
                            {"path": "hello.py", "content": "print('Hello, Agent!')\n"}
                        ),
                    )
                ],
            )

        tool_result = json.loads(messages[-1]["content"])
        assert tool_result["ok"] is True
        if self.call_count == 2:
            assert tool_result["action"] == "created"
            return ModelTurn(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_run_program",
                        name="run_command",
                        arguments=json.dumps(
                            {"argv": [sys.executable, "hello.py"], "cwd": "."}
                        ),
                    )
                ],
            )

        assert tool_result["exit_code"] == 0
        assert tool_result["stdout"] == "Hello, Agent!\n"
        return ModelTurn(
            content="Created hello.py and verified its output.",
            finish_reason="stop",
        )


class RenameClient:
    """Fake model that renames a file before reporting completion."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, messages, tools):
        self.call_count += 1
        if self.call_count == 1:
            return ModelTurn(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_rename",
                        name="rename_file",
                        arguments=json.dumps(
                            {
                                "source": "bubble_sort.py",
                                "destination": "selection_sort.py",
                            }
                        ),
                    )
                ],
            )

        tool_result = json.loads(messages[-1]["content"])
        assert tool_result["ok"] is True
        assert tool_result["action"] == "renamed"
        return ModelTurn(
            content="Renamed bubble_sort.py to selection_sort.py.",
            finish_reason="stop",
        )


def test_minimal_vertical_loop(tmp_path):
    (tmp_path / "README.md").write_text(
        "The main implementation is calculator.py.", encoding="utf-8"
    )
    client = FakeClient()
    agent = Agent(client, tmp_path, max_steps=4)

    result = agent.run("Read README.md and identify the implementation file.")

    assert result.final_answer == "The implementation is in calculator.py."
    assert result.steps == 2
    assert result.tool_calls == 1
    assert client.call_count == 2


def test_agent_enforces_step_limit(tmp_path):
    (tmp_path / "README.md").write_text("demo", encoding="utf-8")
    agent = Agent(EndlessToolClient(), tmp_path, max_steps=2)

    with pytest.raises(AgentError, match="step limit"):
        agent.run("Keep reading forever")


def test_agent_can_write_and_read_back_a_file(tmp_path):
    client = WriteThenReadClient()
    agent = Agent(client, tmp_path, max_steps=5)

    result = agent.run("Create hello.py and verify its contents.")

    assert result.final_answer == "Created hello.py and verified its contents."
    assert result.steps == 3
    assert result.tool_calls == 2
    assert client.call_count == 3
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == (
        "print('Hello, Agent!')\n"
    )


def test_agent_can_write_and_execute_a_program(tmp_path):
    client = WriteThenRunClient()
    agent = Agent(client, tmp_path, max_steps=5)

    result = agent.run("Create hello.py, run it, and verify its output.")

    assert result.final_answer == "Created hello.py and verified its output."
    assert result.steps == 3
    assert result.tool_calls == 2
    assert client.call_count == 3


def test_agent_can_rename_a_file(tmp_path):
    source = tmp_path / "bubble_sort.py"
    source.write_text("def selection_sort(values):\n    return values\n", encoding="utf-8")
    client = RenameClient()
    agent = Agent(client, tmp_path, max_steps=3)

    result = agent.run("Rename bubble_sort.py to selection_sort.py.")

    assert result.final_answer == "Renamed bubble_sort.py to selection_sort.py."
    assert result.steps == 2
    assert result.tool_calls == 1
    assert client.call_count == 2
    assert not source.exists()
    assert (tmp_path / "selection_sort.py").is_file()


class PatchClient:
    """Fake model that applies a localized edit before finishing."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, messages, tools):
        self.call_count += 1
        if self.call_count == 1:
            return ModelTurn(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_patch",
                        name="apply_patch",
                        arguments=json.dumps(
                            {
                                "path": "calculator.py",
                                "patch": "@@\n-return a - b\n+return a + b",
                            }
                        ),
                    )
                ],
            )

        tool_result = json.loads(messages[-1]["content"])
        assert tool_result["ok"] is True
        assert tool_result["action"] == "patched"
        return ModelTurn(
            content="Patched calculator.py.",
            finish_reason="stop",
        )


def test_agent_can_apply_a_localized_patch(tmp_path):
    target = tmp_path / "calculator.py"
    target.write_text("return a - b\n", encoding="utf-8", newline="")
    client = PatchClient()
    agent = Agent(client, tmp_path, max_steps=3)

    result = agent.run("Fix addition in calculator.py.")

    assert result.final_answer == "Patched calculator.py."
    assert result.steps == 2
    assert result.tool_calls == 1
    assert client.call_count == 2
    assert target.read_text(encoding="utf-8") == "return a + b\n"
