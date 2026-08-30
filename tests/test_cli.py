from typer.testing import CliRunner

from coding_agent.agent import AgentError, AgentResult
from coding_agent.checkpoints import CheckpointManager
from coding_agent.cli import Provider, app, create_client
from coding_agent.llm_client import DeepSeekClient, QianwenClient

runner = CliRunner()


def test_create_client_selects_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-test-model")

    client, model = create_client(Provider.deepseek)

    assert isinstance(client, DeepSeekClient)
    assert model == "deepseek-test-model"


def test_create_client_selects_qianwen(monkeypatch):
    monkeypatch.setenv("QIANWEN_API_KEY", "qianwen-test-key")
    monkeypatch.setenv("QIANWEN_MODEL", "qianwen-test-model")

    client, model = create_client(Provider.qianwen)

    assert isinstance(client, QianwenClient)
    assert model == "qianwen-test-model"


def test_cli_prints_concise_agent_error_without_traceback(tmp_path, monkeypatch):
    class FailingAgent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, task):
            raise AgentError("Qianwen request timed out")

    monkeypatch.setattr(
        "coding_agent.cli.create_client",
        lambda provider: (object(), "test-model"),
    )
    monkeypatch.setattr("coding_agent.cli.Agent", FailingAgent)

    result = runner.invoke(
        app,
        ["run", "test task", "--workspace", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "Error: Qianwen request timed out" in result.output
    assert "Traceback" not in result.output


def test_cli_without_task_enters_interactive_mode_and_reuses_agent(
    tmp_path,
    monkeypatch,
):
    instances = []

    class InteractiveAgent:
        def __init__(self, *args, **kwargs):
            self.tasks = []
            instances.append(self)

        def run(self, task):
            self.tasks.append(task)
            return AgentResult(
                final_answer=f"answer: {task}",
                steps=1,
                tool_calls=0,
                messages=[],
            )

    monkeypatch.setattr(
        "coding_agent.cli.create_client",
        lambda provider: (object(), "test-model"),
    )
    monkeypatch.setattr("coding_agent.cli.Agent", InteractiveAgent)

    result = runner.invoke(
        app,
        ["run", "--workspace", str(tmp_path), "--session", "interactive"],
        input="\nfirst task\nsecond task\nquit\n",
    )

    assert result.exit_code == 0
    assert len(instances) == 1
    assert instances[0].tasks == ["first task", "second task"]
    assert "Interactive mode" in result.output
    assert "answer: first task" in result.output
    assert "answer: second task" in result.output
    assert "会话已结束" in result.output


def test_interactive_mode_reports_task_error_and_accepts_next_question(
    tmp_path,
    monkeypatch,
):
    class RecoveringAgent:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def run(self, task):
            self.calls += 1
            if self.calls == 1:
                raise AgentError("temporary failure")
            return AgentResult(
                final_answer="recovered",
                steps=1,
                tool_calls=0,
                messages=[],
            )

    monkeypatch.setattr(
        "coding_agent.cli.create_client",
        lambda provider: (object(), "test-model"),
    )
    monkeypatch.setattr("coding_agent.cli.Agent", RecoveringAgent)

    result = runner.invoke(
        app,
        ["run", "--workspace", str(tmp_path)],
        input="failing task\nnext task\nexit\n",
    )

    assert result.exit_code == 0
    assert "Error: temporary failure" in result.output
    assert "recovered" in result.output


def test_interactive_diff_and_undo_commands_restore_workspace(tmp_path, monkeypatch):
    target = tmp_path / "code.py"
    target.write_text("before\n", encoding="utf-8")
    checkpoint = CheckpointManager(tmp_path).create("before edit")
    target.write_text("after\n", encoding="utf-8")

    class UnusedAgent:
        def __init__(self, *args, **kwargs):
            self.restore_events = []

        def run(self, task):
            raise AssertionError("Slash commands must not be sent to the model")

        def record_workspace_restore(self, result):
            self.restore_events.append(result)

    monkeypatch.setattr(
        "coding_agent.cli.create_client",
        lambda provider: (object(), "test-model"),
    )
    monkeypatch.setattr("coding_agent.cli.Agent", UnusedAgent)

    result = runner.invoke(
        app,
        ["run", "--workspace", str(tmp_path)],
        input=f"/diff {checkpoint.checkpoint_id}\n/undo {checkpoint.checkpoint_id}\ny\nquit\n",
    )

    assert result.exit_code == 0
    assert "M code.py" in result.output
    assert f"Restored {checkpoint.checkpoint_id}" in result.output
    assert "Workspace restore event recorded" in result.output
    assert target.read_text(encoding="utf-8") == "before\n"
