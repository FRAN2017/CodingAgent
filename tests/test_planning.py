import json
import sys

import pytest

from coding_agent.agent import AgentError
from coding_agent.checkpoints import CheckpointManager
from coding_agent.planning import JsonPlanStore, PlanExecuteAgent, PlanStep, TaskPlan
from coding_agent.protocol import ModelTurn, ToolCall


class SuccessfulPlanClient:
    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, messages, tools):
        self.call_count += 1
        names = [item["function"]["name"] for item in tools]

        if self.call_count == 1:
            assert names == [
                "read_file",
                "list_files",
                "search_text",
                "submit_plan",
            ]
            return ModelTurn(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="submit",
                        name="submit_plan",
                        arguments=json.dumps(
                            {
                                "steps": [
                                    {
                                        "title": "Inspect the project",
                                        "description": "Read the project instructions.",
                                        "success_criteria": "README contents are observed.",
                                    },
                                    {
                                        "title": "Create and test hello.py",
                                        "description": "Write the program and execute it.",
                                        "success_criteria": "文件已创建并且测试命令退出码为零。",
                                    },
                                ]
                            }
                        ),
                    )
                ],
            )
        if self.call_count == 2:
            assert names[-1] == "finish_step"
            return _tool_turn("read", "read_file", {"path": "README.md"})
        if self.call_count == 3:
            return _tool_turn(
                "finish-1",
                "finish_step",
                {"step_id": "step-1", "summary": "Read README.md."},
            )
        if self.call_count == 4:
            return _tool_turn(
                "write",
                "write_file",
                {"path": "hello.py", "content": "print('hello')\n"},
            )
        if self.call_count == 5:
            return _tool_turn(
                "run",
                "run_command",
                {"argv": [sys.executable, "hello.py"], "cwd": "."},
            )
        if self.call_count == 6:
            return _tool_turn(
                "finish-2",
                "finish_step",
                {
                    "step_id": "step-2",
                    "summary": "Created hello.py and its command exited successfully.",
                },
            )

        assert self.call_count == 7
        assert "finalization phase" in messages[0]["content"]
        return ModelTurn(
            content="Created hello.py and verified that it runs successfully.",
            finish_reason="stop",
        )


def _tool_turn(call_id, name, arguments):
    return ModelTurn(
        content=None,
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments=json.dumps(arguments),
            )
        ],
    )


def test_plan_execute_agent_plans_executes_verifies_and_persists(tmp_path):
    (tmp_path / "README.md").write_text("Demo project.\n", encoding="utf-8")
    client = SuccessfulPlanClient()
    checkpoints = CheckpointManager(tmp_path)
    agent = PlanExecuteAgent(
        client,
        tmp_path,
        max_steps=12,
        checkpoint_manager=checkpoints,
    )

    result = agent.run("Create hello.py and verify it.")

    assert result.final_answer == (
        "Created hello.py and verified that it runs successfully."
    )
    assert result.steps == 7
    assert result.tool_calls == 6
    assert result.plan_id is not None
    assert result.checkpoint_id is not None
    assert (tmp_path / "hello.py").read_text(encoding="utf-8") == (
        "print('hello')\n"
    )
    plan = JsonPlanStore(tmp_path).load(result.plan_id)
    assert plan.status == "completed"
    assert [step.status for step in plan.steps] == ["completed", "completed"]
    assert any("run_command: ok" in item for item in plan.steps[1].evidence)
    task_checkpoints = [item for item in checkpoints.list() if item.kind == "task"]
    assert len(task_checkpoints) == 1


def test_finish_step_requires_successful_test_evidence():
    step = PlanStep(
        step_id="step-1",
        title="Run tests",
        description="Verify the implementation.",
        success_criteria="All tests pass.",
    )
    arguments = json.dumps({"step_id": "step-1", "summary": "Tests pass."})

    result, accepted, _, _ = PlanExecuteAgent._finish_step(
        arguments,
        step,
        [("read_file", {"ok": True})],
    )

    assert accepted is False
    assert "run_command" in result["error"]

    result, accepted, summary, evidence = PlanExecuteAgent._finish_step(
        arguments,
        step,
        [("run_command", {"ok": True, "exit_code": 0})],
    )
    assert result["ok"] is True
    assert accepted is True
    assert summary == "Tests pass."
    assert evidence == ["run_command: ok (exit_code=0)"]


def test_finish_step_requires_verification_after_latest_change():
    step = PlanStep(
        step_id="step-1",
        title="Fix and test code",
        description="Modify the implementation and verify it.",
        success_criteria="The modified file passes tests.",
    )
    arguments = json.dumps({"step_id": "step-1", "summary": "Fixed."})

    result, accepted, _, _ = PlanExecuteAgent._finish_step(
        arguments,
        step,
        [
            ("run_command", {"ok": True, "exit_code": 0}),
            ("apply_patch", {"ok": True, "changed": True}),
        ],
    )

    assert accepted is False
    assert "after the latest file change" in result["error"]


def test_json_plan_store_round_trip(tmp_path):
    plan = TaskPlan.create(
        workspace=str(tmp_path.resolve()),
        task="Inspect the repository",
        steps=[
            PlanStep(
                step_id="step-1",
                title="Inspect",
                description="List files.",
                success_criteria="Files are listed.",
            )
        ],
        session_id="demo",
        checkpoint_id="cp-example",
    )
    store = JsonPlanStore(tmp_path)

    store.save(plan)
    restored = store.load(plan.plan_id)

    assert restored.to_dict() == plan.to_dict()
    assert (tmp_path / ".coding-agent" / "plans" / f"{plan.plan_id}.json").is_file()


class SlowDiscoveryPlanClient:
    """Inspect for five turns, then obey the forced submission transition."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, messages, tools):
        self.call_count += 1
        names = [item["function"]["name"] for item in tools]
        if self.call_count <= 5:
            assert names == [
                "read_file",
                "list_files",
                "search_text",
                "submit_plan",
            ]
            if self.call_count > 1:
                assert "Discovery results from earlier rounds" in messages[0]["content"]
            return _tool_turn(
                f"discover-{self.call_count}",
                "list_files",
                {"path": ".", "max_depth": 1},
            )
        if self.call_count == 6:
            assert names == ["submit_plan"]
            assert "FINAL PLAN SUBMISSION ROUND" in messages[0]["content"]
            return _tool_turn(
                "submit-after-discovery",
                "submit_plan",
                {
                    "steps": [
                        {
                            "title": "Inspect workspace",
                            "description": "Observe the workspace contents.",
                            "success_criteria": "A directory listing is available.",
                        },
                        {
                            "title": "Confirm workspace",
                            "description": "Confirm the discovered structure.",
                            "success_criteria": "The structure is observed locally.",
                        },
                    ]
                },
            )
        if self.call_count == 7:
            return _tool_turn("step-1-list", "list_files", {"path": "."})
        if self.call_count == 8:
            return _tool_turn(
                "finish-slow-1",
                "finish_step",
                {"step_id": "step-1", "summary": "Listed the workspace."},
            )
        if self.call_count == 9:
            return _tool_turn("step-2-list", "list_files", {"path": "."})
        if self.call_count == 10:
            return _tool_turn(
                "finish-slow-2",
                "finish_step",
                {"step_id": "step-2", "summary": "Confirmed the workspace."},
            )
        assert self.call_count == 11
        return ModelTurn(content="Workspace inspection completed.")


def test_planner_allows_discovery_then_forces_final_submission(tmp_path):
    client = SlowDiscoveryPlanClient()
    agent = PlanExecuteAgent(client, tmp_path, max_steps=15)

    result = agent.run("Inspect this empty workspace carefully.")

    assert result.final_answer == "Workspace inspection completed."
    assert result.steps == 11
    assert result.plan_id is not None
    assert JsonPlanStore(tmp_path).load(result.plan_id).status == "completed"


class NeverSubmitPlanClient:
    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, messages, tools):
        self.call_count += 1
        return _tool_turn(
            f"ignored-schema-{self.call_count}",
            "list_files",
            {"path": "."},
        )


def test_planner_failure_reports_actual_tool_trace(tmp_path):
    agent = PlanExecuteAgent(NeverSubmitPlanClient(), tmp_path, max_steps=10)

    with pytest.raises(AgentError) as captured:
        agent.run("Inspect the workspace.")

    message = str(captured.value)
    assert "did not call submit_plan" in message
    assert "turn 1=[list_files]" in message
    assert "turn 6=[list_files]" in message
