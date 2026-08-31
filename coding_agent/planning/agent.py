"""A small, self-managed linear Plan-and-Execute agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from coding_agent.agent import Agent, AgentError, AgentResult
from coding_agent.planning.models import PlanError, PlanStep, TaskPlan
from coding_agent.planning.schemas import (
    FINISH_STEP_SCHEMA,
    SUBMIT_PLAN_SCHEMA,
    FinishStepInput,
    SubmitPlanInput,
)
from coding_agent.planning.store import JsonPlanStore
from coding_agent.protocol import ModelClientError, ModelTurn
from coding_agent.tools.registry import DEFAULT_TOOLS, ToolRegistry

READ_ONLY_TOOL_NAMES = {"read_file", "list_files", "search_text"}
MUTATING_TOOL_NAMES = {"write_file", "rename_file", "apply_patch"}
TEST_WORDS = {
    "test",
    "tests",
    "pytest",
    "unittest",
    "lint",
    "ruff",
    "check",
    "测试",
    "验证",
}
MUTATION_WORDS = {
    "implement",
    "modify",
    "create",
    "write",
    "rename",
    "fix",
    "实现",
    "修改",
    "创建",
    "写入",
    "重命名",
    "修复",
}
MAX_PLANNER_TURNS = 6
MAX_STEP_TURNS = 5

PLANNER_PROMPT = """\
Plan-and-Execute planning phase (trusted local controller instruction):
You are planning the user's current programming task. You may inspect the
workspace only with read_file, list_files, and search_text. Do not modify files
or run commands during planning. Produce an ordered, concrete linear plan with
2 to 6 independently checkable steps. Each success criterion must describe
observable local evidence. You must submit the plan with submit_plan; ordinary
assistant text does not finish planning.
"""


@dataclass(slots=True)
class _RunStats:
    turns: int = 0
    tool_calls: int = 0


class PlanExecuteAgent(Agent):
    """Plan first, then execute and locally verify one linear step at a time."""

    def __init__(self, *args: Any, plan_store: JsonPlanStore | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.max_steps < 6:
            raise ValueError("plan-execute mode requires max_steps to be at least 6")
        self.plan_store = plan_store or JsonPlanStore(self.workspace)
        if self.plan_store.workspace != self.workspace:
            raise ValueError("plan_store belongs to a different workspace")
        read_only_specs = [
            tool for tool in DEFAULT_TOOLS if tool.name in READ_ONLY_TOOL_NAMES
        ]
        self.read_only_tools = ToolRegistry(self.workspace, read_only_specs)

    def run(self, task: str) -> AgentResult:
        task = task.strip()
        if not task:
            raise ValueError("task must not be empty")
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace is not a directory: {self.workspace}")

        checkpoint = None
        if self.checkpoint_manager is not None:
            checkpoint = self.checkpoint_manager.create(
                task,
                session_id=self.session_id,
            )
        history = self._prepare_history(task)
        stats = _RunStats()
        plan: TaskPlan | None = None

        try:
            planned_steps = self._request_plan(
                history,
                stats,
                minimum_steps=2,
                prompt=PLANNER_PROMPT,
            )
            self._assign_step_ids(planned_steps, start=1)
            plan = TaskPlan.create(
                workspace=str(self.workspace),
                task=task,
                steps=planned_steps,
                session_id=self.session_id,
                checkpoint_id=(
                    checkpoint.checkpoint_id if checkpoint is not None else None
                ),
            )
            self.plan_store.save(plan)

            replanned = False
            while step := next(
                (item for item in plan.steps if item.status == "pending"),
                None,
            ):
                step.status = "in_progress"
                plan.touch()
                self.plan_store.save(plan)
                failure = self._execute_step(history, plan, step, stats)
                if failure is None:
                    continue

                step.status = "failed"
                step.summary = failure
                plan.touch()
                self.plan_store.save(plan)
                if replanned or self.max_steps - stats.turns < 3:
                    raise AgentError(
                        f"Plan step {step.step_id} failed: {failure}"
                    )

                replacements = self._request_plan(
                    history,
                    stats,
                    minimum_steps=1,
                    prompt=self._replan_prompt(plan, step, failure),
                )
                for old_step in plan.steps:
                    if old_step.status in {"pending", "failed"}:
                        old_step.status = "superseded"
                self._assign_step_ids(replacements, start=len(plan.steps) + 1)
                plan.steps.extend(replacements)
                plan.revision += 1
                replanned = True
                plan.touch()
                self.plan_store.save(plan)

            plan.status = "completed"
            plan.touch()
            self.plan_store.save(plan)
            final_answer = self._request_final_answer(history, plan, stats)
            changes = (
                list(self.checkpoint_manager.diff(checkpoint.checkpoint_id).changes)
                if self.checkpoint_manager is not None and checkpoint is not None
                else []
            )
            return AgentResult(
                final_answer=final_answer,
                steps=stats.turns,
                tool_calls=stats.tool_calls,
                messages=history.messages,
                checkpoint_id=(
                    checkpoint.checkpoint_id if checkpoint is not None else None
                ),
                changes=changes,
                plan_id=plan.plan_id,
            )
        except (AgentError, PlanError):
            if plan is not None:
                plan.status = "failed"
                plan.touch()
                self.plan_store.save(plan)
            raise

    def _request_plan(
        self,
        history: Any,
        stats: _RunStats,
        *,
        minimum_steps: int,
        prompt: str,
    ) -> list[PlanStep]:
        discovery_schemas = [*self.read_only_tools.schemas, SUBMIT_PLAN_SCHEMA]
        attempts = min(MAX_PLANNER_TURNS, self.max_steps - stats.turns - 3)
        if attempts < 1:
            raise AgentError("Not enough model turns remain to create a plan")

        last_error = "the model did not submit a plan"
        planner_trace: list[str] = []
        observed_tools: list[str] = []
        for attempt_index in range(attempts):
            final_submission_round = attempt_index == attempts - 1
            schemas = (
                [SUBMIT_PLAN_SCHEMA]
                if final_submission_round
                else discovery_schemas
            )
            turn = self._complete(
                history,
                schemas,
                self._planner_prompt(
                    prompt,
                    attempt_index=attempt_index,
                    attempts=attempts,
                    observed_tools=observed_tools,
                ),
                stats,
            )
            history.append_assistant(turn.as_assistant_message())
            if not turn.tool_calls:
                if not turn.content or not turn.content.strip():
                    raise AgentError(
                        "Planner returned neither tool calls nor useful text "
                        f"on turn {attempt_index + 1}"
                    )
                self._save_history(history)
                last_error = "planner returned text instead of submit_plan"
                planner_trace.append(f"turn {attempt_index + 1}=text")
                continue

            tool_names = [call.name for call in turn.tool_calls]
            planner_trace.append(
                f"turn {attempt_index + 1}=[{', '.join(tool_names)}]"
            )
            submit_calls = [
                call for call in turn.tool_calls if call.name == "submit_plan"
            ]
            if not submit_calls:
                last_error = "planner used tools but did not call submit_plan"
            candidate: list[PlanStep] | None = None
            for call in turn.tool_calls:
                stats.tool_calls += 1
                if call.name == "submit_plan":
                    if len(submit_calls) != 1:
                        result = {
                            "ok": False,
                            "error": "Exactly one submit_plan call is required",
                        }
                        last_error = str(result["error"])
                    else:
                        candidate, result = self._parse_submitted_plan(
                            call.arguments,
                            minimum_steps=minimum_steps,
                        )
                        if candidate is None:
                            last_error = str(result["error"])
                elif final_submission_round:
                    result = {
                        "ok": False,
                        "error": (
                            "The final planning round only permits submit_plan"
                        ),
                    }
                else:
                    result = self.read_only_tools.execute(
                        call.name,
                        call.arguments,
                    )
                    observed_tools.append(call.name)
                history.append_tool(
                    call.id,
                    json.dumps(result, ensure_ascii=False),
                )
            self._save_history(history)
            if candidate is not None and len(submit_calls) == 1:
                return candidate

        trace = "; ".join(planner_trace) if planner_trace else "no model turns"
        raise AgentError(
            "Planner failed to produce a valid plan: "
            f"{last_error}. Planner trace: {trace}"
        )

    @staticmethod
    def _planner_prompt(
        base_prompt: str,
        *,
        attempt_index: int,
        attempts: int,
        observed_tools: list[str],
    ) -> str:
        round_number = attempt_index + 1
        if round_number == attempts:
            transition = """\
FINAL PLAN SUBMISSION ROUND: Workspace discovery is now closed. The only
available tool is submit_plan. Use the observations already present in the
conversation and call submit_plan now. Do not return an ordinary text plan.
"""
        elif observed_tools:
            observed = ", ".join(dict.fromkeys(observed_tools))
            transition = f"""\
Discovery results from earlier rounds are already available in the conversation
(tools used: {observed}). Do not repeat an inspection that has already answered
the same question. If there is enough evidence to create a safe plan, call
submit_plan now. Planning round {round_number} of {attempts}.
"""
        else:
            transition = (
                f"Planning round {round_number} of {attempts}. Inspect only if "
                "necessary; otherwise call submit_plan immediately."
            )
        return f"{base_prompt.strip()}\n\n{transition.strip()}"

    def _execute_step(
        self,
        history: Any,
        plan: TaskPlan,
        step: PlanStep,
        stats: _RunStats,
    ) -> str | None:
        schemas = [*self.tools.schemas, FINISH_STEP_SCHEMA]
        records: list[tuple[str, dict[str, Any]]] = []
        for _ in range(MAX_STEP_TURNS):
            pending_after = sum(
                item.status == "pending" for item in plan.steps
            )
            if self.max_steps - stats.turns <= pending_after + 1:
                return "global model-turn budget is exhausted"

            turn = self._complete(
                history,
                schemas,
                self._step_prompt(plan, step),
                stats,
            )
            history.append_assistant(turn.as_assistant_message())
            if not turn.tool_calls:
                if not turn.content or not turn.content.strip():
                    raise AgentError(
                        f"Executor returned no action for {step.step_id}"
                    )
                self._save_history(history)
                continue

            completed = False
            completion_summary: str | None = None
            completion_evidence: list[str] = []
            for call in turn.tool_calls:
                stats.tool_calls += 1
                if call.name == "finish_step":
                    result, accepted, summary, evidence = self._finish_step(
                        call.arguments,
                        step,
                        records,
                    )
                    if accepted:
                        completed = True
                        completion_summary = summary
                        completion_evidence = evidence
                else:
                    result = self.tools.execute(call.name, call.arguments)
                    records.append((call.name, result))
                history.append_tool(
                    call.id,
                    json.dumps(result, ensure_ascii=False),
                )
            self._save_history(history)
            if completed:
                step.status = "completed"
                step.summary = completion_summary
                step.evidence = completion_evidence
                plan.touch()
                self.plan_store.save(plan)
                return None

        return f"step exceeded its {MAX_STEP_TURNS}-turn limit"

    def _request_final_answer(
        self,
        history: Any,
        plan: TaskPlan,
        stats: _RunStats,
    ) -> str:
        if stats.turns >= self.max_steps:
            raise AgentError("No model turn remains for the final answer")
        summaries = "\n".join(
            f"- {step.step_id} {step.title}: {step.summary}"
            for step in plan.steps
            if step.status == "completed"
        )
        prompt = f"""\
Plan-and-Execute finalization phase (trusted local controller instruction):
All active plan steps are complete. Do not call any tools. Give the user a
concise final answer based only on the recorded tool evidence. Mention relevant
files and verification results without inventing details.

Completed step summaries:
{summaries}
"""
        turn = self._complete(history, self.tools.schemas, prompt, stats)
        history.append_assistant(turn.as_assistant_message())
        if turn.tool_calls:
            for call in turn.tool_calls:
                stats.tool_calls += 1
                history.append_tool(
                    call.id,
                    json.dumps(
                        {
                            "ok": False,
                            "error": "The plan is complete; finalization forbids tools",
                        },
                        ensure_ascii=False,
                    ),
                )
            self._save_history(history)
            raise AgentError("Model called tools during plan finalization")
        if not turn.content or not turn.content.strip():
            raise AgentError("Model returned an empty final answer")
        self._save_history(history)
        return turn.content.strip()

    def _complete(
        self,
        history: Any,
        schemas: list[dict[str, Any]],
        prompt: str,
        stats: _RunStats,
    ) -> ModelTurn:
        if stats.turns >= self.max_steps:
            raise AgentError(
                f"Agent stopped after reaching the {self.max_steps}-step limit"
            )
        messages = self._build_request(
            history,
            schemas,
            system_addendum=prompt,
        )
        try:
            turn = self.client.complete(messages, schemas)
        except ModelClientError as exc:
            raise AgentError(str(exc)) from exc
        stats.turns += 1
        return turn

    @staticmethod
    def _parse_submitted_plan(
        raw_arguments: str,
        *,
        minimum_steps: int,
    ) -> tuple[list[PlanStep] | None, dict[str, Any]]:
        try:
            decoded = json.loads(raw_arguments)
            submitted = SubmitPlanInput.model_validate(decoded)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            return None, {"ok": False, "error": f"Invalid plan: {exc}"}
        if len(submitted.steps) < minimum_steps:
            return None, {
                "ok": False,
                "error": f"Plan requires at least {minimum_steps} steps",
            }
        steps = [
            PlanStep(
                step_id="unassigned",
                title=item.title,
                description=item.description,
                success_criteria=item.success_criteria,
            )
            for item in submitted.steps
        ]
        return steps, {"ok": True, "accepted_steps": len(steps)}

    @staticmethod
    def _assign_step_ids(steps: list[PlanStep], *, start: int) -> None:
        for index, step in enumerate(steps, start=start):
            step.step_id = f"step-{index}"

    @staticmethod
    def _finish_step(
        raw_arguments: str,
        step: PlanStep,
        records: list[tuple[str, dict[str, Any]]],
    ) -> tuple[dict[str, Any], bool, str | None, list[str]]:
        try:
            decoded = json.loads(raw_arguments)
            requested = FinishStepInput.model_validate(decoded)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            return {"ok": False, "error": f"Invalid completion request: {exc}"}, False, None, []
        if requested.step_id != step.step_id:
            return {
                "ok": False,
                "error": f"Current step is {step.step_id}, not {requested.step_id}",
            }, False, None, []

        successful = [
            item
            for item in records
            if PlanExecuteAgent._is_effective_success(*item)
        ]
        if not successful:
            return {
                "ok": False,
                "error": "At least one successful local tool result is required",
            }, False, None, []
        searchable = f"{step.title} {step.success_criteria}".casefold()
        test_required = PlanExecuteAgent._contains_keyword(searchable, TEST_WORDS)
        mutation_required = PlanExecuteAgent._contains_keyword(
            searchable,
            MUTATION_WORDS,
        )
        command_indexes = [
            index for index, (name, _) in enumerate(records) if name == "run_command"
        ]
        if test_required and (
            not command_indexes
            or not PlanExecuteAgent._is_effective_success(
                *records[command_indexes[-1]]
            )
        ):
            return {
                "ok": False,
                "error": "This step requires the latest run_command to succeed",
            }, False, None, []
        if mutation_required and not any(
            name in MUTATING_TOOL_NAMES for name, _ in successful
        ):
            return {
                "ok": False,
                "error": "This step requires a successful file-changing tool result",
            }, False, None, []
        if test_required and mutation_required:
            mutation_indexes = [
                index
                for index, (name, result) in enumerate(records)
                if name in MUTATING_TOOL_NAMES
                and PlanExecuteAgent._is_effective_success(name, result)
            ]
            if mutation_indexes and command_indexes[-1] < mutation_indexes[-1]:
                return {
                    "ok": False,
                    "error": "Run verification again after the latest file change",
                }, False, None, []

        evidence = [
            PlanExecuteAgent._evidence_line(name, result)
            for name, result in successful
        ]
        return {
            "ok": True,
            "step_id": step.step_id,
            "accepted": True,
            "evidence": evidence,
        }, True, requested.summary, evidence

    @staticmethod
    def _evidence_line(name: str, result: dict[str, Any]) -> str:
        details = []
        for key in ("path", "action", "exit_code", "matches", "entries"):
            if key in result:
                value = result[key]
                if isinstance(value, list):
                    value = len(value)
                details.append(f"{key}={value}")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"{name}: ok{suffix}"

    @staticmethod
    def _contains_keyword(text: str, keywords: set[str]) -> bool:
        for keyword in keywords:
            if keyword.isascii():
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    return True
            elif keyword in text:
                return True
        return False

    @staticmethod
    def _is_effective_success(name: str, result: dict[str, Any]) -> bool:
        if result.get("ok") is not True:
            return False
        if name == "run_command":
            return result.get("exit_code") == 0
        if name in MUTATING_TOOL_NAMES:
            return result.get("changed") is not False
        return True

    @staticmethod
    def _step_prompt(plan: TaskPlan, step: PlanStep) -> str:
        completed = [
            f"{item.step_id}: {item.summary}"
            for item in plan.steps
            if item.status == "completed"
        ]
        prior = "\n".join(completed) if completed else "(none)"
        return f"""\
Plan-and-Execute execution phase (trusted local controller instruction):
Execute only the current step below. Use local tools to gather observable
evidence. Do not declare the entire task complete and do not silently switch to
another step. When this step is genuinely complete, call finish_step with the
exact step_id and a concise summary. The local controller, not the model, decides
whether the evidence is sufficient.

Plan: {plan.plan_id}, revision {plan.revision}
Current step: {step.step_id} - {step.title}
Description: {step.description}
Success criteria: {step.success_criteria}
Previously completed:
{prior}
"""

    @staticmethod
    def _replan_prompt(plan: TaskPlan, step: PlanStep, failure: str) -> str:
        completed = "\n".join(
            f"- {item.step_id} {item.title}: {item.summary}"
            for item in plan.steps
            if item.status == "completed"
        ) or "- none"
        return f"""\
Plan-and-Execute replanning phase (trusted local controller instruction):
The original linear plan could not complete its current step. Create 1 to 6
replacement steps for only the unfinished remainder. Keep completed work intact.
You may inspect using read_file, list_files, and search_text, but may not modify
files or run commands. You must call submit_plan.

Failed step: {step.step_id} - {step.title}
Failure: {failure}
Completed work:
{completed}
"""
