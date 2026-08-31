"""Validated state for the first Plan-and-Execute implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

PLAN_FORMAT_VERSION = 1
STEP_STATUSES = {"pending", "in_progress", "completed", "failed", "superseded"}
PLAN_STATUSES = {"planning", "executing", "completed", "failed"}


class PlanError(RuntimeError):
    """Raised when a plan is invalid or cannot be executed safely."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _required_text(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"Plan field {name!r} must be non-empty text")
    return value.strip()


@dataclass(slots=True)
class PlanStep:
    step_id: str
    title: str
    description: str
    success_criteria: str
    status: str = "pending"
    summary: str | None = None
    evidence: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any, *, position: int) -> PlanStep:
        if not isinstance(data, dict):
            raise PlanError(f"Plan step {position} must be an object")
        allowed = {
            "step_id",
            "title",
            "description",
            "success_criteria",
            "status",
            "summary",
            "evidence",
        }
        if set(data) - allowed:
            raise PlanError(f"Plan step {position} has unsupported fields")
        status = data.get("status", "pending")
        if status not in STEP_STATUSES:
            raise PlanError(f"Plan step {position} has invalid status: {status!r}")
        summary = data.get("summary")
        if summary is not None and (not isinstance(summary, str) or not summary.strip()):
            raise PlanError(f"Plan step {position} summary must be text or null")
        evidence = data.get("evidence", [])
        if not isinstance(evidence, list) or any(
            not isinstance(item, str) or not item for item in evidence
        ):
            raise PlanError(f"Plan step {position} evidence must be a text list")
        return cls(
            step_id=_required_text(data, "step_id"),
            title=_required_text(data, "title"),
            description=_required_text(data, "description"),
            success_criteria=_required_text(data, "success_criteria"),
            status=status,
            summary=summary.strip() if summary is not None else None,
            evidence=list(evidence),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "status": self.status,
            "summary": self.summary,
            "evidence": list(self.evidence),
        }


@dataclass(slots=True)
class TaskPlan:
    plan_id: str
    workspace: str
    task: str
    created_at: str
    updated_at: str
    status: str
    revision: int
    steps: list[PlanStep]
    session_id: str | None = None
    checkpoint_id: str | None = None
    format_version: int = PLAN_FORMAT_VERSION

    @classmethod
    def create(
        cls,
        *,
        workspace: str,
        task: str,
        steps: list[PlanStep],
        session_id: str | None,
        checkpoint_id: str | None,
    ) -> TaskPlan:
        timestamp = utc_now()
        return cls(
            plan_id=f"plan-{uuid4().hex[:16]}",
            workspace=workspace,
            task=task,
            created_at=timestamp,
            updated_at=timestamp,
            status="executing",
            revision=1,
            steps=steps,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
        )

    @classmethod
    def from_dict(cls, data: Any) -> TaskPlan:
        if not isinstance(data, dict):
            raise PlanError("Plan document must be a JSON object")
        if data.get("format_version") != PLAN_FORMAT_VERSION:
            raise PlanError(
                f"Unsupported plan format version: {data.get('format_version')!r}"
            )
        status = data.get("status")
        if status not in PLAN_STATUSES:
            raise PlanError(f"Invalid plan status: {status!r}")
        revision = data.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise PlanError("Plan revision must be a positive integer")
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlanError("Plan steps must be a non-empty list")
        steps = [
            PlanStep.from_dict(item, position=index)
            for index, item in enumerate(raw_steps)
        ]
        step_ids = [step.step_id for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise PlanError("Plan step ids must be unique")
        session_id = data.get("session_id")
        checkpoint_id = data.get("checkpoint_id")
        if session_id is not None and not isinstance(session_id, str):
            raise PlanError("Plan session_id must be text or null")
        if checkpoint_id is not None and not isinstance(checkpoint_id, str):
            raise PlanError("Plan checkpoint_id must be text or null")
        for name in ("created_at", "updated_at"):
            value = _required_text(data, name)
            try:
                datetime.fromisoformat(value)
            except ValueError as exc:
                raise PlanError(f"Plan field {name!r} must be ISO-8601") from exc
        return cls(
            format_version=PLAN_FORMAT_VERSION,
            plan_id=_required_text(data, "plan_id"),
            workspace=_required_text(data, "workspace"),
            task=_required_text(data, "task"),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            status=status,
            revision=revision,
            steps=steps,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
        )

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "plan_id": self.plan_id,
            "workspace": self.workspace,
            "task": self.task,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "revision": self.revision,
            "session_id": self.session_id,
            "checkpoint_id": self.checkpoint_id,
            "steps": [step.to_dict() for step in self.steps],
        }
