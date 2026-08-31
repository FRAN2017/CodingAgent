"""Atomic JSON persistence for local task plans."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from coding_agent.planning.models import PlanError, TaskPlan


class JsonPlanStore:
    """Store plans below protected workspace state, outside model-facing tools."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.directory = self.workspace / ".coding-agent" / "plans"

    def save(self, plan: TaskPlan) -> None:
        if Path(plan.workspace).resolve() != self.workspace:
            raise PlanError("Plan belongs to a different workspace")
        self._ensure_directory()
        destination = self._path(plan.plan_id)
        temporary = self.directory / f".{destination.name}.{uuid4().hex}.tmp"
        payload = json.dumps(
            plan.to_dict(), ensure_ascii=False, indent=2
        ) + "\n"
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except OSError as exc:
            raise PlanError(f"Cannot save plan {plan.plan_id}: {exc}") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def load(self, plan_id: str) -> TaskPlan:
        path = self._path(plan_id)
        if path.is_symlink():
            raise PlanError("Plan files must not be symbolic links")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PlanError(f"Plan does not exist: {plan_id}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise PlanError(f"Cannot load plan {plan_id}: {exc}") from exc
        plan = TaskPlan.from_dict(raw)
        if plan.plan_id != plan_id:
            raise PlanError("Plan id does not match its filename")
        if Path(plan.workspace).resolve() != self.workspace:
            raise PlanError("Plan belongs to a different workspace")
        return plan

    def _ensure_directory(self) -> None:
        state = self.workspace / ".coding-agent"
        if state.is_symlink() or self.directory.is_symlink():
            raise PlanError("Plan state directories must not be symbolic links")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PlanError(f"Cannot create plan directory: {exc}") from exc

    def _path(self, plan_id: str) -> Path:
        if (
            not isinstance(plan_id, str)
            or not plan_id.startswith("plan-")
            or not plan_id[5:].isalnum()
        ):
            raise PlanError(f"Invalid plan id: {plan_id!r}")
        return self.directory / f"{plan_id}.json"
