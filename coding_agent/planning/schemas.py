"""Model-facing control tools used only by the Plan-and-Execute controller."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PlannedStepInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1_000)
    success_criteria: str = Field(min_length=1, max_length=500)


class SubmitPlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[PlannedStepInput] = Field(min_length=1, max_length=6)


class FinishStepInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1, max_length=40)
    summary: str = Field(min_length=1, max_length=1_000)


def function_schema(
    name: str,
    description: str,
    input_model: type[BaseModel],
) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_model.model_json_schema(),
        },
    }


SUBMIT_PLAN_SCHEMA = function_schema(
    "submit_plan",
    "Submit the ordered linear plan. This does not modify the workspace.",
    SubmitPlanInput,
)

FINISH_STEP_SCHEMA = function_schema(
    "finish_step",
    "Request completion of the current plan step after gathering local evidence.",
    FinishStepInput,
)
