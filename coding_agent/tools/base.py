"""Shared definitions for model-facing local tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ToolResult = dict[str, Any]
ToolHandler = Callable[[Path, Any], ToolResult]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Describe one tool and connect its schema to its local handler."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        """Return the OpenAI-compatible function tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }
