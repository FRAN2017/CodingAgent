"""Tool discovery, argument validation, and dispatch."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from coding_agent.tools.base import ToolSpec
from coding_agent.tools.command import RUN_COMMAND_TOOL
from coding_agent.tools.filesystem import FILE_TOOLS
from coding_agent.tools.patch import APPLY_PATCH_TOOL
from coding_agent.tools.rename import RENAME_FILE_TOOL
from coding_agent.tools.search import SEARCH_TEXT_TOOL

DEFAULT_TOOLS = (
    *FILE_TOOLS,
    RENAME_FILE_TOOL,
    SEARCH_TEXT_TOOL,
    RUN_COMMAND_TOOL,
    APPLY_PATCH_TOOL,
)


class ToolRegistry:
    """Validate model arguments and dispatch registered local tools."""

    def __init__(
        self,
        workspace: Path,
        tools: Iterable[ToolSpec] = DEFAULT_TOOLS,
    ) -> None:
        self.workspace = workspace.resolve()
        self._tools = {tool.name: tool for tool in tools}

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, raw_arguments: str) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool: {name}"}

        try:
            decoded = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"Invalid JSON arguments: {exc.msg}"}
        if not isinstance(decoded, dict):
            return {"ok": False, "error": "Tool arguments must be a JSON object"}

        try:
            arguments = tool.input_model.model_validate(decoded)
            return tool.handler(self.workspace, arguments)
        except (ValidationError, ValueError, OSError) as exc:
            return {"ok": False, "error": str(exc)}