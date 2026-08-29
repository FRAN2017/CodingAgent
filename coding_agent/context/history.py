"""Complete in-memory conversation history and atomic interaction blocks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


class HistoryError(ValueError):
    """Raised when messages would form an invalid tool-call sequence."""


@dataclass(frozen=True, slots=True)
class ConversationBlock:
    """Messages that must be retained or compacted as one unit."""

    kind: str
    messages: tuple[dict[str, Any], ...]


class ConversationHistory:
    """Append-only full history used for audit and request construction."""

    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []
        self._pending_tool_call_ids: set[str] = set()

    @classmethod
    def for_task(cls, system_prompt: str, task: str) -> ConversationHistory:
        history = cls()
        history.append_system(system_prompt)
        history.append_user(task)
        return history

    @classmethod
    def from_messages(
        cls,
        messages: list[dict[str, Any]],
    ) -> ConversationHistory:
        """Rebuild and validate history loaded from persistent storage."""
        if not isinstance(messages, list) or not messages:
            raise HistoryError("Conversation history must be a non-empty list")

        history = cls()
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise HistoryError(f"Message {index} must be an object")
            role = message.get("role")
            cls._validate_message_keys(message, role, index)
            if role == "system":
                if index != 0:
                    raise HistoryError("The system message must be first and unique")
                history.append_system(cls._required_content(message, index))
            elif role == "user":
                history.append_user(cls._required_content(message, index))
            elif role == "assistant":
                history.append_assistant(message)
            elif role == "tool":
                call_id = message.get("tool_call_id")
                content = message.get("content")
                if not isinstance(call_id, str) or not call_id:
                    raise HistoryError(
                        f"Tool message {index} must have a non-empty tool_call_id"
                    )
                if not isinstance(content, str):
                    raise HistoryError(f"Tool message {index} content must be a string")
                history.append_tool(call_id, content)
            else:
                raise HistoryError(f"Message {index} has unsupported role: {role!r}")

        if messages[0].get("role") != "system":
            raise HistoryError("Conversation history must begin with a system message")
        history.ensure_complete()
        return history

    @property
    def messages(self) -> list[dict[str, Any]]:
        return deepcopy(self._messages)

    def append_system(self, content: str) -> None:
        self._append_non_tool({"role": "system", "content": content})

    def append_user(self, content: str) -> None:
        self._append_non_tool({"role": "user", "content": content})

    def append_assistant(self, message: dict[str, Any]) -> None:
        if self._pending_tool_call_ids:
            raise HistoryError(
                "Cannot append an assistant message before all tool results arrive"
            )
        if message.get("role") != "assistant":
            raise HistoryError("Assistant message must have role='assistant'")

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise HistoryError("Assistant content must be a string or null")
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise HistoryError("Assistant reasoning_content must be a string or null")

        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls is None:
            tool_calls = []
        elif isinstance(raw_tool_calls, list):
            tool_calls = raw_tool_calls
        else:
            raise HistoryError("Assistant tool_calls must be a list")
        call_ids: list[str] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                raise HistoryError("Every tool call must be an object")
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise HistoryError("Every tool call must have a non-empty string id")
            if call.get("type") != "function":
                raise HistoryError("Every tool call must have type='function'")
            function = call.get("function")
            if not isinstance(function, dict):
                raise HistoryError("Every tool call must contain a function object")
            extra_call_keys = set(call) - {"id", "type", "function"}
            if extra_call_keys:
                raise HistoryError(
                    "Tool call contains unsupported fields: "
                    + ", ".join(sorted(extra_call_keys))
                )
            extra_function_keys = set(function) - {"name", "arguments"}
            if extra_function_keys:
                raise HistoryError(
                    "Tool call function contains unsupported fields: "
                    + ", ".join(sorted(extra_function_keys))
                )
            if not isinstance(function.get("name"), str) or not function["name"]:
                raise HistoryError("Every tool call must have a function name")
            if not isinstance(function.get("arguments"), str):
                raise HistoryError("Tool call arguments must be a JSON string")
            call_ids.append(call_id)
        if len(call_ids) != len(set(call_ids)):
            raise HistoryError("Tool call ids must be unique within an assistant turn")
        if not call_ids and (not isinstance(content, str) or not content.strip()):
            raise HistoryError(
                "Assistant message must contain tool calls or non-empty content"
            )

        self._messages.append(deepcopy(message))
        self._pending_tool_call_ids = set(call_ids)

    def append_tool(self, tool_call_id: str, content: str) -> None:
        if tool_call_id not in self._pending_tool_call_ids:
            raise HistoryError(
                f"Tool result does not match a pending tool call: {tool_call_id}"
            )
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )
        self._pending_tool_call_ids.remove(tool_call_id)

    def ensure_complete(self) -> None:
        if self._pending_tool_call_ids:
            pending = ", ".join(sorted(self._pending_tool_call_ids))
            raise HistoryError(f"Missing tool results for: {pending}")

    def blocks(self) -> list[ConversationBlock]:
        self.ensure_complete()
        blocks: list[ConversationBlock] = []
        index = 0
        while index < len(self._messages):
            message = self._messages[index]
            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                call_ids = {
                    call["id"] for call in message.get("tool_calls", [])
                }
                grouped = [message]
                index += 1
                while index < len(self._messages):
                    candidate = self._messages[index]
                    if candidate.get("role") != "tool":
                        break
                    if candidate.get("tool_call_id") not in call_ids:
                        break
                    grouped.append(candidate)
                    index += 1
                blocks.append(
                    ConversationBlock(
                        kind="tool_interaction",
                        messages=tuple(deepcopy(grouped)),
                    )
                )
                continue

            blocks.append(
                ConversationBlock(
                    kind=str(role),
                    messages=(deepcopy(message),),
                )
            )
            index += 1
        return blocks

    def _append_non_tool(self, message: dict[str, Any]) -> None:
        if self._pending_tool_call_ids:
            raise HistoryError(
                "Cannot append a new message before all tool results arrive"
            )
        self._messages.append(deepcopy(message))

    @staticmethod
    def _required_content(message: dict[str, Any], index: int) -> str:
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise HistoryError(f"Message {index} content must be a non-empty string")
        return content

    @staticmethod
    def _validate_message_keys(message: dict[str, Any], role: Any, index: int) -> None:
        allowed_by_role = {
            "system": {"role", "content"},
            "user": {"role", "content"},
            "assistant": {"role", "content", "reasoning_content", "tool_calls"},
            "tool": {"role", "tool_call_id", "content"},
        }
        allowed = allowed_by_role.get(role)
        if allowed is None:
            return
        extra = set(message) - allowed
        if extra:
            raise HistoryError(
                f"Message {index} contains unsupported fields: "
                + ", ".join(sorted(extra))
            )
