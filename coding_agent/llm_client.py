"""Thin DeepSeek adapter. Agent behavior lives outside this module."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from coding_agent.config import Config
from coding_agent.protocol import ModelTurn, ToolCall


class DeepSeekClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout_seconds,
            max_retries=2,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": False,
        }
        if self.config.thinking_enabled:
            request["reasoning_effort"] = self.config.reasoning_effort
            request["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            request["extra_body"] = {"thinking": {"type": "disabled"}}

        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message

        calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
            for call in (message.tool_calls or [])
        ]

        return ModelTurn(
            content=message.content,
            tool_calls=calls,
            finish_reason=choice.finish_reason,
            reasoning_content=getattr(message, "reasoning_content", None),
        )
