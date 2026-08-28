"""Thin OpenAI-compatible adapters. Agent behavior lives elsewhere."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from coding_agent.config import DeepseekConfig, QianwenConfig
from coding_agent.protocol import ModelTurn, ToolCall


class DeepSeekClient:
    def __init__(self, config: DeepseekConfig) -> None:
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
        extra_body: dict[str, Any] = {}
        if self.config.thinking_enabled:
            extra_body["reasoning_effort"] = self.config.reasoning_effort
            extra_body["thinking"] = {"type": "enabled"}
        else:
            extra_body["thinking"] = {"type": "disabled"}
        request["extra_body"] = extra_body

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


class QianwenClient:
    def __init__(self, config: QianwenConfig) -> None:
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
        extra_body: dict[str, Any] = {
            "enable_thinking": self.config.thinking_enabled,
        }
        if self.config.thinking_enabled:
            extra_body["reasoning_effort"] = self.config.reasoning_effort
        request["extra_body"] = extra_body

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
