"""Build bounded model request contexts from complete conversation history."""

from __future__ import annotations

from typing import Any

from coding_agent.context.config import ContextConfig
from coding_agent.context.history import ConversationBlock, ConversationHistory
from coding_agent.context.summary import ToolEventSummarizer
from coding_agent.context.token_counter import TokenCounter


class ContextBudgetError(RuntimeError):
    """Raised when required recent messages cannot fit the configured budget."""


class ContextManager:
    """Create a compact request view without mutating the full history."""

    def __init__(
        self,
        config: ContextConfig | None = None,
        token_counter: TokenCounter | None = None,
        summarizer: ToolEventSummarizer | None = None,
    ) -> None:
        self.config = config or ContextConfig()
        self.token_counter = token_counter or TokenCounter()
        self.summarizer = summarizer or ToolEventSummarizer()

    def build_request(
        self,
        history: ConversationHistory,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        full_messages = history.messages
        available = self.config.input_token_budget - self.token_counter.count_tools(
            tools
        )
        if available < 1:
            raise ContextBudgetError(
                "Tool schemas consume the entire configured input token budget"
            )
        if self.token_counter.count_messages(full_messages) <= available:
            return full_messages

        blocks = history.blocks()
        if not blocks or blocks[0].kind != "system":
            raise ContextBudgetError("Conversation must begin with a system message")
        latest_user_index = self._latest_user_index(blocks)
        fixed_system = list(blocks[0].messages)
        fixed_user = list(blocks[latest_user_index].messages)
        earlier_blocks = blocks[1:latest_user_index]
        current_blocks = blocks[latest_user_index + 1 :]

        maximum_recent = min(self.config.recent_blocks, len(current_blocks))
        minimum_recent = 1 if current_blocks else 0
        summary_limits = self._summary_limits()
        smallest_candidate: list[dict[str, Any]] | None = None
        for recent_count in range(maximum_recent, minimum_recent - 1, -1):
            if recent_count:
                old_blocks = [*earlier_blocks, *current_blocks[:-recent_count]]
                recent_blocks = current_blocks[-recent_count:]
            else:
                old_blocks = [*earlier_blocks, *current_blocks]
                recent_blocks = []
            for summary_limit in summary_limits:
                candidate = self._candidate(
                    fixed_system,
                    fixed_user,
                    old_blocks,
                    recent_blocks,
                    summary_limit,
                )
                smallest_candidate = candidate
                if self.token_counter.count_messages(candidate) <= available:
                    return candidate

        self._raise_too_large(smallest_candidate or full_messages, available)

    def _candidate(
        self,
        fixed_system: list[dict[str, Any]],
        fixed_user: list[dict[str, Any]],
        old_blocks: list[ConversationBlock],
        recent_blocks: list[ConversationBlock],
        summary_limit: int,
    ) -> list[dict[str, Any]]:
        candidate = [*fixed_system]
        if old_blocks:
            candidate.append(
                {
                    "role": "system",
                    "content": self.summarizer.summarize(
                        old_blocks,
                        max_chars=summary_limit,
                    ),
                }
            )
        candidate.extend(fixed_user)
        for block in recent_blocks:
            candidate.extend(block.messages)
        return candidate

    def _latest_user_index(self, blocks: list[ConversationBlock]) -> int:
        for index in range(len(blocks) - 1, 0, -1):
            if blocks[index].kind == "user":
                return index
        raise ContextBudgetError("Conversation is missing a user message")

    def _summary_limits(self) -> list[int]:
        limits: list[int] = []
        value = self.config.summary_max_chars
        while value >= 512:
            limits.append(value)
            value //= 2
        if 256 not in limits:
            limits.append(256)
        return limits

    def _raise_too_large(
        self,
        messages: list[dict[str, Any]],
        available: int,
    ) -> None:
        estimated = self.token_counter.count_messages(messages)
        raise ContextBudgetError(
            "Required system, latest user, and recent interaction messages exceed the "
            f"context budget (estimated={estimated}, available={available})"
        )
