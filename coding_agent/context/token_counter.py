"""Dependency-free conservative token estimation."""

from __future__ import annotations

import json
import math
from typing import Any


class TokenCounter:
    """Estimate tokens without relying on a provider-specific tokenizer."""

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text.encode("utf-8")) / 3)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        serialized = json.dumps(
            messages,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return self.count_text(serialized) + 4 * len(messages)

    def count_tools(self, tools: list[dict[str, Any]]) -> int:
        serialized = json.dumps(
            tools,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return self.count_text(serialized)
