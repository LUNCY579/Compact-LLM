from __future__ import annotations

import re
from typing import Protocol

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenEstimator(Protocol):
    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


class ConservativeTokenEstimator:
    """Tokenizer-free estimate; inject the exact Rineya tokenizer for release measurements."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        lexical = len(_TOKEN_PATTERN.findall(text))
        byte_floor = (len(text.encode("utf-8")) + 3) // 4
        return max(lexical, byte_floor)

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        if self.count(text) <= max_tokens:
            return text
        suffix = " …[truncated]"
        include_suffix = self.count(suffix) <= max_tokens
        low, high = 0, len(text)
        while low < high:
            midpoint = (low + high + 1) // 2
            base = text[:midpoint].rstrip()
            candidate = base + suffix if include_suffix and base else base
            if self.count(candidate) <= max_tokens:
                low = midpoint
            else:
                high = midpoint - 1
        base = text[:low].rstrip()
        result = base + suffix if include_suffix and base else base
        # Very small budgets can make even one Unicode code point too expensive under the byte floor.
        while result and self.count(result) > max_tokens:
            result = result[:-1].rstrip()
        return result
