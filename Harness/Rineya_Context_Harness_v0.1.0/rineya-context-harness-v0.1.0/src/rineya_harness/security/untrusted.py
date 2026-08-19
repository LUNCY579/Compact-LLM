from __future__ import annotations

import re
from dataclasses import dataclass


_PATTERNS = {
    "instruction_override": re.compile(
        r"\b(ignore|disregard|override)\b.{0,40}\b(previous|system|developer|instructions?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "tool_request": re.compile(
        r"\b(call|invoke|execute|run)\b.{0,30}\b(tool|shell|command|function)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "secret_request": re.compile(
        r"\b(api key|password|secret|token|credentials?)\b.{0,40}\b(send|reveal|print|upload)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "role_impersonation": re.compile(
        r"\b(system message|developer message|you are now|new instructions)\b",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True, slots=True)
class InjectionScan:
    suspicious: bool
    flags: tuple[str, ...]


def scan_untrusted_text(text: str) -> InjectionScan:
    flags = tuple(name for name, pattern in _PATTERNS.items() if pattern.search(text))
    return InjectionScan(suspicious=bool(flags), flags=flags)
