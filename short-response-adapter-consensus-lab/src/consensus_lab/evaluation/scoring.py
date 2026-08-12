from __future__ import annotations

import re
from collections import Counter
from typing import Any


ABSTENTION_TEXT = "I don't know based on the available information."


def normalize_answer(text: str) -> str:
    value = (text or "").strip().casefold()
    value = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", value)
    value = re.sub(r"[.!?]+$", "", value)
    return re.sub(r"\s+", " ", value)


def answer_correct(answer: str, row: dict[str, Any]) -> bool:
    candidates = [str(row.get("approved_answer") or row.get("answer") or "")]
    candidates.extend(str(value) for value in (row.get("answer_aliases") or []))
    normalized = normalize_answer(answer)
    return any(normalized == normalize_answer(candidate) for candidate in candidates if normalize_answer(candidate))


def is_abstention(text: str) -> bool:
    normalized = normalize_answer(text)
    return (
        normalized == normalize_answer(ABSTENTION_TEXT)
        or normalized.startswith("i don't know")
        or normalized.startswith("i do not know")
    )


def repetition_failure(token_ids: list[int]) -> bool:
    if len(token_ids) < 6:
        return False
    for n in (1, 2, 3):
        grams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
        counts = Counter(grams)
        if counts and max(counts.values()) >= 4:
            return True
    return False

