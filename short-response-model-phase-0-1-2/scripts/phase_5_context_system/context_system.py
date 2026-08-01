# Purpose: Implements deterministic memory retrieval, filtering, budgeting, prompt assembly, and prefix keys.
from __future__ import annotations

"""Deterministic retrieval and bounded context assembly for Phase 5.

The module intentionally has no model dependency.  It can be unit-tested on
CPU and then used by the model-aware evaluator.  Retrieval combines lexical
relevance with explicit salience and recency, while always rejecting stale or
contradicted memories and deduplicating equivalent facts.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class Budget:
    """Independent token budgets for the assembled prompt tiers."""

    total: int = 1024
    memory: int = 256
    recent: int = 512


class JsonlMemoryStore:
    """Minimal append-only memory store used by the local-first prototype."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{self.path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{self.path}:{line_number}: memory row must be an object")
            rows.append(value)
        return rows

    def append(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokens(text: str) -> list[str]:
    """Return stable word/punctuation units for dependency-free budgeting."""

    return re.findall(r"[\w.+-]+|[^\w\s]", text, re.UNICODE)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def memory_text(row: dict[str, Any]) -> str:
    """Return the user-visible memory text included in prompts."""

    return str(row.get("text") or row.get("value") or "").strip()


def memory_search_text(row: dict[str, Any]) -> str:
    """Build the retrieval-only text from the fact plus optional aliases.

    Structured memories often contain a compact fact whose wording does not
    repeat the user's question.  ``keywords``/``aliases`` make those explicit
    relationships auditable without injecting hidden prose into the model
    prompt.  They are metadata only; ``memory_text`` remains the prompt value.
    """

    parts = [memory_text(row)]
    for field in ("keywords", "aliases", "topics"):
        value = row.get(field, [])
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value if str(item).strip())
    return " ".join(part for part in parts if part).strip()


def _lexical_score(query: str, text: str) -> float:
    """Compute a bounded lexical relevance score with phrase bonus."""

    query_tokens = [item.casefold() for item in tokens(query) if item.isalnum()]
    text_tokens = [item.casefold() for item in tokens(text) if item.isalnum()]
    if not query_tokens or not text_tokens:
        return 0.0
    query_set = set(query_tokens)
    text_set = set(text_tokens)
    overlap = len(query_set & text_set)
    # Dice coefficient behaves well for short conversational queries.
    dice = 2.0 * overlap / max(1, len(query_set) + len(text_set))
    phrase_bonus = 0.20 if normalize(query) in normalize(text) or normalize(text) in normalize(query) else 0.0
    return min(1.0, dice + phrase_bonus)


def retrieve_with_trace(
    query: str,
    rows: Iterable[dict[str, Any]],
    top_k: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retrieve memories and return a complete scored trace for auditing."""

    scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    seen_text: set[str] = set()
    for index, row in enumerate(rows):
        text = memory_text(row)
        normalized_text = normalize(text)
        excluded_reason: str | None = None
        if not text:
            excluded_reason = "empty"
        elif row.get("stale"):
            excluded_reason = "stale"
        elif row.get("contradicted"):
            excluded_reason = "contradicted"
        elif normalized_text in seen_text:
            excluded_reason = "duplicate"

        lexical = _lexical_score(query, memory_search_text(row))
        salience = max(0.0, min(1.0, float(row.get("salience", 0.0))))
        recency = max(0.0, min(1.0, float(row.get("recency", 0.0))))
        explicit_priority = max(0.0, min(1.0, float(row.get("priority", 0.0))))
        score = 0.70 * lexical + 0.15 * salience + 0.10 * recency + 0.05 * explicit_priority
        trace_row = {
            "id": row.get("id", index),
            "text": text,
            "lexical_score": lexical,
            "salience": salience,
            "recency": recency,
            "priority": explicit_priority,
            "score": score,
            "excluded_reason": excluded_reason,
            "selected": False,
        }
        if excluded_reason is None:
            seen_text.add(normalized_text)
            scored.append((score, row, trace_row))
        else:
            scored.append((-math.inf, row, trace_row))

    eligible = sorted(
        (
            item
            for item in scored
            if item[2]["excluded_reason"] is None
            and (item[2]["lexical_score"] > 0.0 or item[2]["priority"] > 0.0)
        ),
        key=lambda item: (item[0], float(item[1].get("recency", 0.0))),
        reverse=True,
    )
    selected_triplets = [item for item in eligible[:top_k] if item[0] > 0.0]
    selected_ids = {id(item[2]) for item in selected_triplets}
    for _, _, trace_row in scored:
        trace_row["selected"] = id(trace_row) in selected_ids
    selected = [row for _, row, _ in selected_triplets]
    trace = [trace_row for _, _, trace_row in sorted(scored, key=lambda item: item[0], reverse=True)]
    return selected, trace


def retrieve(query: str, rows: Iterable[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only selected memories."""

    return retrieve_with_trace(query, rows, top_k)[0]


def fit_texts(
    texts: Iterable[str],
    budget: int,
    token_counter: Callable[[str], int] | None = None,
    *,
    prefer_last: bool = False,
) -> list[str]:
    """Keep complete items within budget in the requested priority order.

    Retrieved memories arrive most-relevant first, while recent turns arrive
    oldest-to-newest and should prioritize the tail.  ``prefer_last`` makes
    that distinction explicit instead of accidentally budgeting low-ranked
    memories first.
    """

    count = token_counter or (lambda value: len(tokens(value)))
    values = list(texts)
    candidates = list(reversed(values)) if prefer_last else values
    output: list[str] = []
    used = 0
    for value in candidates:
        size = count(value)
        if size <= 0 or used + size > budget:
            continue
        output.append(value)
        used += size
    return list(reversed(output)) if prefer_last else output


def assemble(
    policy: str,
    memories: Iterable[dict[str, Any]],
    recent: Iterable[str],
    user: str,
    budget: Budget = Budget(),
    token_counter: Callable[[str], int] | None = None,
) -> str:
    """Assemble a bounded text prompt while preserving current user input."""

    count = token_counter or (lambda value: len(tokens(value)))
    memory_texts = fit_texts((memory_text(row) for row in memories), budget.memory, count)
    recent_texts = fit_texts(recent, budget.recent, count, prefer_last=True)
    sections = [policy]
    if memory_texts:
        sections.append("Relevant memory:\n- " + "\n- ".join(memory_texts))
    if recent_texts:
        sections.append("Recent conversation:\n" + "\n".join(recent_texts))
    sections.append("User: " + user)
    sections.append("Assistant:")

    # Drop oldest optional sections until the hard total budget is satisfied.
    while count("\n\n".join(sections)) > budget.total:
        if memory_texts:
            memory_texts.pop(0)
            if memory_texts:
                sections[1] = "Relevant memory:\n- " + "\n- ".join(memory_texts)
            else:
                sections.pop(1)
            continue
        recent_index = next((i for i, value in enumerate(sections) if value.startswith("Recent conversation:")), None)
        if recent_index is not None and recent_texts:
            recent_texts.pop(0)
            if recent_texts:
                sections[recent_index] = "Recent conversation:\n" + "\n".join(recent_texts)
            else:
                sections.pop(recent_index)
            continue
        raise ValueError("Policy plus current user message exceed the total context budget.")
    return "\n\n".join(sections)


def prefix_key(
    model_sha: str,
    tokenizer_sha: str,
    prefix_tokens: list[int],
    context_config: dict[str, Any],
) -> str:
    """Hash every value that makes a cached KV prefix semantically valid."""

    payload = {
        "model_sha": model_sha,
        "tokenizer_sha": tokenizer_sha,
        "prefix_tokens": prefix_tokens,
        "context_config": context_config,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
