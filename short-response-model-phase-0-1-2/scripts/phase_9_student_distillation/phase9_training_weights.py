from __future__ import annotations

"""Observable completion-token weighting for Phase 9 grounding SFT/KD."""

from typing import Any

from scripts.phase_9_student_distillation.phase9_metrics import decision_class


def find_subsequence_positions(sequence: list[int], pattern: list[int]) -> set[int]:
    positions: set[int] = set()
    if not pattern or len(pattern) > len(sequence):
        return positions
    for start in range(0, len(sequence) - len(pattern) + 1):
        if sequence[start : start + len(pattern)] == pattern:
            positions.update(range(start, start + len(pattern)))
    return positions


def required_anchor_phrases(row: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    contract = row.get("response_contract") or {}
    for requirement in contract.get("requirements") or []:
        match = requirement.get("match") or {}
        for key in ("any_phrases", "all_phrases"):
            for value in match.get(key) or []:
                text = str(value).strip()
                if text and text not in phrases:
                    phrases.append(text)
    return phrases


def completion_loss_weights(
    row: dict[str, Any],
    tokenizer: Any,
    completion_ids: list[int],
    *,
    first_answer_token_weight: float,
    evidence_anchor_token_weight: float,
    abstention_token_weight: float,
) -> list[float]:
    weights = [1.0 for _ in completion_ids]
    if weights:
        weights[0] = max(weights[0], float(first_answer_token_weight))
    for phrase in required_anchor_phrases(row):
        phrase_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
        for position in find_subsequence_positions(completion_ids, phrase_ids):
            weights[position] = max(weights[position], float(evidence_anchor_token_weight))
    if decision_class(row) == "unsupported":
        weights = [max(value, float(abstention_token_weight)) for value in weights]
    return weights
