"""Frozen evaluation definitions for STR-QUAL-001."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


def pass_at_k(groups: Iterable[Iterable[bool]], k: int) -> float:
    """Empirical pass@k: fraction of prompt groups with a correct result in k."""

    if k <= 0:
        raise ValueError("k must be positive")
    materialized = [list(group) for group in groups]
    if not materialized or any(len(group) < k for group in materialized):
        raise ValueError("each prompt group must contain at least k samples")
    return sum(any(group[:k]) for group in materialized) / len(materialized)


def correct_solution_diversity(groups: Iterable[Iterable[Mapping[str, object]]]) -> float:
    """Mean fraction of correct samples with distinct normalized solutions.

    Each row must contain ``correct`` and ``answer_norm``.  A prompt with no
    correct samples contributes 0.0.  This is frozen before confirmatory data
    and distinguishes repeated correct solutions from genuinely varied ones.
    """

    values: list[float] = []
    for group in groups:
        correct = [str(row["answer_norm"]) for row in group if bool(row["correct"])]
        values.append(len(set(correct)) / len(correct) if correct else 0.0)
    if not values:
        raise ValueError("at least one prompt group is required")
    return sum(values) / len(values)


def paired_repairs_damages(
    baseline: Mapping[str, bool], candidate: Mapping[str, bool]
) -> dict[str, int]:
    """Count paired baseline/candidate changes without netting losses away."""

    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("baseline and candidate IDs must match and be non-empty")
    repairs = sum(not baseline[key] and candidate[key] for key in baseline)
    damages = sum(baseline[key] and not candidate[key] for key in baseline)
    unchanged_correct = sum(baseline[key] and candidate[key] for key in baseline)
    return {"repairs": repairs, "damages": damages, "unchanged_correct": unchanged_correct}


def retention(baseline: Mapping[str, bool], candidate: Mapping[str, bool]) -> float | None:
    """Candidate correctness on rows the base model got right."""

    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("baseline and candidate IDs must match and be non-empty")
    base_correct = [key for key, value in baseline.items() if value]
    if not base_correct:
        return None
    return sum(candidate[key] for key in base_correct) / len(base_correct)


def reasoning_length(tokens: Iterable[int]) -> int:
    return sum(1 for _ in tokens)


def entropy(probabilities: Iterable[float]) -> float:
    import math

    values = [float(p) for p in probabilities]
    if not values or any(p < 0 for p in values):
        raise ValueError("probabilities must be non-negative and non-empty")
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("probabilities must have a positive finite sum")
    normalized = [p / total for p in values]
    return -sum(p * math.log(p) for p in normalized if p > 0)
