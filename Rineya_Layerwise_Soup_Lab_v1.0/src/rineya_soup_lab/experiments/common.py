from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..analysis.bootstrap import BootstrapDifference, paired_bootstrap_objective_difference
from ..candidates import CandidateResult


def compare_candidates(
    candidate: CandidateResult,
    reference: CandidateResult,
    *,
    objective_config: Mapping[str, Any],
    statistics_config: Mapping[str, Any],
    seed_offset: int = 0,
) -> BootstrapDifference:
    return paired_bootstrap_objective_difference(
        candidate.evidence.rows,
        reference.evidence.rows,
        objective_config,
        samples=int(statistics_config.get("bootstrap_samples", 2000)),
        confidence_level=float(statistics_config.get("confidence_level", 0.95)),
        seed=int(statistics_config.get("seed", 0)) + seed_offset,
    )


def rank_candidates(candidates: Sequence[CandidateResult]) -> list[CandidateResult]:
    return sorted(
        candidates,
        key=lambda result: (result.objective.score, result.candidate_id),
        reverse=True,
    )


def accept_change(
    candidate: CandidateResult,
    current: CandidateResult,
    *,
    objective_config: Mapping[str, Any],
    statistics_config: Mapping[str, Any],
    seed_offset: int = 0,
) -> tuple[bool, BootstrapDifference]:
    comparison = compare_candidates(
        candidate,
        current,
        objective_config=objective_config,
        statistics_config=statistics_config,
        seed_offset=seed_offset,
    )
    minimum_delta = float(statistics_config.get("minimum_objective_delta", 0.0))
    minimum_probability = float(statistics_config.get("minimum_accept_probability", 0.5))
    accepted = (
        comparison.observed_delta >= minimum_delta
        and comparison.probability_improvement >= minimum_probability
    )
    return accepted, comparison
