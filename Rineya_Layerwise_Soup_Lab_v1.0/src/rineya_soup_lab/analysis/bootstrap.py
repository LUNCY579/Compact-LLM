from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..evaluation.objective import (
    align_rows,
    case_utility,
    objective_summary_from_rows,
    score_summary,
)


@dataclass(frozen=True)
class BootstrapDifference:
    observed_delta: float
    bootstrap_mean_delta: float
    ci_low: float
    ci_high: float
    probability_improvement: float
    samples: int
    confidence_level: float
    paired_case_mean_delta: float
    paired_case_win_rate: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def paired_bootstrap_objective_difference(
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    objective_config: Mapping[str, Any],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> BootstrapDifference:
    candidate, reference = align_rows(candidate_rows, reference_rows)
    observed_candidate = score_summary(
        objective_summary_from_rows(candidate, objective_config), objective_config
    ).score
    observed_reference = score_summary(
        objective_summary_from_rows(reference, objective_config), objective_config
    ).score
    observed_delta = observed_candidate - observed_reference
    n = len(candidate)
    rng = np.random.default_rng(seed)
    deltas = np.empty(max(1, int(samples)), dtype=np.float64)
    for index in range(len(deltas)):
        sample_indices = rng.integers(0, n, size=n)
        sampled_candidate = [candidate[int(i)] for i in sample_indices]
        sampled_reference = [reference[int(i)] for i in sample_indices]
        # Repeated bootstrap draws are statistical multiplicity, not repeated model
        # outputs. Preserve one copy of each sampled case for duplicate-reply metrics.
        unique_indices = sorted({int(i) for i in sample_indices})
        duplicate_candidate = [candidate[i] for i in unique_indices]
        duplicate_reference = [reference[i] for i in unique_indices]
        candidate_score = score_summary(
            objective_summary_from_rows(
                sampled_candidate,
                objective_config,
                duplicate_rate_rows=duplicate_candidate,
            ),
            objective_config,
        ).score
        reference_score = score_summary(
            objective_summary_from_rows(
                sampled_reference,
                objective_config,
                duplicate_rate_rows=duplicate_reference,
            ),
            objective_config,
        ).score
        deltas[index] = candidate_score - reference_score
    alpha = (1.0 - confidence_level) / 2.0
    case_deltas = np.array(
        [
            case_utility(c, objective_config) - case_utility(r, objective_config)
            for c, r in zip(candidate, reference)
        ],
        dtype=np.float64,
    )
    return BootstrapDifference(
        observed_delta=float(observed_delta),
        bootstrap_mean_delta=float(deltas.mean()),
        ci_low=float(np.quantile(deltas, alpha)),
        ci_high=float(np.quantile(deltas, 1.0 - alpha)),
        probability_improvement=float(np.mean(deltas > 0.0)),
        samples=len(deltas),
        confidence_level=confidence_level,
        paired_case_mean_delta=float(case_deltas.mean()),
        paired_case_win_rate=float(np.mean(case_deltas > 0.0)),
    )


def bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot bootstrap an empty sequence.")
    rng = np.random.default_rng(seed)
    means = np.empty(max(1, int(samples)), dtype=np.float64)
    for index in range(len(means)):
        means[index] = rng.choice(array, size=array.size, replace=True).mean()
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "mean": float(array.mean()),
        "bootstrap_mean": float(means.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
        "probability_positive": float(np.mean(means > 0.0)),
    }


def exact_two_sided_sign_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    probability = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * probability)
