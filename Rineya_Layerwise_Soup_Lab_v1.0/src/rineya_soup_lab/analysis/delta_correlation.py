from __future__ import annotations

"""Descriptive association between parameter movement and behavioral influence.

These statistics are deliberately diagnostic.  They are never used to choose a
merge coefficient: a large checkpoint delta is not evidence that the Greedy
version of a layer is behaviorally better, and a small delta can still cross an
important decision boundary.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CorrelationEstimate:
    coefficient: float | None
    permutation_p_value: float | None
    bootstrap_ci_low: float | None
    bootstrap_ci_high: float | None
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _rank_average_ties(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, matching the standard Spearman definition."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    denominator = float(np.linalg.norm(centered_x) * np.linalg.norm(centered_y))
    if denominator <= 0.0:
        return None
    return float(np.dot(centered_x, centered_y) / denominator)


def _coefficient(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    if method == "pearson":
        return _pearson(x, y)
    if method == "spearman":
        return _pearson(_rank_average_ties(x), _rank_average_ties(y))
    raise ValueError(f"Unsupported correlation method: {method!r}")


def estimate_correlation(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    method: str,
    permutations: int = 5000,
    bootstrap_samples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> CorrelationEstimate:
    """Estimate correlation with descriptive permutation and bootstrap uncertainty."""

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("Correlation inputs must have the same shape.")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    observed = _coefficient(x, y, method)
    if observed is None:
        return CorrelationEstimate(None, None, None, None, int(len(x)))

    rng = np.random.default_rng(int(seed))
    permutation_count = max(0, int(permutations))
    permutation_p: float | None = None
    if permutation_count > 0:
        at_least_as_extreme = 0
        valid_permutations = 0
        for _ in range(permutation_count):
            permuted = _coefficient(x, rng.permutation(y), method)
            if permuted is None:
                continue
            valid_permutations += 1
            if abs(permuted) >= abs(observed) - 1e-15:
                at_least_as_extreme += 1
        if valid_permutations:
            # Add-one correction prevents a misleading p=0 in a finite randomization test.
            permutation_p = (at_least_as_extreme + 1) / (valid_permutations + 1)

    bootstrap_count = max(0, int(bootstrap_samples))
    ci_low: float | None = None
    ci_high: float | None = None
    if bootstrap_count > 0 and len(x) >= 3:
        estimates: list[float] = []
        for _ in range(bootstrap_count):
            indices = rng.integers(0, len(x), size=len(x))
            estimate = _coefficient(x[indices], y[indices], method)
            if estimate is not None:
                estimates.append(estimate)
        if estimates:
            tail = (1.0 - float(confidence_level)) / 2.0
            ci_low, ci_high = (
                float(np.quantile(estimates, tail)),
                float(np.quantile(estimates, 1.0 - tail)),
            )

    return CorrelationEstimate(
        coefficient=observed,
        permutation_p_value=permutation_p,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        sample_count=int(len(x)),
    )


def join_layer_delta_and_influence(
    delta_statistics: Sequence[Mapping[str, Any]],
    influence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join compatibility-time delta statistics with intervention measurements."""

    deltas = {
        int(row["layer_index"]): row
        for row in delta_statistics
        if row.get("layer_index") is not None
    }
    joined: list[dict[str, Any]] = []
    for influence in influence_rows:
        layer = int(influence["layer"])
        delta = deltas.get(layer)
        if delta is None:
            continue
        combined_mean = float(influence["combined_mean"])
        joined.append(
            {
                "layer": layer,
                "parameter_count": int(delta.get("parameter_count", 0)),
                "delta_l2_norm": float(delta.get("delta_l2_norm", 0.0)),
                "relative_delta_norm": float(delta.get("relative_delta_norm", 0.0)),
                "anchor_greedy_cosine": (
                    None
                    if delta.get("anchor_greedy_cosine") is None
                    else float(delta["anchor_greedy_cosine"])
                ),
                "angular_distance_proxy": (
                    None
                    if delta.get("anchor_greedy_cosine") is None
                    else 1.0 - float(delta["anchor_greedy_cosine"])
                ),
                "combined_behavioral_influence": combined_mean,
                "absolute_behavioral_influence": abs(combined_mean),
                "influence_reliability": float(influence.get("reliability", 0.0)),
                "normalized_importance": float(influence.get("normalized_importance", 0.0)),
            }
        )
    return sorted(joined, key=lambda row: int(row["layer"]))


def build_delta_influence_correlation_report(
    joined_rows: Sequence[Mapping[str, Any]],
    *,
    permutations: int = 5000,
    bootstrap_samples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a diagnostic matrix of delta-magnitude/influence associations."""

    predictor_names = ("delta_l2_norm", "relative_delta_norm", "angular_distance_proxy")
    outcome_names = ("combined_behavioral_influence", "absolute_behavioral_influence")
    correlations: dict[str, Any] = {}
    for predictor_index, predictor in enumerate(predictor_names):
        correlations[predictor] = {}
        for outcome_index, outcome in enumerate(outcome_names):
            pairs = [
                (float(row[predictor]), float(row[outcome]))
                for row in joined_rows
                if row.get(predictor) is not None and row.get(outcome) is not None
            ]
            x_values = [pair[0] for pair in pairs]
            y_values = [pair[1] for pair in pairs]
            correlations[predictor][outcome] = {
                method: estimate_correlation(
                    x_values,
                    y_values,
                    method=method,
                    permutations=permutations,
                    bootstrap_samples=bootstrap_samples,
                    confidence_level=confidence_level,
                    seed=seed + predictor_index * 1000 + outcome_index * 100 + method_index,
                ).to_dict()
                for method_index, method in enumerate(("pearson", "spearman"))
            }

    return {
        "matched_layer_count": len(joined_rows),
        "correlations": correlations,
        "methodological_boundary": (
            "Parameter movement is reported only as a descriptive diagnostic. Delta magnitude, "
            "relative norm, and cosine distance are not used to select layer coefficients, because "
            "weight movement does not establish the direction or value of behavioral contribution."
        ),
        "small_sample_warning": (
            "Layer-level correlation is based on the number of Transformer layers in this model; "
            "confidence intervals and p-values are exploratory, especially for a 12-layer network."
        ),
    }
