"""Cluster-aware paired analysis for repeated case variants.

Rendered depths, schedules, and policies derived from one latent case are not
independent. These helpers first collapse paired differences within immutable
clusters, then perform inference on clusters only.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True, slots=True)
class ClusteredPairedResult:
    cluster_count: int
    row_count: int
    cluster_differences: dict[str, float]
    mean_difference: float
    median_difference: float
    confidence_interval: tuple[float, float]
    p_value_two_sided: float
    test_method: str
    randomization_samples: int
    eligible: bool
    eligibility_reason: str


def cluster_differences(
    cluster_ids: Sequence[str],
    baseline: Sequence[float],
    candidate: Sequence[float],
) -> dict[str, float]:
    """Average paired row differences inside each immutable cluster."""

    if not cluster_ids or len(cluster_ids) != len(baseline) or len(baseline) != len(candidate):
        raise ValueError("cluster IDs and paired outcomes must be non-empty and equally sized")
    grouped: dict[str, list[float]] = defaultdict(list)
    for cluster_id, left, right in zip(cluster_ids, baseline, candidate, strict=True):
        name = str(cluster_id)
        if not name:
            raise ValueError("cluster IDs must be non-empty")
        values = (float(left), float(right))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("paired outcomes must be finite")
        grouped[name].append(values[1] - values[0])
    return {name: float(np.mean(grouped[name])) for name in sorted(grouped)}


def exact_or_sampled_sign_flip(
    differences: Sequence[float],
    *,
    exact_max_clusters: int = 20,
    monte_carlo_samples: int = 100_000,
    seed: int = 0,
) -> tuple[float, str, int]:
    """Two-sided paired sign-flip test over cluster-level differences."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("differences must be a non-empty finite vector")
    observed = abs(float(values.mean()))
    tolerance = 1e-15
    if values.size <= exact_max_clusters:
        extreme = 0
        total = 1 << int(values.size)
        for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
            statistic = abs(float(np.mean(values * np.asarray(signs))))
            extreme += int(statistic + tolerance >= observed)
        return extreme / total, "exact_cluster_sign_flip", total
    if monte_carlo_samples <= 0:
        raise ValueError("monte_carlo_samples must be positive")
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(monte_carlo_samples):
        sampled_signs = rng.choice(np.array([-1.0, 1.0]), size=values.size)
        statistic = abs(float(np.mean(values * sampled_signs)))
        extreme += int(statistic + tolerance >= observed)
    # Phipson-Smyth correction: a sampled randomization p-value is never zero.
    return (extreme + 1) / (monte_carlo_samples + 1), "sampled_cluster_sign_flip", monte_carlo_samples


def cluster_bootstrap_interval(
    differences: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("differences must be a non-empty finite vector")
    if not 0 < confidence < 1 or samples <= 0:
        raise ValueError("invalid confidence/bootstrap sample count")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def clustered_paired_analysis(
    cluster_ids: Sequence[str],
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    minimum_clusters: int,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    randomization_samples: int = 100_000,
    seed: int = 0,
) -> ClusteredPairedResult:
    if minimum_clusters <= 0:
        raise ValueError("minimum_clusters must be positive")
    by_cluster = cluster_differences(cluster_ids, baseline, candidate)
    differences = list(by_cluster.values())
    eligible = len(differences) >= minimum_clusters
    reason = "PASS" if eligible else f"requires at least {minimum_clusters} independent clusters"
    interval = cluster_bootstrap_interval(
        differences,
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed,
    )
    p_value, method, samples = exact_or_sampled_sign_flip(
        differences,
        monte_carlo_samples=randomization_samples,
        seed=seed,
    )
    return ClusteredPairedResult(
        cluster_count=len(differences),
        row_count=len(cluster_ids),
        cluster_differences=by_cluster,
        mean_difference=float(np.mean(differences)),
        median_difference=float(np.median(differences)),
        confidence_interval=interval,
        p_value_two_sided=p_value,
        test_method=method,
        randomization_samples=samples,
        eligible=eligible,
        eligibility_reason=reason,
    )


def normal_approximation_mde(
    *,
    independent_clusters: int,
    cluster_difference_sd: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Planning MDE for a mean cluster-level paired difference.

    This is a planning approximation, not the registered analysis. Pilot data
    must supply a defensible between-cluster SD before the value is actionable.
    """

    if independent_clusters < 2 or cluster_difference_sd <= 0:
        raise ValueError("at least two clusters and a positive cluster SD are required")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be in (0, 1)")
    normal = NormalDist()
    critical = normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)
    return float(critical * cluster_difference_sd / math.sqrt(independent_clusters))


def validate_cluster_splits(
    split_by_case: Iterable[tuple[str, str]],
) -> dict[str, str]:
    """Reject any cluster that appears in more than one dataset split."""

    observed: dict[str, str] = {}
    for cluster_id, split in split_by_case:
        if not cluster_id or not split:
            raise ValueError("cluster_id and split must be non-empty")
        previous = observed.setdefault(cluster_id, split)
        if previous != split:
            raise ValueError(f"cluster {cluster_id!r} leaks across {previous!r} and {split!r}")
    return dict(sorted(observed.items()))
