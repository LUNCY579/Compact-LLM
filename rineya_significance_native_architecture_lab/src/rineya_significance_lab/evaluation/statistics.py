"""Small dependency-free paired statistics for Tier-A outputs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def paired_bootstrap_difference(
    baseline: Iterable[float],
    candidate: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 42,
    cluster_ids: Sequence[str] | None = None,
) -> dict[str, float | int]:
    left = np.asarray(list(baseline), dtype=np.float64)
    right = np.asarray(list(candidate), dtype=np.float64)
    if left.ndim != 1 or left.size == 0 or left.shape != right.shape:
        raise ValueError("paired bootstrap requires non-empty equal-length vectors")
    if not 0 < confidence < 1 or samples <= 0:
        raise ValueError("invalid confidence/sample count")
    difference = right - left
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    if cluster_ids is None:
        for index in range(samples):
            sample = rng.integers(0, difference.size, size=difference.size)
            estimates[index] = float(np.mean(difference[sample]))
    else:
        clusters = np.asarray(list(cluster_ids), dtype=str)
        if clusters.shape != difference.shape:
            raise ValueError("cluster_ids must align with paired vectors")
        unique = np.unique(clusters)
        members = {value: np.flatnonzero(clusters == value) for value in unique}
        for index in range(samples):
            sampled_clusters = rng.choice(unique, size=unique.size, replace=True)
            sampled = np.concatenate([members[value] for value in sampled_clusters])
            estimates[index] = float(np.mean(difference[sampled]))
    alpha = 1.0 - confidence
    return {
        "estimate": float(np.mean(difference)),
        "lower": float(np.quantile(estimates, alpha / 2)),
        "upper": float(np.quantile(estimates, 1 - alpha / 2)),
        "confidence": confidence,
        "samples": samples,
    }


def _binomial_probability(n: int, k: int) -> float:
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2))


def exact_mcnemar(baseline: Iterable[bool], candidate: Iterable[bool]) -> dict[str, int | float]:
    left = list(map(bool, baseline))
    right = list(map(bool, candidate))
    if not left or len(left) != len(right):
        raise ValueError("McNemar inputs must be non-empty and paired")
    baseline_only = sum(a and not b for a, b in zip(left, right, strict=True))
    candidate_only = sum((not a) and b for a, b in zip(left, right, strict=True))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(baseline_only, candidate_only)
        p_value = min(1.0, 2.0 * sum(_binomial_probability(discordant, k) for k in range(tail + 1)))
    return {
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "discordant": discordant,
        "p_value_two_sided": p_value,
    }


def summarize_paired_binary(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    seed: int = 42,
    cluster_ids: Sequence[str] | None = None,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("binary results must be non-empty and paired")
    return {
        "pairs": len(baseline),
        "baseline_rate": float(np.mean(baseline)),
        "candidate_rate": float(np.mean(candidate)),
        "candidate_minus_baseline": paired_bootstrap_difference(
            baseline,
            candidate,
            seed=seed,
            samples=bootstrap_samples,
            cluster_ids=cluster_ids,
        ),
        "mcnemar": exact_mcnemar(baseline, candidate),
    }
