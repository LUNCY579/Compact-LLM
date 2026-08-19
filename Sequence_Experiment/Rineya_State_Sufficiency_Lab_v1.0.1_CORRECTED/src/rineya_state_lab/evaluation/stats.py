from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def paired_bootstrap_difference(
    baseline: Iterable[float],
    candidate: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    left = np.asarray(list(baseline), dtype=np.float64)
    right = np.asarray(list(candidate), dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("Paired bootstrap requires non-empty vectors of equal length")
    diff = right - left
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    chunk = 1000
    for start in range(0, samples, chunk):
        size = min(chunk, samples - start)
        indexes = rng.integers(0, diff.size, size=(size, diff.size))
        estimates[start : start + size] = diff[indexes].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "estimate": float(diff.mean()),
        "lower": float(np.quantile(estimates, alpha / 2)),
        "upper": float(np.quantile(estimates, 1 - alpha / 2)),
        "confidence": confidence,
        "samples": samples,
    }


def _binomial_probability(n: int, k: int) -> float:
    if n < 0 or k < 0 or k > n:
        return 0.0
    return math.exp(
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2.0)
    )


def exact_mcnemar(baseline: Iterable[bool], candidate: Iterable[bool]) -> dict[str, Any]:
    left = list(map(bool, baseline))
    right = list(map(bool, candidate))
    if len(left) != len(right):
        raise ValueError("McNemar inputs must be paired")
    baseline_only = sum(a and not b for a, b in zip(left, right))
    candidate_only = sum((not a) and b for a, b in zip(left, right))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(baseline_only, candidate_only)
        p_value = min(1.0, 2.0 * sum(_binomial_probability(discordant, index) for index in range(tail + 1)))
    return {
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "discordant": discordant,
        "p_value_two_sided": p_value,
    }


def sign_flip_randomization(
    baseline: Iterable[float],
    candidate: Iterable[float],
    *,
    samples: int = 20000,
    seed: int = 42,
) -> dict[str, float]:
    left = np.asarray(list(baseline), dtype=np.float64)
    right = np.asarray(list(candidate), dtype=np.float64)
    diff = right - left
    nonzero = diff[diff != 0]
    observed = abs(float(diff.mean()))
    if nonzero.size == 0:
        return {"observed_absolute_difference": observed, "p_value_two_sided": 1.0, "samples": samples}
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(samples):
        signs = rng.choice(np.array([-1.0, 1.0]), size=nonzero.size)
        permuted = abs(float((nonzero * signs).sum() / diff.size))
        if permuted >= observed - 1e-15:
            extreme += 1
    return {
        "observed_absolute_difference": observed,
        "p_value_two_sided": (extreme + 1) / (samples + 1),
        "samples": samples,
    }


def proportion_ci_wilson(successes: int, total: int, confidence: float = 0.95) -> dict[str, float]:
    if total <= 0:
        return {"estimate": 0.0, "lower": 0.0, "upper": 0.0, "confidence": confidence}
    # Common z values avoid adding scipy as a required runtime dependency.
    z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}.get(confidence)
    if z is None:
        raise ValueError("Wilson CI currently supports confidence 0.90, 0.95, or 0.99")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return {"estimate": p, "lower": max(0.0, center - half), "upper": min(1.0, center + half), "confidence": confidence}
