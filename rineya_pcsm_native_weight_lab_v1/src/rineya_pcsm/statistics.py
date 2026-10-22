"""Small, dependency-free paired statistics for PCSM qualification results."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import sqrt
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_CONFIDENCE: Final = 0.95


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    low: float
    high: float
    confidence: float
    method: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "method": self.method,
        }


@dataclass(frozen=True, slots=True)
class SignFlipResult:
    p_value: float
    observed_mean: float
    permutations: int
    exact: bool
    alternative: str = "two-sided"

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "p_value": self.p_value,
            "observed_mean": self.observed_mean,
            "permutations": self.permutations,
            "exact": self.exact,
            "alternative": self.alternative,
        }


def _one_dimensional(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def paired_differences(first: ArrayLike, second: ArrayLike) -> NDArray[np.float64]:
    """Return ``first - second`` after strict paired-shape validation."""

    first_array = _one_dimensional(first, "first")
    second_array = _one_dimensional(second, "second")
    if first_array.shape != second_array.shape:
        raise ValueError("paired samples must have identical shapes")
    return first_array - second_array


def percentile_bootstrap_interval(
    values: ArrayLike,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = 10_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Compute a deterministic percentile bootstrap interval for a sample mean."""

    sample = _one_dimensional(values, "values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, sample.size, size=(resamples, sample.size))
    means = np.mean(sample[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    quantiles = np.asarray(np.quantile(means, [alpha, 1.0 - alpha]), dtype=np.float64)
    return ConfidenceInterval(
        low=float(quantiles[0]),
        high=float(quantiles[1]),
        confidence=confidence,
        method=f"paired_percentile_bootstrap_{resamples}",
    )


def paired_bootstrap_interval(
    first: ArrayLike,
    second: ArrayLike,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = 10_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Bootstrap the mean of paired ``first - second`` differences."""

    return percentile_bootstrap_interval(
        paired_differences(first, second),
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )


def paired_sign_flip_test(
    first: ArrayLike,
    second: ArrayLike,
    *,
    max_exact_pairs: int = 20,
    monte_carlo_permutations: int = 100_000,
    seed: int = 0,
) -> SignFlipResult:
    """Two-sided paired randomization test under exchangeable difference signs."""

    differences = paired_differences(first, second)
    observed = float(np.mean(differences))
    threshold = abs(observed) - np.finfo(np.float64).eps * 16.0

    if differences.size <= max_exact_pairs:
        signs = np.asarray(
            list(product((-1.0, 1.0), repeat=differences.size)),
            dtype=np.float64,
        )
        permuted = np.mean(signs * differences, axis=1)
        extreme = int(np.count_nonzero(np.abs(permuted) >= threshold))
        total = int(permuted.size)
        return SignFlipResult(
            p_value=extreme / total,
            observed_mean=observed,
            permutations=total,
            exact=True,
        )

    if monte_carlo_permutations < 1_000:
        raise ValueError("monte_carlo_permutations must be at least 1000")
    rng = np.random.default_rng(seed)
    signs = rng.choice(
        np.asarray((-1.0, 1.0)),
        size=(monte_carlo_permutations, differences.size),
    )
    permuted = np.mean(signs * differences, axis=1)
    extreme = int(np.count_nonzero(np.abs(permuted) >= threshold))
    return SignFlipResult(
        p_value=(extreme + 1.0) / (monte_carlo_permutations + 1.0),
        observed_mean=observed,
        permutations=monte_carlo_permutations,
        exact=False,
    )


def summarize_sample(values: ArrayLike) -> dict[str, float | int]:
    """Return transparent descriptive statistics without distributional claims."""

    sample = _one_dimensional(values, "values")
    standard_deviation = float(np.std(sample, ddof=1)) if sample.size > 1 else 0.0
    return {
        "n": int(sample.size),
        "mean": float(np.mean(sample)),
        "median": float(np.median(sample)),
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / sqrt(float(sample.size)),
        "minimum": float(np.min(sample)),
        "maximum": float(np.max(sample)),
    }


def paired_comparison(
    first: ArrayLike,
    second: ArrayLike,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    bootstrap_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Bundle effect size, paired bootstrap CI, and a sign-flip p-value."""

    differences = paired_differences(first, second)
    return {
        "difference_definition": "first_minus_second",
        "differences": [float(value) for value in differences],
        "effect": summarize_sample(differences),
        "confidence_interval": percentile_bootstrap_interval(
            differences,
            confidence=confidence,
            resamples=bootstrap_resamples,
            seed=seed,
        ).to_dict(),
        "sign_flip_test": paired_sign_flip_test(
            first,
            second,
            seed=seed,
        ).to_dict(),
    }
