"""Deterministic exploratory uncertainty from already recorded paired outcomes.

The bootstrap resamples the empirical parent differences {-1, 0, 1}. Its full
finite distribution is evaluated by convolution, using floating-point arithmetic
rather than Monte Carlo or a normal approximation. "Exact empirical" describes
that distribution; it does not promise exact frequentist coverage. Sparse events
and unobserved future harms can make percentile intervals misleading, including
an interval [0, 0] when no differences were observed. Independent parent identity,
source custody and a prospective confirmation design remain external requirements.
These calculations do not retrofit any registered experimental acceptance rule.
"""

from __future__ import annotations

import math

MAX_N = 512


def _integer(value: int, name: str, *, positive: bool = False) -> None:
    minimum = 1 if positive else 0
    if type(value) is not int or not minimum <= value <= MAX_N:
        raise ValueError(name + " must be an integer from " + str(minimum) + " through 512")


def _validate_counts(benefits: int, harms: int, n: int) -> None:
    _integer(n, "n", positive=True)
    _integer(benefits, "benefits")
    _integer(harms, "harms")
    if benefits + harms > n:
        raise ValueError("benefits and harms exceed the paired-parent denominator")


def _empirical_pmf(benefits: int, harms: int, n: int) -> list[float]:
    """Return mass indexed by bootstrap sum+n, for integer sums -n through n."""
    _validate_counts(benefits, harms, n)
    minus, zero, plus = harms / n, (n - benefits - harms) / n, benefits / n
    mass = [1.0]
    for _ in range(n):
        next_mass = [0.0] * (len(mass) + 2)
        for index, probability in enumerate(mass):
            if probability:
                next_mass[index] += probability * minus
                next_mass[index + 1] += probability * zero
                next_mass[index + 2] += probability * plus
        mass = next_mass
    total = math.fsum(mass)
    if not math.isfinite(total) or total <= 0 or any(not math.isfinite(value) or value < 0 for value in mass):
        raise ValueError("invalid empirical bootstrap probability mass")
    # Normalize accumulated floating-point roundoff, preserving the finite PMF.
    return [min(1.0, max(0.0, value / total)) for value in mass]


def _quantile(mass: list[float], probability: float, n: int) -> float:
    """Inverse-CDF percentile: smallest supported sum whose CDF reaches q."""
    cumulative = 0.0
    for index, value in enumerate(mass):
        cumulative = math.fsum((cumulative, value))
        if value and cumulative >= probability:
            return (index - n) / n
    # A last-bit cumulative deficit must not select an unsupported outer endpoint.
    for index in range(len(mass) - 1, -1, -1):
        if mass[index] > 0:
            return (index - n) / n
    raise ValueError("empty empirical bootstrap support")


def paired_bootstrap_interval(benefits: int, harms: int, n: int, confidence: float = .90) -> dict:
    """Percentile interval for the empirical paired-parent mean difference.

    Benefits count source-wrong/candidate-correct pairs; harms count the reverse.
    The remaining n-benefits-harms observations have difference zero. Every parent
    has equal empirical weight. Pooling families resamples their mixture as well;
    this is not a fixed-family stratified bootstrap. No RNG or model is involved.
    """
    _validate_counts(benefits, harms, n)
    if type(confidence) not in (int, float) or not 0 < confidence < 1 or not math.isfinite(confidence):
        raise ValueError("confidence must be a finite number strictly between zero and one")
    mass = _empirical_pmf(benefits, harms, n)
    tail = (1 - confidence) / 2
    return {
        "net_rate": (benefits - harms) / n,
        "lower": _quantile(mass, tail, n),
        "upper": _quantile(mass, 1 - tail, n),
        "confidence": confidence,
        "method": "exact_empirical_paired_percentile",
        "coverage": "EXPLORATORY_NOT_CONFIRMATION",
    }


def stratified_paired_bootstrap_interval(strata: list[tuple[int, int, int]], confidence: float = .90) -> dict:
    """Resample pairs within fixed equal-sized families, then combine differences.

    Each tuple is (benefits, harms, n) for one family. All n values must agree,
    making the equally weighted family macro mean equal to the overall paired
    rate. The allocation remains fixed; family prevalence is not resampled.
    The empirical percentile distribution has the same exploratory, rare-event
    and unobserved-future-harm limitations as the pooled calculation.
    """
    if type(strata) is not list or not strata:
        raise ValueError("strata must be a nonempty list of count tuples")
    for stratum in strata:
        if type(stratum) is not tuple or len(stratum) != 3:
            raise ValueError("each stratum must be a (benefits, harms, n) tuple")
        _validate_counts(*stratum)
    allocation = [stratum[2] for stratum in strata]
    if len(set(allocation)) != 1:
        raise ValueError("equal family denominators required; unequal strata cannot be silently pooled")
    total_n = sum(allocation)
    _integer(total_n, "total n", positive=True)
    if type(confidence) not in (int, float) or not 0 < confidence < 1 or not math.isfinite(confidence):
        raise ValueError("confidence must be a finite number strictly between zero and one")

    mass = [1.0]
    for benefits, harms, n in strata:
        family_mass = _empirical_pmf(benefits, harms, n)
        next_mass = [0.0] * (len(mass) + len(family_mass) - 1)
        for left_index, left_probability in enumerate(mass):
            if left_probability:
                for right_index, right_probability in enumerate(family_mass):
                    if right_probability:
                        next_mass[left_index + right_index] += left_probability * right_probability
        mass = next_mass
    total_mass = math.fsum(mass)
    if not math.isfinite(total_mass) or total_mass <= 0 or any(not math.isfinite(value) or value < 0 for value in mass):
        raise ValueError("invalid stratified empirical bootstrap probability mass")
    mass = [min(1.0, max(0.0, value / total_mass)) for value in mass]
    tail = (1 - confidence) / 2
    return {
        "net_rate": sum(benefits - harms for benefits, harms, _ in strata) / total_n,
        "lower": _quantile(mass, tail, total_n),
        "upper": _quantile(mass, 1 - tail, total_n),
        "confidence": confidence,
        "method": "exact_empirical_stratified_paired_percentile",
        "coverage": "EXPLORATORY_NOT_CONFIRMATION",
        "strata_count": len(strata),
        "fixed_allocation": allocation,
    }


def exact_mcnemar(benefits: int, harms: int) -> float:
    """Two-sided exact conditional McNemar p-value, with zero discordance=1.

    Uses twice the fair-binomial upper tail at max(benefits, harms), clamped to1.
    Counts describe recorded pairs; this cannot establish valid population trials.
    """
    _integer(benefits, "benefits")
    _integer(harms, "harms")
    discordant = benefits + harms
    if discordant > MAX_N:
        raise ValueError("discordant pair count exceeds 512")
    if discordant == 0:
        return 1.0
    tail_numerator = sum(math.comb(discordant, index) for index in range(max(benefits, harms), discordant + 1))
    return min(1.0, max(0.0, 2 * tail_numerator / (2 ** discordant)))
