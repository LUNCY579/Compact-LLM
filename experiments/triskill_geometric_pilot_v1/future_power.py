"""Prospective, CPU-only power for a separately approved paired confirmation.

No model, outcome, fixture, or protected-bank inputs are used. A source world is
the independent sampling unit. Meaning-preserving format variants must be
collapsed within that world; they cannot increase N. The joint probabilities
below are assumptions, not estimates from the exploratory pilot.

The candidate must show a beneficial semantic-correctness difference against
each of eight comparators. Each comparison uses a two-sided exact McNemar test
at .05/8 and must have more gains than losses. This calculator sums over the
number of discordant pairs D~Bin(N, gain+loss), then the number of gains
X|D~Bin(D, gain/(gain+loss)). It uses the exact discrete rejection region, with
floating-point summation; it is not an asymptotic or normal approximation.

Power is per comparator. Shared worlds and candidate predictions correlate the
eight tests. A simultaneous-power lower bound is reported using the union
bound; no independence of comparator tests is assumed. All sample sizes,
operators, layers, angles, stopping rules, and scoring must be frozen before
any confirmation C outcomes. This module does not authorize that experiment.
"""
from __future__ import annotations

from functools import lru_cache
import math
import time


COMPARATORS = ("BASE", "T", "H", "TH", "IDENTITY", "RANDOM", "PROMPT", "REVERSE")
FAMILY_ALPHA = 0.05
PER_COMPARISON_ALPHA = FAMILY_ALPHA / len(COMPARATORS)
SAMPLE_SIZES = (256, 384, 512)
SCENARIOS = (
    ("primary net .10", 0.15, 0.05),
    ("sensitivity net .05", 0.10, 0.05),
    ("sensitivity net .08", 0.12, 0.04),
)


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability")
    return value


def _alpha(value: float) -> float:
    value = _probability(value, "alpha")
    if not 0.0 < value < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    return value


def binomial_pmf(n: int, p: float) -> tuple[float, ...]:
    """Stable normalized binomial masses, grown outward from the mode.

    This avoids factorial overflow and starting from an underflowed p**n tail.
    For the prospectively bounded N<=512, no scientific library is required.
    """
    n = _count(n, "n")
    p = _probability(p, "p")
    if p == 0.0:
        return (1.0,) + (0.0,) * n
    if p == 1.0:
        return (0.0,) * n + (1.0,)
    mode = min(n, int(math.floor((n + 1) * p)))
    weights = [0.0] * (n + 1)
    weights[mode] = 1.0
    for k in range(mode, 0, -1):
        weights[k - 1] = weights[k] * (k / (n - k + 1)) * ((1.0 - p) / p)
    for k in range(mode, n):
        weights[k + 1] = weights[k] * ((n - k) / (k + 1)) * (p / (1.0 - p))
    total = math.fsum(weights)
    return tuple(weight / total for weight in weights)


def exact_two_sided_p(gains: int, losses: int) -> float:
    """Symmetric exact paired p-value; zero discordance yields p=1."""
    gains, losses = _count(gains, "gains"), _count(losses, "losses")
    d = gains + losses
    if d == 0:
        return 1.0
    extreme = max(gains, losses)
    tail = math.fsum(binomial_pmf(d, 0.5)[extreme:])
    return min(1.0, 2.0 * tail)


def rejects_beneficial(gains: int, losses: int, alpha: float = PER_COMPARISON_ALPHA) -> bool:
    """A significant harmful difference cannot satisfy a gain claim."""
    alpha = _alpha(alpha)
    gains, losses = _count(gains, "gains"), _count(losses, "losses")
    return gains > losses and exact_two_sided_p(gains, losses) <= alpha


@lru_cache(maxsize=None)
def critical_gain_count(d: int, alpha: float = PER_COMPARISON_ALPHA) -> int | None:
    """Minimum beneficial gain count that rejects, or None if impossible."""
    d, alpha = _count(d, "d"), _alpha(alpha)
    if d == 0:
        return None
    null = binomial_pmf(d, 0.5)
    tail = 0.0
    critical = None
    for gains in range(d, d // 2, -1):
        tail = math.fsum((tail, null[gains]))
        if 2.0 * tail <= alpha:
            critical = gains
        else:
            break
    return critical


def paired_gain_power(
    n: int, gain: float, loss: float, alpha: float = PER_COMPARISON_ALPHA
) -> float:
    """Probability of a beneficial rejection under assumed joint rates.

    gain=P(candidate correct, comparator wrong); loss=P(candidate wrong,
    comparator correct). The remaining mass is concordance. A world contributes
    at most one pair to this computation, irrespective of prompt variants.
    """
    n, gain, loss, alpha = _count(n, "n"), _probability(gain, "gain"), _probability(loss, "loss"), _alpha(alpha)
    discordance = gain + loss
    if discordance > 1.0:
        raise ValueError("gain + loss must not exceed one")
    if n == 0 or discordance == 0.0 or gain == 0.0:
        return 0.0
    conditional_gain = gain / discordance
    terms = []
    for d, d_mass in enumerate(binomial_pmf(n, discordance)):
        threshold = critical_gain_count(d, alpha)
        if threshold is not None and d_mass:
            conditional_tail = math.fsum(binomial_pmf(d, conditional_gain)[threshold:])
            terms.append(d_mass * conditional_tail)
    return min(1.0, max(0.0, math.fsum(terms)))


def simultaneous_power_lower_bound(per_comparator_powers: tuple[float, ...]) -> float:
    """Union-bound power for passing every comparison; dependence is allowed."""
    if not per_comparator_powers:
        raise ValueError("at least one comparator power is required")
    powers = tuple(_probability(p, "power") for p in per_comparator_powers)
    return max(0.0, 1.0 - math.fsum(1.0 - p for p in powers))


def prospective_rows() -> list[dict[str, float | int | str]]:
    rows = []
    for label, gain, loss in SCENARIOS:
        for n in SAMPLE_SIZES:
            power = paired_gain_power(n, gain, loss)
            rows.append({
                "scenario": label, "n": n, "gain": gain, "loss": loss,
                "net_gain": gain - loss, "per_comparator_power": power,
                "all_eight_power_lower_bound": simultaneous_power_lower_bound((power,) * len(COMPARATORS)),
            })
    return rows


def main() -> None:
    started = time.perf_counter()
    print(f"Prospective exact paired tests: alpha={PER_COMPARISON_ALPHA:.6f}, comparators={len(COMPARATORS)}")
    print("Assumed joint rates only; independent source worlds; format variants do not increase N.")
    print("scenario\tN\tgain\tloss\tper-comparator power\tall-eight lower bound")
    for row in prospective_rows():
        print(f"{row['scenario']}\t{row['n']}\t{row['gain']:.2f}\t{row['loss']:.2f}\t"
              f"{row['per_comparator_power']:.9f}\t{row['all_eight_power_lower_bound']:.9f}")
    print(f"Exploratory 3 gains / 0 losses: two-sided p={exact_two_sided_p(3, 0):.2f}; cannot confirm.")
    print(f"CPU calculator elapsed seconds: {time.perf_counter() - started:.6f}")


if __name__ == "__main__":
    main()
