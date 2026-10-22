"""Bounded stdlib statistics for descriptive historical pattern checks.

These calculations distinguish a one-sided binomial rejection threshold from
the lower bound of a two-sided Clopper-Pearson interval. They do not change any
historical registration, establish independent trials, or supply new efficacy
evidence. Exact binomial distributions are evaluated with floating-point sums;
no normal approximation, model, tokenizer, or optional scientific library is used.
"""
from __future__ import annotations

import math

MAX_N = 512
BISECTION_STEPS = 180


def _trials(n: int) -> None:
    if type(n) is not int or not 1 <= n <= MAX_N:
        raise ValueError("n must be an integer from 1 through 512")


def _count(k: int, n: int, name: str = "k") -> None:
    _trials(n)
    if type(k) is not int or not 0 <= k <= n:
        raise ValueError(name + " must be an integer from 0 through n")


def _probability(value: float, name: str, *, open_interval: bool = False) -> None:
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError(name + " must be a finite number")
    valid = 0 < value < 1 if open_interval else 0 <= value <= 1
    if not valid:
        raise ValueError(name + " outside its probability interval")


def upper_tail(k: int, n: int, p: float) -> float:
    """Return ``P(X >= k)`` for ``X ~ Binomial(n, p)``; ``1 <= n <= 512``.

    Counts and probabilities are validated even at analytic endpoints. ``fsum``
    accumulates the finite binomial terms; clamping removes sum roundoff outside
    [0, 1]. The name "exact" describes the distribution, not arbitrary precision.
    """
    _count(k, n)
    _probability(p, "p")
    if k == 0:
        return 1.0
    if p == 0:
        return 0.0
    if p == 1:
        return 1.0
    value = math.fsum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                      for i in range(k, n + 1))
    return min(1.0, max(0.0, value))


def _lower_bound(k: int, n: int, tail_probability: float) -> float:
    if k == 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2
        if upper_tail(k, n, middle) > tail_probability:
            high = middle
        else:
            low = middle
    return (low + high) / 2


def clopper_pearson(k: int, n: int, confidence: float = .95) -> tuple[float, float]:
    """Return the equal-tailed two-sided exact binomial interval.

    Each tail receives ``(1-confidence)/2``. Analytic boundaries are lower=0
    for k=0 and upper=1 for k=n. Bisection uses a fixed iteration count; the
    upper bound is obtained by binomial symmetry to avoid subtracting a CDF.
    """
    _count(k, n)
    _probability(confidence, "confidence", open_interval=True)
    tail_probability = (1 - confidence) / 2
    lower = _lower_bound(k, n, tail_probability)
    upper = 1.0 - _lower_bound(n - k, n, tail_probability)
    return lower, upper


def critical_count(n: int, p: float, alpha: float = .05) -> int:
    """Return the first count with one-sided tail strictly less than alpha.

    Return n+1 when no attainable count clears. Strict rejection matches the
    historical implemented rule's strictly-above lower-bound comparison; equality
    is not silently promoted to a pass. A two-sided confidence-C lower-bound
    comparison uses alpha=(1-C)/2, not alpha=1-C.
    """
    _trials(n)
    _probability(p, "p")
    _probability(alpha, "alpha", open_interval=True)
    for k in range(n + 1):
        if upper_tail(k, n, p) < alpha:
            return k
    return n + 1


def is_floor_clear(k: int, n: int, floor_count: int, confidence: float = .95) -> bool:
    """Test whether the two-sided interval lower bound exceeds floor_count/n.

    The shortcut floor uses the exact integer ratio, never a rounded report rate.
    This is a numerical check of the stated historical rule, not a claim that the
    binomial independence assumptions or experiment provenance have been met.
    """
    _count(k, n)
    _count(floor_count, n, "floor_count")
    lower, _ = clopper_pearson(k, n, confidence)
    return lower > floor_count / n
