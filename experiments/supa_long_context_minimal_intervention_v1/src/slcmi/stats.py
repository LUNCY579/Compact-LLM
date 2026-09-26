"""Paired statistics for the pilot.

Every comparison in this lab is paired: the same example is scored under
different conditions or different checkpoints, so the resampling unit is the
example, not the observation.

No p-value is reported as "significant". The lab's own precedent
(``Rineya-Supa-A1.1/CORE.json``) is explicit that this corpus is underpowered --
roughly 576 rows would be needed for 80% power at the effects seen here, and
there are 256. Intervals are reported so the reader can see the width.
"""

from __future__ import annotations

import random
from typing import Sequence

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260905


def paired_bootstrap_delta(
    a: Sequence[bool],
    b: Sequence[bool],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI for mean(b) - mean(a) over paired per-example outcomes.

    Resamples example indices with replacement, so the pairing is preserved and
    the interval reflects the correlation between conditions.
    """
    if len(a) != len(b):
        raise ValueError(f"unpaired inputs: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return {"n": 0, "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "resamples": 0}

    av = [1.0 if x else 0.0 for x in a]
    bv = [1.0 if x else 0.0 for x in b]
    observed = sum(bv) / n - sum(av) / n

    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(bv[i] - av[i] for i in idx) / n)
    deltas.sort()
    lo = deltas[int((alpha / 2) * resamples)]
    hi = deltas[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return {
        "n": n,
        "delta": round(observed, 6),
        "delta_points": round(100 * observed, 3),
        "ci_low_points": round(100 * lo, 3),
        "ci_high_points": round(100 * hi, 3),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "resamples": resamples,
        "seed": seed,
    }


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Two-sided exact McNemar on the discordant pairs."""
    if len(a) != len(b):
        raise ValueError(f"unpaired inputs: {len(a)} vs {len(b)}")
    b01 = sum(1 for x, y in zip(a, b) if not x and y)  # a wrong, b right
    b10 = sum(1 for x, y in zip(a, b) if x and not y)  # a right, b wrong
    n = b01 + b10
    if n == 0:
        return {"b_only_correct": 0, "a_only_correct": 0, "discordant": 0, "p_value": 1.0}

    from math import comb

    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {
        "b_only_correct": b01,
        "a_only_correct": b10,
        "discordant": n,
        "p_value": round(min(1.0, 2 * tail), 6),
    }


def relative_error_reduction(baseline_acc: float, arm_acc: float) -> float:
    """Fraction of the baseline's errors removed by the arm."""
    err = 1.0 - baseline_acc
    if err <= 0:
        return 0.0
    return (arm_acc - baseline_acc) / err


def summarise(correct: Sequence[bool]) -> dict:
    n = len(correct)
    k = sum(1 for c in correct if c)
    return {
        "n": n,
        "correct": k,
        "accuracy": round(k / n, 6) if n else 0.0,
        "accuracy_points": round(100 * k / n, 2) if n else 0.0,
    }
