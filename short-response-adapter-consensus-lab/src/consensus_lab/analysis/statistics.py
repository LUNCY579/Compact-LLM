from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import binomtest


@dataclass(frozen=True)
class PairedBinaryResult:
    n: int
    rate_a: float
    rate_b: float
    difference: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    mcnemar_b: int
    mcnemar_c: int
    mcnemar_exact_p: float
    sign_flip_p: float

    def to_dict(self) -> dict[str, int | float]:
        return self.__dict__.copy()


def paired_binary_statistics(
    a: Iterable[bool],
    b: Iterable[bool],
    *,
    seed: int,
    bootstrap_samples: int = 20_000,
    permutation_samples: int = 20_000,
) -> PairedBinaryResult:
    left = np.asarray(list(a), dtype=np.int8)
    right = np.asarray(list(b), dtype=np.int8)
    if left.shape != right.shape or left.ndim != 1 or len(left) == 0:
        raise ValueError("Paired inputs must be nonempty one-dimensional arrays of equal length")
    rng = np.random.default_rng(seed)
    differences = left.astype(float) - right.astype(float)
    indices = rng.integers(0, len(left), size=(bootstrap_samples, len(left)))
    boot = differences[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    b_only = int(np.sum((left == 1) & (right == 0)))
    c_only = int(np.sum((left == 0) & (right == 1)))
    discordant = b_only + c_only
    mcnemar_p = 1.0 if discordant == 0 else float(binomtest(min(b_only, c_only), discordant, 0.5).pvalue)
    observed = abs(float(differences.mean()))
    if discordant == 0:
        permutation_p = 1.0
    elif discordant <= 20:
        # Exact sign-flip distribution for small paired pilots.
        magnitudes = np.abs(differences[differences != 0])
        exceed = 0
        total = 1 << discordant
        for mask in range(total):
            signs = np.fromiter(
                (1.0 if mask & (1 << index) else -1.0 for index in range(discordant)),
                dtype=float,
            )
            exceed += abs(float(np.sum(magnitudes * signs) / len(left))) >= observed - 1e-15
        permutation_p = exceed / total
    else:
        magnitudes = np.abs(differences[differences != 0])
        signs = rng.choice((-1.0, 1.0), size=(permutation_samples, discordant))
        simulated = np.abs((signs * magnitudes).sum(axis=1) / len(left))
        permutation_p = (float(np.sum(simulated >= observed - 1e-15)) + 1.0) / (
            permutation_samples + 1.0
        )
    return PairedBinaryResult(
        n=len(left),
        rate_a=float(left.mean()),
        rate_b=float(right.mean()),
        difference=float(differences.mean()),
        bootstrap_ci_low=float(low),
        bootstrap_ci_high=float(high),
        mcnemar_b=b_only,
        mcnemar_c=c_only,
        mcnemar_exact_p=mcnemar_p,
        sign_flip_p=float(permutation_p),
    )


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [math.nan] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted
