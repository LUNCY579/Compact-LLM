"""High-precision outcome-blind simulation of the registered decision rule.

Nothing in this module reads a checkpoint, a prediction file, or any TGSA
outcome. It draws synthetic paired differences from an explicit generative model
that mirrors the registered crossed seed x example structure, then runs the
*registered* estimators from :mod:`tgsa.phase1_inference` over them.

Generative model, matching the decomposition the inference module documents::

    d[s, k, e] in {-1, 0, +1}
    E d[s, k, e] = theta + a_s + b_[k, e]
    a_s      ~ N(0, sigma_a^2)          per training seed
    b_[k, e] ~ N(0, sigma_b^2)          per example, SHARED across seeds
    P(+1) = (rate + mu) / 2,  P(-1) = (rate - mu) / 2

``rate`` is the disagreement rate: the fraction of examples on which the two arms
give different answers. This parameterisation is calibrated so that the expected
paired difference is exactly ``mu``, which makes ``theta`` directly comparable to
the registered accuracy-delta threshold.

The shared ``b`` term is the whole point. It is why seed and example are crossed,
why ``sigma_b^2 / (K * n)`` does not shrink with replication, and why an
estimator that treats the design as nested is miscalibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .phase1_inference import (
    ANALYTIC_ESTIMATORS,
    BOOTSTRAP_ESTIMATORS,
    _weights,
    crossed_cluster_t,
    point_estimate,
    resolve_interval_estimator,
    seed_cluster,
    two_way_cluster_variance,
)

SIMULATION_SCHEMA_VERSION = "tgsa.phase1-simulation.v2"


@dataclass(frozen=True, slots=True)
class DesignPoint:
    """One simulated world plus the replication level being evaluated."""

    seed_count: int
    true_effect: float
    seed_sd: float
    example_sd: float = 0.05
    disagreement_rate: float = 0.30
    stratum_sd: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "seed_count": self.seed_count,
            "true_effect": self.true_effect,
            "seed_sd": self.seed_sd,
            "example_sd": self.example_sd,
            "disagreement_rate": self.disagreement_rate,
            "stratum_sd": self.stratum_sd,
        }


@dataclass(frozen=True, slots=True)
class StratumLayout:
    """The registered example/stratum structure the simulation must preserve."""

    stratum_count: int
    examples_per_stratum: tuple[int, ...]

    @classmethod
    def balanced(cls, stratum_count: int, per_stratum: int) -> "StratumLayout":
        return cls(stratum_count, tuple([per_stratum] * stratum_count))

    @property
    def example_count(self) -> int:
        return sum(self.examples_per_stratum)

    @property
    def width(self) -> int:
        return max(self.examples_per_stratum)

    def mask(self) -> np.ndarray:
        mask = np.zeros((self.stratum_count, self.width), dtype=bool)
        for index, count in enumerate(self.examples_per_stratum):
            mask[index, :count] = True
        return mask

    def to_dict(self) -> dict[str, object]:
        return {
            "stratum_count": self.stratum_count,
            "examples_per_stratum": list(self.examples_per_stratum),
            "example_count": self.example_count,
        }


def simulate_panels(
    design: DesignPoint,
    layout: StratumLayout,
    *,
    trials: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Draw ``trials`` independent panels of shape ``(trials, m, K, width)``."""

    mask = layout.mask()
    shape = (trials, design.seed_count, layout.stratum_count, layout.width)

    seed_effect = generator.normal(0.0, design.seed_sd, size=(trials, design.seed_count, 1, 1))
    # One example effect per (stratum, example), reused by every seed. This is
    # the shared structure that makes the design crossed.
    example_effect = generator.normal(
        0.0, design.example_sd, size=(trials, 1, layout.stratum_count, layout.width)
    )
    mean = design.true_effect + seed_effect + example_effect
    if design.stratum_sd > 0.0:
        # DIAGNOSTIC MISSPECIFICATION, off by default. The registered strata are
        # a FIXED set chosen by the generator, so the estimand macro-averages
        # over those exact strata and a stratum-level random effect is not a
        # source of sampling variability. Injecting one adds variance that no
        # correctly-specified estimator can see, which manufactures apparent
        # anticonservatism. This knob exists to quantify that, not to use it.
        mean = mean + generator.normal(
            0.0, design.stratum_sd, size=(trials, 1, layout.stratum_count, 1)
        )

    rate = design.disagreement_rate
    positive = np.clip(0.5 * (rate + mean), 0.0, rate)
    negative = np.clip(rate - positive, 0.0, rate)

    uniform = generator.random(size=shape)
    panel = np.where(uniform < positive, 1.0, np.where(uniform < positive + negative, -1.0, 0.0))
    return panel * mask[None, None, :, :]


def _pigeonhole_counts(
    seed_count: int,
    stratum_sizes: np.ndarray,
    width: int,
    *,
    samples: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicities of each seed and each example in ``samples`` resamples.

    The pigeonhole bootstrap draws seed IDs and example IDs independently with
    replacement and keeps the intersection cells. A resample is fully described
    by how many times each seed and each example was drawn, which is what makes
    the counts formulation below exact rather than an approximation.
    """

    seed_pick = generator.integers(0, seed_count, size=(samples, seed_count))
    seed_counts = np.zeros((samples, seed_count), dtype=np.float64)
    np.add.at(seed_counts, (np.arange(samples)[:, None], seed_pick), 1.0)

    example_counts = np.zeros((samples, stratum_sizes.size, width), dtype=np.float64)
    for stratum in range(stratum_sizes.size):
        size = int(stratum_sizes[stratum])
        pick = generator.integers(0, size, size=(samples, size))
        np.add.at(example_counts[:, stratum, :], (np.arange(samples)[:, None], pick), 1.0)
    return seed_counts, example_counts


def _bootstrap_intervals(
    panel: np.ndarray,
    weights: np.ndarray,
    *,
    estimator: str,
    confidence: float,
    samples: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pigeonhole bootstrap intervals via exact resample sufficient statistics.

    Materialising every resampled panel costs ``trials x samples x m x K x n``
    floats, which is hundreds of gigabytes at the precision this audit needs.
    Because the estimand and the two-way cluster variance are both linear or
    quadratic in the resample multiplicities, each resample statistic can be
    evaluated directly from the seed and example counts. This is algebraically
    identical to resampling, not an approximation.
    """

    trials, seed_count, stratum_count, width = panel.shape
    stratum_sizes = (weights > 0.0).sum(axis=1)
    per_stratum_weight = weights.max(axis=1)  # W_k, constant within a stratum

    theta = point_estimate(panel, weights)
    alpha = (1.0 - confidence) / 2.0
    lows = np.empty(trials, dtype=np.float64)
    highs = np.empty(trials, dtype=np.float64)
    spreads = np.empty(trials, dtype=np.float64)

    if estimator == "crossed_pigeonhole_t":
        observed_error = np.sqrt(
            np.maximum(two_way_cluster_variance(panel, weights, theta), 0.0)
        )
    else:
        observed_error = np.zeros(trials)

    correction_seed = seed_count / max(seed_count - 1, 1)
    example_total = int(stratum_sizes.sum())
    correction_example = example_total / max(example_total - 1, 1)

    for trial in range(trials):
        board = panel[trial]  # (m, K, width)
        seed_counts, example_counts = _pigeonhole_counts(
            seed_count, stratum_sizes, width, samples=samples, generator=generator
        )
        # Q[b, k, e] = sum over resampled seed slots of the cell value.
        flat = board.reshape(seed_count, -1)
        draws_q = (seed_counts @ flat).reshape(samples, stratum_count, width)
        weighted = per_stratum_weight[None, :, None] * example_counts
        draws = np.einsum("bke,bke->b", weighted, draws_q)

        if estimator == "crossed_pigeonhole":
            lows[trial], highs[trial] = np.quantile(draws, [alpha, 1.0 - alpha])
            spreads[trial] = draws.std(ddof=1)
            continue

        # Two-way cluster variance of each resample, from the same counts.
        # R[b, s] = sum_k W_k sum_e n_count[b,k,e] * P[s,k,e]
        contribution = np.einsum(
            "bke,ske->bs", weighted, board
        )
        seed_term = np.einsum(
            "bs,bs->b", seed_counts, np.square(contribution - draws[:, None] / seed_count)
        )
        example_term = np.einsum(
            "bke,bke->b",
            example_counts,
            np.square(per_stratum_weight[None, :, None] * (draws_q - seed_count * draws[:, None, None])),
        )
        # Expand the cell term instead of materialising (samples, m, K, width):
        #   sum_k W_k^2 * sum_s c_s * sum_e n_e * (P - theta*)^2
        # = sum_k W_k^2 * [ S2 - 2*theta* * S1 + theta*^2 * m * n_k ]
        # where S2 and S1 contract the example axis first, leaving (b, s, k).
        squared_board = np.square(board)
        s2 = np.einsum("bs,bke,ske->bk", seed_counts, example_counts, squared_board)
        s1 = np.einsum("bs,bke,ske->bk", seed_counts, example_counts, board)
        totals = seed_count * stratum_sizes.astype(np.float64)[None, :]
        cell_term = np.einsum(
            "k,bk->b",
            np.square(per_stratum_weight),
            s2 - 2.0 * draws[:, None] * s1 + np.square(draws)[:, None] * totals,
        )
        variance = (
            correction_seed * seed_term + correction_example * example_term - cell_term
        )
        variance = np.where(variance > 0.0, variance, correction_seed * seed_term)
        pivots = (draws - theta[trial]) / np.maximum(np.sqrt(np.maximum(variance, 0.0)), 1.0e-12)
        lower_pivot, upper_pivot = np.quantile(pivots, [alpha, 1.0 - alpha])
        lows[trial] = theta[trial] - upper_pivot * observed_error[trial]
        highs[trial] = theta[trial] - lower_pivot * observed_error[trial]
        spreads[trial] = observed_error[trial]

    return theta, lows, highs, spreads


def evaluate_design(
    design: DesignPoint,
    layout: StratumLayout,
    *,
    estimator: str,
    confidence: float,
    practical_threshold: float,
    trials: int,
    rng_seed: int,
    bootstrap_samples: int = 999,
    chunk: int = 250,
) -> dict[str, object]:
    """Run the registered decision rule ``trials`` times and summarise it."""

    resolve_interval_estimator(estimator)
    mask = layout.mask()
    weights = _weights(mask, design.seed_count)
    generator = np.random.default_rng(rng_seed)

    observed: list[np.ndarray] = []
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    errors: list[np.ndarray] = []

    remaining = trials
    while remaining > 0:
        size = min(chunk, remaining)
        remaining -= size
        panel = simulate_panels(design, layout, trials=size, generator=generator)
        if estimator in ANALYTIC_ESTIMATORS:
            theta, low, high, standard_error = ANALYTIC_ESTIMATORS[estimator](
                panel, weights, confidence=confidence
            )
        else:
            theta, low, high, standard_error = _bootstrap_intervals(
                panel,
                weights,
                estimator=estimator,
                confidence=confidence,
                samples=bootstrap_samples,
                generator=generator,
            )
        observed.append(np.asarray(theta))
        lows.append(np.asarray(low))
        highs.append(np.asarray(high))
        errors.append(np.asarray(standard_error))

    theta = np.concatenate(observed)
    low = np.concatenate(lows)
    high = np.concatenate(highs)
    standard_error = np.concatenate(errors)

    lower_above_zero = low > 0.0
    meets_practical = theta >= practical_threshold
    conjunction = lower_above_zero & meets_practical
    covers_truth = (low <= design.true_effect) & (design.true_effect <= high)
    width = high - low

    def _rate(flags: np.ndarray) -> dict[str, float]:
        value = float(flags.mean())
        # Wilson interval: honest at rates near 0 or 1, where a Wald interval
        # would run outside [0, 1].
        n = float(flags.size)
        z = 1.959963984540054
        centre = (value + z * z / (2 * n)) / (1 + z * z / n)
        spread = (z / (1 + z * z / n)) * np.sqrt(value * (1 - value) / n + z * z / (4 * n * n))
        return {
            "rate": value,
            "monte_carlo_standard_error": float(np.sqrt(max(value * (1.0 - value), 0.0) / n)),
            "ci_low": float(max(0.0, centre - spread)),
            "ci_high": float(min(1.0, centre + spread)),
        }

    return {
        "design": design.to_dict(),
        "estimator": estimator,
        "trials": int(theta.size),
        "practical_threshold": practical_threshold,
        "gate_conjunction": _rate(conjunction),
        "gate_lower_bound_above_zero": _rate(lower_above_zero),
        "gate_practical_effect": _rate(meets_practical),
        "interval_coverage_of_truth": _rate(covers_truth),
        "observed_effect": {
            "mean": float(theta.mean()),
            "sd": float(theta.std(ddof=1)),
            "q025": float(np.quantile(theta, 0.025)),
            "q250": float(np.quantile(theta, 0.25)),
            "median": float(np.median(theta)),
            "q750": float(np.quantile(theta, 0.75)),
            "q975": float(np.quantile(theta, 0.975)),
        },
        "interval_width": {
            "mean": float(width.mean()),
            "sd": float(width.std(ddof=1)),
            "median": float(np.median(width)),
            "q025": float(np.quantile(width, 0.025)),
            "q975": float(np.quantile(width, 0.975)),
        },
        "standard_error": {
            "mean_estimated": float(standard_error.mean()),
            "empirical_sd_of_estimate": float(theta.std(ddof=1)),
            # <1 means the estimator understates the true sampling variability,
            # which is exactly how an interval becomes anticonservative.
            "calibration_ratio": float(standard_error.mean() / max(theta.std(ddof=1), 1e-12)),
        },
    }
