"""Interval estimators for the crossed seed x example Phase-1 design.

## The design

Every training seed is evaluated on the *same* materialized test split, so the
paired per-example difference

    d[s, k, e] = correct(candidate, s, k, e) - correct(baseline, s, k, e)

is indexed by a training seed ``s``, a registered stratum ``k``, and an example
``e`` inside that stratum. Seed and example are **crossed**, not nested: example
``e`` appears once under every seed.

The estimand is the registered macro-average

    theta = (1/m) * sum_s (1/K) * sum_k mean_e d[s, k, e]

which is a fixed-weight linear functional ``theta = sum_i w_i * d_i`` with
``w_i = 1 / (m * K * n_k)``.

## Why the estimator matters

Writing ``d[s,k,e] = theta + a_s + b_[k,e] + eps[s,k,e]`` with a seed effect
``a``, a *shared* example effect ``b``, and interaction noise ``eps``:

    Var(theta_hat) = sigma_a^2 / m
                   + sigma_b^2 / (K * n)
                   + sigma_eps^2 / (m * K * n)

Two consequences drive the whole v3 design:

* the ``sigma_b^2`` term does **not** shrink as seeds are added, so replication
  buys less than a naive nested calculation suggests, and there is a hard floor
  on achievable precision;
* an estimator that mishandles the shared example structure will get the second
  term wrong, and with few seeds the first term is estimated from very few
  degrees of freedom.

## The estimators

``crossed_pigeonhole`` (v2 incumbent)
    Owen's pigeonhole bootstrap: resample seed IDs and example IDs independently
    with replacement and take the intersection cells, then take percentile
    limits. Asymptotically valid for crossed data, but percentile limits from a
    handful of seed clusters are known to be anticonservative.

``crossed_cluster_t`` (v3 candidate)
    Two-way cluster-robust variance in the sense of Cameron, Gelbach & Miller
    (2011), with a ``t_{m-1}`` reference distribution:

        V = c_s * V_seed + c_e * V_example - V_cell

    where each ``V`` is a sum of squared cluster totals of the weighted
    residuals ``u_i = w_i * (d_i - theta_hat)``. Taking expectations term by
    term recovers ``sigma_a^2/m + sigma_b^2/(Kn) + sigma_eps^2/(mKn)`` up to an
    ``O(1/(mKn))`` correction, so the estimator is essentially unbiased for the
    true sampling variance. The ``t_{m-1}`` reference is the standard
    few-clusters correction: the seed component carries only ``m - 1`` degrees
    of freedom.

``crossed_pigeonhole_t`` (v3 candidate)
    Studentized pigeonhole bootstrap. Each resample is standardised by its own
    two-way cluster-robust standard error, so the interval is built from the
    bootstrap distribution of a pivot rather than of the raw statistic.
    Studentizing generally improves small-sample coverage over percentile limits.

``seed_cluster`` (sensitivity only)
    Resample training seeds only, treating the generated test split as fixed.
    Narrower claim: "across training seeds, for this materialized test split".
    Retained as a preregistered sensitivity analysis. It must not become the
    primary analysis merely because it yields significance more easily.

Nothing here inspects a confirmatory outcome. The same functions serve the
outcome-blind simulation and the real analysis, so the procedure that is
calibrated is exactly the procedure that is registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

INFERENCE_SCHEMA_VERSION = "tgsa.phase1-inference.v1"

# Registered estimator names. `resolve_interval_estimator` refuses anything else.
PRIMARY_ESTIMATOR = "crossed_cluster_t"


@dataclass(frozen=True, slots=True)
class IntervalResult:
    """One interval estimate plus everything a gate or an audit needs."""

    observed: float
    confidence_low: float
    confidence_high: float
    standard_error: float
    degrees_of_freedom: float
    probability_positive: float
    estimator: str
    confidence: float
    seed_count: int
    stratum_count: int
    example_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "observed": self.observed,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "standard_error": self.standard_error,
            "degrees_of_freedom": self.degrees_of_freedom,
            "probability_positive": self.probability_positive,
            "estimator": self.estimator,
            "confidence": self.confidence,
            "seed_count": self.seed_count,
            "stratum_count": self.stratum_count,
            "example_count": self.example_count,
        }


def _student_t_quantile(probability: float, degrees_of_freedom: float) -> float:
    """Two-sided Student-t quantile, SciPy when present, else a good fallback."""

    try:  # pragma: no cover - exercised whichever branch the environment has
        from scipy.stats import t as _t

        return float(_t.ppf(probability, degrees_of_freedom))
    except Exception:  # pragma: no cover
        # Cornish-Fisher expansion of the t quantile about the normal quantile.
        # Accurate to <1% for df >= 3, which covers every registered seed count.
        from statistics import NormalDist

        z = NormalDist().inv_cdf(probability)
        nu = float(degrees_of_freedom)
        return float(
            z
            + (z**3 + z) / (4.0 * nu)
            + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * nu**2)
            + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * nu**3)
        )


def _probability_positive(
    observed: float, standard_error: float, degrees_of_freedom: float
) -> float:
    """One-sided evidence summary under the estimator's own reference law.

    For the analytic estimators this is the ``t_{m-1}`` tail probability that the
    effect exceeds zero. It is a descriptive summary, never a gate: the
    registered gates are the practical-effect threshold and the lower confidence
    bound.
    """

    if standard_error <= 0.0:
        # A degenerate estimate carries no directional evidence at exactly zero.
        if observed == 0.0:
            return 0.5
        return 1.0 if observed > 0.0 else 0.0
    statistic = observed / standard_error
    try:  # pragma: no cover - environment dependent
        from scipy.stats import t as _t

        return float(_t.cdf(statistic, degrees_of_freedom))
    except Exception:  # pragma: no cover
        from statistics import NormalDist

        return float(NormalDist().cdf(statistic))


def _as_panel(
    values: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Convert the analysis dict form to a dense ``(m, K, n_max)`` panel.

    Returns ``(panel, mask, seed_ids, strata)``. Ragged strata are padded and
    masked rather than dropped, so unequal stratum sizes are handled exactly.
    """

    seed_ids = sorted(values)
    if not seed_ids:
        raise ValueError("no paired seeds")
    strata = sorted(values[seed_ids[0]])
    if not strata:
        raise ValueError("no paired strata")
    example_ids: dict[str, list[str]] = {}
    for stratum in strata:
        ids = sorted(values[seed_ids[0]][stratum])
        if not ids:
            raise ValueError("all seed/stratum cells must be non-empty")
        example_ids[stratum] = ids
    for seed in seed_ids:
        if sorted(values[seed]) != strata:
            raise ValueError("all seeds must share the same strata")
        for stratum in strata:
            if sorted(values[seed][stratum]) != example_ids[stratum]:
                raise ValueError(
                    "all seeds must share the same examples inside each stratum; "
                    "seed and example are not crossed"
                )
    width = max(len(example_ids[stratum]) for stratum in strata)
    panel = np.zeros((len(seed_ids), len(strata), width), dtype=np.float64)
    mask = np.zeros((len(strata), width), dtype=bool)
    for stratum_index, stratum in enumerate(strata):
        ids = example_ids[stratum]
        mask[stratum_index, : len(ids)] = True
        for seed_index, seed in enumerate(seed_ids):
            cell = values[seed][stratum]
            panel[seed_index, stratum_index, : len(ids)] = [cell[key] for key in ids]
    return panel, mask, seed_ids, strata


def _weights(mask: np.ndarray, seed_count: int) -> np.ndarray:
    """Registered macro-average weights ``1 / (m * K * n_k)``, zero where masked."""

    counts = mask.sum(axis=1).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("a registered stratum has no examples")
    per_example = 1.0 / (seed_count * mask.shape[0] * counts)
    return mask.astype(np.float64) * per_example[:, None]


def point_estimate(panel: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Registered macro-average. Leading axes are treated as trial axes."""

    return np.einsum("...ske,ke->...", panel, weights)


def two_way_cluster_variance(
    panel: np.ndarray,
    weights: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Cameron-Gelbach-Miller two-way cluster-robust variance of the estimand.

    ``V = c_s * V_seed + c_e * V_example - V_cell`` on the weighted residuals.
    The seed dimension is corrected by ``m / (m - 1)`` and the example dimension
    by ``N / (N - 1)``; the intersection term needs no correction because its
    clusters are the individual cells.

    Leading axes of ``panel`` are trial axes and are vectorised over.
    """

    seed_count = panel.shape[-3]
    example_count = int(np.count_nonzero(weights))
    residual = weights * (panel - theta[..., None, None, None])

    seed_totals = residual.sum(axis=(-2, -1))
    variance_seed = np.square(seed_totals).sum(axis=-1)

    example_totals = residual.sum(axis=-3)
    variance_example = np.square(example_totals).sum(axis=(-2, -1))

    variance_cell = np.square(residual).sum(axis=(-3, -2, -1))

    correction_seed = seed_count / max(seed_count - 1, 1)
    correction_example = example_count / max(example_count - 1, 1)
    variance = (
        correction_seed * variance_seed
        + correction_example * variance_example
        - variance_cell
    )
    # A two-way estimator can go negative in small samples. CGM's recommended
    # remedy is to fall back on the (always non-negative) seed component, which
    # is the dominant term here.
    fallback = correction_seed * variance_seed
    return np.where(variance > 0.0, variance, fallback)


def crossed_cluster_t(
    panel: np.ndarray,
    weights: np.ndarray,
    *,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-way cluster-robust interval with a ``t_{m-1}`` reference."""

    seed_count = panel.shape[-3]
    theta = point_estimate(panel, weights)
    variance = two_way_cluster_variance(panel, weights, theta)
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    degrees_of_freedom = max(seed_count - 1, 1)
    critical = _student_t_quantile(0.5 + confidence / 2.0, degrees_of_freedom)
    half_width = critical * standard_error
    return theta, theta - half_width, theta + half_width, standard_error


def _pigeonhole_draws(
    panel: np.ndarray,
    weights: np.ndarray,
    *,
    samples: int,
    generator: np.random.Generator,
    studentized: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Owen pigeonhole resampling: seeds and examples resampled independently."""

    seed_count, stratum_count, width = panel.shape[-3:]
    counts = (weights > 0.0).sum(axis=1)
    seed_pick = generator.integers(0, seed_count, size=(samples, seed_count))
    example_pick = np.empty((samples, stratum_count, width), dtype=np.int64)
    for stratum in range(stratum_count):
        size = int(counts[stratum])
        example_pick[:, stratum, :size] = generator.integers(0, size, size=(samples, size))
        example_pick[:, stratum, size:] = 0

    draws = np.empty(samples, dtype=np.float64)
    errors = np.empty(samples, dtype=np.float64) if studentized else None
    stratum_axis = np.arange(stratum_count)[None, :, None]
    for index in range(samples):
        resampled = panel[seed_pick[index]][:, stratum_axis[0], example_pick[index]]
        theta = point_estimate(resampled, weights)
        draws[index] = theta
        if errors is not None:
            variance = two_way_cluster_variance(resampled, weights, np.asarray(theta))
            errors[index] = float(np.sqrt(max(float(variance), 0.0)))
    return draws, errors


def crossed_pigeonhole(
    panel: np.ndarray,
    weights: np.ndarray,
    *,
    confidence: float,
    samples: int,
    rng_seed: int,
) -> tuple[float, float, float, float]:
    """Percentile pigeonhole bootstrap. The v2 incumbent."""

    generator = np.random.default_rng(rng_seed)
    theta = float(point_estimate(panel, weights))
    draws, _ = _pigeonhole_draws(
        panel, weights, samples=samples, generator=generator, studentized=False
    )
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return theta, float(low), float(high), float(draws.std(ddof=1))


def crossed_pigeonhole_t(
    panel: np.ndarray,
    weights: np.ndarray,
    *,
    confidence: float,
    samples: int,
    rng_seed: int,
) -> tuple[float, float, float, float]:
    """Studentized pigeonhole bootstrap (bootstrap-t)."""

    generator = np.random.default_rng(rng_seed)
    theta = float(point_estimate(panel, weights))
    standard_error = float(
        np.sqrt(max(float(two_way_cluster_variance(panel, weights, np.asarray(theta))), 0.0))
    )
    draws, errors = _pigeonhole_draws(
        panel, weights, samples=samples, generator=generator, studentized=True
    )
    assert errors is not None
    safe = np.maximum(errors, 1.0e-12)
    pivots = (draws - theta) / safe
    alpha = (1.0 - confidence) / 2.0
    # Bootstrap-t inverts the pivot, so the upper pivot quantile sets the LOWER
    # confidence limit.
    lower_pivot, upper_pivot = np.quantile(pivots, [alpha, 1.0 - alpha])
    return (
        theta,
        theta - float(upper_pivot) * standard_error,
        theta - float(lower_pivot) * standard_error,
        standard_error,
    )


def seed_cluster(
    panel: np.ndarray,
    weights: np.ndarray,
    *,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample training seeds only; the generated test split is treated as fixed."""

    seed_count = panel.shape[-3]
    per_seed = np.einsum("...ske,ke->...s", panel, weights) * seed_count
    theta = per_seed.mean(axis=-1)
    standard_error = per_seed.std(axis=-1, ddof=1) / np.sqrt(seed_count)
    degrees_of_freedom = max(seed_count - 1, 1)
    critical = _student_t_quantile(0.5 + confidence / 2.0, degrees_of_freedom)
    half_width = critical * standard_error
    return theta, theta - half_width, theta + half_width, standard_error


ANALYTIC_ESTIMATORS = {"crossed_cluster_t": crossed_cluster_t, "seed_cluster": seed_cluster}
BOOTSTRAP_ESTIMATORS = {
    "crossed_pigeonhole": crossed_pigeonhole,
    "crossed_pigeonhole_t": crossed_pigeonhole_t,
}
INTERVAL_ESTIMATORS = tuple(sorted({*ANALYTIC_ESTIMATORS, *BOOTSTRAP_ESTIMATORS}))


def resolve_interval_estimator(name: str) -> str:
    if name not in INTERVAL_ESTIMATORS:
        raise ValueError(
            f"unsupported interval estimator {name!r}; registered estimators are "
            f"{list(INTERVAL_ESTIMATORS)}"
        )
    return name


def estimate_interval(
    values: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    estimator: str,
    confidence: float,
    samples: int = 10000,
    rng_seed: int = 91273,
) -> IntervalResult:
    """Registered entry point used by the frozen analyzer.

    The simulation calls the same underlying estimators on the same panel
    representation, so the calibrated procedure is the registered procedure.
    """

    resolve_interval_estimator(estimator)
    panel, mask, seed_ids, strata = _as_panel(values)
    if len(seed_ids) < 2:
        # Seed-level variability is unidentifiable from a single training seed.
        # A single-seed panel silently yields a NaN or zero-width interval, which
        # would read as infinite precision. Refuse instead.
        raise ValueError(
            f"interval estimation requires at least two training seeds; got {len(seed_ids)}"
        )
    weights = _weights(mask, len(seed_ids))
    example_count = int(mask.sum())

    if estimator in ANALYTIC_ESTIMATORS:
        theta, low, high, standard_error = ANALYTIC_ESTIMATORS[estimator](
            panel, weights, confidence=confidence
        )
        degrees_of_freedom = float(max(len(seed_ids) - 1, 1))
        theta, low, high, standard_error = (
            float(theta),
            float(low),
            float(high),
            float(standard_error),
        )
    else:
        theta, low, high, standard_error = BOOTSTRAP_ESTIMATORS[estimator](
            panel, weights, confidence=confidence, samples=samples, rng_seed=rng_seed
        )
        degrees_of_freedom = float("nan")

    return IntervalResult(
        observed=theta,
        confidence_low=low,
        confidence_high=high,
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        probability_positive=_probability_positive(
            theta,
            standard_error,
            degrees_of_freedom if degrees_of_freedom == degrees_of_freedom else 1e9,
        ),
        estimator=estimator,
        confidence=confidence,
        seed_count=len(seed_ids),
        stratum_count=len(strata),
        example_count=example_count,
    )
