"""Tests for the v3 crossed seed x example interval estimators.

The estimator is the part of the design a reader is least able to check by
inspection, so these tests pin its *mathematical* properties rather than its
output values: exactness of the resampling shortcut, unbiasedness of the
variance decomposition, and correct handling of the crossed structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tgsa.phase1_inference import (
    INTERVAL_ESTIMATORS,
    PRIMARY_ESTIMATOR,
    _as_panel,
    _student_t_quantile,
    _weights,
    estimate_interval,
    point_estimate,
    resolve_interval_estimator,
    two_way_cluster_variance,
)
from tgsa.phase1_simulation import (
    DesignPoint,
    StratumLayout,
    _pigeonhole_counts,
    evaluate_design,
    simulate_panels,
)

LAB_ROOT = Path(__file__).resolve().parents[1]


def _values(seeds: int = 5, strata: int = 3, examples: int = 20, effect: float = 0.0) -> dict:
    rng = np.random.default_rng(11)
    return {
        seed: {
            f"s{k}": {
                f"e{k}_{i}": float(rng.choice([-1.0, 0.0, 1.0]) + effect)
                for i in range(examples)
            }
            for k in range(strata)
        }
        for seed in range(seeds)
    }


# --------------------------------------------------------------------------
# Registry and plumbing
# --------------------------------------------------------------------------


def test_primary_estimator_is_registered() -> None:
    assert PRIMARY_ESTIMATOR in INTERVAL_ESTIMATORS
    resolve_interval_estimator(PRIMARY_ESTIMATOR)


def test_unregistered_estimator_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported interval estimator"):
        resolve_interval_estimator("whatever_makes_it_significant")


def test_every_registered_estimator_produces_an_ordered_interval() -> None:
    values = _values(effect=0.1)
    for estimator in INTERVAL_ESTIMATORS:
        result = estimate_interval(
            values, estimator=estimator, confidence=0.95, samples=199, rng_seed=3
        )
        assert result.confidence_low <= result.observed <= result.confidence_high
        assert result.standard_error >= 0.0
        assert 0.0 <= result.probability_positive <= 1.0
        assert result.estimator == estimator


def test_point_estimate_is_the_registered_macro_average() -> None:
    """A rare stratum must weigh as much as a populous one."""

    values = {
        seed: {"rare": {"r0": 1.0}, "common": {f"c{i}": 0.0 for i in range(50)}}
        for seed in range(4)
    }
    result = estimate_interval(
        values, estimator="crossed_cluster_t", confidence=0.95
    )
    assert result.observed == pytest.approx(0.5)


# --------------------------------------------------------------------------
# The crossed structure must be enforced, not assumed
# --------------------------------------------------------------------------


def test_non_crossed_input_is_refused() -> None:
    values = _values()
    values[1]["s0"].pop("e0_0")
    values[1]["s0"]["e0_999"] = 0.0
    with pytest.raises(ValueError, match="not crossed"):
        estimate_interval(values, estimator="crossed_cluster_t", confidence=0.95)


def test_ragged_strata_are_masked_not_dropped() -> None:
    values = {
        seed: {"small": {"a": 1.0, "b": 1.0}, "large": {f"x{i}": 0.0 for i in range(9)}}
        for seed in range(3)
    }
    panel, mask, seed_ids, strata = _as_panel(values)
    assert panel.shape == (3, 2, 9)
    assert mask.sum() == 11
    weights = _weights(mask, len(seed_ids))
    # Macro weighting: each stratum contributes 1/K regardless of its size.
    assert weights[0].sum() == pytest.approx(weights[1].sum())
    assert float(point_estimate(panel, weights)) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# The variance decomposition
# --------------------------------------------------------------------------


def test_two_way_variance_matches_the_true_sampling_variance() -> None:
    """The estimator must track Var = sa^2/m + sb^2/(Kn) + se^2/(mKn).

    This is the property the whole v3 power analysis rests on. It is checked by
    simulating many panels, comparing the mean estimated variance against the
    empirical variance of the point estimate.
    """

    layout = StratumLayout.balanced(6, 40)
    design = DesignPoint(seed_count=8, true_effect=0.0, seed_sd=0.03, example_sd=0.05)
    generator = np.random.default_rng(5)
    panel = simulate_panels(design, layout, trials=4000, generator=generator)
    weights = _weights(layout.mask(), design.seed_count)
    theta = point_estimate(panel, weights)
    variance = two_way_cluster_variance(panel, weights, theta)

    empirical = float(theta.var(ddof=1))
    estimated = float(variance.mean())
    assert estimated == pytest.approx(empirical, rel=0.12), (
        f"variance estimator is off: estimated={estimated:.3e} empirical={empirical:.3e}"
    )


def test_pigeonhole_counts_reproduce_explicit_resampling_exactly() -> None:
    """The counts shortcut must be algebra, not approximation."""

    rng = np.random.default_rng(0)
    seeds, strata, examples = 4, 3, 5
    mask = np.ones((strata, examples), dtype=bool)
    weights = _weights(mask, seeds)
    board = rng.choice([-1.0, 0.0, 1.0], size=(seeds, strata, examples))
    sizes = mask.sum(axis=1)

    generator = np.random.default_rng(7)
    seed_counts, example_counts = _pigeonhole_counts(
        seeds, sizes, examples, samples=8, generator=generator
    )
    per_stratum_weight = weights.max(axis=1)
    resampled_q = (seed_counts @ board.reshape(seeds, -1)).reshape(8, strata, examples)
    shortcut = np.einsum(
        "bke,bke->b", per_stratum_weight[None, :, None] * example_counts, resampled_q
    )

    for draw in range(8):
        seed_slots = np.repeat(np.arange(seeds), seed_counts[draw].astype(int))
        example_slots = [
            np.repeat(np.arange(examples), example_counts[draw, k].astype(int))
            for k in range(strata)
        ]
        explicit_panel = np.stack(
            [
                np.stack([board[s, k][example_slots[k]] for k in range(strata)])
                for s in seed_slots
            ]
        )
        explicit = float(point_estimate(explicit_panel, weights))
        assert shortcut[draw] == pytest.approx(explicit, abs=1e-12)


# --------------------------------------------------------------------------
# Simulation generative model
# --------------------------------------------------------------------------


def test_simulated_effect_is_calibrated_to_the_accuracy_scale() -> None:
    """`true_effect` must BE the expected paired difference, not a proxy.

    A miscalibrated simulator silently reports power for the wrong effect size.
    """

    layout = StratumLayout.balanced(8, 60)
    generator = np.random.default_rng(2)
    for effect in (0.0, 0.02, 0.04):
        design = DesignPoint(seed_count=10, true_effect=effect, seed_sd=0.0, example_sd=0.0)
        panel = simulate_panels(design, layout, trials=400, generator=generator)
        weights = _weights(layout.mask(), design.seed_count)
        observed = float(point_estimate(panel, weights).mean())
        assert observed == pytest.approx(effect, abs=0.004), (
            f"simulator is miscalibrated at effect={effect}: observed {observed:.4f}"
        )


def _mean_cross_seed_correlation(panel: np.ndarray) -> float:
    flat = panel.reshape(panel.shape[0], -1)
    pairs = [
        float(np.corrcoef(flat[i], flat[j])[0, 1])
        for i in range(panel.shape[0])
        for j in range(i + 1, panel.shape[0])
    ]
    return float(np.mean(pairs))


def test_example_effects_are_shared_across_seeds() -> None:
    """The crossed structure must actually be present in the simulated data.

    With no seed effect, the only thing that can correlate two seeds' paired
    differences is the shared per-example effect. Theory gives
    ``corr = sigma_b^2 / Var(d)``, so the test compares the shared world against
    an otherwise identical unshared world rather than asserting a bare
    threshold: a single pairwise correlation is far too noisy to test directly.
    """

    layout = StratumLayout.balanced(4, 500)
    common = dict(seed_count=6, true_effect=0.0, seed_sd=0.0, disagreement_rate=1.0)
    generator = np.random.default_rng(4)

    shared = simulate_panels(
        DesignPoint(**common, example_sd=0.4), layout, trials=6, generator=generator
    )
    unshared = simulate_panels(
        DesignPoint(**common, example_sd=0.0), layout, trials=6, generator=generator
    )
    shared_corr = float(np.mean([_mean_cross_seed_correlation(p) for p in shared]))
    unshared_corr = float(np.mean([_mean_cross_seed_correlation(p) for p in unshared]))

    # sigma_b^2 / Var(d) with Var(d) = rate = 1.0 gives an expected 0.16.
    assert shared_corr == pytest.approx(0.16, abs=0.04), (
        f"shared example effect has the wrong magnitude: {shared_corr:.4f}"
    )
    assert abs(unshared_corr) < 0.02, (
        f"seeds correlate without a shared example effect: {unshared_corr:.4f}"
    )
    assert shared_corr > unshared_corr + 0.10


def test_stratum_random_effect_knob_is_off_by_default() -> None:
    assert DesignPoint(seed_count=5, true_effect=0.0, seed_sd=0.02).stratum_sd == 0.0


# --------------------------------------------------------------------------
# Calibration of the registered estimator
# --------------------------------------------------------------------------


def test_registered_estimator_has_acceptable_null_calibration() -> None:
    """Type-I control at the smallest candidate seed count.

    Kept small enough for CI; the authoritative numbers come from the frozen
    high-precision audit.
    """

    layout = StratumLayout.balanced(10, 50)
    design = DesignPoint(seed_count=5, true_effect=0.0, seed_sd=0.02)
    result = evaluate_design(
        design,
        layout,
        estimator=PRIMARY_ESTIMATOR,
        confidence=0.95,
        practical_threshold=0.02,
        trials=1500,
        rng_seed=17,
    )
    # Wilson upper bound keeps this from flaking on Monte Carlo noise.
    assert result["gate_conjunction"]["ci_high"] <= 0.075
    assert result["interval_coverage_of_truth"]["rate"] >= 0.90


def test_power_increases_with_seed_count() -> None:
    layout = StratumLayout.balanced(10, 50)
    powers = []
    for seed_count in (5, 15):
        result = evaluate_design(
            DesignPoint(seed_count=seed_count, true_effect=0.03, seed_sd=0.02),
            layout,
            estimator=PRIMARY_ESTIMATOR,
            confidence=0.95,
            practical_threshold=0.02,
            trials=800,
            rng_seed=23,
        )
        powers.append(result["gate_conjunction"]["rate"])
    assert powers[1] > powers[0], f"power did not increase with replication: {powers}"


def test_student_t_quantile_falls_back_accurately() -> None:
    from statistics import NormalDist

    # The SciPy-free fallback must stay close enough to matter at small df.
    for df, expected in ((4, 2.776), (9, 2.262), (19, 2.093)):
        assert _student_t_quantile(0.975, df) == pytest.approx(expected, abs=0.02)
    assert _student_t_quantile(0.975, 1e9) == pytest.approx(NormalDist().inv_cdf(0.975), abs=1e-3)
