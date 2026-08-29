from __future__ import annotations

import pytest

from tgsa.phase1_analysis import (
    _accuracy_delta,
    _quantile,
    hierarchical_paired_bootstrap,
    paired_seed_stratum_values,
)


def _row(
    *,
    example_id: str,
    seed: int,
    family: str,
    stratum: str,
    correct: bool,
    answer_nll: float = 1.0,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "split": "test",
        "arm": "candidate",
        "seed": seed,
        "intervention": "identity",
        "family": family,
        "stratum": stratum,
        "answer_token": 7,
        "prediction_token": 7 if correct else 8,
        "correct": correct,
        "answer_nll": answer_nll,
        "target_probability": 0.5,
        "target_margin": 0.0,
    }


def test_quantile_uses_linear_interpolation() -> None:
    values = [0.0, 10.0]
    assert _quantile(values, 0.0) == 0.0
    assert _quantile(values, 0.25) == 2.5
    assert _quantile(values, 0.5) == 5.0
    assert _quantile(values, 1.0) == 10.0


def test_hierarchical_bootstrap_macro_averages_strata_not_examples() -> None:
    values = {
        1: {"rare": [1.0], "common": [0.0, 0.0, 0.0, 0.0]},
        2: {"rare": [1.0], "common": [0.0, 0.0, 0.0, 0.0]},
    }
    estimate = hierarchical_paired_bootstrap(
        values,
        samples=500,
        confidence=0.95,
        seed=42,
    )

    assert estimate.observed == pytest.approx(0.5)
    assert estimate.confidence_low == pytest.approx(0.5)
    assert estimate.confidence_high == pytest.approx(0.5)
    assert estimate.probability_positive == 1.0
    assert estimate.seed_count == 2
    assert estimate.stratum_count == 2
    assert estimate.example_seed_pairs == 10


def test_hierarchical_bootstrap_is_seed_deterministic() -> None:
    values = {
        10: {"a": [1.0, 0.0, 1.0], "b": [0.0, 1.0]},
        20: {"a": [0.0, 1.0, 1.0], "b": [1.0, 1.0]},
    }
    first = hierarchical_paired_bootstrap(values, samples=1000, confidence=0.95, seed=19)
    second = hierarchical_paired_bootstrap(values, samples=1000, confidence=0.95, seed=19)

    assert first == second


def test_paired_rows_preserve_seed_instance_and_stratum_pairing() -> None:
    candidate = {
        1: [
            _row(example_id="a", seed=1, family="pointer_chase", stratum="h4", correct=True),
            _row(example_id="b", seed=1, family="state_update", stratum="s6", correct=True),
        ],
        2: [
            _row(example_id="a", seed=2, family="pointer_chase", stratum="h4", correct=True),
            _row(example_id="b", seed=2, family="state_update", stratum="s6", correct=True),
        ],
    }
    baseline = {
        seed: [dict(row, arm="baseline", correct=False, prediction_token=8) for row in rows]
        for seed, rows in candidate.items()
    }

    values = paired_seed_stratum_values(
        candidate,
        baseline,
        score=_accuracy_delta,
        include_families={"pointer_chase", "state_update"},
    )

    assert values == {
        1: {"h4": [1.0], "s6": [1.0]},
        2: {"h4": [1.0], "s6": [1.0]},
    }


def test_paired_analysis_rejects_metadata_mismatch() -> None:
    candidate = {
        1: [_row(example_id="a", seed=1, family="pointer_chase", stratum="h4", correct=True)]
    }
    baseline = {
        1: [_row(example_id="a", seed=1, family="pointer_chase", stratum="h5", correct=False)]
    }

    with pytest.raises(ValueError, match="metadata mismatch"):
        paired_seed_stratum_values(candidate, baseline, score=_accuracy_delta)


def test_paired_analysis_rejects_unpaired_seed_sets() -> None:
    candidate = {
        1: [_row(example_id="a", seed=1, family="pointer_chase", stratum="h4", correct=True)]
    }
    baseline = {
        2: [_row(example_id="a", seed=2, family="pointer_chase", stratum="h4", correct=False)]
    }

    with pytest.raises(ValueError, match="seed sets differ"):
        paired_seed_stratum_values(candidate, baseline, score=_accuracy_delta)
