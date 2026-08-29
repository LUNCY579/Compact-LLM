"""Tests for the secondary difficulty-response analysis.

The scientific risk here is contamination: a difficulty analysis that chooses its
bins after seeing outcomes, or that quietly acquires gating power, would let a
failed primary endpoint be rescued by a post-hoc slice. These tests pin the
structural guarantees that make that impossible.
"""

from __future__ import annotations

import pytest

from tgsa.phase1_response import (
    CONTINUOUS_BIN_COUNT,
    CONTINUOUS_VARIABLES,
    REGISTERED_DIFFICULTY_VARIABLES,
    _bin_panel,
    _contrast_estimate,
    _orthogonal_linear_contrast,
    difficulty_response,
    registered_bins,
)


def _rows(seed: int, correct_by_id: dict[str, bool], family: str = "pointer_chase") -> list[dict]:
    return [
        {
            "example_id": example_id,
            "seed": seed,
            "split": "test",
            "family": family,
            "stratum": "s0",
            "answer_token": 1,
            "correct": correct,
        }
        for example_id, correct in correct_by_id.items()
    ]


def test_every_registered_variable_is_generator_recorded() -> None:
    """No variable may be introduced that the generator does not already emit."""

    from tgsa.phase1_data import SyntheticVocabulary, generate_split

    vocab = SyntheticVocabulary(domain_size=16, max_hops=6, max_abs_step=4)
    spec = {
        "families": {
            "pointer_chase": {"min_nodes": 6, "max_nodes": 10, "min_hops": 2, "max_hops": 4},
            "state_update": {"min_steps": 3, "max_steps": 5},
        }
    }
    examples = generate_split(
        split="test",
        count_by_family={"pointer_chase": 12, "state_update": 12},
        dataset_seed=5,
        vocab=vocab,
        generation_spec=spec,
    )
    available = {family: set() for family in ("pointer_chase", "state_update")}
    for example in examples:
        available[example.family].update(example.metadata)
    for variable, families, _description in REGISTERED_DIFFICULTY_VARIABLES:
        for family in families:
            assert variable in available[family], (
                f"registered difficulty variable {variable!r} is not recorded for {family}"
            )


def test_discrete_bins_are_the_registered_values_themselves() -> None:
    metadata = {f"e{i}": {"effective_hops": 2 + (i % 3)} for i in range(30)}
    bins = registered_bins(metadata, variable="effective_hops")
    assert set(bins.values()) == {
        "effective_hops=2",
        "effective_hops=3",
        "effective_hops=4",
    }


def test_continuous_bins_come_from_the_dataset_not_from_outcomes() -> None:
    """Quantile cuts must depend only on the materialized metadata.

    Two callers with identical metadata but wildly different notional outcomes
    must receive byte-identical bin assignments, because outcomes are never an
    input to binning.
    """

    metadata = {f"e{i}": {"evidence_position": i} for i in range(40)}
    first = registered_bins(metadata, variable="evidence_position")
    second = registered_bins(dict(metadata), variable="evidence_position")
    assert first == second
    assert len(set(first.values())) == CONTINUOUS_BIN_COUNT
    assert "evidence_position" in CONTINUOUS_VARIABLES


def test_linear_contrast_is_centred_and_unit_norm() -> None:
    for count in (2, 3, 4, 7):
        weights = _orthogonal_linear_contrast(count)
        assert sum(weights) == pytest.approx(0.0, abs=1e-12)
        assert sum(value * value for value in weights) == pytest.approx(1.0)
        assert weights == sorted(weights), "contrast must follow the difficulty ordering"


def test_trend_contrast_recovers_the_hand_computed_value() -> None:
    panel = {
        seed: {f"b{j}": {f"e{i}": float(j) for i in range(25)} for j in range(4)}
        for seed in range(5)
    }
    labels = [f"b{j}" for j in range(4)]
    weights = _orthogonal_linear_contrast(4)
    result = _contrast_estimate(
        panel,
        labels=labels,
        weights=weights,
        estimator="crossed_cluster_t",
        confidence=0.95,
        bootstrap_samples=99,
        rng_seed=1,
    )
    expected = sum(weight * index for index, weight in enumerate(weights))
    assert result.observed == pytest.approx(expected, abs=1e-9)


def test_increasing_effect_with_difficulty_produces_a_positive_trend() -> None:
    """The signature the mechanism predicts must actually be detected."""

    panel = {
        seed: {
            f"hops={j}": {f"e{j}_{i}": (0.6 if i < 25 * j // 3 else 0.0) for i in range(25)}
            for j in range(1, 5)
        }
        for seed in range(6)
    }
    labels = sorted(next(iter(panel.values())))
    result = _contrast_estimate(
        panel,
        labels=labels,
        weights=_orthogonal_linear_contrast(len(labels)),
        estimator="crossed_cluster_t",
        confidence=0.95,
        bootstrap_samples=99,
        rng_seed=2,
    )
    assert result.observed > 0.0
    assert result.confidence_low > 0.0


def test_bins_missing_from_some_seed_disable_the_variable_rather_than_bias_it() -> None:
    candidate = {0: _rows(0, {"a": True, "b": True}), 1: _rows(1, {"a": True, "b": True})}
    baseline = {0: _rows(0, {"a": False, "b": False}), 1: _rows(1, {"a": False, "b": False})}
    bin_of = {"a": "x", "b": "y"}
    panel = _bin_panel(
        candidate, baseline, bin_of=bin_of, families=frozenset({"pointer_chase"})
    )
    assert set(next(iter(panel.values()))) == {"x", "y"}

    # Drop a bin from one seed only: the panel must refuse rather than silently
    # compare different example sets across seeds.
    lopsided_bin_of = {"a": "x", "b": "y"}
    lopsided_candidate = {0: _rows(0, {"a": True, "b": True}), 1: _rows(1, {"a": True})}
    lopsided_baseline = {0: _rows(0, {"a": False, "b": False}), 1: _rows(1, {"a": False})}
    assert (
        _bin_panel(
            lopsided_candidate,
            lopsided_baseline,
            bin_of=lopsided_bin_of,
            families=frozenset({"pointer_chase"}),
        )
        == {}
    )


def test_difficulty_response_is_marked_non_gating() -> None:
    metadata = {f"e{i}": {"effective_hops": 2 + (i % 3)} for i in range(60)}
    families = {example_id: "pointer_chase" for example_id in metadata}
    correct = {example_id: (index % 2 == 0) for index, example_id in enumerate(metadata)}
    candidate = {seed: _rows(seed, correct) for seed in range(4)}
    baseline = {
        seed: _rows(seed, {key: False for key in correct}) for seed in range(4)
    }
    report = difficulty_response(
        candidate,
        baseline,
        examples_metadata=metadata,
        example_families=families,
        estimator="crossed_cluster_t",
        confidence=0.95,
        bootstrap_samples=99,
        rng_seed=7,
    )
    assert report["gating"] is False
    assert "cannot rescue a failed primary" in str(report["claim_scope"])
    variables = report["variables"]
    assert "effective_hops" in variables
    assert variables["effective_hops"]["status"] == "estimated"
    assert variables["effective_hops"]["linear_trend"] is not None


def test_no_advancement_gate_reads_the_secondary_analysis() -> None:
    """Structural guarantee: the gate names must not reference the secondary key."""

    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "tgsa" / "phase1_analysis.py").read_text(
        encoding="utf-8"
    )
    gate_block = source[source.index('gates: dict[str, dict[str, object]] = {') :]
    gate_block = gate_block[: gate_block.index("all_pass = ")]
    assert "secondary_difficulty_response" not in gate_block
    assert "difficulty_response" not in gate_block
