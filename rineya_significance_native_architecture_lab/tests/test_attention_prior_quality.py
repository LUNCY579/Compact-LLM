from __future__ import annotations

import pytest

from rineya_significance_lab.evaluation.attention_prior_quality import (
    ATTENTION_PRIOR_QUALITY_CONDITION_IDS,
    expected_cell_order,
    recover_prediction_prefix,
    retained_set_diagnostics,
    specificity_decision,
    validate_attention_prior_conditions,
)


def _conditions() -> list[dict[str, object]]:
    return [
        {"id": "C0_B3_NO_PRIOR", "source": "normalized", "lambda": 0.0},
        {"id": "C1_NORMALIZED_SC", "source": "normalized", "lambda": 0.01},
        {"id": "C5_RAW_S", "source": "raw", "lambda": 0.01},
        {"id": "C4_SHUFFLED_SC", "source": "shuffled", "lambda": 0.01},
        {"id": "C6_AGE_ONLY", "source": "age", "lambda": 0.01},
    ]


def test_five_conditions_use_only_the_frozen_lambda() -> None:
    parsed = validate_attention_prior_conditions(_conditions(), frozen_lambda=0.01)
    assert tuple(row.condition_id for row in parsed) == ATTENTION_PRIOR_QUALITY_CONDITION_IDS
    bad = _conditions()
    bad[2]["lambda"] = 0.02
    with pytest.raises(ValueError, match="frozen lambda"):
        validate_attention_prior_conditions(bad, frozen_lambda=0.01)


def test_restart_checkpoint_must_be_an_exact_unique_prefix() -> None:
    order = expected_cell_order(["a", "b"], ["x", "y"])
    rows = [
        {"case_id": "a", "condition": "x", "artifact": "one"},
        {"case_id": "a", "condition": "y", "artifact": "two"},
    ]
    assert len(recover_prediction_prefix(rows, order)) == 2
    with pytest.raises(RuntimeError, match="exact frozen-order prefix"):
        recover_prediction_prefix(list(reversed(rows)), order)
    with pytest.raises(RuntimeError, match="repeats cell"):
        recover_prediction_prefix([rows[0], rows[0]], order)


def test_retained_set_diagnostics_measure_diversity_churn_and_position_capture() -> None:
    first = [
        [[0, 1, 2], [0, 2, 3]],
        [[4, 5, 6], [4, 6, 7]],
    ]
    second = [
        [[0, 2, 4], [0, 3, 4]],
        [[4, 6, 8], [4, 7, 8]],
    ]
    initial = retained_set_diagnostics(first, input_tokens=9)
    changed = retained_set_diagnostics(second, input_tokens=9, previous_positions=first)
    assert initial["mean_boundary_churn_jaccard_distance"] is None
    assert 0 < float(changed["mean_boundary_churn_jaccard_distance"]) <= 1
    assert 0 < float(changed["mean_cross_head_jaccard_distance"]) <= 1
    total = sum(
        float(changed[name])
        for name in ("early_position_fraction", "middle_position_fraction", "late_position_fraction")
    )
    assert total == pytest.approx(1.0)


def test_specificity_decision_uses_all_three_controls() -> None:
    rows = {
        control: {
            "mean_difference": mean,
            "confidence_interval": interval,
        }
        for control, mean, interval in (
            ("C5_RAW_S", 0.1, (0.02, 0.2)),
            ("C4_SHUFFLED_SC", 0.08, (0.01, 0.16)),
            ("C6_AGE_ONLY", 0.05, (0.001, 0.12)),
        )
    }
    assert specificity_decision(rows)["decision"] == "SUPPORTED_EXPLORATORY_SIGNAL"
    rows["C6_AGE_ONLY"] = {"mean_difference": -0.01, "confidence_interval": (-0.1, 0.1)}
    assert specificity_decision(rows)["decision"] == "NOT_SUPPORTED"

