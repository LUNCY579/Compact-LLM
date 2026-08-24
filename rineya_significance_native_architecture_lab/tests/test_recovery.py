from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rineya_significance_lab.evaluation.recovery import (
    RecoveryCaseOutcome,
    aggregate_recovery_clusters,
    assert_pre_answer_feature_rows,
    descriptive_error_roc_auc,
    extract_cornerstone_risk_features,
    risk_coverage_curve,
)
from rineya_significance_lab.recovery import (
    RecoveryAction,
    RecoverySignal,
    ThresholdRecoveryController,
    execute_recovery_decision,
)


def _arrays() -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    rows = {
        0: {
            "entry_ids": [0, 1, 2, 3],
            "scores": [1.0, 2.0, 3.0, 4.0],
            "true_counts": [1.0, 1.0, 1.0, 1.0],
            "ranking": [1.0, 2.0, 3.0, 4.0],
            "selected": [1, 0, 1, 0],
        },
        1: {
            "entry_ids": [0, 2, 4, 5],
            "scores": [1.5, 4.0, 2.0, 1.0],
            "true_counts": [2.0, 2.0, 1.0, 1.0],
            "ranking": [0.75, 2.0, 2.0, 1.0],
            "selected": [0, 1, 1, 0],
        },
    }
    for boundary, fields in rows.items():
        for suffix, row in fields.items():
            dtype = np.int64 if suffix == "entry_ids" else np.float64
            if suffix == "selected":
                dtype = np.uint8
            values[f"b{boundary:03d}_l000_{suffix}"] = np.asarray(
                row, dtype=dtype
            ).reshape(1, 1, -1)
    return values


def _features():
    return extract_cornerstone_risk_features(
        _arrays(),
        case_id="case-a",
        cluster_id="cluster-a",
        task_family="R1",
        boundary_index=1,
        layer_count=1,
        top_mass_fraction=0.5,
        weights={
            "entropy": 1.0,
            "concentration": 1.0,
            "margin": 1.0,
            "churn": 1.0,
            "count": 1.0,
            "query_delta": 1.0,
        },
    )


def test_cornerstone_risk_features_reconstruct_query_delta_and_churn() -> None:
    features = _features()
    # Delta is [0.5, 1.0, 2.0, 1.0] after aligning prior survivors by entry_id.
    assert features.query_delta_fraction == pytest.approx(4.5 / 8.5)
    # Previous retained {0, 2}, current retained {2, 4}.
    assert features.retained_set_churn == pytest.approx(2.0 / 3.0)
    assert 0.0 <= features.risk_score <= 1.0
    assert set(features.risk_components) == {
        "entropy",
        "concentration",
        "margin",
        "churn",
        "count",
        "query_delta",
    }
    row = features.to_dict()
    assert_pre_answer_feature_rows([row])
    assert "correct" not in row
    assert "prediction" not in row


def test_query_delta_alignment_rejects_evicted_entry_reappearance() -> None:
    arrays = _arrays()
    arrays["b001_l000_entry_ids"] = np.asarray([1, 2, 4, 5], dtype=np.int64).reshape(
        1, 1, -1
    )
    with pytest.raises(RuntimeError, match="reappeared"):
        extract_cornerstone_risk_features(
            arrays,
            case_id="case-a",
            cluster_id="cluster-a",
            task_family="R1",
            boundary_index=1,
            layer_count=1,
            top_mass_fraction=0.5,
            weights=dict.fromkeys(_features().risk_components, 1.0),
        )


def test_pre_answer_feature_view_rejects_outcome_leakage() -> None:
    row = _features().to_dict()
    row["correct"] = True
    with pytest.raises(RuntimeError, match="forbidden"):
        assert_pre_answer_feature_rows([row])


def test_threshold_controller_accept_is_literal_identity_and_never_recomputes() -> None:
    signal = RecoverySignal("case", "cluster", 0.25, {"entropy": 0.25})
    decision = ThresholdRecoveryController(threshold=0.5).decide(signal)
    accepted = object()
    calls = 0

    def recompute(_: RecoveryAction) -> object:
        nonlocal calls
        calls += 1
        return object()

    execution = execute_recovery_decision(decision, accepted, recompute=recompute)
    assert execution.output is accepted
    assert execution.recomputation_calls == 0
    assert calls == 0


def test_threshold_controller_executes_one_full_cache_recomputation() -> None:
    signal = RecoverySignal("case", "cluster", 0.75, {"entropy": 0.75})
    decision = ThresholdRecoveryController(threshold=0.5).decide(signal)
    recovered = object()
    execution = execute_recovery_decision(
        decision,
        object(),
        recompute=lambda action: recovered
        if action is RecoveryAction.RECOMPUTE_FULL_CACHE
        else object(),
    )
    assert execution.output is recovered
    assert execution.recomputation_calls == 1


def _outcome(
    case: str,
    cluster: str,
    risk: float,
    compressed_correct: bool,
    *,
    family: str = "R1",
) -> RecoveryCaseOutcome:
    return RecoveryCaseOutcome(
        case_id=case,
        cluster_id=cluster,
        task_family=family,
        risk_score=risk,
        compressed_correct=compressed_correct,
        full_cache_correct=True,
        compressed_wall_seconds=2.0,
        fallback_wall_seconds=3.0,
        compressed_persistent_bytes=100,
        fallback_persistent_bytes=200,
        recompute_input_bytes=32,
    )


def test_whole_cluster_gate_never_selects_individual_case() -> None:
    clusters = aggregate_recovery_clusters(
        [
            _outcome("a-1", "a", 0.2, True),
            _outcome("a-2", "a", 0.8, False),
            _outcome("b-1", "b", 0.4, True),
        ]
    )
    a = next(row for row in clusters if row.cluster_id == "a")
    assert a.case_ids == ("a-1", "a-2")
    assert a.risk_score == 0.8
    assert not a.compressed_correct

    curve = risk_coverage_curve(clusters)
    point = next(row for row in curve if row.threshold == 0.8)
    assert point.recovered_clusters == 1
    assert point.rescued_errors == 1
    assert point.system_accuracy == 1.0
    assert point.wall_multiplier_vs_compressed == pytest.approx(2.0)
    assert point.recompute_input_bytes == 64
    assert point.incremental_peak_persistent_bytes == 100


def test_risk_curve_is_tie_safe_and_descriptive_auc_uses_cluster_units() -> None:
    rows = [
        _outcome("a", "a", 0.8, False),
        _outcome("b", "b", 0.8, True),
        _outcome("c", "c", 0.2, True),
    ]
    clusters = aggregate_recovery_clusters(rows)
    curve = risk_coverage_curve(clusters)
    at_tie = next(point for point in curve if point.threshold == 0.8)
    assert at_tie.recovered_clusters == 2
    assert at_tie.accepted_clusters == 1
    assert descriptive_error_roc_auc(clusters) == pytest.approx(0.75)

    with pytest.raises(ValueError, match="multiple task families"):
        aggregate_recovery_clusters(
            [rows[0], replace(rows[1], cluster_id="a", task_family="R3")]
        )
