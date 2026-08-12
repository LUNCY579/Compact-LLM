from __future__ import annotations

from consensus_lab.analysis.statistics import holm_adjust, paired_binary_statistics
from consensus_lab.evaluation.summary import _uncertainty_metrics


def test_paired_statistics_detects_known_direction_deterministically() -> None:
    result = paired_binary_statistics(
        [True, True, True, False],
        [False, False, True, False],
        seed=3,
        bootstrap_samples=1_000,
    )
    assert result.difference == 0.5
    assert result.mcnemar_b == 2
    assert result.mcnemar_c == 0
    assert result.mcnemar_exact_p == 0.5


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]


def test_missing_uncertainty_signal_is_not_coerced_to_zero() -> None:
    metrics = _uncertainty_metrics(
        [
            {"correct": True, "mean_agreement": 1.0, "mean_vote_entropy": 0.0, "mean_pairwise_js": None},
            {"correct": False, "mean_agreement": 1.0, "mean_vote_entropy": 0.0, "mean_pairwise_js": None},
        ]
    )
    assert metrics["pairwise_js_error_auroc"] is None
    assert metrics["disagreement_error_auroc"] == 0.5
