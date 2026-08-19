from rineya_harness.evals.rsf import RSFEvaluator, RSFTrial


def test_rsf_metrics() -> None:
    trial = RSFTrial(
        case_id="route",
        before_active_ids={"goal", "A", "B", "C"},
        after_active_ids={"goal", "B", "C"},
        expected_retained_ids={"goal", "B", "C"},
        expected_evicted_ids={"A"},
        later_needed_ids={"goal", "B", "C"},
        pinned_ids={"goal"},
        before_estimated_tokens=100,
        after_estimated_tokens=60,
    )
    metrics = RSFEvaluator().evaluate([trial])
    assert metrics.task_success_rate == 1.0
    assert metrics.obsolete_eviction_precision == 1.0
    assert metrics.obsolete_eviction_recall == 1.0
    assert metrics.future_utility_recall == 1.0
    assert metrics.deletion_regret_rate == 0.0
    assert metrics.mean_token_reduction == 0.4
