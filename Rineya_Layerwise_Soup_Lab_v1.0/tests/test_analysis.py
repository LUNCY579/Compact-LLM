from rineya_soup_lab.analysis.bootstrap import paired_bootstrap_objective_difference
from rineya_soup_lab.analysis.layer_influence import smooth_layer_targets


OBJECTIVE = {
    "metrics": [{"path": "mean_quality_score", "weight": 1.0, "transform": "maximize_01"}],
    "case_utility": {"quality_weight": 1.0},
    "tail_fraction": 0.1,
}


def _rows(delta: float):
    return [
        {
            "id": f"x{i}",
            "quality_score": 0.5 + delta,
            "complete_utterance": True,
            "observable_quality_pass": True,
            "visible_generated_tokens": 3,
            "reply": f"r{i}",
            "quality_dimensions": {"task_adequacy": 0.5 + delta},
        }
        for i in range(20)
    ]


def test_paired_bootstrap_detects_improvement() -> None:
    result = paired_bootstrap_objective_difference(
        _rows(0.1), _rows(0.0), OBJECTIVE, samples=300, confidence_level=0.95, seed=3
    )
    assert result.observed_delta > 0
    assert result.probability_improvement == 1.0
    assert result.ci_low > 0


def test_bootstrap_resampling_does_not_create_fake_duplicate_replies() -> None:
    objective = {
        "metrics": [
            {"path": "duplicate_reply_rate", "weight": 1.0, "transform": "minimize_01"}
        ],
        "case_utility": {"quality_weight": 1.0},
        "tail_fraction": 0.1,
    }
    rows = _rows(0.0)
    result = paired_bootstrap_objective_difference(
        rows,
        rows,
        objective,
        samples=200,
        confidence_level=0.95,
        seed=11,
    )
    assert result.observed_delta == 0.0
    assert result.bootstrap_mean_delta == 0.0


def test_smoothing_shrinks_unreliable_layer() -> None:
    result = smooth_layer_targets(
        {0: 0.0, 1: 1.0, 2: 0.0},
        {0: 1.0, 1: 0.0, 2: 1.0},
        global_alpha=0.5,
        smoothing_penalty=0.2,
        global_shrinkage_penalty=0.2,
    )
    assert 0.2 < result[1] < 0.8
    assert 0.0 <= min(result.values()) <= max(result.values()) <= 1.0


def test_delta_influence_correlation_is_descriptive_and_ranked() -> None:
    from rineya_soup_lab.analysis.delta_correlation import (
        build_delta_influence_correlation_report,
        join_layer_delta_and_influence,
    )

    delta_rows = [
        {
            "layer_index": layer,
            "parameter_count": 10,
            "delta_l2_norm": float(layer + 1),
            "relative_delta_norm": float(layer + 1) / 10.0,
            "anchor_greedy_cosine": 1.0 - float(layer + 1) / 100.0,
        }
        for layer in range(5)
    ]
    influence_rows = [
        {
            "layer": layer,
            "combined_mean": float(layer + 1) / 20.0,
            "reliability": 1.0,
            "normalized_importance": 0.2,
        }
        for layer in range(5)
    ]
    joined = join_layer_delta_and_influence(delta_rows, influence_rows)
    report = build_delta_influence_correlation_report(
        joined,
        permutations=100,
        bootstrap_samples=100,
        confidence_level=0.95,
        seed=7,
    )
    estimate = report["correlations"]["relative_delta_norm"][
        "combined_behavioral_influence"
    ]["spearman"]
    assert abs(float(estimate["coefficient"]) - 1.0) < 1e-12
    assert report["matched_layer_count"] == 5
    assert "not used to select" in report["methodological_boundary"]
