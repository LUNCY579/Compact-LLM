from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from rineya_significance_lab.manifest import sha256_file

LAB_ROOT = Path(__file__).resolve().parents[1]
CONFIG = LAB_ROOT / "configs" / "halo_attention_prior_quality_screen_v1.yaml"


def _runner():
    path = LAB_ROOT / "scripts" / "run_halo_attention_prior_quality_screen.py"
    spec = importlib.util.spec_from_file_location("attention_prior_quality_runner_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quality_config_reuses_registered_ids_and_untuned_smoke_lambda() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    registered_path = LAB_ROOT / "configs" / "halo_endpoint_validation_v2_registered.yaml"
    registered = yaml.safe_load(registered_path.read_text(encoding="utf-8"))
    assert config["experiment"]["classification"] == "EXPLORATORY_NOT_REGISTERED"
    assert config["protocol"]["total_cells"] == 60
    assert config["selection"]["selected_case_ids"] == registered["selection"]["selected_case_ids"]
    assert config["frozen_case_source"]["registered_config_sha256"] == sha256_file(
        registered_path
    )
    smoke_config = LAB_ROOT / "configs" / "halo_attention_prior_smoke.yaml"
    assert config["frozen_lambda_source"]["config_sha256"] == sha256_file(smoke_config)
    assert config["frozen_lambda_source"]["lambda"] == 0.01
    assert config["frozen_lambda_source"]["tuning_on_quality_outcomes"] is False
    conditions = _runner().validate_quality_config(config)
    assert len(conditions) == 5
    assert conditions[0].lambda_value == 0.0
    assert {row.lambda_value for row in conditions[1:]} == {0.01}


def test_runner_does_not_access_evaluation_until_predictions_are_durable() -> None:
    source = (LAB_ROOT / "scripts" / "run_halo_attention_prior_quality_screen.py").read_text(
        encoding="utf-8"
    )
    durable = source.index('predictions_path = output / "predictions.unscored.jsonl"')
    access = source.index("# First evaluation-file access occurs")
    assert durable < access
    assert '"answer_key_accessed": False' in source[:access]


def test_synthetic_12_by_5_analysis_uses_whole_clusters_and_specificity_controls() -> None:
    runner = _runner()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    endpoint_by_condition = {
        "C0_B3_NO_PRIOR": 0.30,
        "C1_NORMALIZED_SC": 0.50,
        "C5_RAW_S": 0.35,
        "C4_SHUFFLED_SC": 0.32,
        "C6_AGE_ONLY": 0.31,
    }
    rows = []
    for case_id in config["selection"]["selected_case_ids"]:
        family = case_id.split("-", 1)[0]
        for condition, endpoint in endpoint_by_condition.items():
            rows.append(
                {
                    "case_id": case_id,
                    "condition": condition,
                    "task_family": family,
                    "correct": condition == "C1_NORMALIZED_SC",
                    "evidence": {
                        "layer_fact_exact_completion_rate": endpoint,
                        "all_required_survived": False,
                    },
                    "attention_feedback": {
                        "mean_attention_entropy_normalized": 0.7,
                        "mean_source_entropy_normalized": 0.6,
                        "maximum_abs_bias": 0.02,
                    },
                    "retained_set": {
                        "mean_mean_cross_head_jaccard_distance": 0.5,
                        "mean_boundary_churn_jaccard_distance": 0.4,
                        "mean_early_position_fraction": 0.3,
                        "mean_late_position_fraction": 0.3,
                    },
                }
            )
    analysis = runner.analyze_scored(rows, config)
    assert analysis["clusters"] == 12
    assert analysis["cells"] == 60
    assert analysis["specificity_decision"]["decision"] == "SUPPORTED_EXPLORATORY_SIGNAL"
    primary = analysis["whole_cluster_paired_comparisons"][
        "C1_NORMALIZED_SC_minus_C4_SHUFFLED_SC__primary_endpoint"
    ]
    assert primary["cluster_count"] == 12
    assert primary["row_count"] == 12


def test_runner_rejects_any_scope_drift() -> None:
    runner = _runner()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["conditions"][1]["lambda"] = 0.02
    with pytest.raises(ValueError, match="frozen lambda"):
        runner.validate_quality_config(config)


def test_matching_audit_rejects_query_or_storage_asymmetry() -> None:
    runner = _runner()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rows = []
    for case_id in config["selection"]["selected_case_ids"]:
        for condition in runner.ATTENTION_PRIOR_QUALITY_CONDITION_IDS:
            rows.append(
                {
                    "case_id": case_id,
                    "condition": condition,
                    "resident_after_prefill": 175,
                    "query_access": {
                        "query_token_sha256": f"query-{case_id}",
                        "query_tokens": 11,
                        "answer_access_during_inference": False,
                    },
                }
            )
    assert runner.matching_audit(rows, config)["status"] == "PASS"
    rows[1]["resident_after_prefill"] = 176
    audit = runner.matching_audit(rows, config)
    assert audit["status"] == "FAIL"
    assert audit["checks"]["same_resident_prefill_size_within_case"] is False
