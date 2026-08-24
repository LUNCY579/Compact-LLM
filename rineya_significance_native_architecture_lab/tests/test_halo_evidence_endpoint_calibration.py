from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def test_endpoint_config_is_outcome_blind_and_future_frozen() -> None:
    lab = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((lab / "configs" / "halo_evidence_endpoint_v1.yaml").read_text())

    assert config["endpoint"]["version"] == "layer-fact-retention-v1"
    assert config["endpoint"]["primary"]["name"] == "layer_fact_exact_completion_rate"
    assert config["endpoint"]["primary"]["threshold"] is None
    forbidden = set(config["calibration"]["forbidden_for_endpoint_selection"])
    assert {"condition", "correct", "prediction", "accepted_answers"} <= forbidden
    assert config["freeze"]["scope"] == "future_evaluations_only"
    assert config["freeze"]["confirmatory_status"] == "NOT_REGISTERED"


def test_calibration_artifact_proves_primary_non_saturation_and_strict_saturation() -> None:
    lab = Path(__file__).resolve().parents[1]
    artifact = json.loads(
        (lab / "results" / "statistics" / "halo_evidence_endpoint_calibration_v1.json")
        .read_text()
    )
    source = lab.parent / artifact["source"]["path"]
    observed_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    assert artifact["status"] == "PASS"
    assert observed_hash == artifact["source"]["sha256"]
    assert artifact["source"]["evidence_record_count"] == 33
    assert artifact["selection_blinding_audit"]["answer_outcomes_used"] is False
    primary = artifact["candidate_summaries"]["layer_fact_exact_completion_rate"]
    strict = artifact["candidate_summaries"][
        "strict_all_required_fact_tokens_available_in_every_evaluated_layer"
    ]
    assert primary["distinct_value_count"] == 14
    assert primary["fraction_at_zero"] < 0.8
    assert strict["distinct_value_count"] == 1
    assert strict["fraction_at_zero"] == 1.0
    assert artifact["decision"]["strict_diagnostic_retained"] is True


def test_existing_pilot_application_is_labeled_posthoc_and_answer_free() -> None:
    lab = Path(__file__).resolve().parents[1]
    artifact = json.loads(
        (lab / "results" / "statistics" / "halo_retention_endpoint_analysis_v1.json")
        .read_text()
    )

    assert artifact["classification"] == "EXPLORATORY_POSTHOC_NO_ANSWER_OUTCOMES"
    assert artifact["analysis_access_audit"]["answer_outcomes_accessed"] is False
    assert artifact["source"]["cluster_count"] == 11
    assert artifact["source"]["row_count"] == 33
    assert set(artifact["by_condition"]) == {"B2", "B3", "B4"}
    comparison = artifact["clustered_paired_comparisons"][
        "B3_minus_B2__layer_fact_exact_completion_rate"
    ]
    assert comparison["cluster_count"] == 11
    assert comparison["eligible"] is True
