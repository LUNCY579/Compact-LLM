from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from rineya_tgsa.trial_queue import validate_trial, validate_trial_queue

LAB = Path(__file__).resolve().parents[1]


def _trial() -> dict:
    return {
        "id": "trial-a",
        "mechanism": "Uniform soup",
        "mechanism_class": "CHECKPOINT",
        "eligibility_stage": "RESEARCH_POSITIVE_TRIAL_ELIGIBLE",
        "source_era_phase": "Phase 4",
        "source_commit": "1" * 40,
        "source_checkpoint": "parent",
        "positive_result": ["evidence/result.json"],
        "control_baseline": "baseline",
        "preservation_result": "no regression",
        "artifact_type": "full checkpoint",
        "bakeability": "high",
        "current_parent_compatibility": "high",
        "expected_integration_cost": "SMALL",
        "known_negative_transfers": [],
        "trial_configuration": {"alpha": 0.5},
        "trial_rank": 1,
        "trial_status": "QUEUED",
        "transfer_result": "NOT_RUN",
        "reason": "bounded trial",
    }


def test_queue_status_is_not_promotion_status() -> None:
    trial = _trial()
    assert validate_trial(trial) == []
    queue = {
        "schema_version": "rineya.production-trial-queue.v1",
        "selected_trial_id": "trial-a",
        "trials": [trial],
        "promotable": True,
    }
    assert any("promotable" in error for error in validate_trial_queue(queue))


def test_pass_requires_transfer_positive() -> None:
    trial = _trial()
    trial["trial_status"] = "PASS"
    assert any("TRANSFER_POSITIVE" in error for error in validate_trial(trial))
    trial["transfer_result"] = "TRANSFER_POSITIVE"
    assert validate_trial(trial) == []


def test_fail_requires_transfer_fail() -> None:
    trial = _trial()
    trial["trial_status"] = "FAIL"
    assert any("TRANSFER_FAIL" in error for error in validate_trial(trial))


def test_duplicate_trial_rank_is_rejected() -> None:
    queue = {
        "schema_version": "rineya.production-trial-queue.v1",
        "selected_trial_id": "trial-a",
        "trials": [_trial(), {**deepcopy(_trial()), "id": "trial-b"}],
    }
    assert any("duplicate trial_rank" in error for error in validate_trial_queue(queue))


def test_current_trial_queue_passes_schema_and_semantics() -> None:
    queue = json.loads(
        (LAB / "evidence" / "production_trial_queue.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (LAB / "schemas" / "production_trial_queue.schema.json").read_text(encoding="utf-8")
    )
    assert list(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(queue)
    ) == []
    assert validate_trial_queue(queue) == []
    assert queue["selected_trial_id"] == "phase4-uniform-delta-soup"
    assert "promotable" not in queue


def test_frozen_transfer_trial_has_one_candidate_and_strict_tie_failure() -> None:
    config = json.loads(
        (LAB / "configs" / "historical_soup_a11_baya_midpoint_trial.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["registration_status"] == "FROZEN_BEFORE_OUTCOME"
    assert config["recipe"]["candidate_count"] == 1
    assert config["recipe"]["ingredient_coefficient"] == 0.5
    assert config["transfer_gate"]["logical_operator"] == "AND"
    assert config["transfer_gate"]["tie_policy"] == "FAIL"
    amendment = config["compute_only_amendment"]
    assert amendment["invalidated_rows"] == 21
    assert config["evaluation"]["protocol"]["logits_to_keep"] == 1
    assert amendment["amended_runner_sha256"] == config["evaluation"]["runner"]["sha256"]
    assert amendment["attempt_1_status"].startswith("REJECTED_BEFORE_PREDICTION")
    assert config["evaluation"]["protocol"]["trust_remote_code"] is False
    assert config["evaluation"]["protocol"]["portable_logit_bias_token_id"] == 51378
