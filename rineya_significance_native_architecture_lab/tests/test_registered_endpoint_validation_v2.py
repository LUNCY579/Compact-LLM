from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

from rineya_significance_lab.manifest import sha256_file

LAB = Path(__file__).resolve().parents[1]


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_registration_precedes_and_exactly_matches_execution() -> None:
    registration = json.loads(
        (
            LAB
            / "results"
            / "registrations"
            / "halo_endpoint_validation_v2.json"
        ).read_text(encoding="utf-8")
    )
    execution = json.loads(
        (
            LAB
            / "results"
            / "registrations"
            / "halo_endpoint_validation_v2_execution.json"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            LAB
            / "results"
            / "runs"
            / "halo_endpoint_validation_v2_registered"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert execution["status"] == "PASS"
    assert all(execution["checks"].values())
    assert _time(registration["registered_at"]) < _time(manifest["created_at"])
    config_path = LAB / registration["registered_config"]
    assert registration["registered_config_sha256"] == sha256_file(config_path)


def test_registered_config_has_balanced_fixed_selection_and_primary() -> None:
    config = yaml.safe_load(
        (LAB / "configs" / "halo_endpoint_validation_v2_registered.yaml").read_text(
            encoding="utf-8"
        )
    )
    selection = config["selection"]
    assert selection["independent_clusters"] == 12
    assert set(selection["selected_case_ids_by_family"]) == {"R1", "R3", "R4"}
    assert all(len(values) == 4 for values in selection["selected_case_ids_by_family"].values())
    assert config["endpoint"]["version"] == "layer-fact-retention-v1"
    assert config["endpoint"]["primary_contrast"] == "B3_minus_B2"
    assert config["analysis"]["multiplicity"] == "one_declared_primary_endpoint_contrast"


def test_registered_primary_retention_positive_but_answer_not_improved() -> None:
    execution = json.loads(
        (
            LAB
            / "results"
            / "registrations"
            / "halo_endpoint_validation_v2_execution.json"
        ).read_text(encoding="utf-8")
    )
    primary = execution["primary_result"]
    answer = execution["secondary_answer_result"]
    assert primary["mean_difference"] > 0
    assert primary["confidence_interval"][0] > 0
    assert primary["p_value_two_sided"] == 0.01953125
    assert answer["mean_difference"] < 0
    assert answer["p_value_two_sided"] == 1.0
