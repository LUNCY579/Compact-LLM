from __future__ import annotations

import json
from pathlib import Path

from experiments.phase2_mechanism.audit_phase2_completion import audit_jsonl
from experiments.phase2_mechanism.analyze_cross_v212 import CONTROL_RUN, FULL_RUN
from experiments.phase2_mechanism.analyze_v22_calibrated_test import RUN as V22_CALIBRATED_RUN
from experiments.phase2_mechanism.analyze_selected_committee_dev import (
    RUN as SELECTED_COMMITTEE_RUN,
)
from experiments.phase2_mechanism.run_remaining_pipeline import live_lock_owner


def test_audit_jsonl_counts_unique_keys_methods_and_hashes(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [
        {"dataset": "d", "committee": "c", "method": "hard", "example_id": "1", "config_sha256": "abc"},
        {"dataset": "d", "committee": "c", "method": "soft", "example_id": "1", "config_sha256": "abc"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    audit = audit_jsonl(
        path,
        key_fields=("dataset", "committee", "method", "example_id"),
        expected_config_sha256="abc",
    )
    assert audit["rows"] == 2
    assert audit["unique_keys"] == 2
    assert audit["duplicate_keys"] == 0
    assert audit["methods"] == {"hard": 1, "soft": 1}
    assert audit["config_matches"] is True
    assert audit["malformed_rows"] == []


def test_audit_jsonl_reports_duplicate_and_malformed_rows(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    row = {"dataset": "d", "committee": "c", "method": "hard", "example_id": "1"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n{" + "\n", encoding="utf-8")
    audit = audit_jsonl(path, key_fields=("dataset", "committee", "method", "example_id"))
    assert audit["rows"] == 2
    assert audit["unique_keys"] == 1
    assert audit["duplicate_keys"] == 1
    assert len(audit["malformed_rows"]) == 1


def test_cross_analyzer_reads_generic_run_dev_output_names() -> None:
    assert CONTROL_RUN.name == "phase2_cross_v212_controls_dev_full.jsonl"
    assert FULL_RUN.name == "phase2_cross_v212_full_k7_dev_full.jsonl"


def test_v22_calibrated_analyzer_reads_generic_run_dev_output_name() -> None:
    assert V22_CALIBRATED_RUN.name == "phase2_v22_calibrated_test_confirmation_dev_full.jsonl"


def test_selected_committee_analyzer_reads_generic_run_dev_output_name() -> None:
    assert SELECTED_COMMITTEE_RUN.name == "phase2_selected_committee_dev_free_run_dev_full.jsonl"


def test_v22_calibrated_confirmation_was_frozen_before_output() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "artifacts/phase2_mechanism/v22_calibrated_test_confirmation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "V22_CALIBRATED_TEST_CONFIRMATION_FROZEN"
    assert manifest["calibrated_confirmation_outcomes_read_before_freeze"] is False
    assert manifest["selection_used_test_outcomes"] is False
    assert manifest["config"]["canonical_sha256"] == (
        "e4f77f7c7d68dc332eadc4decfb839d3a4b526addefd5c61538d92cf58df2794"
    )


def test_selected_committee_panel_is_frozen_and_dev_only() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "artifacts/phase2_mechanism/selected_committee_dev_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "DEV_SELECTED_COMMITTEE_PANEL_FROZEN"
    assert manifest["selection_used_test_outcomes"] is False
    assert manifest["committee_count"] == 12
    assert manifest["expected_rows"] == 48600
    assert manifest["config"]["canonical_sha256"] == (
        "8ba792929c3f5691670ed72ec43b81b695de631304d53d34a3d611507eb0b820"
    )


def test_pipeline_lock_reader_ignores_missing_and_dead_pid(tmp_path) -> None:
    path = tmp_path / "run.lock"
    assert live_lock_owner(path) is None
    path.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")
    assert live_lock_owner(path) is None
