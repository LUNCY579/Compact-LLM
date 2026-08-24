from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import yaml

LAB_ROOT = Path(__file__).resolve().parents[1]
SCREEN = LAB_ROOT / "results" / "runs" / "halo_recovery_controller_screen_v1"
SMOKE = LAB_ROOT / "results" / "runs" / "halo_recovery_controller_smoke_v1"
CONFIG = LAB_ROOT / "configs" / "halo_recovery_controller.yaml"


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_recovery_config_and_resolved_snapshots_are_identical() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    screen_module = _module(
        "recovery_screen_test_module", LAB_ROOT / "scripts" / "run_halo_recovery_screen.py"
    )
    screen_module.validate_config(config)
    assert config["protocol"]["execution_mode"] == "closed_loop_model_execution"
    assert config["query"]["mode"] == "conditioned"
    assert config["significance"]["enabled"] is True
    for output in (SCREEN, SMOKE):
        resolved = yaml.safe_load((output / "resolved_config.yaml").read_text(encoding="utf-8"))
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert resolved == config
        assert manifest["resolved_config"] == config
        assert manifest["experiment"]["query_conditioned"] is True


def test_pre_answer_feature_artifact_has_no_oracle_or_outcome_fields() -> None:
    rows = _jsonl(SCREEN / "features.pre_answer.jsonl")
    assert len(rows) == 11
    forbidden = {
        "accepted_answers",
        "answer",
        "correct",
        "evidence",
        "full_cache_correct",
        "gold",
        "prediction",
        "target",
    }
    assert all(not forbidden.intersection(row) for row in rows)
    assert len({row["cluster_id"] for row in rows}) == 11
    assert all(row["feature_provenance"] == "PRE_ANSWER_ON_POLICY_CORNERSTONE_STATE" for row in rows)


def test_open_loop_curve_is_whole_cluster_tie_safe_and_reports_selection_bias() -> None:
    summary = json.loads((SCREEN / "summary.json").read_text(encoding="utf-8"))
    curve = _jsonl(SCREEN / "risk_coverage.jsonl")
    assert summary["whole_cluster_results"]["independent_clusters"] == 11
    assert summary["whole_cluster_results"]["descriptive_error_roc_auc"] == 0.5
    assert summary["controller"]["deployment_threshold_selected"] is False
    assert len(summary["selection_bias"]) == 7
    assert "not schedule-matched" in summary["cost_accounting"]["wall_time_interpretation"]
    assert curve[0]["recovered_clusters"] == 0
    assert curve[0]["coverage"] == 1.0
    assert curve[-1]["recovered_clusters"] == 11
    assert curve[-1]["coverage"] == 0.0
    assert all(
        int(row["accepted_clusters"]) + int(row["recovered_clusters"]) == 11
        for row in curve
    )


def test_live_smoke_proves_identity_and_fresh_position_safe_recovery() -> None:
    summary = json.loads((SMOKE / "summary.json").read_text(encoding="utf-8"))
    rows = {str(row["action"]): row for row in _jsonl(SMOKE / "paths.jsonl")}
    accept = rows["accept"]
    recovered = rows["recompute_full_cache"]
    assert summary["decision_before_answer_decode"] is True
    assert summary["answer_key_accessed"] is False
    assert summary["quality_scored"] is False
    assert accept["recomputation_calls"] == 0
    assert accept["literal_accept_payload_identity"] is True
    assert accept["matches_prior_same_path_prediction"] is True
    assert recovered["recomputation_calls"] == 1
    assert recovered["fresh_cache_zero_state_verified"] is True
    assert recovered["compressed_cache_reused"] is False
    assert recovered["source_context_fully_restored"] is True
    assert recovered["recompute_input_position_start"] == 0
    assert recovered["recompute_positions_contiguous"] is True
    assert recovered["matches_prior_same_path_prediction"] is True
    assert float(recovered["recompute_wall_seconds"]) > 0.0
    assert int(recovered["recompute_context_kv_bytes"]) > 0
    assert int(recovered["recompute_input_token_bytes"]) > 0
