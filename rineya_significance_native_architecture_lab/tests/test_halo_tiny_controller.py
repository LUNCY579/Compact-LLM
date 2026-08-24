from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import jsonschema
import numpy as np
import pytest
import yaml

from rineya_significance_lab.evaluation.tiny_controller import (
    FEATURE_NAMES,
    assert_each_split_has_both_classes,
    assert_pre_answer_rows,
    calibrate_threshold,
    deterministic_cluster_split,
    fit_logistic_controller,
    fit_shallow_tree,
)
from rineya_significance_lab.integrity import artifact_hash_map, content_tree_sha256, hash_tree

LAB_ROOT = Path(__file__).resolve().parents[1]
CONFIG = LAB_ROOT / "configs" / "halo_tiny_significance_controller.yaml"
OUTPUT = LAB_ROOT / "results" / "experiment_f" / "halo_tiny_controller_v1"
SOURCE = LAB_ROOT / "results" / "runs" / "halo_recovery_controller_screen_v1"


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _runner_module() -> ModuleType:
    path = LAB_ROOT / "scripts" / "run_halo_tiny_controller.py"
    spec = importlib.util.spec_from_file_location("halo_tiny_controller_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cluster_hash_split_is_label_blind_exact_and_frozen() -> None:
    cluster_ids = [
        "R1-s000116",
        "R1-s000170",
        "R1-s000202",
        "R1-s000259",
        "R3-s000170",
        "R3-s000202",
        "R3-s000276",
        "R4-s000116",
        "R4-s000118",
        "R4-s000170",
        "R4-s000202",
    ]
    split = deterministic_cluster_split(
        cluster_ids,
        salt="rineya-halo-experiment-f-v1",
        train_count=5,
        calibration_count=3,
        evaluation_count=3,
    )
    assert {key for key, value in split.items() if value == "train"} == {
        "R3-s000202",
        "R4-s000170",
        "R1-s000259",
        "R3-s000170",
        "R1-s000116",
    }
    assert {key for key, value in split.items() if value == "calibration"} == {
        "R1-s000202",
        "R4-s000116",
        "R4-s000202",
    }
    assert {key for key, value in split.items() if value == "evaluation"} == {
        "R4-s000118",
        "R3-s000276",
        "R1-s000170",
    }
    assert split == deterministic_cluster_split(
        list(reversed(cluster_ids)),
        salt="rineya-halo-experiment-f-v1",
        train_count=5,
        calibration_count=3,
        evaluation_count=3,
    )


def test_split_support_failure_aborts_without_reassignment() -> None:
    split = {"a": "train", "b": "train", "c": "calibration", "d": "evaluation"}
    labels = {"a": False, "b": True, "c": False, "d": True}
    frozen = dict(split)
    with pytest.raises(RuntimeError, match="outcome-based reassignment is forbidden"):
        assert_each_split_has_both_classes(labels, split)
    assert split == frozen


def test_preanswer_contract_rejects_outcome_fields() -> None:
    row = {
        "case_id": "case",
        "cluster_id": "cluster",
        "feature_provenance": "PRE_ANSWER_ON_POLICY_CORNERSTONE_STATE",
        "risk_components": dict.fromkeys(FEATURE_NAMES, 0.5),
        "risk_score": 0.5,
        "compressed_correct": True,
    }
    with pytest.raises(RuntimeError, match="forbidden fields"):
        assert_pre_answer_rows([row])


def test_outcome_reader_cannot_open_labels_if_durable_split_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_module()
    feature = tmp_path / "features.pre_answer.jsonl"
    split = tmp_path / "split.pre_answer.jsonl"
    outcome = tmp_path / "outcomes.jsonl"
    feature.write_text("{}\n", encoding="utf-8")
    split.write_text("{}\n", encoding="utf-8")
    outcome.write_text('{"compressed_correct": true}\n', encoding="utf-8")
    freeze = {
        "stage": "PRE_LABEL_FEATURE_AND_SPLIT_FREEZE_COMPLETE",
        "durable_feature_snapshot_sha256": runner.sha256_file(feature),
        "durable_split_snapshot_sha256": runner.sha256_file(split),
    }
    split.write_text('{"tampered": true}\n', encoding="utf-8")
    opened: list[Path] = []

    def spy(path: Path) -> list[dict[str, object]]:
        opened.append(path)
        return []

    monkeypatch.setattr(runner, "_jsonl", spy)
    with pytest.raises(RuntimeError, match="split changed before outcome access"):
        runner._load_outcomes_after_freeze(
            outcome_source=outcome,
            output=tmp_path,
            freeze=freeze,
        )
    assert opened == []


def test_tiny_models_are_deterministic_and_parameter_bounded() -> None:
    features = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([False, False, True, True], dtype=np.bool_)
    first = fit_logistic_controller(
        features, labels, l2_penalty=1.0, max_iterations=100, tolerance=1e-10
    )
    second = fit_logistic_controller(
        features, labels, l2_penalty=1.0, max_iterations=100, tolerance=1e-10
    )
    assert first == second
    assert first.converged
    assert first.trainable_parameter_count == 7
    assert np.array_equal(first.score(features), second.score(features))

    tree = fit_shallow_tree(
        features, labels, max_depth=2, min_leaf_samples=1, laplace=1.0
    )
    assert tree.root.node_count <= 7
    assert tree.learned_scalar_count <= 7
    threshold = calibrate_threshold(labels, first.score(features))
    assert threshold.candidate_count >= 2
    assert threshold.tie_break == "fewer_recoveries_then_higher_threshold"


def test_artifact_freeze_precedes_label_access_and_split_has_support() -> None:
    feature_rows = _jsonl(OUTPUT / "features.pre_answer.jsonl")
    split_rows = _jsonl(OUTPUT / "split.pre_answer.jsonl")
    freeze = _json(OUTPUT / "prelabel_freeze.json")
    ledger = _json(OUTPUT / "data_access_ledger.json")
    assert_pre_answer_rows(feature_rows)
    assert freeze["stage"] == "PRE_LABEL_FEATURE_AND_SPLIT_FREEZE_COMPLETE"
    assert freeze["labels_opened"] is False
    events = ledger["ordered_events"]
    assert isinstance(events, list)
    assert [row["event"] for row in events] == [
        "PRE_ANSWER_FEATURE_SNAPSHOT_DURABLE",
        "CLUSTER_HASH_SPLIT_DURABLE",
        "ANSWER_DERIVED_OUTCOMES_OPENED",
    ]
    assert ledger["outcome_based_reassignment_attempted"] is False
    assert ledger["oracle_evidence_opened"] is False
    assert ledger["source_artifacts_modified"] is False
    assert len(feature_rows) == len(split_rows) == 11
    assert {row["assignment_provenance"] for row in split_rows} == {
        "PRE_LABEL_SHA256_CLUSTER_RANK"
    }
    support = ledger["split_class_support_checked_after_freeze"]
    assert all(
        split_support["compressed_success"] > 0 and split_support["compressed_error"] > 0
        for split_support in support.values()
    )


def test_artifact_models_metrics_and_underpowered_classification() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    resolved = yaml.safe_load((OUTPUT / "resolved_config.yaml").read_text(encoding="utf-8"))
    models = _json(OUTPUT / "models.json")
    calibration = _json(OUTPUT / "calibration.json")
    metrics = _json(OUTPUT / "metrics.json")
    summary = _json(OUTPUT / "summary.json")
    assert resolved == config
    assert models["backbone"] == "FROZEN_NOT_LOADED"
    assert models["reinforcement_learning"] is False
    logistic = models["logistic"]
    assert logistic["trainable_parameter_count"] == 7
    assert logistic["backbone_parameters_trainable"] == 0
    assert logistic["optimizer"] == "damped_Newton_IRLS"
    assert models["shallow_tree_control"]["max_depth"] == 2
    assert calibration["evaluation_labels_used"] is False
    held_out = metrics["held_out_evaluation"]
    assert set(held_out) == {
        "logistic",
        "shallow_tree",
        "deterministic_equal_weight",
        "constant_accept_all",
        "constant_recover_all",
        "calibration_budget_hash",
    }
    assert all(method["n"] == 3 for method in held_out.values())
    assert all(method["class_support"] == {"compressed_error": 2, "compressed_success": 1} for method in held_out.values())
    assert summary["classification"] == "UNDERPOWERED_OVERFIT_SCREEN_ONLY"
    assert summary["dataset"]["total_clusters"] == 11
    assert summary["dataset"]["evaluation_clusters"] == 3
    assert len(summary["inferential_limits"]) >= 5


def test_manifest_schema_and_content_integrity() -> None:
    manifest = _json(OUTPUT / "manifest.json")
    schema = _json(LAB_ROOT / "schemas" / "manifest.schema.json")
    jsonschema.Draft202012Validator(schema).validate(manifest)
    assert manifest["status"] == "COMPLETE"
    assert manifest["experiment"]["compute_tier"] == "C"
    assert manifest["learning"]["backbone_trainable_parameters"] == 0
    assert manifest["data_access"]["outcome_based_split_reassignment"] is False

    content = _json(OUTPUT / "content_manifest.json")
    repo_root = LAB_ROOT.parent
    recorded_code = content["code_files"]
    assert hash_tree(repo_root, recorded_code) == recorded_code
    assert content_tree_sha256(recorded_code) == content["code_tree_sha256"]
    recorded_artifacts = content["artifact_files"]
    current_artifacts = artifact_hash_map(OUTPUT)
    current_artifacts.pop("manifest.json", None)
    assert current_artifacts == recorded_artifacts
    assert content_tree_sha256(recorded_artifacts) == content["artifact_tree_sha256"]
    assert manifest["artifact_integrity"]["tree_sha256"] == content["artifact_tree_sha256"]
    assert (SOURCE / "features.pre_answer.jsonl").is_file()
