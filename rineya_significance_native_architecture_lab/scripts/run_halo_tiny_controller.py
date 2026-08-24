#!/usr/bin/env python3
"""Run leakage-safe Architecture Experiment F on durable Halo features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import yaml

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.evaluation.tiny_controller import (  # noqa: E402
    FEATURE_NAMES,
    SPLIT_NAMES,
    LogisticController,
    ShallowTreeController,
    assert_each_split_has_both_classes,
    assert_pre_answer_rows,
    binary_metrics,
    calibrate_threshold,
    cluster_digest,
    deterministic_cluster_split,
    exact_budget_actions,
    feature_vector,
    fit_logistic_controller,
    fit_shallow_tree,
    hash_budget_scores,
    risk_coverage_curve,
)
from rineya_significance_lab.manifest import (  # noqa: E402
    atomic_write_json,
    build_manifest,
    sha256_file,
    utc_now,
)


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("tiny-controller config must be a mapping")
    return value


def _resolve_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    raw = dict(config["paths"])
    base = REPO_ROOT if raw.get("base") == "repository_root" else LAB_ROOT
    return {
        name: (base / str(value)).resolve()
        for name, value in raw.items()
        if name != "base"
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    _write_text(path, payload)


def validate_config(config: Mapping[str, Any]) -> None:
    protocol = dict(config.get("protocol", {}))
    if protocol.get("execution_mode") != "open_loop_trace_simulation":
        raise ValueError("Experiment F reuses frozen features and must be open-loop")
    if protocol.get("backbone") != "frozen_no_model_execution":
        raise ValueError("Experiment F must keep the Halo backbone frozen")
    if protocol.get("target") != "compressed_error":
        raise ValueError("the controller target must be compressed error")
    if protocol.get("answer_access") != (
        "only_after_feature_snapshot_and_cluster_split_are_durable"
    ):
        raise ValueError("pre-answer leakage contract is not frozen")
    if not bool(protocol.get("source_artifacts_read_only")):
        raise ValueError("source recovery artifacts must remain read-only")
    if bool(protocol.get("reinforcement_learning", True)):
        raise ValueError("reinforcement learning is excluded")

    model = dict(config.get("model", {}))
    core_id = str(model.get("core_id", ""))
    if ".halo." not in core_id.lower() or model.get("expected_source_base") != "Qwen3-0.6B":
        raise ValueError("Experiment F is scoped to published Halo Qwen3-0.6B")
    if tuple(model.get("allowed_core_ids", ())) != (core_id,):
        raise ValueError("allowed_core_ids must contain only active Halo")
    if int(model.get("frozen_backbone_trainable_parameters", -1)) != 0:
        raise ValueError("Halo backbone parameters must remain frozen")
    if bool(model.get("trust_remote_code", True)):
        raise ValueError("trust_remote_code must remain false")

    features = dict(config.get("features", {}))
    if tuple(features.get("order", ())) != FEATURE_NAMES:
        raise ValueError(f"feature order must be exactly {FEATURE_NAMES}")
    if int(features.get("dimension", -1)) != len(FEATURE_NAMES):
        raise ValueError("feature dimension does not match the six-feature contract")
    if bool(features.get("answer_access", True)) or bool(
        features.get("oracle_evidence_access", True)
    ):
        raise ValueError("pre-answer controller features cannot access answers or evidence")

    split = dict(config.get("split", {}))
    if split.get("algorithm") != "sha256_rank_exact_counts_v1":
        raise ValueError("split algorithm must be frozen SHA-256 rank")
    if not bool(split.get("immutable_single_attempt")):
        raise ValueError("split salt cannot be searched after outcomes")
    if split.get("failure_policy") != "abort_without_outcome_based_reassignment":
        raise ValueError("split failure must abort without outcome reassignment")
    counts = dict(split.get("counts", {}))
    if tuple(counts) != SPLIT_NAMES or any(int(counts[name]) < 2 for name in SPLIT_NAMES):
        raise ValueError("train/calibration/evaluation counts must each allow both classes")

    logistic = dict(config.get("logistic_controller", {}))
    if logistic.get("type") != "l2_logistic_regression":
        raise ValueError("the primary tiny controller must be logistic regression")
    if logistic.get("optimizer") != "damped_Newton_IRLS":
        raise ValueError("unexpected logistic optimizer")
    if int(logistic.get("expected_trainable_parameters", -1)) != len(FEATURE_NAMES) + 1:
        raise ValueError("logistic parameter budget must be six weights plus one intercept")

    tree = dict(config.get("tree_control", {}))
    if tree.get("type") != "deterministic_shallow_CART_control":
        raise ValueError("the control must be a deterministic shallow CART")
    if not 1 <= int(tree.get("maximum_depth", 0)) <= 2:
        raise ValueError("tree control must remain at depth one or two")

    calibration = dict(config.get("calibration", {}))
    if calibration.get("partition") != "calibration_only":
        raise ValueError("thresholds may use only the calibration split")
    if calibration.get("objective") != "maximum_balanced_accuracy":
        raise ValueError("unexpected calibration objective")


def _freeze_prelabel_view(
    *,
    feature_source: Path,
    output: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """Read no outcome path; atomically freeze feature and split artifacts."""

    features = _jsonl(feature_source)
    assert_pre_answer_rows(features)
    features.sort(key=lambda row: str(row["case_id"]))
    split_config = dict(config["split"])
    counts = {name: int(dict(split_config["counts"])[name]) for name in SPLIT_NAMES}
    split_map = deterministic_cluster_split(
        [str(row["cluster_id"]) for row in features],
        salt=str(split_config["salt"]),
        train_count=counts["train"],
        calibration_count=counts["calibration"],
        evaluation_count=counts["evaluation"],
    )
    ranked = sorted(
        (str(row["cluster_id"]) for row in features),
        key=lambda cluster_id: (
            cluster_digest(cluster_id, salt=str(split_config["salt"])),
            cluster_id,
        ),
    )
    by_cluster = {str(row["cluster_id"]): row for row in features}
    split_rows = [
        {
            "schema_version": "1.0",
            "assignment_provenance": "PRE_LABEL_SHA256_CLUSTER_RANK",
            "rank": rank,
            "case_id": str(by_cluster[cluster_id]["case_id"]),
            "cluster_id": cluster_id,
            "task_family": str(by_cluster[cluster_id]["task_family"]),
            "split": split_map[cluster_id],
            "sha256": cluster_digest(cluster_id, salt=str(split_config["salt"])),
        }
        for rank, cluster_id in enumerate(ranked)
    ]

    output.mkdir(parents=True, exist_ok=True)
    feature_snapshot = output / "features.pre_answer.jsonl"
    split_snapshot = output / "split.pre_answer.jsonl"
    resolved_config = output / "resolved_config.yaml"
    _write_jsonl(feature_snapshot, features)
    _write_jsonl(split_snapshot, split_rows)
    _write_text(
        resolved_config,
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
    )
    freeze = {
        "schema_version": "1.0",
        "stage": "PRE_LABEL_FEATURE_AND_SPLIT_FREEZE_COMPLETE",
        "labels_opened": False,
        "oracle_evidence_opened": False,
        "feature_source_sha256": sha256_file(feature_source),
        "durable_feature_snapshot_sha256": sha256_file(feature_snapshot),
        "durable_split_snapshot_sha256": sha256_file(split_snapshot),
        "split_algorithm": split_config["algorithm"],
        "split_salt": split_config["salt"],
        "split_counts": counts,
        "cluster_count": len(features),
        "no_outcome_retry_path": True,
    }
    atomic_write_json(output / "prelabel_freeze.json", freeze)
    return features, split_map, split_rows, freeze


def _load_outcomes_after_freeze(
    *,
    outcome_source: Path,
    output: Path,
    freeze: Mapping[str, Any],
) -> list[dict[str, Any]]:
    feature_snapshot = output / "features.pre_answer.jsonl"
    split_snapshot = output / "split.pre_answer.jsonl"
    if freeze.get("stage") != "PRE_LABEL_FEATURE_AND_SPLIT_FREEZE_COMPLETE":
        raise RuntimeError("pre-label freeze was not completed")
    if sha256_file(feature_snapshot) != freeze.get("durable_feature_snapshot_sha256"):
        raise RuntimeError("durable feature view changed before outcome access")
    if sha256_file(split_snapshot) != freeze.get("durable_split_snapshot_sha256"):
        raise RuntimeError("durable cluster split changed before outcome access")
    # This is the first call in the runner that receives or opens an answer-
    # derived path.  The source recovery directory itself remains untouched.
    return _jsonl(outcome_source)


def _join_after_freeze(
    features: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    split_map: Mapping[str, str],
) -> list[dict[str, Any]]:
    by_case = {str(row["case_id"]): row for row in outcomes}
    if len(by_case) != len(outcomes):
        raise RuntimeError("outcome case IDs must be unique")
    joined: list[dict[str, Any]] = []
    for feature in features:
        case_id = str(feature["case_id"])
        if case_id not in by_case:
            raise RuntimeError(f"missing outcome for {case_id}")
        outcome = by_case[case_id]
        cluster_id = str(feature["cluster_id"])
        if cluster_id != str(outcome["cluster_id"]):
            raise RuntimeError(f"{case_id}: feature/outcome cluster mismatch")
        if str(feature["task_family"]) != str(outcome["task_family"]):
            raise RuntimeError(f"{case_id}: feature/outcome family mismatch")
        if outcome.get("outcome_access_phase") != "AFTER_DURABLE_PRE_ANSWER_FEATURE_VIEW":
            raise RuntimeError(f"{case_id}: source outcome lacks its own access-phase proof")
        if not math.isclose(
            float(feature["risk_score"]), float(outcome["risk_score"]), abs_tol=1e-12
        ):
            raise RuntimeError(f"{case_id}: deterministic risk changed across source views")
        joined.append(
            {
                "schema_version": "1.0",
                "case_id": case_id,
                "cluster_id": cluster_id,
                "task_family": str(feature["task_family"]),
                "split": split_map[cluster_id],
                "features": list(feature_vector(feature)),
                "equal_weight_risk": float(feature["risk_score"]),
                "compressed_error": not bool(outcome["compressed_correct"]),
                "full_cache_correct": bool(outcome["full_cache_correct"]),
                "compressed_wall_seconds": float(outcome["compressed_wall_seconds"]),
                "fallback_wall_seconds": float(outcome["recorded_full_cache_fallback_wall_seconds"]),
                "compressed_persistent_bytes": int(outcome["compressed_persistent_bytes"]),
                "fallback_persistent_bytes": int(outcome["full_cache_fallback_kv_bytes"]),
                "recompute_input_bytes": int(outcome["recompute_input_token_bytes"]),
                "outcome_access_phase": "AFTER_DURABLE_EXPERIMENT_F_FEATURE_AND_SPLIT_FREEZE",
            }
        )
    if set(by_case) != {str(row["case_id"]) for row in features}:
        raise RuntimeError("source outcome set differs from durable feature case set")
    joined.sort(key=lambda row: (str(row["split"]), str(row["cluster_id"])))
    return joined


class ResourceProbe:
    def __init__(self) -> None:
        self._process = psutil.Process()
        info = self._process.memory_info()
        self.baseline_rss = int(info.rss)
        self.peak_sampled_rss = self.baseline_rss
        self.process_lifetime_peak_rss = int(getattr(info, "peak_wset", info.rss))

    def sample(self) -> None:
        info = self._process.memory_info()
        self.peak_sampled_rss = max(self.peak_sampled_rss, int(info.rss))
        self.process_lifetime_peak_rss = max(
            self.process_lifetime_peak_rss, int(getattr(info, "peak_wset", info.rss))
        )

    def result(self, *, wall_seconds: float, python_peak_bytes: int) -> dict[str, Any]:
        self.sample()
        return {
            "wall_seconds": wall_seconds,
            "baseline_process_rss_bytes": self.baseline_rss,
            "peak_process_rss_bytes_sampled": self.peak_sampled_rss,
            "incremental_process_rss_bytes_sampled": max(
                0, self.peak_sampled_rss - self.baseline_rss
            ),
            "process_lifetime_peak_rss_bytes": self.process_lifetime_peak_rss,
            "peak_python_tracemalloc_bytes": python_peak_bytes,
            "measurement_limit": (
                "RSS is sampled at optimizer iterations and boundaries; process-lifetime peak includes "
                "imports, while tracemalloc excludes native NumPy allocations."
            ),
        }


def _partition(rows: Sequence[Mapping[str, Any]], split: str) -> list[Mapping[str, Any]]:
    result = [row for row in rows if row["split"] == split]
    result.sort(key=lambda row: str(row["cluster_id"]))
    if not result:
        raise RuntimeError(f"partition {split} is empty")
    return result


def _arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    labels = np.asarray([bool(row["compressed_error"]) for row in rows], dtype=np.bool_)
    return features, labels


def _fit_controllers(
    config: Mapping[str, Any], train_rows: Sequence[Mapping[str, Any]]
) -> tuple[LogisticController, ShallowTreeController, dict[str, Any]]:
    features, labels = _arrays(train_rows)
    logistic_config = dict(config["logistic_controller"])
    logistic_probe = ResourceProbe()
    tracemalloc.start()
    start = time.perf_counter()
    logistic = fit_logistic_controller(
        features,
        labels,
        l2_penalty=float(logistic_config["l2_penalty"]),
        max_iterations=int(logistic_config["maximum_iterations"]),
        tolerance=float(logistic_config["convergence_tolerance"]),
        observer=logistic_probe.sample,
    )
    logistic_wall = time.perf_counter() - start
    _, logistic_python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if not logistic.converged:
        raise RuntimeError("deterministic logistic optimizer did not converge")
    if logistic.trainable_parameter_count != int(logistic_config["expected_trainable_parameters"]):
        raise RuntimeError("logistic controller exceeded its frozen parameter budget")

    tree_config = dict(config["tree_control"])
    tree_probe = ResourceProbe()
    tracemalloc.start()
    start = time.perf_counter()
    tree = fit_shallow_tree(
        features,
        labels,
        max_depth=int(tree_config["maximum_depth"]),
        min_leaf_samples=int(tree_config["minimum_leaf_samples"]),
        laplace=float(tree_config["leaf_laplace_smoothing"]),
        observer=tree_probe.sample,
    )
    tree_wall = time.perf_counter() - start
    _, tree_python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    dimension = len(FEATURE_NAMES) + 1
    resources = {
        "dataset": {
            "total_clusters": 11,
            "training_clusters": len(train_rows),
            "feature_count": len(FEATURE_NAMES),
        },
        "backbone": {
            "status": "FROZEN_NOT_LOADED",
            "trainable_parameters": 0,
            "model_execution": False,
        },
        "logistic": {
            **logistic_probe.result(
                wall_seconds=logistic_wall,
                python_peak_bytes=int(logistic_python_peak),
            ),
            "trainable_parameter_count": logistic.trainable_parameter_count,
            "optimizer": logistic.optimizer,
            "optimizer_state": {
                "persisted_after_fit_bytes": 0,
                "transient_state": "gradient + Hessian + Newton step",
                "estimated_transient_optimizer_bytes": int(
                    (dimension + dimension * dimension + dimension) * 8
                ),
                "estimate_excludes_design_matrix_and_predictions": True,
            },
        },
        "tree_control": {
            **tree_probe.result(
                wall_seconds=tree_wall,
                python_peak_bytes=int(tree_python_peak),
            ),
            "learned_scalar_count": tree.learned_scalar_count,
            "optimizer": "exhaustive_gini_split_search_with_deterministic_ties",
            "optimizer_state": {
                "persisted_after_fit_bytes": 0,
                "transient_state": "one best-split tuple per recursive node",
            },
        },
    }
    return logistic, tree, resources


def _cost_metrics(rows: Sequence[Mapping[str, Any]], actions: np.ndarray[Any, Any]) -> dict[str, Any]:
    if actions.shape != (len(rows),):
        raise ValueError("cost actions are not aligned to rows")
    compressed_wall = sum(float(row["compressed_wall_seconds"]) for row in rows)
    fallback_wall = sum(
        float(row["fallback_wall_seconds"])
        for row, recover in zip(rows, actions.tolist(), strict=True)
        if recover
    )
    baseline_peak = max(int(row["compressed_persistent_bytes"]) for row in rows)
    fallback_peak = max(
        (
            int(row["fallback_persistent_bytes"])
            for row, recover in zip(rows, actions.tolist(), strict=True)
            if recover
        ),
        default=0,
    )
    system_correct = [
        bool(row["full_cache_correct"]) if recover else not bool(row["compressed_error"])
        for row, recover in zip(rows, actions.tolist(), strict=True)
    ]
    return {
        "system_accuracy": sum(system_correct) / len(system_correct),
        "compressed_baseline_wall_seconds": compressed_wall,
        "fallback_wall_seconds": fallback_wall,
        "total_wall_seconds": compressed_wall + fallback_wall,
        "wall_multiplier_vs_compressed": (compressed_wall + fallback_wall) / compressed_wall,
        "recompute_input_bytes": sum(
            int(row["recompute_input_bytes"])
            for row, recover in zip(rows, actions.tolist(), strict=True)
            if recover
        ),
        "sequential_peak_persistent_bytes": max(baseline_peak, fallback_peak),
        "incremental_peak_persistent_bytes": max(0, fallback_peak - baseline_peak),
        "cost_limit": (
            "Reuses prior same-prompt measurements with mismatched source chunk schedules; "
            "not a contemporaneous latency benchmark."
        ),
    }


def _evaluate_method(
    rows: Sequence[Mapping[str, Any]],
    *,
    scores: np.ndarray[Any, Any],
    actions: np.ndarray[Any, Any],
    threshold: float | None,
    threshold_source: str,
) -> dict[str, Any]:
    labels = np.asarray([bool(row["compressed_error"]) for row in rows], dtype=np.bool_)
    result = binary_metrics(labels, scores, actions)
    result["threshold"] = threshold
    result["threshold_source"] = threshold_source
    result["cost_and_system"] = _cost_metrics(rows, actions)
    return result


def _write_curve_artifacts(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    score_map: Mapping[str, np.ndarray[Any, Any]],
) -> None:
    labels = np.asarray([bool(row["compressed_error"]) for row in rows], dtype=np.bool_)
    full = np.asarray([bool(row["full_cache_correct"]) for row in rows], dtype=np.bool_)
    curve_rows: list[dict[str, Any]] = []
    for method, scores in score_map.items():
        for point in risk_coverage_curve(labels, full, scores):
            curve_rows.append({"method": method, **point})
    _write_jsonl(output / "risk_coverage.evaluation.jsonl", curve_rows)
    with (output / "risk_coverage.evaluation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for method in score_map:
        points = [row for row in curve_rows if row["method"] == method]
        coverage = [float(row["coverage"]) for row in points]
        selective = [
            math.nan if row["selective_risk"] is None else float(row["selective_risk"])
            for row in points
        ]
        system = [float(row["system_accuracy"]) for row in points]
        axes[0].plot(coverage, selective, marker="o", label=method)
        axes[1].plot([1.0 - value for value in coverage], system, marker="o", label=method)
    axes[0].set(xlabel="Coverage (accepted fraction)", ylabel="Selective compressed-error risk")
    axes[1].set(xlabel="Recovery fraction", ylabel="Counterfactual system accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8)
    fig.suptitle("Experiment F held-out risk/coverage (n=3; descriptive only)")
    fig.savefig(output / "risk_coverage.evaluation.png", dpi=180)
    plt.close(fig)


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    validate_config(config)
    paths = _resolve_paths(config)
    output = paths["output"]
    started_at = utc_now()

    features, split_map, _split_rows, freeze = _freeze_prelabel_view(
        feature_source=paths["source_features"], output=output, config=config
    )
    manifest = build_manifest(
        repo=REPO_ROOT,
        config=config,
        run_id="halo_tiny_controller_v1",
        execution_mode="OPEN_LOOP_TRACE_SIMULATION",
        model={
            "status": "VERIFIED",
            "id": config["model"]["core_id"],
            "path": None,
            "architecture": config["model"]["expected_architecture"],
            "hashes": dict(config["model"]["expected_hashes"]),
        },
        benchmark={"name": "11 B0-selected durable Halo recovery clusters"},
        architecture_variant="F tiny learned significance recovery controller",
        seed=int(config["runtime"]["seed"]),
        compute_tier="C",
        status="RUNNING",
        resolved_config_path="configs/halo_tiny_significance_controller.yaml",
    )
    manifest["created_at"] = started_at
    manifest["data_access"] = {
        "phase": "PRE_LABEL_FREEZE_COMPLETE",
        "prelabel_freeze": "prelabel_freeze.json",
        "outcomes_opened": False,
    }
    atomic_write_json(output / "manifest.json", manifest)

    outcomes = _load_outcomes_after_freeze(
        outcome_source=paths["source_outcomes"], output=output, freeze=freeze
    )
    joined = _join_after_freeze(features, outcomes, split_map)
    labels_by_cluster = {
        str(row["cluster_id"]): bool(row["compressed_error"]) for row in joined
    }
    support = assert_each_split_has_both_classes(labels_by_cluster, split_map)
    _write_jsonl(output / "joined.post_answer.jsonl", joined)
    access_ledger = {
        "schema_version": "1.0",
        "ordered_events": [
            {
                "sequence": 1,
                "event": "PRE_ANSWER_FEATURE_SNAPSHOT_DURABLE",
                "sha256": freeze["durable_feature_snapshot_sha256"],
            },
            {
                "sequence": 2,
                "event": "CLUSTER_HASH_SPLIT_DURABLE",
                "sha256": freeze["durable_split_snapshot_sha256"],
            },
            {
                "sequence": 3,
                "event": "ANSWER_DERIVED_OUTCOMES_OPENED",
                "sha256": sha256_file(paths["source_outcomes"]),
            },
        ],
        "split_class_support_checked_after_freeze": support,
        "outcome_based_reassignment_attempted": False,
        "oracle_evidence_opened": False,
        "source_artifacts_modified": False,
    }
    atomic_write_json(output / "data_access_ledger.json", access_ledger)

    train_rows = _partition(joined, "train")
    calibration_rows = _partition(joined, "calibration")
    evaluation_rows = _partition(joined, "evaluation")
    logistic, tree, resources = _fit_controllers(config, train_rows)
    resources["dataset"] = {
        "total_clusters": len(joined),
        "train_clusters": len(train_rows),
        "calibration_clusters": len(calibration_rows),
        "evaluation_clusters": len(evaluation_rows),
        "feature_count": len(FEATURE_NAMES),
        "class_support": support,
        "independence_unit": "cluster",
    }

    train_x, train_y = _arrays(train_rows)
    calibration_x, calibration_y = _arrays(calibration_rows)
    logistic_train_scores = logistic.score(train_x)
    tree_train_scores = tree.score(train_x)
    logistic_calibration_scores = logistic.score(calibration_x)
    tree_calibration_scores = tree.score(calibration_x)
    equal_calibration_scores = np.asarray(
        [float(row["equal_weight_risk"]) for row in calibration_rows], dtype=np.float64
    )
    logistic_calibration = calibrate_threshold(calibration_y, logistic_calibration_scores)
    tree_calibration = calibrate_threshold(calibration_y, tree_calibration_scores)
    equal_calibration = calibrate_threshold(calibration_y, equal_calibration_scores)
    calibration_artifact = {
        "schema_version": "1.0",
        "partition": "calibration_only",
        "evaluation_labels_used": False,
        "selection_objective": "maximum_balanced_accuracy",
        "models": {
            "logistic": logistic_calibration.to_dict(),
            "shallow_tree": tree_calibration.to_dict(),
            "deterministic_equal_weight": equal_calibration.to_dict(),
        },
        "budget_control": {
            "source": "logistic_calibration_recovery_fraction",
            "calibration_recovered": logistic_calibration.recovered,
            "calibration_n": len(calibration_rows),
            "fraction": logistic_calibration.recovered / len(calibration_rows),
            "evaluation_recover_count": math.floor(
                logistic_calibration.recovered
                * len(evaluation_rows)
                / len(calibration_rows)
                + 0.5
            ),
        },
    }
    model_artifact = {
        "schema_version": "1.0",
        "architecture_experiment": "F_TINY_LEARNED_SIGNIFICANCE_CONTROLLER",
        "feature_provenance": "PRE_ANSWER_ON_POLICY_CORNERSTONE_STATE",
        "backbone": "FROZEN_NOT_LOADED",
        "reinforcement_learning": False,
        "logistic": logistic.to_dict(),
        "shallow_tree_control": tree.to_dict(),
    }
    atomic_write_json(output / "models.json", model_artifact)
    atomic_write_json(output / "training_resources.json", resources)
    # Thresholds and models are durable before held-out scoring is performed.
    atomic_write_json(output / "calibration.json", calibration_artifact)

    evaluation_x, _evaluation_y = _arrays(evaluation_rows)
    inference_times: dict[str, float] = {}

    start = time.perf_counter()
    logistic_scores = logistic.score(evaluation_x)
    inference_times["logistic_vector_seconds"] = time.perf_counter() - start
    logistic_actions = logistic_scores >= logistic_calibration.threshold

    start = time.perf_counter()
    tree_scores = tree.score(evaluation_x)
    inference_times["shallow_tree_vector_seconds"] = time.perf_counter() - start
    tree_actions = tree_scores >= tree_calibration.threshold

    equal_scores = np.asarray(
        [float(row["equal_weight_risk"]) for row in evaluation_rows], dtype=np.float64
    )
    equal_actions = equal_scores >= equal_calibration.threshold
    accept_scores = np.zeros(len(evaluation_rows), dtype=np.float64)
    accept_actions = np.zeros(len(evaluation_rows), dtype=np.bool_)
    recover_scores = np.ones(len(evaluation_rows), dtype=np.float64)
    recover_actions = np.ones(len(evaluation_rows), dtype=np.bool_)
    budget_config = dict(dict(config["baselines"])["calibration_budget_hash"])
    budget_scores = hash_budget_scores(
        [str(row["cluster_id"]) for row in evaluation_rows], salt=str(budget_config["salt"])
    )
    budget_count = int(calibration_artifact["budget_control"]["evaluation_recover_count"])
    budget_actions = exact_budget_actions(budget_scores, recover_count=budget_count)

    method_values: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float | None, str]] = {
        "logistic": (
            logistic_scores,
            logistic_actions,
            logistic_calibration.threshold,
            "calibration_only",
        ),
        "shallow_tree": (
            tree_scores,
            tree_actions,
            tree_calibration.threshold,
            "calibration_only",
        ),
        "deterministic_equal_weight": (
            equal_scores,
            equal_actions,
            equal_calibration.threshold,
            "calibration_only",
        ),
        "constant_accept_all": (accept_scores, accept_actions, None, "fixed_constant"),
        "constant_recover_all": (recover_scores, recover_actions, None, "fixed_constant"),
        "calibration_budget_hash": (
            budget_scores,
            budget_actions,
            None,
            "calibration_budget_plus_label_blind_hash_rank",
        ),
    }
    evaluation_metrics = {
        method: _evaluate_method(
            evaluation_rows,
            scores=scores,
            actions=actions,
            threshold=threshold,
            threshold_source=threshold_source,
        )
        for method, (scores, actions, threshold, threshold_source) in method_values.items()
    }
    training_descriptives = {
        "logistic": binary_metrics(
            train_y,
            logistic_train_scores,
            logistic_train_scores >= logistic_calibration.threshold,
        ),
        "shallow_tree": binary_metrics(
            train_y,
            tree_train_scores,
            tree_train_scores >= tree_calibration.threshold,
        ),
    }
    calibration_descriptives = {
        "logistic": binary_metrics(
            calibration_y,
            logistic_calibration_scores,
            logistic_calibration_scores >= logistic_calibration.threshold,
        ),
        "shallow_tree": binary_metrics(
            calibration_y,
            tree_calibration_scores,
            tree_calibration_scores >= tree_calibration.threshold,
        ),
        "deterministic_equal_weight": binary_metrics(
            calibration_y,
            equal_calibration_scores,
            equal_calibration_scores >= equal_calibration.threshold,
        ),
    }
    metrics_artifact = {
        "schema_version": "1.0",
        "target": "compressed_error",
        "positive_action": "recompute_full_cache",
        "training_descriptive_at_calibrated_threshold": training_descriptives,
        "calibration_resubstitution": calibration_descriptives,
        "held_out_evaluation": evaluation_metrics,
        "controller_inference_timing": {
            **inference_times,
            "evaluation_rows": len(evaluation_rows),
            "timing_limit": "single tiny vector call; below stable benchmarking resolution",
        },
    }
    atomic_write_json(output / "metrics.json", metrics_artifact)

    prediction_rows: list[dict[str, Any]] = []
    for index, row in enumerate(evaluation_rows):
        prediction_rows.append(
            {
                "schema_version": "1.0",
                "case_id": row["case_id"],
                "cluster_id": row["cluster_id"],
                "task_family": row["task_family"],
                "split": "evaluation",
                "compressed_error": row["compressed_error"],
                "full_cache_correct": row["full_cache_correct"],
                "methods": {
                    method: {
                        "risk_score": float(values[0][index]),
                        "recover": bool(values[1][index]),
                    }
                    for method, values in method_values.items()
                },
            }
        )
    _write_jsonl(output / "predictions.evaluation.post_answer.jsonl", prediction_rows)
    _write_curve_artifacts(
        output,
        evaluation_rows,
        {
            "logistic": logistic_scores,
            "shallow_tree": tree_scores,
            "deterministic_equal_weight": equal_scores,
            "calibration_budget_hash": budget_scores,
        },
    )

    logistic_eval = evaluation_metrics["logistic"]
    equal_eval = evaluation_metrics["deterministic_equal_weight"]
    train_gap = float(training_descriptives["logistic"]["balanced_accuracy"]) - float(
        logistic_eval["balanced_accuracy"]
    )
    summary = {
        "schema_version": "1.0",
        "status": "COMPLETE",
        "architecture_experiment": "F_TINY_LEARNED_SIGNIFICANCE_CONTROLLER",
        "classification": "UNDERPOWERED_OVERFIT_SCREEN_ONLY",
        "claim": (
            "Eleven selected clusters (5 train/3 calibration/3 evaluation) cannot establish "
            "that Cornerstone state supports a reusable learned controller. Results are a "
            "leakage-safe mechanism screen only."
        ),
        "dataset": resources["dataset"],
        "split": {
            "algorithm": config["split"]["algorithm"],
            "salt": config["split"]["salt"],
            "prelabel_sha256": freeze["durable_split_snapshot_sha256"],
            "outcome_based_reassignment": False,
        },
        "primary_logistic": {
            "trainable_parameter_count": logistic.trainable_parameter_count,
            "optimizer": logistic.optimizer,
            "training_wall_seconds": resources["logistic"]["wall_seconds"],
            "peak_process_rss_bytes_sampled": resources["logistic"][
                "peak_process_rss_bytes_sampled"
            ],
            "calibration_threshold": logistic_calibration.threshold,
            "held_out_error_roc_auc": logistic_eval["error_roc_auc"],
            "held_out_balanced_accuracy": logistic_eval["balanced_accuracy"],
            "held_out_coverage": logistic_eval["coverage"],
            "held_out_selective_risk": logistic_eval["selective_risk"],
            "held_out_system_accuracy": logistic_eval["cost_and_system"]["system_accuracy"],
            "train_minus_evaluation_balanced_accuracy": train_gap,
        },
        "held_out_comparison": {
            method: {
                "error_roc_auc": values["error_roc_auc"],
                "balanced_accuracy": values["balanced_accuracy"],
                "coverage": values["coverage"],
                "selective_risk": values["selective_risk"],
                "system_accuracy": values["cost_and_system"]["system_accuracy"],
                "recovered": values["recovered"],
            }
            for method, values in evaluation_metrics.items()
        },
        "comparison_to_equal_weight": {
            "auc_delta": float(logistic_eval["error_roc_auc"])
            - float(equal_eval["error_roc_auc"]),
            "balanced_accuracy_delta": float(logistic_eval["balanced_accuracy"])
            - float(equal_eval["balanced_accuracy"]),
            "no_inferential_test": True,
        },
        "inferential_limits": [
            "Only 11 B0-selected clusters exist; the held-out evaluation has three clusters.",
            "Each evaluation AUC is based on only two positive-negative pairwise comparisons.",
            "Thresholds are calibrated on three clusters and are highly unstable.",
            "R2 benchmark-family cases are absent and only one evidence depth is represented.",
            "The source deterministic feature design preceded this split, but the overall program is exploratory.",
            "Training or calibration perfection is compatible with overfit and supports no deployment claim.",
        ],
        "decision": (
            "Retain the controller API and leakage guard, but do not advance the learned controller "
            "until a materially larger, prospectively split capability-conditioned dataset exists."
        ),
    }
    atomic_write_json(output / "summary.json", summary)

    manifest["status"] = "COMPLETE"
    manifest["completed_at"] = utc_now()
    manifest["input_provenance"] = {
        "source_run_id": "halo_recovery_controller_screen_v1",
        "source_feature_sha256": sha256_file(paths["source_features"]),
        "source_outcome_sha256": sha256_file(paths["source_outcomes"]),
        "source_manifest_sha256": sha256_file(paths["source_manifest"]),
        "source_artifacts_modified": False,
    }
    manifest["data_access"] = {
        "phase": "COMPLETE",
        "prelabel_freeze": "prelabel_freeze.json",
        "access_ledger": "data_access_ledger.json",
        "outcomes_opened_after_freeze": True,
        "oracle_evidence_opened": False,
        "outcome_based_split_reassignment": False,
    }
    manifest["dataset"] = resources["dataset"]
    manifest["learning"] = {
        "backbone_trainable_parameters": 0,
        "logistic_trainable_parameters": logistic.trainable_parameter_count,
        "tree_learned_scalars": tree.learned_scalar_count,
        "reinforcement_learning": False,
        "calibration_thresholds": {
            "logistic": logistic_calibration.threshold,
            "shallow_tree": tree_calibration.threshold,
            "deterministic_equal_weight": equal_calibration.threshold,
        },
        "resources": resources,
    }
    manifest["artifacts"] = {
        "resolved_config": "resolved_config.yaml",
        "pre_answer_features": "features.pre_answer.jsonl",
        "pre_answer_split": "split.pre_answer.jsonl",
        "prelabel_freeze": "prelabel_freeze.json",
        "data_access_ledger": "data_access_ledger.json",
        "post_answer_join": "joined.post_answer.jsonl",
        "models": "models.json",
        "calibration": "calibration.json",
        "training_resources": "training_resources.json",
        "metrics": "metrics.json",
        "evaluation_predictions": "predictions.evaluation.post_answer.jsonl",
        "risk_coverage": "risk_coverage.evaluation.jsonl",
        "risk_coverage_csv": "risk_coverage.evaluation.csv",
        "risk_coverage_figure": "risk_coverage.evaluation.png",
        "summary": "summary.json",
    }
    manifest["notes"] = [
        "Architecture F only; the Halo backbone was frozen and not loaded.",
        "Feature and SHA-256 cluster split artifacts were fsync-backed and durable before outcomes opened.",
        "Every frozen split had both classes; no salt search, stratification, or outcome reassignment exists.",
        "The three-cluster evaluation is underpowered and supports no controller-quality claim.",
        "Source recovery artifacts were read-only and remain the authority for feature/outcome provenance.",
    ]
    atomic_write_json(output / "manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_tiny_significance_controller.yaml",
    )
    args = parser.parse_args()
    summary = run(args.config.resolve(strict=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
