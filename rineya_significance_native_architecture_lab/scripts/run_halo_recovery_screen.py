#!/usr/bin/env python3
"""Build the deterministic Architecture-E risk/coverage screen from Halo traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rineya_significance_lab.evaluation.recovery import (  # noqa: E402
    RISK_COMPONENT_NAMES,
    RecoveryCaseOutcome,
    aggregate_recovery_clusters,
    assert_pre_answer_feature_rows,
    descriptive_error_roc_auc,
    extract_cornerstone_risk_features,
    risk_coverage_curve,
)
from rineya_significance_lab.manifest import (  # noqa: E402
    atomic_write_json,
    build_manifest,
    sha256_file,
    utc_now,
)


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("recovery config must be a mapping")
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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_config(config: Mapping[str, Any]) -> None:
    protocol = dict(config.get("protocol", {}))
    if protocol.get("open_loop_execution_mode") != "OPEN_LOOP_TRACE_SIMULATION":
        raise ValueError("screen requires OPEN_LOOP_TRACE_SIMULATION")
    if protocol.get("source_condition") != "true_realized_count":
        raise ValueError("screen must use the on-policy true-realized-count condition")
    if protocol.get("pairing_unit") != "whole_cluster":
        raise ValueError("recovery decisions must use whole clusters")
    if protocol.get("answer_access") != "only_after_pre_answer_features_are_durable":
        raise ValueError("the no-leakage answer-access contract is not frozen")

    model = dict(config.get("model", {}))
    core_id = str(model.get("core_id", ""))
    if ".halo." not in core_id.lower() or model.get("expected_source_base") != "Qwen3-0.6B":
        raise ValueError("Architecture E is scoped to the published Halo Qwen3-0.6B core")
    if tuple(model.get("allowed_core_ids", ())) != (core_id,):
        raise ValueError("allowed_core_ids must contain only the active Halo core")
    if bool(model.get("trust_remote_code", True)):
        raise ValueError("trust_remote_code must remain false")

    risk = dict(config.get("risk_signal", {}))
    if risk.get("decision_boundary") != "final_available_nonfinal_chunk_boundary":
        raise ValueError("this trace contract supports only the final available boundary")
    weights = dict(risk.get("weights", {}))
    if set(weights) != set(RISK_COMPONENT_NAMES):
        raise ValueError(f"risk weights must cover exactly {RISK_COMPONENT_NAMES}")
    if any(float(value) < 0.0 or not math.isfinite(float(value)) for value in weights.values()):
        raise ValueError("risk weights must be finite and non-negative")
    if sum(float(value) for value in weights.values()) <= 0.0:
        raise ValueError("at least one risk weight must be positive")

    controller = dict(config.get("controller", {}))
    if controller.get("type") != "deterministic_fixed_threshold":
        raise ValueError("learned or adaptive recovery controllers are outside this screen")
    if controller.get("recovery_action") != "recompute_full_cache":
        raise ValueError("the bounded screen requires the full-cache recovery action")
    if int(controller.get("learned_parameters", -1)) != 0:
        raise ValueError("the deterministic screen must have zero learned parameters")


def build_pre_answer_features(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> list[dict[str, Any]]:
    """Read only prompt/state artifacts; answer-bearing files are not arguments."""

    source = paths["source_run"]
    source_condition = str(dict(config["protocol"])["source_condition"])
    unscored = {
        (str(row["case_id"]), str(row["condition"])): row
        for row in _jsonl(source / "predictions.unscored.jsonl")
    }
    boundaries = [
        row
        for row in _jsonl(source / "boundary_diagnostics.jsonl")
        if str(row.get("condition")) == source_condition
    ]
    if not boundaries:
        raise RuntimeError("no on-policy true-count boundary traces were found")
    rows: list[dict[str, Any]] = []
    risk = dict(config["risk_signal"])
    weights = {name: float(value) for name, value in dict(risk["weights"]).items()}
    for index in boundaries:
        case_id = str(index["case_id"])
        source_row = unscored[(case_id, source_condition)]
        summaries = index.get("boundaries")
        if not isinstance(summaries, list) or len(summaries) < 2:
            raise RuntimeError(f"{case_id}: risk extraction needs at least two boundaries")
        final = summaries[-1]
        if not isinstance(final, dict):
            raise RuntimeError(f"{case_id}: malformed final boundary summary")
        trace_path = source / str(index["diagnostic_npz"])
        with np.load(trace_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        features = extract_cornerstone_risk_features(
            arrays,
            case_id=case_id,
            cluster_id=str(source_row["cluster_id"]),
            task_family=str(source_row["family"]),
            boundary_index=int(final["boundary_index"]),
            layer_count=int(final["layer_count"]),
            top_mass_fraction=float(risk["top_mass_fraction"]),
            weights=weights,
        )
        row = features.to_dict()
        row["source_condition"] = source_condition
        row["source_trace"] = str(index["diagnostic_npz"])
        rows.append(row)
    rows.sort(key=lambda row: str(row["case_id"]))
    assert_pre_answer_feature_rows(rows)
    return rows


def _full_cache_kv_bytes(config: Mapping[str, Any], resident_entries: int) -> int:
    model = dict(config["model"])
    accounting = dict(config["accounting"])
    return int(
        resident_entries
        * int(model["expected_layers"])
        * int(model["expected_kv_heads"])
        * int(model["expected_head_dim"])
        * int(accounting["kv_tensor_count"])
        * int(accounting["kv_dtype_bytes"])
    )


def join_outcomes_after_feature_freeze(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    feature_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[RecoveryCaseOutcome], list[dict[str, Any]]]:
    """Open answer-derived outcomes only after the feature view is durable."""

    source = paths["source_run"]
    full = paths["full_cache_run"]
    source_condition = str(dict(config["protocol"])["source_condition"])
    source_cases = {
        str(row["case_id"]): row
        for row in _jsonl(source / "cases.jsonl")
        if str(row.get("condition")) == source_condition
    }
    full_cases = {str(row["case_id"]): row for row in _jsonl(full / "cases.jsonl")}
    full_predictions = {
        str(row["case_id"]): row for row in _jsonl(full / "predictions.unscored.jsonl")
    }
    prompt_rows = {
        str(row["case_id"]): row for row in _jsonl(paths["prompt_only_inference"])
    }
    input_dtype_bytes = int(dict(config["accounting"])["input_token_dtype_bytes"])
    outcomes: list[RecoveryCaseOutcome] = []
    scored: list[dict[str, Any]] = []
    for feature in feature_rows:
        case_id = str(feature["case_id"])
        compressed = source_cases[case_id]
        reference = full_cases[case_id]
        reference_prediction = full_predictions[case_id]
        prompt = prompt_rows[case_id]
        if not bool(compressed.get("full_cache_correct")):
            raise RuntimeError(f"{case_id}: source selection is not B0-eligible")
        if str(reference_prediction["token_ids_sha256"]) != str(prompt["token_ids_sha256"]):
            raise RuntimeError(f"{case_id}: full-cache reference used different prompt tokens")
        input_tokens = int(reference_prediction["input_tokens"])
        if input_tokens != int(prompt["token_count"]):
            raise RuntimeError(f"{case_id}: prompt token count drift")
        fallback_bytes = _full_cache_kv_bytes(
            config, int(reference_prediction["resident_entries"])
        )
        outcome = RecoveryCaseOutcome(
            case_id=case_id,
            cluster_id=str(feature["cluster_id"]),
            task_family=str(feature["task_family"]),
            risk_score=float(feature["risk_score"]),
            compressed_correct=bool(compressed["correct"]),
            full_cache_correct=bool(reference["correct"]),
            compressed_wall_seconds=float(compressed["efficiency"]["wall_seconds"]),
            fallback_wall_seconds=float(reference["efficiency"]["wall_seconds"]),
            compressed_persistent_bytes=int(
                compressed["efficiency"]["total_persistent_bytes"]
            ),
            fallback_persistent_bytes=fallback_bytes,
            recompute_input_bytes=input_tokens * input_dtype_bytes,
        )
        outcomes.append(outcome)
        scored.append(
            {
                "schema_version": "1.0",
                "case_id": outcome.case_id,
                "cluster_id": outcome.cluster_id,
                "task_family": outcome.task_family,
                "risk_score": outcome.risk_score,
                "compressed_correct": outcome.compressed_correct,
                "full_cache_correct": outcome.full_cache_correct,
                "compressed_wall_seconds": outcome.compressed_wall_seconds,
                "recorded_full_cache_fallback_wall_seconds": outcome.fallback_wall_seconds,
                "compressed_persistent_bytes": outcome.compressed_persistent_bytes,
                "full_cache_fallback_kv_bytes": outcome.fallback_persistent_bytes,
                "recompute_input_token_bytes": outcome.recompute_input_bytes,
                "outcome_access_phase": "AFTER_DURABLE_PRE_ANSWER_FEATURE_VIEW",
            }
        )
    return outcomes, scored


def _write_curve_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty risk curve")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    coverage = [float(row["coverage"]) for row in rows]
    selective = [
        math.nan if row["selective_risk"] is None else float(row["selective_risk"])
        for row in rows
    ]
    accuracy = [float(row["system_accuracy"]) for row in rows]
    wall = [float(row["wall_multiplier_vs_compressed"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.25), constrained_layout=True)
    axes[0].plot(coverage, selective, marker="o", label="risk among accepted")
    axes[0].plot(coverage, [1.0 - value for value in accuracy], marker="s", label="system error")
    axes[0].set(xlabel="accept coverage", ylabel="error rate", ylim=(-0.03, 1.03))
    axes[0].set_title("Open-loop whole-cluster risk/coverage")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(wall, accuracy, marker="o", color="#7b3294")
    axes[1].set(xlabel="wall-time multiplier vs compressed", ylabel="system accuracy")
    axes[1].set_title("Recorded fallback cost tradeoff")
    axes[1].grid(alpha=0.25)
    figure.suptitle("Exploratory: 11 B0-selected Halo clusters; no held-out threshold")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config_path: Path) -> dict[str, Any]:
    import yaml

    started_at = utc_now()
    config_path = config_path.resolve()
    config = _load_config(config_path)
    validate_config(config)
    paths = _resolve_paths(config)
    source_manifest_data = json.loads(
        (paths["source_run"] / "manifest.json").read_text(encoding="utf-8")
    )
    expected_model = dict(config["model"])
    source_model = dict(source_manifest_data.get("model", {}))
    if source_model.get("id") != expected_model["core_id"]:
        raise RuntimeError("source trace run used a different model core")
    source_hashes = dict(source_model.get("hashes", {}))
    expected_hashes = dict(expected_model["expected_hashes"])
    mismatched_hashes = {
        name: {"source": source_hashes.get(name), "expected": digest}
        for name, digest in expected_hashes.items()
        if source_hashes.get(name) != digest
    }
    if mismatched_hashes:
        raise RuntimeError(f"source trace model hash mismatch: {mismatched_hashes}")
    output = paths["screen_output"]
    if output.exists():
        # A failed process before manifest creation is an explicitly incomplete
        # checkpoint.  Replacing only this runner's known partial artifacts
        # makes the bounded analysis restartable without weakening immutable
        # completed-run protection.
        if (output / "manifest.json").exists():
            raise RuntimeError(f"refusing to overwrite completed recovery screen: {output}")
        allowed_partial = {
            "features.pre_answer.jsonl",
            "resolved_config.yaml",
            "risk_coverage.csv",
            "risk_coverage.jsonl",
            "risk_coverage.png",
            "scored_cases.jsonl",
            "summary.json",
        }
        unexpected = {path.name for path in output.iterdir()}.difference(allowed_partial)
        if unexpected:
            raise RuntimeError(f"unexpected files in partial recovery screen: {sorted(unexpected)}")
    else:
        output.mkdir(parents=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )

    # The durable feature file is intentionally written before either answer-bearing
    # cases.jsonl is opened by join_outcomes_after_feature_freeze().
    feature_rows = build_pre_answer_features(config, paths)
    features_path = output / "features.pre_answer.jsonl"
    _write_jsonl(features_path, feature_rows)
    assert_pre_answer_feature_rows(_jsonl(features_path))

    outcomes, scored = join_outcomes_after_feature_freeze(config, paths, feature_rows)
    clusters = aggregate_recovery_clusters(outcomes)
    points = risk_coverage_curve(clusters)
    curve_rows = [point.to_dict() for point in points]
    _write_jsonl(output / "scored_cases.jsonl", scored)
    _write_jsonl(output / "risk_coverage.jsonl", curve_rows)
    _write_curve_csv(output / "risk_coverage.csv", curve_rows)
    _make_figure(output / "risk_coverage.png", curve_rows)

    source_manifest = paths["source_run"] / "manifest.json"
    source_content = paths["source_run"] / "content_manifest.json"
    full_manifest = paths["full_cache_run"] / "manifest.json"
    compressed_correct = sum(row.compressed_correct for row in clusters)
    full_correct = sum(row.full_cache_correct for row in clusters)
    errors = len(clusters) - compressed_correct
    summary = {
        "schema_version": "1.0",
        "status": "COMPLETE",
        "classification": "EXPLORATORY_OPEN_LOOP_RECOVERY_SCREEN",
        "execution_mode": "OPEN_LOOP_TRACE_SIMULATION",
        "architecture_experiment": "E_SIGNIFICANCE_CONTROLLED_RECOMPUTATION",
        "controller": {
            "type": "deterministic fixed-threshold sweep",
            "learned_parameters": 0,
            "action": "full-cache recomputation",
            "deployment_threshold_selected": False,
        },
        "feature_contract": {
            "answer_blind": True,
            "oracle_evidence_blind": True,
            "features": list(RISK_COMPONENT_NAMES),
            "feature_rows_sha256": sha256_file(features_path),
            "query_delta_source": (
                "adjacent cumulative S difference aligned by semantic entry_id; "
                "the source run scored only temporary query probes"
            ),
        },
        "whole_cluster_results": {
            "independent_clusters": len(clusters),
            "families": sorted({row.task_family for row in clusters}),
            "compressed_correct": compressed_correct,
            "compressed_accuracy": compressed_correct / len(clusters),
            "full_cache_correct": full_correct,
            "full_cache_accuracy": full_correct / len(clusters),
            "compressed_errors": errors,
            "descriptive_error_roc_auc": descriptive_error_roc_auc(clusters),
            "risk_score_minimum": min(row.risk_score for row in clusters),
            "risk_score_maximum": max(row.risk_score for row in clusters),
            "risk_curve_points": len(points),
        },
        "cost_accounting": {
            "compressed_baseline_wall_seconds": sum(
                row.compressed_wall_seconds for row in clusters
            ),
            "full_recovery_total_wall_seconds": points[-1].total_wall_seconds,
            "full_recovery_wall_multiplier": points[-1].wall_multiplier_vs_compressed,
            "full_recovery_recompute_input_bytes": points[-1].recompute_input_bytes,
            "sequential_peak_persistent_bytes_at_full_recovery": points[-1].peak_persistent_bytes,
            "incremental_peak_persistent_bytes_at_full_recovery": (
                points[-1].incremental_peak_persistent_bytes
            ),
            "wall_time_interpretation": (
                "sum of recorded compressed execution (chunk 128) and prior same-prompt "
                "B0 full-cache execution (chunk 256); not schedule-matched or a "
                "contemporaneous latency benchmark"
            ),
        },
        "selection_bias": list(config["selection_bias"]),
        "inferential_limit": (
            "The same 11 B0-selected clusters supply risk scores and outcomes. This is "
            "resubstitution-only mechanism evidence, not held-out controller validation."
        ),
        "claim": (
            "This artifact only tests deterministic risk ordering and counterfactual fallback "
            "cost/coverage. It supports no broad Halo quality or deployable-controller claim."
        ),
    }
    atomic_write_json(output / "summary.json", summary)

    model = dict(config["model"])
    manifest = build_manifest(
        repo=REPO_ROOT,
        config=config,
        run_id=output.name,
        execution_mode="OPEN_LOOP_TRACE_SIMULATION",
        model={
            "status": "VERIFIED",
            "id": model["core_id"],
            "path": str(config["paths"]["model"]),
            "architecture": model["expected_architecture"],
            "hashes": dict(model["expected_hashes"]),
        },
        benchmark={"name": "11 B0-selected Halo count-control clusters"},
        architecture_variant="E deterministic significance-controlled full-cache recovery",
        seed=int(dict(config["runtime"])["seed"]),
        compute_tier="A",
        status="COMPLETE",
        resolved_config_path="configs/halo_recovery_controller.yaml",
    )
    manifest["created_at"] = started_at
    manifest["completed_at"] = utc_now()
    manifest["runtime"]["precision"] = "numpy-float64-analysis-of-float32-traces"
    manifest["runtime"]["model_loaded"] = False
    manifest["input_provenance"] = {
        "source_run_id": paths["source_run"].name,
        "source_model_hash_contract": "PASS",
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_content_manifest_sha256": sha256_file(source_content),
        "source_boundary_index_sha256": sha256_file(
            paths["source_run"] / "boundary_diagnostics.jsonl"
        ),
        "source_unscored_predictions_sha256": sha256_file(
            paths["source_run"] / "predictions.unscored.jsonl"
        ),
        "full_cache_run_id": paths["full_cache_run"].name,
        "full_cache_manifest_sha256": sha256_file(full_manifest),
        "prompt_only_inference_sha256": sha256_file(paths["prompt_only_inference"]),
    }
    manifest["artifacts"] = {
        "resolved_config": "resolved_config.yaml",
        "pre_answer_features": "features.pre_answer.jsonl",
        "scored_cases": "scored_cases.jsonl",
        "risk_coverage": "risk_coverage.jsonl",
        "risk_coverage_csv": "risk_coverage.csv",
        "risk_coverage_figure": "risk_coverage.png",
        "summary": "summary.json",
    }
    manifest["notes"] = [
        "Features were durable before answer-derived outcomes were opened.",
        "Whole clusters are indivisible gate units; tied risk scores are never split.",
        "Thresholds are enumerated, not fitted; there is no held-out calibration split.",
        "Fallback outcomes and timing are reused from an earlier same-prompt full-cache Halo run whose prefill chunk differs (256 versus 128).",
        "No broad quality, causality, or deployability claim is supported.",
    ]
    atomic_write_json(output / "manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "halo_recovery_controller.yaml",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
