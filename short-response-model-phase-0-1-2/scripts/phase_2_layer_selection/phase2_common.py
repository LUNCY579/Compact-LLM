from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PHASE_1 = ROOT / "scripts" / "phase_1_dense_sft"


def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def contribution(score_subset: float, score_base: float, score_all: float) -> float | None:
    denominator = score_all - score_base
    if abs(denominator) < 1e-6:
        return None
    return (score_subset - score_base) / denominator


def train_and_evaluate(
    *,
    model_id: str,
    train_file: Path,
    dev_file: Path,
    benchmark: Path,
    output_dir: Path,
    selected_layers: str,
    max_steps: int,
    seed: int,
    qlora: bool,
    gradient_checkpointing: bool,
    target_module_group: str = "both",
) -> dict[str, Any]:
    train_dir = output_dir / "training"
    eval_dir = output_dir / "evaluation"
    train_cmd = [
        sys.executable,
        str(PHASE_1 / "train_dense_sft.py"),
        "--model-id", model_id,
        "--train-file", str(train_file),
        "--dev-file", str(dev_file),
        "--output-dir", str(train_dir),
        "--selected-layers", selected_layers,
        "--target-module-group", target_module_group,
        "--max-steps", str(max_steps),
        "--seed", str(seed),
    ]
    if qlora:
        train_cmd.append("--qlora")
    if gradient_checkpointing:
        train_cmd.append("--gradient-checkpointing")
    run(train_cmd)

    run([
        sys.executable,
        str(PHASE_1 / "evaluate_model.py"),
        "--model-id", model_id,
        "--adapter-path", str(train_dir / "adapter"),
        "--benchmark", str(benchmark),
        "--output-dir", str(eval_dir),
    ])
    summary = load_json(eval_dir / "summary.json")
    manifest = load_json(train_dir / "training_manifest.json")
    metrics = manifest.get("train_metrics", {})
    return {
        "selected_layers": selected_layers,
        "seed": seed,
        "quality_evaluation_version": summary.get("quality_evaluation_version"),
        "quality_policy_version": summary.get("quality_policy_version"),
        "case_count": int(summary.get("case_count", 0)),
        "mean_quality_score": float(summary.get("mean_quality_score", summary["mean_heuristic_score"])),
        "mean_quality_dimensions": dict(summary.get("mean_quality_dimensions", summary.get("quality_dimension_means", {}))),
        "quality_dimension_means": dict(summary.get("quality_dimension_means", summary.get("mean_quality_dimensions", {}))),
        "mean_issue_cost": float(summary.get("mean_issue_cost", 0.0)),
        "severe_issue_case_rate": float(summary.get("severe_issue_case_rate", 0.0)),
        "mean_heuristic_score": float(summary["mean_heuristic_score"]),
        "mean_legacy_heuristic_score": float(summary.get("mean_legacy_heuristic_score", summary["mean_heuristic_score"])),
        "quality_pass_rate": float(summary.get("quality_pass_rate", 0.0)),
        "intelligibility_pass_rate": float(summary.get("intelligibility_pass_rate", 0.0)),
        "hard_failure_case_count": int(summary.get("hard_failure_case_count", 0)),
        "worst_decile_mean_quality_score": float(summary.get("worst_decile_mean_quality_score", 0.0)),
        "p10_quality_score": float(summary.get("p10_quality_score", 0.0)),
        "mean_reference_alignment_score": summary.get("mean_reference_alignment_score"),
        "median_reply_tokens": float(summary["median_reply_tokens"]),
        "p90_reply_tokens": float(summary["p90_reply_tokens"]),
        "complete_utterance_rate": float(summary["complete_utterance_rate"]),
        "category_scores": summary.get("category_quality_scores", summary.get("category_scores", {})),
        "category_quality_scores": summary.get("category_quality_scores", summary.get("category_scores", {})),
        "duplicate_reply_count": int(summary.get("duplicate_reply_count", 0)),
        "critical_failure_count": int(summary.get("critical_failure_count", 0)),
        "critical_case_count": int(summary.get("critical_case_count", 0)),
        "major_issue_count": int(summary.get("major_issue_count", 0)),
        "major_case_count": int(summary.get("major_case_count", 0)),
        "semantic_warning_count": int(summary.get("semantic_warning_count", 0)),
        "observable_quality_pass_rate": float(summary.get("observable_quality_pass_rate", summary.get("semantic_pass_rate", 1.0))),
        "peak_eval_vram_gb": float(summary["peak_vram_gb"]),
        "peak_train_vram_gb": float(manifest["peak_allocated_vram_gb"]),
        "train_runtime_seconds": float(metrics.get("train_runtime", 0.0)),
        "train_samples_per_second": float(metrics.get("train_samples_per_second", 0.0)),
        "target_module_count": int(manifest["target_module_count"]),
        "artifact_dir": str(output_dir),
    }
