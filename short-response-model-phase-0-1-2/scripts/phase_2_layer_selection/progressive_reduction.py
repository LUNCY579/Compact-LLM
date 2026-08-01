from __future__ import annotations

"""Run Phase 2B progressive decoder-depth reduction with recovery branches.

Design principles:
- Every current layer is tested as a one-layer deletion at each stage.
- The first pass is cheap, reversible, and uses the configured development probe.
- Only the strongest raw deletions receive short recovery fine-tuning by default.
- Every accepted frontier stage is materialized and preserved; rejected full
  checkpoints may be regenerated from their parent plus recorded drop ID.
- Progress continues only while proportional loaded weight-memory reduction exceeds
  combined proportional quality opportunity loss. No absolute quality floor is used.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, run, write_json
from scripts.phase_2_layer_selection.create_depth_candidate import lineage_for_model
from scripts.common.quality_evaluation import require_quality_v2, summary_quality_score
from scripts.common.quality_gate import assert_valid_reference
from scripts.common.opportunity_evaluation import evaluate_opportunity, resource_reduction

PYTHON = sys.executable
EVALUATOR = ROOT / "scripts/phase_1_dense_sft/evaluate_model.py"
TRAIN_SFT = ROOT / "scripts/phase_1_dense_sft/train_dense_sft.py"
MATERIALIZE = ROOT / "scripts/common/materialize_model.py"
MEASURE = ROOT / "scripts/common/measure_model_footprint.py"
CREATE_CANDIDATE = ROOT / "scripts/phase_2_layer_selection/create_depth_candidate.py"
BUILD_PROBE = ROOT / "scripts/phase_2_layer_selection/build_structural_probe.py"
TEACHER_FORCE = ROOT / "scripts/phase_2_layer_selection/evaluate_teacher_forcing.py"
COMPARE = ROOT / "scripts/phase_2_layer_selection/compare_outputs.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run progressive Phase 2B structural reduction.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--sft-adapter", type=Path, default=ROOT / "artifacts/dense_sft/adapter")
    parser.add_argument("--train-file", type=Path, default=ROOT / "data/train/train.jsonl")
    parser.add_argument("--dev-file", type=Path, default=ROOT / "data/dev/dev.jsonl")
    parser.add_argument("--probe-file", type=Path, default=ROOT / "data/phase_2_structural_reduction/probe_50.jsonl")
    parser.add_argument("--phase-benchmark", type=Path, default=ROOT / "data/benchmark/shortchatbench_pilot.jsonl")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/phase_2_progressive_reduction")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/PHASE_2B_PROGRESSIVE_REDUCTION_REPORT.md")
    parser.add_argument("--gate-report", type=Path, default=ROOT / "reports/PHASE_2B_STOP_GATE.md")
    parser.add_argument("--probe-limit", type=int, default=50)
    parser.add_argument("--probe-seed", type=int, default=20260728)
    parser.add_argument("--max-drops", type=int, default=6)
    parser.add_argument("--recovery-top-k", type=int, default=3)
    parser.add_argument("--recovery-steps", type=int, default=50)
    parser.add_argument("--recovery-seed", type=int, default=11)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--quality-profile",
        choices=("balanced", "logic_oriented", "conversational_permissive"),
        default="balanced",
        help="Relative quality-opportunity weighting profile.",
    )
    # Historical threshold flags remain parseable so old commands fail safely
    # neither at the shell nor by silently changing the new decision. They are
    # ignored and omitted from help because absolute promotion floors are banned.
    for deprecated in (
        "--minimum-reduction-advantage", "--max-score-loss",
        "--balanced-max-score-loss", "--max-semantic-pass-loss",
        "--max-category-loss", "--max-nll-increase",
        "--min-complete-utterance-rate", "--max-worst-decile-loss",
        "--min-quality-pass-rate", "--min-intelligibility-pass-rate",
        "--max-reference-alignment-loss",
    ):
        parser.add_argument(deprecated, type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-new-major-issues", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--skip-recovery", action="store_true")
    parser.add_argument("--keep-candidate-checkpoints", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def relative_loss(reference: float, candidate: float) -> float:
    """Return non-negative relative loss when larger is better."""

    return max(0.0, reference - candidate) / max(abs(reference), 1e-12)


def max_category_loss(reference: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Compatibility helper backed by current continuous category scores."""

    require_quality_v2(reference, "reference category summary")
    require_quality_v2(candidate, "candidate category summary")
    ref_scores = reference.get("category_quality_scores") or {}
    cand_scores = candidate.get("category_quality_scores") or {}
    losses: dict[str, float] = {}
    for category, ref_score in ref_scores.items():
        cand_score = cand_scores.get(category)
        losses[str(category)] = 1.0 if cand_score is None else max(0.0, float(ref_score) - float(cand_score))
    return (max(losses.values()) if losses else 0.0), losses


def assess_candidate(
    *,
    baseline_generation: dict[str, Any],
    candidate_generation: dict[str, Any],
    baseline_nll: dict[str, Any],
    candidate_nll: dict[str, Any],
    baseline_footprint: dict[str, Any],
    candidate_footprint: dict[str, Any],
    output_deviation: dict[str, Any],
    **deprecated_thresholds: Any,
) -> dict[str, Any]:
    """Apply the relative opportunity rule for structural depth reduction.

    All former absolute quality floors/ceilings are accepted as compatibility
    arguments but deliberately ignored. Loaded weight-tensor reduction is the
    primary benefit. Diverse quality dimensions, tail quality, issue costs,
    categories, and target NLL contribute proportionally to opportunity loss.
    """

    base_bytes = float(baseline_footprint["parameter_tensor_bytes_loaded"])
    cand_bytes = float(candidate_footprint["parameter_tensor_bytes_loaded"])
    weight_reduction = resource_reduction(base_bytes, cand_bytes)
    base_nll_value = float(baseline_nll["mean_target_nll"])
    cand_nll_value = float(candidate_nll["mean_target_nll"])
    opportunity = evaluate_opportunity(
        baseline_generation,
        candidate_generation,
        resource_gain=weight_reduction,
        resource_name="loaded_weight_tensor_memory_reduction",
        reference_target_nll=base_nll_value,
        candidate_target_nll=cand_nll_value,
        profile=str(deprecated_thresholds.get("quality_profile") or "balanced"),
    )

    base_score = summary_quality_score(baseline_generation)
    cand_score = summary_quality_score(candidate_generation)
    simple_mean_loss = max(0.0, base_score - cand_score) / max(abs(base_score), 1e-12)
    base_eval_vram = float(baseline_generation.get("peak_vram_gb", 0.0))
    cand_eval_vram = float(candidate_generation.get("peak_vram_gb", 0.0))
    eval_vram_reduction = resource_reduction(base_eval_vram, cand_eval_vram) if base_eval_vram > 0 else None
    base_latency = float(baseline_generation.get("mean_generation_seconds", 0.0))
    cand_latency = float(candidate_generation.get("mean_generation_seconds", 0.0))
    latency_reduction = 1.0 - cand_latency / base_latency if base_latency > 0 else None
    base_parameters = float(baseline_footprint.get("parameters", {}).get("total", 0.0))
    cand_parameters = float(candidate_footprint.get("parameters", {}).get("total", 0.0))
    parameter_reduction = resource_reduction(base_parameters, cand_parameters) if base_parameters > 0 else None
    quality_loss = float(opportunity["quality_opportunity_loss_fraction"])
    margin = float(opportunity["opportunity_margin"])
    efficiency_ratio = float(weight_reduction or 0.0) / quality_loss if quality_loss > 1e-12 else math.inf
    reasons = list(opportunity.get("technical_invalidity_reasons", []))
    if not opportunity["eligible"] and not reasons:
        reasons.append(opportunity["decision_explanation"])

    return {
        "eligible_for_progression": bool(opportunity["eligible"]),
        "rejection_reasons": reasons,
        "decision_rule": "weight-memory reduction > combined proportional quality opportunity loss",
        "weight_memory_reduction_fraction": float(weight_reduction or 0.0),
        "parameter_count_reduction_fraction": parameter_reduction,
        "peak_evaluation_vram_reduction_fraction": eval_vram_reduction,
        "mean_generation_latency_reduction_fraction": latency_reduction,
        "quality_opportunity_loss_fraction": quality_loss,
        "quality_score_loss_fraction": simple_mean_loss,
        "heuristic_score_loss_fraction": simple_mean_loss,
        "reduction_minus_quality_loss": margin,
        "reduction_minus_score_loss": margin,
        "opportunity_margin": margin,
        "memory_to_quality_loss_ratio": efficiency_ratio,
        "memory_to_score_loss_ratio": efficiency_ratio,
        "quality_dimension_losses": opportunity.get("quality_dimension_losses", {}),
        "category_quality_losses": opportunity.get("category_quality_relative_losses", {}),
        "category_score_losses": opportunity.get("category_quality_relative_losses", {}),
        "maximum_category_quality_loss": opportunity.get("maximum_category_relative_loss", 0.0),
        "maximum_category_score_loss": opportunity.get("maximum_category_relative_loss", 0.0),
        "target_nll_increase_fraction": opportunity.get("quality_dimension_losses", {}).get("target_nll", 0.0),
        "quality_profile": opportunity.get("quality_profile"),
        "opportunity_policy_version": opportunity.get("opportunity_policy_version"),
        "technical_valid": opportunity.get("technical_valid"),
        "deprecated_threshold_arguments_ignored": sorted(deprecated_thresholds),
        "quality_pass_rate": float(candidate_generation.get("quality_pass_rate", 0.0)),
        "intelligibility_pass_rate": float(candidate_generation.get("intelligibility_pass_rate", 0.0)),
        "hard_failure_case_count": int(candidate_generation.get("hard_failure_case_count", 0)),
        "new_issue_diagnostics": {
            "new_critical_failure_count": int(output_deviation.get("new_critical_failure_count", 0)),
            "new_major_issue_count": int(output_deviation.get("new_major_issue_count", 0)),
            "new_hard_failure_count": int(output_deviation.get("new_hard_failure_count", 0)),
        },
        "exact_reply_match_rate": float(output_deviation.get("exact_reply_match_rate", 0.0)),
        "mean_fuzzy_reply_similarity": float(output_deviation.get("mean_fuzzy_reply_similarity", 0.0)),
    }


def ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    assessment = row["assessment"]
    generation = row["generation_summary"]
    return (
        bool(assessment["eligible_for_progression"]),
        float(assessment.get("opportunity_margin", float("-inf"))),
        float(assessment.get("weight_memory_reduction_fraction", 0.0)),
        -float(assessment.get("quality_opportunity_loss_fraction", float("inf"))),
        float(generation.get("worst_decile_mean_quality_score", 0.0)),
        float(generation.get("mean_quality_score", 0.0)),
    )


def run_if_needed(command: list[str], sentinel: Path, resume: bool) -> None:
    if resume and sentinel.exists():
        print(f"[reuse] {sentinel}")
        return
    run(command, cwd=ROOT)


def evaluate_variant(
    *,
    model_id: str,
    adapter_path: Path | None,
    evidence_dir: Path,
    probe_file: Path,
    baseline_results: Path,
    baseline_generation: dict[str, Any],
    baseline_nll: dict[str, Any],
    baseline_footprint: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    generation_dir = evidence_dir / "generation"
    nll_dir = evidence_dir / "teacher_forcing"
    footprint_path = evidence_dir / "footprint.json"
    deviation_json = evidence_dir / "output_deviation.json"
    deviation_report = evidence_dir / "OUTPUT_DEVIATION.md"

    summary_sentinel = generation_dir / "summary.json"
    if args.resume and summary_sentinel.exists():
        try:
            require_quality_v2(read_json(summary_sentinel), str(summary_sentinel))
        except ValueError:
            print(f"[stale evaluator] rebuilding {generation_dir}")
            shutil.rmtree(generation_dir, ignore_errors=True)

    eval_cmd = [
        PYTHON,
        str(EVALUATOR),
        "--model-id",
        model_id,
        "--benchmark",
        str(probe_file),
        "--output-dir",
        str(generation_dir),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if adapter_path is not None:
        eval_cmd += ["--adapter-path", str(adapter_path)]
    run_if_needed(eval_cmd, generation_dir / "summary.json", args.resume)

    nll_cmd = [
        PYTHON,
        str(TEACHER_FORCE),
        "--model-id",
        model_id,
        "--input",
        str(probe_file),
        "--output-dir",
        str(nll_dir),
    ]
    if adapter_path is not None:
        nll_cmd += ["--adapter-path", str(adapter_path)]
    run_if_needed(nll_cmd, nll_dir / "summary.json", args.resume)

    run_if_needed(
        [PYTHON, str(MEASURE), "--model", model_id, "--out", str(footprint_path)],
        footprint_path,
        args.resume,
    )
    run_if_needed(
        [
            PYTHON,
            str(COMPARE),
            "--reference",
            str(baseline_results),
            "--candidate",
            str(generation_dir / "results.jsonl"),
            "--out-json",
            str(deviation_json),
            "--out-report",
            str(deviation_report),
        ],
        deviation_json,
        args.resume,
    )

    generation_summary = read_json(generation_dir / "summary.json")
    nll_summary = read_json(nll_dir / "summary.json")
    footprint = read_json(footprint_path)
    deviation_payload = read_json(deviation_json)
    deviation_summary = deviation_payload["summary"]
    assessment = assess_candidate(
        baseline_generation=baseline_generation,
        candidate_generation=generation_summary,
        baseline_nll=baseline_nll,
        candidate_nll=nll_summary,
        baseline_footprint=baseline_footprint,
        candidate_footprint=footprint,
        output_deviation=deviation_summary,
        quality_profile=args.quality_profile,
    )
    payload = {
        "model_id": model_id,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "generation_summary": generation_summary,
        "nll_summary": nll_summary,
        "footprint": footprint,
        "output_deviation_summary": deviation_summary,
        "assessment": assessment,
        "evidence_dir": str(evidence_dir),
    }
    write_json(evidence_dir / "combined_metrics.json", payload)
    return payload


def recover_candidate(
    *,
    raw_model: Path,
    candidate_evidence: Path,
    args: argparse.Namespace,
) -> Path:
    recovery_root = candidate_evidence / "recovery_training"
    command = [
        PYTHON,
        str(TRAIN_SFT),
        "--model-id",
        str(raw_model),
        "--train-file",
        str(args.train_file),
        "--dev-file",
        str(args.dev_file),
        "--output-dir",
        str(recovery_root),
        "--selected-layers",
        "all",
        "--target-module-group",
        "both",
        "--max-steps",
        str(args.recovery_steps),
        "--seed",
        str(args.recovery_seed),
    ]
    if args.qlora:
        command.append("--qlora")
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    run_if_needed(command, recovery_root / "adapter/adapter_config.json", args.resume)
    return recovery_root / "adapter"


def copy_or_materialize_selected(
    *,
    selected: dict[str, Any],
    stage_model_dir: Path,
    retained_original_ids: list[int],
    parent_model: str,
) -> None:
    if stage_model_dir.exists():
        shutil.rmtree(stage_model_dir)
    source_model = Path(selected["raw_model"])
    adapter = selected.get("adapter_path")
    if adapter:
        run(
            [
                PYTHON,
                str(MATERIALIZE),
                "--model-id",
                str(source_model),
                "--adapter-path",
                str(adapter),
                "--output-dir",
                str(stage_model_dir),
                "--source-label",
                "phase_2b_recovered_frontier_stage",
            ],
            cwd=ROOT,
        )
        shutil.copy2(
            source_model / "depth_reduction_manifest.json",
            stage_model_dir / "depth_reduction_manifest.json",
        )
    else:
        shutil.copytree(source_model, stage_model_dir)

    write_json(
        stage_model_dir / "phase_2b_stage_manifest.json",
        {
            "parent_model": parent_model,
            "selected_variant": selected["variant_id"],
            "dropped_original_layer_id": selected["dropped_original_layer_id"],
            "retained_original_layer_ids": retained_original_ids,
            "recovery_applied": bool(adapter),
            "recovery_adapter": str(adapter) if adapter else None,
            "assessment": selected["assessment"],
            "source_evidence": selected["evidence_dir"],
        },
    )


def select_profiles(
    baseline_profile: dict[str, Any],
    frontier: list[dict[str, Any]],
    balanced_max_score_loss: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Select profiles from the relative Pareto frontier.

    ``balanced_max_score_loss`` is retained only for command compatibility and
    does not control selection.
    """

    eligible = [row for row in frontier if bool(row["assessment"].get("eligible_for_progression"))]
    if not eligible:
        return {"full_quality": baseline_profile, "balanced": baseline_profile, "min_memory": baseline_profile}
    balanced = max(
        eligible,
        key=lambda row: (
            float(row["assessment"].get("opportunity_margin", float("-inf"))),
            float(row["assessment"].get("weight_memory_reduction_fraction", 0.0)),
        ),
    )
    minimum = max(eligible, key=lambda row: float(row["assessment"].get("weight_memory_reduction_fraction", 0.0)))
    return {"full_quality": baseline_profile, "balanced": balanced, "min_memory": minimum}


def format_optional_percent(value: Any) -> str:
    """Render optional measured deltas without converting missing CPU data to zero."""

    return "n/a" if value is None else f"{float(value):+.2%}"


def render_report(
    *,
    baseline: dict[str, Any],
    stages: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    first_stage = [row for row in candidates if row["stage"] == 1]
    first_lines: list[str] = []
    for row in sorted(first_stage, key=lambda item: item["dropped_original_layer_id"]):
        a = row["assessment"]
        first_lines.append(
            f"| {row['dropped_original_layer_id']} | {row['variant_kind']} | "
            f"{row['generation_summary']['mean_quality_score']:.3f} | "
            f"{a['weight_memory_reduction_fraction']:.2%} | "
            f"{a['quality_opportunity_loss_fraction']:.2%} | "
            f"{a['opportunity_margin']:+.2%} | "
            f"{format_optional_percent(a.get('peak_evaluation_vram_reduction_fraction'))} | "
            f"{format_optional_percent(a.get('mean_generation_latency_reduction_fraction'))} | "
            f"{row['nll_summary']['mean_target_nll']:.4f} | "
            f"{a.get('quality_profile', args.quality_profile)} | "
            f"{'WORTHWHILE' if a['eligible_for_progression'] else 'NOT WORTHWHILE'} |"
        )
    if not first_lines:
        first_lines.append("| - | - | - | - | - | - | - | - | - | - | - |")

    stage_lines: list[str] = []
    for row in stages:
        a = row["assessment"]
        stage_lines.append(
            f"| {row['stage']} | {row['dropped_original_layer_ids']} | "
            f"{row['remaining_layer_count']} | {row['generation_summary']['mean_quality_score']:.3f} | "
            f"{a['weight_memory_reduction_fraction']:.2%} | "
            f"{a['quality_opportunity_loss_fraction']:.2%} | "
            f"{a['opportunity_margin']:+.2%} | "
            f"{format_optional_percent(a.get('peak_evaluation_vram_reduction_fraction'))} | "
            f"{format_optional_percent(a.get('mean_generation_latency_reduction_fraction'))} | "
            f"{row['variant_kind']} |"
        )
    if not stage_lines:
        stage_lines.append("| 0 | [] | - | - | - | - | - | - | - | no proportionally worthwhile reduction |")

    def profile_line(name: str, row: dict[str, Any]) -> str:
        if name == "full_quality" or "assessment" not in row:
            return f"| {name} | {row['remaining_layer_count']} | 0.00% | 0.00% | 0.00% | `{row['inference_model']}` |"
        a = row["assessment"]
        return (
            f"| {name} | {row['remaining_layer_count']} | "
            f"{a['weight_memory_reduction_fraction']:.2%} | "
            f"{a['quality_opportunity_loss_fraction']:.2%} | "
            f"{a['opportunity_margin']:+.2%} | `{row['inference_model']}` |"
        )

    profile_lines = [profile_line(name, row) for name, row in profiles.items()]
    return f"""# PHASE 2B — PROGRESSIVE STRUCTURAL REDUCTION REPORT

## Purpose

Test which complete layers can be removed, optionally recovered, and combined
progressively. This structural experiment remains distinct from selective-layer
training.

## Evidence boundary

- Probe cases: {args.probe_limit}, selected deterministically from development data.
- Sealed evaluation used: **No**.
- Quality evaluator: **3.0**.
- Relative opportunity profile: **{args.quality_profile}**.
- Full-reference quality: {baseline['generation_summary']['mean_quality_score']:.3f}.
- Full-reference target NLL: {baseline['nll_summary']['mean_target_nll']:.4f}.
- Every current layer is tested as a raw deletion at each progressive stage.
- Recovery fine-tuning is limited to the strongest {args.recovery_top_k} raw candidates per stage unless configured otherwise.

## Stage 1 individual-layer results

| Dropped original layer | Variant | Quality | Weight-memory gain | Quality opportunity loss | Opportunity margin | Eval VRAM change | Generation latency change | Target NLL | Profile | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(first_lines)}

## Accepted progressive frontier

| Stage | Original layers dropped so far | Remaining layers | Quality | Weight-memory gain | Quality opportunity loss | Opportunity margin | Eval VRAM change | Generation latency change | Selected form |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(stage_lines)}

## Retained deployment/training profiles

| Profile | Remaining layers | Weight-memory gain | Quality opportunity loss | Opportunity margin | Model |
|---|---:|---:|---:|---:|---|
{chr(10).join(profile_lines)}

## Decision rule

A technically comparable candidate enters the frontier when its measured
proportional loaded weight-tensor memory reduction is **strictly greater** than
its combined proportional quality opportunity loss. Mean quality, tail quality,
category behavior, comprehensibility, task adequacy, logic, safety, context,
completeness, content anchors, issue cost, severe-collapse rate, and target NLL
contribute continuously according to the selected profile. None is an independent
absolute floor or issue-count veto. Latency and evaluation VRAM remain secondary
measured evidence and do not silently replace the declared primary resource.

Human review and later phase controls remain mandatory. Technical invalidity—such
as missing evidence, a corrupt checkpoint, mismatched evaluator versions, or
non-reproducible provenance—still blocks comparison because no valid opportunity
calculation exists.

## Workflow handoff

- `full_quality` preserves the original all-layer SFT reference.
- `balanced` selects the eligible frontier point with the largest opportunity margin.
- `min_memory` selects the eligible frontier point with the largest weight-memory gain.
- Phase 4 soups must use ingredients from the same selected architecture profile.
"""


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output_root / "model_catalog.json"
    if args.resume and catalog_path.exists():
        print(f"Completed catalog already exists: {catalog_path}")
        return

    if not args.sft_adapter.exists():
        raise FileNotFoundError(args.sft_adapter)
    if not args.probe_file.exists():
        run(
            [
                PYTHON,
                str(BUILD_PROBE),
                "--input",
                str(args.dev_file),
                "--output",
                str(args.probe_file),
                "--limit",
                str(args.probe_limit),
                "--seed",
                str(args.probe_seed),
            ],
            cwd=ROOT,
        )

    reference_model = args.output_root / "reference/full_model"
    run_if_needed(
        [
            PYTHON,
            str(MATERIALIZE),
            "--model-id",
            args.model_id,
            "--adapter-path",
            str(args.sft_adapter),
            "--output-dir",
            str(reference_model),
            "--source-label",
            "phase_1_all_layer_sft_reference_for_phase_2b",
        ],
        reference_model / "materialization_manifest.json",
        args.resume,
    )

    reference_evidence = args.output_root / "reference/evidence"
    baseline_generation_dir = reference_evidence / "generation"
    baseline_nll_dir = reference_evidence / "teacher_forcing"
    baseline_footprint_path = reference_evidence / "footprint.json"
    if args.resume and (baseline_generation_dir / "summary.json").exists():
        try:
            require_quality_v2(read_json(baseline_generation_dir / "summary.json"), "Phase 2B reference evaluation")
        except ValueError:
            print(f"[stale evaluator] rebuilding {baseline_generation_dir}")
            shutil.rmtree(baseline_generation_dir, ignore_errors=True)
    run_if_needed(
        [
            PYTHON,
            str(EVALUATOR),
            "--model-id",
            str(reference_model),
            "--benchmark",
            str(args.probe_file),
            "--output-dir",
            str(baseline_generation_dir),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ],
        baseline_generation_dir / "summary.json",
        args.resume,
    )
    run_if_needed(
        [
            PYTHON,
            str(TEACHER_FORCE),
            "--model-id",
            str(reference_model),
            "--input",
            str(args.probe_file),
            "--output-dir",
            str(baseline_nll_dir),
        ],
        baseline_nll_dir / "summary.json",
        args.resume,
    )
    run_if_needed(
        [PYTHON, str(MEASURE), "--model", str(reference_model), "--out", str(baseline_footprint_path)],
        baseline_footprint_path,
        args.resume,
    )
    baseline_generation = read_json(baseline_generation_dir / "summary.json")
    require_quality_v2(baseline_generation, "Phase 2B reference evaluation")
    assert_valid_reference(baseline_generation, label="Phase 2B full reference")
    baseline_nll = read_json(baseline_nll_dir / "summary.json")
    baseline_footprint = read_json(baseline_footprint_path)
    baseline = {
        "generation_summary": baseline_generation,
        "nll_summary": baseline_nll,
        "footprint": baseline_footprint,
        "results": str(baseline_generation_dir / "results.jsonl"),
    }

    current_model = reference_model
    dropped_original: list[int] = []
    all_candidate_rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    work_root = args.output_root / "work_models"
    evidence_root = args.output_root / "evidence"
    frontier_root = args.output_root / "frontier"

    for stage in range(1, args.max_drops + 1):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(current_model)
        current_count = int(config.num_hidden_layers)
        if current_count <= 2:
            break
        original_ids = lineage_for_model(current_model, current_count)
        stage_raw_rows: list[dict[str, Any]] = []

        for current_index, original_id in enumerate(original_ids):
            variant_id = f"stage_{stage:02d}_drop_original_{original_id:02d}_raw"
            raw_model = work_root / variant_id
            candidate_evidence = evidence_root / f"stage_{stage:02d}" / variant_id
            run_if_needed(
                [
                    PYTHON,
                    str(CREATE_CANDIDATE),
                    "--model",
                    str(current_model),
                    "--drop-current-layer",
                    str(current_index),
                    "--output-dir",
                    str(raw_model),
                ],
                raw_model / "depth_reduction_manifest.json",
                args.resume,
            )
            metrics = evaluate_variant(
                model_id=str(raw_model),
                adapter_path=None,
                evidence_dir=candidate_evidence,
                probe_file=args.probe_file,
                baseline_results=baseline_generation_dir / "results.jsonl",
                baseline_generation=baseline_generation,
                baseline_nll=baseline_nll,
                baseline_footprint=baseline_footprint,
                args=args,
            )
            row = {
                **metrics,
                "variant_id": variant_id,
                "variant_kind": "raw deletion",
                "stage": stage,
                "parent_model": str(current_model),
                "raw_model": str(raw_model),
                "adapter_path": None,
                "drop_current_layer_id": current_index,
                "dropped_original_layer_id": original_id,
                "retained_original_layer_ids": [value for value in original_ids if value != original_id],
            }
            stage_raw_rows.append(row)
            all_candidate_rows.append(row)

        # Recovery is intentionally focused by default. Set recovery-top-k to
        # the current layer count to fine-tune every raw deletion branch.
        recovered_rows: list[dict[str, Any]] = []
        if not args.skip_recovery and args.recovery_steps > 0 and args.recovery_top_k > 0:
            recovery_targets = sorted(stage_raw_rows, key=ranking_key, reverse=True)[: args.recovery_top_k]
            for raw_row in recovery_targets:
                variant_id = raw_row["variant_id"].replace("_raw", "_recovered")
                candidate_evidence = evidence_root / f"stage_{stage:02d}" / variant_id
                adapter = recover_candidate(
                    raw_model=Path(raw_row["raw_model"]),
                    candidate_evidence=candidate_evidence,
                    args=args,
                )
                metrics = evaluate_variant(
                    model_id=raw_row["raw_model"],
                    adapter_path=adapter,
                    evidence_dir=candidate_evidence,
                    probe_file=args.probe_file,
                    baseline_results=baseline_generation_dir / "results.jsonl",
                    baseline_generation=baseline_generation,
                    baseline_nll=baseline_nll,
                    baseline_footprint=baseline_footprint,
                    args=args,
                )
                row = {
                    **metrics,
                    "variant_id": variant_id,
                    "variant_kind": "deletion + recovery SFT",
                    "stage": stage,
                    "parent_model": str(current_model),
                    "raw_model": raw_row["raw_model"],
                    "adapter_path": str(adapter),
                    "drop_current_layer_id": raw_row["drop_current_layer_id"],
                    "dropped_original_layer_id": raw_row["dropped_original_layer_id"],
                    "retained_original_layer_ids": raw_row["retained_original_layer_ids"],
                }
                recovered_rows.append(row)
                all_candidate_rows.append(row)

        stage_options = stage_raw_rows + recovered_rows
        eligible = [row for row in stage_options if row["assessment"]["eligible_for_progression"]]
        if not eligible:
            print(f"Stopping at stage {stage}: no deletion satisfies the reduction/quality rule.")
            break
        selected = max(eligible, key=ranking_key)
        dropped_original.append(int(selected["dropped_original_layer_id"]))
        stage_dir = frontier_root / f"stage_{stage:02d}_drop_{'_'.join(map(str, dropped_original))}"
        stage_model = stage_dir / "model"
        copy_or_materialize_selected(
            selected=selected,
            stage_model_dir=stage_model,
            retained_original_ids=selected["retained_original_layer_ids"],
            parent_model=str(current_model),
        )

        frontier_row = {
            **selected,
            "stage": stage,
            "inference_model": str(stage_model),
            "training_model_id": str(stage_model),
            "training_adapter": None,
            "evaluation_summary": str(Path(selected["evidence_dir"]) / "generation/summary.json"),
            "dropped_original_layer_ids": list(dropped_original),
            "remaining_layer_count": len(selected["retained_original_layer_ids"]),
        }
        write_json(stage_dir / "stage_result.json", frontier_row)
        frontier_rows.append(frontier_row)
        current_model = stage_model

        if not args.keep_candidate_checkpoints:
            # Evidence, recovery adapters, and selected frontier checkpoints are
            # retained. Rejected full checkpoints are reproducible from the
            # preserved parent model and recorded drop-current-layer ID.
            if work_root.exists():
                shutil.rmtree(work_root)

    baseline_profile = {
        "profile": "full_quality",
        "training_model_id": args.model_id,
        "training_adapter": str(args.sft_adapter),
        "inference_model": str(reference_model),
        "evaluation_summary": str(baseline_generation_dir / "summary.json"),
        "remaining_layer_count": int(baseline_footprint["num_hidden_layers"]),
        "assessment": {
            "eligible_for_progression": True,
            "weight_memory_reduction_fraction": 0.0,
            "quality_opportunity_loss_fraction": 0.0,
            "quality_score_loss_fraction": 0.0,
            "heuristic_score_loss_fraction": 0.0,
            "opportunity_margin": 0.0,
            "reduction_minus_quality_loss": 0.0,
            "reduction_minus_score_loss": 0.0,
        },
    }
    profiles = select_profiles(
        baseline_profile,
        frontier_rows,
    )
    for name, profile in profiles.items():
        profile["profile"] = name

    # Phase 3 gates compare against ShortChatBench Pilot, not the development
    # structural probe. Re-evaluate each retained profile on that common
    # benchmark so downstream comparisons remain apples-to-apples.
    profile_eval_cache: dict[str, str] = {}
    for name, profile in profiles.items():
        inference_model = str(profile["inference_model"])
        if inference_model not in profile_eval_cache:
            profile_eval_dir = args.output_root / "profile_validation" / name / "generation"
            run_if_needed(
                [
                    PYTHON,
                    str(EVALUATOR),
                    "--model-id",
                    inference_model,
                    "--benchmark",
                    str(args.phase_benchmark),
                    "--output-dir",
                    str(profile_eval_dir),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                ],
                profile_eval_dir / "summary.json",
                args.resume,
            )
            profile_eval_cache[inference_model] = str(profile_eval_dir / "summary.json")
        profile["evaluation_summary"] = profile_eval_cache[inference_model]
        profile["phase_benchmark"] = str(args.phase_benchmark)

    catalog = {
        "manifest_version": 1,
        "status": "PENDING_HUMAN_REVIEW",
        "source_model_id": args.model_id,
        "source_sft_adapter": str(args.sft_adapter),
        "probe_file": str(args.probe_file),
        "phase_benchmark": str(args.phase_benchmark),
        "probe_case_count": args.probe_limit,
        "baseline": baseline,
        "opportunity_policy": {
            "decision_rule": "loaded weight-tensor reduction > combined proportional quality opportunity loss",
            "quality_profile": args.quality_profile,
            "absolute_promotion_thresholds": "prohibited",
            "quality_evaluation_version": "3.0",
        },
        "frontier": frontier_rows,
        "profiles": profiles,
        "all_candidate_registry": str(args.output_root / "candidate_registry.json"),
        "gate_report": str(args.gate_report),
        "downstream_rule": (
            "Phase 3 may use full_quality, balanced, or min_memory only after "
            "the non-full profile receives explicit Phase 2B human approval."
        ),
    }
    write_json(args.output_root / "candidate_registry.json", {"candidates": all_candidate_rows})
    write_json(catalog_path, catalog)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_report(
            baseline=baseline,
            stages=frontier_rows,
            candidates=all_candidate_rows,
            profiles=profiles,
            args=args,
        ),
        encoding="utf-8",
    )
    gate = f"""# PHASE 2B STOP GATE\n\n## Automated status\n\n`PENDING_HUMAN_REVIEW`\n\n- Model catalog: `{catalog_path}`\n- Progressive report: `{args.report}`\n- Accepted frontier stages: {len(frontier_rows)}\n- Recommended balanced profile: `{profiles['balanced']['inference_model']}`\n- Recommended minimum-memory profile: `{profiles['min_memory']['inference_model']}`\n\n## Required human decision\n\nReview the configured probe side-by-side outputs for every accepted stage and the lowest-scoring rejected neighbors. Confirm that the chosen profile preserves meaning, tone, context, clarification, and safety. The automated rule is relative opportunity only: weight-memory reduction must exceed combined proportional quality opportunity loss. Human review remains required before downstream use.\n\n- Human decision: `PENDING`\n- Reviewer:\n- Date:\n- Approved profiles: `full_quality`\n- Notes:\n"""
    args.gate_report.parent.mkdir(parents=True, exist_ok=True)
    args.gate_report.write_text(gate, encoding="utf-8")
    print(catalog_path)
    print(args.report)
    print(args.gate_report)


if __name__ == "__main__":
    main()
