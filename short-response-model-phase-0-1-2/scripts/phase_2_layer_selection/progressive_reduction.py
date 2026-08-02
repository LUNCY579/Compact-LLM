from __future__ import annotations

"""Run Phase 2B progressive decoder-depth reduction with recovery branches.

Design principles:
- Every current layer is tested as a one-layer deletion at each stage.
- The first pass is cheap, reversible, and uses the configured development probe.
- Only the strongest raw deletions receive short recovery fine-tuning initially;
  every remaining branch is recovered once if the focused search would stop.
- Every accepted frontier stage is materialized and preserved; rejected full
  checkpoints may be regenerated from their parent plus recorded drop ID.
- Resume restores the last accepted frontier directly instead of replaying it.
- Progress requires positive cumulative and immediate-parent opportunity margins.
  No absolute quality floor is used.
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, run, sha256, sha256_tree, write_json
from scripts.phase_2_layer_selection.create_depth_candidate import lineage_for_model
from scripts.common.quality_evaluation import QUALITY_EVALUATION_VERSION, require_quality_v4, summary_quality_score
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
    parser.add_argument("--probe-file", type=Path, default=ROOT / "data/phase_2_structural_reduction/probe_50_v4_3.jsonl")
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



RESUME_SCHEMA_VERSION = 3
EVIDENCE_PROVENANCE_SCHEMA_VERSION = 1
TERMINAL_NO_PROGRESS_REASONS = {
    "no_marginally_worthwhile_deletion_after_exhaustive_recovery",
    "no_marginally_worthwhile_deletion_recovery_disabled",
    "minimum_layer_guard_reached",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normal_path(value: str | Path) -> str:
    """Return a stable comparison token for repository paths.

    Existing Phase 2B catalogs can contain relative paths, absolute paths, or
    Windows-style separators. Resolution is intentionally anchored at the
    repository root so continuation does not depend on the caller's cwd.
    """

    raw = str(value).strip().replace("\\", "/")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        candidate = candidate.resolve(strict=False)
    except OSError:
        pass
    return candidate.as_posix().rstrip("/").casefold()


def _repo_path(value: str | Path) -> Path:
    raw = str(value).replace("\\", "/")
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _file_fingerprint(value: str | Path) -> str:
    path = _repo_path(value)
    return sha256(path) if path.is_file() else "missing"


def _tree_fingerprint(value: str | Path) -> str:
    path = _repo_path(value)
    return sha256_tree(path) if path.exists() else "missing"


def build_variant_provenance(
    *,
    stage: int,
    variant_id: str,
    variant_kind: str,
    parent_model: str | Path,
    dropped_original_layer_id: int,
    retained_original_layer_ids: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Describe the exact lineage and evaluator context behind reusable evidence."""

    recovered_variant = "recover" in str(variant_kind).casefold()
    return {
        "schema_version": EVIDENCE_PROVENANCE_SCHEMA_VERSION,
        "stage": int(stage),
        "variant_id": str(variant_id),
        "variant_kind": str(variant_kind),
        "parent_model": _normal_path(parent_model),
        "dropped_original_layer_id": int(dropped_original_layer_id),
        "retained_original_layer_ids": [int(value) for value in retained_original_layer_ids],
        "probe_file": _normal_path(args.probe_file),
        "probe_file_sha256": _file_fingerprint(args.probe_file),
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "quality_config_sha256": _file_fingerprint(ROOT / "configs/quality_evaluation.json"),
        "opportunity_policy_sha256": _file_fingerprint(ROOT / "configs/relative_opportunity_evaluation.json"),
        "max_new_tokens": int(args.max_new_tokens),
        "recovery_steps": int(args.recovery_steps) if recovered_variant else 0,
        "recovery_seed": int(args.recovery_seed) if recovered_variant else None,
        "qlora": bool(args.qlora) if recovered_variant else False,
        "gradient_checkpointing": bool(args.gradient_checkpointing) if recovered_variant else False,
    }


def prepare_variant_evidence(evidence_dir: Path, expected: dict[str, Any]) -> bool:
    """Keep reusable evidence only when its exact parent lineage still matches.

    Returns ``True`` when stale evidence was removed.  This is essential after
    rescoring: if the corrected evaluator selects a different Stage-N parent,
    Stage-(N+1) generations from the old branch must never be silently reused.
    """

    manifest = evidence_dir / "evidence_provenance.json"
    stale = False
    if manifest.exists():
        try:
            existing = read_json(manifest)
        except Exception:
            existing = {}
        stale = existing != expected
    elif evidence_dir.exists() and any(evidence_dir.iterdir()):
        stale = True
    if stale:
        print(f"[stale lineage] removing evidence from a different parent: {evidence_dir}")
        shutil.rmtree(evidence_dir, ignore_errors=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest, expected)
    return stale


def build_run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Persist every setting that can alter resumed Phase 2B evidence.

    ``max_drops`` is recorded but deliberately excluded from strict resume
    equality because increasing it is the supported way to extend a completed
    frontier.
    """

    return {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "model_id": str(args.model_id),
        "sft_adapter": str(args.sft_adapter),
        "train_file": str(args.train_file),
        "dev_file": str(args.dev_file),
        "probe_file": str(args.probe_file),
        "phase_benchmark": str(args.phase_benchmark),
        "probe_limit": int(args.probe_limit),
        "probe_seed": int(args.probe_seed),
        "quality_profile": str(args.quality_profile),
        "recovery_top_k": int(args.recovery_top_k),
        "recovery_steps": int(args.recovery_steps),
        "recovery_seed": int(args.recovery_seed),
        "max_new_tokens": int(args.max_new_tokens),
        "qlora": bool(args.qlora),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "skip_recovery": bool(args.skip_recovery),
        "sft_adapter_fingerprint": _tree_fingerprint(args.sft_adapter),
        "train_file_sha256": _file_fingerprint(args.train_file),
        "dev_file_sha256": _file_fingerprint(args.dev_file),
        "probe_file_sha256": _file_fingerprint(args.probe_file),
        "phase_benchmark_sha256": _file_fingerprint(args.phase_benchmark),
        "quality_evaluation_config_sha256": _file_fingerprint(ROOT / "configs/quality_evaluation.json"),
        "opportunity_policy_config_sha256": _file_fingerprint(ROOT / "configs/relative_opportunity_evaluation.json"),
        "evaluator_script_sha256": _file_fingerprint(EVALUATOR),
        "requested_max_drops": int(args.max_drops),
    }


def validate_resume_catalog(catalog: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Reject continuation when persisted provenance conflicts with this run.

    Catalogs produced before this hotfix did not persist every recovery flag.
    For those legacy catalogs, all fields that *were* persisted are verified and
    the unavailable fields are explicitly reported as assumptions. They are not
    silently claimed to have been verified.
    """

    mismatches: list[str] = []
    verified: list[str] = []
    assumptions: list[str] = []

    scalar_checks = {
        "source_model_id": (catalog.get("source_model_id"), str(args.model_id)),
        "probe_case_count": (catalog.get("probe_case_count"), int(args.probe_limit)),
    }
    for name, (existing, requested) in scalar_checks.items():
        if existing is None:
            assumptions.append(name)
        elif str(existing) != str(requested):
            mismatches.append(f"{name}: catalog={existing!r}, requested={requested!r}")
        else:
            verified.append(name)

    path_checks = {
        "source_sft_adapter": (catalog.get("source_sft_adapter"), args.sft_adapter),
        "probe_file": (catalog.get("probe_file"), args.probe_file),
        "phase_benchmark": (catalog.get("phase_benchmark"), args.phase_benchmark),
    }
    for name, (existing, requested) in path_checks.items():
        if existing is None:
            assumptions.append(name)
        elif _normal_path(existing) != _normal_path(requested):
            mismatches.append(f"{name}: catalog={existing!r}, requested={str(requested)!r}")
        else:
            verified.append(name)

    existing_profile = (catalog.get("opportunity_policy") or {}).get("quality_profile")
    if existing_profile is None:
        assumptions.append("quality_profile")
    elif str(existing_profile) != str(args.quality_profile):
        mismatches.append(
            f"quality_profile: catalog={existing_profile!r}, requested={args.quality_profile!r}"
        )
    else:
        verified.append("quality_profile")

    evaluator_version = (catalog.get("opportunity_policy") or {}).get("quality_evaluation_version")
    if evaluator_version is None:
        assumptions.append("quality_evaluation_version")
    elif str(evaluator_version) != QUALITY_EVALUATION_VERSION:
        mismatches.append(
            f"quality_evaluation_version: catalog={evaluator_version!r}, requested={QUALITY_EVALUATION_VERSION!r}"
        )
    else:
        verified.append("quality_evaluation_version")

    previous = catalog.get("run_configuration")
    current = build_run_configuration(args)
    if isinstance(previous, dict):
        path_fields = {"sft_adapter", "train_file", "dev_file", "probe_file", "phase_benchmark"}
        ignored = {"requested_max_drops", "resume_schema_version"}
        for name, requested in current.items():
            if name in ignored:
                continue
            if name not in previous:
                assumptions.append(f"run_configuration.{name}")
                continue
            existing = previous[name]
            equal = _normal_path(existing) == _normal_path(requested) if name in path_fields else existing == requested
            if not equal:
                mismatches.append(
                    f"run_configuration.{name}: catalog={existing!r}, requested={requested!r}"
                )
            else:
                verified.append(f"run_configuration.{name}")
    else:
        assumptions.extend(
            f"legacy_unpersisted.{name}"
            for name in (
                "train_file", "dev_file", "probe_seed", "recovery_top_k",
                "recovery_steps", "recovery_seed", "max_new_tokens", "qlora",
                "gradient_checkpointing", "skip_recovery", "data_fingerprints",
                "adapter_fingerprint", "evaluator_and_policy_fingerprints",
            )
        )

    if mismatches:
        detail = "\n  - ".join(mismatches)
        raise RuntimeError(
            "Refusing to resume Phase 2B because experiment provenance changed. "
            "Use the original settings or a new OutputRoot.\n  - " + detail
        )
    return {
        "validated_at": utc_now(),
        "verified_fields": sorted(set(verified)),
        "legacy_assumptions": sorted(set(assumptions)),
        "status": "verified_with_legacy_assumptions" if assumptions else "fully_verified",
    }


def load_candidate_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    rows = payload.get("candidates", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        raise ValueError(f"Malformed Phase 2B candidate registry: {path}")
    return list(rows)


def restore_resume_state(
    catalog: dict[str, Any],
    *,
    reference_model: Path,
    candidate_registry_path: Path,
) -> tuple[Path, list[int], list[dict[str, Any]], list[dict[str, Any]], int]:
    """Restore the exact accepted frontier parent without replaying old stages."""

    frontier = list(catalog.get("frontier") or [])
    candidates = load_candidate_registry(candidate_registry_path)
    if not frontier:
        return reference_model, [], candidates, [], 1

    last = max(frontier, key=lambda row: int(row.get("stage", 0)))
    current_model = _repo_path(last["inference_model"])
    if not current_model.exists():
        raise FileNotFoundError(
            f"Cannot resume Phase 2B: accepted frontier parent is missing: {current_model}"
        )
    dropped = [int(value) for value in last.get("dropped_original_layer_ids", [])]
    if len(dropped) != int(last.get("stage", len(dropped))):
        raise ValueError("Phase 2B frontier lineage is inconsistent with its stage number.")
    start_stage = max(int(row.get("stage", 0)) for row in frontier) + 1
    return current_model, dropped, candidates, frontier, start_stage


def combine_progression_assessments(
    cumulative: dict[str, Any],
    incremental: dict[str, Any],
) -> dict[str, Any]:
    """Require both full-reference and parent-relative opportunity to be positive."""

    cumulative_ok = bool(cumulative.get("eligible_for_progression"))
    incremental_ok = bool(incremental.get("eligible_for_progression"))
    reasons: list[str] = []
    if not cumulative_ok:
        reasons.extend(f"cumulative: {reason}" for reason in cumulative.get("rejection_reasons", []))
    if not incremental_ok:
        reasons.extend(f"incremental: {reason}" for reason in incremental.get("rejection_reasons", []))

    combined = dict(cumulative)
    combined.update(
        {
            "eligible_for_progression": cumulative_ok and incremental_ok,
            "cumulative_eligible_for_progression": cumulative_ok,
            "incremental_eligible_for_progression": incremental_ok,
            "rejection_reasons": reasons,
            "decision_rule": (
                "cumulative weight-memory gain > cumulative quality opportunity loss AND "
                "incremental parent-relative weight-memory gain > incremental quality opportunity loss"
            ),
            "incremental_weight_memory_reduction_fraction": incremental.get(
                "weight_memory_reduction_fraction", 0.0
            ),
            "incremental_quality_opportunity_loss_fraction": incremental.get(
                "quality_opportunity_loss_fraction", 0.0
            ),
            "incremental_opportunity_margin": incremental.get("opportunity_margin", 0.0),
            "cumulative_assessment": cumulative,
            "incremental_assessment": incremental,
        }
    )
    return combined


def parent_reference_from_baseline(baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation_summary": baseline["generation_summary"],
        "nll_summary": baseline["nll_summary"],
        "footprint": baseline["footprint"],
        "results": baseline["results"],
        "label": "full_reference",
    }


def parent_reference_from_frontier(row: dict[str, Any]) -> dict[str, Any]:
    evidence_dir = _repo_path(row["evidence_dir"])
    return {
        "generation_summary": row["generation_summary"],
        "nll_summary": row["nll_summary"],
        "footprint": row["footprint"],
        "results": str(evidence_dir / "generation" / "results.jsonl"),
        "label": f"frontier_stage_{row.get('stage')}",
    }


def audit_existing_frontier_incrementally(
    baseline: dict[str, Any],
    frontier: list[dict[str, Any]],
    *,
    quality_profile: str,
) -> None:
    """Backfill and verify the marginal rule on legacy accepted stages."""

    parent = parent_reference_from_baseline(baseline)
    for row in sorted(frontier, key=lambda item: int(item.get("stage", 0))):
        incremental = assess_candidate(
            baseline_generation=parent["generation_summary"],
            candidate_generation=row["generation_summary"],
            baseline_nll=parent["nll_summary"],
            candidate_nll=row["nll_summary"],
            baseline_footprint=parent["footprint"],
            candidate_footprint=row["footprint"],
            output_deviation={},
            quality_profile=quality_profile,
        )
        cumulative = row["assessment"].get("cumulative_assessment", row["assessment"])
        row["assessment"] = combine_progression_assessments(cumulative, incremental)
        if not row["assessment"]["eligible_for_progression"]:
            raise RuntimeError(
                "The existing accepted Phase 2B frontier fails the new parent-relative rule at "
                f"stage {row.get('stage')}. Do not continue from a subsidized deletion; review or "
                "rollback to the preceding accepted stage."
            )
        parent = parent_reference_from_frontier(row)


def profile_validation_dir(output_root: Path, name: str, inference_model: str) -> Path:
    """Use model-keyed profile folders so resumed runs never reuse stale summaries."""

    token = hashlib.sha256(_normal_path(inference_model).encode("utf-8")).hexdigest()[:12]
    return output_root / "profile_validation" / f"{name}_{token}" / "generation"


def relative_loss(reference: float, candidate: float) -> float:
    """Return non-negative relative loss when larger is better."""

    return max(0.0, reference - candidate) / max(abs(reference), 1e-12)


def max_category_loss(reference: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Compatibility helper backed by current continuous category scores."""

    require_quality_v4(reference, "reference category summary")
    require_quality_v4(candidate, "candidate category summary")
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
        float(assessment.get("incremental_opportunity_margin", float("-inf"))),
        float(assessment.get("weight_memory_reduction_fraction", 0.0)),
        -float(assessment.get("quality_opportunity_loss_fraction", float("inf"))),
        float(generation.get("worst_decile_mean_quality_score", 0.0)),
        float(generation.get("mean_quality_score", 0.0)),
    )



def require_explicit_v4_contracts(path: Path) -> None:
    """Reject Phase 2B probes that would force the evaluator to guess intent.

    Evaluator v4 can score mechanical integrity without a contract, but official
    architecture selection also needs case-level requirements for factual,
    context, safety, and strict task cases.  Phase 2B therefore refuses legacy
    probes rather than silently treating every reply as open-ended.
    """

    missing: list[str] = []
    invalid: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("id", f"line-{line_number}"))
            contract = row.get("response_contract")
            if not isinstance(contract, dict):
                missing.append(case_id)
            elif str(contract.get("schema_version")) != QUALITY_EVALUATION_VERSION:
                invalid.append(case_id)
    if missing or invalid:
        details = []
        if missing:
            details.append(f"missing contracts: {missing[:8]}")
        if invalid:
            details.append(f"wrong contract schema: {invalid[:8]}")
        raise ValueError(
            f"Phase 2B requires an explicit Evaluator-v4 response-contract probe ({'; '.join(details)}). "
            "Use data/phase_2_structural_reduction/probe_50_v4_3.jsonl or "
            "probe_150_thorough_v4_3.jsonl, or author a new contracted probe with "
            "scripts/common/build_v4_response_contract_benchmark.py."
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
    parent_reference: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    generation_dir = evidence_dir / "generation"
    nll_dir = evidence_dir / "teacher_forcing"
    footprint_path = evidence_dir / "footprint.json"
    deviation_json = evidence_dir / "output_deviation.json"
    deviation_report = evidence_dir / "OUTPUT_DEVIATION.md"
    parent_deviation_json = evidence_dir / "parent_output_deviation.json"
    parent_deviation_report = evidence_dir / "PARENT_OUTPUT_DEVIATION.md"

    summary_sentinel = generation_dir / "summary.json"
    if args.resume and summary_sentinel.exists():
        try:
            require_quality_v4(read_json(summary_sentinel), str(summary_sentinel))
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
    parent_results = Path(str(parent_reference["results"]))
    if _normal_path(parent_results) == _normal_path(baseline_results):
        parent_deviation_json = deviation_json
    else:
        run_if_needed(
            [
                PYTHON,
                str(COMPARE),
                "--reference",
                str(parent_results),
                "--candidate",
                str(generation_dir / "results.jsonl"),
                "--out-json",
                str(parent_deviation_json),
                "--out-report",
                str(parent_deviation_report),
            ],
            parent_deviation_json,
            args.resume,
        )

    generation_summary = read_json(generation_dir / "summary.json")
    nll_summary = read_json(nll_dir / "summary.json")
    footprint = read_json(footprint_path)
    deviation_payload = read_json(deviation_json)
    deviation_summary = deviation_payload["summary"]
    parent_deviation_payload = read_json(parent_deviation_json)
    parent_deviation_summary = parent_deviation_payload["summary"]
    cumulative_assessment = assess_candidate(
        baseline_generation=baseline_generation,
        candidate_generation=generation_summary,
        baseline_nll=baseline_nll,
        candidate_nll=nll_summary,
        baseline_footprint=baseline_footprint,
        candidate_footprint=footprint,
        output_deviation=deviation_summary,
        quality_profile=args.quality_profile,
    )
    incremental_assessment = assess_candidate(
        baseline_generation=parent_reference["generation_summary"],
        candidate_generation=generation_summary,
        baseline_nll=parent_reference["nll_summary"],
        candidate_nll=nll_summary,
        baseline_footprint=parent_reference["footprint"],
        candidate_footprint=footprint,
        output_deviation=parent_deviation_summary,
        quality_profile=args.quality_profile,
    )
    assessment = combine_progression_assessments(cumulative_assessment, incremental_assessment)
    payload = {
        "model_id": model_id,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "generation_summary": generation_summary,
        "nll_summary": nll_summary,
        "footprint": footprint,
        "output_deviation_summary": deviation_summary,
        "parent_output_deviation_summary": parent_deviation_summary,
        "parent_reference_label": parent_reference.get("label"),
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



def recover_rows(
    *,
    raw_rows: list[dict[str, Any]],
    stage: int,
    evidence_root: Path,
    baseline_generation_dir: Path,
    baseline_generation: dict[str, Any],
    baseline_nll: dict[str, Any],
    baseline_footprint: dict[str, Any],
    parent_reference: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Recovery-train and evaluate the requested raw deletion branches."""

    recovered: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        variant_id = raw_row["variant_id"].replace("_raw", "_recovered")
        candidate_evidence = evidence_root / f"stage_{stage:02d}" / variant_id
        prepare_variant_evidence(
            candidate_evidence,
            build_variant_provenance(
                stage=stage,
                variant_id=variant_id,
                variant_kind="deletion + recovery SFT",
                parent_model=raw_row["parent_model"],
                dropped_original_layer_id=int(raw_row["dropped_original_layer_id"]),
                retained_original_layer_ids=[int(value) for value in raw_row["retained_original_layer_ids"]],
                args=args,
            ),
        )
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
            parent_reference=parent_reference,
            args=args,
        )
        recovered.append(
            {
                **metrics,
                "variant_id": variant_id,
                "variant_kind": "deletion + recovery SFT",
                "stage": stage,
                "parent_model": raw_row["parent_model"],
                "raw_model": raw_row["raw_model"],
                "adapter_path": str(adapter),
                "drop_current_layer_id": raw_row["drop_current_layer_id"],
                "dropped_original_layer_id": raw_row["dropped_original_layer_id"],
                "retained_original_layer_ids": raw_row["retained_original_layer_ids"],
            }
        )
    return recovered


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
    termination: dict[str, Any],
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
            f"{float(a.get('incremental_weight_memory_reduction_fraction', 0.0)):.2%} | "
            f"{float(a.get('incremental_quality_opportunity_loss_fraction', 0.0)):.2%} | "
            f"{float(a.get('incremental_opportunity_margin', 0.0)):+.2%} | "
            f"{format_optional_percent(a.get('peak_evaluation_vram_reduction_fraction'))} | "
            f"{format_optional_percent(a.get('mean_generation_latency_reduction_fraction'))} | "
            f"{row['variant_kind']} |"
        )
    if not stage_lines:
        stage_lines.append(
            "| 0 | [] | - | - | - | - | - | - | - | - | - | - | "
            "no proportionally worthwhile reduction |"
        )

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
- Recovery first evaluates the strongest {args.recovery_top_k} raw candidates per stage.
- If focused recovery finds no eligible candidate, every remaining raw branch is recovery-tested once before the frontier is declared terminal.
- Resume continues directly from the last accepted frontier model; completed stages are not reconstructed.

## Stage 1 individual-layer results

| Dropped original layer | Variant | Quality | Weight-memory gain | Quality opportunity loss | Opportunity margin | Eval VRAM change | Generation latency change | Target NLL | Profile | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(first_lines)}

## Accepted progressive frontier

| Stage | Original layers dropped so far | Remaining layers | Quality | Cumulative memory gain | Cumulative quality loss | Cumulative margin | Incremental memory gain | Incremental quality loss | Incremental margin | Eval VRAM change | Generation latency change | Selected form |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(stage_lines)}

## Terminal condition

- Reason: `{termination.get('reason')}`
- Terminal stage attempted: `{termination.get('terminal_stage_attempted')}`
- Accepted frontier stages: **{termination.get('accepted_frontier_stages', len(stages))}**
- Exhaustive recovery triggered at terminal/last stage: **{termination.get('exhaustive_recovery_triggered', False)}**

## Retained deployment/training profiles

| Profile | Remaining layers | Weight-memory gain | Quality opportunity loss | Opportunity margin | Model |
|---|---:|---:|---:|---:|---|
{chr(10).join(profile_lines)}

## Decision rule

A technically comparable candidate enters the frontier only when **both** are true:

1. Its cumulative loaded weight-tensor memory reduction from the full reference is strictly greater than its cumulative combined quality opportunity loss.
2. Its incremental loaded weight-tensor memory reduction from the immediate accepted parent is strictly greater than its incremental combined quality opportunity loss.

The second condition prevents a later damaging deletion from being subsidized by
the large positive margin accumulated by earlier successful deletions. Mean
quality, tail quality, category behavior, comprehensibility, task adequacy,
logic, safety, context, completeness, content anchors, issue cost,
severe-collapse rate, and target NLL contribute continuously according to the
selected profile. None is an independent absolute floor or issue-count veto.
Latency and evaluation VRAM remain secondary measured evidence.

Human review and later phase controls remain mandatory. Technical invalidity—such
as missing evidence, a corrupt checkpoint, mismatched evaluator versions, or
non-reproducible provenance—still blocks comparison because no valid opportunity
calculation exists.

## Workflow handoff

- `full_quality` preserves the original all-layer SFT reference.
- `balanced` selects the eligible frontier point with the largest cumulative opportunity margin.
- `min_memory` selects the eligible frontier point with the largest weight-memory gain.
- Phase 2C should begin only after a reduced profile receives explicit human approval.
- Phase 4 soups must use ingredients from the same selected architecture profile.
"""


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output_root / "model_catalog.json"
    candidate_registry_path = args.output_root / "candidate_registry.json"
    existing_catalog: dict[str, Any] | None = None
    resume_validation: dict[str, Any] | None = None

    if catalog_path.exists() and not args.resume:
        raise FileExistsError(
            f"Phase 2B output already contains a completed catalog: {catalog_path}. "
            "Use --resume to extend it or choose a new --output-root for a fresh run."
        )

    if args.resume and catalog_path.exists():
        existing_catalog = read_json(catalog_path)
        resume_validation = validate_resume_catalog(existing_catalog, args)
        if resume_validation["legacy_assumptions"]:
            print(
                "[resume warning] Legacy Phase 2B catalog lacks some settings that cannot be "
                "retroactively verified. They are recorded as explicit assumptions in the new catalog."
            )
        write_json(args.output_root / "resume_validation.json", resume_validation)

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

    require_explicit_v4_contracts(args.probe_file)

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
            require_quality_v4(read_json(baseline_generation_dir / "summary.json"), "Phase 2B reference evaluation")
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
    require_quality_v4(baseline_generation, "Phase 2B reference evaluation")
    assert_valid_reference(baseline_generation, label="Phase 2B full reference")
    baseline_nll = read_json(baseline_nll_dir / "summary.json")
    baseline_footprint = read_json(baseline_footprint_path)
    baseline = {
        "generation_summary": baseline_generation,
        "nll_summary": baseline_nll,
        "footprint": baseline_footprint,
        "results": str(baseline_generation_dir / "results.jsonl"),
    }

    if existing_catalog is not None:
        current_model, dropped_original, all_candidate_rows, frontier_rows, start_stage = restore_resume_state(
            existing_catalog,
            reference_model=reference_model,
            candidate_registry_path=candidate_registry_path,
        )
        audit_existing_frontier_incrementally(
            baseline,
            frontier_rows,
            quality_profile=args.quality_profile,
        )
        parent_reference = (
            parent_reference_from_frontier(frontier_rows[-1])
            if frontier_rows
            else parent_reference_from_baseline(baseline)
        )
        print(
            f"[resume] continuing from accepted stage {start_stage - 1}; "
            f"next stage is {start_stage}; parent={current_model}"
        )
    else:
        current_model = reference_model
        dropped_original = []
        all_candidate_rows = []
        frontier_rows = []
        start_stage = 1
        parent_reference = parent_reference_from_baseline(baseline)

    work_root = args.output_root / "work_models"
    evidence_root = args.output_root / "evidence"
    frontier_root = args.output_root / "frontier"
    prior_termination = (existing_catalog or {}).get("termination") or {}
    termination: dict[str, Any] = {
        "reason": "max_drops_reached",
        "terminal_stage_attempted": None,
        "accepted_frontier_stages": len(frontier_rows),
        "exhaustive_recovery_triggered": False,
    }

    skip_stage_loop = False
    if prior_termination.get("reason") in TERMINAL_NO_PROGRESS_REASONS:
        termination = dict(prior_termination)
        skip_stage_loop = True
        print(
            "[resume] the catalog already records a terminal no-progress condition; "
            "no stage will be replayed."
        )

    if not skip_stage_loop:
        for stage in range(start_stage, args.max_drops + 1):
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(current_model)
            current_count = int(config.num_hidden_layers)
            if current_count <= 2:
                termination = {
                    "reason": "minimum_layer_guard_reached",
                    "terminal_stage_attempted": stage,
                    "accepted_frontier_stages": len(frontier_rows),
                    "exhaustive_recovery_triggered": False,
                }
                break
            original_ids = lineage_for_model(current_model, current_count)
            stage_raw_rows: list[dict[str, Any]] = []

            for current_index, original_id in enumerate(original_ids):
                variant_id = f"stage_{stage:02d}_drop_original_{original_id:02d}_raw"
                raw_model = work_root / variant_id
                candidate_evidence = evidence_root / f"stage_{stage:02d}" / variant_id
                retained_ids = [value for value in original_ids if value != original_id]
                stale_lineage = prepare_variant_evidence(
                    candidate_evidence,
                    build_variant_provenance(
                        stage=stage,
                        variant_id=variant_id,
                        variant_kind="raw deletion",
                        parent_model=current_model,
                        dropped_original_layer_id=original_id,
                        retained_original_layer_ids=retained_ids,
                        args=args,
                    ),
                )
                if stale_lineage and raw_model.exists():
                    shutil.rmtree(raw_model, ignore_errors=True)
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
                    parent_reference=parent_reference,
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
                    "retained_original_layer_ids": retained_ids,
                }
                stage_raw_rows.append(row)
                all_candidate_rows.append(row)

            recovered_rows: list[dict[str, Any]] = []
            focused_targets: list[dict[str, Any]] = []
            if not args.skip_recovery and args.recovery_steps > 0 and args.recovery_top_k > 0:
                focused_targets = sorted(stage_raw_rows, key=ranking_key, reverse=True)[: args.recovery_top_k]
                recovered_rows.extend(
                    recover_rows(
                        raw_rows=focused_targets,
                        stage=stage,
                        evidence_root=evidence_root,
                        baseline_generation_dir=baseline_generation_dir,
                        baseline_generation=baseline_generation,
                        baseline_nll=baseline_nll,
                        baseline_footprint=baseline_footprint,
                        parent_reference=parent_reference,
                        args=args,
                    )
                )
                all_candidate_rows.extend(recovered_rows)

            stage_options = stage_raw_rows + recovered_rows
            eligible = [row for row in stage_options if row["assessment"]["eligible_for_progression"]]
            exhaustive_triggered = False

            if (
                not eligible
                and not args.skip_recovery
                and args.recovery_steps > 0
                and args.recovery_top_k > 0
                and len(focused_targets) < len(stage_raw_rows)
            ):
                recovered_original_ids = {
                    int(row["dropped_original_layer_id"]) for row in focused_targets
                }
                remaining_targets = [
                    row for row in stage_raw_rows
                    if int(row["dropped_original_layer_id"]) not in recovered_original_ids
                ]
                if remaining_targets:
                    exhaustive_triggered = True
                    print(
                        f"[stage {stage}] focused recovery found no eligible deletion; "
                        f"recovery-testing all {len(remaining_targets)} remaining branches before stopping."
                    )
                    exhaustive_rows = recover_rows(
                        raw_rows=remaining_targets,
                        stage=stage,
                        evidence_root=evidence_root,
                        baseline_generation_dir=baseline_generation_dir,
                        baseline_generation=baseline_generation,
                        baseline_nll=baseline_nll,
                        baseline_footprint=baseline_footprint,
                        parent_reference=parent_reference,
                        args=args,
                    )
                    recovered_rows.extend(exhaustive_rows)
                    all_candidate_rows.extend(exhaustive_rows)
                    stage_options.extend(exhaustive_rows)
                    eligible = [
                        row for row in stage_options
                        if row["assessment"]["eligible_for_progression"]
                    ]

            if not eligible:
                reason = (
                    "no_marginally_worthwhile_deletion_recovery_disabled"
                    if args.skip_recovery or args.recovery_steps <= 0 or args.recovery_top_k <= 0
                    else "no_marginally_worthwhile_deletion_after_exhaustive_recovery"
                )
                termination = {
                    "reason": reason,
                    "terminal_stage_attempted": stage,
                    "accepted_frontier_stages": len(frontier_rows),
                    "exhaustive_recovery_triggered": exhaustive_triggered,
                    "raw_candidates_tested": len(stage_raw_rows),
                    "recovered_candidates_tested": len(recovered_rows),
                }
                print(
                    f"Stopping at stage {stage}: no candidate satisfies both the cumulative "
                    "and parent-relative opportunity rules."
                )
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
                "exhaustive_recovery_triggered": exhaustive_triggered,
            }
            write_json(stage_dir / "stage_result.json", frontier_row)
            frontier_rows.append(frontier_row)
            current_model = stage_model
            parent_reference = parent_reference_from_frontier(frontier_row)
            termination = {
                "reason": "max_drops_reached",
                "terminal_stage_attempted": stage,
                "accepted_frontier_stages": len(frontier_rows),
                "exhaustive_recovery_triggered": exhaustive_triggered,
            }

            if not args.keep_candidate_checkpoints and work_root.exists():
                shutil.rmtree(work_root)

    # A resumed run can reconstruct a partially completed terminal stage. Keep
    # only the latest durable row for each stage/variant pair.
    deduplicated: dict[tuple[int, str], dict[str, Any]] = {}
    for row in all_candidate_rows:
        deduplicated[(int(row.get("stage", 0)), str(row.get("variant_id", "")))] = row
    all_candidate_rows = list(deduplicated.values())

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
            "incremental_weight_memory_reduction_fraction": 0.0,
            "incremental_quality_opportunity_loss_fraction": 0.0,
            "incremental_opportunity_margin": 0.0,
        },
    }
    profiles = select_profiles(baseline_profile, frontier_rows)
    for name, profile in profiles.items():
        profile["profile"] = name

    profile_eval_cache: dict[str, str] = {}
    for name, profile in profiles.items():
        inference_model = str(profile["inference_model"])
        cache_key = _normal_path(inference_model)
        if cache_key not in profile_eval_cache:
            profile_eval_dir = profile_validation_dir(args.output_root, name, inference_model)
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
            profile_eval_cache[cache_key] = str(profile_eval_dir / "summary.json")
        profile["evaluation_summary"] = profile_eval_cache[cache_key]
        profile["phase_benchmark"] = str(args.phase_benchmark)

    history = list((existing_catalog or {}).get("resume_history") or [])
    if existing_catalog is not None:
        history.append(
            {
                "resumed_at": utc_now(),
                "accepted_stages_before": start_stage - 1,
                "accepted_stages_after": len(frontier_rows),
                "resume_parent": str(_repo_path((existing_catalog.get("frontier") or [{}])[-1].get("inference_model", reference_model))),
                "requested_max_drops": int(args.max_drops),
                "validation": resume_validation,
            }
        )

    catalog = {
        "manifest_version": 2,
        "status": "PENDING_HUMAN_REVIEW",
        "source_model_id": args.model_id,
        "source_sft_adapter": str(args.sft_adapter),
        "probe_file": str(args.probe_file),
        "phase_benchmark": str(args.phase_benchmark),
        "probe_case_count": args.probe_limit,
        "baseline": baseline,
        "run_configuration": build_run_configuration(args),
        "resume_validation": resume_validation,
        "resume_history": history,
        "opportunity_policy": {
            "decision_rule": (
                "cumulative loaded weight-tensor reduction > cumulative quality opportunity loss AND "
                "incremental parent-relative reduction > incremental quality opportunity loss"
            ),
            "quality_profile": args.quality_profile,
            "absolute_promotion_thresholds": "prohibited",
            "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        },
        "frontier": frontier_rows,
        "profiles": profiles,
        "termination": termination,
        "all_candidate_registry": str(candidate_registry_path),
        "gate_report": str(args.gate_report),
        "downstream_rule": (
            "Phase 3 may use full_quality, balanced, or min_memory only after "
            "the non-full profile receives explicit Phase 2B human approval."
        ),
    }
    write_json(candidate_registry_path, {"candidates": all_candidate_rows})
    write_json(catalog_path, catalog)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_report(
            baseline=baseline,
            stages=frontier_rows,
            candidates=all_candidate_rows,
            profiles=profiles,
            termination=termination,
            args=args,
        ),
        encoding="utf-8",
    )
    gate = f"""# PHASE 2B STOP GATE

## Automated status

`PENDING_HUMAN_REVIEW`

- Model catalog: `{catalog_path}`
- Progressive report: `{args.report}`
- Accepted frontier stages: {len(frontier_rows)}
- Termination reason: `{termination.get('reason')}`
- Terminal stage attempted: `{termination.get('terminal_stage_attempted')}`
- Recommended balanced profile: `{profiles['balanced']['inference_model']}`
- Recommended minimum-memory profile: `{profiles['min_memory']['inference_model']}`

## Required human decision

Review the configured probe side-by-side outputs for every accepted stage and the lowest-scoring rejected neighbors. A new frontier stage must be worthwhile both cumulatively against the full reference and incrementally against its immediate parent. When focused recovery cannot find an eligible candidate, the runner recovery-tests every remaining raw branch before declaring the frontier terminal. Human review remains required before downstream use.

- Human decision: `PENDING`
- Reviewer:
- Date:
- Approved profiles: `full_quality`
- Notes:
"""
    args.gate_report.parent.mkdir(parents=True, exist_ok=True)
    args.gate_report.write_text(gate, encoding="utf-8")
    print(catalog_path)
    print(args.report)
    print(args.gate_report)


if __name__ == "__main__":
    main()
