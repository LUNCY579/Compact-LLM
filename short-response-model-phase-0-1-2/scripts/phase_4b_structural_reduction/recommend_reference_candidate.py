# Purpose: Applies conservative, tail-aware quality/size floors to recommend a downstream reference.
from __future__ import annotations

"""Recommend a deployable Phase 4B reference from measured candidate results.

The recommendation never grants the human stop gate. Evaluator-v2 evidence is
mandatory so structural savings cannot hide gibberish, repetition collapse, or
rare catastrophic cases behind an acceptable mean.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, write_json
from scripts.common.quality_gate import (
    assert_valid_reference,
    compare_quality_summaries,
    quality_index,
)
from scripts.common.quality_evaluation import require_quality_v2, summary_quality_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply quality-v2 floors and recommend a Phase 4B reference.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/phase_4b_structural_reduction/candidate_recommendation.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/PHASE_4B_CANDIDATE_RECOMMENDATION.md")
    # Keep the historical option name as a compatibility alias.
    parser.add_argument("--max-mean-quality-loss", "--max-mean-score-loss", dest="max_mean_quality_loss", type=float, default=0.01)
    parser.add_argument("--max-quality-index-loss", type=float, default=0.015)
    parser.add_argument("--max-worst-decile-loss", type=float, default=0.03)
    parser.add_argument("--max-category-quality-loss", "--max-category-score-loss", dest="max_category_quality_loss", type=float, default=0.04)
    parser.add_argument("--max-semantic-pass-loss", type=float, default=0.01)
    parser.add_argument("--max-reference-alignment-loss", type=float, default=0.05)
    parser.add_argument("--min-quality-pass-rate", type=float, default=0.98)
    parser.add_argument("--min-intelligibility-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-semantic-pass-rate", type=float, default=0.98)
    parser.add_argument("--min-complete-utterance-rate", type=float, default=0.99)
    parser.add_argument("--max-major-issue-cases", type=int, default=0)
    parser.add_argument("--min-parameter-reduction", type=float, default=0.03)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        **candidate,
        "summary_data": read_json(project_path(candidate["summary"])),
        "footprint_data": read_json(project_path(candidate["footprint"])),
    }


def assess_candidate(candidate: dict[str, Any], dense: dict[str, Any], **thresholds: Any) -> dict[str, Any]:
    summary = candidate["summary_data"]
    dense_summary = dense["summary_data"]
    require_quality_v2(summary, f"Phase 4B candidate {candidate['name']}")
    footprint = candidate["footprint_data"]
    dense_footprint = dense["footprint_data"]
    dense_params = float(dense_footprint["parameters"]["total"])
    params = float(footprint["parameters"]["total"])
    reduction = 1.0 - params / dense_params

    comparison = compare_quality_summaries(
        dense_summary,
        summary,
        max_mean_quality_loss=float(thresholds["max_mean_quality_loss"]),
        max_quality_index_loss=float(thresholds["max_quality_index_loss"]),
        max_worst_decile_loss=float(thresholds["max_worst_decile_loss"]),
        max_category_quality_loss_allowed=float(thresholds["max_category_quality_loss"]),
        max_semantic_pass_loss=float(thresholds["max_semantic_pass_loss"]),
        max_reference_alignment_loss=float(thresholds["max_reference_alignment_loss"]),
        min_quality_pass_rate=float(thresholds["min_quality_pass_rate"]),
        min_intelligibility_pass_rate=float(thresholds["min_intelligibility_pass_rate"]),
        min_semantic_pass_rate=float(thresholds["min_semantic_pass_rate"]),
        min_complete_utterance_rate=float(thresholds["min_complete_utterance_rate"]),
        max_hard_failure_cases=0,
        max_critical_failures=0,
        max_major_issue_cases=int(thresholds["max_major_issue_cases"]),
    )
    reasons = list(comparison["reasons"])
    if candidate["name"] != dense["name"] and reduction < float(thresholds["min_parameter_reduction"]):
        reasons.append("parameter reduction is too small to justify structural risk")

    return {
        "name": candidate["name"],
        "model": candidate["model"],
        "eligible": not reasons,
        "reasons": reasons,
        "parameter_reduction_fraction": reduction,
        "checkpoint_directory_bytes": footprint.get("local_directory_bytes"),
        "mean_quality_score": summary_quality_score(summary),
        "quality_index": quality_index(summary),
        "mean_quality_delta": summary_quality_score(summary) - summary_quality_score(dense_summary),
        "worst_decile_quality_score": summary.get("worst_decile_mean_quality_score"),
        "quality_pass_rate": summary.get("quality_pass_rate"),
        "intelligibility_pass_rate": summary.get("intelligibility_pass_rate"),
        "semantic_pass_rate": summary.get("semantic_pass_rate"),
        "hard_failure_case_count": summary.get("hard_failure_case_count"),
        "major_case_count": summary.get("major_case_count"),
        "comparison": comparison,
        "peak_vram_gb": summary.get("peak_vram_gb"),
        "mean_generation_seconds": summary.get("mean_generation_seconds"),
    }


def recommend(assessments: list[dict[str, Any]]) -> dict[str, Any]:
    eligible_reduced = [row for row in assessments if row["eligible"] and row["name"] != "dense_reference"]
    if not eligible_reduced:
        return next(row for row in assessments if row["name"] == "dense_reference")

    def ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        vram = row.get("peak_vram_gb")
        latency = row.get("mean_generation_seconds")
        return (
            float(row["parameter_reduction_fraction"]),
            float(row.get("quality_index", row.get("mean_quality_score", 0.0))),
            float(row.get("worst_decile_quality_score", row.get("mean_quality_score", 0.0)) or 0.0),
            -float(vram) if vram is not None and math.isfinite(float(vram)) else -math.inf,
            -float(latency) if latency is not None and math.isfinite(float(latency)) else -math.inf,
        )

    return max(eligible_reduced, key=ranking_key)


def render_report(*, registry: Path, thresholds: dict[str, Any], assessments: list[dict[str, Any]], recommendation: dict[str, Any]) -> str:
    rows: list[str] = []
    for row in assessments:
        reasons = "; ".join(row["reasons"]) if row["reasons"] else "-"
        rows.append(
            f"| {row['name']} | {'YES' if row['eligible'] else 'NO'} | "
            f"{row['parameter_reduction_fraction']:.3%} | {row['mean_quality_score']:.4f} | "
            f"{row['quality_index']:.4f} | {float(row.get('worst_decile_quality_score') or 0.0):.4f} | "
            f"{float(row.get('quality_pass_rate') or 0.0):.4f} | "
            f"{int(row.get('hard_failure_case_count') or 0)} | {int(row.get('major_case_count') or 0)} | {reasons} |"
        )
    return f"""# PHASE 4B CANDIDATE RECOMMENDATION

## Status

`RECOMMENDATION ONLY — HUMAN REVIEW REQUIRED`

- Candidate registry: `{registry}`
- Quality evaluator: `2.0`
- Thresholds: `{thresholds}`

## Candidate decisions

| Candidate | Eligible | Parameter reduction | Mean quality | Quality index | Worst decile | Quality pass | Hard cases | Major cases | Rejection reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## Automated recommendation

- Candidate: `{recommendation['name']}`
- Model: `{recommendation['model']}`
- Parameter reduction: `{recommendation['parameter_reduction_fraction']:.3%}`
- Mean quality: `{recommendation['mean_quality_score']:.4f}`
- Quality index: `{recommendation['quality_index']:.4f}`

The recommendation cannot approve deployment. Review every hard/major/low-tail case,
safety-sensitive prompts, exact replies, reload behavior, repeated runtime measurements,
and maintenance cost. Keeping `dense_reference` remains a successful result.
"""


def main() -> None:
    args = parse_args()
    registry = read_json(args.registry)
    candidates = [load_candidate(row) for row in registry["candidates"]]
    dense = next(row for row in candidates if row["name"] == registry["dense_reference_name"])
    assert_valid_reference(dense["summary_data"], label="Phase 4B dense reference")
    thresholds = {
        "max_mean_quality_loss": args.max_mean_quality_loss,
        "max_quality_index_loss": args.max_quality_index_loss,
        "max_worst_decile_loss": args.max_worst_decile_loss,
        "max_category_quality_loss": args.max_category_quality_loss,
        "max_semantic_pass_loss": args.max_semantic_pass_loss,
        "max_reference_alignment_loss": args.max_reference_alignment_loss,
        "min_quality_pass_rate": args.min_quality_pass_rate,
        "min_intelligibility_pass_rate": args.min_intelligibility_pass_rate,
        "min_semantic_pass_rate": args.min_semantic_pass_rate,
        "min_complete_utterance_rate": args.min_complete_utterance_rate,
        "max_major_issue_cases": args.max_major_issue_cases,
        "min_parameter_reduction": args.min_parameter_reduction,
    }
    assessments = [assess_candidate(row, dense, **thresholds) for row in candidates]
    selected = recommend(assessments)
    payload = {
        "schema_version": 2,
        "quality_evaluation_version": "2.0",
        "registry": str(args.registry),
        "thresholds": thresholds,
        "assessments": assessments,
        "recommended_candidate": selected,
        "human_review_required": True,
    }
    write_json(args.out, payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(registry=args.registry, thresholds=thresholds, assessments=assessments, recommendation=selected), encoding="utf-8")
    print(args.out)
    print(args.report)


if __name__ == "__main__":
    main()
