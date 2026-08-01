from __future__ import annotations

"""Recommend a Phase 4B reference by structural gain versus quality cost."""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, write_json
from scripts.common.opportunity_evaluation import evaluate_opportunity, pareto_frontier, resource_reduction
from scripts.common.quality_evaluation import require_quality_v3, summary_quality_score
from scripts.common.quality_gate import assert_valid_reference, quality_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend a Phase 4B candidate by relative opportunity margin.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/phase_4b_structural_reduction/candidate_recommendation.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/PHASE_4B_CANDIDATE_RECOMMENDATION.md")
    parser.add_argument("--quality-profile", choices=("balanced", "logic_oriented", "conversational_permissive"), default="balanced")
    # Historical fixed floors are accepted but ignored.
    for flag in (
        "--max-mean-quality-loss", "--max-mean-score-loss", "--max-quality-index-loss",
        "--max-worst-decile-loss", "--max-category-quality-loss", "--max-category-score-loss",
        "--max-semantic-pass-loss", "--max-reference-alignment-loss", "--min-quality-pass-rate",
        "--min-intelligibility-pass-rate", "--min-semantic-pass-rate",
        "--min-complete-utterance-rate", "--max-major-issue-cases", "--min-parameter-reduction",
    ):
        parser.add_argument(flag, type=float, default=None, help=argparse.SUPPRESS)
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


def assess_candidate(candidate: dict[str, Any], dense: dict[str, Any], *, quality_profile: str = "balanced", **deprecated: Any) -> dict[str, Any]:
    summary = candidate["summary_data"]
    dense_summary = dense["summary_data"]
    require_quality_v3(summary, f"Phase 4B candidate {candidate['name']}")
    dense_params = float(dense["footprint_data"]["parameters"]["total"])
    params = float(candidate["footprint_data"]["parameters"]["total"])
    reduction = resource_reduction(dense_params, params)
    if candidate["name"] == dense["name"]:
        opportunity = evaluate_opportunity(
            dense_summary, summary, resource_gain=0.0,
            resource_name="structural_parameter_reduction", profile=quality_profile,
        )
        opportunity.update({
            "eligible": True,
            "decision": "DENSE_REFERENCE_FALLBACK",
            "decision_explanation": "dense reference remains the valid zero-reduction fallback",
            "opportunity_margin": 0.0,
        })
    else:
        opportunity = evaluate_opportunity(
            dense_summary,
            summary,
            resource_gain=reduction,
            resource_name="structural_parameter_reduction",
            profile=quality_profile,
        )
    return {
        "name": candidate["name"],
        "model": candidate["model"],
        **opportunity,
        "parameter_reduction_fraction": float(reduction or 0.0),
        "checkpoint_directory_bytes": candidate["footprint_data"].get("local_directory_bytes"),
        "mean_quality_score": summary_quality_score(summary),
        "quality_index": quality_index(summary),
        "mean_quality_delta": summary_quality_score(summary) - summary_quality_score(dense_summary),
        "worst_decile_quality_score": summary.get("worst_decile_mean_quality_score"),
        "diagnostic_issue_free_rate": summary.get("quality_pass_rate"),
        "severe_issue_case_rate": summary.get("severe_issue_case_rate"),
        "mean_issue_cost": summary.get("mean_issue_cost"),
        "peak_vram_gb": summary.get("peak_vram_gb"),
        "mean_generation_seconds": summary.get("mean_generation_seconds"),
        "deprecated_threshold_arguments_ignored": sorted(deprecated),
    }


def recommend(assessments: list[dict[str, Any]]) -> dict[str, Any]:
    reduced = [row for row in assessments if row["name"] != "dense_reference" and row.get("eligible")]
    if not reduced:
        return next(row for row in assessments if row["name"] == "dense_reference")
    return max(
        reduced,
        key=lambda row: (
            float(row.get("opportunity_margin", float("-inf"))),
            float(row.get("resource_gain_fraction", 0.0)),
            -float(row.get("quality_opportunity_loss_fraction", math.inf)),
        ),
    )


def render_report(registry: Path, profile: str, assessments: list[dict[str, Any]], recommendation: dict[str, Any]) -> str:
    frontier_keys = {
        (row["name"], float(row.get("opportunity_margin", 0.0)))
        for row in pareto_frontier(assessments)
    }
    rows = []
    for row in sorted(assessments, key=lambda item: float(item.get("opportunity_margin", 0.0)), reverse=True):
        pareto = (row["name"], float(row.get("opportunity_margin", 0.0))) in frontier_keys
        rows.append(
            f"| {row['name']} | {row['resource_gain_fraction']:.3%} | "
            f"{row['quality_opportunity_loss_fraction']:.3%} | {row['opportunity_margin']:+.3%} | "
            f"{row['mean_quality_score']:.4f} | {'YES' if pareto else 'NO'} | "
            f"{'WORTHWHILE' if row['eligible'] else 'NOT WORTHWHILE'} |"
        )
    return f"""# PHASE 4B CANDIDATE RECOMMENDATION

## Policy

Structural candidates are evaluated by proportional parameter-memory reduction versus combined proportional quality opportunity loss. No absolute score or reduction floor applies.

- Candidate registry: `{registry}`
- Quality evaluator: `3.0`
- Quality profile: `{profile}`

| Candidate | Resource gain | Quality opportunity loss | Opportunity margin | Mean quality | Pareto | Decision |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## Automated recommendation

- Candidate: `{recommendation['name']}`
- Model: `{recommendation['model']}`
- Opportunity margin: `{recommendation['opportunity_margin']:+.3%}`
- Resource gain: `{recommendation['resource_gain_fraction']:.3%}`
- Quality opportunity loss: `{recommendation['quality_opportunity_loss_fraction']:.3%}`

This remains a recommendation. Human review may choose the dense fallback or another Pareto candidate.
"""


def main() -> None:
    args = parse_args()
    registry = read_json(args.registry)
    candidates = [load_candidate(row) for row in registry["candidates"]]
    dense = next(row for row in candidates if row["name"] == registry["dense_reference_name"])
    assert_valid_reference(dense["summary_data"], label="Phase 4B dense reference")
    deprecated = {
        key: value for key, value in vars(args).items()
        if key not in {"registry", "out", "report", "quality_profile"} and value is not None
    }
    assessments = [assess_candidate(row, dense, quality_profile=args.quality_profile, **deprecated) for row in candidates]
    selected = recommend(assessments)
    payload = {
        "schema_version": 3,
        "quality_evaluation_version": "3.0",
        "opportunity_policy_version": "1.0",
        "registry": str(args.registry),
        "quality_profile": args.quality_profile,
        "deprecated_threshold_arguments_ignored": sorted(deprecated),
        "assessments": assessments,
        "recommended_candidate": selected,
        "human_review_required": True,
    }
    write_json(args.out, payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(args.registry, args.quality_profile, assessments, selected), encoding="utf-8")
    print(args.out)
    print(args.report)


if __name__ == "__main__":
    main()
