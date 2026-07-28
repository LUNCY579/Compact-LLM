from __future__ import annotations

"""Recommend a deployable Phase 4B reference from measured candidate results.

The recommendation is intentionally conservative and never grants the human
stop gate. It checks practical quality floors, then prefers the largest proven
parameter reduction. If no reduced model passes, keeping the dense Phase 4
reference is a successful and usable outcome.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply practical quality floors and recommend a Phase 4B reference."
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "artifacts"
        / "phase_4b_structural_reduction"
        / "candidate_recommendation.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "PHASE_4B_CANDIDATE_RECOMMENDATION.md",
    )
    parser.add_argument("--max-mean-score-loss", type=float, default=0.01)
    parser.add_argument("--max-category-score-loss", type=float, default=0.02)
    parser.add_argument("--max-semantic-pass-loss", type=float, default=0.01)
    parser.add_argument("--min-complete-utterance-rate", type=float, default=0.99)
    parser.add_argument("--min-parameter-reduction", type=float, default=0.03)
    return parser.parse_args()


def project_path(value: str) -> Path:
    """Resolve registry paths relative to the repository root."""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Attach complete summary and footprint data to one registry entry."""

    return {
        **candidate,
        "summary_data": read_json(project_path(candidate["summary"])),
        "footprint_data": read_json(project_path(candidate["footprint"])),
    }


def assess_candidate(
    candidate: dict[str, Any],
    dense: dict[str, Any],
    *,
    max_mean_score_loss: float,
    max_category_score_loss: float,
    max_semantic_pass_loss: float,
    min_complete_utterance_rate: float,
    min_parameter_reduction: float,
) -> dict[str, Any]:
    """Return eligibility, reasons, and deployment-oriented deltas."""

    summary = candidate["summary_data"]
    dense_summary = dense["summary_data"]
    footprint = candidate["footprint_data"]
    dense_footprint = dense["footprint_data"]

    dense_params = float(dense_footprint["parameters"]["total"])
    params = float(footprint["parameters"]["total"])
    reduction = 1.0 - params / dense_params

    reasons: list[str] = []
    if int(summary.get("critical_failure_count", 0)) != 0:
        reasons.append("critical semantic failures are nonzero")
    if float(summary["mean_heuristic_score"]) < (
        float(dense_summary["mean_heuristic_score"]) - max_mean_score_loss
    ):
        reasons.append("mean quality score exceeds the allowed loss")
    if float(summary["semantic_pass_rate"]) < (
        float(dense_summary["semantic_pass_rate"]) - max_semantic_pass_loss
    ):
        reasons.append("semantic pass rate exceeds the allowed loss")
    if float(summary["complete_utterance_rate"]) < min_complete_utterance_rate:
        reasons.append("complete-utterance rate is below the configured floor")

    category_deltas: dict[str, float] = {}
    for category, dense_score in dense_summary.get("category_scores", {}).items():
        candidate_score = summary.get("category_scores", {}).get(category)
        if candidate_score is None:
            reasons.append(f"missing category score: {category}")
            continue
        delta = float(candidate_score) - float(dense_score)
        category_deltas[category] = delta
        if delta < -max_category_score_loss:
            reasons.append(f"category regression exceeds limit: {category}")

    if candidate["name"] != dense["name"] and reduction < min_parameter_reduction:
        reasons.append("parameter reduction is too small to justify structural risk")

    return {
        "name": candidate["name"],
        "model": candidate["model"],
        "eligible": not reasons,
        "reasons": reasons,
        "parameter_reduction_fraction": reduction,
        "checkpoint_directory_bytes": footprint.get("local_directory_bytes"),
        "mean_score_delta": float(summary["mean_heuristic_score"])
        - float(dense_summary["mean_heuristic_score"]),
        "semantic_pass_delta": float(summary["semantic_pass_rate"])
        - float(dense_summary["semantic_pass_rate"]),
        "category_score_deltas": category_deltas,
        "peak_vram_gb": summary.get("peak_vram_gb"),
        "mean_generation_seconds": summary.get("mean_generation_seconds"),
    }


def recommend(assessments: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer the largest eligible reduction; fall back to dense safely."""

    eligible_reduced = [
        row
        for row in assessments
        if row["eligible"] and row["name"] != "dense_reference"
    ]
    if not eligible_reduced:
        return next(row for row in assessments if row["name"] == "dense_reference")

    def ranking_key(row: dict[str, Any]) -> tuple[float, float, float]:
        # Missing runtime values rank last rather than being interpreted as a
        # performance gain. Parameter reduction remains the primary objective.
        vram = row.get("peak_vram_gb")
        latency = row.get("mean_generation_seconds")
        return (
            float(row["parameter_reduction_fraction"]),
            -float(vram) if vram is not None and math.isfinite(float(vram)) else -math.inf,
            -float(latency)
            if latency is not None and math.isfinite(float(latency))
            else -math.inf,
        )

    return max(eligible_reduced, key=ranking_key)


def render_report(
    *,
    registry: Path,
    thresholds: dict[str, float],
    assessments: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> str:
    """Render a reviewable Markdown recommendation with no automatic approval."""

    rows: list[str] = []
    for row in assessments:
        reasons = "; ".join(row["reasons"]) if row["reasons"] else "-"
        rows.append(
            f"| {row['name']} | {'YES' if row['eligible'] else 'NO'} | "
            f"{row['parameter_reduction_fraction']:.3%} | "
            f"{row['mean_score_delta']:+.4f} | {reasons} |"
        )

    return f"""# PHASE 4B CANDIDATE RECOMMENDATION

## Status

`RECOMMENDATION ONLY — HUMAN REVIEW REQUIRED`

## Inputs

- Candidate registry: `{registry}`
- Thresholds: `{thresholds}`

## Candidate decisions

| Candidate | Eligible | Parameter reduction | Mean-score delta | Rejection reasons |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

## Automated recommendation

- Candidate: `{recommendation['name']}`
- Model: `{recommendation['model']}`
- Parameter reduction: `{recommendation['parameter_reduction_fraction']:.3%}`

This recommendation does not approve deployment. Review raw replies, safety-sensitive
cases, real latency/VRAM, checkpoint reload behavior, and maintenance cost. Keeping
`dense_reference` is valid when reduced candidates do not produce worthwhile value.

## Next action

1. Record the final choice in `reports/PHASE_4B_STOP_GATE.md`.
2. Add exactly `- Human decision: PASS` after review.
3. Run `finalize_reference_selection.py` for the chosen candidate.
"""


def main() -> None:
    args = parse_args()
    registry = read_json(args.registry)
    candidates = [load_candidate(row) for row in registry["candidates"]]
    dense = next(
        row for row in candidates if row["name"] == registry["dense_reference_name"]
    )

    thresholds = {
        "max_mean_score_loss": args.max_mean_score_loss,
        "max_category_score_loss": args.max_category_score_loss,
        "max_semantic_pass_loss": args.max_semantic_pass_loss,
        "min_complete_utterance_rate": args.min_complete_utterance_rate,
        "min_parameter_reduction": args.min_parameter_reduction,
    }
    assessments = [
        assess_candidate(row, dense, **thresholds) for row in candidates
    ]
    recommendation = recommend(assessments)

    write_json(
        args.out,
        {
            "status": "RECOMMENDATION_PENDING_HUMAN",
            "registry": str(args.registry),
            "thresholds": thresholds,
            "assessments": assessments,
            "recommended_candidate": recommendation["name"],
            "recommended_model": recommendation["model"],
        },
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_report(
            registry=args.registry,
            thresholds=thresholds,
            assessments=assessments,
            recommendation=recommendation,
        ),
        encoding="utf-8",
    )
    print(args.out)
    print(args.report)


if __name__ == "__main__":
    main()
