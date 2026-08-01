from __future__ import annotations

"""Phase 6 gate: active-compute reduction versus quality opportunity loss."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json
from scripts.common.opportunity_evaluation import evaluate_opportunity, pareto_frontier
from scripts.common.quality_gate import assert_valid_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 6 relative-opportunity report and gate.")
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--quality-profile", choices=("balanced", "logic_oriented", "conversational_permissive"), default="balanced")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/SPARSITY_CALIBRATION_REPORT.md")
    parser.add_argument("--gate", type=Path, default=ROOT / "reports/PHASE_6_STOP_GATE.md")
    for flag in ("--max-quality-loss", "--max-score-loss", "--max-worst-decile-loss", "--max-category-quality-loss", "--max-semantic-pass-loss", "--min-complete-rate"):
        parser.add_argument(flag, type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    payload = read_json(args.frontier)
    dense = payload.get("dense_quality")
    comparisons = payload.get("masked_generation_comparison") or []
    technical_reasons: list[str] = []
    assessed: list[dict] = []
    if dense is None:
        technical_reasons.append("masked generation quality was not run")
    else:
        try:
            assert_valid_reference(dense, label="Phase 6 dense reference")
            for row in comparisons:
                ratio = float(row["active_ratio"])
                if ratio >= 1.0:
                    continue
                assessment = evaluate_opportunity(
                    dense,
                    row,
                    resource_gain=max(0.0, 1.0 - ratio),
                    resource_name="active_compute_reduction",
                    profile=args.quality_profile,
                )
                row["opportunity_assessment"] = assessment
                assessed.append(row)
        except (ValueError, TypeError) as exc:
            technical_reasons.append(str(exc))
    worthwhile = [row for row in assessed if row["opportunity_assessment"]["eligible"]]
    selected = max(
        worthwhile,
        key=lambda row: (
            float(row["opportunity_assessment"]["opportunity_margin"]),
            float(row["opportunity_assessment"]["resource_gain_fraction"]),
        ),
        default=None,
    )
    merged = [{**row, **row["opportunity_assessment"]} for row in assessed]
    frontier_keys = {(float(row["active_ratio"]), float(row["opportunity_margin"])) for row in pareto_frontier(merged)}
    status = "PASS_PENDING_HUMAN_REVIEW" if selected and not technical_reasons else (
        "PASS_WITH_DENSE_ONLY_PENDING_HUMAN_REVIEW" if not technical_reasons else "BLOCKED_TECHNICAL_EVIDENCE"
    )
    table = []
    for row in sorted(assessed, key=lambda item: float(item["opportunity_assessment"]["opportunity_margin"]), reverse=True):
        a = row["opportunity_assessment"]
        table.append(
            f"| {float(row['active_ratio']):.3f} | {a['resource_gain_fraction']:.3%} | "
            f"{a['quality_opportunity_loss_fraction']:.3%} | {a['opportunity_margin']:+.3%} | "
            f"{'YES' if (float(row['active_ratio']), float(a['opportunity_margin'])) in frontier_keys else 'NO'} | "
            f"{'WORTHWHILE' if a['eligible'] else 'NOT WORTHWHILE'} |"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join([
        "# SPARSITY CALIBRATION REPORT", "", f"- Automated status: `{status}`", "- Quality evaluator: `3.0`",
        f"- Recommended active ratio: `{selected['active_ratio'] if selected else 'DENSE'}`",
        "- Speed claim allowed in Phase 6: `NO`", "",
        "| Active ratio | Compute reduction | Quality opportunity loss | Opportunity margin | Pareto | Decision |",
        "|---:|---:|---:|---:|---:|---|", *table, "",
        *([f"- Technical blocker: {reason}" for reason in technical_reasons] or ["No absolute quality floor was applied."]),
    ]), encoding="utf-8")
    args.gate.write_text("\n".join([
        "# PHASE 6 STOP GATE", "", "## Automated Gate Status", "", f"`{status}`", "",
        f"- Recommended ratio for Phase 7: `{selected['active_ratio'] if selected else 'NONE'}`",
        f"- Opportunity margin: `{selected['opportunity_assessment']['opportunity_margin'] if selected else 'N/A'}`",
        f"- Production fallback: `{'SPARSE_EXPERIMENT' if selected else 'DENSE_ONLY'}`", "",
        "## Human decision", "", "- Human decision: `PENDING`", "- Reviewer:", "- Date:", "- Notes:",
    ]), encoding="utf-8")
    print(args.report)
    print(args.gate)


if __name__ == "__main__":
    main()
