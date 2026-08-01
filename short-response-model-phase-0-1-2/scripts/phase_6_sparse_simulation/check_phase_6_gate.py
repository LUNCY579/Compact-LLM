# Purpose: Selects the lowest viable active ratio inside quality-v2 floors and writes the Phase 6 gate.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.experiment_utils import read_json
from scripts.common.quality_gate import assert_valid_reference, compare_quality_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 6 quality-v2 report and gate.")
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--max-quality-loss", "--max-score-loss", dest="max_quality_loss", type=float, default=0.01)
    parser.add_argument("--max-worst-decile-loss", type=float, default=0.03)
    parser.add_argument("--max-category-quality-loss", type=float, default=0.04)
    parser.add_argument("--max-semantic-pass-loss", type=float, default=0.01)
    parser.add_argument("--min-complete-rate", type=float, default=0.99)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/SPARSITY_CALIBRATION_REPORT.md")
    parser.add_argument("--gate", type=Path, default=ROOT / "reports/PHASE_6_STOP_GATE.md")
    args = parser.parse_args()

    payload = read_json(args.frontier)
    dense = payload.get("dense_quality")
    comparisons = payload.get("masked_generation_comparison") or []
    eligible: list[dict] = []
    reasons_by_ratio: dict[float, list[str]] = {}
    global_reasons: list[str] = []
    if dense is None:
        automated = "FAIL"; global_reasons.append("masked generation quality was not run")
    else:
        try:
            assert_valid_reference(dense, label="Phase 6 dense reference")
        except ValueError as exc:
            global_reasons.append(str(exc))
        for row in comparisons:
            ratio = float(row["active_ratio"])
            if ratio >= 0.999:
                continue
            comparison = compare_quality_summaries(
                dense, row,
                max_mean_quality_loss=args.max_quality_loss,
                max_quality_index_loss=args.max_quality_loss + 0.005,
                max_worst_decile_loss=args.max_worst_decile_loss,
                max_category_quality_loss_allowed=args.max_category_quality_loss,
                max_semantic_pass_loss=args.max_semantic_pass_loss,
                max_reference_alignment_loss=0.05,
                min_quality_pass_rate=0.98, min_intelligibility_pass_rate=1.0,
                min_semantic_pass_rate=0.98, min_complete_utterance_rate=args.min_complete_rate,
                max_hard_failure_cases=0, max_critical_failures=0, max_major_issue_cases=0,
            )
            reasons_by_ratio[ratio] = comparison["reasons"]
            row["quality_comparison"] = comparison
            if comparison["eligible"] and not global_reasons:
                eligible.append(row)
        automated = "PASS_PENDING_HUMAN_REVIEW" if eligible and not global_reasons else (
            "PASS_WITH_DENSE_ONLY_PENDING_HUMAN_REVIEW" if not global_reasons else "FAIL"
        )

    selected = min(eligible, key=lambda row: float(row["active_ratio"])) if eligible else None
    table_rows = []
    for row in comparisons:
        ratio = float(row["active_ratio"])
        table_rows.append(
            f"| {ratio:.2f} | {float(row.get('mean_quality_score', 0.0)):.4f} | "
            f"{float(row.get('worst_decile_mean_quality_score', 0.0)):.4f} | "
            f"{float(row.get('quality_pass_rate', 0.0)):.4f} | {float(row.get('intelligibility_pass_rate', 0.0)):.4f} | "
            f"{int(row.get('hard_failure_case_count', 0))} | {int(row.get('major_case_count', 0))} | "
            f"{'; '.join(reasons_by_ratio.get(ratio, [])) or '-'} |"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join([
        "# SPARSITY CALIBRATION REPORT", "", f"- Automated status: `{automated}`",
        f"- Quality evaluator: `2.0`", f"- Recommended active ratio: `{selected['active_ratio'] if selected else 'NONE'}`",
        "- Speed claim allowed in Phase 6: `NO`", "",
        "| Active ratio | Mean quality | Worst decile | Quality pass | Intelligibility | Hard cases | Major cases | Issues |",
        "|---:|---:|---:|---:|---:|---:|---:|---|", *table_rows, "",
        *([f"- Global blocker: {reason}" for reason in global_reasons] or []),
        "Review every hard/major/low-tail dense and masked generation before approval.",
    ]), encoding="utf-8")
    args.gate.write_text("\n".join([
        "# PHASE 6 STOP GATE", "", "## Automated Gate Status", "", f"`{automated}`", "",
        f"- Recommended ratio for Phase 7: `{selected['active_ratio'] if selected else 'NONE'}`",
        f"- Production fallback: `{'SPARSE_EXPERIMENT' if selected else 'DENSE_ONLY'}`",
        *([f"- {reason}" for reason in global_reasons] or []), "", "## Human decision", "",
        "- Human decision: `PENDING`", "- Reviewer:", "- Date:", "- Notes:",
    ]), encoding="utf-8")
    print(args.report); print(args.gate)


if __name__ == "__main__":
    main()
