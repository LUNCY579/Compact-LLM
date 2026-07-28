# Purpose: Selects the lowest viable active ratio inside quality floors and writes the Phase 6 gate.
from __future__ import annotations

"""Choose the most aggressive Phase 6 ratio inside configured quality floors."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 6 report and gate.")
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--max-score-loss", type=float, default=0.01)
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
    if dense is None:
        automated = "FAIL"
        global_reasons = ["masked generation quality was not run"]
    else:
        global_reasons = []
        for row in comparisons:
            ratio = float(row["active_ratio"])
            if ratio >= 0.999:
                continue
            reasons: list[str] = []
            if float(row["mean_score_loss_from_dense"]) > args.max_score_loss:
                reasons.append("mean score loss exceeded the limit")
            if float(row["semantic_pass_loss_from_dense"]) > args.max_semantic_pass_loss:
                reasons.append("semantic pass loss exceeded the limit")
            if float(row["complete_utterance_rate"]) < args.min_complete_rate:
                reasons.append("complete-utterance rate fell below the floor")
            if int(row["critical_failure_count"]) > int(dense["critical_failure_count"]):
                reasons.append("critical failures increased")
            reasons_by_ratio[ratio] = reasons
            if not reasons:
                eligible.append(row)
        automated = "PASS_PENDING_HUMAN_REVIEW" if eligible else "PASS_WITH_DENSE_ONLY_PENDING_HUMAN_REVIEW"

    selected = min(eligible, key=lambda row: float(row["active_ratio"])) if eligible else None
    table_rows = []
    for row in comparisons:
        ratio = float(row["active_ratio"])
        table_rows.append(
            f"| {ratio:.2f} | {row['mean_heuristic_score']:.4f} | "
            f"{row['mean_score_loss_from_dense']:.4f} | {row['semantic_pass_rate']:.4f} | "
            f"{row['critical_failure_count']} | "
            f"{'; '.join(reasons_by_ratio.get(ratio, [])) or '-'} |"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "\n".join(
            [
                "# SPARSITY CALIBRATION REPORT",
                "",
                f"- Automated status: `{automated}`",
                f"- Recommended active ratio: `{selected['active_ratio'] if selected else 'NONE'}`",
                "- Speed claim allowed in Phase 6: `NO`",
                "",
                "| Active ratio | Mean score | Score loss | Semantic pass | Critical failures | Issues |",
                "|---:|---:|---:|---:|---:|---|",
                *table_rows,
                "",
                "Review the raw dense/masked generations before approval.",
            ]
        ),
        encoding="utf-8",
    )
    args.gate.write_text(
        "\n".join(
            [
                "# PHASE 6 STOP GATE",
                "",
                "## Automated Gate Status",
                "",
                f"`{automated}`",
                "",
                f"- Recommended ratio for Phase 7: `{selected['active_ratio'] if selected else 'NONE'}`",
                f"- Production fallback: `{'SPARSE_EXPERIMENT' if selected else 'DENSE_ONLY'}`",
                *([f"- {reason}" for reason in global_reasons] or []),
                "",
                "## Human decision",
                "",
                "- Human decision: `PENDING`",
                "- Reviewer:",
                "- Date:",
                "- Notes:",
            ]
        ),
        encoding="utf-8",
    )
    print(args.report)
    print(args.gate)


if __name__ == "__main__":
    main()
