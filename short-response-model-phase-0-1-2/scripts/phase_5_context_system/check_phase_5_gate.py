# Purpose: Evaluates Phase 5 automated floors and writes the report plus pending human gate.
from __future__ import annotations

"""Create the Phase 5 evidence report and pending human stop gate."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 automated floors.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--prefix-cache", type=Path, required=True)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-memory-reduction", type=float, default=0.25)
    parser.add_argument("--max-quality-loss", type=float, default=0.02)
    parser.add_argument("--out-report", type=Path, default=ROOT / "reports/CONTEXT_SYSTEM_REPORT.md")
    parser.add_argument("--out-gate", type=Path, default=ROOT / "reports/PHASE_5_STOP_GATE.md")
    args = parser.parse_args()

    evaluation = read_json(args.evaluation)
    prefix = read_json(args.prefix_cache)
    reasons: list[str] = []
    if float(evaluation["relevant_memory_recall"]) < args.min_recall:
        reasons.append("relevant-memory recall is below the configured floor")
    if int(evaluation["forbidden_memory_selection_count"]) != 0:
        reasons.append("stale/contradicted/forbidden memory was selected")
    if float(evaluation["memory_token_reduction_fraction"]) < args.min_memory_reduction:
        reasons.append("retrieved-memory token reduction is below the configured floor")
    quality_delta = evaluation.get("compact_minus_full_quality")
    if quality_delta is not None and float(quality_delta) < -args.max_quality_loss:
        reasons.append("compact-context generation quality regressed too far")
    if not prefix.get("numerically_consistent"):
        reasons.append("prefix-cache logits do not match full prefill")
    automated = "PASS_PENDING_HUMAN_REVIEW" if not reasons else "FAIL"

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        "\n".join(
            [
                "# CONTEXT SYSTEM REPORT",
                "",
                f"- Automated status: `{automated}`",
                f"- Relevant-memory recall: `{evaluation['relevant_memory_recall']:.4f}`",
                f"- Relevant-memory precision: `{evaluation['relevant_memory_precision']:.4f}`",
                f"- Forbidden selections: `{evaluation['forbidden_memory_selection_count']}`",
                f"- Memory-token reduction: `{evaluation['memory_token_reduction_fraction']:.2%}`",
                f"- Whole-prompt token reduction: `{evaluation['prompt_token_reduction_fraction']:.2%}`",
                f"- Compact minus full generation quality: `{quality_delta}`",
                f"- Median context assembly: `{evaluation['median_assembly_ms']:.4f} ms`",
                f"- Prefix steady-state speedup: `{prefix.get('steady_state_speedup')}`",
                f"- Prefix numerical consistency: `{prefix.get('numerically_consistent')}`",
                "",
                "## Automated issues",
                "",
                *([f"- {reason}" for reason in reasons] or ["- None."]),
                "",
                "Review retrieval_traces.jsonl and compact/full replies before approval.",
            ]
        ),
        encoding="utf-8",
    )
    args.out_gate.write_text(
        "\n".join(
            [
                "# PHASE 5 STOP GATE",
                "",
                "## Automated Gate Status",
                "",
                f"`{automated}`",
                "",
                *([f"- {reason}" for reason in reasons] or ["- All configured automated floors passed."]),
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
    print(args.out_report)
    print(args.out_gate)


if __name__ == "__main__":
    main()
