# Purpose: Evaluates retrieval and quality-v2 floors and writes the Phase 5 pending human gate.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json
from scripts.common.quality_gate import compare_quality_summaries, assert_valid_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 automated floors.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--prefix-cache", type=Path, required=True)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-memory-reduction", type=float, default=0.25)
    parser.add_argument("--max-quality-loss", type=float, default=0.02)
    parser.add_argument("--max-worst-decile-loss", type=float, default=0.04)
    parser.add_argument("--max-category-quality-loss", type=float, default=0.05)
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

    compact = evaluation.get("compact_quality")
    full = evaluation.get("full_quality")
    comparison = None
    if compact is None or full is None:
        reasons.append("compact/full model quality-v2 evaluation was not run")
    else:
        try:
            assert_valid_reference(full, label="Phase 5 full-context reference")
            comparison = compare_quality_summaries(
                full, compact,
                max_mean_quality_loss=args.max_quality_loss,
                max_quality_index_loss=args.max_quality_loss,
                max_worst_decile_loss=args.max_worst_decile_loss,
                max_category_quality_loss_allowed=args.max_category_quality_loss,
                max_semantic_pass_loss=0.01,
                max_reference_alignment_loss=0.05,
                min_quality_pass_rate=0.98,
                min_intelligibility_pass_rate=1.0,
                min_semantic_pass_rate=0.98,
                min_complete_utterance_rate=0.99,
                max_hard_failure_cases=0,
                max_critical_failures=0,
                max_major_issue_cases=0,
            )
            reasons.extend(comparison["reasons"])
        except ValueError as exc:
            reasons.append(str(exc))
    if not prefix.get("numerically_consistent"):
        reasons.append("prefix-cache logits do not match full prefill")
    automated = "PASS_PENDING_HUMAN_REVIEW" if not reasons else "FAIL"

    compact_mean = compact.get("mean_quality_score") if compact else None
    full_mean = full.get("mean_quality_score") if full else None
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join([
        "# CONTEXT SYSTEM REPORT", "", f"- Automated status: `{automated}`",
        f"- Quality evaluator: `2.0`", f"- Relevant-memory recall: `{evaluation['relevant_memory_recall']:.4f}`",
        f"- Relevant-memory precision: `{evaluation['relevant_memory_precision']:.4f}`",
        f"- Forbidden selections: `{evaluation['forbidden_memory_selection_count']}`",
        f"- Memory-token reduction: `{evaluation['memory_token_reduction_fraction']:.2%}`",
        f"- Whole-prompt token reduction: `{evaluation['prompt_token_reduction_fraction']:.2%}`",
        f"- Compact mean quality: `{compact_mean}`", f"- Full-context mean quality: `{full_mean}`",
        f"- Compact/full comparison: `{comparison}`", f"- Median context assembly: `{evaluation['median_assembly_ms']:.4f} ms`",
        f"- Prefix steady-state speedup: `{prefix.get('steady_state_speedup')}`",
        f"- Prefix numerical consistency: `{prefix.get('numerically_consistent')}`", "", "## Automated issues", "",
        *([f"- {reason}" for reason in reasons] or ["- None."]), "",
        "Review retrieval_traces.jsonl, all hard/major/low-tail cases, and compact/full replies before approval.",
    ]), encoding="utf-8")
    args.out_gate.write_text("\n".join([
        "# PHASE 5 STOP GATE", "", "## Automated Gate Status", "", f"`{automated}`", "",
        *([f"- {reason}" for reason in reasons] or ["- All configured automated floors passed."]), "",
        "## Human decision", "", "- Human decision: `PENDING`", "- Reviewer:", "- Date:", "- Notes:",
    ]), encoding="utf-8")
    print(args.out_report); print(args.out_gate)


if __name__ == "__main__":
    main()
