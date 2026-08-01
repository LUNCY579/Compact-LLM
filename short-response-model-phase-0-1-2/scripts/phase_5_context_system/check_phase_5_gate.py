from __future__ import annotations

"""Phase 5 gate: context-token saving versus retrieval-adjusted quality loss."""

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json
from scripts.common.opportunity_evaluation import evaluate_opportunity
from scripts.common.quality_gate import assert_valid_reference


def _retrieval_adjusted(summary: dict, recall: float, forbidden_rate: float) -> dict:
    adjusted = copy.deepcopy(summary)
    dims = dict(adjusted.get("mean_quality_dimensions") or adjusted.get("quality_dimension_means") or {})
    retention = max(0.0, min(1.0, recall)) * max(0.0, 1.0 - forbidden_rate)
    for name in ("context_consistency", "task_adequacy", "content_anchor_preservation"):
        if name in dims:
            dims[name] = float(dims[name]) * retention
    adjusted["mean_quality_dimensions"] = dims
    adjusted["quality_dimension_means"] = dims
    adjusted["mean_issue_cost"] = min(1.0, float(adjusted.get("mean_issue_cost", 0.0)) + forbidden_rate)
    adjusted["severe_issue_case_rate"] = min(1.0, float(adjusted.get("severe_issue_case_rate", 0.0)) + forbidden_rate)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 relative context opportunity.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--prefix-cache", type=Path, required=True)
    parser.add_argument("--quality-profile", choices=("balanced", "logic_oriented", "conversational_permissive"), default="balanced")
    parser.add_argument("--out-report", type=Path, default=ROOT / "reports/CONTEXT_SYSTEM_REPORT.md")
    parser.add_argument("--out-gate", type=Path, default=ROOT / "reports/PHASE_5_STOP_GATE.md")
    for flag in ("--min-recall", "--min-memory-reduction", "--max-quality-loss", "--max-worst-decile-loss", "--max-category-quality-loss"):
        parser.add_argument(flag, type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    evaluation = read_json(args.evaluation)
    prefix = read_json(args.prefix_cache)
    compact, full = evaluation.get("compact_quality"), evaluation.get("full_quality")
    technical_reasons: list[str] = []
    opportunity = None
    if compact is None or full is None:
        technical_reasons.append("compact/full model quality evaluation is missing")
    else:
        try:
            assert_valid_reference(full, label="Phase 5 full-context reference")
            case_count = max(1, int(evaluation.get("case_count") or full.get("case_count") or 1))
            forbidden_rate = float(evaluation.get("forbidden_memory_selection_count", 0)) / case_count
            adjusted = _retrieval_adjusted(compact, float(evaluation.get("relevant_memory_recall", 0.0)), forbidden_rate)
            opportunity = evaluate_opportunity(
                full,
                adjusted,
                resource_gain=float(evaluation.get("memory_token_reduction_fraction", 0.0)),
                resource_name="retrieved_context_token_reduction",
                profile=args.quality_profile,
            )
        except (ValueError, TypeError) as exc:
            technical_reasons.append(str(exc))
    if not prefix.get("numerically_consistent"):
        technical_reasons.append("prefix-cache implementation is not numerically consistent with full prefill")
    worthwhile = bool(opportunity and opportunity["eligible"] and not technical_reasons)
    status = "PASS_PENDING_HUMAN_REVIEW" if worthwhile else (
        "NO_PROPORTIONALLY_WORTHWHILE_CONTEXT_REDUCTION" if opportunity and not technical_reasons else "BLOCKED_TECHNICAL_EVIDENCE"
    )
    deprecated = [name for name, value in vars(args).items() if name.startswith(("min_", "max_")) and value is not None]
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join([
        "# CONTEXT SYSTEM REPORT", "", f"- Automated status: `{status}`", "- Quality evaluator: `3.0`",
        f"- Relevant-memory recall (continuous evidence): `{evaluation.get('relevant_memory_recall')}`",
        f"- Relevant-memory precision (continuous evidence): `{evaluation.get('relevant_memory_precision')}`",
        f"- Forbidden selections: `{evaluation.get('forbidden_memory_selection_count')}`",
        f"- Memory-token reduction: `{float(evaluation.get('memory_token_reduction_fraction',0.0)):.2%}`",
        f"- Whole-prompt token reduction: `{float(evaluation.get('prompt_token_reduction_fraction',0.0)):.2%}`",
        f"- Resource gain: `{opportunity.get('resource_gain_fraction') if opportunity else 'N/A'}`",
        f"- Quality opportunity loss: `{opportunity.get('quality_opportunity_loss_fraction') if opportunity else 'N/A'}`",
        f"- Opportunity margin: `{opportunity.get('opportunity_margin') if opportunity else 'N/A'}`",
        f"- Prefix numerical consistency: `{prefix.get('numerically_consistent')}`",
        f"- Prefix steady-state speedup (diagnostic): `{prefix.get('steady_state_speedup')}`",
        f"- Deprecated threshold arguments supplied and ignored: `{deprecated or 'None'}`", "",
        "## Technical blockers", "", *([f"- {reason}" for reason in technical_reasons] or ["- None."]), "",
        "Retrieval recall, forbidden-memory selection, and language quality contribute proportionally; none is an independent numerical floor.",
    ]), encoding="utf-8")
    args.out_gate.write_text("\n".join([
        "# PHASE 5 STOP GATE", "", "## Automated Gate Status", "", f"`{status}`", "",
        f"- Resource gain: `{opportunity.get('resource_gain_fraction') if opportunity else 'N/A'}`",
        f"- Quality opportunity loss: `{opportunity.get('quality_opportunity_loss_fraction') if opportunity else 'N/A'}`",
        f"- Opportunity margin: `{opportunity.get('opportunity_margin') if opportunity else 'N/A'}`", "",
        "## Human decision", "", "- Human decision: `PENDING`", "- Reviewer:", "- Date:", "- Notes:",
    ]), encoding="utf-8")
    print(args.out_report)
    print(args.out_gate)


if __name__ == "__main__":
    main()
