from __future__ import annotations

"""Phase 3 gate: proportional alignment-quality gains versus regressions."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json
from scripts.common.opportunity_evaluation import evaluate_quality_improvement
from scripts.common.quality_evaluation import require_quality_v4, summary_quality_score
from scripts.common.quality_gate import quality_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SFT and DPO with relative quality opportunity accounting.")
    parser.add_argument("--sft-summary", type=Path, required=True)
    parser.add_argument("--dpo-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/PHASE_3_STOP_GATE.md")
    parser.add_argument("--quality-profile", choices=("balanced", "logic_oriented", "conversational_permissive"), default="balanced")
    args = parser.parse_args()
    sft = read_json(args.sft_summary)
    dpo = read_json(args.dpo_summary)
    require_quality_v4(sft, "Phase 3 SFT summary")
    require_quality_v4(dpo, "Phase 3 DPO summary")
    comparison = evaluate_quality_improvement(sft, dpo, profile=args.quality_profile)
    status = "PASS_PENDING_HUMAN_REVIEW" if comparison["eligible"] else "NO_NET_RELATIVE_ALIGNMENT_GAIN"
    loss_rows = "\n".join(
        f"| {name} | {float(value):.4%} |"
        for name, value in sorted(comparison.get("quality_dimension_losses", {}).items())
    )
    gain_rows = "\n".join(
        f"| {name} | {float(value):.4%} |"
        for name, value in sorted(comparison.get("quality_dimension_gains", {}).items())
    )
    report = f"""# PHASE 3 STOP GATE

## Governing rule

DPO advances when its weighted proportional quality gains are greater than its weighted proportional regressions. No absolute score, pass-rate, issue-count, completion, or token-length floor controls promotion.

| Metric | SFT | DPO | Change |
|---|---:|---:|---:|
| Mean quality | {summary_quality_score(sft):.4f} | {summary_quality_score(dpo):.4f} | {summary_quality_score(dpo)-summary_quality_score(sft):+.4f} |
| Descriptive quality index | {quality_index(sft):.4f} | {quality_index(dpo):.4f} | {quality_index(dpo)-quality_index(sft):+.4f} |
| Worst-decile quality | {float(sft.get('worst_decile_mean_quality_score',0)):.4f} | {float(dpo.get('worst_decile_mean_quality_score',0)):.4f} | {float(dpo.get('worst_decile_mean_quality_score',0))-float(sft.get('worst_decile_mean_quality_score',0)):+.4f} |
| P90 tokens (diagnostic) | {float(sft.get('p90_reply_tokens',0)):.1f} | {float(dpo.get('p90_reply_tokens',0)):.1f} | {float(dpo.get('p90_reply_tokens',0))-float(sft.get('p90_reply_tokens',0)):+.1f} |

## Relative losses

| Dimension | Proportional loss |
|---|---:|
{loss_rows or '| None | 0.0000% |'}

## Relative gains

| Dimension | Proportional gain |
|---|---:|
{gain_rows or '| None | 0.0000% |'}

- Combined quality opportunity loss: `{comparison['combined_quality_opportunity_loss']:.4%}`
- Combined quality gain: `{comparison['combined_quality_opportunity_gain']:.4%}`
- Net relative quality margin: `{comparison['opportunity_margin']:+.4%}`
- Quality profile: `{args.quality_profile}`

## Automated status

`{status}`

## Human decision

Review blinded replies, preference behavior, logic, context, safety, and proportional issue-rate changes.

- Human decision: `PENDING`
- Reviewer:
- Date:
- Notes:
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
