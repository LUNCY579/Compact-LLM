from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.opportunity_evaluation import evaluate_quality_improvement
from scripts.common.quality_evaluation import require_current_quality, summary_quality_score
from scripts.common.quality_gate import quality_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and dense SFT using relative continuous quality evidence.")
    parser.add_argument("--base-summary", type=Path, default=ROOT / "artifacts/baseline_evaluation/summary.json")
    parser.add_argument("--adapter-summary", type=Path, default=ROOT / "artifacts/dense_sft_evaluation/summary.json")
    parser.add_argument("--output-report", type=Path, default=ROOT / "reports/PHASE_1_STOP_GATE.md")
    # Accepted only so old commands do not break. None controls promotion.
    parser.add_argument("--minimum-score-improvement", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--maximum-median-tokens", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--maximum-p90-tokens", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--minimum-complete-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--maximum-category-regression", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--minimum-quality-pass-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--minimum-semantic-pass-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--minimum-worst-decile-quality", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--maximum-hard-failure-cases", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--maximum-major-issue-increase", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require_current_quality(payload, str(path))
    return payload


def main() -> None:
    args = parse_args()
    base, adapter = load(args.base_summary), load(args.adapter_summary)
    result = evaluate_quality_improvement(base, adapter)
    status = "PASS_PENDING_HUMAN_REVIEW" if result["eligible"] else "FAIL"
    base_score, adapter_score = summary_quality_score(base), summary_quality_score(adapter)
    losses = "\n".join(
        f"| {name} | {value:.4%} |"
        for name, value in sorted(result["quality_dimension_losses"].items(), key=lambda item: item[1], reverse=True)
    ) or "| None | 0.0000% |"
    gains = "\n".join(
        f"| {name} | {value:.4%} |"
        for name, value in sorted(result["quality_dimension_gains"].items(), key=lambda item: item[1], reverse=True)
    ) or "| None | 0.0000% |"
    report = f"""# PHASE 1 STOP GATE — RELATIVE OPPORTUNITY POLICY

## Purpose

Compare dense SFT with the untouched base using continuous quality dimensions.
No absolute score, token, completion, pass-rate, issue-count, or category floor
controls promotion. Phase 1 advances only when proportional quality gains exceed
proportional quality losses; human review remains mandatory.

## Inputs

- Base summary: `{args.base_summary}`
- Adapter summary: `{args.adapter_summary}`
- Quality evaluator: `3.0`
- Opportunity policy: `1.0 / balanced`

## Relative comparison

| Metric | Base | Dense SFT | Relative change |
|---|---:|---:|---:|
| Mean quality | {base_score:.4f} | {adapter_score:.4f} | {(adapter_score-base_score)/max(abs(base_score),1e-12):+.4%} |
| Continuous quality index | {quality_index(base):.4f} | {quality_index(adapter):.4f} | {(quality_index(adapter)-quality_index(base))/max(abs(quality_index(base)),1e-12):+.4%} |
| Worst-decile quality | {float(base.get('worst_decile_mean_quality_score',0.0)):.4f} | {float(adapter.get('worst_decile_mean_quality_score',0.0)):.4f} | {(float(adapter.get('worst_decile_mean_quality_score',0.0))-float(base.get('worst_decile_mean_quality_score',0.0)))/max(abs(float(base.get('worst_decile_mean_quality_score',0.0))),1e-12):+.4%} |
| Mean issue cost | {float(base.get('mean_issue_cost',0.0)):.4f} | {float(adapter.get('mean_issue_cost',0.0)):.4f} | {float(adapter.get('mean_issue_cost',0.0))-float(base.get('mean_issue_cost',0.0)):+.4f} |
| Median reply tokens | {float(base.get('median_reply_tokens',0.0)):.1f} | {float(adapter.get('median_reply_tokens',0.0)):.1f} | {float(adapter.get('median_reply_tokens',0.0))-float(base.get('median_reply_tokens',0.0)):+.1f} |
| P90 reply tokens | {float(base.get('p90_reply_tokens',0.0)):.1f} | {float(adapter.get('p90_reply_tokens',0.0)):.1f} | {float(adapter.get('p90_reply_tokens',0.0))-float(base.get('p90_reply_tokens',0.0)):+.1f} |

## Proportional quality gains

| Dimension | Relative gain |
|---|---:|
{gains}

## Proportional quality losses

| Dimension | Relative loss |
|---|---:|
{losses}

- Combined proportional quality gain: `{result['combined_quality_opportunity_gain']:.6f}`
- Combined proportional quality loss: `{result['combined_quality_opportunity_loss']:.6f}`
- Net opportunity margin: `{result['opportunity_margin']:+.6f}`
- Decision: `{result['decision']}`

## Automated status

`{status}`

## Required human review

Review low-tail cases, logic/context/safety changes, and conversational quality.
Minor grammar, informal wording, alternate reasonable replies, and natural
follow-up questions are permitted.

- Human decision: `PENDING`
- Reviewer:
- Date:
- Notes:
"""
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(report, encoding="utf-8")
    print(args.output_report)
    print(f"Automated status: {status}")
    if not result["eligible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
