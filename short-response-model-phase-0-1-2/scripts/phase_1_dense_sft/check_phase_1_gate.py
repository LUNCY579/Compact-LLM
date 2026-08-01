from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.quality_gate import compare_quality_summaries, quality_index
from scripts.common.quality_evaluation import require_quality_v2, summary_quality_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base and dense-SFT evaluations with the quality-v2 contract."
    )
    parser.add_argument("--base-summary", type=Path, default=ROOT / "artifacts/baseline_evaluation/summary.json")
    parser.add_argument("--adapter-summary", type=Path, default=ROOT / "artifacts/dense_sft_evaluation/summary.json")
    parser.add_argument("--output-report", type=Path, default=ROOT / "reports/PHASE_1_STOP_GATE.md")
    parser.add_argument("--minimum-score-improvement", type=float, default=0.02, help="Minimum mean quality-score improvement.")
    parser.add_argument("--maximum-median-tokens", type=float, default=12.0)
    parser.add_argument("--maximum-p90-tokens", type=float, default=24.0)
    parser.add_argument("--minimum-complete-rate", type=float, default=0.99)
    parser.add_argument("--maximum-category-regression", type=float, default=0.08)
    parser.add_argument("--minimum-quality-pass-rate", type=float, default=0.95)
    parser.add_argument("--minimum-semantic-pass-rate", type=float, default=0.98)
    parser.add_argument("--minimum-worst-decile-quality", type=float, default=0.70)
    parser.add_argument("--maximum-hard-failure-cases", type=int, default=0)
    parser.add_argument("--maximum-major-issue-increase", type=int, default=0)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require_quality_v2(payload, str(path))
    return payload


def main() -> None:
    args = parse_args()
    base = load(args.base_summary)
    adapter = load(args.adapter_summary)
    base_score = summary_quality_score(base)
    adapter_score = summary_quality_score(adapter)
    score_gain = adapter_score - base_score

    comparison = compare_quality_summaries(
        base,
        adapter,
        max_mean_quality_loss=1.0,  # Phase 1 requires a positive gain separately.
        max_quality_index_loss=1.0,
        max_worst_decile_loss=1.0,
        max_category_quality_loss_allowed=args.maximum_category_regression,
        max_semantic_pass_loss=max(0.0, float(base.get("semantic_pass_rate", 0.0)) - args.minimum_semantic_pass_rate),
        max_reference_alignment_loss=0.10,
        min_quality_pass_rate=args.minimum_quality_pass_rate,
        min_intelligibility_pass_rate=1.0,
        min_semantic_pass_rate=args.minimum_semantic_pass_rate,
        min_complete_utterance_rate=args.minimum_complete_rate,
        max_hard_failure_cases=args.maximum_hard_failure_cases,
        max_critical_failures=0,
        max_major_issue_cases=int(base.get("major_case_count", 0)) + args.maximum_major_issue_increase,
    )

    checks = {
        "quality_v2_evidence": True,
        "mean_quality_improvement": score_gain >= args.minimum_score_improvement,
        "quality_index_not_regressed": quality_index(adapter) >= quality_index(base),
        "quality_pass_rate": float(adapter.get("quality_pass_rate", 0.0)) >= args.minimum_quality_pass_rate,
        "intelligibility_zero_tolerance": int(adapter.get("hard_failure_case_count", 0)) <= args.maximum_hard_failure_cases,
        "semantic_pass_rate": float(adapter.get("semantic_pass_rate", 0.0)) >= args.minimum_semantic_pass_rate,
        "worst_decile_quality": float(adapter.get("worst_decile_mean_quality_score", 0.0)) >= args.minimum_worst_decile_quality,
        "median_reply_tokens": float(adapter["median_reply_tokens"]) <= args.maximum_median_tokens,
        "p90_reply_tokens": float(adapter["p90_reply_tokens"]) <= args.maximum_p90_tokens,
        "complete_utterance_rate": float(adapter["complete_utterance_rate"]) >= args.minimum_complete_rate,
        "category_quality_regression_limit": comparison["maximum_category_quality_loss"] <= args.maximum_category_regression,
        "no_new_major_issue_cases": int(adapter.get("major_case_count", 0)) <= int(base.get("major_case_count", 0)) + args.maximum_major_issue_increase,
        "no_critical_failures": int(adapter.get("critical_failure_count", 0)) == 0,
    }
    automated_pass = all(checks.values())
    status = "PASS_PENDING_HUMAN_REVIEW" if automated_pass else "FAIL"
    check_rows = "\n".join(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items())
    category_rows = "\n".join(
        f"| {category} | {loss:.3f} |"
        for category, loss in sorted(comparison["category_quality_losses"].items(), key=lambda item: item[1], reverse=True)
    ) or "| None | 0.000 |"

    report = f"""# PHASE 1 STOP GATE

## Purpose

Compare the untouched base model with the dense-SFT adapter using evaluator
schema v2.0. Mean quality is only one input: hard failures, tail quality,
semantic pass, category regressions, completeness, and brevity are independent
gates.

## Inputs

- Base summary: `{args.base_summary}`
- Adapter summary: `{args.adapter_summary}`
- Quality evaluator: `2.0`

## Comparison

| Metric | Base | Dense SFT | Difference |
|---|---:|---:|---:|
| Mean quality score | {base_score:.3f} | {adapter_score:.3f} | {score_gain:+.3f} |
| Quality index | {quality_index(base):.3f} | {quality_index(adapter):.3f} | {quality_index(adapter)-quality_index(base):+.3f} |
| Quality pass rate | {float(base.get('quality_pass_rate', 0.0)):.3f} | {float(adapter.get('quality_pass_rate', 0.0)):.3f} | {float(adapter.get('quality_pass_rate', 0.0))-float(base.get('quality_pass_rate', 0.0)):+.3f} |
| Intelligibility pass rate | {float(base.get('intelligibility_pass_rate', 0.0)):.3f} | {float(adapter.get('intelligibility_pass_rate', 0.0)):.3f} | {float(adapter.get('intelligibility_pass_rate', 0.0))-float(base.get('intelligibility_pass_rate', 0.0)):+.3f} |
| Hard-failure cases | {int(base.get('hard_failure_case_count', 0))} | {int(adapter.get('hard_failure_case_count', 0))} | {int(adapter.get('hard_failure_case_count', 0))-int(base.get('hard_failure_case_count', 0)):+d} |
| Worst-decile mean quality | {float(base.get('worst_decile_mean_quality_score', 0.0)):.3f} | {float(adapter.get('worst_decile_mean_quality_score', 0.0)):.3f} | {float(adapter.get('worst_decile_mean_quality_score', 0.0))-float(base.get('worst_decile_mean_quality_score', 0.0)):+.3f} |
| Semantic pass rate | {float(base.get('semantic_pass_rate', 0.0)):.3f} | {float(adapter.get('semantic_pass_rate', 0.0)):.3f} | {float(adapter.get('semantic_pass_rate', 0.0))-float(base.get('semantic_pass_rate', 0.0)):+.3f} |
| Major-issue cases | {int(base.get('major_case_count', 0))} | {int(adapter.get('major_case_count', 0))} | {int(adapter.get('major_case_count', 0))-int(base.get('major_case_count', 0)):+d} |
| Median reply tokens | {float(base['median_reply_tokens']):.1f} | {float(adapter['median_reply_tokens']):.1f} | {float(adapter['median_reply_tokens'])-float(base['median_reply_tokens']):+.1f} |
| P90 reply tokens | {float(base['p90_reply_tokens']):.1f} | {float(adapter['p90_reply_tokens']):.1f} | {float(adapter['p90_reply_tokens'])-float(base['p90_reply_tokens']):+.1f} |
| Complete utterance rate | {float(base['complete_utterance_rate']):.3f} | {float(adapter['complete_utterance_rate']):.3f} | {float(adapter['complete_utterance_rate'])-float(base['complete_utterance_rate']):+.3f} |

## Automated Checks

| Check | Result |
|---|---|
{check_rows}

## Category Quality Losses

| Category | Loss vs base |
|---|---:|
{category_rows}

## Gate Status

`{status}`

## Required Human Review

Inspect every hard/major flag, the lowest-quality cases, and the worst category.
Confirm naturalness, correctness, context use, clarification, role consistency,
and safety before recording `Human decision: PASS`.

- Human decision: `PENDING`
- Reviewer:
- Date:
- Notes:
"""
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(report, encoding="utf-8")
    print(f"Phase 1 gate report: {args.output_report}")
    print(f"Automated status: {status}")
    if not automated_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
