from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base and dense-SFT evaluations and write the Phase 1 stop-gate report."
    )
    parser.add_argument(
        "--base-summary",
        type=Path,
        default=ROOT / "artifacts" / "baseline_evaluation" / "summary.json",
    )
    parser.add_argument(
        "--adapter-summary",
        type=Path,
        default=ROOT / "artifacts" / "dense_sft_evaluation" / "summary.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=ROOT / "reports" / "PHASE_1_STOP_GATE.md",
    )
    parser.add_argument("--minimum-score-improvement", type=float, default=0.02)
    parser.add_argument("--maximum-median-tokens", type=float, default=12.0)
    parser.add_argument("--maximum-p90-tokens", type=float, default=24.0)
    parser.add_argument("--minimum-complete-rate", type=float, default=0.99)
    parser.add_argument("--maximum-category-regression", type=float, default=0.15)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing evaluation summary: {path}. Run the corresponding Phase 1 evaluation first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    base = load(args.base_summary)
    adapter = load(args.adapter_summary)

    base_score = float(base["mean_heuristic_score"])
    adapter_score = float(adapter["mean_heuristic_score"])
    score_gain = adapter_score - base_score

    base_categories = base.get("category_scores", {})
    adapter_categories = adapter.get("category_scores", {})
    regressions: list[tuple[str, float]] = []
    for category, base_value in base_categories.items():
        if category not in adapter_categories:
            regressions.append((category, 1.0))
            continue
        drop = float(base_value) - float(adapter_categories[category])
        if drop > args.maximum_category_regression:
            regressions.append((category, drop))

    checks = {
        "score_improvement": score_gain >= args.minimum_score_improvement,
        "median_reply_tokens": float(adapter["median_reply_tokens"])
        <= args.maximum_median_tokens,
        "p90_reply_tokens": float(adapter["p90_reply_tokens"])
        <= args.maximum_p90_tokens,
        "complete_utterance_rate": float(adapter["complete_utterance_rate"])
        >= args.minimum_complete_rate,
        "category_regression_limit": not regressions,
    }
    automated_pass = all(checks.values())
    status = "PASS_PENDING_HUMAN_REVIEW" if automated_pass else "FAIL"

    regression_rows = (
        "\n".join(f"| {name} | {drop:.3f} |" for name, drop in regressions)
        if regressions
        else "| None | 0.000 |"
    )
    check_rows = "\n".join(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in checks.items()
    )

    report = f"""# PHASE 1 STOP GATE

## Purpose

Compare the untouched base model with the all-layer dense-SFT adapter using the
same benchmark and decoding configuration. This automated gate is evidence for
human review; it does not approve Phase 2 by itself.

## Inputs

- Base summary: `{args.base_summary}`
- Adapter summary: `{args.adapter_summary}`

## Comparison

| Metric | Base | Dense SFT | Difference |
|---|---:|---:|---:|
| Mean heuristic score | {base_score:.3f} | {adapter_score:.3f} | {score_gain:+.3f} |
| Median reply tokens | {float(base['median_reply_tokens']):.1f} | {float(adapter['median_reply_tokens']):.1f} | {float(adapter['median_reply_tokens']) - float(base['median_reply_tokens']):+.1f} |
| P90 reply tokens | {float(base['p90_reply_tokens']):.1f} | {float(adapter['p90_reply_tokens']):.1f} | {float(adapter['p90_reply_tokens']) - float(base['p90_reply_tokens']):+.1f} |
| Complete utterance rate | {float(base['complete_utterance_rate']):.3f} | {float(adapter['complete_utterance_rate']):.3f} | {float(adapter['complete_utterance_rate']) - float(base['complete_utterance_rate']):+.3f} |
| Peak evaluation VRAM (GB) | {float(base['peak_vram_gb']):.3f} | {float(adapter['peak_vram_gb']):.3f} | {float(adapter['peak_vram_gb']) - float(base['peak_vram_gb']):+.3f} |

## Automated Checks

| Check | Result |
|---|---|
{check_rows}

## Category Regressions Above Allowed Limit

| Category | Score drop |
|---|---:|
{regression_rows}

## Gate Status

`{status}`

## Required Human Review

Before starting the layer scan, inspect the lowest-scoring adapter cases and
confirm that the model is:

- naturally brief rather than mechanically truncated;
- varied rather than relying on canned replies;
- appropriate across the represented personalities and categories;
- not materially worse on any important category;
- clearly improved enough to justify the next experiment.

Record the final human decision below:

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
