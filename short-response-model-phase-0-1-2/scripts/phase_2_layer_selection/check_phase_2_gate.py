from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

from phase2_common import ROOT, load_json

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.quality_evaluation import require_quality_v2
from scripts.common.quality_gate import compare_quality_summaries, summary_quality_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stop Gate 2 with quality-v2 evidence.")
    p.add_argument("--results", type=Path, default=ROOT / "artifacts/phase_2_layer_study/phase_2_results.json")
    p.add_argument("--output-report", type=Path, default=ROOT / "reports/PHASE_2_STOP_GATE.md")
    p.add_argument("--quality-tolerance", type=float, default=0.02)
    p.add_argument("--minimum-vram-reduction", type=float, default=0.15)
    p.add_argument("--maximum-category-regression", type=float, default=0.08)
    p.add_argument("--minimum-complete-rate", type=float, default=0.99)
    p.add_argument("--maximum-p90-tokens", type=float, default=24.0)
    p.add_argument("--minimum-repeat-seeds", type=int, default=3)
    p.add_argument("--maximum-repeat-score-stdev", type=float, default=0.03)
    p.add_argument("--minimum-quality-pass-rate", type=float, default=0.95)
    p.add_argument("--minimum-semantic-pass-rate", type=float, default=0.98)
    p.add_argument("--minimum-worst-decile-quality", type=float, default=0.70)
    return p.parse_args()


def as_summary(row: dict[str, Any]) -> dict[str, Any]:
    require_quality_v2(row, str(row.get("variant", "candidate")))
    return row


def main() -> None:
    args = parse_args()
    data = load_json(args.results)
    all_ref = as_summary(data["all_layer_reference"])
    rows = [*data["ranked_single_layers"], *data["pair_results"], *data["controls"]]
    eligible: list[dict[str, Any]] = []
    for row in rows:
        layers = row.get("layers")
        if not isinstance(layers, list) or len(layers) not in (1, 2, 4):
            continue
        as_summary(row)
        comparison = compare_quality_summaries(
            all_ref,
            row,
            max_mean_quality_loss=args.quality_tolerance,
            max_quality_index_loss=args.quality_tolerance,
            max_worst_decile_loss=0.05,
            max_category_quality_loss_allowed=args.maximum_category_regression,
            max_semantic_pass_loss=0.02,
            max_reference_alignment_loss=0.08,
            min_quality_pass_rate=args.minimum_quality_pass_rate,
            min_intelligibility_pass_rate=1.0,
            min_semantic_pass_rate=args.minimum_semantic_pass_rate,
            min_complete_utterance_rate=args.minimum_complete_rate,
            max_hard_failure_cases=0,
            max_critical_failures=0,
            max_major_issue_cases=int(all_ref.get("major_case_count", 0)),
        )
        passes = (
            comparison["eligible"]
            and float(row["vram_reduction_vs_all"]) >= args.minimum_vram_reduction
            and float(row["p90_reply_tokens"]) <= args.maximum_p90_tokens
            and float(row.get("worst_decile_mean_quality_score", 0.0)) >= args.minimum_worst_decile_quality
        )
        row["gate_assessment"] = comparison
        row["gate_eligible"] = passes
        if passes:
            eligible.append(row)

    repeats = [as_summary(row) for row in data.get("repeat_results", [])]
    repeat_scores = [summary_quality_score(row) for row in repeats]
    repeat_stdev = statistics.pstdev(repeat_scores) if len(repeat_scores) > 1 else float("inf")
    stability_pass = len(repeats) >= args.minimum_repeat_seeds and repeat_stdev <= args.maximum_repeat_score_stdev
    automated_pass = bool(eligible) and stability_pass
    status = "PASS_PENDING_HUMAN_REVIEW" if automated_pass else "FAIL"
    best = max(
        eligible,
        key=lambda row: (
            float(row.get("worst_decile_mean_quality_score", 0.0)),
            summary_quality_score(row),
            float(row.get("quality_pass_rate", 0.0)),
            float(row.get("vram_reduction_vs_all", 0.0)),
        ),
        default=None,
    )

    candidate_rows = "\n".join(
        f"| {r['variant']} | {r['layers']} | {summary_quality_score(r):.3f} | "
        f"{float(r.get('worst_decile_mean_quality_score', 0.0)):.3f} | "
        f"{float(r.get('quality_pass_rate', 0.0)):.3f} | "
        f"{int(r.get('hard_failure_case_count', 0))} | {int(r.get('major_case_count', 0))} | "
        f"{100*float(r['vram_reduction_vs_all']):.1f}% | {'PASS' if r.get('gate_eligible') else 'FAIL'} |"
        for r in sorted(rows, key=lambda x: summary_quality_score(x), reverse=True)
    )
    report = f"""# PHASE 2 STOP GATE

## Purpose

Advance only when a one-, two-, or four-layer configuration preserves mean,
tail, category, semantic, intelligibility, and completion quality while reducing
training resources.

## Candidate Checks

| Variant | Layers | Quality | Worst decile | Pass rate | Hard fails | Major cases | VRAM reduction | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
{candidate_rows}

## Repeated-Seed Stability

- Repeat runs: {len(repeats)}
- Mean quality: {statistics.mean(repeat_scores) if repeat_scores else float('nan'):.3f}
- Quality-score standard deviation: {repeat_stdev:.4f}
- Stability check: {'PASS' if stability_pass else 'FAIL'}

## Automated Gate Status

`{status}`

## Recommended Static Candidate

{('`' + best['variant'] + '`') if best else 'None passed all quality-v2 floors.'}

## Human Decision

Inspect all hard/major flags, low-tail cases, and repeat-seed differences.

- Human decision: `PENDING`
- Reviewer:
- Date:
- Notes:
"""
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(report, encoding="utf-8")
    print(args.output_report)
    if not automated_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
