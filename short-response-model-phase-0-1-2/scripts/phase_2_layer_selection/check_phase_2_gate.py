from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from phase2_common import ROOT, load_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stop Gate 2 from the completed static layer study.")
    p.add_argument("--results", type=Path, default=ROOT / "artifacts/phase_2_layer_study/phase_2_results.json")
    p.add_argument("--output-report", type=Path, default=ROOT / "reports/PHASE_2_STOP_GATE.md")
    p.add_argument("--quality-tolerance", type=float, default=0.02)
    p.add_argument("--minimum-vram-reduction", type=float, default=0.15)
    p.add_argument("--maximum-category-regression", type=float, default=0.10)
    p.add_argument("--minimum-complete-rate", type=float, default=0.99)
    p.add_argument("--maximum-p90-tokens", type=float, default=24.0)
    p.add_argument("--minimum-repeat-seeds", type=int, default=3)
    p.add_argument("--maximum-repeat-score-stdev", type=float, default=0.03)
    p.add_argument("--maximum-critical-failures", type=int, default=0)
    p.add_argument("--minimum-semantic-pass-rate", type=float, default=0.90)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.results)
    all_ref = data["all_layer_reference"]
    all_score = float(all_ref["mean_heuristic_score"])
    all_categories = all_ref.get("category_scores", {})
    rows = [*data["ranked_single_layers"], *data["pair_results"], *data["controls"]]
    eligible: list[dict[str, Any]] = []
    for row in rows:
        layers = row.get("layers")
        if not isinstance(layers, list) or len(layers) not in (1, 2, 4):
            continue
        category_regressions = {
            k: float(v) - float(row.get("category_scores", {}).get(k, 0.0))
            for k, v in all_categories.items()
            if float(v) - float(row.get("category_scores", {}).get(k, 0.0)) > args.maximum_category_regression
        }
        passes = (
            all_score - float(row["mean_heuristic_score"]) <= args.quality_tolerance
            and float(row["vram_reduction_vs_all"]) >= args.minimum_vram_reduction
            and float(row["complete_utterance_rate"]) >= args.minimum_complete_rate
            and float(row["p90_reply_tokens"]) <= args.maximum_p90_tokens
            and int(row.get("critical_failure_count", 0)) <= args.maximum_critical_failures
            and float(row.get("semantic_pass_rate", 1.0)) >= args.minimum_semantic_pass_rate
            and not category_regressions
        )
        row["gate_category_regressions"] = category_regressions
        row["gate_eligible"] = passes
        if passes:
            eligible.append(row)

    repeats = data.get("repeat_results", [])
    repeat_scores = [float(x["mean_heuristic_score"]) for x in repeats]
    repeat_stdev = statistics.pstdev(repeat_scores) if len(repeat_scores) > 1 else float("inf")
    stability_pass = len(repeats) >= args.minimum_repeat_seeds and repeat_stdev <= args.maximum_repeat_score_stdev
    automated_pass = bool(eligible) and stability_pass
    status = "PASS_PENDING_HUMAN_REVIEW" if automated_pass else "FAIL"
    best = sorted(eligible, key=lambda x: x["mean_heuristic_score"], reverse=True)[0] if eligible else None

    candidate_rows = "\n".join(
        f"| {r['variant']} | {r['layers']} | {r['mean_heuristic_score']:.3f} | "
        f"{r.get('critical_failure_count', 0)} | {r.get('major_issue_count', 0)} | "
        f"{r.get('semantic_pass_rate', 1.0):.3f} | {100*r['vram_reduction_vs_all']:.1f}% | "
        f"{'PASS' if r['gate_eligible'] else 'FAIL'} |"
        for r in sorted(rows, key=lambda x: x["mean_heuristic_score"], reverse=True)
    )
    report = f"""# PHASE 2 STOP GATE\n\n## Purpose\n\nAdvance only when a one-, two-, or four-layer static configuration matches the all-layer quality reference within tolerance and materially reduces training resources.\n\n## Candidate Checks\n\n| Variant | Layers | Score | VRAM reduction | Result |\n|---|---|---:|---:|---|\n{candidate_rows}\n\n## Repeated-Seed Stability\n\n- Repeat runs: {len(repeats)}\n- Mean score: {statistics.mean(repeat_scores) if repeat_scores else float('nan'):.3f}\n- Score standard deviation: {repeat_stdev:.4f}\n- Stability check: {'PASS' if stability_pass else 'FAIL'}\n\n## Automated Gate Status\n\n`{status}`\n\n## Recommended Static Candidate\n\n{('`' + best['variant'] + '` layers ' + str(best['layers'])) if best else 'No candidate passed every automated requirement.'}\n\n## Required Human Review\n\nReview `reports/PHASE_2_HUMAN_REVIEW_PACKET.md`, raw replies, and category scores for the recommended candidate, the all-layer reference, RANDOM4, and BOTTOM4. Confirm that the ranking is behaviorally meaningful and not a heuristic artifact. A critical semantic failure is disqualifying even when the average heuristic score is high.\n\n- Human decision: `PENDING`\n- Reviewer:\n- Date:\n- Notes:\n\nDynamic IST-style selection is optional after this static gate. It must be reported separately because gradient masking does not automatically prove optimizer-state memory savings.\n"""
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(report, encoding="utf-8")
    print(f"Phase 2 gate report: {args.output_report}")
    print(f"Automated status: {status}")
    if not automated_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
