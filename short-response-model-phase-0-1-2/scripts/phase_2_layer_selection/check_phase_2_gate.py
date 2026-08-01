from __future__ import annotations

"""Phase 2 selective-tuning gate using relative opportunity cost only."""

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

from phase2_common import ROOT, load_json

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.opportunity_evaluation import evaluate_opportunity, pareto_frontier
from scripts.common.quality_evaluation import require_quality_v3, summary_quality_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 by training-resource gain versus quality opportunity loss.")
    parser.add_argument("--results", type=Path, default=ROOT / "artifacts/phase_2_layer_study/phase_2_results.json")
    parser.add_argument("--output-report", type=Path, default=ROOT / "reports/PHASE_2_STOP_GATE.md")
    parser.add_argument("--quality-profile", choices=("balanced", "logic_oriented", "conversational_permissive"), default="balanced")
    # Deprecated aliases remain accepted so old commands do not break. They do not control selection.
    for flag in (
        "--quality-tolerance", "--minimum-vram-reduction", "--maximum-category-regression",
        "--minimum-complete-rate", "--maximum-p90-tokens", "--minimum-repeat-seeds",
        "--maximum-repeat-score-stdev", "--minimum-quality-pass-rate",
        "--minimum-semantic-pass-rate", "--minimum-worst-decile-quality",
    ):
        parser.add_argument(flag, type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.results)
    reference = data["all_layer_reference"]
    require_quality_v3(reference, "Phase 2 all-layer reference")
    rows = [*data.get("ranked_single_layers", []), *data.get("pair_results", []), *data.get("controls", [])]
    assessed: list[dict[str, Any]] = []
    for row in rows:
        layers = row.get("layers")
        if not isinstance(layers, list) or len(layers) not in (1, 2, 4):
            continue
        require_quality_v3(row, str(row.get("variant", "Phase 2 candidate")))
        resource_gain = row.get("vram_reduction_vs_all")
        if resource_gain is None:
            all_trainable = data.get("all_layer_reference", {}).get("trainable_parameter_count")
            candidate_trainable = row.get("trainable_parameter_count")
            if all_trainable and candidate_trainable is not None:
                resource_gain = max(0.0, 1.0 - float(candidate_trainable) / float(all_trainable))
        assessment = evaluate_opportunity(
            reference,
            row,
            resource_gain=None if resource_gain is None else float(resource_gain),
            resource_name="training_resource_reduction",
            profile=args.quality_profile,
        )
        row["gate_assessment"] = assessment
        row["gate_eligible"] = assessment["eligible"]
        assessed.append(row)

    eligible = [row for row in assessed if row["gate_eligible"]]
    frontier_ids = {id(row) for row in pareto_frontier([
        {**row, **row["gate_assessment"]} for row in assessed
    ])}
    # Recompute membership by (variant, margin) because merged dictionaries have new identities.
    frontier_keys = {
        (str(row.get("variant")), float(row.get("opportunity_margin", 0.0)))
        for row in pareto_frontier([{**row, **row["gate_assessment"]} for row in assessed])
    }
    for row in assessed:
        row["pareto_frontier"] = (
            str(row.get("variant")), float(row["gate_assessment"].get("opportunity_margin", 0.0))
        ) in frontier_keys

    repeats = data.get("repeat_results", [])
    repeat_scores: list[float] = []
    for row in repeats:
        require_quality_v3(row, "Phase 2 repeated-seed evidence")
        repeat_scores.append(summary_quality_score(row))
    repeat_stdev = statistics.pstdev(repeat_scores) if len(repeat_scores) > 1 else None
    best = max(
        eligible,
        key=lambda row: (
            float(row["gate_assessment"]["opportunity_margin"]),
            float(row["gate_assessment"]["resource_gain_fraction"]),
            -float(row["gate_assessment"]["quality_opportunity_loss_fraction"]),
        ),
        default=None,
    )
    status = "PASS_PENDING_HUMAN_REVIEW" if best else "NO_PROPORTIONALLY_WORTHWHILE_CANDIDATE"
    table = "\n".join(
        f"| {row.get('variant')} | {row.get('layers')} | {summary_quality_score(row):.4f} | "
        f"{float(row['gate_assessment']['resource_gain_fraction']):.3%} | "
        f"{float(row['gate_assessment']['quality_opportunity_loss_fraction']):.3%} | "
        f"{float(row['gate_assessment']['opportunity_margin']):+.3%} | "
        f"{'YES' if row.get('pareto_frontier') else 'NO'} | "
        f"{'WORTHWHILE' if row.get('gate_eligible') else 'NOT WORTHWHILE'} |"
        for row in sorted(assessed, key=lambda item: float(item["gate_assessment"]["opportunity_margin"]), reverse=True)
    )
    deprecated = [name for name, value in vars(args).items() if name not in {"results", "output_report", "quality_profile"} and value is not None]
    report = f"""# PHASE 2 STOP GATE

## Governing rule

Selective tuning advances only when proportional training-resource reduction is greater than combined proportional quality opportunity loss. No absolute quality floor controls promotion.

| Variant | Layers | Quality | Resource gain | Quality opportunity loss | Opportunity margin | Pareto | Decision |
|---|---|---:|---:|---:|---:|---:|---|
{table}

## Repeated-seed evidence

- Repeat runs recorded: `{len(repeat_scores)}`
- Mean quality: `{statistics.mean(repeat_scores) if repeat_scores else 'N/A'}`
- Quality standard deviation (diagnostic only): `{repeat_stdev if repeat_stdev is not None else 'N/A'}`

Repeated seeds inform uncertainty and human review; no fixed repeat-count or standard-deviation threshold accepts or rejects a candidate.

## Automated status

`{status}`

- Recommended candidate: `{best.get('variant') if best else 'NONE'}`
- Quality profile: `{args.quality_profile}`
- Deprecated threshold arguments supplied and ignored: `{deprecated or 'None'}`

## Human decision

- Human decision: `PENDING`
- Reviewer:
- Date:
- Notes:
"""
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(report, encoding="utf-8")
    print(args.output_report)
    if not best:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
