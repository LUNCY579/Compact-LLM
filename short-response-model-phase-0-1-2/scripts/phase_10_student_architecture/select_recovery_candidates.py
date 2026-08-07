from __future__ import annotations

"""Rank unrecovered Phase 10 candidates and select a bounded recovery set."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, write_json
from scripts.common.opportunity_evaluation import evaluate_opportunity, resource_reduction
from scripts.phase_10_student_architecture.phase10_common import candidate_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Phase 10 candidates for recovery KD.")
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], help="name=summary.json")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--purpose", choices=("recovery", "final"), default="recovery")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    reference = read_json(args.reference_summary)
    packets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in args.candidate:
        name, path = candidate_assignment(raw)
        if name in seen:
            raise ValueError(f"Duplicate candidate: {name}")
        seen.add(name)
        summary = read_json(path)
        gain = resource_reduction(reference.get("model_weight_bytes"), summary.get("model_weight_bytes"))
        assessment = evaluate_opportunity(
            reference,
            summary,
            resource_gain=gain,
            resource_name="deployed_model_weight_bytes",
        )
        packets.append(
            {
                "name": name,
                "summary": str(path),
                "model": summary.get("model"),
                "adapter": summary.get("adapter"),
                "assessment": assessment,
                "resource_gain_fraction": gain,
            }
        )
    if not packets:
        raise ValueError("No Phase 10 candidates were supplied")
    # Recovery is allowed for promising raw candidates even when their current
    # opportunity margin is negative. Prioritize positive resource gain, then
    # the least quality damage / best margin. This is a search decision, not a
    # promotion decision.
    ranked = sorted(
        packets,
        key=lambda item: (
            float(item["resource_gain_fraction"] or 0.0) > 0.0,
            float((item["assessment"] or {}).get("opportunity_margin", float("-inf"))),
            -float((item["assessment"] or {}).get("quality_opportunity_loss_fraction", float("inf"))),
        ),
        reverse=True,
    )
    if args.purpose == "final":
        # Final-test selection is frozen on dev. Prefer candidates already
        # eligible under the relative-opportunity policy; if none are eligible,
        # still freeze the least-bad research candidate so the sealed test can
        # quantify the failure without using test to choose another model.
        eligible = [item for item in ranked if bool((item.get("assessment") or {}).get("eligible"))]
        selection_pool = eligible or ranked
        selected = [item["name"] for item in selection_pool[:1]]
        selection_status = "DEV_ELIGIBLE_CANDIDATE_SELECTED" if eligible else "NO_DEV_ELIGIBLE_CANDIDATE_RESEARCH_ONLY"
    else:
        selected = [item["name"] for item in ranked[: min(args.top_k, len(ranked))]]
        selection_status = "RECOVERY_SET_SELECTED"
    write_json(
        args.out,
        {
            "schema_version": "1.1",
            "reference_summary": str(args.reference_summary),
            "purpose": args.purpose,
            "top_k": args.top_k,
            "selection_status": selection_status,
            "selected_candidates": selected,
            "ranking": ranked,
            "interpretation": (
                "Recovery selection authorizes bounded recovery experiments only. Final selection is frozen on dev before "
                "the sealed test and does not use test metrics to choose a different candidate."
            ),
        },
    )
    print(json.dumps(selected))


if __name__ == "__main__":
    main()
