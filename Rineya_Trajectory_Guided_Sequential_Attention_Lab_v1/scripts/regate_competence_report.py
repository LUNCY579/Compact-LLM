"""Re-evaluate a stored competence report under the current gate.

A competence report carries two things: the raw per-run measurements, and a
verdict computed from them under whatever thresholds were in force when it was
written. Only the first is evidence. When the gate changes — as it did when
`competence_is_reliable_across_seeds` was added, and again when that check was
split into a competence part and a power part — the stored verdict becomes
stale, and `require_go` rejects it on schema version rather than silently
honouring a rule that no longer exists.

This script recomputes the verdict from the stored raw runs, so a report does
not have to be *re-measured* just because the gate moved. It never edits the
input: the re-gated report is written to a new path and records what it was
derived from.

It cannot manufacture a `GO`. If the raw runs do not satisfy the current
thresholds, the output says `NO_GO`, and re-running this script will keep saying
`NO_GO` for as long as that is true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from tgsa.phase1_competence import (
    ArmCompetence,
    CompetenceThresholds,
    evaluate_competence,
)

# Frozen in PHASE1_V4_PILOT_REPORT.md section 7, outcome-blind, before any v4
# confirmatory run. Changing a value here is an amendment, not a tweak.
FROZEN = CompetenceThresholds(
    minimum_margin_over_trivial=0.030,
    minimum_absolute_accuracy=0.070,
    maximum_answer_nll_fraction_of_uniform=0.97,
    minimum_distinct_predictions=4,
    maximum_family_accuracy=0.95,
    maximum_generalization_gap=0.10,
    maximum_seed_coefficient_of_variation=0.35,
    minimum_seed_runs=3,
)
PRIMARY_FAMILIES = ("pointer_chase", "state_update")


def arms_from_runs(runs: list[dict]) -> list[ArmCompetence]:
    """Rebuild per-arm evidence from the raw per-run measurements."""

    arms: list[ArmCompetence] = []
    for arm in sorted({run["arm"] for run in runs}):
        subset = [run for run in runs if run["arm"] == arm]
        families = sorted({f for run in subset for f in run["holdout"]["per_family_accuracy"]})
        arms.append(
            ArmCompetence(
                arm=arm,
                holdout_accuracy=statistics.fmean(r["holdout"]["accuracy"] for r in subset),
                seen_accuracy=statistics.fmean(r["seen"]["accuracy"] for r in subset),
                answer_nll=statistics.fmean(r["holdout"]["answer_nll"] for r in subset),
                distinct_predictions=min(
                    int(r["holdout"]["distinct_predictions"]) for r in subset
                ),
                per_family_accuracy={
                    family: statistics.fmean(
                        r["holdout"]["per_family_accuracy"].get(family, 0.0) for r in subset
                    )
                    for family in families
                },
                per_seed_holdout_accuracy=[r["holdout"]["accuracy"] for r in subset],
            )
        )
    return arms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain-size", type=int, default=32)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    runs = source.get("runs")
    if not runs:
        raise ValueError(
            f"{args.input} carries no raw runs; a stored verdict alone is not evidence "
            f"and cannot be re-gated"
        )

    report = evaluate_competence(
        arms_from_runs(runs),
        thresholds=FROZEN,
        trivial_baselines=source["trivial_baselines"],
        uniform_answer_nll=math.log(args.domain_size),
        primary_families=PRIMARY_FAMILIES,
    )
    report["regime"] = source["regime"]
    report["runs"] = runs
    report["derived_from"] = {
        "path": args.input.name,
        "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "original_schema_version": source.get("schema_version"),
        "original_verdict": source.get("verdict"),
        "note": (
            "verdict recomputed from the stored raw runs under the current gate; "
            "no measurement was repeated and the input was not modified"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{args.input.name}: {source.get('verdict')} ({source.get('schema_version')})")
    print(f"{args.output.name}: {report['verdict']} ({report['schema_version']})")
    for name, ok in report["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for reason in report["reasons"]:
        print(f"  reason: {reason}")
    return 0 if report["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
