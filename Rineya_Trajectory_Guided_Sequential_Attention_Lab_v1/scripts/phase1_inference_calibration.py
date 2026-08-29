#!/usr/bin/env python3
"""Stage 3: high-precision null-calibration and coverage audit of the estimators.

Outcome-blind. Simulates the registered crossed seed x example design and asks,
for each candidate interval estimator and each candidate seed count:

* what is the empirical type-I rejection rate of the registered conjunction gate;
* what is the coverage of the nominal 95% interval;
* does the estimated standard error match the true sampling variability.

It also runs an explicit misspecification probe that quantifies how much of a
reported null rate is attributable to treating the registered FIXED strata as a
random effect, which is not a source of sampling variability for this estimand.

No TGSA confirmatory outcome is read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from tgsa.phase1_inference import INTERVAL_ESTIMATORS
from tgsa.phase1_runner import atomic_write_json
from tgsa.phase1_simulation import (
    SIMULATION_SCHEMA_VERSION,
    DesignPoint,
    StratumLayout,
    evaluate_design,
)

# Bootstrap estimators cost orders of magnitude more per trial, so they run on a
# decision-relevant subset of the grid. What is dropped is logged, never silent.
BOOTSTRAP_SEED_COUNTS = (5, 10, 15, 20)
BOOTSTRAP_SEED_SDS = (0.02, 0.04)


def _job(payload: dict) -> dict:
    layout = StratumLayout.balanced(payload["stratum_count"], payload["examples_per_stratum"])
    design = DesignPoint(
        seed_count=payload["seed_count"],
        true_effect=payload["true_effect"],
        seed_sd=payload["seed_sd"],
        example_sd=payload["example_sd"],
        disagreement_rate=payload["disagreement_rate"],
        stratum_sd=payload["stratum_sd"],
    )
    result = evaluate_design(
        design,
        layout,
        estimator=payload["estimator"],
        confidence=payload["confidence"],
        practical_threshold=payload["practical_threshold"],
        trials=payload["trials"],
        rng_seed=payload["rng_seed"],
        bootstrap_samples=payload["bootstrap_samples"],
        chunk=payload["chunk"],
    )
    result["layout"] = layout.to_dict()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--bootstrap-trials", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=999)
    parser.add_argument("--stratum-count", type=int, default=17)
    parser.add_argument("--examples-per-stratum", type=int, default=70)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--practical-threshold", type=float, default=0.02)
    parser.add_argument("--example-sd", type=float, default=0.05)
    parser.add_argument("--disagreement-rate", type=float, default=0.30)
    parser.add_argument("--rng-seed", type=int, default=91273)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--seed-counts", default="5,8,10,12,15,20")
    parser.add_argument("--seed-sds", default="0.01,0.02,0.04,0.06")
    args = parser.parse_args()

    seed_counts = [int(value) for value in args.seed_counts.split(",")]
    seed_sds = [float(value) for value in args.seed_sds.split(",")]

    jobs: list[dict] = []
    dropped: list[str] = []
    index = 0
    for estimator in INTERVAL_ESTIMATORS:
        is_bootstrap = estimator.startswith("crossed_pigeonhole")
        for seed_count in seed_counts:
            for seed_sd in seed_sds:
                if is_bootstrap and (
                    seed_count not in BOOTSTRAP_SEED_COUNTS or seed_sd not in BOOTSTRAP_SEED_SDS
                ):
                    dropped.append(f"{estimator}@m={seed_count},sigma_a={seed_sd}")
                    continue
                index += 1
                jobs.append(
                    {
                        "estimator": estimator,
                        "seed_count": seed_count,
                        "seed_sd": seed_sd,
                        "true_effect": 0.0,
                        "stratum_sd": 0.0,
                        "example_sd": args.example_sd,
                        "disagreement_rate": args.disagreement_rate,
                        "stratum_count": args.stratum_count,
                        "examples_per_stratum": args.examples_per_stratum,
                        "confidence": args.confidence,
                        "practical_threshold": args.practical_threshold,
                        "trials": args.bootstrap_trials if is_bootstrap else args.trials,
                        "bootstrap_samples": args.bootstrap_samples,
                        "rng_seed": args.rng_seed + 7919 * index,
                        "chunk": 100 if is_bootstrap else 250,
                    }
                )

    # Misspecification probe: identical design, strata treated as random.
    probe_jobs: list[dict] = []
    for stratum_sd in (0.0, 0.02, 0.05):
        index += 1
        probe_jobs.append(
            {
                "estimator": "crossed_cluster_t",
                "seed_count": 5,
                "seed_sd": 0.02,
                "true_effect": 0.0,
                "stratum_sd": stratum_sd,
                "example_sd": args.example_sd,
                "disagreement_rate": args.disagreement_rate,
                "stratum_count": args.stratum_count,
                "examples_per_stratum": args.examples_per_stratum,
                "confidence": args.confidence,
                "practical_threshold": args.practical_threshold,
                "trials": args.trials,
                "bootstrap_samples": args.bootstrap_samples,
                "rng_seed": args.rng_seed + 7919 * index,
                "chunk": 250,
            }
        )

    print(f"calibration: {len(jobs)} grid conditions + {len(probe_jobs)} probes", flush=True)
    if dropped:
        print(f"bootstrap estimators restricted to a subset; skipped: {sorted(dropped)}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_job, jobs))
        probes = list(pool.map(_job, probe_jobs))

    report = {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "kind": "null_calibration_and_coverage",
        "confidence": args.confidence,
        "practical_threshold": args.practical_threshold,
        "example_sd": args.example_sd,
        "disagreement_rate": args.disagreement_rate,
        "trials_analytic": args.trials,
        "trials_bootstrap": args.bootstrap_trials,
        "bootstrap_samples": args.bootstrap_samples,
        "seed_counts": seed_counts,
        "seed_sds": seed_sds,
        "bootstrap_grid_restricted_to": {
            "seed_counts": list(BOOTSTRAP_SEED_COUNTS),
            "seed_sds": list(BOOTSTRAP_SEED_SDS),
        },
        "skipped_conditions": sorted(dropped),
        "conditions": results,
        "stratum_misspecification_probe": probes,
        "interpretation": (
            "Outcome-blind. The registered strata are a FIXED set, so the default "
            "stratum_sd=0 is the correctly specified world. The probe reports what "
            "happens when a stratum random effect is injected, which inflates true "
            "variance in a way no correctly specified estimator can track."
        ),
    }
    atomic_write_json(args.output.resolve(), report)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
