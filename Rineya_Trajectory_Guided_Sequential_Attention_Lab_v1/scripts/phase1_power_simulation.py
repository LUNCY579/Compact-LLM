#!/usr/bin/env python3
"""Stage 2: high-precision outcome-blind power audit across seed counts.

Runs the *registered* decision rule over a grid of candidate replication levels,
true effect sizes, and seed-effect standard deviations, and reports detection
probability with Monte Carlo standard errors and Wilson confidence intervals.

Outcome-blind by construction: the simulator draws synthetic paired differences
from an explicit generative model. No checkpoint, prediction file, or TGSA
confirmatory outcome is read.
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

from tgsa.phase1_inference import PRIMARY_ESTIMATOR, resolve_interval_estimator
from tgsa.phase1_runner import atomic_write_json
from tgsa.phase1_simulation import (
    SIMULATION_SCHEMA_VERSION,
    DesignPoint,
    StratumLayout,
    evaluate_design,
)


def _job(payload: dict) -> dict:
    layout = StratumLayout.balanced(payload["stratum_count"], payload["examples_per_stratum"])
    design = DesignPoint(
        seed_count=payload["seed_count"],
        true_effect=payload["true_effect"],
        seed_sd=payload["seed_sd"],
        example_sd=payload["example_sd"],
        disagreement_rate=payload["disagreement_rate"],
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


PRIMARY_GATE_KEYS = {
    "lower_bound_above_zero": "gate_lower_bound_above_zero",
    "conjunction": "gate_conjunction",
}


def power_curves(conditions: list[dict], *, gate_key: str = "gate_conjunction") -> dict[str, object]:
    """Detection probability versus seed count, keyed by effect and seed SD."""

    curves: dict[str, dict[str, list[dict[str, float]]]] = {}
    for entry in conditions:
        design = entry["design"]
        key = f"effect={design['true_effect']:.3f}|seed_sd={design['seed_sd']:.3f}"
        curves.setdefault(key, {"points": []})["points"].append(
            {
                "seed_count": design["seed_count"],
                "power": entry[gate_key]["rate"],
                "ci_low": entry[gate_key]["ci_low"],
                "ci_high": entry[gate_key]["ci_high"],
                "monte_carlo_standard_error": entry[gate_key]["monte_carlo_standard_error"],
                "conjunction_rate": entry["gate_conjunction"]["rate"],
            }
        )
    for value in curves.values():
        value["points"].sort(key=lambda point: point["seed_count"])
    return curves


def smallest_adequate_seed_count(
    conditions: list[dict],
    *,
    target_effect: float,
    central_seed_sd: float,
    elevated_seed_sd: float,
    target_power: float,
    robust_power: float,
    max_null_rate: float,
    gate_key: str = "gate_conjunction",
) -> dict[str, object]:
    """Smallest replication level meeting every registered adequacy criterion.

    A seed count qualifies only if, at that count, the lower confidence bound of
    the estimated detection probability clears the target at the central variance
    scenario, clears the weaker robustness target at the elevated one, and the
    upper confidence bound of the null rejection rate stays within tolerance.
    Using confidence bounds rather than point estimates keeps Monte Carlo noise
    from selecting a seed count that merely looks adequate.
    """

    by_count: dict[int, dict[str, dict]] = {}
    for entry in conditions:
        design = entry["design"]
        by_count.setdefault(design["seed_count"], {})[
            f"{design['true_effect']:.3f}|{design['seed_sd']:.3f}"
        ] = entry

    evaluations: list[dict[str, object]] = []
    for seed_count in sorted(by_count):
        cells = by_count[seed_count]
        central = cells.get(f"{target_effect:.3f}|{central_seed_sd:.3f}")
        elevated = cells.get(f"{target_effect:.3f}|{elevated_seed_sd:.3f}")
        null = cells.get(f"{0.0:.3f}|{central_seed_sd:.3f}")
        if central is None or elevated is None or null is None:
            continue
        central_ok = central[gate_key]["ci_low"] >= target_power
        elevated_ok = elevated[gate_key]["ci_low"] >= robust_power
        null_ok = null[gate_key]["ci_high"] <= max_null_rate
        evaluations.append(
            {
                "seed_count": seed_count,
                "central_power": central[gate_key]["rate"],
                "central_power_ci_low": central[gate_key]["ci_low"],
                "elevated_power": elevated[gate_key]["rate"],
                "elevated_power_ci_low": elevated[gate_key]["ci_low"],
                "null_rate": null[gate_key]["rate"],
                "null_rate_ci_high": null[gate_key]["ci_high"],
                "meets_central_power": central_ok,
                "meets_elevated_power": elevated_ok,
                "meets_null_control": null_ok,
                "adequate": bool(central_ok and elevated_ok and null_ok),
            }
        )

    adequate = [entry for entry in evaluations if entry["adequate"]]
    return {
        "criteria": {
            "target_effect": target_effect,
            "central_seed_sd": central_seed_sd,
            "elevated_seed_sd": elevated_seed_sd,
            "target_power": target_power,
            "robust_power": robust_power,
            "max_null_rate": max_null_rate,
            "rule": "select on confidence bounds, not point estimates",
        },
        "evaluations": evaluations,
        "recommended_seed_count": adequate[0]["seed_count"] if adequate else None,
        "no_candidate_is_adequate": not adequate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--estimator", default=PRIMARY_ESTIMATOR)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed-counts", default="5,8,10,12,15,20")
    parser.add_argument("--effects", default="0.000,0.005,0.010,0.015,0.020,0.030,0.040")
    parser.add_argument("--seed-sds", default="0.01,0.02,0.04,0.06")
    parser.add_argument("--stratum-count", type=int, default=17)
    parser.add_argument("--examples-per-stratum", type=int, default=70)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--practical-threshold", type=float, default=0.02)
    parser.add_argument("--example-sd", type=float, default=0.05)
    parser.add_argument("--disagreement-rate", type=float, default=0.30)
    parser.add_argument("--bootstrap-samples", type=int, default=999)
    parser.add_argument("--rng-seed", type=int, default=91273)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--robust-power", type=float, default=0.60)
    parser.add_argument("--max-null-rate", type=float, default=0.075)
    parser.add_argument("--central-seed-sd", type=float, default=0.02)
    parser.add_argument(
        "--primary-gate",
        choices=tuple(PRIMARY_GATE_KEYS),
        default="lower_bound_above_zero",
        help=(
            "lower_bound_above_zero: the registered v3 gate; the practical threshold is "
            "the effect the study is POWERED for and is reported descriptively. "
            "conjunction: the v2 rule, which caps power at 50%% when the true effect "
            "equals the threshold."
        ),
    )
    parser.add_argument("--elevated-seed-sd", type=float, default=0.04)
    args = parser.parse_args()

    resolve_interval_estimator(args.estimator)
    seed_counts = [int(value) for value in args.seed_counts.split(",")]
    effects = [float(value) for value in args.effects.split(",")]
    seed_sds = [float(value) for value in args.seed_sds.split(",")]

    jobs: list[dict] = []
    index = 0
    for seed_count in seed_counts:
        for effect in effects:
            for seed_sd in seed_sds:
                index += 1
                jobs.append(
                    {
                        "estimator": args.estimator,
                        "seed_count": seed_count,
                        "true_effect": effect,
                        "seed_sd": seed_sd,
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

    print(f"power audit: {len(jobs)} conditions x {args.trials} trials", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_job, jobs))

    gate_key = PRIMARY_GATE_KEYS[args.primary_gate]
    recommendation = smallest_adequate_seed_count(
        results,
        gate_key=gate_key,
        target_effect=args.practical_threshold,
        central_seed_sd=args.central_seed_sd,
        elevated_seed_sd=args.elevated_seed_sd,
        target_power=args.target_power,
        robust_power=args.robust_power,
        max_null_rate=args.max_null_rate,
    )

    report = {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "kind": "power_audit",
        "estimator": args.estimator,
        "primary_gate": args.primary_gate,
        "primary_gate_key": gate_key,
        "trials": args.trials,
        "confidence": args.confidence,
        "practical_threshold": args.practical_threshold,
        "example_sd": args.example_sd,
        "disagreement_rate": args.disagreement_rate,
        "seed_counts": seed_counts,
        "effects": effects,
        "seed_sds": seed_sds,
        "conditions": results,
        "power_curves": power_curves(results, gate_key=gate_key),
        "seed_count_recommendation": recommendation,
        "interpretation": (
            "Outcome-blind. Detection probability is the rate at which the registered "
            "conjunction gate fires (observed effect at or above the practical threshold "
            "AND lower confidence bound above zero). The null row is the type-I rate. "
            "Selection uses confidence bounds so Monte Carlo noise cannot pick a seed "
            "count that only looks adequate."
        ),
    }
    atomic_write_json(args.output.resolve(), report)
    print(f"wrote {args.output}", flush=True)
    print(json.dumps(recommendation["evaluations"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
