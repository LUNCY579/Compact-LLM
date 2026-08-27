#!/usr/bin/env python3
"""Evaluate the fail-closed real-peer Phase-C advancement gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rineya_pcsm.hashing import canonical_json_hash, sha256_file
from rineya_pcsm.manifest import atomic_write_json, utc_now
from rineya_pcsm.statistics import paired_comparison


def mean_loss(records: list[dict[str, Any]]) -> float:
    if not records:
        raise ValueError("Cannot aggregate an empty arm")
    return sum(float(record["evaluation"]["mean_nll"]) for record in records) / len(
        records
    )


def self_hash_valid(value: dict[str, Any], key: str) -> bool:
    declared = value.get(key)
    payload = dict(value)
    payload.pop(key, None)
    return isinstance(declared, str) and declared == canonical_json_hash(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.run.read_text(encoding="utf-8"))
    plans = json.loads(args.plans.read_text(encoding="utf-8"))
    budget = json.loads(args.budget.read_text(encoding="utf-8"))
    qualification = budget["qualification"]
    expected_seeds = [int(seed) for seed in qualification["seeds"]]
    expected_budgets = [float(value) for value in qualification["budget_fractions"]]
    required_arms = set(budget.get("required_arms", qualification["arms"]))
    required_arms.add("layer_shuffled_peer_pcsm")
    reasons: list[str] = []
    budget_sha = sha256_file(args.budget)
    plans_sha = sha256_file(args.plans)
    run_sha = sha256_file(args.run)
    if not self_hash_valid(run, "run_sha256"):
        reasons.append("run self-seal is missing or invalid")
    if not self_hash_valid(plans, "result_sha256"):
        reasons.append("plan-result self-seal is missing or invalid")
    if not self_hash_valid(budget, "budget_manifest_sha256"):
        reasons.append("budget self-seal is missing or invalid")
    if any(
        not self_hash_valid(record, "record_sha256")
        for record in run.get("records", [])
    ):
        reasons.append("one or more run record self-seals are invalid")
    if any(
        not self_hash_valid(record, "record_sha256")
        for record in plans.get("records", [])
    ):
        reasons.append("one or more plan record self-seals are invalid")
    if run.get("status") != "complete" or run.get("stage") != "qualification":
        reasons.append("complete qualification run is absent")
    if plans.get("status") != "complete" or plans.get("stage") != "qualification":
        reasons.append("complete qualification plan set is absent")
    if run.get("budget_manifest_sha256") != budget_sha:
        reasons.append("run is not bound to the supplied budget manifest")
    if run.get("plan_result_sha256") != plans_sha:
        reasons.append("run is not bound to the supplied plan result")
    base_hash = budget["base_model"]["weight_sha256"]
    if (
        run.get("base_weight_sha256") != base_hash
        or plans.get("base_fingerprint") != base_hash
    ):
        reasons.append("base checkpoint identity differs across evidence")

    grouped: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for record in run.get("records", []):
        grouped[
            (int(record["seed"]), float(record["budget_fraction"]), str(record["arm"]))
        ].append(record)
    expected_run_cells = {
        (seed, fraction, arm)
        for seed in expected_seeds
        for fraction in expected_budgets
        for arm in sorted(required_arms)
    }
    missing = sorted(cell for cell in expected_run_cells if not grouped[cell])
    duplicate_run_cells = sorted(
        cell for cell in expected_run_cells if len(grouped[cell]) != 1
    )
    if missing:
        reasons.append(f"missing {len(missing)} required seed/budget/arm cells")
    if duplicate_run_cells:
        reasons.append(f"non-unique {len(duplicate_run_cells)} run cells")
    extra_run_cells = set(grouped) - expected_run_cells
    if extra_run_cells:
        reasons.append(f"unexpected {len(extra_run_cells)} run cells")

    all_audits_valid = all(
        bool(record.get("delta_audit", {}).get("valid"))
        and bool(record.get("unplanned_parameter_hash_unchanged"))
        for record in run.get("records", [])
    )
    if not all_audits_valid:
        reasons.append("one or more exact native-delta audits failed")
    if (
        run.get("sealed_test_read") is not False
        or plans.get("sealed_test_read") is not False
    ):
        reasons.append("sealed-data non-access is not proven")

    matched_support = True
    for seed in expected_seeds:
        for fraction in expected_budgets:
            counts = {
                int(record["k"])
                for arm in required_arms - {"no_update_frozen"}
                for record in grouped[(seed, fraction, arm)]
            }
            if len(counts) != 1:
                matched_support = False
    if not matched_support:
        reasons.append("sparse arms do not have identical exact K")

    plan_groups: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for record in plans.get("records", []):
        plan_groups[
            (
                int(record["seed"]),
                float(record["budget_fraction"]),
                str(record["arm"]),
            )
        ].append(record)
    expected_plan_cells = {
        (seed, fraction, arm)
        for seed in expected_seeds
        for fraction in expected_budgets
        for arm in required_arms - {"no_update_frozen"}
    }
    if set(plan_groups) != expected_plan_cells or any(
        len(plan_groups[cell]) != 1 for cell in expected_plan_cells
    ):
        reasons.append("plan cells are missing, duplicated, or unexpected")
    capacity_matched = True
    for seed in expected_seeds:
        for fraction in expected_budgets:
            records = [
                plan_groups.get((seed, fraction, arm), [None])[0]
                for arm in (
                    "coordinate_shuffled_peer_pcsm",
                    "layer_shuffled_peer_pcsm",
                    "real_peer_pcsm_calibrated",
                )
            ]
            if any(record is None for record in records):
                capacity_matched = False
                continue
            signatures = {
                (
                    record["k"],
                    record["metadata"].get("steps"),
                    record["metadata"].get("generator_learning_rate"),
                    len(record.get("calibration_trace", [])),
                )
                for record in records
                if record is not None
            }
            if len(signatures) != 1:
                capacity_matched = False
    if not capacity_matched:
        reasons.append("real/shuffled generator capacity or meta compute differs")

    statistics: dict[str, Any] = {}
    if not missing and not duplicate_run_cells:
        real_losses: list[float] = []
        coordinate_losses: list[float] = []
        layer_losses: list[float] = []
        for seed in expected_seeds:
            real_losses.append(
                mean_loss(
                    [
                        record
                        for fraction in expected_budgets
                        for record in grouped[
                            (seed, fraction, "real_peer_pcsm_calibrated")
                        ]
                    ]
                )
            )
            coordinate_losses.append(
                mean_loss(
                    [
                        record
                        for fraction in expected_budgets
                        for record in grouped[
                            (seed, fraction, "coordinate_shuffled_peer_pcsm")
                        ]
                    ]
                )
            )
            layer_losses.append(
                mean_loss(
                    [
                        record
                        for fraction in expected_budgets
                        for record in grouped[
                            (seed, fraction, "layer_shuffled_peer_pcsm")
                        ]
                    ]
                )
            )
        coordinate_comparison = paired_comparison(
            coordinate_losses, real_losses, seed=20260827
        )
        layer_comparison = paired_comparison(layer_losses, real_losses, seed=20260828)
        coordinate_differences = coordinate_comparison["differences"]
        layer_differences = layer_comparison["differences"]
        coordinate_wins = sum(value > 0 for value in coordinate_differences)
        layer_wins = sum(value > 0 for value in layer_differences)
        coordinate_ci = coordinate_comparison["confidence_interval"]
        layer_ci = layer_comparison["confidence_interval"]
        statistics = {
            "difference_definition": (
                "shuffled_loss_minus_real_loss; positive favors real"
            ),
            "per_seed_real_loss": dict(zip(expected_seeds, real_losses, strict=True)),
            "coordinate_shuffle": coordinate_comparison,
            "coordinate_wins": coordinate_wins,
            "layer_identity_permutation": layer_comparison,
            "layer_wins": layer_wins,
        }
        if coordinate_wins < 4:
            reasons.append(
                "real peers beat coordinate-shuffled peers in fewer than 4/5 seeds"
            )
        if float(coordinate_ci["low"]) <= 0:
            reasons.append("paired coordinate-shuffle CI does not exclude zero")
        if layer_wins < 4 or float(layer_ci["low"]) <= 0:
            reasons.append(
                "real superiority does not survive layer-identity permutation"
            )

    passed = not reasons
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": utc_now(),
        "analysis_class": "PRE_SPECIFIED_CONFIRMATORY",
        "verdict": "PASS_ADVANCE_TO_PHASE_D" if passed else "STOP_NO_PHASE_D",
        "passed": passed,
        "reasons": reasons,
        "required_seed_count": 5,
        "matched_support_counts": matched_support,
        "capacity_and_compute_matched": capacity_matched,
        "all_delta_audits_valid": all_audits_valid,
        "statistics": statistics,
        "evidence": {
            "run_sha256": run_sha,
            "plans_sha256": plans_sha,
            "budget_sha256": budget_sha,
        },
        "sealed_test_read": False,
    }
    result["gate_sha256"] = canonical_json_hash(result)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
