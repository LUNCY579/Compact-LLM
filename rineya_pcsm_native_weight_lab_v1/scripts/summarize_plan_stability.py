#!/usr/bin/env python3
"""Summarize support/attention stability from a complete fixed-plan set."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from rineya_pcsm.hashing import canonical_json_hash, sha256_file
from rineya_pcsm.manifest import atomic_write_json, utc_now
from rineya_pcsm.persistence import load_plan

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def layer_index(name: str) -> int:
    match = _LAYER.search(name)
    if match is None:
        raise ValueError(f"Missing layer index: {name}")
    return int(match.group(1))


def overlap(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    intersection = int((left & right).sum().item())
    union = int((left | right).sum().item())
    left_k = int(left.sum().item())
    right_k = int(right.sum().item())
    return {
        "left_k": left_k,
        "right_k": right_k,
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 1.0,
        "overlap_over_min_k": intersection / min(left_k, right_k),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.plans.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise RuntimeError("Plan set is incomplete")
    base = str(result["base_fingerprint"])
    masks: dict[tuple[int, float, str], torch.Tensor] = {}
    records: dict[tuple[int, float, str], dict[str, Any]] = {}
    for record in result["records"]:
        key = (
            int(record["seed"]),
            float(record["budget_fraction"]),
            str(record["arm"]),
        )
        if key in records:
            raise RuntimeError(f"Duplicate plan cell: {key}")
        plan = load_plan(
            record["plan_directory"], expected_base_fingerprint=base
        )
        masks[key] = plan.entries[0].support_mask
        records[key] = record

    by_arm_budget: dict[tuple[str, float], list[tuple[int, torch.Tensor]]] = defaultdict(
        list
    )
    for (seed, fraction, arm), mask in masks.items():
        by_arm_budget[(arm, fraction)].append((seed, mask))
    seed_stability: list[dict[str, Any]] = []
    for (arm, fraction), values in sorted(by_arm_budget.items()):
        comparisons = [
            {
                "left_seed": left_seed,
                "right_seed": right_seed,
                **overlap(left, right),
            }
            for (left_seed, left), (right_seed, right) in itertools.combinations(
                sorted(values), 2
            )
        ]
        if comparisons:
            seed_stability.append(
                {
                    "arm": arm,
                    "budget_fraction": fraction,
                    "pair_count": len(comparisons),
                    "mean_jaccard": sum(item["jaccard"] for item in comparisons)
                    / len(comparisons),
                    "minimum_jaccard": min(item["jaccard"] for item in comparisons),
                    "comparisons": comparisons,
                }
            )

    cross_arm: list[dict[str, Any]] = []
    comparison_arms = (
        "coordinate_shuffled_peer_pcsm",
        "layer_shuffled_peer_pcsm",
        "gradient_topk_sparse_native",
        "target_only_sparse_native",
    )
    for (seed, fraction, arm), mask in sorted(masks.items()):
        if arm != "real_peer_pcsm_calibrated":
            continue
        for comparison_arm in comparison_arms:
            comparison = masks.get((seed, fraction, comparison_arm))
            if comparison is not None:
                cross_arm.append(
                    {
                        "seed": seed,
                        "budget_fraction": fraction,
                        "left_arm": arm,
                        "right_arm": comparison_arm,
                        **overlap(mask, comparison),
                    }
                )

    attention: list[dict[str, Any]] = []
    for key, record in sorted(records.items()):
        peer_names = record.get("attention", {}).get("peer_names", [])
        alphas = record.get("attention", {}).get("alphas", [])
        if not peer_names:
            continue
        target_depth = 14
        contributions = defaultdict(float)
        for name, alpha in zip(peer_names, alphas, strict=True):
            distance = abs(layer_index(name) - target_depth)
            bucket = "nearest" if distance == 1 else "mid" if distance <= 7 else "distant"
            contributions[bucket] += float(alpha)
        attention.append(
            {
                "seed": key[0],
                "budget_fraction": key[1],
                "arm": key[2],
                "depth_contribution": dict(contributions),
                "maximum_alpha": max(float(value) for value in alphas),
                "minimum_alpha": min(float(value) for value in alphas),
            }
        )

    artifact: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": utc_now(),
        "plan_result_sha256": sha256_file(args.plans),
        "seed_support_stability": seed_stability,
        "cross_arm_support_overlap": cross_arm,
        "attention_depth_contribution": attention,
        "calibration_subset_stability": {
            "status": "NOT_RUN",
            "reason": (
                "Only the pre-specified MASK_CAL sample was used; alternate-subset "
                "stability remains an explicit limitation and cannot affect the gate."
            ),
        },
    }
    artifact["summary_sha256"] = canonical_json_hash(artifact)
    atomic_write_json(args.output, artifact)
    print(json.dumps({"output": str(args.output), "cells": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

