#!/usr/bin/env python3
"""Freeze quality-blind resource stages before any real-model execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rineya_pcsm.hashing import canonical_json_hash, sha256_file
from rineya_pcsm.manifest import atomic_write_json, utc_now

ARMS = [
    "no_update_frozen",
    "random_sparse_native",
    "gradient_topk_sparse_native",
    "target_only_sparse_native",
    "coordinate_shuffled_peer_pcsm",
    "layer_shuffled_peer_pcsm",
    "real_peer_pcsm_bootstrap",
    "real_peer_pcsm_calibrated",
]


def shortest(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or count > len(rows):
        raise ValueError("Invalid resource sample count")
    return sorted(rows, key=lambda row: (int(row["token_count"]), row["row_id"]))[
        :count
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    sections = samples["sections"]
    target = "model.layers.14.mlp.down_proj.weight"
    calibration = {
        "meta_train": shortest(sections["meta_train"], 1),
        "mask_cal": shortest(sections["mask_cal"], 1),
    }
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": utc_now(),
        "selection_policy": (
            "quality-blind shortest-token resource rule frozen before model execution"
        ),
        "base_model": {
            "path": str(args.model.resolve()),
            "weight_sha256": sha256_file(args.model / "model.safetensors"),
            "config_sha256": sha256_file(args.model / "config.json"),
        },
        "sample_manifest": {
            "path": str(args.samples.resolve()),
            "sha256": sha256_file(args.samples),
        },
        "target_role": "mlp.down_proj",
        "target_parameter": target,
        "peer_policy": "all other same-role layers; target self-weight exactly zero",
        "generator": {
            "descriptor_dimension": 24,
            "calibration_steps": 20,
            "calibration_learning_rate": 0.01,
            "bootstrap_steps": 0,
            "stabilizer_bounds": [0.25, 1.0],
            "topk_ste_temperature": 1.0,
            "structural_regularizer_primary": 0.0,
            "structural_regularizer_ablation": 0.05,
            "structural_proxy_learning_rate": 0.0001,
        },
        "native_optimizer": {
            "kind": "ExactSupportAdamW",
            "learning_rate_grid": [0.00005, 0.0001, 0.0002],
            "selection_split": "META_TRAIN_only",
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "weight_decay": 0.01,
        },
        "calibration": calibration,
        "pilot": {
            "purpose": "resource and correctness only; cannot pass advancement gate",
            "seeds": [17],
            "budget_fractions": [0.001],
            "arms": [
                "no_update_frozen",
                "random_sparse_native",
                "gradient_topk_sparse_native",
                "coordinate_shuffled_peer_pcsm",
                "real_peer_pcsm_calibrated",
            ],
            "one_step_dev": shortest(sections["one_step_dev"], 1),
        },
        "qualification": {
            "run_only_if_pilot_is_valid": True,
            "seeds": [11, 17, 23, 29, 31],
            "budget_fractions": [0.0001, 0.0005, 0.001, 0.005, 0.01],
            "arms": ARMS,
            "one_step_dev": shortest(sections["one_step_dev"], 4),
            "context_scope": (
                "three ~2.8k examples plus one 6.1k long-context stress example"
            ),
            "fairness": (
                "same base, target, examples, selected-coordinate count, optimizer, "
                "learning-rate grid, and per-arm evaluation budget"
            ),
            "required_causal_ablations": [
                "coordinate_shuffle",
                "layer_shuffle",
                "module_role_shuffle",
                "uniform_peer_attention",
                "nearest_layer_only",
                "target_only",
                "gradient_feature_only",
                "no_step_scale_head",
                "random_generator",
                "structural_regularizer_off_on",
            ],
        },
        "advancement_gate": {
            "pilot_can_advance": False,
            "requires_complete_qualification": True,
            "real_over_shuffled_seed_wins": "at_least_4_of_5",
            "paired_bootstrap_ci_must_exclude_zero": True,
            "layer_permutation_test_required": True,
        },
        "sealed_test_read": False,
    }
    manifest["budget_manifest_sha256"] = canonical_json_hash(manifest)
    atomic_write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
