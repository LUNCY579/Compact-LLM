# Purpose: Fail early when soup adapters differ in base model, tensor shapes, target modules, or PEFT configuration.
from __future__ import annotations

"""Validate adapter compatibility before reconstructing any model soup.

Configuration equality alone is not enough: two adapter directories can carry
matching JSON while their saved tensor keys or shapes differ.  This script
checks both the PEFT configuration contract and the actual adapter state schema,
then writes a durable report used by the Phase 4 runner.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, sha256_tree, write_json

CONFIG_KEYS = (
    "base_model_name_or_path",
    "r",
    "lora_alpha",
    "target_modules",
    "peft_type",
    "task_type",
    "bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 4 PEFT ingredients.")
    parser.add_argument("adapters", nargs="+", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/PHASE_4_COMPATIBILITY.json",
    )
    return parser.parse_args()


def normalized_config(path: Path) -> dict[str, Any]:
    config_path = path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    config = read_json(config_path)
    normalized: dict[str, Any] = {}
    for key in CONFIG_KEYS:
        value = config.get(key)
        # PEFT may serialize target modules as a list in one run and a set-like
        # order in another. Sorting preserves semantic equality.
        if key == "target_modules" and isinstance(value, list):
            value = sorted(str(item) for item in value)
        normalized[key] = value
    return normalized


def tensor_schema(path: Path) -> dict[str, list[int]]:
    """Read only adapter tensor names/shapes, preferring safetensors metadata."""

    safe_path = path / "adapter_model.safetensors"
    if safe_path.exists():
        from safetensors import safe_open

        schema: dict[str, list[int]] = {}
        with safe_open(safe_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                schema[key] = list(handle.get_slice(key).get_shape())
        return schema

    bin_path = path / "adapter_model.bin"
    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected adapter state format in {bin_path}")
        return {str(key): list(value.shape) for key, value in state.items() if hasattr(value, "shape")}

    raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin in {path}")


def main() -> None:
    args = parse_args()
    if len(args.adapters) < 1:
        raise SystemExit("At least one adapter is required.")

    rows = []
    for adapter in args.adapters:
        rows.append(
            {
                "path": str(adapter),
                "fingerprint": sha256_tree(adapter),
                "config": normalized_config(adapter),
                "tensor_schema": tensor_schema(adapter),
            }
        )

    reference = rows[0]
    mismatches: list[dict[str, Any]] = []
    for index, row in enumerate(rows[1:], 1):
        for key in CONFIG_KEYS:
            if row["config"].get(key) != reference["config"].get(key):
                mismatches.append(
                    {
                        "ingredient_index": index,
                        "field": f"config.{key}",
                        "expected": reference["config"].get(key),
                        "actual": row["config"].get(key),
                    }
                )
        if row["tensor_schema"] != reference["tensor_schema"]:
            expected_keys = set(reference["tensor_schema"])
            actual_keys = set(row["tensor_schema"])
            mismatches.append(
                {
                    "ingredient_index": index,
                    "field": "tensor_schema",
                    "missing_keys": sorted(expected_keys - actual_keys),
                    "extra_keys": sorted(actual_keys - expected_keys),
                    "shape_mismatches": {
                        key: {
                            "expected": reference["tensor_schema"].get(key),
                            "actual": row["tensor_schema"].get(key),
                        }
                        for key in sorted(expected_keys & actual_keys)
                        if reference["tensor_schema"].get(key) != row["tensor_schema"].get(key)
                    },
                }
            )

    payload = {
        "ingredients": rows,
        "status": "PASS" if not mismatches else "FAIL",
        "mismatches": mismatches,
    }
    write_json(args.out, payload)
    if mismatches:
        raise SystemExit(f"Incompatible adapters; inspect {args.out}")
    print(f"Compatibility PASS: {args.out}")


if __name__ == "__main__":
    main()
