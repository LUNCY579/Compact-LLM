#!/usr/bin/env python
"""Emit a repeatable FMA qualification environment/budget packet."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.factorized_memory_attention_v1.fma.budget import parameter_budget
from experiments.factorized_memory_attention_v1.fma.config import FMAConfig, load_shape


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        default=str(ROOT / "Rineya_Published_Cores" / "Rineya-Supa-1.7.0-devfreeze" / "config.json"),
    )
    parser.add_argument(
        "--core",
        default=str(ROOT / "Rineya_Published_Cores" / "Rineya-Supa-1.7.0-devfreeze" / "CORE.json"),
    )
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--head-sha", required=True, help="immutable code/docs/tests commit")
    parser.add_argument("--output")
    args = parser.parse_args()

    shape = load_shape(args.model_config)
    core = json.loads(Path(args.core).read_text(encoding="utf-8"))
    baseline_params = int(core["checks"]["parameter_count_serialized"])
    budget = parameter_budget(shape, FMAConfig(rank=args.rank), baseline_params=baseline_params)

    cuda = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cuda.update(
            device=torch.cuda.get_device_name(0),
            total_bytes=props.total_memory,
            allocated_bytes=torch.cuda.memory_allocated(0),
            reserved_bytes=torch.cuda.memory_reserved(0),
        )
    else:
        cuda.update(status="BLOCKED", reason="CPU-only runtime; exact RTX 4070 probe not executed")

    receipt = {
        "experiment_id": "FMA-QUAL-001",
        "base_sha": "7d9b4fd450bc56405f6cec41459a7e83df7958ae",
        "qualified_head_sha": args.head_sha,
        "model": "Rineya-Supa-1.7.0-devfreeze",
        "model_config": str(Path(args.model_config).relative_to(ROOT)),
        "checkpoint_weight_sha256": core["checks"]["weight_sha256"],
        "config_sha256": core["checks"]["preserved_asset_sha256"]["config.json"],
        "tokenizer_sha256": core["checks"]["preserved_asset_sha256"]["tokenizer.json"],
        "dtype": next(iter(core["checks"]["dtype_counts"])).lower(),
        "shape": shape.to_dict(),
        "modified_layers": list(range(shape.num_hidden_layers)),
        "rank": args.rank,
        "budget": budget.to_dict(),
        "seed": args.seed,
        "calibration_performed": False,
        "calibration_data_ids": [],
        "sealed_data_access": False,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "pytest": package_version("pytest"),
        },
        "resources": {"cuda": cuda},
        "evidence_classification": "mechanical/runtime packet only; no architecture efficacy result",
    }
    text_out = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text_out, encoding="utf-8")
    else:
        print(text_out, end="")


if __name__ == "__main__":
    main()
