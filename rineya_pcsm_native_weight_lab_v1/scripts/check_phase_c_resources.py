#!/usr/bin/env python3
"""Record a quality-blind hardware feasibility decision for real Phase C."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import psutil
import torch

from rineya_pcsm.hashing import canonical_json_hash
from rineya_pcsm.manifest import atomic_write_json, environment_manifest, utc_now


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-minimum-free-gib", type=float, default=6.0)
    args = parser.parse_args()
    memory = psutil.virtual_memory()
    cuda: dict[str, object] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        cuda.update(
            {
                "device": torch.cuda.get_device_name(0),
                "free_bytes": free,
                "total_bytes": total,
                "free_gib": free / 1024**3,
                "total_gib": total / 1024**3,
            }
        )
    cuda_eligible = bool(
        cuda.get("available")
        and float(cuda.get("free_gib", 0.0)) >= args.cuda_minimum_free_gib
    )
    numel = 1024 * 3072
    artifact: dict[str, object] = {
        "schema_version": "1.0",
        "captured_at_utc": utc_now(),
        "policy": (
            "device selected from available memory before model execution; no quality "
            "outcome used"
        ),
        "platform": platform.platform(),
        "system_memory": {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "total_gib": memory.total / 1024**3,
            "available_gib": memory.available / 1024**3,
        },
        "cuda": cuda,
        "cuda_minimum_free_gib": args.cuda_minimum_free_gib,
        "selected_device": "cuda" if cuda_eligible else "cpu",
        "target_matrix": {
            "shape": [1024, 3072],
            "numel": numel,
            "bf16_weight_bytes": numel * 2,
            "fp32_score_bytes": numel * 4,
            "fp32_stabilizer_bytes": numel * 4,
            "bool_mask_bytes": numel,
            "note": "Dense backward GEMMs remain dense; no sparse-compute claim.",
        },
        "environment": environment_manifest(),
    }
    artifact["preflight_sha256"] = canonical_json_hash(artifact)
    atomic_write_json(args.output, artifact)
    print(json.dumps(artifact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
