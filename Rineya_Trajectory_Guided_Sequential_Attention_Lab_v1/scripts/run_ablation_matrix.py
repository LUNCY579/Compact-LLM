#!/usr/bin/env python3
"""Execute every reference ablation as an engineering-only sanity matrix."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from tgsa import (
    ParallelCausalSelfAttention,
    TGSAConfig,
    TrajectoryGuidedSequentialAttention,
    count_parameters,
    telemetry_summary,
)

ARMS = (
    "parallel_mha",
    "cascade",
    "history_control",
    "raw_delta",
    "learned_delta",
    "qkv_conditioned",
    "predictor_corrector",
    "full",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "configs" / "phase0_smoke.json")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "phase0_ablation_matrix.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loaded = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(loaded["seed"])
    model_spec = loaded["model"]
    runtime = loaded["runtime"]
    torch.manual_seed(seed)
    inputs = torch.randn(
        int(runtime["batch_size"]),
        int(runtime["sequence_length"]),
        int(model_spec["embed_dim"]),
    )
    results: list[dict[str, object]] = []

    for arm in ARMS:
        torch.manual_seed(seed)
        start = time.perf_counter()
        if arm == "parallel_mha":
            model = ParallelCausalSelfAttention(
                int(model_spec["embed_dim"]),
                int(model_spec["num_heads"]),
                dropout=0.0,
                bias=bool(model_spec["bias"]),
            )
            output = model(inputs)
            telemetry = None
        else:
            config = TGSAConfig.for_ablation(
                embed_dim=int(model_spec["embed_dim"]),
                num_heads=int(model_spec["num_heads"]),
                arm=arm,  # type: ignore[arg-type]
                trajectory_rank=int(model_spec["trajectory_rank"]),
                dropout=0.0,
                bias=bool(model_spec["bias"]),
                telemetry_level="summary",
            )
            model = TrajectoryGuidedSequentialAttention(config)
            output, telemetry = model(inputs, telemetry_level="summary")
        output.square().mean().backward()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        results.append(
            {
                "arm": arm,
                "trainable_parameters": count_parameters(model),
                "finite": bool(torch.isfinite(output).all()),
                "output_rms": float(output.float().square().mean().sqrt().item()),
                "single_smoke_elapsed_ms": elapsed_ms,
                "trajectory": None if telemetry is None else telemetry_summary(telemetry),
            }
        )

    manifest = {
        "schema_version": "tgsa.ablation-smoke.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "results": results,
        "all_passed": all(bool(item["finite"]) for item in results),
        "claim_scope": "engineering-smoke-only; timings are not benchmark-quality",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"all_passed": manifest["all_passed"], "output": str(args.output)}, indent=2))
    return 0 if manifest["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
