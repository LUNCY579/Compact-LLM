#!/usr/bin/env python3
"""Run the Phase-0 engineering proof and emit a machine-readable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tgsa import (
    ParallelCausalSelfAttention,
    TGSAConfig,
    TrajectoryGuidedSequentialAttention,
    count_parameters,
    telemetry_summary,
)


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def forward_tensor(model: nn.Module, inputs: Tensor) -> Tensor:
    if isinstance(model, TrajectoryGuidedSequentialAttention):
        return model(inputs, telemetry_level="none")
    return model(inputs)


def timing_summary(
    model: nn.Module,
    inputs: Tensor,
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            forward_tensor(model, inputs)
        synchronize(device)
        samples: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            forward_tensor(model, inputs)
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, max(0, int(0.9 * len(ordered)) - 1))
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": ordered[p90_index],
        "iterations": float(repeats),
    }


def discover_git_commit(root: Path) -> str | None:
    from_environment = os.environ.get("GITHUB_SHA")
    if from_environment:
        return from_environment
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def max_causal_prefix_error(model: nn.Module, inputs: Tensor) -> float:
    model.eval()
    split = max(1, inputs.shape[1] // 2)
    altered = inputs.detach().clone()
    generator = torch.Generator(device=inputs.device).manual_seed(991)
    replacement = torch.randn(
        altered[:, split:, :].shape,
        generator=generator,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    altered[:, split:, :] = replacement * 10.0
    with torch.no_grad():
        first = forward_tensor(model, inputs)
        second = forward_tensor(model, altered)
    return float((first[:, :split] - second[:, :split]).abs().max().float().item())


def run(config_path: Path, output_path: Path) -> dict[str, Any]:
    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(loaded["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    runtime = loaded["runtime"]
    device = resolve_device(str(runtime["device"]))
    dtype = resolve_dtype(str(runtime["dtype"]))
    model_config = TGSAConfig(**loaded["model"])

    tgsa = TrajectoryGuidedSequentialAttention(model_config).to(device=device, dtype=dtype)
    baseline = ParallelCausalSelfAttention(
        model_config.embed_dim,
        model_config.num_heads,
        dropout=model_config.dropout,
        bias=model_config.bias,
    ).to(device=device, dtype=dtype)
    batch = int(runtime["batch_size"])
    sequence = int(runtime["sequence_length"])
    inputs = torch.randn(
        batch,
        sequence,
        model_config.embed_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    tgsa.train()
    output, telemetry = tgsa(inputs, telemetry_level="summary")
    loss = output.float().square().mean()
    loss.backward()
    trajectory_gradients = [
        parameter.grad
        for name, parameter in tgsa.named_parameters()
        if "trajectory_encoders" in name and parameter.requires_grad
    ]
    finite_backward = inputs.grad is not None and bool(torch.isfinite(inputs.grad).all())
    trajectory_gradient = any(
        gradient is not None
        and bool(torch.isfinite(gradient).all())
        and float(gradient.abs().sum().item()) > 0.0
        for gradient in trajectory_gradients
    )

    detached_inputs = inputs.detach()
    causal_error = max_causal_prefix_error(tgsa, detached_inputs)
    warmup = int(runtime["warmup_iterations"])
    repeats = int(runtime["timed_iterations"])
    tgsa_timing = timing_summary(tgsa, detached_inputs, warmup=warmup, repeats=repeats, device=device)
    baseline_timing = timing_summary(
        baseline,
        detached_inputs,
        warmup=warmup,
        repeats=repeats,
        device=device,
    )

    acceptance = loaded["acceptance"]
    checks = {
        "finite_forward": bool(torch.isfinite(output).all()),
        "finite_backward": finite_backward,
        "trajectory_gradient_present": trajectory_gradient,
        "causal_prefix_error_within_limit": causal_error
        <= float(acceptance["max_causal_prefix_error"]),
    }
    status = "pass" if all(checks.values()) else "fail"
    root = Path(__file__).resolve().parents[1]
    hardware: dict[str, Any] = {
        "device": str(device),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    if device.type == "cuda":
        hardware.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_version": torch.version.cuda,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": "tgsa.run-manifest.v1",
        "run_id": f"phase0-smoke-seed-{seed}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "branch": "research/trajectory-guided-sequential-attention",
        "git_commit": discover_git_commit(root),
        "config_path": str(config_path),
        "config_sha256": canonical_hash(loaded),
        "seed": seed,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "hardware": hardware,
        },
        "model": {
            "tgsa_trainable_parameters": count_parameters(tgsa),
            "parallel_baseline_trainable_parameters": count_parameters(baseline),
            "configuration": loaded["model"],
        },
        "input": {
            "batch_size": batch,
            "sequence_length": sequence,
            "dtype": str(dtype),
            "tensor_sha256": hashlib.sha256(
                detached_inputs.float().cpu().numpy().tobytes()
            ).hexdigest(),
        },
        "checks": checks,
        "metrics": {
            "loss": float(loss.detach().cpu().item()),
            "max_causal_prefix_error": causal_error,
            "tgsa_forward_timing": tgsa_timing,
            "parallel_baseline_forward_timing": baseline_timing,
            "synthetic_latency_ratio": tgsa_timing["median_ms"]
            / max(baseline_timing["median_ms"], 1.0e-12),
            "trajectory": telemetry_summary(telemetry),
        },
        "claim_scope": "engineering-smoke-only; no quality or novelty claim",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "configs" / "phase0_smoke.json")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "phase0_smoke.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.config, arguments.output)
    print(json.dumps({"status": result["status"], "output": str(arguments.output)}, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)
