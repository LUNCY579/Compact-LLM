#!/usr/bin/env python3
"""Fail-closed wrapper for the real-checkpoint TGSA + MTP adapter warm start."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rineya_tgsa.run_integrity import (
    attach_training_integrity,
    capture_device_identity,
    source_tree_manifest,
)


def _capture_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--stage", default="adapters")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    namespace, _ = parser.parse_known_args(arguments)
    return namespace


def main() -> int:
    raw = sys.argv[1:]
    if "--development-only" not in raw:
        raise SystemExit(
            "real-checkpoint warm starts require the explicit --development-only marker"
        )
    native_arguments = [argument for argument in raw if argument != "--development-only"]
    captured = _capture_arguments(native_arguments)
    if captured.output_dir.exists():
        raise SystemExit(f"output directory already exists: {captured.output_dir}")
    if not captured.train_jsonl.is_file():
        raise SystemExit(f"training JSONL not found: {captured.train_jsonl}")

    # Bind source weights and device before the native trainer can create output.
    source = source_tree_manifest(captured.model_dir)
    device = capture_device_identity(captured.device)
    invocation = {
        "development_only": True,
        "model_dir": str(captured.model_dir.resolve()),
        "train_jsonl": str(captured.train_jsonl.resolve()),
        "output_dir": str(captured.output_dir.resolve()),
        "layers": captured.layers,
        "stage": captured.stage,
        "steps": captured.steps,
        "batch_size": captured.batch_size,
        "gradient_accumulation": captured.gradient_accumulation,
        "learning_rate": captured.learning_rate,
        "seed": captured.seed,
        "device": captured.device,
    }

    native = Path(__file__).with_name("train_tgsa_mtp_adapter.py")
    completed = subprocess.run(
        [sys.executable, str(native), *native_arguments],
        check=False,
    )
    if completed.returncode != 0:
        return int(completed.returncode)

    manifest = attach_training_integrity(
        output_dir=captured.output_dir,
        source_manifest=source,
        device_identity=device,
        invocation=invocation,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "development_only": manifest["development_only"],
                "effective_epochs": manifest["training_dose"]["effective_epochs"],
                "run_fingerprint": manifest["run_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
