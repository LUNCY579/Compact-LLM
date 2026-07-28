# Purpose: Reconstructs PEFT deltas in FP32, averages them, and saves a normal full Transformers model.
from __future__ import annotations

"""Merge compatible PEFT adapters into one full Hugging Face checkpoint.

The script reconstructs each adapter's full weight delta against the same base
model, accumulates those deltas in FP32, applies the weighted average to a fresh
base checkpoint, and saves a normal (adapter-free) model directory.  The output
can therefore be consumed directly by Phase 4B and later Transformers runtimes.
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import environment, sha256_tree, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge compatible adapters in reconstructed FP32 delta space.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapters", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory. Normal phase runners avoid this.",
    )
    return parser.parse_args()


def validate_weights(adapters: list[Path], weights: list[float] | None) -> list[float]:
    """Return normalized merge coefficients after strict validation."""

    if not adapters:
        raise ValueError("At least one adapter is required.")
    for adapter in adapters:
        if not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing adapter_config.json in {adapter}")
    values = weights or [1.0 / len(adapters)] * len(adapters)
    if len(values) != len(adapters):
        raise ValueError("--weights must contain one value per adapter.")
    if any(value < 0 for value in values):
        raise ValueError("Soup weights must be non-negative.")
    total = sum(values)
    if total <= 0:
        raise ValueError("Soup weights must sum to a positive value.")
    normalized = [value / total for value in values]
    if abs(sum(normalized) - 1.0) > 1e-6:
        raise ValueError("Normalized weights do not sum to one.")
    return normalized


def merged_state_dict(model_id: str, adapter: Path) -> dict[str, torch.Tensor]:
    """Load one adapter on a fresh FP32 CPU base and return merged tensors."""

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    # state_dict tensors remain owned by ``merged`` until the caller finishes
    # accumulating them.  The caller deletes the dictionary immediately after.
    return merged.state_dict()


def main() -> None:
    args = parse_args()
    weights = validate_weights(args.adapters, args.weights)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory is not empty: {args.output_dir}. Use a new path or --overwrite."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # A fresh base object supplies the exact architecture/configuration and the
    # immutable reference values used to reconstruct weighted adapter deltas.
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    base_state = base.state_dict()
    accumulators: dict[str, torch.Tensor] = {
        name: torch.zeros_like(value, dtype=torch.float32)
        for name, value in base_state.items()
        if value.is_floating_point()
    }

    for coefficient, adapter in zip(weights, args.adapters):
        print(f"Accumulating {adapter} with weight {coefficient:.6f}", flush=True)
        merged_state = merged_state_dict(args.model_id, adapter)
        missing = sorted(set(accumulators) - set(merged_state))
        if missing:
            raise KeyError(f"Merged adapter is missing {len(missing)} base tensors; first: {missing[0]}")
        for name, accumulator in accumulators.items():
            # Accumulate reconstructed deltas rather than averaging LoRA A/B
            # factors, which is not generally equivalent.
            delta = merged_state[name].float() - base_state[name].float()
            accumulator.add_(delta, alpha=coefficient)
        del merged_state
        gc.collect()

    with torch.no_grad():
        for name, value in base_state.items():
            if name in accumulators:
                value.copy_((value.float() + accumulators[name]).to(value.dtype))

    base.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(args.output_dir)
    write_json(
        args.output_dir / "soup_manifest.json",
        {
            "format_version": 2,
            "model_id": args.model_id,
            "base_fingerprint": sha256_tree(Path(args.model_id)),
            "adapters": [str(path) for path in args.adapters],
            "adapter_fingerprints": [sha256_tree(path) for path in args.adapters],
            "weights": weights,
            "merge_space": "FP32 reconstructed adapter deltas",
            "environment": environment(),
        },
    )
    print(f"Saved merged full checkpoint: {args.output_dir}")


if __name__ == "__main__":
    main()
