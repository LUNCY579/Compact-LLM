# Purpose: Records parameter count, checkpoint bytes, architecture metadata, and generation smoke results.
from __future__ import annotations

"""Measure real checkpoint size for a Phase 4B candidate.

The output separates parameter count, in-memory tensor bytes, buffer bytes, and
local serialized directory bytes. These values are evidence about model size;
they are not substitutes for measured latency, VRAM, or RAM in the target
runtime.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import write_json
from scripts.common.model_introspection import count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure parameters and serialized size of a saved model."
    )
    parser.add_argument("--model", required=True, help="Local path or model ID.")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def directory_bytes(path: Path) -> int | None:
    """Return recursive local bytes, or None for a remote model identifier."""

    if not path.exists() or not path.is_dir():
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    # Keep Transformers optional during import so unit tests for helper logic do
    # not require loading the model stack.
    from transformers import AutoModelForCausalLM

    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", low_cpu_mem_usage=True
    )

    parameters = count_parameters(model)
    tensor_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    buffers_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )

    dtype_counts: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + parameter.numel()

    payload = {
        "model": args.model,
        "parameters": parameters,
        "parameter_tensor_bytes_loaded": int(tensor_bytes),
        "buffer_tensor_bytes_loaded": int(buffers_bytes),
        "local_directory_bytes": directory_bytes(Path(args.model)),
        "dtype_parameter_counts": dtype_counts,
        "reporting_boundary": (
            "Size metrics only. Measure latency and peak VRAM/RAM separately in "
            "the production runtime."
        ),
    }
    write_json(args.out, payload)
    print(args.out)


if __name__ == "__main__":
    main()
