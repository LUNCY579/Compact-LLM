from __future__ import annotations

"""Measure model parameter count, loaded tensor bytes, and serialized bytes."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import write_json
from scripts.common.model_introspection import count_parameters, find_decoder_stack


def directory_bytes(path: Path) -> int | None:
    if not path.exists() or not path.is_dir():
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def measure_loaded_model(model: torch.nn.Module) -> dict[str, object]:
    parameters = count_parameters(model)
    tensor_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    dtype_counts: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + parameter.numel()
    return {
        "parameters": parameters,
        "parameter_tensor_bytes_loaded": int(tensor_bytes),
        "buffer_tensor_bytes_loaded": int(buffer_bytes),
        "dtype_parameter_counts": dtype_counts,
        "num_hidden_layers": len(find_decoder_stack(model).layers),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure a saved model footprint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForCausalLM

    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    payload = {
        "model": args.model,
        **measure_loaded_model(model),
        "local_directory_bytes": directory_bytes(Path(args.model)),
        "reporting_boundary": (
            "Weight/tensor size evidence only. Runtime KV cache, temporary buffers, "
            "latency, and peak process memory must be measured separately."
        ),
    }
    write_json(args.out, payload)
    print(args.out)


if __name__ == "__main__":
    main()
