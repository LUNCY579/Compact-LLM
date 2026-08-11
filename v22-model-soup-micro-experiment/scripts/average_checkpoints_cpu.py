from __future__ import annotations

"""Uniform-average compatible full HF checkpoints on CPU (streaming, no CUDA)."""

import argparse
import gc
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


META_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
)


def find_weight_file(model_dir: Path) -> Path:
    sf = model_dir / "model.safetensors"
    if sf.exists():
        return sf
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if len(shards) == 1:
        return shards[0]
    if shards:
        raise RuntimeError(f"Sharded checkpoints are not supported yet: {model_dir}")
    bins = list(model_dir.glob("pytorch_model*.bin"))
    if len(bins) == 1:
        return bins[0]
    raise FileNotFoundError(f"No model weights under {model_dir}")


def load_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {path}")
    return obj


def average_state_dicts(paths: list[Path]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not paths:
        raise ValueError("Need at least one checkpoint")
    n = float(len(paths))
    first = load_state(find_weight_file(paths[0]))
    dtypes = {k: v.dtype for k, v in first.items()}
    acc = {k: v.detach().to(torch.float32) / n for k, v in first.items()}
    del first
    gc.collect()
    for path in paths[1:]:
        state = load_state(find_weight_file(path))
        if set(state) != set(acc):
            missing = sorted(set(acc) - set(state))
            extra = sorted(set(state) - set(acc))
            raise RuntimeError(f"Incompatible keys for {path}: missing={missing[:5]} extra={extra[:5]}")
        for key, value in state.items():
            if tuple(value.shape) != tuple(acc[key].shape):
                raise RuntimeError(f"Shape mismatch on {key} in {path}")
            acc[key].add_(value.detach().to(torch.float32), alpha=1.0 / n)
        del state
        gc.collect()
    out = {k: v.to(dtypes[k]) for k, v in acc.items()}
    meta = {
        "ingredient_count": len(paths),
        "ingredients": [str(p) for p in paths],
        "formula": "W = (1/N) * sum_i W_i",
        "device": "cpu",
        "tensor_count": len(out),
    }
    return out, meta


def copy_tokenizer_and_config(src: Path, dst: Path) -> None:
    for name in META_FILES:
        path = src / name
        if path.exists():
            shutil.copy2(path, dst / name)


def save_checkpoint(state: dict[str, torch.Tensor], template: Path, output: Path, manifest: dict[str, Any]) -> None:
    if output.exists():
        shutil.rmtree(output)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Staging writes a full second copy; require ~2× weight-file headroom.
    weight_bytes = sum(v.numel() * v.element_size() for v in state.values())
    need = max(int(weight_bytes * 1.15), 2 * 1024**3)
    free = shutil.disk_usage(parent).free
    if free < need:
        raise RuntimeError(
            f"Not enough disk to save soup under {parent}: free={free / 1024**3:.2f} GiB, "
            f"need~{need / 1024**3:.2f} GiB for staging+final model.safetensors"
        )
    staging = parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        copy_tokenizer_and_config(template, staging)
        save_file(state, str(staging / "model.safetensors"))
        (staging / "soup_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.replace(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU uniform average of full checkpoints")
    parser.add_argument("--model", action="append", type=Path, required=True, dest="models")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.force:
        raise SystemExit(f"Output exists: {args.output_dir} (pass --force)")
    state, meta = average_state_dicts(args.models)
    save_checkpoint(state, args.models[0], args.output_dir, meta)
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
