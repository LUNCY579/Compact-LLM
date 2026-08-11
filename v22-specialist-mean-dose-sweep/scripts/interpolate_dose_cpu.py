from __future__ import annotations

"""Interpolate two full checkpoints on CPU: W = (1-d)*A + d*B."""

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


def weight_file(model_dir: Path) -> Path:
    sf = model_dir / "model.safetensors"
    if sf.exists():
        return sf
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if len(shards) == 1:
        return shards[0]
    raise FileNotFoundError(f"Expected single-file weights under {model_dir}")


def load_state(model_dir: Path) -> dict[str, torch.Tensor]:
    return load_file(str(weight_file(model_dir)), device="cpu")


def interpolate(anchor: Path, soup: Path, dose: float) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not 0.0 <= dose <= 1.0:
        raise ValueError(f"dose must be in [0,1], got {dose}")
    a = load_state(anchor)
    b = load_state(soup)
    if set(a) != set(b):
        raise RuntimeError("Checkpoint key mismatch between anchor and soup")
    out: dict[str, torch.Tensor] = {}
    for key in a:
        if tuple(a[key].shape) != tuple(b[key].shape):
            raise RuntimeError(f"Shape mismatch on {key}")
        dtype = a[key].dtype
        mixed = (1.0 - dose) * a[key].float() + dose * b[key].float()
        out[key] = mixed.to(dtype)
    del a, b
    gc.collect()
    meta = {
        "formula": "W(d)=(1-d)*W_anchor + d*W_phase4_mean",
        "equivalent_delta_form": "W_anchor + d * mean_i(Delta_i)",
        "dose": dose,
        "anchor": str(anchor),
        "soup": str(soup),
        "device": "cpu",
    }
    return out, meta


def save_checkpoint(state: dict[str, torch.Tensor], template: Path, output: Path, manifest: dict[str, Any]) -> None:
    if output.exists():
        shutil.rmtree(output)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        for name in META_FILES:
            src = template / name
            if src.exists():
                shutil.copy2(src, staging / name)
        save_file(state, str(staging / "model.safetensors"))
        (staging / "dose_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.replace(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", type=Path, required=True)
    p.add_argument("--soup", type=Path, required=True)
    p.add_argument("--dose", type=float, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.output_dir.exists() and not args.force:
        raise SystemExit(f"Output exists: {args.output_dir}")
    state, meta = interpolate(args.anchor, args.soup, args.dose)
    save_checkpoint(state, args.anchor, args.output_dir, meta)
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
