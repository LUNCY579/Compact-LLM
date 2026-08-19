#!/usr/bin/env python3
"""Stitch a 19-layer Rineya donor into 28-layer Qwen3-0.6B on Phase 2B kept layers."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def layer_suffixes(state: dict[str, torch.Tensor], index: int) -> dict[str, str]:
    prefix = f"model.layers.{index}."
    mapping = {key[len(prefix) :]: key for key in state if key.startswith(prefix)}
    if not mapping:
        raise KeyError(f"No tensors found for {prefix}*")
    return mapping


def copy_qwen3_sidecar(qwen3: Path, output_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(qwen3), trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    for name in (
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        src = qwen3 / name
        if src.exists() and not (output_dir / name).exists():
            shutil.copy2(src, output_dir / name)


def stitch(
    qwen3: Path,
    donor: Path,
    output_dir: Path,
    layer_map: dict[str, Any],
    alpha: float,
    overwrite: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SystemExit(
            f"Output directory is not empty: {output_dir}. Use a new path or --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    kept = [int(value) for value in layer_map["kept_original_layer_ids"]]
    dropped = [int(value) for value in layer_map["dropped_original_layer_ids"]]
    mapping = {int(src): int(dst) for src, dst in layer_map["rineya_to_qwen3"].items()}
    if sorted(mapping.values()) != sorted(kept):
        raise ValueError("rineya_to_qwen3 values must match kept_original_layer_ids")
    if sorted(kept + dropped) != list(range(int(layer_map["qwen3_layers"]))):
        raise ValueError("kept + dropped must cover all Qwen3 layers")
    if len(mapping) != int(layer_map["rineya_layers"]):
        raise ValueError("rineya_to_qwen3 must have one entry per Rineya layer")

    print(f"[{utc_now()}] Loading Qwen3 from {qwen3}", flush=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        str(qwen3),
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"[{utc_now()}] Loading donor from {donor}", flush=True)
    donor_model = AutoModelForCausalLM.from_pretrained(
        str(donor),
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    if int(qwen_model.config.num_hidden_layers) != int(layer_map["qwen3_layers"]):
        raise SystemExit(
            f"Qwen3 layer count {qwen_model.config.num_hidden_layers} "
            f"!= {layer_map['qwen3_layers']}"
        )
    if int(donor_model.config.num_hidden_layers) != int(layer_map["rineya_layers"]):
        raise SystemExit(
            f"Donor layer count {donor_model.config.num_hidden_layers} "
            f"!= {layer_map['rineya_layers']}"
        )

    qwen_state = qwen_model.state_dict()
    donor_state = donor_model.state_dict()
    blended_keys: list[str] = []
    unchanged_dropped: list[str] = []

    with torch.no_grad():
        for rineya_i, qwen_i in sorted(mapping.items()):
            src_map = layer_suffixes(donor_state, rineya_i)
            dst_map = layer_suffixes(qwen_state, qwen_i)
            if set(src_map) != set(dst_map):
                raise SystemExit(
                    f"Layer tensor mismatch rineya {rineya_i} vs qwen3 {qwen_i}: "
                    f"missing={sorted(set(dst_map) - set(src_map))} "
                    f"extra={sorted(set(src_map) - set(dst_map))}"
                )
            for suffix, dst_key in dst_map.items():
                src = donor_state[src_map[suffix]]
                dst = qwen_state[dst_key]
                if src.shape != dst.shape:
                    raise SystemExit(
                        f"Shape mismatch {dst_key}: donor {tuple(src.shape)} vs qwen3 {tuple(dst.shape)}"
                    )
                if not dst.is_floating_point():
                    continue
                dst.copy_(((1.0 - alpha) * dst.float() + alpha * src.float()).to(dst.dtype))
                blended_keys.append(dst_key)

        for qwen_i in dropped:
            unchanged_dropped.extend(sorted(layer_suffixes(qwen_state, qwen_i).values()))

    qwen_model.load_state_dict(qwen_state)
    del donor_model
    del donor_state

    print(f"[{utc_now()}] Saving hybrid to {output_dir}", flush=True)
    qwen_model.save_pretrained(output_dir, safe_serialization=True)
    copy_qwen3_sidecar(qwen3, output_dir)

    manifest = {
        "format_version": 1,
        "merge_space": "FP32 entire decoder-block average on Phase 2B kept layers",
        "generated_at": utc_now(),
        "qwen3_model": str(qwen3),
        "donor_model": str(donor),
        "alpha_rineya": alpha,
        "alpha_qwen3": 1.0 - alpha,
        "kept_original_layer_ids": kept,
        "dropped_original_layer_ids": dropped,
        "rineya_to_qwen3": {str(src): dst for src, dst in mapping.items()},
        "blended_tensor_count": len(blended_keys),
        "unchanged": [
            "dropped decoder blocks",
            "embed_tokens",
            "final norm",
            "tied lm_head",
        ],
        "unchanged_dropped_tensor_count": len(unchanged_dropped),
    }
    write_json(output_dir / "soup_manifest.json", manifest)
    print(f"[{utc_now()}] Saved hybrid: {output_dir}", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="50/50-blend Rineya kept decoder layers into 28-layer Qwen3-0.6B."
    )
    parser.add_argument("--qwen3", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-map", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="Rineya donor weight on kept layers")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer_map = load_json(args.layer_map)
    default_alpha = float(layer_map.get("alpha", 0.5))
    alpha = default_alpha if args.alpha is None else float(args.alpha)
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")
    stitch(
        qwen3=args.qwen3.expanduser().resolve(),
        donor=args.donor.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        layer_map=layer_map,
        alpha=alpha,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
