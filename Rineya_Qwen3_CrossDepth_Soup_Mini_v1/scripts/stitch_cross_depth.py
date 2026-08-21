#!/usr/bin/env python3
"""Stitch a 19-layer Rineya donor into 28-layer Qwen3-0.6B on Phase 2B kept layers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
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


def unlink_tmp_shards(directory: Path) -> None:
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.is_file() and path.name.startswith(".tmp"):
            try:
                path.unlink()
            except OSError:
                time.sleep(1.0)
                path.unlink()


def save_hybrid_weights(model: Any, output_dir: Path, attempts: int = 5) -> None:
    """Write via a sibling staging dir. Windows often locks in-progress `.tmp*` shards (os error 32)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        unlink_tmp_shards(output_dir)
        staging = output_dir.parent / f".tmp-stitch-{output_dir.name}-{os.getpid()}-{attempt}"
        try:
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            model.save_pretrained(staging, safe_serialization=True)
            for src in staging.iterdir():
                dest = output_dir / src.name
                if dest.exists() and dest.is_file():
                    dest.unlink()
                shutil.move(str(src), str(dest))
            shutil.rmtree(staging, ignore_errors=True)
            return
        except Exception as exc:
            last_error = exc
            text = str(exc).lower()
            transient = "being used by another process" in text or "os error 32" in text
            print(
                f"[{utc_now()}] save_pretrained attempt {attempt}/{attempts} failed: {exc}",
                flush=True,
            )
            shutil.rmtree(staging, ignore_errors=True)
            unlink_tmp_shards(output_dir)
            if not transient or attempt == attempts:
                raise
            time.sleep(2.0 * attempt)
    if last_error is not None:
        raise last_error


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


def load_rineya_alphas(payload: Any, n_layers: int) -> list[float]:
    raw: Any
    if isinstance(payload, dict) and "rineya_alphas" in payload:
        raw = payload["rineya_alphas"]
    else:
        raw = payload
    values: list[float]
    if isinstance(raw, dict):
        values = []
        missing: list[int] = []
        for index in range(n_layers):
            if str(index) in raw:
                values.append(float(raw[str(index)]))
            elif index in raw:
                values.append(float(raw[index]))
            else:
                missing.append(index)
        if missing:
            raise SystemExit(f"alpha json missing Rineya layers: {missing}")
    elif isinstance(raw, list):
        if len(raw) != n_layers:
            raise SystemExit(f"alpha json must have {n_layers} Rineya alphas, got {len(raw)}")
        values = [float(item) for item in raw]
    else:
        raise SystemExit("alpha json must be a list of 19 floats or an object with rineya_alphas")
    for index, value in enumerate(values):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"rineya_alphas[{index}]={value} is outside [0, 1]")
    return values


def stitch(
    qwen3: Path,
    donor: Path,
    output_dir: Path,
    layer_map: dict[str, Any],
    alpha: float | None,
    overwrite: bool,
    rineya_alphas: list[float] | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SystemExit(
            f"Output directory is not empty: {output_dir}. Use a new path or --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    kept = [int(value) for value in layer_map["kept_original_layer_ids"]]
    dropped = [int(value) for value in layer_map["dropped_original_layer_ids"]]
    mapping = {int(src): int(dst) for src, dst in layer_map["rineya_to_qwen3"].items()}
    n_rineya = int(layer_map["rineya_layers"])
    if sorted(mapping.values()) != sorted(kept):
        raise ValueError("rineya_to_qwen3 values must match kept_original_layer_ids")
    if sorted(kept + dropped) != list(range(int(layer_map["qwen3_layers"]))):
        raise ValueError("kept + dropped must cover all Qwen3 layers")
    if len(mapping) != n_rineya:
        raise ValueError("rineya_to_qwen3 must have one entry per Rineya layer")
    if rineya_alphas is None:
        if alpha is None:
            raise SystemExit("Provide --alpha or --alpha-json")
        rineya_alphas = [float(alpha)] * n_rineya
    if len(rineya_alphas) != n_rineya:
        raise SystemExit(f"Expected {n_rineya} Rineya alphas, got {len(rineya_alphas)}")
    for index, value in enumerate(rineya_alphas):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"rineya_alphas[{index}]={value} is outside [0, 1]")

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
    if int(donor_model.config.num_hidden_layers) != n_rineya:
        raise SystemExit(
            f"Donor layer count {donor_model.config.num_hidden_layers} "
            f"!= {n_rineya}"
        )

    qwen_state = qwen_model.state_dict()
    donor_state = donor_model.state_dict()
    blended_keys: list[str] = []
    unchanged_dropped: list[str] = []
    layer_alpha_used: dict[str, float] = {}

    with torch.no_grad():
        for rineya_i, qwen_i in sorted(mapping.items()):
            layer_alpha = float(rineya_alphas[rineya_i])
            layer_alpha_used[str(qwen_i)] = layer_alpha
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
                dst.copy_(
                    ((1.0 - layer_alpha) * dst.float() + layer_alpha * src.float()).to(dst.dtype)
                )
                blended_keys.append(dst_key)

        for qwen_i in dropped:
            unchanged_dropped.extend(sorted(layer_suffixes(qwen_state, qwen_i).values()))

    qwen_model.load_state_dict(qwen_state)
    del donor_model
    del donor_state

    print(f"[{utc_now()}] Saving hybrid to {output_dir}", flush=True)
    save_hybrid_weights(qwen_model, output_dir)
    copy_qwen3_sidecar(qwen3, output_dir)

    mean_alpha = sum(rineya_alphas) / len(rineya_alphas)
    uniform = all(abs(value - rineya_alphas[0]) < 1e-12 for value in rineya_alphas)
    manifest = {
        "format_version": 2,
        "merge_space": "FP32 entire decoder-block average on Phase 2B kept layers",
        "generated_at": utc_now(),
        "qwen3_model": str(qwen3),
        "donor_model": str(donor),
        "alpha_rineya": mean_alpha,
        "alpha_qwen3": 1.0 - mean_alpha,
        "alpha_mean": mean_alpha,
        "alpha_min": min(rineya_alphas),
        "alpha_max": max(rineya_alphas),
        "alpha_uniform": uniform,
        "rineya_alphas": rineya_alphas,
        "qwen3_kept_alphas": layer_alpha_used,
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
        description="Blend Rineya kept decoder layers into 28-layer Qwen3-0.6B."
    )
    parser.add_argument("--qwen3", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-map", type=Path, required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Uniform Rineya donor weight on kept layers. Default 0.5 if --alpha-json is omitted.",
    )
    parser.add_argument(
        "--alpha-json",
        type=Path,
        help="JSON list of 19 Rineya alphas, or an object with rineya_alphas.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer_map = load_json(args.layer_map)
    n_rineya = int(layer_map["rineya_layers"])
    rineya_alphas: list[float] | None = None
    scalar = args.alpha
    if args.alpha_json is not None:
        rineya_alphas = load_rineya_alphas(load_json(args.alpha_json), n_rineya)
    elif scalar is None:
        scalar = float(layer_map.get("alpha", 0.5))
    if rineya_alphas is None:
        if scalar is None or not 0.0 <= float(scalar) <= 1.0:
            raise SystemExit("--alpha must be in [0, 1]")
    stitch(
        qwen3=args.qwen3.expanduser().resolve(),
        donor=args.donor.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        layer_map=layer_map,
        alpha=None if rineya_alphas is not None else float(scalar),
        overwrite=args.overwrite,
        rineya_alphas=rineya_alphas,
    )


if __name__ == "__main__":
    main()
