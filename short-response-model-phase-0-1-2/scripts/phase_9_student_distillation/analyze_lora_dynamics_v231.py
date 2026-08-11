from __future__ import annotations

"""Checkpoint-wise LoRA norm/rank diagnostics for v2.3.1.

The non-zero singular values of B@A are computed from a small r×r core via QR,
so the analysis does not materialize full dense deltas for every module.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.grokking_utils_v231 import discover_adapters, read_json, write_csv, write_json


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def adapter_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    cfg = read_json(path / "adapter_config.json")
    safe = path / "adapter_model.safetensors"
    binary = path / "adapter_model.bin"
    if safe.exists():
        from safetensors.torch import load_file
        state = load_file(str(safe), device="cpu")
    elif binary.exists():
        value = torch.load(binary, map_location="cpu", weights_only=True)
        state = dict(value)
    else:
        raise FileNotFoundError(f"No adapter_model.safetensors/bin in {path}")
    return state, cfg


def pair_key(name: str, token: str) -> str:
    return name.replace(token, "lora_PAIR")


def effective_rank(singular_values: torch.Tensor) -> float:
    values = singular_values.float().abs()
    total = float(values.sum().item())
    if total <= 0.0:
        return 0.0
    p = values / total
    entropy = float(-(p * torch.log(p.clamp_min(1e-30))).sum().item())
    return float(math.exp(entropy))


def module_stats(state: dict[str, torch.Tensor], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    a_map: dict[str, tuple[str, torch.Tensor]] = {}
    b_map: dict[str, tuple[str, torch.Tensor]] = {}
    for name, tensor in state.items():
        if ".lora_A." in name or name.endswith(".lora_A.weight"):
            a_map[pair_key(name, "lora_A")] = (name, tensor.float())
        elif ".lora_B." in name or name.endswith(".lora_B.weight"):
            b_map[pair_key(name, "lora_B")] = (name, tensor.float())
    rank_cfg = cfg.get("r") or cfg.get("rank_pattern") or 1
    alpha_cfg = cfg.get("lora_alpha") or cfg.get("alpha_pattern") or rank_cfg
    rows: list[dict[str, Any]] = []
    for key in sorted(set(a_map) & set(b_map)):
        aname, A = a_map[key]
        bname, B = b_map[key]
        rank = int(A.shape[0])
        alpha = float(alpha_cfg if isinstance(alpha_cfg, (int, float)) else rank)
        scaling = alpha / max(1, rank)
        # B = Qb Rb ; A^T = Qa Ra ; BA = Qb (Rb Ra^T) Qa^T.
        _qb, rb = torch.linalg.qr(B, mode="reduced")
        _qa, ra = torch.linalg.qr(A.T, mode="reduced")
        core = rb @ ra.T
        sv = torch.linalg.svdvals(core) * scaling
        delta_norm = float(torch.linalg.vector_norm(sv).item())
        layer_match = LAYER_RE.search(aname)
        rows.append(
            {
                "module": key,
                "layer": int(layer_match.group(1)) if layer_match else None,
                "rank": rank,
                "alpha": alpha,
                "scaling": scaling,
                "a_norm": float(torch.linalg.vector_norm(A).item()),
                "b_norm": float(torch.linalg.vector_norm(B).item()),
                "delta_norm": delta_norm,
                "delta_effective_rank": effective_rank(sv),
                "delta_singular_max": float(sv.max().item()) if sv.numel() else 0.0,
            }
        )
    return rows


def base_target_norms(base_model: str, module_names: list[str]) -> dict[str, float]:
    """Best-effort suffix mapping to anchor target-module weights."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype="auto", device_map=None, low_cpu_mem_usage=True)
    params = {name: param.detach().float().cpu() for name, param in model.named_parameters()}
    out: dict[str, float] = {}
    for module in module_names:
        # Strip PEFT wrapper prefixes and lora_PAIR suffix.
        root = module.replace("base_model.model.", "").replace("lora_PAIR.default.weight", "weight").replace("lora_PAIR.weight", "weight")
        candidates = [name for name in params if name.endswith(root)]
        if not candidates:
            root2 = root.replace("model.model.", "model.")
            candidates = [name for name in params if name.endswith(root2)]
        if candidates:
            out[module] = float(torch.linalg.vector_norm(params[min(candidates, key=len)]).item())
    del model
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", type=Path, required=True)
    ap.add_argument("--base-model")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--include-anchor-relative-norm", action="store_true")
    args = ap.parse_args()
    checkpoints = [item for item in discover_adapters(args.training_root) if item.get("adapter")]
    per_checkpoint: list[dict[str, Any]] = []
    base_norms: dict[str, float] = {}
    for index, item in enumerate(checkpoints):
        state, cfg = adapter_state(Path(str(item["adapter"])))
        modules = module_stats(state, cfg)
        if args.include_anchor_relative_norm and args.base_model and not base_norms and modules:
            base_norms = base_target_norms(args.base_model, [row["module"] for row in modules])
        layer_sq: dict[int, float] = defaultdict(float)
        for row in modules:
            if row["layer"] is not None:
                layer_sq[int(row["layer"])] += float(row["delta_norm"]) ** 2
        delta_sq = sum(float(row["delta_norm"]) ** 2 for row in modules)
        base_sq = sum(float(base_norms.get(row["module"], 0.0)) ** 2 for row in modules)
        per_checkpoint.append(
            {
                "step": int(item["step"]),
                "adapter": str(item["adapter"]),
                "module_count": len(modules),
                "lora_A_frobenius_norm": math.sqrt(sum(float(row["a_norm"]) ** 2 for row in modules)),
                "lora_B_frobenius_norm": math.sqrt(sum(float(row["b_norm"]) ** 2 for row in modules)),
                "effective_delta_frobenius_norm": math.sqrt(delta_sq),
                "relative_delta_norm_to_anchor_target_weights": (math.sqrt(delta_sq) / math.sqrt(base_sq)) if base_sq > 0 else None,
                "delta_effective_rank_mean": (sum(float(row["delta_effective_rank"]) for row in modules) / len(modules)) if modules else None,
                "delta_effective_rank_max": max([float(row["delta_effective_rank"]) for row in modules] or [0.0]),
                "delta_singular_max": max([float(row["delta_singular_max"]) for row in modules] or [0.0]),
                "per_layer_delta_norm": {str(layer): math.sqrt(value) for layer, value in sorted(layer_sq.items())},
            }
        )
    payload = {
        "schema_version": "2.3.1",
        "training_root": str(args.training_root),
        "anchor_relative_norm_computed": bool(base_norms),
        "checkpoints": per_checkpoint,
    }
    write_json(args.output_json, payload)
    flat = [{key: value for key, value in row.items() if key != "per_layer_delta_norm"} for row in per_checkpoint]
    write_csv(args.output_csv, flat)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
