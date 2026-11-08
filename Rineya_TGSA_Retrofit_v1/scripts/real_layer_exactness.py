#!/usr/bin/env python3
"""Per-layer real-weight exactness across dtypes and attention implementations.

Answers the numerics question the full-model report leaves open: native Qwen3
dispatches all heads through one batched interface (sdpa by default), while the
retrofit computes attention per head because the carrier must evolve between
heads. Those are mathematically identical but reduce in a different order.

Memory-cheap: loads one layer's real tensors, never the whole model.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding
from rineya_tgsa import RineyaTGSAConfig
from rineya_tgsa.qwen3_adapter import Qwen3TGSAAttention


def load_layer(path: Path, layer: int, dtype):
    keys = ("q_proj.weight","k_proj.weight","v_proj.weight","o_proj.weight",
            "q_norm.weight","k_norm.weight")
    out = {}
    with safe_open(str(path), framework="pt") as handle:
        available = set(handle.keys())
        for key in keys:
            full = f"model.layers.{layer}.self_attn.{key}"
            if full in available:
                out[key] = handle.get_tensor(full).to(dtype)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layers", default="0,14,27")
    p.add_argument("--seq", type=int, default=32)
    p.add_argument("--batch", type=int, default=2)
    args = p.parse_args()

    cfg = AutoConfig.from_pretrained(str(args.model_dir), trust_remote_code=True)
    source = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    tcfg = RineyaTGSAConfig.from_source_config(source, trajectory_rank=32)
    weights = args.model_dir / "model.safetensors"

    rows = []
    for dtype, dname in ((torch.float64,"float64"), (torch.float32,"float32"), (torch.bfloat16,"bfloat16")):
        for impl in ("eager", "sdpa"):
            for layer in [int(x) for x in args.layers.split(",")]:
                cfg._attn_implementation = impl
                torch.manual_seed(100 + layer)
                native = Qwen3Attention(cfg, layer).eval().to(dtype)
                tgsa = Qwen3TGSAAttention(cfg, layer, tcfg).eval().to(dtype)
                real = load_layer(weights, layer, dtype)
                missing = [k for k in ("q_proj.weight","k_proj.weight","v_proj.weight","o_proj.weight") if k not in real]
                if missing:
                    print(f"ERROR: layer {layer} missing {missing}", file=sys.stderr); return 1
                native.load_state_dict(real, strict=False)
                reused = tgsa.load_native_(native)

                B, T = args.batch, args.seq
                x = torch.randn(B, T, cfg.hidden_size, dtype=dtype)
                rot = Qwen3RotaryEmbedding(cfg).to(dtype)
                pos = torch.arange(T)[None].expand(B, -1)
                cos, sin = rot(x, pos)
                mask = torch.full((B, 1, T, T), 0.0, dtype=dtype)
                mask = mask + torch.full((T, T), torch.finfo(dtype).min, dtype=dtype).triu(1)

                with torch.no_grad():
                    a, _ = native(x, (cos, sin), mask)
                    b, _ = tgsa(x, (cos, sin), mask)
                diff = (b.float() - a.float()).abs()
                rows.append({
                    "dtype": dname, "attn_implementation": impl, "layer": layer,
                    "reused_tensor_count": len(reused), "reused_tensors": reused,
                    "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
                    "relative_max": float(diff.max() / a.float().abs().max().clamp_min(1e-12)),
                    "bitwise_identical": bool(torch.equal(a, b)),
                    "native_output_rms": float(a.float().square().mean().sqrt()),
                })
                print(f"  {dname:9s} {impl:6s} L{layer:<3d} reused={len(reused)} "
                      f"max|d|={rows[-1]['max_abs_diff']:.3e} rel={rows[-1]['relative_max']:.3e} "
                      f"bitwise={rows[-1]['bitwise_identical']}", flush=True)
                del native, tgsa, real

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"schema_version":"rineya.tgsa-real-layer-exactness.v1",
         "model_dir": str(args.model_dir), "batch": args.batch, "seq": args.seq,
         "results": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
