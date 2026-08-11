from __future__ import annotations

"""Compare a candidate full checkpoint against an official Phase-4 delta soup."""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_weights(model_dir: Path) -> dict[str, torch.Tensor]:
    path = model_dir / "model.safetensors"
    if not path.exists():
        shards = sorted(model_dir.glob("model-*.safetensors"))
        if len(shards) != 1:
            raise FileNotFoundError(model_dir)
        path = shards[0]
    return load_file(str(path), device="cpu")


def compare(a: Path, b: Path) -> dict:
    sa, sb = load_weights(a), load_weights(b)
    if set(sa) != set(sb):
        return {
            "equivalent": False,
            "reason": "key_mismatch",
            "missing_in_b": sorted(set(sa) - set(sb))[:10],
            "extra_in_b": sorted(set(sb) - set(sa))[:10],
        }
    max_abs = 0.0
    sum_sq_a = 0.0
    sum_sq_b = 0.0
    sum_ab = 0.0
    for key in sa:
        xa = sa[key].float().reshape(-1)
        xb = sb[key].float().reshape(-1)
        diff = (xa - xb).abs().max().item()
        max_abs = max(max_abs, diff)
        sum_sq_a += float(torch.dot(xa, xa))
        sum_sq_b += float(torch.dot(xb, xb))
        sum_ab += float(torch.dot(xa, xb))
    cosine = sum_ab / math.sqrt(sum_sq_a * sum_sq_b) if sum_sq_a > 0 and sum_sq_b > 0 else None
    # LoRA merge / dtype casting can create tiny numeric drift; 1e-4 is generous for fp16/bf16 roundtrip.
    equivalent = max_abs < 1e-3
    return {
        "equivalent": equivalent,
        "max_abs_diff": max_abs,
        "cosine_similarity": cosine,
        "tensor_count": len(sa),
        "left": str(a),
        "right": str(b),
        "tolerance": 1e-3,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--left", type=Path, required=True, help="Candidate soup checkpoint")
    p.add_argument("--right", type=Path, required=True, help="Phase-4 merge_delta_soup checkpoint")
    p.add_argument("--output-json", type=Path)
    args = p.parse_args()
    result = compare(args.left, args.right)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if not result["equivalent"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
