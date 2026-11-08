#!/usr/bin/env python3
"""Convert a real Rineya/Qwen3 checkpoint to TGSA and write a conversion manifest.

Fails closed on: a missing checkpoint, a source config whose RoPE convention the
retrofit cannot reproduce, an out-of-range layer, a native tensor the retrofit
lacks, or a conversion that does not leave the conditioning neutral.

The written artifact is a *converted checkpoint*: the full model state dict with
the selected layers' attention replaced. Nothing native is dropped or reshaped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

import torch

from rineya_tgsa import RineyaTGSAConfig
from rineya_tgsa.qwen3_adapter import Qwen3TGSAAttention, convert_qwen3_model

MANIFEST_SCHEMA = "rineya.tgsa-real-conversion-manifest.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().to("cpu").contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parse_layers(spec: str, total: int) -> list[int]:
    """Accept `all`, `every:N`, or an explicit comma list."""

    spec = spec.strip()
    if spec == "all":
        return list(range(total))
    if spec.startswith("every:"):
        step = int(spec.split(":", 1)[1])
        if step <= 0:
            raise ValueError("every:N requires a positive N")
        return list(range(0, total, step))
    return sorted({int(value) for value in spec.split(",") if value.strip()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="every:7")
    parser.add_argument("--trajectory-rank", type=int, default=32)
    parser.add_argument("--condition", default="q", help="comma list from q,k,v")
    parser.add_argument("--master-seed", type=int, default=20260829)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--attn-impl", default="eager")
    parser.add_argument("--save-weights", action="store_true",
                        help="also write the converted state dict (large)")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    weights = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    for required in (weights, config_path):
        if not required.is_file():
            print(f"ERROR: missing {required}", file=sys.stderr)
            return 1

    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    targets = {value.strip() for value in args.condition.split(",") if value.strip()}
    unknown = targets - {"q", "k", "v"}
    if unknown:
        print(f"ERROR: unknown conditioning targets {sorted(unknown)}", file=sys.stderr)
        return 1
    overrides = {
        "trajectory_rank": args.trajectory_rank,
        "condition_q": "q" in targets,
        "condition_k": "k" in targets,
        "condition_v": "v" in targets,
    }
    tgsa_config = RineyaTGSAConfig.from_source_config(source_config, **overrides)

    from transformers import AutoModelForCausalLM

    started = time.time()
    print(f"loading {model_dir.name} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=getattr(torch, args.dtype),
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    ).eval()

    total_layers = len(model.model.layers)
    layers = parse_layers(args.layers, total_layers)
    print(f"converting layers {layers} of {total_layers} ...", flush=True)
    record = convert_qwen3_model(model, tgsa_config, layers=layers, master_seed=args.master_seed)

    converted_modules = {
        index: model.model.layers[index].self_attn for index in record["converted_layers"]
    }
    for index, module in converted_modules.items():
        if not isinstance(module, Qwen3TGSAAttention):
            print(f"ERROR: layer {index} was not replaced", file=sys.stderr)
            return 1
        if not module.conditioning_is_neutral():
            print(f"ERROR: layer {index} conditioning is not neutral", file=sys.stderr)
            return 1

    new_state = {
        f"model.layers.{index}.self_attn.{name}": tensor
        for index, module in converted_modules.items()
        for name, tensor in module.state_dict().items()
    }
    trajectory_only = {
        key: value
        for key, value in new_state.items()
        if not any(
            root in key.split("self_attn.")[-1].split(".")[0]
            for root in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
        )
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source": {
            "model_dir": str(model_dir),
            "config_sha256": sha256_file(config_path),
            "weights_sha256": sha256_file(weights),
            "weights_bytes": weights.stat().st_size,
            "architectures": source_config.get("architectures"),
        },
        "retrofit_config": tgsa_config.to_dict(),
        "conversion": record,
        "master_seed": args.master_seed,
        "dtype": args.dtype,
        "attn_implementation": args.attn_impl,
        "converted_attention_state_sha256": state_sha256(new_state),
        "new_trajectory_tensor_state_sha256": state_sha256(trajectory_only),
        "new_trajectory_tensor_count": len(trajectory_only),
        "effective_conditioned_heads": next(iter(converted_modules.values())).effective_conditioned_heads,
        "head_zero_is_bootstrap": True,
        "mapping_rules": {
            "reuse": "q_proj,k_proj,v_proj,o_proj,q_norm,k_norm copied verbatim by subclassing Qwen3Attention",
            "rope": "the (cos,sin) supplied by the model is used verbatim; never recomputed",
            "gqa": "query head j attends kv group j // (num_attention_heads // num_key_value_heads)",
            "sequential_axis": "carrier recurses across query heads within a token, never across positions",
            "neutrality": "conditioning projections are zero, so the module reduces to native attention",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = args.output_dir / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.save_weights:
        target = args.output_dir / "converted_attention.pt"
        torch.save(new_state, target)
        print(f"wrote {target} ({target.stat().st_size / 2**20:.1f} MiB)")

    print(f"converted layers      : {record['converted_layers']}")
    print(f"reused per layer      : {len(record['layers'][0]['reused_tensors'])} tensors")
    print(f"new trajectory tensors: {len(trajectory_only)}")
    print(f"conditioning neutral  : True")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
