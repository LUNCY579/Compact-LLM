from __future__ import annotations

"""Create one reloadable Phase 10 structural student candidate.

The implementation supports four orthogonal reductions on Qwen/Llama-style
causal LMs: complete-layer deletion, gated-MLP channel slicing, grouped-query
attention group slicing, and residual hidden-coordinate slicing. The operations
create physically smaller dense tensors; they are not masks.
"""

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.depth_reduction import reduce_model_depth
from scripts.common.experiment_utils import environment, sha256, sha256_tree, write_json
from scripts.common.model_introspection import count_parameters, find_decoder_stack
from scripts.phase_9_student_distillation.phase9_common import tree_bytes
from scripts.phase_10_student_architecture.phase10_common import (
    apply_attention_group_slicing,
    apply_hidden_coordinate_slicing,
    apply_mlp_channel_slicing,
    retained_layer_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Phase 10 structural student candidate.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--target-role", default="experimental_candidate")
    parser.add_argument("--target-parameter-min", type=int, default=None)
    parser.add_argument("--target-parameter-max", type=int, default=None)
    parser.add_argument("--depth-drop", type=int, default=0)
    parser.add_argument("--depth-mode", choices=("uniform-drop", "drop-last"), default="uniform-drop")
    parser.add_argument("--hidden-prune-ratio", type=float, default=0.0)
    parser.add_argument("--mlp-prune-ratio", type=float, default=0.0)
    parser.add_argument("--attention-group-prune-ratio", type=float, default=0.0)
    parser.add_argument("--hidden-multiple", type=int, default=64)
    parser.add_argument("--mlp-multiple", type=int, default=64)
    parser.add_argument("--smoke-text", default="Use the supplied evidence and answer briefly.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-reload-check", action="store_true")
    return parser.parse_args()


def publish(staging: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    moved = False
    try:
        if destination.exists():
            destination.replace(backup)
            moved = True
        staging.replace(destination)
    except BaseException:
        if moved and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite intentionally: {args.output}")
    for name, value in (
        ("hidden_prune_ratio", args.hidden_prune_ratio),
        ("mlp_prune_ratio", args.mlp_prune_ratio),
        ("attention_group_prune_ratio", args.attention_group_prune_ratio),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0,1)")
    if args.depth_drop < 0:
        raise ValueError("depth_drop cannot be negative")
    if args.target_parameter_min is not None and args.target_parameter_min <= 0:
        raise ValueError("target_parameter_min must be positive")
    if args.target_parameter_max is not None and args.target_parameter_max <= 0:
        raise ValueError("target_parameter_max must be positive")
    if (
        args.target_parameter_min is not None
        and args.target_parameter_max is not None
        and args.target_parameter_min > args.target_parameter_max
    ):
        raise ValueError("target_parameter_min cannot exceed target_parameter_max")
    if args.depth_drop == 0 and not any(
        value > 0 for value in (
            args.hidden_prune_ratio,
            args.mlp_prune_ratio,
            args.attention_group_prune_ratio,
        )
    ):
        raise ValueError("Candidate performs no structural reduction")
    if not args.profile.exists():
        raise FileNotFoundError(args.profile)

    profile = torch.load(args.profile, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    original_stack = find_decoder_stack(model)
    original_layer_count = len(original_stack.layers)
    if int(profile.get("layer_count", -1)) != original_layer_count:
        raise ValueError(
            f"Profile layer count {profile.get('layer_count')} does not match model {original_layer_count}"
        )
    before = count_parameters(model)
    retained_current = retained_layer_indices(
        original_layer_count,
        drop_count=args.depth_drop,
        mode=args.depth_mode,
    )
    current_original_layer_ids = list(range(original_layer_count))
    depth_manifest: dict[str, Any] = {
        "original_layer_count": original_layer_count,
        "retained_original_layer_ids": current_original_layer_ids,
        "dropped_original_layer_ids": [],
    }
    if len(retained_current) < original_layer_count:
        result = reduce_model_depth(model, retained_current_layer_ids=retained_current)
        current_original_layer_ids = list(result.retained_original_layer_ids)
        depth_manifest = {
            "original_layer_count": result.original_layer_count,
            "retained_original_layer_ids": list(result.retained_original_layer_ids),
            "dropped_original_layer_ids": list(result.dropped_original_layer_ids),
        }

    mlp_importance = {
        int(index): tensor for index, tensor in (profile.get("mlp_importance") or {}).items()
    }
    attention_importance = {
        int(index): tensor for index, tensor in (profile.get("attention_head_importance") or {}).items()
    }
    mlp_manifest = apply_mlp_channel_slicing(
        model,
        importance_by_original_layer=mlp_importance,
        current_original_layer_ids=current_original_layer_ids,
        prune_ratio=args.mlp_prune_ratio,
        multiple=args.mlp_multiple,
    )
    attention_manifest = apply_attention_group_slicing(
        model,
        importance_by_original_layer=attention_importance,
        current_original_layer_ids=current_original_layer_ids,
        prune_ratio=args.attention_group_prune_ratio,
    )
    hidden_manifest = apply_hidden_coordinate_slicing(
        model,
        hidden_importance=profile["hidden_importance"],
        prune_ratio=args.hidden_prune_ratio,
        multiple=args.hidden_multiple,
    )

    encoded = tokenizer(args.smoke_text, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded, use_cache=False)
    if not torch.isfinite(output.logits).all():
        raise RuntimeError("Structural candidate smoke forward produced non-finite logits")
    after = count_parameters(model)
    if after["total"] >= before["total"]:
        raise RuntimeError("Structural candidate did not reduce parameter count")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_name(f".{args.output.name}.tmp-{uuid.uuid4().hex}")
    try:
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        reload_shape = None
        if not args.skip_reload_check:
            reloaded = AutoModelForCausalLM.from_pretrained(
                staging,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map=None,
                low_cpu_mem_usage=True,
            )
            reloaded.eval()
            with torch.no_grad():
                reloaded_output = reloaded(**encoded, use_cache=False)
            if not torch.isfinite(reloaded_output.logits).all():
                raise RuntimeError("Reloaded structural candidate produced non-finite logits")
            reload_shape = list(reloaded_output.logits.shape)

        manifest = {
            "phase": "10_student_architecture",
            "schema_version": "1.0",
            "name": args.name,
            "description": args.description or args.name,
            "target": {
                "role": args.target_role,
                "parameter_min": args.target_parameter_min,
                "parameter_max": args.target_parameter_max,
                "parameter_band_match": (
                    (args.target_parameter_min is None or after["total"] >= args.target_parameter_min)
                    and (args.target_parameter_max is None or after["total"] <= args.target_parameter_max)
                ),
            },
            "source_model": args.model,
            "source_tree_sha256": sha256_tree(Path(args.model)) if Path(args.model).exists() else None,
            "profile": str(args.profile),
            "profile_sha256": sha256(args.profile),
            "operations": {
                "depth_drop": args.depth_drop,
                "depth_mode": args.depth_mode,
                "hidden_prune_ratio": args.hidden_prune_ratio,
                "mlp_prune_ratio": args.mlp_prune_ratio,
                "attention_group_prune_ratio": args.attention_group_prune_ratio,
            },
            "depth": depth_manifest,
            "mlp": mlp_manifest,
            "attention": attention_manifest,
            "hidden": hidden_manifest,
            "parameters_before": before,
            "parameters_after": after,
            "parameter_reduction_fraction": 1.0 - after["total"] / before["total"],
            "checkpoint_bytes": tree_bytes(staging),
            "smoke_logits_shape": list(output.logits.shape),
            "reload_check_performed": not args.skip_reload_check,
            "reload_smoke_logits_shape": reload_shape,
            "environment": environment(),
            "limitations": [
                "Activation-magnitude selection is an experimental coordinate/channel/head heuristic, not a proof of semantic redundancy.",
                "Hidden-coordinate slicing is not a full SliceGPT orthogonal-invariance implementation and may require substantial recovery.",
                "Passing reload and smoke checks establishes technical validity only; promotion requires Phase 10 RAG evaluation and human review.",
            ],
        }
        write_json(staging / "phase_10_candidate_manifest.json", manifest)
        publish(staging, args.output)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    print(f"Candidate: {args.output}")
    print(f"Parameters: {before['total']:,} -> {after['total']:,}")
    print(f"Reduction: {1.0 - after['total'] / before['total']:.4%}")


if __name__ == "__main__":
    main()
