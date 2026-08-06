from __future__ import annotations

"""Create one of the explicit Phase 9 student initialization tracks.

The default ``depth_copy`` track is warm-started by copying complete teacher
blocks while preserving width/tokenizer compatibility. ``custom_qwen_random``
is an aggressive same-tokenizer architecture control; it is intentionally
marked non-production until sufficient continued pretraining/distillation proves
language readiness. ``pretrained_clone`` supports pretrained challengers such as
SmolLM without claiming cross-tokenizer logit compatibility.
"""

import argparse
import copy
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.depth_reduction import reduce_model_depth
from scripts.common.experiment_utils import environment, replace_path_with_retry, sha256_tree, write_json
from scripts.common.model_introspection import find_decoder_stack, update_config_field
from scripts.phase_9_student_distillation.phase9_common import (
    model_parameter_breakdown,
    tokenizer_fingerprint,
    tree_bytes,
)


def parse_layer_ids(value: str | None, count: int, target_count: int) -> list[int]:
    if value:
        ids = [int(part.strip()) for part in value.split(",") if part.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError("--layer-ids contains duplicates")
        if not ids or min(ids) < 0 or max(ids) >= count:
            raise ValueError(f"Layer IDs must be within 0..{count - 1}")
        return ids
    if target_count <= 0 or target_count > count:
        raise ValueError(f"--num-hidden-layers must be in 1..{count}")
    if target_count == count:
        return list(range(count))
    # Evenly cover the complete stack and preserve both endpoints.
    return sorted({round(index * (count - 1) / (target_count - 1)) for index in range(target_count)}) \
        if target_count > 1 else [count // 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Phase 9 student initialization checkpoint.")
    parser.add_argument("--source", required=True, help="Teacher checkpoint or pretrained challenger model ID/path.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("depth_copy", "custom_qwen_random", "pretrained_clone"),
        default="depth_copy",
    )
    parser.add_argument("--layer-ids", default=None)
    parser.add_argument("--num-hidden-layers", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--num-attention-heads", type=int, default=6)
    parser.add_argument("--num-key-value-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--max-position-embeddings", type=int, default=1024)
    parser.add_argument(
        "--acknowledge-random-control",
        action="store_true",
        help="Required for custom_qwen_random. This checkpoint has no inherited language weights.",
    )
    parser.add_argument("--torch-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--target-params",
        type=int,
        default=None,
        help="Optional expected total parameter count. Fails if the created student is outside tolerance.",
    )
    parser.add_argument(
        "--target-params-tolerance",
        type=float,
        default=0.12,
        help="Relative tolerance for --target-params (default 0.12 = +/-12%).",
    )
    parser.add_argument(
        "--profile-name",
        default=None,
        help="Optional profile label recorded in the creation manifest (for hybrid suites).",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> Any:
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def validate_custom_shape(args: argparse.Namespace, config: Any) -> None:
    positive = {
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "head_dim": args.head_dim,
        "max_position_embeddings": args.max_position_embeddings,
    }
    for name, value in positive.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_attention_heads % args.num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads for GQA")
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError("custom_qwen_random currently requires a Qwen3 source config")


def publish_directory(staging: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    moved = False
    try:
        if destination.exists():
            replace_path_with_retry(destination, backup)
            moved = True
        replace_path_with_retry(staging, destination)
    except BaseException:
        if moved and backup.exists() and not destination.exists():
            replace_path_with_retry(backup, destination)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite intentionally: {args.output}")
    if args.mode == "custom_qwen_random" and not args.acknowledge_random_control:
        raise ValueError(
            "custom_qwen_random is a randomly initialized control. Pass --acknowledge-random-control "
            "only when you intend to run language-readiness/continued-distillation experiments."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    source_config = AutoConfig.from_pretrained(args.source, trust_remote_code=True)
    dtype = dtype_from_name(args.torch_dtype)
    selected_layers: list[int] | None = None
    initialization_state: str

    if args.mode == "custom_qwen_random":
        validate_custom_shape(args, source_config)
        config = copy.deepcopy(source_config)
        for name, value in (
            ("hidden_size", args.hidden_size),
            ("intermediate_size", args.intermediate_size),
            ("num_hidden_layers", args.num_hidden_layers),
            ("num_attention_heads", args.num_attention_heads),
            ("num_key_value_heads", args.num_key_value_heads),
            ("head_dim", args.head_dim),
            ("max_position_embeddings", args.max_position_embeddings),
            ("max_window_layers", args.num_hidden_layers),
            ("use_cache", False),
        ):
            update_config_field(config, name, value)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=None if dtype == "auto" else dtype)
        initialization_state = "random_control_requires_language_readiness_proof"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.source,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        if args.mode == "depth_copy":
            stack = find_decoder_stack(model)
            selected_layers = sorted(
                parse_layer_ids(args.layer_ids, len(stack.layers), args.num_hidden_layers)
            )
            if len(selected_layers) < len(stack.layers):
                # Use the same proven depth-reduction helper as Phases 2B/4B.
                # Besides replacing the physical decoder stack, it keeps Qwen3
                # per-layer config metadata (notably ``layer_types``) aligned and
                # reindexes cache-facing layer IDs before serialization.
                reduce_model_depth(
                    model,
                    retained_current_layer_ids=selected_layers,
                )
            if hasattr(model.config, "max_window_layers"):
                update_config_field(model.config, "max_window_layers", len(selected_layers))
            initialization_state = "warm_started_exact_complete_layer_copy"
        else:
            initialization_state = "pretrained_challenger_clone"

    breakdown = model_parameter_breakdown(model)
    if args.target_params is not None:
        if args.target_params <= 0:
            raise ValueError("--target-params must be positive")
        if not 0.0 <= args.target_params_tolerance < 1.0:
            raise ValueError("--target-params-tolerance must be in [0, 1)")
        total = int(breakdown["total"])
        target = int(args.target_params)
        lower = int(target * (1.0 - args.target_params_tolerance))
        upper = int(target * (1.0 + args.target_params_tolerance))
        if total < lower or total > upper:
            raise ValueError(
                f"Student parameter count {total:,} is outside the target band "
                f"[{lower:,}, {upper:,}] for --target-params {target:,} "
                f"(tolerance={args.target_params_tolerance:.3f})"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_name(f".{args.output.name}.tmp-{uuid.uuid4().hex}")
    try:
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        if not (staging / "config.json").exists():
            raise RuntimeError("Student serialization is missing config.json")
        if not any(staging.glob("*.safetensors")) and not any(staging.glob("pytorch_model*.bin")):
            raise RuntimeError("Student serialization contains no model weights")
        manifest = {
            "phase": "9_student_distillation",
            "schema_version": "1.0",
            "profile_name": args.profile_name,
            "source": args.source,
            "source_tree_sha256": sha256_tree(Path(args.source)) if Path(args.source).exists() else None,
            "mode": args.mode,
            "initialization_state": initialization_state,
            "selected_teacher_layers": selected_layers,
            "architecture": model.config.to_dict(),
            "parameter_breakdown": breakdown,
            "target_params": args.target_params,
            "target_params_tolerance": args.target_params_tolerance if args.target_params is not None else None,
            "idealized_weight_bytes": {
                "fp32": breakdown["total"] * 4,
                "fp16_bf16": breakdown["total"] * 2,
                "int8": breakdown["total"],
                "int4_raw_only": breakdown["total"] / 2,
            },
            "embedding_parameter_fraction": breakdown["input_embedding"] / max(1, breakdown["total"]),
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
            "tokenizer_vocab_size": len(tokenizer),
            "environment": environment(),
            "production_eligibility_note": (
                "Random narrow students are controls until continued pretraining/distillation and the complete "
                "Phase 9 evaluation prove language readiness. Depth-copy and pretrained tracks inherit language weights."
            ),
        }
        write_json(staging / "student_creation_manifest.json", manifest)
        if args.output.exists():
            shutil.rmtree(args.output)
        publish_directory(staging, args.output)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    print(f"Student checkpoint: {args.output}")
    if args.profile_name:
        print(f"Profile: {args.profile_name}")
    print(f"Initialization: {initialization_state}")
    print(f"Parameters: {breakdown['total']:,}")
    print(f"Checkpoint bytes: {tree_bytes(args.output):,}")


if __name__ == "__main__":
    main()
