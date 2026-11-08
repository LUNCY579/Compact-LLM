"""Bind a dense Titan 0.6B checkpoint using files and safetensors headers only.

This command imports no ML library, allocates no tensor, and runs no model. The
output directory must not exist: prior 344M receipts and prior bindings survive.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys


TITAN_WEIGHT_SHA256 = "537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7"
TITAN_CORE_ID = "rineya.core.titan-experimental.v1.0.0-devfreeze.28L.supa17-rrp015-control-seed17"
PARAMETERS = 596_049_920
REQUIRED_FILES = (
    "model.safetensors", "config.json", "CORE.json", "tokenizer.json",
    "tokenizer_config.json", "generation_config.json", "merges.txt", "vocab.json", "LICENSE",
)
GEOMETRY = {
    "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
    "num_hidden_layers": 28, "hidden_size": 1024, "intermediate_size": 3072,
    "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
    "vocab_size": 151936, "tie_word_embeddings": True,
    "max_position_embeddings": 40960, "rope_theta": 1000000,
    "attention_bias": False, "use_sliding_window": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def inspect_safetensors(path: Path) -> dict:
    """Validate the complete byte layout without reading any tensor payload."""
    file_bytes = path.stat().st_size
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        require(len(length_bytes) == 8, "truncated safetensors length")
        header_bytes = struct.unpack("<Q", length_bytes)[0]
        require(2 <= header_bytes <= 32 * 1024 * 1024, "unreasonable safetensors header size")
        require(8 + header_bytes <= file_bytes, "header exceeds file")
        header = json.loads(stream.read(header_bytes), object_pairs_hook=unique_object)
    require(isinstance(header, dict), "header must be an object")
    tensors, intervals = {}, []
    dtype_counts = Counter()
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        require(isinstance(entry, dict), f"invalid tensor descriptor: {name}")
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        offsets = entry.get("data_offsets")
        require(isinstance(shape, list) and all(type(d) is int and d > 0 for d in shape),
                f"invalid shape: {name}")
        require(dtype in {"BF16", "F32"}, f"unexpected dtype: {name}: {dtype}")
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(type(d) is int and d >= 0 for d in offsets), f"invalid offsets: {name}")
        count = math.prod(shape)
        require(offsets[1] - offsets[0] == count * {"BF16": 2, "F32": 4}[dtype],
                f"shape/byte count mismatch: {name}")
        tensors[name] = {"shape": shape, "dtype": dtype, "data_offsets": offsets,
                         "parameter_count": count}
        intervals.append((offsets[0], offsets[1], name))
        dtype_counts[dtype] += 1
    require(bool(tensors), "checkpoint has no tensors")
    end = 0
    for start, stop, name in sorted(intervals):
        require(start == end, f"overlap or gap before tensor: {name}")
        end = stop
    require(end + 8 + header_bytes == file_bytes, "payload size disagrees with tensor inventory")
    return {"tensor_count": len(tensors), "parameter_count_serialized":
            sum(t["parameter_count"] for t in tensors.values()), "dtype_counts": dict(dtype_counts),
            "header_bytes": header_bytes, "payload_bytes": end, "file_bytes": file_bytes,
            "metadata": header.get("__metadata__", {}), "tensors": tensors}


def validate_geometry(config: dict, inventory: dict) -> None:
    for key, expected in GEOMETRY.items():
        require(config.get(key) == expected, f"Titan config mismatch: {key}")
    require(not config.get("auto_map") and not config.get("quantization_config"),
            "custom loader or quantized model is outside this binding")
    require(config.get("torch_dtype") == "bfloat16", "dense Titan storage must remain BF16")
    expected = {"model.embed_tokens.weight": [151936, 1024], "model.norm.weight": [1024]}
    for i in range(28):
        prefix = f"model.layers.{i}."
        for suffix, shape in {
            "input_layernorm.weight": [1024], "post_attention_layernorm.weight": [1024],
            "mlp.down_proj.weight": [1024, 3072], "mlp.gate_proj.weight": [3072, 1024],
            "mlp.up_proj.weight": [3072, 1024], "self_attn.k_norm.weight": [128],
            "self_attn.q_norm.weight": [128], "self_attn.k_proj.weight": [1024, 1024],
            "self_attn.v_proj.weight": [1024, 1024], "self_attn.q_proj.weight": [2048, 1024],
            "self_attn.o_proj.weight": [1024, 2048],
        }.items():
            expected[prefix + suffix] = shape
    actual = {name: row["shape"] for name, row in inventory["tensors"].items()}
    require(actual == expected, "checkpoint tensor names/shapes differ from exact dense Titan geometry")
    require(inventory["parameter_count_serialized"] == PARAMETERS, "parameter count mismatch")
    require(inventory["dtype_counts"] == {"BF16": 310}, "dtype or tensor count mismatch")


def git_state() -> dict:
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"code_sha": git("rev-parse", "HEAD"), "code_branch": git("branch", "--show-current"),
            "dirty_status": git("status", "--porcelain")}


def create_receipts(checkpoint: Path, expected_weight_sha256: str) -> dict[str, dict]:
    checkpoint = checkpoint.resolve(strict=True)
    require(checkpoint.is_dir(), "checkpoint must be a directory")
    require(re.fullmatch(r"[0-9a-f]{64}", expected_weight_sha256) is not None,
            "expected weight hash must be a lowercase SHA256")
    require(expected_weight_sha256 == TITAN_WEIGHT_SHA256,
            "this binder admits only the exact preregistered dense Titan checkpoint")
    for name in REQUIRED_FILES:
        require((checkpoint / name).is_file() and not (checkpoint / name).is_symlink(),
                f"missing/nonregular required asset: {name}")
    files = {}
    for path in sorted(checkpoint.iterdir()):
        if path.is_file():
            require(not path.is_symlink(), f"symlink root asset: {path.name}")
            before = path.stat()
            digest = sha256_file(path)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    f"asset changed during hash: {path.name}")
            files[path.name] = {"bytes": after.st_size, "sha256": digest}
    require(files["model.safetensors"]["sha256"] == expected_weight_sha256, "weight SHA256 mismatch")
    inventory = inspect_safetensors(checkpoint / "model.safetensors")
    config = read_json(checkpoint / "config.json")
    validate_geometry(config, inventory)
    core = read_json(checkpoint / "CORE.json")
    require(core.get("core_id") == TITAN_CORE_ID, "CORE identity mismatch")
    require(core.get("checks", {}).get("weight_sha256") == expected_weight_sha256,
            "CORE weight binding mismatch")
    published_assets = core.get("checks", {}).get("preserved_asset_sha256", {})
    for name in REQUIRED_FILES:
        if name in {"model.safetensors", "CORE.json"}:
            continue
        require(published_assets.get(name) == files[name]["sha256"],
                f"asset hash disagrees with published CORE: {name}")
    tokenizer_config = read_json(checkpoint / "tokenizer_config.json")
    tokenizer_data = read_json(checkpoint / "tokenizer.json")
    template = tokenizer_config.get("chat_template")
    require(isinstance(template, str) and bool(template), "unbound chat template")
    added = {row["content"]: row["id"] for row in tokenizer_data["added_tokens"]}
    require(added.get("<|endoftext|>") == 151643 and added.get("<|im_start|>") == 151644
            and added.get("<|im_end|>") == 151645, "tokenizer special IDs mismatch")
    require(tokenizer_config.get("eos_token") == "<|im_end|>"
            and tokenizer_config.get("pad_token") == "<|endoftext|>", "EOS/PAD contract mismatch")
    # Reject a concurrent sidecar edit between hashing and interpretation.
    for name in ("config.json", "CORE.json", "tokenizer.json", "tokenizer_config.json",
                 "generation_config.json"):
        require(sha256_file(checkpoint / name) == files[name]["sha256"], f"asset drift: {name}")
    common = {"schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "evidence_level": "MEASURED_STATIC_FILES_AND_HEADERS", "native_model_loads": 0,
              "gpu_runs": 0, "training_updates": 0, "checkpoint_path": str(checkpoint),
              "weight_sha256": expected_weight_sha256, "binder_path": str(Path(__file__).resolve()),
              "binder_sha256": sha256_file(Path(__file__)), **git_state()}
    tokenizer_receipt = {
        **common, "status": "STATIC_BOUND_RUNTIME_UNQUALIFIED",
        "tokenizer_identity": tokenizer_config.get("tokenizer_class"),
        "tokenizer_sha256": files["tokenizer.json"]["sha256"],
        "tokenizer_files": {n: files[n] for n in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")},
        "chat_template": template, "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "special_token_ids": added, "eos_token_id": 151645, "pad_token_id": 151643,
        "generation_config": read_json(checkpoint / "generation_config.json"),
        "tokenizer_declared_max_context": tokenizer_config.get("model_max_length"),
        "model_declared_max_context": config["max_position_embeddings"],
        "qualified_max_context": None, "runtime_tokenizer_initialized": False,
        "prompt_token_ids_sha256": None, "registered_template_kwargs": None,
        "unmeasured": ["runtime rendering", "raw tokenization", "completion masks", "EOS and PAD edge cases"],
    }
    primary = {
        **common, "status": "PRIMARY_0P6B_STATIC_BOUND_E0_NATIVE_REQUIRED", "primary_student": TITAN_CORE_ID,
        "model_family": "Rineya Titan", "primary_scale": "0.6B-class", "parameter_count": PARAMETERS,
        "selection_reason": "Exact available dense Titan is the simplest Titan-series 0.6B starting point for isolated LoRA qualification. This is not a claim of superiority over Titan-Sparse or Titan-Transition.",
        "architecture": config, "architecture_config_sha256": files["config.json"]["sha256"],
        "tensor_count": inventory["tensor_count"], "stored_dtype": "BF16", "files": files,
        "file_inventory_scope": "All regular checkpoint-root files; evidence subdirectories are not recursively inventoried.",
        "tied_embeddings": {"declared": True, "serialized_lm_head_present": False,
                            "unique_serialized_parameters": PARAMETERS, "loaded_pointer_tie_verified": False},
        "tokenizer_sha256": tokenizer_receipt["tokenizer_sha256"],
        "chat_template_sha256": tokenizer_receipt["chat_template_sha256"],
        "declared_max_context": config["max_position_embeddings"], "qualified_max_context": None,
        "candidate_loader": "transformers.Qwen3ForCausalLM.from_pretrained(local_files_only=True, trust_remote_code=False, use_safetensors=True, attn_implementation='eager')",
        "qualified_loader": None, "qualified_runtime": None,
        "prior_quality_evidence": "RECORDED_FROM_PRIOR_WORK_NOT_REPRODUCED; stored BF16 Titan was evaluated through FP32 DenseReference; native BF16 serving was not qualified.",
        "blockers": [
            "Stock Qwen3 and historical DenseReference semantics/parity are not yet qualified on this binding.",
            "BF16 training/inference arithmetic cannot inherit FP32 historical quality measurements.",
            "E0 tokenizer, likelihood, masking, zero-LoRA, gradient confinement, optimizer update/save/resume and resource gates remain unmeasured.",
            "40960 is the config context declaration, not a tested operating envelope; start at 256/512/1024 only after admission.",
        ],
        "planned_rank8_qv_lora_parameters": 1_146_880,
    }
    parameter_receipt = {**common, "status": "HEADER_INVENTORY_VERIFIED_NO_TENSOR_LOAD",
                         **inventory, "unique_parameter_count": PARAMETERS,
                         "unique_parameter_count_basis": "Exact expected tensor set, tied embeddings declared, no serialized lm_head alias; loaded tie remains unqualified."}
    return {"PRIMARY_STUDENT_RECEIPT.json": primary,
            "BASE_MODEL_RECEIPT.json": {**primary, "receipt_role": "base alias of exact primary student"},
            "PARAMETER_INVENTORY.json": parameter_receipt, "TOKENIZER_RECEIPT.json": tokenizer_receipt}


def write_receipts(output_dir: Path, receipts: dict[str, dict]) -> None:
    require(not output_dir.exists(), "output directory already exists; bindings are write-once")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, receipt in receipts.items():
        with (output_dir / name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(receipt, stream, indent=2, ensure_ascii=False)
            stream.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        require(not args.output_dir.exists(), "output directory already exists; bindings are write-once")
        receipts = create_receipts(args.checkpoint, args.expected_weight_sha256)
        write_receipts(args.output_dir, receipts)
    except (ValueError, OSError, KeyError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"BINDING_FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "PRIMARY_0P6B_STATIC_BOUND_E0_NATIVE_REQUIRED",
                      "output_dir": str(args.output_dir.resolve()), "parameter_count": PARAMETERS,
                      "native_model_loads": 0, "gpu_runs": 0, "training_updates": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
