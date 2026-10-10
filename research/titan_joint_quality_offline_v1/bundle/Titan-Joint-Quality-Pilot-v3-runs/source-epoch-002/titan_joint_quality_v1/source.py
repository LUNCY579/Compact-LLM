"""Read-only source identity and safetensors-header audit; no tensor loading."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
import struct
from .common import BUSINESS, PARENT_SHA, SUPA_SHA, file_sha, utc

ROOT = BUSINESS / "Chat-Bubble/Rineya_Published_Cores"
MODEL_SOURCES = {
    "supa": (ROOT / "Rineya-Supa-1.7.0-devfreeze", SUPA_SHA, "BF16"),
    "sparse": (ROOT / "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze", PARENT_SHA, "F32"),
    "transition": (ROOT / "Rineya-Titan-Transition-1.0.0-devfreeze",
        "a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567", "F32"),
}
QUALIFIED_SOURCE_ROOT = BUSINESS / "ChatBubble-rineya-reasoning-posttraining-v1/experiments/model_quality_gpu_v1"
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
CONFIG_PINS = {"supa": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "sparse": "25ace0268b891881bf1d5a95d18611a74ce1896922107d10adc526257a1962e8",
    "transition": "a3c57d2869c3a44c0ea88f6c5538c6db47b5092986e098f076dddac5d6524ff4"}
ASSET_PINS = {
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"}

def header(path):
    with Path(path).open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("missing safetensors header")
        size = struct.unpack("<Q", raw)[0]
        if not 0 < size < 1024 * 1024:
            raise ValueError("unreasonable safetensors header size")
        value = json.loads(f.read(size))
    if not isinstance(value, dict):
        raise ValueError("header is not an object")
    return {k: v for k, v in value.items() if k != "__metadata__"}

def canonical_config(config):
    return {k: v for k, v in config.items() if k not in ("dtype", "torch_dtype")}

def audit_sources():
    results, errors, configs = {}, [], {}
    for name, (directory, expected, stored_dtype) in MODEL_SOURCES.items():
        try:
            weight = directory / "model.safetensors"
            actual = file_sha(weight)
            inventory = header(weight)
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            assets = {n: file_sha(directory / n) for n in SIDECARS}
            count = sum(__import__("math").prod(v["shape"]) for v in inventory.values())
            expected_geometry = {"num_hidden_layers": 28, "hidden_size": 1024, "intermediate_size": 3072,
                                 "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
                                 "vocab_size": 151936, "tie_word_embeddings": True}
            okay = (actual == expected and count == 596049920 and len(inventory) == 310
                and assets["config.json"] == CONFIG_PINS[name]
                and all(assets[k] == v for k, v in ASSET_PINS.items())
                and all(v.get("dtype") == stored_dtype for v in inventory.values())
                and all(config.get(k) == v for k, v in expected_geometry.items())
                and config.get("architectures") == ["Qwen3ForCausalLM"]
                and "abstain_logit_bias" not in config and "abstain_token_id" not in config)
            if not okay:
                errors.append("source identity/geometry differs:" + name)
            configs[name] = canonical_config(config)
            results[name] = {"path": str(directory), "expected_weight_sha256": expected,
                "actual_weight_sha256": actual, "stored_dtype": stored_dtype, "parameters": count,
                "tensor_count": len(inventory), "sidecars": assets, "geometry": expected_geometry,
                "config_eos": config.get("eos_token_id"), "identity_pass": okay,
                "tensor_payload_deserialized": False}
        except (OSError, ValueError, KeyError, TypeError) as e:
            errors.append(name + ":" + str(e))
    if len(configs) == 3 and not all(v == configs["supa"] for v in configs.values()):
        errors.append("canonical configs differ beyond storage dtype")
    if len(results) == 3:
        for sidecar in SIDECARS[1:]:
            if len({v["sidecars"][sidecar] for v in results.values()}) != 1:
                errors.append("tokenizer/generation sidecar differs:" + sidecar)
    dependencies = {}
    for name in ("config.py", "optimizer.py", "losses.py", "checkpoints.py", "native.py"):
        f = QUALIFIED_SOURCE_ROOT / name
        if f.exists():
            dependencies[str(f)] = file_sha(f)
        else:
            errors.append("inherited primitive missing:" + name)
    installed = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "jinja2"):
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    return {"created_utc": utc(), "status": "PASS_METADATA_ONLY" if not errors else "BLOCKED_SOURCE_IDENTITY",
        "models": results, "dependencies": dependencies, "installed_packages_offline_only": installed,
        "errors": errors, "native_qualified": False, "model_loads": 0, "optimizer_steps": 0,
        "teacher": "independent canonical program solvers; no external LLM teacher",
        "limit": "Metadata and header checks do not qualify a runtime, modified loss or model behavior."}
