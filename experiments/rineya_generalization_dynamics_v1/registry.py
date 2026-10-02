"""Immutable local snapshot registration; no training or network access."""
from __future__ import annotations

import json
import re
from pathlib import Path
from .common import digest, file_hash, normalized_prompt_hash, read, require

TOKENIZER_NAMES = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt", "tokenizer.model", "chat_template.jinja"}


def snapshot(directory: str, kind: str) -> dict:
    root = Path(directory).expanduser().resolve()
    require(root.is_dir(), f"Missing local {kind} export: {root}")
    files = {}
    for p in sorted(root.iterdir()):
        wanted = (p.name in TOKENIZER_NAMES if kind == "tokenizer" else
                  p.suffix == ".safetensors" or p.name in {"config.json", "generation_config.json", "adapter_config.json", "model.safetensors.index.json"})
        if wanted:
            require(p.is_file() and not p.is_symlink(), f"Unsafe/nonregular asset: {p}")
            require(p.stat().st_size > 0, f"Empty asset: {p}")
            files[p.name] = file_hash(p)
    if kind == "tokenizer":
        require("tokenizer_config.json" in files and bool(set(files) & {"tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt"}), "Incomplete tokenizer export")
        require("auto_map" not in read(root / "tokenizer_config.json"), "Remote/custom tokenizer code is not admitted")
    else:
        config_name = "adapter_config.json" if kind == "peft" else "config.json"
        require(config_name in files, f"Missing {config_name}")
        config = read(root / config_name)
        require(not set(config) & {"auto_map", "quantization_config", "abstain_logit_bias"}, "Custom/quantized/readout exports are not admitted")
        weights = {name for name in files if name.endswith(".safetensors")}
        require(bool(weights), "Real safetensors weights required; no bin/LFS/download fallback")
        if kind == "peft":
            require(config.get("peft_type") == "LORA", "Only standard LoRA adapters admitted")
            require(weights == {"adapter_model.safetensors"}, "Ambiguous adapter weight inventory")
        elif "model.safetensors.index.json" in files:
            index = read(root / "model.safetensors.index.json")
            required = set(index["weight_map"].values())
            require(required == weights, "Shards missing or extra weights present")
        else:
            require(weights == {"model.safetensors"}, "Ambiguous full-model inventory")
        # Reject Git-LFS text pointers masquerading as binary weights.
        for name in weights:
            with (root / name).open("rb") as f:
                require(not f.read(128).startswith(b"version https://git-lfs"), "Unmaterialized LFS weight pointer")
    return {"path": str(root), "kind": kind, "files": files, "content_sha256": digest(files)}


def model_shape(directory: str) -> str:
    config = read(Path(directory) / "config.json")
    for key in ("_name_or_path", "torch_dtype", "dtype", "transformers_version"):
        config.pop(key, None)
    return digest(config)


def training_inventory(path: str) -> dict:
    source = Path(path).expanduser().resolve()
    require(source.is_file(), "An explicit training prompt manifest is required")
    prompts, parents = set(), set()
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            require(isinstance(item.get("prompt"), str) and bool(item["prompt"].strip()), "Training manifest rows need a nonempty prompt")
            prompts.add(normalized_prompt_hash(item["prompt"]))
            if "parent_id" in item:
                require(isinstance(item["parent_id"], str), "Invalid training parent ID")
                parents.add(item["parent_id"])
    require(bool(prompts), "Empty training manifest")
    return {"path": str(source), "sha256": file_hash(source),
            "prompt_hashes": sorted(prompts), "parent_ids": sorted(parents)}


def audit_training_overlap(rows: list[dict], registry: dict) -> None:
    inv = registry["training"]
    require(not ({normalized_prompt_hash(r["prompt"]) for r in rows} & set(inv["prompt_hashes"])), "Evaluation prompt appears in training manifest")
    require(not ({r["parent_id"] for r in rows} & set(inv["parent_ids"])), "Evaluation parent appears in training manifest")


def freeze_registry(spec: dict) -> dict:
    require(isinstance(spec["trajectory_id"], str) and bool(spec["trajectory_id"]), "Missing trajectory ID")
    count = spec["expected_base_parameters"]
    require(type(count) is int and 500_000_000 <= count <= 700_000_000, "Only the exact 0.6B-class campaign is admitted")
    checkpoints = spec["checkpoints"]
    require(3 <= len(checkpoints) <= 20, "Register reference plus 2-19 snapshots")
    ids, steps = [c["id"] for c in checkpoints], [c["step"] for c in checkpoints]
    require(all(isinstance(c, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", c) for c in ids), "Unsafe checkpoint ID")
    require(len(set(ids)) == len(ids), "Duplicate checkpoint ID")
    require(all(type(s) is int and s >= 0 for s in steps) and steps[0] == 0 and steps == sorted(set(steps)), "Steps must start at zero and strictly increase")
    base = snapshot(spec["base_model"], "hf")
    tokenizer = snapshot(spec["tokenizer"], "tokenizer")
    shape = model_shape(base["path"])
    registered = []
    for index, c in enumerate(checkpoints):
        require(c["kind"] in {"base", "hf", "peft"}, "Unknown checkpoint kind")
        require((index == 0) == (c["kind"] == "base"), "Only step zero is the unmodified base")
        asset = base if c["kind"] == "base" else snapshot(c["path"], c["kind"])
        if c["kind"] == "hf":
            require(model_shape(asset["path"]) == shape, "Architecture/config differs within trajectory")
        registered.append({"id": c["id"], "step": c["step"], "kind": c["kind"], "asset": asset})
    kinds = {c["kind"] for c in registered[1:]}
    require(len(kinds) == 1, "Do not mix full and adapter checkpoints in one trajectory")
    if kinds == {"peft"}:
        configs = [read(Path(c["asset"]["path"]) / "adapter_config.json") for c in registered[1:]]
        require(all(digest(c) == digest(configs[0]) for c in configs), "Adapter configuration changed within trajectory")
    return {"trajectory_id": spec["trajectory_id"], "expected_base_parameters": count,
            "base": base, "tokenizer": tokenizer, "checkpoints": registered,
            "training": training_inventory(spec["training_manifest"]),
            "lineage_status": "caller-declared single run; content hashes verified, training origin not independently reconstructed"}


def verify_registry(registry: dict) -> None:
    assets = [registry["base"], registry["tokenizer"]] + [c["asset"] for c in registry["checkpoints"]]
    seen = set()
    for asset in assets:
        if (asset["path"], asset["kind"]) not in seen:
            require(snapshot(asset["path"], asset["kind"]) == asset, "Registered asset changed")
            seen.add((asset["path"], asset["kind"]))
    require(file_hash(Path(registry["training"]["path"])) == registry["training"]["sha256"], "Training inventory changed")
