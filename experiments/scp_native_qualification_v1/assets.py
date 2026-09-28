"""Byte/hash/header preflight only. No tensor materialization or pretrained imports."""
from __future__ import annotations

import math
from pathlib import Path
import struct
from .contract import (WEIGHTS, SIDECARS, SOURCE_CONFIG, canonical, digest, file_hash,
                       loads, require, safe_path)


def inventory(tail_only=False):
    shapes = {
        "self_attn.q_proj.weight": [2048, 1024], "self_attn.k_proj.weight": [1024, 1024],
        "self_attn.v_proj.weight": [1024, 1024], "self_attn.o_proj.weight": [1024, 2048],
        "self_attn.q_norm.weight": [128], "self_attn.k_norm.weight": [128],
        "mlp.gate_proj.weight": [3072, 1024], "mlp.up_proj.weight": [3072, 1024],
        "mlp.down_proj.weight": [1024, 3072], "input_layernorm.weight": [1024],
        "post_attention_layernorm.weight": [1024],
    }
    result = {f"model.layers.{i}.{suffix}": shape for i in range(8 if tail_only else 0, 28)
              for suffix, shape in shapes.items()}
    if not tail_only:
        result.update({"model.embed_tokens.weight": [151936, 1024], "model.norm.weight": [1024]})
    return result


def read_header(path):
    path = safe_path(path)
    with path.open("rb") as stream:
        size = path.stat().st_size
        prefix = stream.read(8)
        require(len(prefix) == 8, "truncated safetensors prefix")
        length, = struct.unpack("<Q", prefix)
        require(2 <= length <= 8 * 2**20 and 8 + length <= size, "invalid/bounded header length")
        header = loads(stream.read(length).decode("utf-8"))
    require(type(header) is dict, "header object required")
    metadata = header.pop("__metadata__", {})
    require(type(metadata) is dict and all(type(k) is str and type(v) is str for k, v in metadata.items()),
            "invalid safetensors metadata")
    spans = []
    for name, entry in header.items():
        require(type(entry) is dict and set(entry) == {"dtype", "shape", "data_offsets"}, "tensor header keys differ")
        require(entry["dtype"] in ("F16", "BF16", "F32"), "unqualified tensor dtype")
        shape, offsets = entry["shape"], entry["data_offsets"]
        require(type(shape) is list and shape and all(type(n) is int and n > 0 for n in shape), "invalid shape")
        require(type(offsets) is list and len(offsets) == 2 and all(type(n) is int and n >= 0 for n in offsets),
                "invalid tensor offsets")
        start, end = offsets
        require(end - start == math.prod(shape) * (4 if entry["dtype"] == "F32" else 2), "tensor byte count differs")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        require(start == cursor and end > start, "overlap/gap in tensor payload")
        cursor = end
    require(cursor == size - 8 - length, "trailing or missing tensor bytes")
    return header


def validate_inventory(header, *, partial=False, require_fp32=True):
    wanted = inventory(tail_only=partial)
    require(set(header) == set(wanted), "exact tensor-name inventory mismatch")
    require(all(entry["shape"] == wanted[name] for name, entry in header.items()), "tensor geometry mismatch")
    if require_fp32:
        require(all(entry["dtype"] == "F32" for entry in header.values()), "actual serialized FP32 required")
    return {"tensor_count": len(header), "parameters": sum(math.prod(s) for s in wanted.values()),
            "name_shape_sha256": digest(wanted)}


def validate_config_delta(source, subject):
    require(type(source) is dict and type(subject) is dict, "config object required")
    wanted = {"model_type": "qwen3", "hidden_size": 1024, "intermediate_size": 3072,
              "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
              "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": True,
              "use_sliding_window": False, "attention_dropout": 0.0}
    for key, expected in wanted.items():
        require(key in source and canonical(source[key]) == canonical(expected), "unqualified source geometry: " + key)
    require(not subject.get("auto_map") and not subject.get("quantization_config"), "remote/quantized config refused")
    require(subject == {**source, "torch_dtype": "float32"}, "only documented torch_dtype=float32 config exception allowed")


def audit_subject(source, partial, subject):
    source, partial, subject = map(safe_path, (source, partial, subject))
    files = {"source": source / "model.safetensors", "partial": partial,
             "subject": subject / "model.safetensors"}
    require(len({p.resolve() for p in files.values()}) == 3, "three distinct provenance assets required")
    before = {}
    for key, path in files.items():
        actual = file_hash(path)
        require(actual == WEIGHTS[key], "weight-file SHA-256 mismatch: " + key)
        before[key] = actual
    headers = {key: read_header(path) for key, path in files.items()}
    summaries = {key: validate_inventory(value, partial=key == "partial", require_fp32=key != "source")
                 for key, value in headers.items()}
    sidecars = {}
    for label, directory in (("source", source), ("subject", subject)):
        present = {p.name for p in directory.iterdir() if p.suffix in (".bin", ".pt", ".pth", ".safetensors", ".gguf")}
        require(present == {"model.safetensors"}, "unexpected weight files: " + label)
        sidecars[label] = {name: file_hash(directory / name) for name in SIDECARS}
        require(sidecars[label] == SIDECARS, "tokenizer/generation sidecar mismatch: " + label)
    require(file_hash(source / "config.json") == SOURCE_CONFIG, "source config mismatch")
    source_config = loads((source / "config.json").read_text("utf-8"))
    subject_config = loads(safe_path(subject / "config.json").read_text("utf-8"))
    validate_config_delta(source_config, subject_config)
    config_sha = file_hash(subject / "config.json")
    require({key: file_hash(path) for key, path in files.items()} == before, "assets changed during audit")
    return {"schema": "SCP_NQ_ASSET_AUDIT_V1", "classification": "ASSET_PREFLIGHT_ONLY",
            "weight_sha256": before, "sidecars": sidecars, "subject_config_sha256": config_sha,
            "inventory": summaries, "native_execution_permitted": False, "permit_consumed": False,
            "not_proven": ["finite materialized tensors", "tied loaded storage", "masked-delta algebra recomputation",
                           "native tokenizer/template parity", "cache/window parity", "host resource fit"]}
