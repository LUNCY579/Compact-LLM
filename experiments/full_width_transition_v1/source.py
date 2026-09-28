"""Strict local-only import of the locked official source; no fallback or download."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import torch
from .guards import audit_source, require_runtime, source_lock
from experiments.architecture_benchmark_001.budget import presets
from experiments.architecture_benchmark_001.model import DenseReference


def state_digest(model, *, backbone_only=False):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if backbone_only and name.startswith(("memory.", "reads.")):
            continue
        h.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
        h.update(memoryview(raw).cast("B"))
    return h.hexdigest()


def key_map(c):
    result = {"model.embed_tokens.weight": "embed_tokens.weight", "model.norm.weight": "final_norm.weight"}
    for i in range(c.total_layers):
        a, b = f"model.layers.{i}.", f"encoder.{i}."
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            result[a + "self_attn." + name + ".weight"] = b + name + ".weight"
        result[a + "input_layernorm.weight"] = b + "input_norm.weight"
        result[a + "post_attention_layernorm.weight"] = b + "post_norm.weight"
        for name in ("gate_proj", "up_proj", "down_proj"):
            result[a + "mlp." + name + ".weight"] = b + "mlp." + name + ".weight"
    return result


def load_source(directory: Path, *, include_native=False):
    audit_source(directory)
    require_runtime()
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    c = presets()["B28_1024_official_shape"]
    lock = source_lock()
    state = load_file(str(Path(directory) / "model.safetensors"), device="cpu")
    mapping = key_map(c)
    if set(state) != set(mapping) or len(state) != lock["tensor_count"]:
        raise ValueError("official tensor inventory mismatch")
    if sum(t.numel() for t in state.values()) != lock["parameters"]:
        raise ValueError("official parameter count mismatch")
    with torch.device("meta"):
        reference = DenseReference(c)
    expected = reference.state_dict()
    for name, value in state.items():
        if value.dtype != torch.bfloat16 or value.shape != expected[mapping[name]].shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid official tensor: {name}")
    converted = {mapping[k]: v.float() for k, v in state.items()}
    reference.load_state_dict(converted, strict=True, assign=True)
    reference.eval()
    native = None
    if include_native:
        config = Qwen3Config.from_pretrained(str(directory), local_files_only=True)
        config._attn_implementation = "eager"
        with torch.device("meta"):
            native = Qwen3ForCausalLM(config)
        ns = {k: converted[v].detach().clone() for k, v in mapping.items()}
        ns["lm_head.weight"] = ns["model.embed_tokens.weight"]
        native.load_state_dict(ns, strict=True, assign=True)
        native.tie_weights()
        native.eval()
        if native.lm_head.weight.data_ptr() != native.model.embed_tokens.weight.data_ptr():
            raise ValueError("native output head is not tied")
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    del state, converted
    gc.collect()
    return reference, native, tokenizer
