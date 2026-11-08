"""Pinned Supa and Qwen3-4B safetensors loaders. Local-only, no auto fallback."""
from __future__ import annotations
import gc
import hashlib
from pathlib import Path
import torch
from experiments.architecture_benchmark_001.model import DenseReference
from .config import official_config
from .guards import PACKAGE, read_json, regular_file, sha256_file, git_blob, digest, canonical


def source_profile(role: str) -> dict:
    if role not in ("encoder", "decoder"):
        raise ValueError("unknown source role")
    return read_json(PACKAGE/"sources.json")[role]


def key_map(c) -> dict:
    result = {"model.embed_tokens.weight": "embed_tokens.weight", "model.norm.weight": "final_norm.weight"}
    for i in range(c.encoder_layers):
        a, b = f"model.layers.{i}.", f"encoder.{i}."
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            result[a+"self_attn."+name+".weight"] = b+name+".weight"
        result[a+"input_layernorm.weight"] = b+"input_norm.weight"
        result[a+"post_attention_layernorm.weight"] = b+"post_norm.weight"
        for name in ("gate_proj", "up_proj", "down_proj"):
            result[a+"mlp."+name+".weight"] = b+"mlp."+name+".weight"
    return result


def validate_config(raw: dict, role: str):
    c = getattr(official_config(), role)
    expected = dict(model_type="qwen3", architectures=["Qwen3ForCausalLM"], hidden_size=c.hidden_size,
                    intermediate_size=c.intermediate_size, num_hidden_layers=c.encoder_layers,
                    num_attention_heads=c.num_attention_heads, num_key_value_heads=c.num_key_value_heads,
                    head_dim=c.head_dim, rms_norm_eps=c.rms_norm_eps, rope_theta=c.rope_theta,
                    vocab_size=c.vocab_size, max_position_embeddings=c.max_position_embeddings,
                    tie_word_embeddings=True, hidden_act="silu", attention_bias=False,
                    attention_dropout=0.0, rope_scaling=None, use_sliding_window=False, sliding_window=None)
    if any(raw.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{role} native configuration mismatch")
    if raw.get("auto_map") or any("abstain" in k or "quantization" in k for k in raw):
        raise ValueError("custom code, biased readout and quantization are forbidden")
    return c


def audit_source(directory: Path, role: str) -> dict:
    directory = Path(directory).absolute()
    profile = source_profile(role)
    actual = {}
    for name, expected in profile["assets_sha256"].items():
        p = directory/name
        regular_file(p)
        if sha256_file(p) != expected:
            raise ValueError(f"{role} source hash mismatch: {name}")
        actual[name] = expected
    expected_weights = {k for k in actual if k.endswith(".safetensors")}
    present = {p.name for p in directory.iterdir() if p.suffix in (".safetensors", ".bin", ".pt", ".pth", ".gguf")}
    if present != expected_weights:
        raise ValueError("missing or unexpected source weight files")
    c = validate_config(read_json(directory/"config.json"), role)
    if role == "encoder":
        core = directory/"CORE.json"
        regular_file(core)
        if git_blob(core.read_bytes()) != profile["core_git_blob_sha1"] or read_json(core)["core_id"] != profile["id"]:
            raise ValueError("official Supa CORE metadata mismatch")
        actual["CORE.json"] = sha256_file(core)
    if role == "decoder":
        mapping = read_json(directory/"model.safetensors.index.json")["weight_map"]
        if set(mapping) != set(key_map(c)) or set(mapping.values()) != expected_weights:
            raise ValueError("shard index inventory/path mismatch")
    return dict(role=role, id=profile["id"], revision=profile["revision"], assets=actual,
                source_sha256=digest(actual), parameters=profile["parameters"])


def load_reference(directory: Path, role: str) -> DenseReference:
    """Rehash before reading; stream one tensor at a time into an FP32 meta scaffold."""
    audit_source(directory, role)
    from safetensors import safe_open
    c = getattr(official_config(), role)
    mapping = key_map(c)
    with torch.device("meta"):
        model = DenseReference(c)
    expected = model.state_dict()
    names = sorted(k for k in source_profile(role)["assets_sha256"] if k.endswith(".safetensors"))
    index = (read_json(Path(directory)/"model.safetensors.index.json")["weight_map"]
             if role == "decoder" else {k: names[0] for k in mapping})
    converted, seen = {}, set()
    for name in names:
        with safe_open(str(Path(directory)/name), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key not in mapping or key in seen or index[key] != name:
                    raise ValueError("unexpected, duplicate or incorrectly sharded tensor")
                t = f.get_tensor(key)
                if t.dtype != torch.bfloat16 or t.shape != expected[mapping[key]].shape or not torch.isfinite(t).all():
                    raise ValueError(f"source tensor dtype/shape/finiteness mismatch: {key}")
                converted[mapping[key]] = t.float()
                seen.add(key)
    if seen != set(mapping) or len(seen) != source_profile(role)["tensor_count"]:
        raise ValueError("incomplete source tensors")
    # Reject concurrent source-file replacement between the initial hash and reads.
    audit_source(directory, role)
    model.load_state_dict(converted, strict=True, assign=True)
    if sum(p.numel() for p in model.parameters()) != source_profile(role)["parameters"]:
        raise ValueError("source unique-parameter mismatch")
    model.eval().requires_grad_(False)
    del converted
    gc.collect()
    return model


def native_from_reference(reference: DenseReference, directory: Path):
    """Read-only native comparison view sharing already checked source tensors.

    Sharing avoids a redundant 16 GB decoder copy. Never train or mutate this view.
    """
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config.from_pretrained(str(directory), local_files_only=True)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    own = reference.state_dict()
    state = {key: own[value] for key, value in key_map(reference.config).items()}
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    model.eval().requires_grad_(False)
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError("native decoder head must remain tied")
    return model


def tokenizer_pair(encoder_dir: Path, decoder_dir: Path):
    from transformers import AutoTokenizer
    tok = [AutoTokenizer.from_pretrained(str(p), local_files_only=True, trust_remote_code=False,
                                         use_fast=True) for p in (encoder_dir, decoder_dir)]
    probes = ["A blue key opens the north door.", "42 + 17 =", "Line one.\nLine two.", "café 東京"]
    for text in probes:
        if tok[0].encode(text) != tok[1].encode(text):
            raise ValueError("native tokenizer token IDs differ")
    if tok[0].get_vocab() != tok[1].get_vocab():
        raise ValueError("native vocabularies differ")
    return tok[1]


def state_digest(model, *, backbone_only=False):
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        if backbone_only and name.startswith(("memory.", "reads.")):
            continue
        h.update(canonical([name, list(t.shape), str(t.dtype)]))
        data = t.detach().cpu().contiguous().view(torch.uint8).numpy()
        h.update(memoryview(data).cast("B"))
    return h.hexdigest()
