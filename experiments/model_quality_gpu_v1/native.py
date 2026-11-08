"""Independent local-only native Qwen lane, not a borrowed RAB qualification.

No imports from CLI-driven historical experiments; no manual meta construction.
Only exact dense Supa source is active. Opallo continuation/4B/quantization are deferred.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import platform
import sys
import os
import torch
from .integrity import PACKAGE, canonical, digest, file_hash, read_json

TOKEN_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
PINS = {"torch": "2.10.0", "transformers": "4.57.1", "safetensors": "0.7.0"}


def lock() -> dict:
    return read_json(PACKAGE / "source_lock.json")


def audit_assets(source: Path, *, tokenizer_only=False) -> dict:
    expected = lock()["assets"]
    names = TOKEN_FILES if tokenizer_only else tuple(expected)
    actual = {name: file_hash(source / name) for name in names}
    if any(actual[name] != expected[name] for name in names):
        raise ValueError("source asset SHA-256 mismatch; no substitution permitted")
    if not tokenizer_only:
        present = {p.name for p in Path(source).iterdir()
                   if p.suffix in (".bin", ".pt", ".pth", ".safetensors", ".gguf")}
        if present != {"model.safetensors"}:
            raise ValueError("unexpected/missing source weight files")
        c = read_json(source / "config.json")
        wanted = {"model_type": "qwen3", "hidden_size": 1024, "intermediate_size": 3072,
                  "num_hidden_layers": 28, "num_attention_heads": 16,
                  "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
                  "tie_word_embeddings": True, "use_sliding_window": False,
                  "attention_dropout": 0.0}
        if any(c.get(k) != v for k, v in wanted.items()) or c.get("auto_map") or c.get("quantization_config"):
            raise ValueError("source is not the registered dense Qwen geometry")
    return {"source_id": lock()["source_id"], "assets": actual, "sha256": digest(actual)}


def runtime(*, require=False) -> dict:
    libs = {}
    for name in (*PINS, "tokenizers", "numpy", "psutil"):
        try:
            libs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libs[name] = None
    if require and any((libs[k] or "").split("+")[0] != v for k, v in PINS.items()):
        raise ValueError(f"dedicated native runtime pins unavailable: {libs}")
    gpu = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {"name": p.name, "total_memory": p.total_memory,
               "capability": [p.major, p.minor]}
    return {"python": sys.version, "platform": platform.platform(), "libraries": libs,
            "cuda_build": torch.version.cuda, "gpu0": gpu, "arithmetic": "fp32_eager_no_tf32",
            "optimizer": "cpu_moments_fp32", "threads": 1,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest()}


def configure_runtime(seed: int) -> None:
    import os
    import random
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_tokenizer(source: Path):
    runtime(require=True)
    audit_assets(source, tokenizer_only=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id != 151645:
        raise ValueError("wrong native tokenizer/EOS")
    return tokenizer


def load_model(source: Path, device: str):
    runtime(require=True)
    audit_assets(source)
    from transformers import Qwen3ForCausalLM
    model = Qwen3ForCausalLM.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float32, attn_implementation="eager",
        device_map=None)
    audit_assets(source)
    if sum(p.numel() for p in model.parameters()) != lock()["parameters"]:
        raise ValueError("loaded model parameter inventory mismatch")
    for _, tensor in (*model.named_parameters(), *model.named_buffers()):
        if tensor.is_meta or not torch.isfinite(tensor).all():
            raise ValueError("unmaterialized or nonfinite native tensor/buffer")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError("native vocabulary weight tying lost")
    model.config.use_cache = False
    return model.to(device).eval()


def select_tail(model, tail: int) -> list[tuple[str, torch.nn.Parameter]]:
    n = len(model.model.layers)
    if type(tail) is not int or not 1 <= tail <= n:
        raise ValueError("invalid trainable tail")
    prefixes = tuple(f"model.layers.{i}." for i in range(n - tail, n))
    result = []
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(prefixes))
        if p.requires_grad:
            result.append((name, p))
    if not result:
        raise ValueError("empty trainable inventory")
    return result


def hidden_states(model, ids):
    return model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state


def parameter_digest(model, *, frozen_only=False) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if frozen_only and p.requires_grad:
            continue
        h.update(canonical([name, list(p.shape), str(p.dtype)]))
        flat = p.detach().reshape(-1)
        for start in range(0, flat.numel(), 262144):
            part = flat[start:start + 262144].cpu().contiguous().view(torch.uint8).numpy()
            h.update(memoryview(part))
    return h.hexdigest()
