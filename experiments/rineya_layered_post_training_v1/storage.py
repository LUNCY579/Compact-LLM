"""Immutable artifacts and local-model/resource qualification."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .config import canonical, digest, require
from .runtime import Runtime
from .soup import state_hash


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        out.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def save_snapshot(directory, runtime, state, parents=()):
    directory.mkdir(parents=True, exist_ok=False)
    save_file({k: v.contiguous() for k, v in state.items()}, str(directory / "weights.safetensors"))
    metadata = {"schema": 1, "identity_hash": runtime.identity_hash, "identity": runtime.identity, "config_hash": runtime.config.fingerprint, "state_hash": state_hash(state), "weights_sha256": file_hash(directory / "weights.safetensors"), "parents": list(parents), "representation": "dense_trainable_subset", "resume_optimizer": False}
    write_json(directory / "metadata.json", metadata)
    return metadata


def load_snapshot(directory, runtime):
    metadata = json.loads((directory / "metadata.json").read_text())
    require(metadata["schema"] == 1 and metadata["identity_hash"] == runtime.identity_hash and digest(metadata["identity"]) == runtime.identity_hash, "checkpoint lineage/base/tokenizer mismatch")
    require(metadata["config_hash"] == runtime.config.fingerprint, "checkpoint configuration mismatch")
    require(metadata["weights_sha256"] == file_hash(directory / "weights.safetensors"), "checkpoint file tampering")
    state = load_file(str(directory / "weights.safetensors"))
    require(metadata["state_hash"] == state_hash(state), "checkpoint state hash mismatch")
    return state


def resource_report(config, directory, *, worker=False):
    import psutil
    host = psutil.virtual_memory().available / 2**30
    disk = shutil.disk_usage(directory).free / 2**30
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.mem_get_info()[0] / 2**30 if cuda else 0.
    limits = (config.host_worker_gib, config.gpu_worker_gib) if worker else (config.host_pre_gib, config.gpu_pre_gib)
    blockers = []
    if host < limits[0]:
        blockers.append("insufficient host RAM")
    if not cuda or gpu < limits[1]:
        blockers.append("CUDA unavailable or insufficient free VRAM")
    if disk < config.disk_gib:
        blockers.append("insufficient free disk")
    return {"host_free_gib": host, "gpu_free_gib": gpu, "disk_free_gib": disk, "cuda": cuda, "blockers": blockers}


def native_preflight(model_dir, config, output_parent):
    report = resource_report(config, output_parent)
    if not importlib.util.find_spec("transformers"):
        report["blockers"].append("transformers not installed")
    if not model_dir.is_dir():
        report["blockers"].append("local checkpoint directory missing")
    else:
        if not (model_dir / "config.json").is_file():
            report["blockers"].append("model config missing")
        if not list(model_dir.glob("*.safetensors")):
            report["blockers"].append("local safetensors weights missing")
    report["status"] = "PREFLIGHT_READY_NOT_QUALIFIED" if not report["blockers"] else "BLOCKED_NATIVE_UNQUALIFIED"
    return report


def native_runtime(model_dir, config, output_parent, *, allow_model_load=False):
    require(allow_model_load, "explicit --allow-model-load required")
    preflight = native_preflight(model_dir, config, output_parent)
    require(not preflight["blockers"], "; ".join(preflight["blockers"]))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    files = sorted(p for p in model_dir.rglob("*") if p.is_file() and ".git" not in p.parts)
    require(not any(p.is_symlink() for p in files), "use a materialized local snapshot, not mutable symlinks")
    hashes = {str(p.relative_to(model_dir)): file_hash(p) for p in files}
    config_json = json.loads((model_dir / "config.json").read_text())
    require(config_json.get("model_type") == "qwen3", "only native Qwen3-format 0.6B checkpoints are qualified by this loader; custom Rineya needs its own audited adapter")
    require(config.max_context_tokens <= config_json.get("max_position_embeddings", 0), "context exceeds checkpoint capacity")
    require(torch.cuda.is_bf16_supported(), "BF16 CUDA support required")
    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    require(tokenizer.eos_token_id is not None, "explicit EOS required")
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    after_files = sorted(p for p in model_dir.rglob("*") if p.is_file() and ".git" not in p.parts)
    require(hashes == {str(p.relative_to(model_dir)): file_hash(p) for p in after_files}, "checkpoint changed during load")
    total = sum(p.numel() for p in model.parameters())
    require(500_000_000 <= total <= 700_000_000, "this experiment targets the 0.6B family only")
    layers = model.config.num_hidden_layers
    require(config.train_last_layers <= layers, "trainable layer request exceeds depth")
    names = [n for n, _ in model.named_parameters() if any(n == f"model.layers.{i}.self_attn.{proj}.weight" for i in range(layers - config.train_last_layers, layers) for proj in ("q_proj", "v_proj", "o_proj"))]
    params = dict(model.named_parameters())
    require(len(names) == config.train_last_layers * 3 and sum(params[n].numel() for n in names) / total <= config.trainable_fraction_cap, "unexpected projection names or capacity excess")
    runtime = Runtime(model, tokenizer, config, names, {"kind": "native_qwen3_0.6b", "model_files": hashes, "torch": torch.__version__, "transformers": transformers.__version__, "total_parameters": total}, device="cuda", native=True)
    require(not resource_report(config, output_parent, worker=True)["blockers"], "post-load resource guard failed")
    return runtime
