"""Source identity and create-only external custody; no native model entry point."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUSINESS = Path("C:/Users/clclc/BusinessProject").resolve()
ARTIFACT_ROOT = BUSINESS / "ChatBubble-agent-prompts-20261002/run-artifacts/titan-quality-repair-v1"
CORE = BUSINESS / "Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Transition-1.0.0-devfreeze"
WEIGHT_SHA256 = "a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567"
WEIGHT_BYTES = 2384235008
PACKAGES = ("numpy", "tokenizers", "psutil", "torch", "transformers", "peft", "accelerate", "safetensors")
SIDECARS = ("CORE.json", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def external(path, *, create=False):
    path = Path(path).resolve()
    if (not path.is_relative_to(ARTIFACT_ROOT.resolve()) or path.is_relative_to(REPO)
            or ".codex" in [p.lower() for p in path.parts]
            or any((p / ".git").exists() for p in (path, *path.parents))):
        raise ValueError("STOP_OUTPUT_OUTSIDE_CANONICAL_QUALITY_ARTIFACT_ROOT")
    if create:
        path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    path = Path(path)
    external(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_pinned(path, expected):
    data = Path(path).read_bytes()
    if sha_bytes(data) != expected:
        raise ValueError(f"STOP_SOURCE_DIGEST_MISMATCH:{Path(path).name}")
    return data


def source_identity(core=CORE):
    """Metadata and sidecars only. Weight bytes are NOT read or rehashed offline."""
    core = Path(core)
    manifest_bytes = (core / "CORE.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    config_bytes = (core / "config.json").read_bytes()
    config = json.loads(config_bytes)
    checks = manifest.get("checks", {})
    if (checks.get("weight_sha256") != WEIGHT_SHA256 or checks.get("weight_bytes") != WEIGHT_BYTES
            or checks.get("architecture") != "Qwen3ForCausalLM"
            or checks.get("dtype_counts") != {"F32": 310}
            or config.get("architectures") != ["Qwen3ForCausalLM"]
            or config.get("torch_dtype", config.get("dtype")) != "float32"
            or float(config.get("abstain_logit_bias", 0)) != 0):
        raise ValueError("STOP_TITAN_SOURCE_IDENTITY")
    weight = core / "model.safetensors"
    if weight.stat().st_size != WEIGHT_BYTES:
        raise ValueError("STOP_TITAN_WEIGHT_LENGTH")
    pins = {name: sha_bytes((core / name).read_bytes()) for name in SIDECARS}
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"core_id": manifest["core_id"], "core_directory": str(core),
        "weight_path": str(weight), "declared_weight_sha256": WEIGHT_SHA256,
        "declared_weight_bytes": WEIGHT_BYTES, "observed_weight_bytes": weight.stat().st_size,
        "fresh_weight_digest_verified": False, "checkpoint_weight_reads": 0,
        "architecture": "Qwen3ForCausalLM", "storage_dtype": "float32", "custom_abstention": False,
        "sidecar_sha256": pins, "runtime_packages": packages,
        "release_qualified": False, "native_execution_authorized": False,
        "metadata_verified": True, "native_source_verification_required": True}


def source_manifest():
    return {p.name: sha_bytes(p.read_bytes()) for p in sorted(HERE.iterdir()) if p.is_file()}


def refuse_native(stage="qualification"):
    """The offline implementation has no model load path; permits cannot be imported."""
    raise RuntimeError(f"STOP_NATIVE_{stage.upper()}_REQUIRES_SEPARATE_SLOT_AND_SOURCE_BOUND_PROPOSAL")


def git_identity():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, timeout=10).strip()
    return {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
            "git_common_dir": git("rev-parse", "--path-format=absolute", "--git-common-dir")}
