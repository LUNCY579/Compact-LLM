"""Verify a materialized HF candidate before using the existing DenseReference loader.

Avoid a redundant FP32 research export on a disk-constrained host. Conversion is
the same import_qwen_state -> float32 path used by write_research_format and every
frozen-source probe; this changes storage layout, not the model or decoding.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SIDECARS = ("config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_hf_candidate(folder: Path, source: Path) -> dict:
    folder, source = Path(folder).resolve(), Path(source).resolve()
    receipt_path = folder / "MATERIALIZATION.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("record") != "CHECKPOINT_MATERIALIZATION" or receipt.get("layout") != "huggingface":
        raise ValueError("HF retention requires an explicit HF materialization receipt")
    source_sha = file_hash(source / "model.safetensors")
    if receipt.get("source_weight_sha256") != source_sha:
        raise ValueError("HF candidate was materialized from a different source")
    weight_sha = file_hash(folder / "model.safetensors")
    if receipt.get("out_weight_sha256") != weight_sha:
        raise ValueError("HF candidate weight bytes differ from the receipt")
    sidecars = {}
    for name in SIDECARS:
        actual = file_hash(folder / name)
        if actual != file_hash(source / name):
            raise ValueError(f"HF candidate source sidecar differs: {name}")
        sidecars[name] = actual
    return {"layout": "huggingface", "directory": str(folder),
            "source_weight_sha256": source_sha, "weight_sha256": weight_sha,
            "materialization_sha256": file_hash(receipt_path),
            "checkpoint_manifest_sha256": receipt.get("checkpoint_manifest_sha256"),
            "sidecars_sha256": sidecars,
            "evaluation_path": "run._load: import_qwen_state then float32 DenseReference",
            "redundant_research_export_created": False}


def load_hf_candidate(folder: Path, source: Path, device: str):
    identity = verify_hf_candidate(folder, source)
    from experiments.architecture_benchmark_001 import run
    model, _, _ = run._load(Path(folder), device)
    return model, identity
