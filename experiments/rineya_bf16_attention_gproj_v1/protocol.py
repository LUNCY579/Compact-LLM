"""Provenance, development-only input contracts and frozen experiment identity."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import torch

from . import BASE_COMMIT, EXPERIMENT_ID


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an earlier run or silently reinterpret its evidence.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def checkpoint_manifest(path: Path) -> dict:
    """Local-only checkpoint. No download, alias substitution or remote code."""
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("Supply the exact local Rineya checkpoint directory")
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("A safetensors checkpoint is required; pickle checkpoints are not loaded")
    if any(p.is_symlink() for p in files):
        raise ValueError("Materialize checkpoint symlinks before pinning")
    records = [{"path": str(p.relative_to(path)), "bytes": p.stat().st_size,
                "sha256": file_hash(p)} for p in files]
    return {"files": records, "sha256": digest_json(records)}


def parameter_hash(named_parameters: Iterable) -> str:
    """Streaming represented bytes; at most one 4 MiB chunk copied to CPU."""
    h = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        h.update(json.dumps([name, list(parameter.shape), str(parameter.dtype)]).encode())
        flat = parameter.detach().reshape(-1)
        count = max(1, (4 * 1024 * 1024) // parameter.element_size())
        for start in range(0, flat.numel(), count):
            raw = flat[start:start + count].contiguous().cpu().view(torch.uint8).numpy().tobytes()
            h.update(raw)
    return h.hexdigest()


def validate_config(config: dict) -> None:
    required = {"checkpoint", "checkpoint_sha256", "data", "data_sha256", "device", "backend",
                "trainable_names", "max_trainable_parameters", "max_tokens", "max_steps",
                "seeds", "learning_rate", "clip_norm", "min_free_cuda_gib",
                "audit_relative_tolerance", "audit_absolute_tolerance", "capture_call_indices"}
    if set(config) != required:
        raise ValueError(f"Config fields mismatch: missing={required-set(config)}, extra={set(config)-required}")
    for key in ("checkpoint_sha256", "data_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", config[key]):
            raise ValueError(f"Freeze a real SHA256 for {key}")
    if config["backend"] not in {"math", "flash", "efficient", "cudnn"}:
        raise ValueError("Backend must be explicit, never auto")
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("Only explicit cpu/cuda devices supported")
    for key, cap in (("max_tokens", 256), ("max_steps", 32), ("max_trainable_parameters", 20_000_000)):
        if type(config[key]) is not int or not 1 <= config[key] <= cap:
            raise ValueError(f"Invalid bounded budget: {key}")
    indices = config["capture_call_indices"]
    if not isinstance(indices, list) or not indices or len(set(indices)) != len(indices):
        raise ValueError("Freeze unique SDPA capture-call indices")
    if len(indices) > 8 or any(type(i) is not int or not 1 <= i <= 64 for i in indices):
        raise ValueError("Capture-call index budget exceeded")
    names = config["trainable_names"]
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("Provide unique exact trainable parameter names; no wildcard selection")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Invalid trainable name")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be a nonempty unique list")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Invalid seed")
    for key in ("learning_rate", "clip_norm", "min_free_cuda_gib",
                "audit_relative_tolerance", "audit_absolute_tolerance"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid positive finite value: {key}")


def load_rows(path: Path, max_tokens: int) -> list[dict]:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Development micro-pilot data budget is 4 MiB")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len(rows) > 128:
        raise ValueError("Supply 1--128 prospectively admitted rows")
    seen_ids, groups, sequences = set(), {}, {}
    for row in rows:
        required = {"id", "group_id", "split", "purpose", "source", "admitted_for", "input_ids", "labels"}
        if set(row) != required:
            raise ValueError("Each tokenized row needs the exact documented provenance schema")
        if row["split"] not in {"train", "dev"} or row["purpose"] not in {"acquisition", "retention"}:
            raise ValueError("Only TRAIN/DEV acquisition/retention rows; no sealed/confirmation inputs")
        if row["admitted_for"] != row["split"]:
            raise ValueError("Row lacks admission for this use")
        for field in ("id", "group_id", "source"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError("Missing provenance")
        if row["id"] in seen_ids:
            raise ValueError("Duplicate row id")
        seen_ids.add(row["id"])
        tokens, labels = row["input_ids"], row["labels"]
        if not isinstance(tokens, list) or not isinstance(labels, list):
            raise ValueError("Tokens and labels must be lists")
        if not 2 <= len(tokens) <= max_tokens or len(labels) != len(tokens):
            raise ValueError("No truncation: row length/labels exceed contract")
        if any(type(x) is not int or x < 0 for x in tokens):
            raise ValueError("Invalid input token")
        if any(type(x) is not int or (x < 0 and x != -100) for x in labels):
            raise ValueError("Invalid target token")
        if not any(x != -100 for x in labels[1:]):
            raise ValueError("No supervised shifted target")
        for table, key in ((groups, row["group_id"]), (sequences, digest_json(tokens))):
            if key in table and table[key] != row["split"]:
                raise ValueError("TRAIN/DEV group or exact-token overlap")
            table[key] = row["split"]
    for purpose in ("acquisition", "retention"):
        if not any(r["split"] == "dev" and r["purpose"] == purpose for r in rows):
            raise ValueError(f"Missing independent DEV {purpose} cohort")
    if not any(r["split"] == "train" for r in rows):
        raise ValueError("Missing TRAIN rows")
    return rows


def identity(config: dict, checkpoint: dict, runtime: dict) -> dict:
    code = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"experiment": EXPERIMENT_ID, "base_commit": BASE_COMMIT,
            "config_sha256": digest_json(config), "checkpoint_sha256": checkpoint["sha256"],
            "data_sha256": file_hash(Path(config["data"])), "runtime": runtime,
            "code_sha256": digest_json(code)}


def verify_qualification(report: dict, expected_identity: dict) -> None:
    if report.get("identity") != expected_identity:
        raise ValueError("Stale qualification: code/config/checkpoint/data/runtime changed")
    if report.get("status") != "NATIVE_NUMERICAL_QUALIFIED":
        raise ValueError("Native numerical qualification has not passed")
    if not report.get("capture_count", 0) or not report.get("backend_events"):
        raise ValueError("Qualification lacks captures or observed dispatch evidence")
    if report.get("cohorts") != ["acquisition", "retention"]:
        raise ValueError("Qualification must cover both development cohorts")
