"""Atomic, canonical and fail-closed PCSM plan persistence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from .hashing import mask_hash
from .plan import PCSMPlan, TensorPlan

_MANIFEST_NAME = "plan.json"
_TENSORS_NAME = "tensors.safetensors"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_mask_hash(mask: Tensor) -> str:
    """Use the project-wide shape + row-major binary mask encoding."""

    return mask_hash(mask)


def fingerprint_state_dict(state_dict: Mapping[str, Tensor]) -> str:
    """Canonical SHA-256 fingerprint over names, dtypes, shapes, and raw bytes."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, Tensor) or tensor.layout != torch.strided:
            raise TypeError(f"state entry {name!r} is not a dense tensor")
        value = tensor.detach().cpu().contiguous()
        header = {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        digest.update(_canonical_json_bytes(header))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _plan_payload(plan: PCSMPlan, tensor_sha256: str) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for index, entry in enumerate(plan.entries):
        prefix = f"entry.{index}"
        entries.append(
            {
                "name": entry.name,
                "role": entry.role,
                "shape": list(entry.shape),
                "numel": entry.numel,
                "k": entry.k,
                "mask_sha256": canonical_mask_hash(entry.support_mask),
                "peer_names": list(entry.peer_names),
                "stabilizer_lower": entry.stabilizer_lower,
                "stabilizer_upper": entry.stabilizer_upper,
                "tensor_keys": {
                    "support_mask": f"{prefix}.support_mask",
                    "stabilizer": f"{prefix}.stabilizer",
                    "attention_alphas": f"{prefix}.attention_alphas",
                },
            }
        )
    return {
        "schema_version": plan.schema_version,
        "base_fingerprint": plan.base_fingerprint,
        "descriptor_dimension": plan.descriptor_dimension,
        "metadata": dict(plan.metadata),
        "support_stats": plan.support_stats(),
        "tensor_file": _TENSORS_NAME,
        "tensor_file_sha256": tensor_sha256,
        "entries": entries,
    }


def save_plan(plan: PCSMPlan, directory: str | Path) -> Path:
    """Atomically publish an immutable JSON+safetensors plan directory.

    Existing destinations are refused rather than overwritten.  The directory
    rename is the only publication point, so readers never observe a half-plan.
    """

    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {destination}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        tensors: dict[str, Tensor] = {}
        for index, entry in enumerate(plan.entries):
            prefix = f"entry.{index}"
            tensors[f"{prefix}.support_mask"] = entry.support_mask.to(torch.uint8)
            tensors[f"{prefix}.stabilizer"] = entry.stabilizer
            tensors[f"{prefix}.attention_alphas"] = entry.attention_alphas
        tensor_path = staging / _TENSORS_NAME
        save_file(tensors, tensor_path)
        manifest = _plan_payload(plan, _file_sha256(tensor_path))
        manifest_path = staging / _MANIFEST_NAME
        with manifest_path.open("wb") as handle:
            handle.write(_canonical_json_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def load_plan(directory: str | Path, *, expected_base_fingerprint: str) -> PCSMPlan:
    """Load and fully verify a plan against an expected base fingerprint."""

    source = Path(directory)
    if not source.is_dir():
        raise FileNotFoundError(f"plan directory does not exist: {source}")
    actual_files = {path.name for path in source.iterdir()}
    if actual_files != {_MANIFEST_NAME, _TENSORS_NAME}:
        raise ValueError("plan directory has missing or unexpected files")
    try:
        payload: object = json.loads((source / _MANIFEST_NAME).read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("plan manifest is unreadable") from error
    manifest = _require_mapping(payload, "plan manifest")
    base_fingerprint = manifest.get("base_fingerprint")
    if base_fingerprint != expected_base_fingerprint:
        raise ValueError("base fingerprint mismatch; refusing to load plan")
    tensor_path = source / _TENSORS_NAME
    if manifest.get("tensor_file") != _TENSORS_NAME or manifest.get(
        "tensor_file_sha256"
    ) != _file_sha256(tensor_path):
        raise ValueError("safetensors artifact hash mismatch")
    try:
        tensors = load_file(tensor_path, device="cpu")
    except Exception as error:
        raise ValueError("safetensors artifact is unreadable") from error
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("plan entries must be a nonempty list")
    entries: list[TensorPlan] = []
    expected_keys: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        entry = _require_mapping(raw_entry, f"entry {index}")
        keys = _require_mapping(entry.get("tensor_keys"), f"entry {index} tensor_keys")
        try:
            mask_key = str(keys["support_mask"])
            stabilizer_key = str(keys["stabilizer"])
            alpha_key = str(keys["attention_alphas"])
            shape = tuple(int(item) for item in entry["shape"])
            peer_names = tuple(str(item) for item in entry["peer_names"])
            name = str(entry["name"])
            role = str(entry["role"])
            lower = float(entry["stabilizer_lower"])
            upper = float(entry["stabilizer_upper"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"entry {index} has invalid metadata") from error
        expected_keys.update((mask_key, stabilizer_key, alpha_key))
        if not expected_keys.issubset(tensors):
            raise ValueError(f"entry {index} references a missing tensor")
        plan_entry = TensorPlan(
            name=name,
            role=role,
            shape=shape,
            support_mask=tensors[mask_key],
            stabilizer=tensors[stabilizer_key],
            peer_names=peer_names,
            attention_alphas=tensors[alpha_key],
            stabilizer_lower=lower,
            stabilizer_upper=upper,
        )
        if entry.get("numel") != plan_entry.numel or entry.get("k") != plan_entry.k:
            raise ValueError(f"entry {index} support counts do not match tensors")
        if entry.get("mask_sha256") != canonical_mask_hash(plan_entry.support_mask):
            raise ValueError(f"entry {index} canonical mask hash mismatch")
        entries.append(plan_entry)
    if set(tensors) != expected_keys:
        raise ValueError("safetensors artifact contains unexpected tensors")
    metadata = _require_mapping(manifest.get("metadata", {}), "plan metadata")
    try:
        descriptor_dimension = int(manifest["descriptor_dimension"])
        schema_version = str(manifest["schema_version"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("plan header is invalid") from error
    plan = PCSMPlan(
        base_fingerprint=str(base_fingerprint),
        descriptor_dimension=descriptor_dimension,
        entries=tuple(entries),
        metadata=dict(metadata),
        schema_version=schema_version,
    )
    if manifest.get("support_stats") != plan.support_stats():
        raise ValueError("plan support statistics do not match tensor supports")
    return plan


__all__ = [
    "canonical_mask_hash",
    "fingerprint_state_dict",
    "load_plan",
    "save_plan",
]
