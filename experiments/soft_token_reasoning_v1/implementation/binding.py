"""Strict dataset/split and model-contract binding helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bind_dataset_manifest(manifest_path: Path, dataset_id: str, split_names: tuple[str, ...]) -> dict[str, Any]:
    """Bind split IDs to the repository-published immutable manifest.

    This function checks the manifest contract, but does not read sealed data
    and does not authorize a run.  A native runner must additionally hash the
    actual split file before launch and compare it to this result.
    """

    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be an object")
    if not dataset_id or not split_names:
        raise ValueError("dataset ID and split names are required")
    splits: dict[str, Any] = {}
    for split in split_names:
        manifest_key = split if split in manifest else f"{split}.jsonl"
        record = manifest.get(manifest_key)
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise ValueError(f"manifest has no hash for split {split!r}")
        if not isinstance(record.get("rows"), int) or record["rows"] <= 0:
            raise ValueError(f"manifest has no positive row count for split {split!r}")
        splits[split] = {
            "manifest_key": manifest_key,
            "rows": record["rows"],
            "bytes": record.get("bytes"),
            "sha256": record["sha256"],
        }
    return {
        "dataset_id": dataset_id,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "splits": splits,
        "sealed_access": False,
    }


def verify_split_file(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Verify a materialized split without changing it or reading other splits."""

    if not path.is_file():
        raise FileNotFoundError(path)
    observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if observed["sha256"] != expected["sha256"] or observed["bytes"] != expected.get("bytes"):
        raise ValueError(f"split binding mismatch for {path}")
    return {"path": path.as_posix(), **observed, "rows": expected["rows"]}
