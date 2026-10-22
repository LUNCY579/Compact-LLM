from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import atomic_write_json, sha256_file


class LeakageError(RuntimeError):
    pass


FORBIDDEN_FRAGMENTS = (
    "rineya_invisible_limit_lab_v1/data/test2_answer_key.jsonl",
    "rineya_invisible_limit_lab_v1/data/test2",
    "rineya_invisible_limit_lab_v1/outputs/stage6",
)


def normalized_path(path: Path) -> str:
    return str(path.expanduser().resolve()).replace("\\", "/").lower()


def guard_non_sealed_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    value = normalized_path(resolved)
    if any(fragment in value for fragment in FORBIDDEN_FRAGMENTS):
        raise LeakageError(f"Invisible-Limit TEST2 firewall rejected path: {resolved}")
    if "/10_sealed_eval/private/" in value or "/local_data/sealed" in value:
        raise LeakageError(
            f"sealed data requires a frozen authorization manifest: {resolved}"
        )
    return resolved


def authorize_sealed_path(
    path: Path, authorization_manifest: Path, candidate_hash: str
) -> Path:
    resolved = path.expanduser().resolve()
    value = normalized_path(resolved)
    if any(fragment in value for fragment in FORBIDDEN_FRAGMENTS):
        raise LeakageError(f"Invisible-Limit TEST2 is never authorized: {resolved}")
    manifest_path = authorization_manifest.expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "FROZEN_FOR_SINGLE_SEALED_RUN":
        raise LeakageError("sealed authorization status is not frozen")
    if manifest.get("dataset_sha256") != sha256_file(resolved):
        raise LeakageError("sealed dataset hash differs from authorization manifest")
    if manifest.get("candidate_hash") != candidate_hash:
        raise LeakageError("candidate hash differs from authorization manifest")
    if manifest.get("access_started_at_utc") or manifest.get("consumed_at_utc"):
        raise LeakageError("sealed authorization has already been consumed")
    return resolved


def begin_sealed_access(
    authorization_manifest: Path,
    *,
    dataset_sha256: str,
    candidate_hash: str,
    output_directory: Path,
) -> None:
    manifest_path = authorization_manifest.expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("access_started_at_utc") or manifest.get("consumed_at_utc"):
        raise LeakageError("sealed authorization has already been consumed")
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise LeakageError("cannot consume authorization for a different dataset")
    if manifest.get("candidate_hash") != candidate_hash:
        raise LeakageError("cannot consume authorization for a different candidate")
    manifest["access_started_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["access_pid"] = os.getpid()
    manifest["output_directory"] = str(output_directory.resolve())
    atomic_write_json(manifest_path, manifest)


def append_access_ledger(
    ledger_path: Path,
    *,
    dataset_path: Path,
    dataset_sha256: str,
    mode: str,
    model_hash: str,
    details: dict[str, Any] | None = None,
) -> None:
    event = {
        "schema_version": "1.0",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "mode": mode,
        "model_hash": model_hash,
        "details": details or {},
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
