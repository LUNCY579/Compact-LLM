from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from consensus_lab.experiments.config import ExperimentConfig
from consensus_lab.io import canonical_json_sha256, read_json, sha256_file, write_json_atomic
from consensus_lab.paths import LabPaths, is_relative_to


RUNTIME_SOURCE_PATHS = (
    Path("scripts/_bootstrap.py"),
    Path("scripts/run_dev.py"),
    Path("src/consensus_lab/io.py"),
    Path("src/consensus_lab/paths.py"),
    Path("src/consensus_lab/run_lock.py"),
    Path("src/consensus_lab/run_provenance.py"),
    Path("src/consensus_lab/decoding/decoder.py"),
    Path("src/consensus_lab/evaluation/direct.py"),
    Path("src/consensus_lab/evaluation/sampling.py"),
    Path("src/consensus_lab/evaluation/scoring.py"),
    Path("src/consensus_lab/evaluation/summary.py"),
    Path("src/consensus_lab/experiments/config.py"),
    Path("src/consensus_lab/models/committee.py"),
    Path("src/consensus_lab/voting/rules.py"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry(path: Path, *, relative_to: Path | None = None, known_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Runtime provenance input is missing: {resolved}")
    relative = str(resolved.relative_to(relative_to.resolve())).replace("\\", "/") if relative_to else None
    return {
        "path": str(resolved),
        "relative_path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": known_sha256 or sha256_file(resolved),
    }


def _official_hash_cache(paths: LabPaths) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    before_path = paths.lab_root / "artifacts/manifests/official_sources_before.json"
    after_path = paths.lab_root / "artifacts/manifests/official_sources_after.json"
    before = read_json(before_path)
    after = read_json(after_path)
    before_hashes = {str(item["relative_path"]): str(item["sha256"]) for item in before.get("entries", [])}
    after_hashes = {str(item["relative_path"]): str(item["sha256"]) for item in after.get("entries", [])}
    if before_hashes != after_hashes:
        changed = sorted(path for path in set(before_hashes) | set(after_hashes) if before_hashes.get(path) != after_hashes.get(path))
        raise RuntimeError(f"Official read-only source manifests disagree: {changed}")
    cache = {str(item["relative_path"]): item for item in before.get("entries", [])}
    return cache, {
        "before_path": str(before_path.resolve()),
        "before_sha256": sha256_file(before_path),
        "after_path": str(after_path.resolve()),
        "after_sha256": sha256_file(after_path),
        "official_tree_manifest_sha256": before.get("manifest_sha256"),
        "verified_equal": True,
    }


def _official_entry(path: Path, *, paths: LabPaths, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = paths.assert_official_source(path)
    relative = str(source.relative_to(paths.official_root)).replace("\\", "/")
    cached = cache.get(relative)
    if cached is not None:
        if int(cached["bytes"]) != source.stat().st_size:
            raise RuntimeError(f"Official source size changed since the frozen manifest: {relative}")
        return _entry(source, relative_to=paths.official_root, known_sha256=str(cached["sha256"]))
    return _entry(source, relative_to=paths.official_root)


def _model_input_paths(config: ExperimentConfig) -> list[Path]:
    directories = [config.backbone, *(expert.adapter_path for expert in config.experts if expert.adapter_path)]
    files: set[Path] = set()
    for directory in directories:
        files.update(path for path in directory.iterdir() if path.is_file())
    return sorted(files)


def build_runtime_identity(
    *,
    paths: LabPaths,
    config_path: Path,
    config: ExperimentConfig,
    selected_datasets: list[str],
    limit: int | None,
    committee_size: int | None,
    requested_device: str,
    requested_precision: str,
    resolved_device: str,
    resolved_dtype: str,
) -> dict[str, Any]:
    cache, official_manifest = _official_hash_cache(paths)
    source_entries = [_entry(paths.lab_root / relative, relative_to=paths.lab_root) for relative in RUNTIME_SOURCE_PATHS]
    model_entries = [_official_entry(path, paths=paths, cache=cache) for path in _model_input_paths(config)]
    dataset_entries = {
        name: _official_entry(config.datasets[name], paths=paths, cache=cache)
        for name in selected_datasets
    }
    identity_payload = {
        "schema_version": "1.0",
        "config": {
            "path": str(config_path.resolve()),
            "file_sha256": sha256_file(config_path),
            "canonical_sha256": config.config_sha256,
        },
        "selection": {
            "datasets": selected_datasets,
            "limit": limit,
            "committee_size_filter": committee_size,
            "requested_device": requested_device,
            "requested_precision": requested_precision,
            "resolved_device": resolved_device,
            "resolved_dtype": resolved_dtype,
        },
        "datasets": dataset_entries,
        "model_inputs": model_entries,
        "runtime_sources": source_entries,
        "official_source_manifest": official_manifest,
    }
    return {
        "identity_sha256": canonical_json_sha256(identity_payload),
        "identity": identity_payload,
    }


def _nonempty_line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def begin_runtime_invocation(
    *,
    manifest_path: Path,
    output_path: Path,
    identity: dict[str, Any],
    runtime_controls: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    existing_rows = _nonempty_line_count(output_path)
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError(
                "Refusing to resume results under different source/config/model/runtime identity: "
                f"{output_path}"
            )
    elif existing_rows:
        raise RuntimeError(
            "Existing rows have no launch-time runtime manifest; refuse provenance backfill after the fact: "
            f"{output_path}"
        )
    else:
        manifest = {
            "schema_version": "1.0",
            "status": "RUNNING",
            "output_path": str(output_path.resolve()),
            "identity_sha256": identity["identity_sha256"],
            "identity": identity["identity"],
            "invocations": [],
        }
    invocations = list(manifest.get("invocations", []))
    if invocations and invocations[-1].get("status") == "STARTED":
        invocations[-1]["status"] = "INTERRUPTED_OR_UNCLEAN_EXIT"
        invocations[-1]["observed_at_next_launch_utc"] = utc_now()
        invocations[-1]["rows_observed_at_next_launch"] = existing_rows
    invocation_id = str(uuid.uuid4())
    invocations.append(
        {
            "invocation_id": invocation_id,
            "status": "STARTED",
            "started_at_utc": utc_now(),
            "pid": os.getpid(),
            "python": sys.executable,
            "rows_before": existing_rows,
            "runtime_controls": runtime_controls,
        }
    )
    manifest["invocations"] = invocations
    manifest["status"] = "RUNNING"
    manifest["updated_at_utc"] = utc_now()
    write_json_atomic(manifest_path, manifest)
    return invocation_id, manifest


def finish_runtime_invocation(
    *,
    manifest_path: Path,
    invocation_id: str,
    output_path: Path,
    expected_rows: int,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    rows = _nonempty_line_count(output_path)
    matched = False
    for invocation in manifest.get("invocations", []):
        if invocation.get("invocation_id") == invocation_id:
            invocation["status"] = "RETURNED_CLEANLY"
            invocation["finished_at_utc"] = utc_now()
            invocation["rows_after"] = rows
            invocation["rows_added"] = rows - int(invocation.get("rows_before", 0))
            matched = True
            break
    if not matched:
        raise RuntimeError(f"Runtime invocation is absent from its manifest: {invocation_id}")
    manifest["observed_rows"] = rows
    manifest["expected_rows"] = expected_rows
    manifest["status"] = "COMPLETE" if rows == expected_rows else "INCOMPLETE_RESUMABLE"
    manifest["updated_at_utc"] = utc_now()
    write_json_atomic(manifest_path, manifest)
    return manifest
