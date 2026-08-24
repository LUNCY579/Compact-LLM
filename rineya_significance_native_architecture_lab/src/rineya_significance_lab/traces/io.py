"""Atomic JSONL/NPZ trace persistence and structural validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..core.state import SignificanceState
from .schema import TraceEvent, TraceManifest


class TraceValidationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        try:
            Path(temporary).unlink()
        except OSError:
            pass
        raise


class TraceWriter:
    def __init__(self, root: str | Path, manifest: TraceManifest) -> None:
        manifest.validate()
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.events_path = self.root / "events.jsonl"
        if self.events_path.exists() and self.events_path.stat().st_size:
            raise FileExistsError(f"refusing to overwrite existing trace events: {self.events_path}")
        _atomic_json(self.root / "trace_manifest.json", manifest.to_dict())
        self._next_index = 0

    def write_event(self, event: TraceEvent) -> Path:
        event.validate()
        if event.case_id != self.manifest.case_id:
            raise TraceValidationError("event case_id differs from manifest")
        if event.event_index != self._next_index:
            raise TraceValidationError(
                f"event index {event.event_index} is not contiguous; expected {self._next_index}"
            )
        filename = f"event_{event.event_index:05d}.npz"
        destination = self.root / filename
        fd, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=self.root)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as handle:
                state = event.state_before
                np.savez_compressed(
                    handle,
                    entry_ids=state.entry_ids,
                    positions=state.positions,
                    token_ids=state.token_ids,
                    scores=state.scores,
                    counts=state.counts,
                    active=state.active,
                    score_delta=event.score_delta,
                    count_delta=event.count_delta,
                    recorded_indices=event.recorded_indices,
                    oracle_evidence=(
                        np.asarray(event.oracle_evidence, dtype=np.bool_)
                        if event.oracle_evidence is not None
                        else np.zeros((0,), dtype=np.bool_)
                    ),
                )
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        observed = event.observed_state()
        row = {
            "schema_version": self.manifest.schema_version,
            "case_id": event.case_id,
            "event_index": event.event_index,
            "event_id": event.event_id,
            "kind": event.kind,
            "boundary_position": event.boundary_position,
            "policy_id": event.policy_id,
            "npz": filename,
            "npz_sha256": _sha256(destination),
            "state_before_sha256": event.state_before.fingerprint(),
            "state_observed_sha256": observed.fingerprint(),
            "metadata": dict(event.metadata),
        }
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._next_index += 1
        return destination


class TraceReader:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "trace_manifest.json"
        if not manifest_path.exists():
            raise TraceValidationError(f"missing trace manifest: {manifest_path}")
        self.manifest = TraceManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        self.events_path = self.root / "events.jsonl"

    def rows(self) -> Iterator[dict[str, Any]]:
        if not self.events_path.exists():
            return
        with self.events_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TraceValidationError(f"invalid events JSON at line {line_number}") from exc
                if not isinstance(payload, dict):
                    raise TraceValidationError(f"event line {line_number} is not an object")
                yield payload

    def events(self, *, verify_checksums: bool = True) -> Iterator[TraceEvent]:
        for row in self.rows():
            npz_path = self.root / str(row["npz"])
            if not npz_path.is_file():
                raise TraceValidationError(f"missing event shard: {npz_path}")
            if verify_checksums and _sha256(npz_path) != str(row.get("npz_sha256", "")):
                raise TraceValidationError(f"event shard checksum mismatch: {npz_path.name}")
            with np.load(npz_path, allow_pickle=False) as arrays:
                state = SignificanceState(
                    entry_ids=arrays["entry_ids"],
                    positions=arrays["positions"],
                    token_ids=arrays["token_ids"],
                    scores=arrays["scores"],
                    counts=arrays["counts"],
                    active=arrays["active"],
                )
                oracle_array = np.asarray(arrays["oracle_evidence"], dtype=np.bool_)
                event = TraceEvent(
                    case_id=str(row["case_id"]),
                    event_index=int(row["event_index"]),
                    event_id=str(row["event_id"]),
                    kind=str(row["kind"]),
                    boundary_position=int(row["boundary_position"]),
                    state_before=state,
                    score_delta=np.asarray(arrays["score_delta"], dtype=np.float32),
                    count_delta=np.asarray(arrays["count_delta"], dtype=np.float32),
                    recorded_indices=np.asarray(arrays["recorded_indices"], dtype=np.int64),
                    policy_id=str(row["policy_id"]),
                    oracle_evidence=None if oracle_array.size == 0 else oracle_array,
                    metadata=dict(row.get("metadata", {})),
                )
            event.validate()
            if event.case_id != self.manifest.case_id:
                raise TraceValidationError("event case_id differs from manifest")
            if state.fingerprint() != str(row.get("state_before_sha256", "")):
                raise TraceValidationError("state-before fingerprint mismatch")
            if event.observed_state().fingerprint() != str(row.get("state_observed_sha256", "")):
                raise TraceValidationError("observed-state fingerprint mismatch")
            yield event


def validate_trace(root: str | Path) -> dict[str, Any]:
    reader = TraceReader(root)
    expected = 0
    event_count = 0
    for event in reader.events(verify_checksums=True):
        if event.event_index != expected:
            raise TraceValidationError(f"non-contiguous event index {event.event_index}; expected {expected}")
        expected += 1
        event_count += 1
    if event_count == 0:
        raise TraceValidationError("trace contains no events")
    return {
        "status": "PASS",
        "schema_version": reader.manifest.schema_version,
        "case_id": reader.manifest.case_id,
        "support": reader.manifest.support,
        "execution_mode": reader.manifest.execution_mode,
        "events": event_count,
    }
