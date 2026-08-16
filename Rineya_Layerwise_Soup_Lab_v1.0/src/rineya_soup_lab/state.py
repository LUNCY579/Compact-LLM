from __future__ import annotations

import os
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import RineyaSoupError
from .utils import read_json, utc_timestamp, write_json


@dataclass
class RunState:
    """Durable state for an experiment run.

    The state file is deliberately small. Large evidence remains in its own JSON/JSONL
    artifacts; this registry stores only paths, statuses, and selection metadata.
    """

    path: Path
    payload: dict[str, Any]

    @classmethod
    def open(cls, path: Path, *, run_id: str, config_digest: str) -> "RunState":
        if path.exists():
            payload = read_json(path)
            if payload.get("run_id") != run_id:
                raise RineyaSoupError(f"Run-state identity mismatch at {path}")
            if payload.get("config_digest") != config_digest:
                raise RineyaSoupError(
                    "Resolved configuration changed after this run started. Create a new run "
                    "directory, or restore the original configuration."
                )
        else:
            payload = {
                "schema_version": "1.0",
                "run_id": run_id,
                "config_digest": config_digest,
                "created_at": utc_timestamp(),
                "updated_at": utc_timestamp(),
                "stages": {},
                "candidates": {},
                "aliases": {},
                "events": [],
                "selection": None,
            }
            write_json(path, payload)
        return cls(path=path, payload=payload)

    def save(self) -> None:
        self.payload["updated_at"] = utc_timestamp()
        write_json(self.path, self.payload)

    def event(self, kind: str, **details: Any) -> None:
        events = self.payload.setdefault("events", [])
        events.append({"at": utc_timestamp(), "kind": kind, **details})
        # Keep the state useful without turning it into a giant trace file.
        if len(events) > 2000:
            self.payload["events"] = events[-2000:]
        self.save()

    def stage_status(self, stage: str) -> str | None:
        value = (self.payload.get("stages") or {}).get(stage)
        return str(value.get("status")) if isinstance(value, Mapping) else None

    def begin_stage(self, stage: str) -> None:
        stages = self.payload.setdefault("stages", {})
        previous = stages.get(stage) or {}
        stages[stage] = {
            **previous,
            "status": "running",
            "started_at": previous.get("started_at") or utc_timestamp(),
            "last_started_at": utc_timestamp(),
            "error": None,
        }
        self.save()

    def complete_stage(self, stage: str, **outputs: Any) -> None:
        stages = self.payload.setdefault("stages", {})
        previous = stages.get(stage) or {}
        stages[stage] = {
            **previous,
            "status": "complete",
            "completed_at": utc_timestamp(),
            "outputs": outputs,
            "error": None,
        }
        self.save()

    def fail_stage(self, stage: str, error: BaseException) -> None:
        stages = self.payload.setdefault("stages", {})
        previous = stages.get(stage) or {}
        stages[stage] = {
            **previous,
            "status": "failed",
            "failed_at": utc_timestamp(),
            "error": f"{type(error).__name__}: {error}",
        }
        self.save()

    def register_candidate(self, candidate_id: str, record: Mapping[str, Any]) -> None:
        candidates = self.payload.setdefault("candidates", {})
        existing = candidates.get(candidate_id)
        incoming = dict(record)
        if existing:
            # Labels/metadata can differ for the same numerical recipe. Numerical fields cannot.
            def identity(recipe: Mapping[str, Any] | None) -> dict[str, Any]:
                recipe = recipe or {}
                return {
                    "schema_version": recipe.get("schema_version", "1.0"),
                    "kind": recipe.get("kind"),
                    "endpoint": recipe.get("endpoint"),
                    "global_alpha": recipe.get("global_alpha"),
                    "layer_alphas": recipe.get("layer_alphas") or {},
                    "group_alphas": recipe.get("group_alphas") or {},
                }
            if identity(existing.get("recipe")) != identity(incoming.get("recipe")):
                raise RineyaSoupError(f"Candidate ID collision with a different numerical recipe: {candidate_id}")
            merged = dict(existing)
            labels = set(existing.get("labels") or [existing.get("label")])
            labels.add(incoming.get("label"))
            merged.update(incoming)
            merged["recipe"] = existing.get("recipe") or incoming.get("recipe")
            merged["label"] = existing.get("label") or incoming.get("label")
            merged["labels"] = sorted(str(item) for item in labels if item)
            merged["retained"] = bool(existing.get("retained") or incoming.get("retained"))
            existing_evaluations = dict(existing.get("evaluations") or {})
            existing_evaluations.update(incoming.get("evaluations") or {})
            merged["evaluations"] = existing_evaluations
            if existing.get("model_path") and not incoming.get("model_path"):
                merged["model_path"] = existing.get("model_path")
            merged["materialized"] = bool(
                existing.get("materialized") or incoming.get("materialized")
            )
            incoming = merged
        candidates[candidate_id] = incoming
        self.save()

    def update_candidate(self, candidate_id: str, **updates: Any) -> None:
        candidates = self.payload.setdefault("candidates", {})
        if candidate_id not in candidates:
            raise KeyError(f"Unknown candidate: {candidate_id}")
        candidates[candidate_id].update(updates)
        self.save()

    def candidate(self, candidate_id: str) -> dict[str, Any]:
        return dict((self.payload.get("candidates") or {})[candidate_id])

    def set_alias(self, alias: str, candidate_id: str) -> None:
        if candidate_id not in (self.payload.get("candidates") or {}):
            raise KeyError(f"Cannot alias an unknown candidate: {candidate_id}")
        self.payload.setdefault("aliases", {})[alias] = candidate_id
        self.save()

    def resolve_alias(self, alias_or_id: str) -> str:
        aliases = self.payload.get("aliases") or {}
        return str(aliases.get(alias_or_id, alias_or_id))


class RunLock(AbstractContextManager["RunLock"]):
    """Cross-platform best-effort exclusive lock using atomic file creation."""

    def __init__(self, path: Path, *, stale_after_seconds: float = 72 * 3600) -> None:
        self.path = path
        self.stale_after_seconds = stale_after_seconds
        self.acquired = False

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age > self.stale_after_seconds:
                self.path.unlink(missing_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            details = ""
            try:
                details = self.path.read_text(encoding="utf-8")
            except OSError:
                pass
            raise RineyaSoupError(
                f"Another process appears to own this run: {self.path}\n{details}"
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\nstarted={utc_timestamp()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False
