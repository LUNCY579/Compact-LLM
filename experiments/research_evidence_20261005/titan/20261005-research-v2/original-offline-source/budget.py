"""Create-only cumulative attempt accounting; this module authorizes no work.

Declared full attempt caps are charged before entry. Failures/unknown outcomes
never refund them; measured overruns are added rather than hidden. The canonical
ledger is fixed across permits and source freezes. No reclaim, reset or retry.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import uuid

from . import registry, schema

ARTIFACT_ROOT = Path("C:/Users/clclc/BusinessProject/ChatBubble-agent-prompts-20261002/run-artifacts/titan-quality-repair-v1/20261005-offline")
CANONICAL_LEDGER_ROOT = ARTIFACT_ROOT / "budget-ledger"
HEADER = "BUDGET_PLAN.json"
LOCK = "BUDGET_WRITE.lock"
EVENT_PATTERN = re.compile(r"^EVENT_(\d{6})_([0-9a-f]{32})\.json$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
RESULT_STATES = {"SUCCEEDED", "FAILED", "BLOCKED", "ABORTED", "OUTCOME_UNKNOWN"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def seconds(value, *, positive=False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("STOP_INVALID_COST_SECONDS")
    return float(value)


def valid_id(value: str) -> str:
    if not isinstance(value, str) or ID_PATTERN.fullmatch(value) is None:
        raise ValueError("STOP_INVALID_ATTEMPT_ID")
    return value


class BudgetLedger:
    def __init__(self, path: Path | None = None, *, recipe_registry: dict | None = None):
        requested = Path(path if path is not None else CANONICAL_LEDGER_ROOT)
        if any(parent.is_symlink() for parent in (requested, *requested.parents)):
            raise ValueError("STOP_BUDGET_LEDGER_SYMLINK")
        self.path = requested.resolve()
        if self.path != CANONICAL_LEDGER_ROOT.resolve():
            raise ValueError("STOP_BUDGET_LEDGER_ROOT_ROTATION")
        if any(part.lower() == ".codex" for part in self.path.parts):
            raise ValueError("STOP_BUDGET_LEDGER_INSIDE_CODEX")
        for parent in (self.path, *self.path.parents):
            if (parent / ".git").exists():
                raise ValueError("STOP_BUDGET_LEDGER_INSIDE_SOURCE")
        self.registry = registry.build_registry() if recipe_registry is None else registry.validate_registry(recipe_registry)
        self.plan = registry.make_plan(self.registry)
        self.trial_ids = {t["trial_id"] for t in self.registry["trials"]}
        self.path.mkdir(parents=True, exist_ok=True)
        with self._locked():
            header = self.path / HEADER
            if header.exists():
                self._check_header()
            else:
                # A missing header with surviving events is lost provenance, not
                # an empty budget. Preserve it and refuse instead of repairing.
                if any(p.name.startswith("EVENT_") for p in self.path.iterdir()):
                    raise RuntimeError("STOP_BUDGET_HEADER_MISSING_PRESERVE_LEDGER")
                write_new(header, {"experiment": schema.EXPERIMENT, "registry": self.registry,
                                   "plan": self.plan, "created_utc": utc_now(), "native_execution_authorized": False})
            self._events()

    @contextmanager
    def _locked(self):
        token = uuid.uuid4().hex
        lock = self.path / LOCK
        write_new(lock, {"pid": os.getpid(), "token": token, "utc": utc_now()})
        try:
            yield
        finally:
            # Never reclaim an unknown/stale writer. A release deletes only this
            # exact token. A sharing/error path retains the lock fail-closed.
            try:
                if json.loads(lock.read_bytes()).get("token") == token:
                    lock.unlink()
            except (OSError, ValueError):
                pass

    def _check_header(self):
        try:
            value = json.loads((self.path / HEADER).read_bytes())
        except (OSError, ValueError) as exc:
            raise RuntimeError("STOP_BUDGET_HEADER_UNREADABLE") from exc
        if (not isinstance(value, dict) or value.get("registry") != self.registry or value.get("plan") != self.plan
                or value.get("experiment") != schema.EXPERIMENT or value.get("native_execution_authorized") is not False):
            raise RuntimeError("STOP_BUDGET_BINDING_DRIFT")

    def _events(self) -> list[dict]:
        self._check_header()
        paths = sorted(p for p in self.path.iterdir() if p.name.startswith("EVENT_"))
        events, attempts, results, sequence = [], {}, set(), 0
        for path in paths:
            match = EVENT_PATTERN.fullmatch(path.name)
            if match is None or int(match.group(1)) != sequence + 1:
                raise RuntimeError("STOP_BUDGET_EVENT_SEQUENCE_OR_NAME")
            sequence += 1
            try:
                event = json.loads(path.read_bytes())
                expected = event.pop("event_sha256")
                if schema.digest(event) != expected:
                    raise ValueError("event digest mismatch")
                event["event_sha256"] = expected
            except (OSError, ValueError, KeyError, AttributeError) as exc:
                raise RuntimeError("STOP_BUDGET_EVENT_UNREADABLE_OR_CHANGED") from exc
            if event.get("registry_sha256") != self.registry["registry_sha256"]:
                raise RuntimeError("STOP_BUDGET_EVENT_BINDING_DRIFT")
            attempt = valid_id(event.get("attempt_id"))
            kind = event.get("kind")
            if kind in {"CHARGED", "SKIPPED_BUDGET_EXHAUSTED"}:
                if attempt in attempts or event.get("stage") not in self.plan["budget_seconds"]:
                    raise RuntimeError("STOP_BUDGET_DUPLICATE_OR_INVALID_ATTEMPT")
                seconds(event.get("declared_seconds"))
                attempts[attempt] = event
            elif kind == "RESULT":
                if (attempt not in attempts or attempts[attempt]["kind"] != "CHARGED" or attempt in results
                        or event.get("stage") != attempts[attempt]["stage"]
                        or event.get("status") not in RESULT_STATES):
                    raise RuntimeError("STOP_BUDGET_RESULT_LINK_OR_STATE")
                if event.get("observed_seconds") is not None:
                    seconds(event["observed_seconds"])
                results.add(attempt)
            else:
                raise RuntimeError("STOP_BUDGET_UNKNOWN_EVENT_KIND")
            events.append(event)
        return events

    def _append(self, value: dict, events: list[dict]) -> dict:
        event = {"experiment": schema.EXPERIMENT, "registry_sha256": self.registry["registry_sha256"],
                 "utc": utc_now(), **value}
        event["event_sha256"] = schema.digest(event)
        write_new(self.path / f"EVENT_{len(events)+1:06d}_{uuid.uuid4().hex}.json", event)
        return event

    def _snapshot(self, events: list[dict]) -> dict:
        attempts = {e["attempt_id"]: e for e in events if e["kind"] in {"CHARGED", "SKIPPED_BUDGET_EXHAUSTED"}}
        results = {e["attempt_id"]: e for e in events if e["kind"] == "RESULT"}
        used = {stage: 0.0 for stage in self.plan["budget_seconds"]}
        for attempt, event in attempts.items():
            if event["kind"] == "CHARGED":
                observed = results.get(attempt, {}).get("observed_seconds")
                used[event["stage"]] += max(event["declared_seconds"], observed or 0.0)
        return {"registry_sha256": self.registry["registry_sha256"], "ledger_root": str(self.path),
                "charged_seconds": used, "total_charged_seconds": sum(used.values()),
                "remaining_seconds": {s: max(0.0, cap-used[s]) for s, cap in self.plan["budget_seconds"].items()},
                "total_remaining_seconds": max(0.0, self.plan["total_budget_seconds"]-sum(used.values())),
                "attempts": list(attempts.values()), "results": list(results.values()),
                "budget_overrun": any(used[s] > cap for s, cap in self.plan["budget_seconds"].items()),
                "no_refunds_or_resets": True, "native_execution_authorized": False}

    def snapshot(self) -> dict:
        return self._snapshot(self._events())

    def charge_attempt(self, stage: str, attempt_id: str, declared_seconds, *, trial_id: str | None = None) -> dict:
        """Charge full declared cost before entry, or retain a terminal skip.

        The caller must separately obtain all genuine source, permit, ownership,
        resource and measured-throughput gates. This method launches nothing.
        """
        attempt_id, cost = valid_id(attempt_id), seconds(declared_seconds, positive=True)
        if stage not in self.plan["budget_seconds"]:
            raise ValueError("STOP_UNKNOWN_BUDGET_STAGE")
        if stage == "training" and (trial_id not in self.trial_ids or cost != self.plan["trial_wall_seconds"]):
            raise ValueError("STOP_TRAINING_TRIAL_OR_FIXED_ATTEMPT_COST")
        if stage != "training" and trial_id is not None:
            raise ValueError("STOP_TRIAL_ID_OUTSIDE_TRAINING")
        with self._locked():
            events = self._events()
            prior = [e for e in events if e["kind"] in {"CHARGED", "SKIPPED_BUDGET_EXHAUSTED"}]
            if any(e["attempt_id"] == attempt_id or (trial_id is not None and e.get("trial_id") == trial_id) for e in prior):
                raise RuntimeError("STOP_ATTEMPT_OR_TRIAL_ALREADY_USED")
            state = self._snapshot(events)
            skip = (state["budget_overrun"] or cost > state["remaining_seconds"][stage]
                    or cost > state["total_remaining_seconds"])
            return self._append({"kind": "SKIPPED_BUDGET_EXHAUSTED" if skip else "CHARGED", "stage": stage,
                                 "attempt_id": attempt_id, "trial_id": trial_id, "declared_seconds": 0 if skip else cost,
                                 "requested_seconds": cost, "status": "SKIPPED" if skip else "RESERVED_COST_CHARGED"}, events)

    charge = charge_attempt

    def record_result(self, attempt_id: str, status: str, *, observed_seconds=None, receipt: str | None = None) -> dict:
        attempt_id = valid_id(attempt_id)
        if status not in RESULT_STATES:
            raise ValueError("STOP_INVALID_ATTEMPT_RESULT_STATE")
        actual = None if observed_seconds is None else seconds(observed_seconds)
        with self._locked():
            events = self._events()
            charges = [e for e in events if e["kind"] == "CHARGED" and e["attempt_id"] == attempt_id]
            if len(charges) != 1 or any(e["kind"] == "RESULT" and e["attempt_id"] == attempt_id for e in events):
                raise RuntimeError("STOP_ATTEMPT_RESULT_ALREADY_RECORDED_OR_UNCHARGED")
            return self._append({"kind": "RESULT", "attempt_id": attempt_id, "stage": charges[0]["stage"],
                                 "status": status, "observed_seconds": actual, "receipt": receipt,
                                 "declared_seconds_retained": charges[0]["declared_seconds"],
                                 "additional_overrun_seconds": max(0.0, (actual or 0.0)-charges[0]["declared_seconds"])}, events)
