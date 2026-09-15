"""Create-only, hash-chained accounting for one explicitly approved envelope.

Callers serialize access with the cooperative model lease. Unsettled launches
retain their entire reservation and block another launch, even after a restart.
Only an observed terminal supervisor result can settle a launch.
"""
from __future__ import annotations
import math
import re
from pathlib import Path

from experiments.model_quality_gpu_v1.integrity import digest, read_json, seal, unseal, write_new
from .preflight import LIMITS


def validate_grant(grant: dict, data_sha: str, plan_sha: str) -> None:
    if (grant.get("experiment") != "RRP-001-MICROFIT" or grant.get("decision") != "APPROVED"
            or grant.get("training_permission") is not True
            or not isinstance(grant.get("approved_by"), str) or not grant["approved_by"].strip()
            or not isinstance(grant.get("authorization_reference"), str) or not grant["authorization_reference"].strip()):
        raise ValueError("explicit session engineering-budget approval is absent")
    if grant.get("data_contract_sha256") != data_sha or grant.get("plan_sha256") != plan_sha:
        raise ValueError("approval does not bind the exact prepared data and plan")
    limits = grant.get("limits")
    if not isinstance(limits, dict) or set(limits) != set(LIMITS):
        raise ValueError("every numeric budget must be explicit")
    if any(type(v) is not int or not 0 <= v <= LIMITS[k] for k, v in limits.items()):
        raise ValueError("invalid or expanded execution budget")
    excluded = {"broad_SFT", "RL", "larger_teacher", "cloud", "release", "architectural_change"}
    if set(grant.get("excludes", [])) != excluded:
        raise ValueError("engineering grant cannot expand into later research lanes")


class Journal:
    def __init__(self, directory: Path, grant: dict, *, create=False):
        self.directory = Path(directory).absolute()
        if any(p.is_symlink() for p in (self.directory, *self.directory.parents)):
            raise ValueError("unsafe budget journal path")
        if create:
            self.directory.mkdir(parents=True, exist_ok=False)
            (self.directory / "events").mkdir()
            (self.directory / "runs").mkdir()
            write_new(self.directory / "grant.json", grant)
        self.grant = read_json(self.directory / "grant.json")
        if self.grant != grant:
            raise ValueError("budget journal is bound to a different grant")
        self.read()

    def read(self):
        paths = sorted((self.directory / "events").iterdir())
        if len(paths) > 10000:
            raise ValueError("journal event ceiling exceeded")
        used = {key: 0 for key in self.grant["limits"]}
        previous, active, phases, settled, started = None, {}, [], set(), set()
        for index, path in enumerate(paths):
            if path.name != f"{index:06d}.json":
                raise ValueError("incomplete or unexpected budget event inventory")
            raw = read_json(path); event = unseal(raw)
            if event["index"] != index or event["previous"] != previous:
                raise ValueError("budget event chain differs")
            charges = event.get("charges", {})
            if set(charges) - set(used) or any(type(v) is not int or v < 0 for v in charges.values()):
                raise ValueError("invalid budget charge")
            for key, amount in charges.items(): used[key] += amount
            if event["kind"] == "start":
                if active or event["run_id"] in settled:
                    raise ValueError("overlapping or reused launch")
                active[event["run_id"]] = event
                started.add(event["run_id"])
                phases.append(event["phase"])
            elif event["kind"] == "settle":
                run = active.pop(event["run_id"], None)
                if run is None or event["run_id"] in settled:
                    raise ValueError("settlement has no active reservation")
                for key in ("total_owned_work_wall_seconds", "model_worker_wall_seconds_and_conservative_gpu_seconds"):
                    actual = event["actual_wall"][key]
                    if type(actual) is not int or actual < 0:
                        raise ValueError("invalid measured wall time")
                    used[key] += actual - run["charges"][key]
                settled.add(event["run_id"])
            elif event["kind"] != "charge":
                raise ValueError("unknown budget event")
            previous = raw["sha256"]
        if {p.name for p in (self.directory / "runs").iterdir()} - started:
            raise ValueError("run evidence has no corresponding budget reservation")
        self.used, self.previous, self.index = used, previous, len(paths)
        self.active, self.phases = active, phases
        self.started, self.settled = started, settled
        self.breached = any(used[k] > v for k, v in self.grant["limits"].items())
        return {"used_or_reserved": used, "active_reservations": list(active), "budget_breached": self.breached,
                "remaining": {k: max(0, v - used[k]) for k, v in self.grant["limits"].items()}}

    def _append(self, kind, **fields):
        self.read()
        event = seal({"index": self.index, "previous": self.previous, "kind": kind, **fields})
        write_new(self.directory / "events" / f"{self.index:06d}.json", event)
        self.read()
        return event

    def reserve(self, charges):
        self.read()
        if (self.breached or set(charges) - set(self.used)
                or any(type(v) is not int or v < 0 for v in charges.values())):
            raise ValueError("invalid charge or previously breached budget")
        if any(self.used[k] + v > self.grant["limits"][k] for k, v in charges.items()):
            raise RuntimeError("approved budget exhausted; preserve the attempt")
        return self._append("charge", charges=charges)

    def start(self, run_id, phase, timeout):
        self.read()
        if not isinstance(run_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,59}", run_id) is None:
            raise ValueError("unsafe run ID")
        # Reject a reused ID *before* appending. read() would also catch it, but only
        # after the duplicate event was durable, which bricks the journal for good and
        # strands the rest of the approved envelope behind an unreadable chain.
        if run_id in self.started:
            raise ValueError("reused run ID; every attempt needs its own preserved evidence")
        if self.active:
            raise RuntimeError("unsettled prior worker; verify its actual process/terminal receipt before any new launch")
        if phase not in ("qualify", "train", "resume") or type(timeout) is not int or timeout < 1:
            raise ValueError("invalid phase/timeout")
        charges = {"model_worker_launches": 1, "new_disk_bytes": 4 * 2**20,
                   "model_worker_wall_seconds_and_conservative_gpu_seconds": timeout + 20,
                   "total_owned_work_wall_seconds": timeout + 140}
        if phase in self.phases: charges["retries"] = 1
        if phase == "train" and phase not in self.phases: charges["adaptive_model_selection_rounds"] = 1
        if self.breached or any(self.used[k] + v > self.grant["limits"][k] for k, v in charges.items()):
            raise RuntimeError("insufficient approved budget for this bounded launch")
        if (self.directory / "runs" / run_id).exists():
            raise FileExistsError("run evidence already exists")
        return self._append("start", run_id=run_id, phase=phase, charges=charges)

    def settle(self, run_id, supervisor: dict, receipt_sha: str, owned_wall: float):
        self.read()
        if run_id not in self.active:
            raise ValueError("no matching launch to settle")
        cleanup = supervisor.get("cleanup")
        if cleanup is not None and not cleanup.get("child_exited"):
            raise RuntimeError("worker termination unverified; reservation remains active")
        if supervisor.get("returncode") is None and supervisor.get("child_pid") is not None:
            raise RuntimeError("worker has no observed exit code")
        worker_wall = supervisor.get("wall_seconds")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in (worker_wall, owned_wall)):
            raise ValueError("invalid terminal timing")
        if owned_wall + 1e-6 < worker_wall:
            raise ValueError("owned wall time cannot omit worker time")
        # A measured overrun remains recorded as a breach; it is never clamped away.
        return self._append("settle", run_id=run_id, receipt_sha256=receipt_sha, charges={},
            actual_wall={"total_owned_work_wall_seconds": math.ceil(owned_wall),
                         "model_worker_wall_seconds_and_conservative_gpu_seconds": math.ceil(worker_wall)})
