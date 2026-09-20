"""One explicitly authorized, source-bound Stage C worker; no retries or polling.

The controller imports no training stack. It reuses the existing cooperative GPU
lease and foreground supervisor, leaving every other research lane untouched.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.model_quality_gpu_v1.resources import lease
from experiments.model_quality_gpu_v1.supervisor import bounded_child
from experiments.rab001_exploration.hostbudget import snapshot

from . import receipts as r

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("student_base", "donor_base", "donor_skilled")
HOST_GIB = {"student_base": 7, "donor_base": 12, "donor_skilled": 12}


def deadline_seconds(spec, now=None):
    deadline = dt.datetime.fromisoformat(spec["deadline_utc"].replace("Z", "+00:00"))
    r.require(deadline.tzinfo is not None, "session deadline needs timezone")
    remaining = deadline.timestamp() - (time.time() if now is None else now)
    r.require(remaining > 1, "session expired")
    return min(600, int(remaining))


def validate_source(spec):
    r.require(spec.get("schema") == 1 and spec.get("experiment") == "CSC-001" and spec.get("stage") == "C",
              "wrong session registration")
    r.require(spec.get("native_capture_authorized") is True, "no native capture authority")
    r.require(spec.get("authorization_evidence") ==
              "Commit and push the branch as you change and progress. Let's proceed with the goal to test if the model's intelligence can be improved further.",
              "missing current user authorization evidence")
    r.require(spec.get("max_total_forwards") == 96 and spec.get("max_role_forwards") == 32,
              "session call envelope differs")
    r.require(spec.get("max_optimizer_steps") == 0 and spec.get("max_generated_tokens") == 0,
              "capture cannot train or generate")
    r.verify_source(ROOT, spec.get("source_commit"), clean=True)
    r.require(isinstance(spec.get("source_files"), dict) and bool(spec["source_files"]), "missing source inventory")
    inventory = r.implementation_digest(ROOT, list(spec["source_files"]))
    r.require(inventory["files"] == spec["source_files"] and inventory["sha256"] == spec["implementation_digest"],
              "native source bytes changed")
    for required in ("native_session.py", "native_capture.py", "receipts.py"):
        r.require("experiments/cross_scale_composable_001/" + required in spec["source_files"],
                  "unbound native source file")
    deadline_seconds(spec)


def prelaunch(spec, role, memory):
    r.require(role in ROLES, "unknown capture role")
    r.require(not (Path(spec["attempt_root"]).parent / "SESSION_CLOSED.json").exists(),
              "session is terminal; a fresh prospective registration is required")
    validate_source(spec)
    r.require(memory.get("available") is True, "host resource measurement unavailable")
    r.require(memory.get("free_phys_gib", 0) >= HOST_GIB[role],
              f"{role} requires unchanged {HOST_GIB[role]} GiB available host RAM")
    # Never remove or overwrite an existing lease, even if an owner looks terminal.
    r.require(not Path(spec["lease_path"]).exists(), "shared GPU lease occupied")
    runtime = spec["runtime"]["student_base" if role == "student_base" else "donor"]
    r.require(Path(runtime["python_executable"]).is_file(), "native interpreter missing")
    return runtime


def run(spec_path, spec_sha256, role, attempt):
    r.verify_file(spec_path, spec_sha256)
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    attempt = Path(attempt).resolve()
    r.require(not attempt.exists(), "attempt path already exists; no retry/reuse")
    registered_root = Path(spec["attempt_root"]).resolve()
    r.require(attempt.parent == registered_root and attempt.name == role, "attempt is outside registered role directory")
    attempt.mkdir(parents=True)
    memory = snapshot()
    r.write_once(attempt / "PRELAUNCH.json", {"memory": memory, "role": role,
        "required_host_gib": HOST_GIB.get(role), "spec_sha256": spec_sha256,
        "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    status, code, worker = "FAILED_BEFORE_DISPATCH", None, None
    try:
        runtime = prelaunch(spec, role, memory)
        with lease(Path(spec["lease_path"])):
            own_lease = json.loads(Path(spec["lease_path"]).read_text(encoding="utf-8"))
            runtime_spec = dict(spec)
            runtime_spec["lease"] = {"path": spec["lease_path"], "token": own_lease["token"],
                                     "owner_pid": own_lease["pid"]}
            runtime_spec["frozen_registration_sha256"] = spec_sha256
            runtime_path = attempt / "WORKER_SPEC.json"
            runtime_sha = r.write_once(runtime_path, runtime_spec)
            supervision = attempt / "supervision"
            capture = supervision / "capture"
            command = [runtime["python_executable"], "-B", "-m",
                       "experiments.cross_scale_composable_001.native_capture",
                       "--spec", str(runtime_path), "--spec-sha256", runtime_sha,
                       "--role", role, "--out", str(capture), "--supervision", str(supervision)]
            status = "FAILED_CHILD"
            code = bounded_child(command, supervision, deadline_seconds(spec))
            r.require(code == 0, "native child failed; reservation remains consumed")
            # The worker's terminal receipt is mandatory: successful process exit is insufficient.
            terminal = capture / "RESULT.json"
            r.require(terminal.is_file(), "worker exited without capture receipt")
            worker = json.loads(terminal.read_text(encoding="utf-8"))
            r.complete_child(code, worker)
            r.require(worker.get("native_model_forwards") == 32 and worker.get("optimizer_steps") == 0,
                      "native child call budget/completion differs")
            status = "COMPLETE"
    except Exception as exc:
        r.write_once(attempt / "FAILURE.json", {"type": type(exc).__name__, "message": str(exc),
                                              "status": status, "automatic_retry": False})
        raise
    finally:
        known_count = worker.get("native_model_forwards") if worker else (0 if status == "FAILED_BEFORE_DISPATCH" else None)
        r.write_once(attempt / "SESSION_TERMINAL.json", {"status": status, "completed": status == "COMPLETE",
            "role": role, "child_returncode": code, "native_forward_count": known_count,
            "forward_count_note": "Consult durable role reservation/journal when child absent or failed; never infer zero.",
            "optimizer_steps": 0, "scientific_improvement_claim": False,
            "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        closed = registered_root.parent / "SESSION_CLOSED.json"
        if status != "COMPLETE" and not closed.exists():
            r.write_once(closed, {"status": "TERMINAL_AFTER_REFUSAL_OR_FAILED_CHILD", "role": role,
                "spec_sha256": spec_sha256, "automatic_retry": False,
                "native_forward_count": known_count,
                "next_requirement": "Genuine external resource recovery when resource-blocked, then fresh prospective registration."})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    return run(args.spec, args.spec_sha256, args.role, args.attempt)


if __name__ == "__main__":
    raise SystemExit(main())
