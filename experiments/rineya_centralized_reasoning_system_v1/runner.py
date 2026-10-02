"""Bounded CRS coordinator/evaluator. Native worker dispatch lives in native.py.

The scorer and gold stay at this host boundary; the runtime worker receives only
an allowlisted public task and frozen procedure snapshot. Same-account filesystem
access is not OS-isolated; no stronger secrecy claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest
import uuid

from .atlas import paired_parent_interval
from .core import ProcedureSnapshot, canonical, digest, supplied_snapshot
from .data import audit_splits, build_dataset, reader_cases, shortcut_baselines
from .operations import OperationsError, OperationsStore, host_state_path
from .scoring import score

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent.parent
MODULE = "experiments.rineya_centralized_reasoning_system_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding():
    paths = [*PACKAGE.glob("*.py"), *PACKAGE.glob("contracts/*.json"),
             *PACKAGE.glob("tests/*.py"), PACKAGE / "AUTHORIZATION_2026-10-01.md"]
    paths = [p for p in paths if p.name != "native.py" and p.name != "test_native.py" and p.name != "NATIVE_DIAGNOSTIC_V1.json"]
    pins = {str(p.relative_to(WORKSPACE)).replace("\\", "/"): sha(p) for p in sorted(paths)}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True).strip()
    return {"code_commit": commit, "source_sha256": digest(pins), "source_files": pins,
            "source_scope": "CPU pipeline only; native path qualified separately",
            "python": sys.version, "interpreter": sys.executable, "platform": platform.platform(),
            "authority_sha256": sha(PACKAGE/"AUTHORIZATION_2026-10-01.md"),
            "owner_request_sha256": sha(PACKAGE/"authorization/owner-request-2026-10-01.txt"),
            "model_sha256": None, "config_sha256": None, "tokenizer_sha256": None,
            "model_binding_reason": "CPU fixture; zero model loads/calls/updates",
            "prompt": "controlled public grammar only; no neural prompt or decoder",
            "decoding": None, "training": None,
            "scorer_sha256": sha(PACKAGE/"scoring.py"),
            "cpu_contract_sha256": sha(PACKAGE/"contracts/CPU_V1.json")}


def irrelevant_snapshot(reference):
    entries = reference.entries()
    entries[0]["id"] = "irrelevant"  # padding validation matches total byte length
    entries[0]["applicability"] = "irrelevant_nonownership_v1"
    entries[0]["validation"] = "IRRELEVANT"
    target = len(reference._bytes)
    current = len(canonical(entries))
    if current > target:
        raise ValueError("D snapshot cannot exceed registered C bytes")
    entries[0]["validation"] += " "*(target-current)
    snapshot = ProcedureSnapshot(entries)
    if len(snapshot._bytes) != target:
        raise ValueError("D size mismatch")
    return snapshot


def worker_request(case, arm, snapshot):
    raw = canonical({"case": case, "arm": arm, "snapshot": snapshot.entries() if snapshot else None})
    start = time.monotonic()
    try:
        p = subprocess.run([sys.executable, "-m", MODULE+".runtime_worker"], input=raw,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=WORKSPACE,
                           timeout=15, check=False)
        output = json.loads(p.stdout)
        if p.returncode and "error" not in output:
            raise ValueError("worker exit/result mismatch")
        output["public_request_sha256"] = hashlib.sha256(raw).hexdigest()
        output["worker_wall_seconds"] = time.monotonic()-start
        output["worker_exit"] = p.returncode
        output["worker_stderr"] = p.stderr.decode("utf-8", errors="replace")[:4096]
        return output
    except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
        return {"case_id": case["id"], "arm": arm, "error": str(exc), "stop_reason": "WORKER_UNAVAILABLE",
                "worker_wall_seconds": time.monotonic()-start, "public_request_sha256": hashlib.sha256(raw).hexdigest()}


def execute_case(store, key, case, arm, snapshot, epoch):
    request_identity = {"public_case_sha256": digest(case), "arm": arm,
                        "snapshot_sha256": snapshot.sha256 if snapshot else None}
    try:
        old = store.get_job(key)
    except OperationsError as exc:
        if str(exc) != "unregistered job":
            raise
        old = None
    job = store.register_job(key, request_identity, epoch)
    if old is not None:
        if job["state"] in {"SUCCEEDED", "FAILED"}:
            return store.verify_receipt(job["result"]["evidence"]["case_receipt"])["payload"]
        # An unacknowledged prior attempt is retained as a failure, never re-executed.
        recovered = {"case_id": case["id"], "arm": arm, "error": "OUTCOME_UNKNOWN_CPU_PRIOR_ATTEMPT",
                     "stop_reason": "RECONCILIATION_REQUIRED", "worker_wall_seconds": 0,
                     "prior_partial_artifacts_preserved": True}
        receipt = store.write_receipt("case-recovery", key, recovered)
        store.finish_cpu(key, "FAILED", {"case_receipt": receipt})
        return recovered
    store.ready(key)
    result = worker_request(case, arm, snapshot)
    receipt = store.write_receipt("case", key, result)
    store.finish_cpu(key, "SUCCEEDED" if result.get("error") is None else "FAILED", {"case_receipt": receipt})
    return result


def cpu_epoch(store, epoch="CRS-EPOCH-001", count=64, seed=20261001):
    if type(count) is not int or not 16 <= count <= 256:
        raise ValueError("bounded CPU development count 16..256 required")
    train, dev, reader = build_dataset("TRAIN", 32, seed), build_dataset("DEV", count, seed), reader_cases()
    split = audit_splits(train, dev)
    library = supplied_snapshot()
    irrelevant = irrelevant_snapshot(library)
    source = binding()
    payload = {"binding": source, "seed": seed, "count": count,
               "datasets": [train["manifest"], dev["manifest"], reader["manifest"]],
               "procedure_snapshot_sha256": library.sha256, "D_snapshot_sha256": irrelevant.sha256,
               "qualification": "independent CPU mechanics review required for CPU_QUALIFIED; never native",
               "budget": {"source": "contracts/CPU_V1.json", "max_case_worker_wall_seconds": 15,
                          "case_limit": 3*count+16, "native_seconds": 0},
               "split_audit": split, "exposure": "TRAIN_DEV_ALREADY_OPEN_NO_CONFIRMATION"}
    key = "cpu-"+digest(payload)
    job = store.register_job(key, payload, epoch)
    if job["state"] == "SUCCEEDED":
        return store.verify_receipt(job["result"]["evidence"]["result_receipt"])["payload"]
    if job["state"] != "PENDING" and job["state"] != "READY":
        raise OperationsError("failed or unknown epoch preserved; register a changed successor")
    if job["state"] == "PENDING":
        store.ready(key)
    owner = "CPU-"+str(uuid.uuid4())
    lease = store.acquire_lease(owner, 600, kind="coordinator")
    raw, scores = {}, {}
    try:
        store.write_receipt("registration", key, payload)
        for panel, bundle, arms in [("reader", reader, ["B"]), ("procedure", dev, ["B", "C", "D"])]:
            predictions = {arm: {} for arm in arms}
            raw[panel] = {arm: [] for arm in arms}
            for case in bundle["cases"]:
                for arm in arms:
                    case_key = key+":"+panel+":"+case["id"]+":"+arm
                    result = execute_case(store, case_key, case, arm,
                                          library if arm == "C" else irrelevant if arm == "D" else None, epoch)
                    raw[panel][arm].append(result)
                    predictions[arm][case["id"]] = result["answer"] if result.get("error") is None and "answer" in result else {"error": result.get("error", "unavailable")}
                store.heartbeat_lease(owner, lease["fence"], 600, kind="coordinator")
            scores[panel] = {arm: score(predictions[arm], bundle["gold"]) for arm in arms}
        pairs = [{"parent_id": c["parent_id"], "family": c["family"],
                  "B": float(scores["procedure"]["B"]["rows"][c["id"]]["semantic_exact"]),
                  "C": float(scores["procedure"]["C"]["rows"][c["id"]]["semantic_exact"])} for c in dev["cases"]]
        output = {"job_key": key, "epoch": epoch, "status": "CPU_ENGINEERING_COMPLETED",
                  "binding": source, "datasets": payload["datasets"], "split_audit": split,
                  "scores": scores, "B_C_parent_comparison": paired_parent_interval(pairs),
                  "shortcuts": {name: score(pred, dev["gold"]) for name, pred in shortcut_baselines(dev).items()},
                  "raw": raw, "all_scheduled_cases": 3*count+16,
                  "model_only_A": {"status": "UNAVAILABLE_NO_NATIVE_MODEL", "planned_cases": count},
                  "cost": {"model_loads": 0, "model_calls": 0, "training_updates": 0, "native_worker_seconds": 0,
                           "cpu_worker_seconds": sum(r["worker_wall_seconds"] for panel in raw.values() for arm in panel.values() for r in arm)},
                  "D_matching": {"unit": "canonical_UTF8_bytes_CPU_only", "C_bytes": len(library._bytes), "D_bytes": len(irrelevant._bytes), "native_tokens_matched": False},
                  "claim": "supplied fixture infrastructure and strong primitive parity only; no Rineya efficacy/discovery/weight learning/retention/calibration claim",
                  "stop_reason": "PLANNED_CPU_CASES_COMPLETED_NATIVE_GATES_CLOSED"}
        receipt = store.write_receipt("result", key, output)
        store.finish_cpu(key, "SUCCEEDED", {"result_receipt": receipt})
        return output
    except BaseException as exc:
        store.write_receipt("failure", key, {"exception": f"{type(exc).__name__}: {exc}",
                                            "partial_cases_preserved": True, "native_seconds": 0})
        raise
    finally:
        store.release_lease(owner, lease["fence"], kind="coordinator")


def qualify_cpu(store):
    source = binding()
    key = "tests-"+digest({"source": source["source_sha256"], "commit": source["code_commit"]})
    job = store.register_job(key, source, "CRS-CPU-QUALIFICATION")
    if job["state"] in {"SUCCEEDED", "FAILED"}:
        return store.verify_receipt(job["result"]["evidence"]["test_receipt"])["payload"]
    if job["state"] == "PENDING":
        store.ready(key)
    suite = unittest.defaultTestLoader.discover(str(PACKAGE/"tests"), pattern="test_*.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    payload = {"binding": source, "tests_run": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skips": result.skipped,
               "passed": result.wasSuccessful(), "raw_log": stream.getvalue(),
               "evidence_class": "CPU_ENGINEERING_FIXTURE", "native_qualified": False}
    receipt = store.write_receipt("tests", key, payload)
    store.finish_cpu(key, "SUCCEEDED" if result.wasSuccessful() else "FAILED", {"test_receipt": receipt})
    return payload


def status(store):
    jobs = [dict(r) for r in store.db.execute("SELECT key,epoch,state,reserved_seconds,actual_seconds FROM jobs ORDER BY created")]
    return {"track": "CRS-001", "native_launch": "CLOSED_UNTIL_EXACT_QUALIFICATION",
            "jobs": jobs, "budget": store.budget_snapshot("CRS-EPOCH-001"),
            "host_state": str(store.db_path), "existing_central_CHECK_FINAL": "CLOSED",
            "next": "native preflight, numerical diagnostic under reviewed containment; reader only after exact qualification"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["status", "qualify-cpu", "cpu-epoch", "resume", "preflight"])
    parser.add_argument("--state", type=Path, default=PACKAGE/"state/operations.sqlite3")
    parser.add_argument("--host-wide", action="store_true")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "preflight":
        from .native import CONTRACT, preflight
        output = preflight(contract_sha256=sha(CONTRACT), review_path=args.review)
    else:
        db = host_state_path() if args.host_wide else args.state
        store = OperationsStore(db, PACKAGE/"outputs/receipts")
        try:
            output = qualify_cpu(store) if args.command == "qualify-cpu" else cpu_epoch(store, count=args.count) if args.command in {"cpu-epoch", "resume"} else status(store)
        finally:
            store.close()
    if args.command in {"cpu-epoch", "resume"}:
        print(json.dumps({k: output[k] for k in ("job_key", "status", "all_scheduled_cases", "cost", "stop_reason")}, indent=2))
    elif args.command == "qualify-cpu":
        print(json.dumps({k: output[k] for k in ("tests_run", "failures", "errors", "skips", "passed")}, indent=2))
    else:
        print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
