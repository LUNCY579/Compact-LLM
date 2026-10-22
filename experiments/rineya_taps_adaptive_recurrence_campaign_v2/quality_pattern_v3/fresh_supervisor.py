"""Coordinator-only prospective fresh parent; no model import in this module."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import psutil
from .fresh_scope import V3, CAMPAIGN, ROOT, GIB, sha, verify_dispatch, budget_and_unused
from ..bulk_restart_supervisor import gpu_inventory
from ..native_supervisor import append
from ..supervision import WindowsJob, resource_sample


def disk_ready():
    return (shutil.disk_usage(CAMPAIGN).free >= 15 * GIB and
            sum(p.stat().st_size for p in CAMPAIGN.rglob("*") if p.is_file()) <= 1536 * 1024**2)


def check_parent(source, out, admission_path):
    admission = verify_dispatch(admission_path)
    if V3 not in out.parents or out.exists() or str(source) != admission["source"] or sys.executable != admission["interpreter"]:
        raise PermissionError("exact source/interpreter and exclusive new V3 output required")
    events = [json.loads(s) for s in (CAMPAIGN / "BUDGET_LEDGER.jsonl").read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    budget_and_unused(events)
    for relative, expected in admission["checkpoint_files"].items():
        if sha(source / relative) != expected:
            raise PermissionError("canonical asset changed:" + relative)
    inventory = gpu_inventory()
    if (inventory["uuid"] != admission["gpu_uuid"] or inventory["driver_version"] != admission["gpu_driver_version"]
            or inventory["free_bytes"] < 6 * GIB):
        raise MemoryError("original GPU identity/six-GiB initial floor unavailable")
    if psutil.virtual_memory().available < 7 * GIB:
        raise MemoryError("original seven-GiB parent host floor unavailable")
    if not disk_ready():
        raise MemoryError("unchanged evidence disk reserve unavailable")
    if (CAMPAIGN / "native_worker.lock").exists():
        raise RuntimeError("live or unknown campaign lock; do not remove")
    return admission, inventory


def run(source, out, admission_path):
    source, out, admission_path = source.resolve(), out.resolve(), admission_path.resolve()
    admission, inventory = check_parent(source, out, admission_path)
    token = uuid.uuid4().hex
    lock = CAMPAIGN / "native_worker.lock"
    lock_record = {"token": token, "parent_pid": os.getpid(), "parent_created": psutil.Process().create_time(),
                   "runtime_exe": psutil.Process().exe(), "out": str(out), "lane": admission["lane"]}
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(lock_record, stream)
    supervisor_started = time.perf_counter()
    child = job = child_started = None
    assigned = created = False
    receipt = {"status": "FAIL_BEFORE_WORKER", "study_id": admission["study_id"], "lane": admission["lane"],
               "run": str(out), "admission_sha256": sha(admission_path), "native_worker_seconds": 0.,
               "started_utc": datetime.now(timezone.utc).isoformat(), "peak_rss_bytes": None,
               "gpu_initial_inventory": inventory}
    try:
        out.mkdir(parents=True, exist_ok=False); created = True
        (out / "admission.json").write_bytes(admission_path.read_bytes())
        (out / "review.json").write_bytes((V3 / admission["review_receipt"]).read_bytes())
        (out / "source_manifest.json").write_text(json.dumps({"sha256": admission["dispatch_hashes"], "head": admission["head"]}, indent=2) + "\n", encoding="utf-8")
        cache = out / "result/hf_cache"
        scratch = out / "temp"
        scratch.mkdir(parents=True, exist_ok=False)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "HF_HOME": str(cache), "HF_MODULES_CACHE": str(cache / "modules"), "HF_HUB_CACHE": str(cache / "hub"),
               "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "TMP": str(scratch), "TEMP": str(scratch), "CUDA_CACHE_PATH": str(out / "cuda_cache")}
        release = out / "release.json"
        command = [sys.executable, "-B", "-m", "experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.fresh_entry",
                   "--release", str(release), "--source", str(source), "--out", str(out / "result"), "--admission", str(admission_path)]
        receipt["command"] = command
        job = WindowsJob()
        with (out / "stdout.log").open("w", encoding="utf-8") as stdout, (out / "stderr.log").open("w", encoding="utf-8") as stderr:
            child_started = time.perf_counter()
            env["FRESH_PARENT_STARTED_PERF_COUNTER"] = repr(child_started)
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x4)
            receipt["worker_pid"] = child.pid
            job.assign(child); assigned = True
            launcher = psutil.Process(child.pid)
            lock_record.update(launcher_pid=child.pid, launcher_created=launcher.create_time(), launcher_exe=launcher.exe())
            replacement = out / "lock_record.tmp"
            replacement.write_text(json.dumps(lock_record), encoding="utf-8")
            os.replace(replacement, lock)
            job.resume_suspended(child)
            append(CAMPAIGN / "BUDGET_LEDGER.jsonl", {"event": "native_started", "study_id": admission["study_id"],
                   "lane": admission["lane"], "run": str(out), "reserved_seconds": 15600,
                   "worker_pid": child.pid, "timestamp_utc": datetime.now(timezone.utc).isoformat()})
            release.write_text(json.dumps({"contained": True, "token": token, "study_id": admission["study_id"],
                                           "admission_sha256": sha(admission_path)}), encoding="utf-8")
            last_progress = time.monotonic(); last_sizes = None; last_gpu = 0.
            while child.poll() is None:
                now = time.monotonic()
                try:
                    sample = resource_sample(child)
                except psutil.NoSuchProcess:
                    if child.poll() is not None:
                        break
                    raise
                sample["worker_seconds"] = time.perf_counter() - child_started
                if now - last_gpu >= 10:
                    inventory = gpu_inventory(); last_gpu = now
                    sample["gpu_global_inventory"] = inventory
                append(out / "resources.jsonl", sample)
                receipt["peak_rss_bytes"] = max(receipt["peak_rss_bytes"] or 0, sample["rss_bytes"])
                names = ("registration.json", "qualification.json", "forwards.jsonl", "trajectories.jsonl", "outcomes.jsonl", "receipt.json")
                paths = [out / "result" / name for name in names]
                sizes = tuple((p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in paths)
                if sizes != last_sizes:
                    last_sizes = sizes; last_progress = now
                reason = ("HARD_DEADLINE" if sample["worker_seconds"] > 15600 else
                          "HUNG_FORWARD" if now - last_progress > 180 else
                          "RSS_CEILING" if sample["rss_bytes"] > 6 * GIB else
                          "HOST_RESERVE" if sample["host_available_bytes"] < GIB else
                          "GPU_FREE_RESERVE" if inventory["free_bytes"] < GIB else
                          "GPU_IDENTITY" if inventory["uuid"] != admission["gpu_uuid"] else
                          "DISK_RESERVE" if not disk_ready() else None)
                if reason:
                    receipt.update(status="SUPERVISOR_SAFETY_STOP", stop_reason=reason)
                    job.terminate(); break
                time.sleep(1)
            child.wait(timeout=15)
            receipt.update(exit_code=child.returncode, native_worker_seconds=time.perf_counter() - child_started)
            if receipt["status"] != "SUPERVISOR_SAFETY_STOP":
                receipt["status"] = "WORKER_TERMINAL" if child.returncode == 0 else "WORKER_NONZERO"
    except BaseException as error:
        receipt["error"] = type(error).__name__ + ": " + str(error)
        if child is not None:
            job.terminate() if assigned else child.kill()
            child.wait(timeout=15)
            receipt["native_worker_seconds"] = time.perf_counter() - child_started
    finally:
        if child is not None and child_started is not None:
            receipt["native_worker_seconds"] = time.perf_counter() - child_started
        if job is not None:
            try:
                job.close()
            except BaseException as error:
                receipt.update(status="SUPERVISOR_CLEANUP_UNVERIFIED", cleanup_error=type(error).__name__ + ": " + str(error))
        terminated = child is None
        if child is not None:
            try:
                terminated = child.poll() is not None
            except BaseException as error:
                receipt["termination_check_error"] = type(error).__name__ + ": " + str(error)
            if receipt.get("cleanup_error"):
                terminated = False
        receipt["worker_termination_verified"] = terminated
        receipt["native_accounting_complete"] = terminated
        if not terminated:
            receipt["status"] = "SUPERVISOR_CLEANUP_UNVERIFIED"
        receipt.update(supervisor_wall_seconds=time.perf_counter() - supervisor_started,
                       ended_utc=datetime.now(timezone.utc).isoformat())
        if created:
            (out / "terminal_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        append(CAMPAIGN / "BUDGET_LEDGER.jsonl", {"event": "native_completed", **receipt})
        if terminated and lock.exists() and json.loads(lock.read_text(encoding="utf-8"))["token"] == token:
            lock.unlink()
    print(json.dumps(receipt))
    return 0 if receipt["status"] == "WORKER_TERMINAL" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "out", "admission"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source, args.out, args.admission))
