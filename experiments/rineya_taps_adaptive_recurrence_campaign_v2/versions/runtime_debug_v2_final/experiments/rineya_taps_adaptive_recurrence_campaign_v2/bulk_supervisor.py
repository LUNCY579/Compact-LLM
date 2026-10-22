"""Parent supervision for the explicitly registered larger-bank GPU FP32 screen only."""
from __future__ import annotations
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

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.run_apparatus import ROOT, PACKAGE, sha, sources, git
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.supervision import WindowsJob, resource_sample
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.native_supervisor import append

GIB = 1024**3


def gpu_inventory():
    answer = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version", "--format=csv,noheader,nounits"],
        text=True, timeout=5).strip().splitlines()
    if len(answer) != 1:
        raise RuntimeError("registered single-GPU inventory changed")
    values = [x.strip() for x in answer[0].split(",")]
    return {"index": int(values[0]), "uuid": values[1], "name": values[2],
            "total_bytes": int(values[3]) * 1024**2, "free_bytes": int(values[4]) * 1024**2,
            "utilization_percent": int(values[5]), "driver_version": values[6]}


def run(source, out, admission_path):
    source, out = source.resolve(), out.resolve()
    if PACKAGE.resolve() not in out.parents:
        raise ValueError("GPU output must be campaign-local")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission["status"] != "READY_FOR_BULK_FP32_DISPATCH":
        raise RuntimeError("GPU admission not ready")
    if admission["source"] != str(source) or admission["interpreter"] != sys.executable:
        raise RuntimeError("GPU source/interpreter binding mismatch")
    for relative, expected in admission["dispatch_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("reviewed GPU dispatch source changed: " + relative)
    review_path = PACKAGE / admission["review_receipt"]
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review["status"] != "PASS_BULK_FP32_DISPATCH" or review["admission_sha256"] != sha(admission_path):
        raise RuntimeError("GPU independent review is not bound")
    if psutil.virtual_memory().available < 7 * GIB:
        raise RuntimeError("new GPU lane6GiB host floor plus1GiB import reserve unavailable; no worker launched")
    inventory = gpu_inventory()
    if (inventory["uuid"] != admission["gpu_uuid"] or inventory["driver_version"] != admission["gpu_driver_version"]
            or inventory["free_bytes"] < 6 * GIB):
        raise RuntimeError("registered GPU identity/6GiB free floor unavailable; no worker launched")
    ledger = PACKAGE / "BUDGET_LEDGER.jsonl"
    events = [json.loads(x) for x in ledger.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    spent = sum(x.get("native_worker_seconds", 0) for x in events if x.get("event") == "native_completed")
    deadline = admission["hard_deadline_seconds"]
    if spent + deadline + 1800 > 21600:
        raise RuntimeError("complete GPU deadline does not fit cumulative allowance and reserve")
    if shutil.disk_usage(PACKAGE).free < 15 * GIB or sum(p.stat().st_size for p in PACKAGE.rglob("*") if p.is_file()) > 1536 * 1024**2:
        raise RuntimeError("GPU evidence disk reserve unavailable")
    lock, token = PACKAGE / "native_worker.lock", uuid.uuid4().hex
    with lock.open("x", encoding="utf-8") as f:
        json.dump({"token": token, "parent_pid": os.getpid(), "out": str(out), "lane": "GPU_FP32"}, f)
    child, job, assigned = None, None, False
    started = time.monotonic()
    receipt = {"status": "FAIL_BEFORE_WORKER", "lane": "BULK_GPU_FP32_SCREEN", "native_worker_seconds": 0,
               "started_utc": datetime.now(timezone.utc).isoformat(), "peak_rss_bytes": None,
               "run": str(out), "gpu_initial_inventory": inventory}
    try:
        out.mkdir(parents=True, exist_ok=False)
        (out / "source_manifest.json").write_text(json.dumps({"head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain=v1"), "sha256": sources()}, indent=2) + "\n")
        (out / "admission.json").write_bytes(admission_path.read_bytes())
        (out / "review.json").write_bytes(review_path.read_bytes())
        release = out / "release.json"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "HF_HOME": str(PACKAGE / "gpu_runtime_cache"), "HF_MODULES_CACHE": str(PACKAGE / "gpu_runtime_cache/modules"),
               "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        command = [sys.executable, "-B", "-m", "experiments.rineya_taps_adaptive_recurrence_campaign_v2.bulk_entry",
                   "--release", str(release), "--source", str(source), "--out", str(out / "result"),
                   "--admission", str(admission_path)]
        receipt["command"] = command
        job = WindowsJob()
        with (out / "stdout.log").open("w", encoding="utf-8") as stdout, (out / "stderr.log").open("w", encoding="utf-8") as stderr:
            child_started = time.monotonic()
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x4)
            receipt["worker_pid"] = child.pid
            job.assign(child)
            assigned = True
            job.resume_suspended(child)
            append(ledger, {"event": "native_started", "lane": "BULK_GPU_FP32_SCREEN", "run": str(out),
                            "reserved_seconds": deadline, "worker_pid": child.pid,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat()})
            release.write_text(json.dumps({"contained": True, "token": token}), encoding="utf-8")
            last_progress, last_sizes, last_gpu = time.monotonic(), None, 0
            while child.poll() is None:
                now = time.monotonic()
                try:
                    sample = resource_sample(child)
                except psutil.NoSuchProcess:
                    if child.poll() is not None:
                        break
                    raise
                sample["worker_seconds"] = now - child_started
                if now - last_gpu >= 10:
                    inventory, last_gpu = gpu_inventory(), now
                    sample["gpu_global_inventory"] = inventory
                append(out / "resources.jsonl", sample)
                receipt["peak_rss_bytes"] = max(receipt["peak_rss_bytes"] or 0, sample["rss_bytes"])
                paths = [out / "result" / name for name in ("preflight.json", "registration.json", "qualification.json", "outcomes.jsonl", "trajectories.jsonl", "forwards.jsonl", "qualification_journal.jsonl")]
                sizes = tuple((p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in paths)
                if sizes != last_sizes:
                    last_sizes, last_progress = sizes, now
                reason = ("HARD_DEADLINE" if now - child_started > deadline else
                          "HUNG_FORWARD_DEADLINE" if now - last_progress > 950 else
                          "RSS_CEILING" if sample["rss_bytes"] > 6 * GIB else
                          "HOST_RESERVE" if sample["host_available_bytes"] < GIB else
                          "GPU_GLOBAL_RESERVE" if inventory["free_bytes"] < GIB else
                          "GPU_IDENTITY_CHANGED" if inventory["uuid"] != admission["gpu_uuid"] else
                          "DISK_RESERVE" if shutil.disk_usage(PACKAGE).free < 15 * GIB or sum(p.stat().st_size for p in PACKAGE.rglob("*") if p.is_file()) > 1536 * 1024**2 else None)
                if reason:
                    receipt.update(status="SUPERVISOR_SAFETY_STOP", stop_reason=reason)
                    job.terminate()
                    break
                time.sleep(1)
            child.wait(timeout=15)
            receipt.update(exit_code=child.returncode, native_worker_seconds=time.monotonic() - child_started)
            if receipt["status"] != "SUPERVISOR_SAFETY_STOP":
                receipt["status"] = "WORKER_TERMINAL" if child.returncode == 0 else "WORKER_NONZERO"
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        if child is not None:
            if assigned:
                job.terminate()
            else:
                child.kill()
            child.wait(timeout=15)
            receipt["native_worker_seconds"] = time.monotonic() - child_started
    finally:
        if job is not None:
            job.close()
        receipt.update(supervisor_wall_seconds=time.monotonic() - started, ended_utc=datetime.now(timezone.utc).isoformat())
        if out.is_dir():
            (out / "terminal_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        append(ledger, {"event": "native_completed", **receipt})
        if json.loads(lock.read_text())["token"] == token:
            lock.unlink()
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "WORKER_TERMINAL" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source, args.out, args.admission))
