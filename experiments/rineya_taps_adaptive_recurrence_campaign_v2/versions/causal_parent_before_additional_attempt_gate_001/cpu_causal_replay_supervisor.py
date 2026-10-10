"""CPU causal replay parent: exact CPU floors, contained worker, measured deadline."""
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

GIB = 1024**3


def progress_stamp(out):
    paths = [out / "result" / name for name in (
        "preflight.json", "registration.json", "population.json", "qualification.json",
        "attempts.jsonl", "outcomes.jsonl", "receipt.json")]
    return tuple((p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in paths)


def append(path, value):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run(source, out, admission_path):
    source, out = source.resolve(), out.resolve()
    if PACKAGE.resolve() not in out.parents:
        raise ValueError("output must be campaign-local")
    if out.exists():
        raise FileExistsError("CPU causal replay output must be new; existing evidence is immutable")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission["status"] != "READY_FOR_CPU_CAUSAL_REPLAY_DISPATCH":
        raise RuntimeError("admission not ready")
    if str(source) != admission["source"] or sys.executable != admission["interpreter"]:
        raise RuntimeError("source/interpreter differ from reviewed binding")
    for relative, expected in admission["dispatch_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("reviewed dispatch source changed: " + relative)
    review = json.loads((PACKAGE / admission["review_receipt"]).read_text(encoding="utf-8"))
    if review["status"] != "PASS_CPU_CAUSAL_REPLAY_DISPATCH" or review["admission_sha256"] != sha(admission_path):
        raise RuntimeError("independent review not bound to this admission")
    if psutil.virtual_memory().available < 8.4 * GIB:
        raise RuntimeError("parent admission requires historical7.4GiB floor plus1GiB import reserve; no worker launched")
    if shutil.disk_usage(PACKAGE).free < 15 * GIB:
        raise RuntimeError("disk floor unavailable")
    ledger = PACKAGE / "BUDGET_LEDGER.jsonl"
    events = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()] if ledger.exists() else []
    spent = sum(x.get("native_worker_seconds", 0) for x in events if x.get("event") == "native_completed")
    deadline = admission["hard_deadline_seconds"]
    if spent + deadline + 1800 > 21600:
        raise RuntimeError("complete matrix deadline exceeds remaining native allowance/reserve")
    if sum(p.stat().st_size for p in PACKAGE.rglob("*") if p.is_file()) > 1536 * 1024**2:
        raise RuntimeError("campaign finalization disk reserve unavailable")
    lock = PACKAGE / "native_worker.lock"
    token = uuid.uuid4().hex
    with lock.open("x", encoding="utf-8") as f:
        json.dump({"token": token, "parent_pid": os.getpid(), "out": str(out)}, f)
    child, job, assigned, created_out = None, None, False, False
    started = time.monotonic()
    receipt = {"status": "FAIL_BEFORE_WORKER", "lane": "CPU_CAUSAL_REPLAY_FP32", "native_worker_seconds": 0, "started_utc": datetime.now(timezone.utc).isoformat(), "peak_rss_bytes": None, "run": str(out)}
    try:
        out.mkdir(parents=True, exist_ok=False)
        created_out = True
        (out / "source_manifest.json").write_text(json.dumps({"head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain=v1"), "sha256": sources()}, indent=2) + "\n")
        (out / "admission.json").write_bytes(admission_path.read_bytes())
        (out / "review.json").write_bytes((PACKAGE / admission["review_receipt"]).read_bytes())
        release = out / "release.json"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": str(PACKAGE / "runtime_cache"), "HF_MODULES_CACHE": str(PACKAGE / "runtime_cache/modules"), "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "", "CPU_CAUSAL_REPLAY_PARENT_STARTED_PERF_COUNTER": str(time.perf_counter())}
        command = [sys.executable, "-B", "-m", "experiments.rineya_taps_adaptive_recurrence_campaign_v2.cpu_causal_replay_entry", "--release", str(release), "--source", str(source), "--out", str(out / "result"), "--admission", str(admission_path)]
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
            append(ledger, {"event": "native_started", "lane": "CPU_CAUSAL_REPLAY_FP32", "run": str(out), "reserved_seconds": deadline, "worker_pid": child.pid, "timestamp_utc": datetime.now(timezone.utc).isoformat()})
            release.write_text(json.dumps({"contained": True, "token": token}), encoding="utf-8")
            last_progress, last_sizes = time.monotonic(), None
            while child.poll() is None:
                now = time.monotonic()
                try:
                    sample = resource_sample(child)
                except psutil.NoSuchProcess:
                    if child.poll() is not None:
                        break
                    raise
                sample["worker_seconds"] = now - child_started
                append(out / "resources.jsonl", sample)
                receipt["peak_rss_bytes"] = max(receipt["peak_rss_bytes"] or 0, sample["rss_bytes"])
                progress_paths = [out / "result" / name for name in ("preflight.json", "registration.json", "population.json", "qualification.json", "attempts.jsonl", "outcomes.jsonl", "receipt.json")]
                sizes = tuple((p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in progress_paths)
                if sizes != last_sizes:
                    last_sizes, last_progress = sizes, now
                campaign_bytes = sum(p.stat().st_size for p in PACKAGE.rglob("*") if p.is_file())
                reason = ("HARD_DEADLINE" if now - child_started > deadline else
                          "HUNG_FORWARD_DEADLINE" if now - last_progress > 950 else
                          "RSS_CEILING" if sample["rss_bytes"] > 6 * GIB else
                          "HOST_RESERVE" if sample["host_available_bytes"] < GIB else
                          "DISK_RESERVE" if campaign_bytes > 1536 * 1024**2 or shutil.disk_usage(PACKAGE).free < 15 * GIB else None)
                if reason:
                    receipt.update(status="SUPERVISOR_SAFETY_STOP", stop_reason=reason)
                    job.terminate()
                    break
                time.sleep(1)
            child.wait(timeout=15)
            receipt["exit_code"] = child.returncode
            receipt["native_worker_seconds"] = time.monotonic() - child_started
            if receipt["status"] != "SUPERVISOR_SAFETY_STOP":
                receipt["status"] = "WORKER_TERMINAL" if child.returncode == 0 else "WORKER_NONZERO"
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        if child is not None:
            if job is not None and assigned:
                job.terminate()
            else:
                child.kill()
            child.wait(timeout=15)
            receipt["native_worker_seconds"] = time.monotonic() - child_started
    finally:
        if job is not None:
            job.close()
        receipt.update(supervisor_wall_seconds=time.monotonic() - started, ended_utc=datetime.now(timezone.utc).isoformat())
        if created_out:
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
