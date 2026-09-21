"""COMPOSE4B-001: generate and score the composition arms under the shared GPU lease.

Smoke-tests each adapter on a few rows first, because TH_s is rank 32 (the exact
concatenated representation of delta_T + delta_H) and a loader that silently refuses a
non-16 rank would otherwise be discovered an hour into a full battery.

One lease acquisition covers the whole battery: the lease exists to serialize GPU work,
and dropping/retaking it between arms would let another job interleave mid-experiment.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

BASE = Path("C:/Users/clclc/BusinessProject")
RECORDS = BASE / "rineya-dataset-retention-records"
OUT = RECORDS / "compose4b001-20260921"
Q = RECORDS / "qwen001-20260918"
PY = BASE / "Chat-Bubble/.venv/Scripts/python.exe"
MODEL = "C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
LOCK = Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"
FLOOR = 15032385536

ARMS = [
    ("th_s",      OUT / "arm-th-s/adapter"),
    ("scaled_t",  OUT / "arm-scaled-t/adapter"),
    ("h_only",    OUT / "run-h/adapter"),
]


def publish_worker_pids(root_pid):
    """Record the pids that actually hold the card so a staleness rule cannot misjudge.

    The lock's own pid is a shim three levels above the CUDA process. A peer's
    "holder pid dead => free" rule read that shim and would have reclaimed a live lease.
    """
    try:
        import psutil
        pids = [root_pid] + [c.pid for c in psutil.Process(root_pid).children(recursive=True)]
        d = json.loads(LOCK.read_text(encoding="utf-8"))
        d["worker_pids"] = pids
        d["worker_pids_updated"] = time.time()
        tmp = LOCK.with_suffix(".lock.tmp")
        tmp.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, LOCK)
    except Exception as exc:
        print(f"  (could not publish worker pids: {exc})", flush=True)


def gpu_used_mib():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=30).strip().splitlines()[0])
    except Exception:
        return None


def require():
    free = shutil.disk_usage(RECORDS).free
    if free < FLOOR:
        raise SystemExit(f"REFUSED: free disk {free/2**30:.3f} GiB below the 14 GiB floor")
    return {"free_disk_gib": round(free / 2**30, 3), "gpu_used_mib": gpu_used_mib()}


EXPECTED_ROWS = 384


def complete(out):
    """A generation is complete only if its predictions file has every row."""
    f = out / "predictions.jsonl"
    if not f.is_file():
        return False
    n = sum(1 for l in f.open(encoding="utf-8") if l.strip())
    return n >= EXPECTED_ROWS


def generate(label, adapter, rows=None):
    out = OUT / f"gen-{label}" if rows is None else OUT / f"smoke-{label}"
    if rows is None and complete(out):
        print(f"  [{label}] already complete ({EXPECTED_ROWS} rows) - skipping", flush=True)
        return out
    if out.exists():
        shutil.rmtree(out)
    cmd = [str(PY), str(Q / "stage2_generate.py"), "--model", MODEL, "--label", label,
           "--prompts", str(Q / "render-process/prompts.jsonl"), "--out", str(out),
           "--max-new-tokens", "96", "--dtype", "bf16", "--four-bit",
           "--adapter", str(adapter)]
    if rows:
        cmd += ["--rows", str(rows)]
    t0 = time.time()
    # Stream instead of buffering: stage2_generate prints per-row progress, and
    # capture_output=True hid it until exit, leaving a 12-minute arm unsupervisable.
    # "90% done" and "hung" were indistinguishable from inside this process.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    publish_worker_pids(proc.pid)
    captured = []
    for line in proc.stdout:
        captured.append(line)
        s = line.rstrip()
        if s.startswith("  ") and "/" in s[:14] and "it/s" not in s:
            print(f"    {s.strip()}", flush=True)
    proc.wait()
    rc = type("R", (), {"returncode": proc.returncode,
                        "stdout": "".join(captured), "stderr": ""})()
    tail = "\n".join(l for l in (rc.stdout or "").splitlines()
                     if not any(x in l for x in ("Loading weights", "it/s]", "triton")))[-1500:]
    print(f"  [{label}{' smoke' if rows else ''}] rc={rc.returncode} {time.time()-t0:.0f}s")
    if rc.returncode:
        print((rc.stderr or "")[-2000:])
        raise SystemExit(f"generation failed for {label}")
    print(tail[-600:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true")
    a = ap.parse_args()

    require()
    if LOCK.exists() or LOCK.is_symlink():
        raise SystemExit("REFUSED: shared GPU lease is held by another process")
    token = uuid.uuid4().hex
    with LOCK.open("x", encoding="utf-8", newline="\n") as f:
        json.dump({
            "pid": os.getpid(), "token": token, "created": time.time(),
            "supervisor_pid": os.getpid(), "worker_pids": [],
            # A detached worker outlives its supervisor BY DESIGN (that is what
            # CREATE_BREAKAWAY_FROM_JOB is for), so the supervisor pid is NOT a liveness
            # signal and a "holder pid dead => stale" rule would steal a LIVE lease.
            "staleness_rule": ("STALE only if EVERY worker_pid is absent from BOTH "
                               "`nvidia-smi --query-compute-apps=pid` AND the process table. "
                               "supervisor_pid is a shim and is NOT a liveness signal."),
            "grace_note": ("empty worker_pids means the worker has not reached its first CUDA "
                           "allocation yet; treat the lease as LIVE"),
            "job": "COMPOSE4B-001 battery",
        }, f)
    try:
        print("=== SMOKE (8 rows each) ===", flush=True)
        for label, adapter in ARMS:
            if not (adapter / "adapter_model.safetensors").is_file():
                raise SystemExit(f"missing adapter for {label}: {adapter}")
            if complete(OUT / f"gen-{label}"):
                print(f"  [{label}] full output present - smoke not needed", flush=True)
                continue
            generate(label, adapter, rows=8)
        if a.smoke_only:
            print("smoke only; stopping before the full battery")
            return 0
        print("\n=== FULL BATTERY (384 rows each) ===", flush=True)
        for label, adapter in ARMS:
            require()
            generate(label, adapter)
        print("\n=== SCORING ===", flush=True)
        (OUT / "scored").mkdir(exist_ok=True)
        for label, _ in ARMS:
            cmd = [str(PY), str(Q / "stage3_score.py"), "--kind", "process",
                   "--render", str(Q / "render-process"), "--gen", str(OUT / f"gen-{label}"),
                   "--out", str(OUT / "scored" / f"{label}.json")]
            if not complete(OUT / f"gen-{label}"):
                print(f"  scored {label}: SKIPPED, generation incomplete")
                continue
            rc = subprocess.run(cmd, capture_output=True, text=True)
            print(f"  scored {label}: rc={rc.returncode} {(rc.stdout or '').strip()[-300:]}")
            if rc.returncode:
                print((rc.stderr or "")[-1500:])
        return 0
    finally:
        if LOCK.exists():
            try:
                if json.loads(LOCK.read_text(encoding="utf-8")).get("token") == token:
                    LOCK.unlink(); print("lease released", flush=True)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
