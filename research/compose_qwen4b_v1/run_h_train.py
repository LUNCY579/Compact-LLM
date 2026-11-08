"""COMPOSE4B-001: train delta_H on Qwen3-4B under the shared GPU lease.

Gates are checked before the lease is taken and again immediately before dispatch.
The lease is created create-only and released only if our own token still holds it,
matching experiments/model_quality_gpu_v1/resources.py:lease. Never deletes another
process's lock, including a suspected stale one.
"""
from pathlib import Path
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
HERE = Path(__file__).resolve().parent
PY = BASE / "Chat-Bubble/.venv/Scripts/python.exe"
MODEL = "C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
H = RECORDS / "compose001-20260918/H"
LOCK = Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"
FLOOR = 15032385536          # 14 GiB, unchanged programme floor
GPU_IDLE_MIB = 1500          # QWENFT001's wait_gpu threshold


def gpu_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=30)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def gates():
    free = shutil.disk_usage(RECORDS).free
    used = gpu_used_mib()
    return {
        "free_disk_bytes": free,
        "free_disk_gib": round(free / 2**30, 3),
        "disk_floor_ok": free >= FLOOR,
        "gpu_used_mib": used,
        "gpu_idle_ok": used is not None and used < GPU_IDLE_MIB,
        "lease_present": LOCK.exists() or LOCK.is_symlink(),
    }


def require(snapshot):
    if not snapshot["disk_floor_ok"]:
        raise SystemExit(f"REFUSED: free disk {snapshot['free_disk_gib']} GiB below the 14 GiB floor")
    if not snapshot["gpu_idle_ok"]:
        raise SystemExit(f"REFUSED: GPU busy ({snapshot['gpu_used_mib']} MiB used); not contending")
    return snapshot


def main():
    if OUT.exists():
        raise SystemExit("create-only records root already exists")
    before = require(gates())
    if before["lease_present"]:
        raise SystemExit("REFUSED: shared GPU lease is held by another process")

    token = uuid.uuid4().hex
    with LOCK.open("x", encoding="utf-8", newline="\n") as f:
        json.dump({"pid": os.getpid(), "token": token, "created": time.time()}, f)
    try:
        after = require(gates())          # revalidate immediately before dispatch
        OUT.mkdir(parents=True)
        shutil.copyfile(HERE / "GRANT.json", OUT / "GRANT.json")
        shutil.copyfile(HERE / "PREREG.md", OUT / "PREREG.md")
        (OUT / "GATES.json").write_text(
            json.dumps({"before_lease": before, "after_lease": after,
                        "floor_bytes": FLOOR, "model": MODEL,
                        "lowered_any_gate": False}, indent=2) + "\n", encoding="utf-8")
        cmd = [str(PY), str(HERE / "train_qwen_lora.py"),
               "--model", MODEL,
               "--prepared", str(H / "H-prepared"),
               "--plan", str(H / "H-PLAN.json"),
               "--out", str(OUT / "run-h"),
               "--rank", "16", "--lr", "1e-4", "--four-bit"]
        (OUT / "COMMAND.json").write_text(json.dumps({"argv": cmd}, indent=2) + "\n", encoding="utf-8")
        print("DISPATCH", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=str(HERE)).returncode
        print("returncode", rc, flush=True)
        return rc
    finally:
        if LOCK.exists():
            try:
                if json.loads(LOCK.read_text(encoding="utf-8")).get("token") == token:
                    LOCK.unlink()
                    print("lease released", flush=True)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
