"""Wait without a GPU lease for Q1's bound native-loader resource window."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVID = HERE / "evidence"
LOG = EVID / "Q1_NATIVE_RUN.log"
STATUS = EVID / "Q1_NATIVE_STATUS.json"
RESULT = EVID / "Q1_NATIVE_LOADER.json"
LOCK = Path(os.environ["TEMP"]) / "rineya-gpu-0.lock"
MAX_WAIT_S = 4 * 3600
POLL_S = 60


def free_vram_mib() -> int | None:
    try:
        run = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        values = [int(line.strip()) for line in run.stdout.splitlines() if line.strip()]
        return values[0] if run.returncode == 0 and len(values) == 1 else None
    except Exception:
        return None


def reading(log) -> bool:
    ram = psutil.virtual_memory().available / 2**30
    disk = shutil.disk_usage("C:/").free / 2**30
    vram = free_vram_mib()
    free_lease = not LOCK.exists()
    log.write(f"[wait] RAM={ram:.2f} GiB disk={disk:.2f} GiB VRAM={vram} MiB lease_free={free_lease}\n")
    log.flush()
    return ram >= 5.5 and disk >= 6.0 and vram is not None and vram >= 4096 and free_lease


def main() -> int:
    if LOG.exists() or STATUS.exists() or RESULT.exists():
        raise SystemExit("Q1 native log/status/result already exists (create-only)")
    start = time.monotonic()
    EVID.mkdir(parents=True, exist_ok=True)
    with LOG.open("x", encoding="utf-8") as log:
        while time.monotonic() - start < MAX_WAIT_S:
            if reading(log):
                time.sleep(10)
                if reading(log):
                    cmd = [sys.executable, "-u", "-m", "experiments.maxrl_comp002_pretraining_v1.q1_native_loader",
                           "--asset-receipt", str(EVID / "Q1_ASSET_CANDIDATE.json"),
                           "--tokenizer-receipt", str(EVID / "Q1_TOKENIZER_PROMPT_LENGTHS.json"),
                           "--out", str(RESULT)]
                    log.write("[wait] launch " + " ".join(cmd) + "\n")
                    log.flush()
                    rc = subprocess.call(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
                    log.write(f"[wait] child exit rc={rc} result_exists={RESULT.exists()}\n")
                    log.flush()
                    if rc == 0 and RESULT.exists():
                        status = "Q1_NATIVE_LOADER_PASS"
                        break
                    if rc == 3 and not RESULT.exists():
                        time.sleep(POLL_S)
                        continue
                    status = "Q1_NATIVE_LOADER_FAILED"
                    break
            time.sleep(POLL_S)
        else:
            rc = 3
            status = "NOT_STARTED_WAIT_TIMEOUT"
        summary = {"status": status, "child_rc": rc, "result_exists": RESULT.exists(),
                   "elapsed_seconds": round(time.monotonic() - start),
                   "model_loads_if_timeout": "inspect Q1_NATIVE_RUN.log; timeout alone is not a native failure"}
        with STATUS.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
        log.write("[wait] final " + json.dumps(summary) + "\n")
        log.flush()
    return 0 if status == "Q1_NATIVE_LOADER_PASS" else (3 if status == "NOT_STARTED_WAIT_TIMEOUT" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
