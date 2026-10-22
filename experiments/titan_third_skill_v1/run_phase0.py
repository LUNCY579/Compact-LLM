"""Execution layer for TRISKILL-001 Phase 0: wait for host RAM and a free lease, then run the
registered screen (screen_source.py) in a FRESH process. Launch through WMI; wait on PHASE0_RC.json.

Amendment 1 (ATTEMPT_LOG.md): attempt 1 died on a GPU driver reset (nvlddmkm 153), which kills
the process's CUDA context. Each attempt is now its own subprocess and resumes from completed
(arm, bank) units; at most MAX_ATTEMPTS. The RAM threshold is a LAUNCH condition (two reads 10 s
apart), not a measurement.
"""
from __future__ import annotations

import subprocess, sys, time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
LOCK = Path(r"C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock")
NEED_GIB = 4.5
MAX_WAIT_S = 4 * 3600
POLL_S = 60
MAX_ATTEMPTS = 3


def avail_gib() -> float:
    return psutil.virtual_memory().available / 2 ** 30


def wait_for_gates(t0: float) -> bool:
    while True:
        a1 = avail_gib()
        if a1 >= NEED_GIB and not LOCK.exists():
            time.sleep(10)
            a2 = avail_gib()
            if a2 >= NEED_GIB and not LOCK.exists():
                print(f"[run_phase0] gates met: RAM {a1:.2f}/{a2:.2f} GiB, lease free, waited {time.time()-t0:.0f}s", flush=True)
                return True
        if time.time() - t0 > MAX_WAIT_S:
            print(f"[run_phase0] REFUSING after {MAX_WAIT_S}s: RAM {a1:.2f} GiB, lock={'held' if LOCK.exists() else 'free'}", flush=True)
            return False
        time.sleep(POLL_S)


def main() -> int:
    t0 = time.time()
    rc = 1
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not wait_for_gates(t0):
            return 3
        print(f"[run_phase0] attempt {attempt}/{MAX_ATTEMPTS} (this runner; attempt 1 of the experiment is archived in attempt1/)", flush=True)
        rc = subprocess.call([sys.executable, "-u", str(HERE / "screen_source.py")], cwd=str(HERE))
        print(f"[run_phase0] attempt {attempt} rc={rc}", flush=True)
        if rc == 0:
            return 0
        time.sleep(60)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
