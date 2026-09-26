"""Execution layer for TRISKILL-001 Phase 0: wait for host RAM and a free lease, then run the
registered screen (screen_source.main) unchanged. Launch through WMI; wait on PHASE0_RC.json.

The threshold is a LAUNCH condition (two reads 10 s apart), not a measurement. One 2.4 GB FP32
model is resident at a time; ~3.5 GB host was measured for this shape of job in the soup lane.
"""
from __future__ import annotations

import json, time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
LOCK = Path(r"C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock")
NEED_GIB = 4.5
MAX_WAIT_S = 4 * 3600
POLL_S = 60


def avail_gib() -> float:
    return psutil.virtual_memory().available / 2 ** 30


def main() -> int:
    t0 = time.time()
    while True:
        a1 = avail_gib()
        if a1 >= NEED_GIB and not LOCK.exists():
            time.sleep(10)
            a2 = avail_gib()
            if a2 >= NEED_GIB and not LOCK.exists():
                print(f"[run_phase0] launching: RAM {a1:.2f}/{a2:.2f} GiB, lease free, waited {time.time()-t0:.0f}s", flush=True)
                break
        if time.time() - t0 > MAX_WAIT_S:
            print(f"[run_phase0] REFUSING after {MAX_WAIT_S}s: RAM {a1:.2f} GiB, lock={'held' if LOCK.exists() else 'free'}", flush=True)
            return 3
        time.sleep(POLL_S)
    import screen_source
    return screen_source.main()


if __name__ == "__main__":
    raise SystemExit(main())
