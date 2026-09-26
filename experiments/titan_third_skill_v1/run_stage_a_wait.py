"""Wait for the registered Stage A resource gates, then launch its unchanged runner.

This wrapper makes a detached start safe when free RAM hovers around the 5 GiB
floor. It never opens a score bank or changes an experiment rule.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = Path(r"C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock")
RAM_GIB = 5.0
DISK_GIB = 6.0
POLL_SECONDS = 60
MAX_WAIT_SECONDS = 4 * 3600


def available_ram_gib() -> float:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.ullAvailPhys / 2**30


def gates_met() -> bool:
    ram = available_ram_gib()
    disk = shutil.disk_usage("C:/").free / 2**30
    free_lease = not LOCK.exists()
    print(f"[stage-a-wait] RAM {ram:.2f} GiB, disk {disk:.2f} GiB, "
          f"lease {'free' if free_lease else 'held'}", flush=True)
    return ram >= RAM_GIB and disk >= DISK_GIB and free_lease


def main() -> int:
    start = time.monotonic()
    if (HERE / "STAGE_A_RESULT.json").exists():
        raise SystemExit("refusing launch: Stage A result already exists")
    if (HERE / "stage_a").exists():
        print("[stage-a-wait] resuming registered runner from completed units", flush=True)
    while time.monotonic() - start < MAX_WAIT_SECONDS:
        if gates_met():
            time.sleep(10)
            if gates_met():
                print("[stage-a-wait] gates met twice; launching registered runner", flush=True)
                return subprocess.call([sys.executable, "-u", str(HERE / "run_stage_a.py")], cwd=str(HERE))
        time.sleep(POLL_SECONDS)
    print("[stage-a-wait] timeout; Stage A never launched", flush=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
