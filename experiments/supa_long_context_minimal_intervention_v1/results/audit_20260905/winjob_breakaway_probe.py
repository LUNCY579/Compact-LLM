"""Finite Windows lifecycle probe; never imports model or GPU code.

Modes: audit (read-only), launch (one finite probe), child, observe (read-only
process inspection plus receipts). Every output stays beside this source.
"""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve()
AUDIT = SOURCE.parent
ROOT = AUDIT.parents[3]
PYTHON = ROOT.parent / "Chat-Bubble" / ".venv" / "Scripts" / "python.exe"
OUT = AUDIT / "winjob_breakaway_probe_artifacts"
SECONDS = 20
BREAKAWAY_OK = 0x0800
KILL_ON_JOB_CLOSE = 0x2000


class BasicLimit(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
        ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t),
        ("max_ws", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
        ("scheduling", wintypes.DWORD),
    ]


class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes",
        "other_bytes")]


class ExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("basic", BasicLimit), ("io", IoCounters),
        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.GetCurrentProcess.argtypes = []
kernel.GetCurrentProcess.restype = wintypes.HANDLE
kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
                                ctypes.POINTER(wintypes.BOOL)]
kernel.IsProcessInJob.restype = wintypes.BOOL
kernel.QueryInformationJobObject.argtypes = [
    wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD)]
kernel.QueryInformationJobObject.restype = wintypes.BOOL
kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
    ctypes.POINTER(wintypes.FILETIME)] * 4
kernel.GetProcessTimes.restype = wintypes.BOOL
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL
kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel.WaitForSingleObject.restype = wintypes.DWORD
kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE,
                                    ctypes.POINTER(wintypes.DWORD)]
kernel.GetExitCodeProcess.restype = wintypes.BOOL


def checked(ok):
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())


def now():
    return datetime.now(timezone.utc).isoformat()


def save(name, data):
    target = OUT / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def identity(handle, pid):
    times = [wintypes.FILETIME() for _ in range(4)]
    checked(kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)))
    ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    return {"pid": pid, "creation_filetime_100ns": str(ticks),
            "creation_utc": datetime.fromtimestamp(
                (ticks - 116444736000000000) / 10000000,
                timezone.utc).isoformat()}


def in_job(handle):
    assigned = wintypes.BOOL()
    checked(kernel.IsProcessInJob(handle, None, ctypes.byref(assigned)))
    return bool(assigned.value)


def self_audit():
    handle = kernel.GetCurrentProcess()
    result = {"timestamp_utc": now(), "identity": identity(handle, os.getpid()),
              "parent_pid": os.getppid(), "executable": sys.executable,
              "cwd": str(Path.cwd()), "in_any_job": in_job(handle)}
    if result["in_any_job"]:
        limits, length = ExtendedLimit(), wintypes.DWORD()
        checked(kernel.QueryInformationJobObject(
            None, 9, ctypes.byref(limits), ctypes.sizeof(limits),
            ctypes.byref(length)))
        if length.value != ctypes.sizeof(limits):
            raise RuntimeError("Unexpected extended-limit structure size")
        flags = limits.basic.flags
        result["immediate_job"] = {
            "flags_decimal": flags, "flags_hex": f"0x{flags:08X}",
            "breakaway_ok": bool(flags & BREAKAWAY_OK),
            "silent_breakaway_ok": bool(flags & 0x1000),
            "kill_on_job_close": bool(flags & KILL_ON_JOB_CLOSE),
            "returned_bytes": length.value,
        }
    return result


def child():
    start = self_audit()
    start["finite_duration_seconds"] = SECONDS
    save("child_start.json", start)
    print(json.dumps({"event": "start", **start}), flush=True)
    began = time.monotonic()
    with (OUT / "heartbeats.jsonl").open("x", encoding="utf-8") as heartbeat:
        for sequence in range(SECONDS + 1):
            remaining = began + sequence - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            item = {"sequence": sequence, "timestamp_utc": now(),
                    "identity": start["identity"],
                    "elapsed_seconds": round(time.monotonic() - began, 6)}
            heartbeat.write(json.dumps(item) + "\n")
            heartbeat.flush()
            os.fsync(heartbeat.fileno())
            save("latest_heartbeat.json", item)
    completed = {"timestamp_utc": now(), "identity": start["identity"],
                 "elapsed_seconds": round(time.monotonic() - began, 6),
                 "heartbeat_count": SECONDS + 1, "completed": True}
    save("child_completion.json", completed)
    print(json.dumps({"event": "completed", **completed}), flush=True)


def launch():
    parent = self_audit()
    if not parent["in_any_job"] or not parent["immediate_job"]["breakaway_ok"]:
        raise RuntimeError("Probe requires observed immediate BREAKAWAY_OK")
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Probe must run from repository root")
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("Probe requires the specified Chat-Bubble venv Python")
    OUT.mkdir(exist_ok=True)
    with (OUT / "launch_once.json").open("x", encoding="utf-8") as stream:
        json.dump({"timestamp_utc": now(), "parent": parent}, stream, indent=2)
    flags = subprocess.CREATE_BREAKAWAY_FROM_JOB | subprocess.CREATE_NO_WINDOW
    command = [str(PYTHON), str(SOURCE), "child"]
    # Explicit standard handles only; no shell or inherited Codex pipes/handles.
    with (OUT / "child.stdout.log").open("xb") as stdout, \
         (OUT / "child.stderr.log").open("xb") as stderr:
        process = subprocess.Popen(
            command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, close_fds=True,
            creationflags=flags, shell=False)
    receipt = {"timestamp_utc": now(), "parent": parent,
               "popen_pid": process.pid, "command": command,
               "creation_flags_hex": f"0x{flags:08X}",
               "close_fds": True, "stdin": "DEVNULL",
               "stdout": "child.stdout.log", "stderr": "child.stderr.log"}
    deadline = time.monotonic() + 3
    while not (OUT / "child_start.json").exists() and time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Probe child exited early: {process.returncode}")
        time.sleep(0.05)
    if not (OUT / "child_start.json").exists():
        raise RuntimeError("Probe child produced no startup receipt")
    receipt["child_start"] = read("child_start.json")
    receipt["parent_returning_utc"] = now()
    save("launch_receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def observe():
    start = read("child_start.json")
    pid = start["identity"]["pid"]
    # Read-only query plus SYNCHRONIZE; never requests terminate/set rights.
    handle = kernel.OpenProcess(0x1000 | 0x00100000, False, pid)
    checked(handle)
    try:
        actual = identity(handle, pid)
        if actual != start["identity"]:
            raise RuntimeError("PID identity changed; refusing further inspection")
        initial_wait = kernel.WaitForSingleObject(handle, 0)
        if initial_wait == 0xFFFFFFFF:
            checked(False)
        observation = {
            "timestamp_utc": now(), "identity": actual,
            "inspection_context": "separate tool invocation after launcher returned",
            "alive_after_tool_return": initial_wait == 258,
            "in_any_job": in_job(handle), "latest_heartbeat": read("latest_heartbeat.json"),
        }
        save("observed_after_return.json", observation)
        print(json.dumps(observation, indent=2), flush=True)
        waited = kernel.WaitForSingleObject(handle, 25000)
        if waited == 0xFFFFFFFF:
            checked(False)
        if waited != 0:
            raise RuntimeError("Probe did not finish within bounded observation")
        exit_code = wintypes.DWORD()
        checked(kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)))
        final = {"timestamp_utc": now(), "observation": observation,
                 "exit_code": exit_code.value, "completion": read("child_completion.json"),
                 "stderr_bytes": (OUT / "child.stderr.log").stat().st_size}
        save("final_receipt.json", final)
        print(json.dumps(final, indent=2), flush=True)
    finally:
        checked(kernel.CloseHandle(handle))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("audit", "launch", "child", "observe"):
        raise SystemExit("Use audit, launch, child, or observe")
    if sys.argv[1] == "audit":
        print(json.dumps(self_audit(), indent=2))
    else:
        # An exception or access denial exits immediately; no fallback path.
        globals()[sys.argv[1]]()
