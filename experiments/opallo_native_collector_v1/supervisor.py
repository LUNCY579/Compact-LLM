"""Independent bounded subprocess supervisor. Only its own worker tree is killed."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .contract import Blocked, require, write_new

GIB = 1024**3


class WindowsJob:
    """Kill-on-close Windows Job Object, assigned before the model-load GO file."""
    def __init__(self, process):
        import ctypes as c
        from ctypes import wintypes as w
        class Basic(c.Structure):
            _fields_ = [("PerProcessUserTimeLimit", c.c_longlong), ("PerJobUserTimeLimit", c.c_longlong),
                        ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", c.c_size_t),
                        ("MaximumWorkingSetSize", c.c_size_t), ("ActiveProcessLimit", w.DWORD),
                        ("Affinity", c.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]
        class IO(c.Structure):
            _fields_ = [(n, c.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                       "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
        class Extended(c.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO),
                        ("ProcessMemoryLimit", c.c_size_t), ("JobMemoryLimit", c.c_size_t),
                        ("PeakProcessMemoryUsed", c.c_size_t), ("PeakJobMemoryUsed", c.c_size_t)]
        self.kernel = c.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        self.kernel.SetInformationJobObject.restype = w.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = w.BOOL
        self.kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.kernel.TerminateJobObject.restype = w.BOOL
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.CloseHandle.restype = w.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        require(bool(self.handle), "Windows Job Object creation failed")
        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE.
        try:
            require(bool(self.kernel.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits))),
                    "Windows kill-on-close setup failed")
            require(bool(self.kernel.AssignProcessToJobObject(self.handle, int(process._handle))),
                    "Windows worker job assignment failed; do not run uncontained")
        except BaseException:
            self.close()
            raise

    def kill(self):
        if self.handle:
            self.kernel.TerminateJobObject(self.handle, 2)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def contain_worker(parent_pid: int) -> None:
    """Linux parent-death containment; other POSIX platforms fail closed."""
    require(os.getppid() == parent_pid, "native worker parent identity mismatch")
    if sys.platform.startswith("linux"):
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        require(libc.prctl(1, signal.SIGKILL, 0, 0, 0) == 0, "Linux parent-death containment unavailable")
        require(os.getppid() == parent_pid, "supervisor died before containment")
    else:
        require(os.name == "nt", "only Linux/Windows worker containment implemented")


def memory_snapshot(process=None) -> tuple[float, int]:
    import psutil
    free = psutil.virtual_memory().available / GIB
    rss = 0
    if process is not None:
        try:
            root = psutil.Process(process.pid)
            for p in [root] + root.children(recursive=True):
                try:
                    rss += p.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass
    return free, rss


def run_bounded(command: list[str], cwd: Path, out: Path, *, seconds: float,
                min_free_gib: float, max_rss_gib: float, go: dict | None = None) -> dict:
    """Time/RSS/free-memory limits are enforced OUTSIDE the inference loop.

    Smaller test limits are internal test arguments, never native CLI overrides.
    """
    require(seconds > 0 and min_free_gib >= 0 and max_rss_gib > 0, "invalid watchdog limits")
    require(sys.platform.startswith("linux") or os.name == "nt", "unsupported process containment platform")
    require(memory_snapshot()[0] >= min_free_gib, "host memory floor blocks model launch")
    out.mkdir(parents=True, exist_ok=True)
    process = job = None
    peak, started, reason, code = 0, time.monotonic(), None, None
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    try:
        with (out / "stdout.log").open("xb") as stdout, (out / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=str(cwd), stdout=stdout, stderr=stderr, env=env,
                                       start_new_session=os.name != "nt",
                                       creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
            if os.name == "nt":
                job = WindowsJob(process)
            if go is not None:
                write_new(out / "GO.json", go)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                free, rss = memory_snapshot(process)
                peak = max(peak, rss)
                if elapsed >= seconds:
                    reason = "WALL_TIME_LIMIT"
                elif free < min_free_gib:
                    reason = "HOST_MEMORY_FLOOR"
                elif rss > max_rss_gib * GIB:
                    reason = "PROCESS_TREE_RSS_LIMIT"
                elif stdout.tell() + stderr.tell() > 16 * 1024**2:
                    reason = "LOG_SIZE_LIMIT"
                if reason:
                    break
                time.sleep(min(0.1, max(0.001, seconds - elapsed)))
            if reason:
                if job:
                    job.kill()
                elif process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
            code = process.wait(timeout=10)
            if code != 0 and reason is None:
                reason = "WORKER_NONZERO_EXIT"
    except BaseException as exc:
        reason = reason or f"SUPERVISOR_ERROR:{type(exc).__name__}"
        raise
    finally:
        if process is not None:
            if job:
                job.close()  # Also prevents surviving Windows descendants after normal exit.
            elif os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        receipt = {"status": "PASS_PROCESS_ONLY" if reason is None and code == 0 else "BLOCKED",
                   "reason": reason, "exit_code": code, "elapsed_seconds": time.monotonic() - started,
                   "peak_observed_process_tree_rss_bytes": peak, "limit_seconds": seconds,
                   "memory_poll_seconds": 0.1, "native_qualification": False}
        write_new(out / "SUPERVISOR.json", receipt)
    require(receipt["status"] == "PASS_PROCESS_ONLY", f"native capture blocked: {reason}")
    return receipt
