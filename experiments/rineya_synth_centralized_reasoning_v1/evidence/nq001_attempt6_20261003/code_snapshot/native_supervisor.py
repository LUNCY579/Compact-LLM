"""External, framework-free Windows containment for mechanical qualification.

Nothing is launched at import. ``supervise`` accepts only the immutable scope
validated by native_contract.preflight; this is never a study-training permit.
Every invocation writes to a new direct child of this package's artifacts/.
The worker starts suspended and cannot execute before Windows Job assignment.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

PACKAGE = Path(__file__).resolve().parent
GIB = 1024 ** 3
BOUNDS = {
    "preworker_available_GiB": 3.3125,
    "system_available_reserve_GiB": 1.25,
    "worker_process_tree_RSS_GiB": 2.0,
    "GPU_free_at_admission_GiB": 4.0,
    "GPU_free_during_work_GiB": 0.5,
    "disk_free_GiB": 4.0,
    "external_watchdog_milliseconds": 100,
    "whole_scope_seconds": 600,
}
CPU_INTERVAL = 0.1
CPU_FRESHNESS = 0.2
GPU_INTERVAL = 0.5
GPU_FRESHNESS = 2.0
GPU_TIMEOUT = 2.0
ADMISSION_INTERVAL = 2.0
ADMISSION_DEADLINE = 30.0
HEARTBEAT_MAX_BYTES = 1024 ** 2
_LOCAL_LEASE_LOCK = threading.Lock()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _json_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_safe(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _heartbeat(output, supervisor_pid):
    # One writer, before resume and then in the CPU thread only. Appending avoids
    # replacing a pathname while the Windows worker reader has it open. A reader
    # may see a trailing partial record and must select the last complete line.
    target = Path(output) / "supervisor_heartbeat.jsonl"
    record = (json.dumps({"unix_seconds": time.time(), "supervisor_pid": supervisor_pid},
                         allow_nan=False) + "\n").encode("utf-8")
    if not target.exists():
        stream = target.open("xb")  # first creation is exclusive
    else:
        # No O_CREAT: a disappearing ledger is an error rather than a reset.
        descriptor = os.open(target, os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0))
        stream = os.fdopen(descriptor, "ab")
    with stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() + len(record) > HEARTBEAT_MAX_BYTES:
            raise ValueError("heartbeat ledger exceeds prospective 1MiB byte cap")
        stream.write(record)
        stream.flush()


def _new_output(output_dir):
    artifacts = PACKAGE / "artifacts"
    if artifacts.is_symlink():
        raise ValueError("artifact root must not be a symlink")
    artifacts.mkdir(exist_ok=True)
    requested = Path(output_dir).absolute()
    if requested.parent.resolve() != artifacts.resolve() or requested.name.startswith("_"):
        raise ValueError("output must be a new direct, non-reserved child of package/artifacts")
    if requested.is_symlink():
        raise ValueError("output must not be a symlink")
    requested.mkdir(exist_ok=False)
    return requested.resolve()


def _psutil():
    # Importing this helper never imports a model framework.
    import psutil
    return psutil


def _pid_identity(pid):
    process = _psutil().Process(pid)
    return {"pid": int(pid), "create_time": process.create_time()}


def _pid_alive(identity):
    psutil = _psutil()
    try:
        process = psutil.Process(identity["pid"])
        return (process.create_time() == identity["create_time"]
                and process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False
    # Access denial or malformed identity is not evidence of death.


class _LeaseMutex:
    """Serialize current-PID verification and exclusive lease enrollment."""

    def __enter__(self):
        if os.name != "nt":
            _LOCAL_LEASE_LOCK.acquire()
            self.handle = None
            return self
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.api.CreateMutexW.restype = wintypes.HANDLE
        self.api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.api.WaitForSingleObject.restype = wintypes.DWORD
        self.api.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        suffix = hashlib.sha256(str(PACKAGE).encode()).hexdigest()[:24]
        self.handle = self.api.CreateMutexW(None, False, "Local\\SYNTH_CRS_001_GPU0_" + suffix)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        result = self.api.WaitForSingleObject(self.handle, 100)
        if result not in (0, 0x80):  # acquired, or abandoned by a dead owner
            self.api.CloseHandle(self.handle)
            raise RuntimeError("GPU lease registration mutex is busy or unverifiable")
        return self

    def __exit__(self, *unused):
        if self.handle is None:
            _LOCAL_LEASE_LOCK.release()
        else:
            self.api.ReleaseMutex(self.handle)
            self.api.CloseHandle(self.handle)


def _acquire_lease(output, config_sha256):
    root = PACKAGE / "artifacts" / "_native_gpu_leases"
    if root.is_symlink():
        raise ValueError("lease root must not be a symlink")
    root.mkdir(exist_ok=True)
    with _LeaseMutex():
        for prior in sorted(root.iterdir()):
            if not prior.is_dir() or prior.is_symlink():
                raise RuntimeError("unverifiable lease entry")
            reservation = _read(prior / "reservation.json")
            worker = _read(prior / "worker.json") if (prior / "worker.json").exists() else None
            terminal = _read(prior / "terminal.json") if (prior / "terminal.json").exists() else None
            if worker and _pid_alive(worker):
                raise RuntimeError("a previously leased owned GPU worker is currently alive")
            if not worker and not terminal and _pid_alive(reservation["supervisor"]):
                raise RuntimeError("a currently alive supervisor owns an unfinished GPU reservation")
            if terminal and not worker and terminal.get("launch_failure", {}).get("worker_pid"):
                failure = terminal["launch_failure"]
                if not failure.get("owned_process_handle_terminal_verified"):
                    raise RuntimeError("prior failed launch process was not verified terminal")
            if (terminal and (terminal.get("sampler_threads_terminal_verified") is False
                              or terminal.get("gpu_telemetry_cleanup_verified") is False)
                    and _pid_alive(reservation["supervisor"])):
                raise RuntimeError("a currently live supervisor has unverified sampler shutdown")
            if (terminal and terminal.get("worker_attempted")
                    and terminal.get("config_sha256") == config_sha256
                    and terminal.get("status") != "SUPERVISED_WORKER_EXITED_ZERO"):
                raise RuntimeError("unchanged failed worker scope cannot automatically retry")
            _json_new(prior / ("death_observation_" + uuid.uuid4().hex + ".json"),
                      {"observed_utc": _utc(), "worker_identity": worker,
                       "worker_currently_alive": False, "reservation_identity": reservation["supervisor"]})
        lease = root / uuid.uuid4().hex
        lease.mkdir()
        _json_new(lease / "reservation.json", {
            "schema_version": 1, "lease_id": lease.name, "created_utc": _utc(),
            "supervisor": _pid_identity(os.getpid()), "output_dir": str(output),
            "config_sha256": config_sha256, "GPU_index": 0,
        })
    return lease


def _cpu_snapshot(output_dir, worker=None):
    psutil = _psutil()
    started = time.monotonic()
    owned = ({"identities": [], "rss_bytes": 0} if worker is None
             else worker.process_resource_snapshot())
    available = psutil.virtual_memory().available
    disk_free = shutil.disk_usage(output_dir).free
    return {"sample_started_monotonic": started, "sample_completed_monotonic": time.monotonic(),
            "host_available_bytes": available,
            "disk_free_bytes": disk_free,
            "worker_tree_rss_bytes": owned["rss_bytes"], "owned_process_identities": owned["identities"]}


def _gpu_snapshot():
    import native_nvml
    return native_nvml.gpu_snapshot()


def _gpu_pending_threads():
    import native_nvml
    return native_nvml.pending_query_threads()


def _shutdown_gpu_probe():
    import native_nvml
    native_nvml.shutdown()


def resource_snapshot(output_dir, worker=None):
    """Fresh CPU/disk and bounded read-only NVML GPU0 observation; no Torch."""
    cpu = _cpu_snapshot(output_dir, worker)
    gpu = _gpu_snapshot()
    # Refresh RAM after telemetry as well, rather than admitting old RAM evidence.
    refreshed = _cpu_snapshot(output_dir, worker)
    return {**cpu, **refreshed, **gpu}


def _bounds_valid(config):
    resources = config.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("missing fixed resource contract")
    for name, expected in BOUNDS.items():
        actual = resources.get(name)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or actual != expected:
            raise ValueError("fixed resource bound changed: " + name)


def _fresh_cpu(snapshot, now):
    return (all(type(snapshot.get(name)) is int and snapshot[name] >= 0 for name in
                ("host_available_bytes", "disk_free_bytes", "worker_tree_rss_bytes"))
            and 0 <= now - snapshot["sample_started_monotonic"] <= CPU_FRESHNESS
            and snapshot["sample_started_monotonic"] <= snapshot["sample_completed_monotonic"] <= now)


def _fresh_gpu(snapshot, now):
    return (all(type(snapshot.get(name)) is int and snapshot[name] >= 0 for name in
                ("gpu_total_bytes", "gpu_free_bytes"))
            and snapshot.get("GPU_index") == 0 and type(snapshot.get("GPU_index")) is int
            and snapshot["gpu_total_bytes"] >= snapshot["gpu_free_bytes"]
            and isinstance(snapshot.get("gpu_uuid"), str) and bool(snapshot["gpu_uuid"])
            and isinstance(snapshot.get("gpu_observation_id"), str) and bool(snapshot["gpu_observation_id"])
            and 0 <= now - snapshot["gpu_sample_started_monotonic"] <= GPU_FRESHNESS
            and snapshot["gpu_sample_started_monotonic"] <= snapshot["gpu_sample_completed_monotonic"] <= now)


def _admission_failure(snapshot, now):
    if not _fresh_cpu(snapshot, now) or not _fresh_gpu(snapshot, now):
        return "stale_or_invalid_admission_observation"
    for key, floor in (("host_available_bytes", 3.3125), ("gpu_free_bytes", 4.0), ("disk_free_bytes", 4.0)):
        if snapshot[key] < floor * GIB:
            return "admission_floor_" + key
    return None


class _IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IOCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _WindowsJob:
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("worker launch requires Windows Job containment")
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            getattr(self.api, name).argtypes = arguments
            getattr(self.api, name).restype = result
        self.name = "Local\\SYNTH_NQ_JOB_" + uuid.uuid4().hex
        ctypes.set_last_error(0)
        self.handle = self.api.CreateJobObjectW(None, self.name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
            # A collision must never attach to or reconfigure another Job.
            self.close()
            raise RuntimeError("new owned Job name already exists")
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE; no BREAKAWAY flags
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process_handle):
        if not self.api.AssignProcessToJobObject(self.handle, process_handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def pids(self):
        # Dynamic list accommodates descendants; never fall back to just the root.
        count = 16
        while count <= 65536:
            class PidList(ctypes.Structure):
                _fields_ = [("assigned", wintypes.DWORD), ("count", wintypes.DWORD),
                            ("pids", ctypes.c_size_t * count)]
            value = PidList()
            if self.api.QueryInformationJobObject(self.handle, 3, ctypes.byref(value), ctypes.sizeof(value), None):
                return list(value.pids[:value.count])
            if ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(ctypes.get_last_error())
            count = max(count * 2, value.assigned)
        raise RuntimeError("owned process list exceeds containment audit bound")

    def terminate(self):
        if self.handle and not self.api.TerminateJobObject(self.handle, 137):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            if not self.api.CloseHandle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = None


def _resume_primary_thread(pid):
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    class ThreadEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ThreadID", wintypes.DWORD),
                    ("th32OwnerProcessID", wintypes.DWORD), ("tpBasePri", wintypes.LONG),
                    ("tpDeltaPri", wintypes.LONG), ("dwFlags", wintypes.DWORD)]
    api.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    api.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    api.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
    api.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
    api.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenThread.restype = wintypes.HANDLE
    api.ResumeThread.argtypes = [wintypes.HANDLE]
    api.ResumeThread.restype = wintypes.DWORD
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = api.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
    if snapshot == ctypes.c_void_p(-1).value or not snapshot:
        raise ctypes.WinError(ctypes.get_last_error())
    ids = []
    try:
        entry = ThreadEntry()
        entry.dwSize = ctypes.sizeof(entry)
        present = api.Thread32First(snapshot, ctypes.byref(entry))
        while present:
            if entry.th32OwnerProcessID == pid:
                ids.append(entry.th32ThreadID)
            entry.dwSize = ctypes.sizeof(entry)
            present = api.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        api.CloseHandle(snapshot)
    if len(ids) != 1:
        raise RuntimeError("suspended worker must have one primary thread")
    thread = api.OpenThread(2, False, ids[0])  # THREAD_SUSPEND_RESUME
    if not thread:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if api.ResumeThread(thread) != 1:
            raise RuntimeError("worker primary thread was not suspended exactly once")
    finally:
        api.CloseHandle(thread)


class _LaunchFailure(RuntimeError):
    def __init__(self, error, evidence):
        super().__init__(type(error).__name__ + ": " + str(error))
        self.evidence = evidence


class _WindowsOwnedWorker:
    def __init__(self, argv, output, *, config_sha256, deadline_monotonic):
        self.job = _WindowsJob()
        self.process = None
        self.logs = []
        self.deadline_monotonic = deadline_monotonic
        self.resumed = False
        assigned = False
        try:
            for name in ("worker.stdout.log", "worker.stderr.log"):
                self.logs.append((output / name).open("x", encoding="utf-8"))
            environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_HUB_KERNELS="NO",
                               CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1")
            self.process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=self.logs[0], stderr=self.logs[1],
                                            cwd=str(PACKAGE), env=environment, shell=False,
                                            creationflags=0x08000000 | 0x00000004)  # NO_WINDOW | SUSPENDED
            self.identity = _pid_identity(self.process.pid)
            self.job.assign(int(self.process._handle))
            assigned = True
            self.identity.update({"process_handle": int(self.process._handle), "job_handle": int(self.job.handle),
                                  "job_name": self.job.name,
                                  "job_kill_on_close": True, "breakaway_allowed": False,
                                  "assigned_before_resume": True})
            supervisor_identity = _pid_identity(os.getpid())
            _json_new(output / "supervisor_ownership.json", {
                "worker_pid": self.process.pid, "root_worker_pid": self.process.pid,
                "job_name": self.job.name, "supervisor_pid": os.getpid(),
                "supervisor_create_time": supervisor_identity["create_time"], "config_sha256": config_sha256})
            _heartbeat(output, os.getpid())
            if time.monotonic() >= deadline_monotonic:
                raise RuntimeError("whole scope elapsed before suspended worker resume")
            # Return still suspended. The external monitor must be started and
            # observe this actual owned tree before resume() can execute code.
        except BaseException as error:
            evidence = {"attempted_worker_argv": argv, "assigned_before_resume": assigned,
                        "resumed_successfully": False, "process_identity": getattr(self, "identity", None),
                        "cleanup_errors": [], "returncode": None}
            if self.process is not None:
                evidence.update({"worker_pid": self.process.pid, "process_handle": int(self.process._handle)})
                try:
                    if assigned:
                        self.job.terminate()
                    else:
                        self.process.kill()  # only the just-created, suspended root
                    self.process.wait(timeout=5)
                    evidence["returncode"] = self.process.poll()
                    evidence["owned_process_handle_terminal_verified"] = evidence["returncode"] is not None
                except BaseException as cleanup_error:
                    evidence["cleanup_errors"].append(type(cleanup_error).__name__ + ": " + str(cleanup_error))
            try:
                self.job.close()
                evidence["job_closed_kill_on_close"] = True
            except BaseException as cleanup_error:
                evidence["cleanup_errors"].append(type(cleanup_error).__name__ + ": " + str(cleanup_error))
            for stream in self.logs:
                stream.close()
            raise _LaunchFailure(error, evidence) from error

    def resume(self):
        if self.resumed:
            raise RuntimeError("owned worker cannot resume twice")
        if time.monotonic() >= self.deadline_monotonic:
            raise RuntimeError("whole scope elapsed before suspended worker resume")
        if self.poll() is not None:
            raise RuntimeError("owned suspended worker is already terminal")
        _resume_primary_thread(self.process.pid)
        self.resumed = True

    def poll(self):
        return self.process.poll()  # Windows Popen polls its actual process handle.

    def is_alive(self):
        return self.poll() is None or bool(self.job.pids())

    def process_identities(self):
        values = []
        for pid in self.job.pids():
            try:
                values.append(_pid_identity(pid))
            except _psutil().NoSuchProcess:
                continue
        return values

    def process_resource_snapshot(self):
        """One fresh Process object per current Job PID for birth and RSS."""
        psutil = _psutil()
        identities = []
        rss = 0
        for pid in self.job.pids():
            try:
                process = psutil.Process(pid)
                birth = process.create_time()
                if pid == self.identity["pid"] and birth != self.identity["create_time"]:
                    raise RuntimeError("owned root PID birth time changed")
                resident = process.memory_info().rss
                identities.append({"pid": pid, "create_time": birth})
                rss += resident
            except psutil.NoSuchProcess:
                continue
        return {"identities": identities, "rss_bytes": rss}

    def terminate(self):
        self.job.terminate()

    def close(self):
        self.job.close()  # closes the only inheritable-false Job handle and kills all descendants
        try:
            self.process.wait(timeout=5)
        finally:
            for stream in self.logs:
                stream.close()


class _Monitor:
    """External CPU/RSS watchdog and independent bounded GPU sampler."""

    def __init__(self, worker, output, deadline, admitted):
        self.worker, self.output, self.deadline = worker, output, deadline
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.cpu, self.gpu = admitted, admitted
        self.reason = None
        self.observations = []
        self.threads = []
        self.ready = threading.Event()
        self.unjoined_threads = []

    def _abort(self, reason):
        with self.lock:
            if self.reason is not None:
                return
            self.reason = reason
        try:
            self.worker.terminate()
        except BaseException as error:
            with self.lock:
                self.observations.append({"termination_error": type(error).__name__ + ": " + str(error)})
            # Close is a second, OS-enforced tree containment action.
            self.worker.job.close()

    def check(self):
        now = time.monotonic()
        with self.lock:
            cpu, gpu, reason = self.cpu, self.gpu, self.reason
        if reason:
            return reason
        if now >= self.deadline:
            return "whole_scope_deadline"
        if not _fresh_cpu(cpu, now):
            return "stale_external_CPU_watchdog"
        if not _fresh_gpu(gpu, now):
            return "stale_external_GPU_observation"
        if cpu["host_available_bytes"] < 1.25 * GIB:
            return "runtime_system_reserve"
        if cpu["worker_tree_rss_bytes"] > 2.0 * GIB:
            return "runtime_owned_process_tree_RSS"
        if gpu["gpu_free_bytes"] < 0.5 * GIB:
            return "runtime_GPU_free"
        if gpu["gpu_uuid"] != self.gpu_uuid:
            return "runtime_GPU_identity_changed"
        return None

    def _cpu_loop(self):
        next_poll = time.monotonic()
        while not self.stop.is_set():
            try:
                snapshot = _cpu_snapshot(self.output, self.worker)
                with self.lock:
                    self.cpu = snapshot
                    self.observations.append({"kind": "CPU", **snapshot})
                # Publish a complete current sample before heartbeat file IO.
                self.ready.set()
                _heartbeat(self.output, os.getpid())
                reason = self.check()
                if reason:
                    self._abort(reason)
                    return
            except BaseException as error:
                self._abort("CPU_watchdog_error:" + type(error).__name__ + ":" + str(error))
                return
            next_poll += CPU_INTERVAL
            self.stop.wait(max(0, next_poll - time.monotonic()))

    def _gpu_loop(self):
        next_poll = time.monotonic() + GPU_INTERVAL
        while not self.stop.wait(max(0, next_poll - time.monotonic())):
            try:
                snapshot = _gpu_snapshot()
                with self.lock:
                    self.gpu = snapshot
                    self.observations.append({"kind": "GPU", **snapshot})
            except BaseException as error:
                self._abort("GPU_probe_error:" + type(error).__name__ + ":" + str(error))
                return
            next_poll = time.monotonic() if time.monotonic() > next_poll + GPU_INTERVAL else next_poll + GPU_INTERVAL

    def start(self):
        self.gpu_uuid = self.gpu["gpu_uuid"]
        # Process creation, ownership writes and thread enumeration need not fit
        # into a 200ms CPU observation window. Keep the process suspended, take
        # genuine current measurements, and start the monitor before resuming.
        if not _fresh_gpu(self.gpu, time.monotonic()):
            self.gpu = _gpu_snapshot()
            self.observations.append({"kind": "GPU", "phase": "worker_suspended_startup", **self.gpu})
        self.cpu = _cpu_snapshot(self.output, self.worker)
        self.observations.append({"kind": "CPU", "phase": "worker_suspended_startup", **self.cpu})
        reason = self.pre_resume_check()
        if reason:
            self._abort(reason)
            return self
        for function in (self._cpu_loop, self._gpu_loop):
            thread = threading.Thread(target=function, daemon=True, name="SYNTH_EXTERNAL" + function.__name__)
            thread.start()
            self.threads.append(thread)
        if not self.ready.wait(CPU_FRESHNESS):
            self._abort("startup_external_CPU_watchdog_not_ready")
        return self

    def pre_resume_check(self):
        reason = self.check()
        if reason:
            return reason
        with self.lock:
            current = {**self.gpu, **self.cpu}
        return _admission_failure(current, time.monotonic())

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=GPU_TIMEOUT + CPU_INTERVAL)
        self.unjoined_threads = [{"name": thread.name, "ident": thread.ident, "kind": "external_sampler"}
                                 for thread in self.threads if thread.is_alive()]
        self.unjoined_threads.extend({"kind": "NVML_native_call", **item} for item in _gpu_pending_threads())
        with self.lock:
            return list(self.observations)


def _preflight(config_path):
    import native_contract
    return native_contract.preflight(config_path)


def _inventory(output):
    values = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            values.append({"path": str(path.relative_to(output)), "type": "symlink_not_followed"})
        elif path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 ** 2), b""):
                    digest.update(block)
            values.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return values


def supervise(config_path, output_dir, *, admission_only=False):
    """Return and preserve one terminal receipt, with no launch retry."""
    started = time.monotonic()
    output = _new_output(output_dir)
    argv = [sys.executable, "-I", "-B", str(PACKAGE / "native_worker.py"),
            "--config", str(Path(config_path).absolute()), "--output-dir", str(output)]
    receipt = {"schema_version": 1, "started_utc": _utc(), "output_dir": str(output),
               "attempted_worker_argv": argv, "worker_attempted": False, "worker_launched": False,
               "worker_resumed": False,
               "admission_only": bool(admission_only), "bounds": dict(BOUNDS),
               "cadence": {"CPU_watchdog_seconds": CPU_INTERVAL, "CPU_freshness_seconds": CPU_FRESHNESS,
                           "GPU_query_interval_seconds": GPU_INTERVAL, "GPU_query_timeout_seconds": GPU_TIMEOUT,
                           "GPU_freshness_seconds": GPU_FRESHNESS, "admission_interval_seconds": ADMISSION_INTERVAL,
                           "admission_deadline_seconds": ADMISSION_DEADLINE, "admission_samples": 2},
               "deadline_monotonic": started + 600, "status": "ADMISSION_DENIED", "returncode": None,
               "gpu_telemetry_backend": "pinned_System32_NVML_read_only",
               "mechanical_qualification_established": False, "production_training_authorized": False,
               "protected_bank_access": False, "admission_observations": []}
    lease = worker = monitor = None
    try:
        config = _preflight(config_path)
        receipt["preflight"] = config
        receipt["config_sha256"] = config["config_sha256"]
        authorization = config.get("authorization", {})
        if (authorization.get("scope") != "mechanical_native_qualification_only"
                or any(authorization.get(name) is not False for name in
                       ("production_training", "protected_bank_access", "paid_services"))):
            raise ValueError("supervisor admits mechanical qualification only")
        _bounds_valid(config)
        if Path(config["worker_path"]).resolve() != (PACKAGE / "native_worker.py").resolve():
            raise ValueError("worker must be the pinned package/native_worker.py")
        argv[3] = config["worker_path"]
        argv[5] = config["config_path"]
        if time.monotonic() >= receipt["deadline_monotonic"]:
            raise RuntimeError("whole scope elapsed during preflight")
        lease = _acquire_lease(output, config["config_sha256"])
        receipt["lease_path"] = str(lease)
        if not admission_only:
            # Byte checks can take longer than the GPU freshness window. Complete
            # them before the final current observation pair; the worker repeats
            # preflight before importing any model framework or loading weights.
            refreshed = _preflight(config_path)
            if refreshed != config:
                raise RuntimeError("source/config/runtime/data binding changed before admission")
            receipt["preflight_revalidated_before_current_observations"] = True
            if time.monotonic() >= receipt["deadline_monotonic"]:
                raise RuntimeError("whole scope elapsed during pin revalidation")
        admission_started = time.monotonic()
        for index in range(2):
            if index:
                time.sleep(ADMISSION_INTERVAL)
            observation = resource_snapshot(output)
            receipt["admission_observations"].append(observation)
            now = time.monotonic()
            if now - admission_started > ADMISSION_DEADLINE or now >= receipt["deadline_monotonic"]:
                raise RuntimeError("admission observation deadline")
            failure = _admission_failure(observation, now)
            if failure:
                raise RuntimeError(failure)
            if index:
                prior = receipt["admission_observations"][0]
                if (observation["gpu_observation_id"] == prior["gpu_observation_id"]
                        or observation["gpu_sample_started_monotonic"] <= prior["gpu_sample_started_monotonic"]
                        or observation["sample_started_monotonic"] <= prior["sample_started_monotonic"]
                        or observation["gpu_uuid"] != prior["gpu_uuid"]):
                    raise RuntimeError("admission requires two distinct current observations of the same GPU")
        receipt["status"] = "ADMISSION_PASSED_NO_WORKER"
        if not admission_only:
            dispatch_cpu = _cpu_snapshot(output)
            receipt["dispatch_observation"] = {**observation, **dispatch_cpu}
            if _admission_failure(receipt["dispatch_observation"], time.monotonic()):
                raise RuntimeError("admission became stale before dispatch")
            observation = receipt["dispatch_observation"]
            receipt["worker_attempted"] = True
            worker = _WindowsOwnedWorker(argv, output, config_sha256=config["config_sha256"],
                                         deadline_monotonic=receipt["deadline_monotonic"])
            receipt["worker_launched"] = True
            receipt["worker_identity"] = dict(worker.identity)
            receipt["owned_handle_alive_verified_after_launch"] = worker.is_alive()
            _json_new(lease / "worker.json", receipt["worker_identity"])
            monitor = _Monitor(worker, output, receipt["deadline_monotonic"], observation)
            monitor.start()
            reason = monitor.pre_resume_check()
            if reason:
                monitor._abort(reason)
                receipt["status"] = "OWNED_WORKER_ABORTED"
                receipt["failure"] = reason
            else:
                worker.resume()
                receipt["worker_resumed"] = True
                receipt["worker_resume_monotonic"] = time.monotonic()
                _json_new(lease / "resume.json", {
                    "worker_identity": receipt["worker_identity"], "resumed_monotonic": receipt["worker_resume_monotonic"],
                    "external_monitor_ready_before_resume": True, "deadline_monotonic": receipt["deadline_monotonic"]})
                while True:
                    reason = monitor.check()
                    if reason:
                        monitor._abort(reason)
                        receipt["status"] = "OWNED_WORKER_ABORTED"
                        receipt["failure"] = reason
                        break
                    returncode = worker.poll()
                    if returncode is not None:
                        receipt["returncode"] = returncode
                        if worker.process_identities():
                            monitor._abort("root_exited_with_live_owned_descendants")
                            receipt["status"] = "OWNED_WORKER_ABORTED"
                            receipt["failure"] = "root_exited_with_live_owned_descendants"
                        else:
                            receipt["status"] = ("SUPERVISED_WORKER_EXITED_ZERO" if returncode == 0
                                                 else "SUPERVISED_WORKER_FAILED")
                        break
                    time.sleep(CPU_INTERVAL)
    except BaseException as error:
        receipt["failure"] = type(error).__name__ + ": " + str(error)
        receipt["status"] = "OWNED_WORKER_ABORTED" if worker else "ADMISSION_DENIED"
        if isinstance(error, _LaunchFailure):
            receipt["status"] = "OWNED_WORKER_LAUNCH_FAILED"
            receipt["launch_failure"] = error.evidence
            receipt["returncode"] = error.evidence["returncode"]
            if lease and error.evidence.get("process_identity"):
                _json_new(lease / "worker.json", error.evidence["process_identity"])
            _json_new(output / "WORKER_LAUNCH_FAILURE.json", error.evidence)
        if worker:
            try:
                worker.terminate()
            except BaseException as containment_error:
                receipt["termination_error"] = type(containment_error).__name__ + ": " + str(containment_error)
    finally:
        unjoined = []
        if monitor:
            try:
                observations = monitor.close()
                unjoined.extend(getattr(monitor, "unjoined_threads", []))
                _json_new(output / "RESOURCE_OBSERVATIONS.json", observations)
            except BaseException as error:
                receipt["sampler_shutdown_error"] = type(error).__name__ + ": " + str(error)
                unjoined.append({"kind": "external_sampler_shutdown_unverified"})
        if worker:
            try:
                worker.close()
                receipt["returncode"] = worker.poll()
                receipt["job_closed_kill_on_close"] = True
                receipt["owned_worker_identity_currently_alive"] = _pid_alive(worker.identity)
            except BaseException as error:
                receipt["containment_failure"] = type(error).__name__ + ": " + str(error)
                receipt["status"] = "OWNED_WORKER_ABORTED"
        if not unjoined:
            try:
                _shutdown_gpu_probe()
                receipt["gpu_telemetry_cleanup_verified"] = True
            except BaseException as error:
                receipt["gpu_telemetry_cleanup_error"] = type(error).__name__ + ": " + str(error)
                receipt["gpu_telemetry_cleanup_verified"] = False
        unjoined.extend({"kind": "NVML_native_call", **item} for item in _gpu_pending_threads())
        unique = {}
        for item in unjoined:
            key = (item.get("kind"), item.get("name"), item.get("ident"), item.get("operation"))
            unique[key] = item
        unjoined = list(unique.values())
        receipt["unjoined_sampler_threads"] = unjoined
        receipt["sampler_threads_terminal_verified"] = not unjoined
        if unjoined or receipt.get("gpu_telemetry_cleanup_verified") is False:
            receipt["status"] = "SUPERVISOR_SHUTDOWN_UNVERIFIED"
        receipt["finished_utc"] = _utc()
        receipt["elapsed_seconds"] = time.monotonic() - started
        if (time.monotonic() >= receipt["deadline_monotonic"]
                and receipt["status"] == "SUPERVISED_WORKER_EXITED_ZERO"):
            receipt["status"] = "WHOLE_SCOPE_DEADLINE_EXCEEDED"
            receipt["failure"] = "whole_scope_deadline_during_shutdown"
        if unjoined:
            receipt["partial_outputs_frozen"] = False
            receipt["partial_outputs"] = [{"path": str(path.relative_to(output)), "frozen": False}
                                           for path in sorted(output.rglob("*")) if path.is_file() or path.is_symlink()]
        else:
            receipt["partial_outputs_frozen"] = True
            receipt["partial_outputs"] = _inventory(output)
        receipt["finished_utc"] = _utc()
        receipt["elapsed_seconds"] = time.monotonic() - started
        if (time.monotonic() >= receipt["deadline_monotonic"]
                and receipt["status"] == "SUPERVISED_WORKER_EXITED_ZERO"):
            receipt["status"] = "WHOLE_SCOPE_DEADLINE_EXCEEDED"
            receipt["failure"] = "whole_scope_deadline_during_inventory"
        receipt = _json_safe(receipt)
        if lease:
            _json_new(lease / "terminal.json", receipt)
        _json_new(output / "SUPERVISOR_RECEIPT.json", receipt)
    return receipt
