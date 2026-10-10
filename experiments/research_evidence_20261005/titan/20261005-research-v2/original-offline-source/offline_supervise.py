"""TITAN quality offline CPU slot: OS-exclusive handle, Windows Job and measured watchdog.

No native model work. Outputs/caches are confined to a fresh external BusinessProject
directory. The child waits for Job assignment before importing fixture dependencies.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import datetime as dt
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BUSINESS = Path("C:/Users/clclc/BusinessProject").resolve()
HARDWARE_LOCK = BUSINESS / "ChatBubble-agent-prompts-20261002/hardware-worker.lock"
MEMORY_CAP = 512 * 1024 ** 2
OUTPUT_CAP = 50 * 1024 ** 2
WALL_CAP = 900
TEST_MODULES = ()  # discover this experiment only

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def source_snapshot():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.iterdir()) if p.is_file()}


def write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def output_bytes(root):
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:  # fixture temporary files can disappear between enumeration and stat
            continue
    return total


def output_root(path):
    path = Path(path).resolve()
    if (not path.is_relative_to(BUSINESS / "ChatBubble-agent-prompts-20261002/run-artifacts/titan-quality-repair-v1") or path.is_relative_to(REPO)
            or ".codex" in [part.lower() for part in path.parts]
            or any((parent / ".git").exists() for parent in (path, *path.parents))):
        raise ValueError("STOP_OUTPUT_MUST_BE_EXTERNAL_BUSINESSPROJECT")
    path.mkdir(parents=True, exist_ok=False)
    return path


class HardwareHandle:
    def __init__(self, path, record):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        self.kernel.CreateFileW.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        self.handle = self.kernel.CreateFileW(str(path), 0xC0000000, 0, None, 4, 0x80, None)
        if self.handle == ctypes.c_void_p(-1).value:
            raise RuntimeError(f"STOP_HARDWARE_SLOT_HELD_OR_UNREADABLE:{ctypes.get_last_error()}")
        # Truncate/write only after our exclusive handle was admitted; file remains on close.
        self.kernel.SetFilePointer.argtypes = (wintypes.HANDLE, wintypes.LONG, ctypes.c_void_p, wintypes.DWORD)
        self.kernel.SetEndOfFile.argtypes = (wintypes.HANDLE,)
        self.kernel.WriteFile.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                          ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p)
        self.kernel.FlushFileBuffers.argtypes = (wintypes.HANDLE,)
        self.kernel.SetFilePointer(self.handle, 0, None, 0)
        if not self.kernel.SetEndOfFile(self.handle):
            self.close()
            raise OSError("STOP_HARDWARE_OWNER_RECORD_TRUNCATE_FAILED")
        data = json.dumps(record).encode("utf-8")
        count = wintypes.DWORD()
        if not self.kernel.WriteFile(self.handle, data, len(data), ctypes.byref(count), None) or count.value != len(data):
            self.close()
            raise OSError("STOP_HARDWARE_OWNER_RECORD_WRITE_FAILED")
        if not self.kernel.FlushFileBuffers(self.handle):
            self.close()
            raise OSError("STOP_HARDWARE_OWNER_RECORD_FLUSH_FAILED")

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class BasicLimit(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount",
        "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class ExtendedLimit(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BasicLimit), ("IoInfo", IOCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class Job:
    def __init__(self):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        self.kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self.kernel.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        self.kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError("STOP_JOB_CREATE_FAILED")
        limit = ExtendedLimit()
        limit.BasicLimitInformation.LimitFlags = 0x2000 | 0x200  # kill on close + total private-commit cap
        limit.JobMemoryLimit = MEMORY_CAP
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limit), ctypes.sizeof(limit)):
            self.close()
            raise OSError("STOP_JOB_LIMIT_FAILED")

    def assign(self, child):
        if not self.kernel.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(child._handle))):
            raise OSError(f"STOP_JOB_ASSIGN_FAILED:{ctypes.get_last_error()}")

    def terminate(self):
        if not self.kernel.TerminateJobObject(self.handle, 125):
            raise OSError("STOP_JOB_TERMINATE_FAILED")

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "tensorflow", "cupy", "jax"}:
            raise ImportError(f"STOP_LIGHTWEIGHT_PROFILE_MODEL_IMPORT:{fullname}")


def payload(stage, root, gate):
    deadline = time.monotonic() + 30
    while not gate.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("STOP_NO_JOB_ADMISSION")
        time.sleep(.05)
    sys.meta_path.insert(0, NoModelImports())
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO))
    if stage == "containment":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.IsProcessInJob.argtypes = (wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL))
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        in_job = wintypes.BOOL()
        if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(in_job)) or not in_job.value:
            raise RuntimeError("STOP_CHILD_NOT_IN_JOB")
        script = ("import ctypes,json,os,time; from ctypes import wintypes; "
            "k=ctypes.WinDLL('kernel32',use_last_error=True); "
            "k.GetCurrentProcess.restype=wintypes.HANDLE; "
            "k.IsProcessInJob.argtypes=(wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)); "
            "flag=wintypes.BOOL(); k.IsProcessInJob(k.GetCurrentProcess(),None,ctypes.byref(flag)); "
            "from pathlib import Path; "
            "Path(__import__('sys').argv[1]).write_text(json.dumps({'pid':os.getpid(),'in_job':bool(flag.value)})); "
            "time.sleep(45)")
        child = subprocess.Popen([sys.executable, "-B", "-c", script, str(root / "GRANDCHILD.json")],
                                  creationflags=0x08000000)
        write(root / "CONTAINMENT_PAYLOAD.json", {"pid": os.getpid(), "in_job": bool(in_job.value),
                                                   "grandchild_pid": child.pid})
        time.sleep(45)
        return 1  # watchdog must terminate the full job first
    if stage == "tests":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_*.py", top_level_dir=str(REPO))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        write(root / "CPU_FIXTURES.json", {"tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "passed": result.wasSuccessful(), "profile": "LIGHTWEIGHT_NO_TORCH",
            "model_loads": 0, "model_forwards": 0, "capability_evidence": False})
        return 0 if result.wasSuccessful() else 1
    if stage == "prepare":
        from experiments.titan_quality_repair_v1.packet import prepare_packet
        result = prepare_packet(root)
        return 0 if result["offline_validation_passed"] else 1
    if stage == "inspect":
        from experiments.titan_quality_repair_v1.replay import load_replay
        rows, receipt = load_replay()
        from experiments.titan_quality_repair_v1.custody import source_identity
        write(root / "INPUT_INSPECTION.json", {"replay": receipt, "source": source_identity()})
        print(__import__("json").dumps({"replay_eligible": len(rows), "families": receipt["eligible_families"]}))
        return 0
    raise ValueError("unknown stage")


def supervise(stage, path):
    import psutil
    root = output_root(path)
    wall_cap = 3 if stage == "containment" else WALL_CAP
    record = {"experiment": "TITAN-QUALITY-REPAIR-V1", "stage": stage, "utc_start": utc(),
        "worktree": str(REPO), "supervisor_pid": os.getpid(), "token": uuid.uuid4().hex,
        "profile": "LIGHTWEIGHT_NO_TORCH", "memory_cap_bytes": MEMORY_CAP,
        "wall_cap_seconds": wall_cap, "output_cap_bytes": OUTPUT_CAP, "threads": 1,
        "native_model_loads": 0, "native_model_forwards": 0, "native_generations": 0,
        "gpu_leases_acquired": 0, "paid_calls": 0, "status": "STARTING"}
    admitted_sources = source_snapshot()
    record["source_admission_sha256"] = admitted_sources
    hardware = job = child = None
    start = time.monotonic()
    peak_rss = peak_output = 0
    minimum_available = psutil.virtual_memory().available
    try:
        if minimum_available < 2 * 1024 ** 3:
            raise RuntimeError("STOP_PRELOAD_HOST_RAM_BELOW_2GIB")
        hardware = HardwareHandle(HARDWARE_LOCK, record)
        if stage == "containment":
            try:
                competing = HardwareHandle(HARDWARE_LOCK, {"stage": "fixture_competing_open"})
            except RuntimeError:
                record["exclusive_handle_refuses_competing_open"] = True
            else:
                competing.close()
                raise RuntimeError("STOP_EXCLUSIVE_HANDLE_FIXTURE_FAILED")
        job = Job()
        gate = root / "JOB_ADMITTED.json"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
            "TMP": str(root), "TEMP": str(root), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(root / "hf_cache"), "XDG_CACHE_HOME": str(root / "cache")}
        with (root / "worker.log").open("xb") as log:
            child = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--payload", stage,
                "--output-dir", str(root), "--gate", str(gate)], stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=env, cwd=str(HERE), creationflags=0x08000000)
            record["worker_pid"] = child.pid
            job.assign(child)
            record["job_assigned_before_fixture_imports"] = True
            process = psutil.Process(child.pid)
            cpu = psutil.Process().cpu_affinity()[0]
            process.cpu_affinity([cpu])
            record["worker_cpu_affinity"] = [cpu]
            write(gate, {"token": record["token"], "utc": utc()})
            # Total Job membership and RSS descendants are enforced for the entire payload.
            while child.poll() is None:
                try:
                    members = [process, *process.children(recursive=True)]
                except psutil.NoSuchProcess:  # normal exit between poll and the process-tree snapshot
                    break
                rss = 0
                for member in members:
                    try:
                        rss += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                available = psutil.virtual_memory().available
                size = output_bytes(root)
                peak_rss, peak_output = max(peak_rss, rss), max(peak_output, size)
                minimum_available = min(minimum_available, available)
                if rss > MEMORY_CAP or available < 1.5 * 1024 ** 3 or size > OUTPUT_CAP or time.monotonic() - start >= wall_cap:
                    record["status"] = "STOP_CPU_RESOURCE_OR_WALL_LIMIT"
                    job.terminate()
                    break
                time.sleep(.1)
            record["worker_returncode"] = child.wait(timeout=10)
        final_output = output_bytes(root)
        peak_output = max(peak_output, final_output)
        if final_output > OUTPUT_CAP:
            record["status"] = "STOP_CPU_OUTPUT_LIMIT"
        record["status"] = record["status"] if record["status"].startswith("STOP_") else ("PASS" if child.returncode == 0 else "FIXTURE_FAILURE")
        if stage == "containment" and record["status"] == "STOP_CPU_RESOURCE_OR_WALL_LIMIT":
            observed = json.loads((root / "CONTAINMENT_PAYLOAD.json").read_bytes())
            grandchild = json.loads((root / "GRANDCHILD.json").read_bytes())
            dead = not psutil.pid_exists(grandchild["pid"])
            record["containment_fixture"] = {"payload_in_job": observed["in_job"],
                "grandchild_in_job": grandchild["in_job"], "grandchild_exited": dead}
            if observed["in_job"] and grandchild["in_job"] and dead:
                record["status"] = "PASS"
    except BaseException as exc:
        record["status"] = "BLOCKED" if child is None else "SUPERVISOR_FAILURE"
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
        if job:
            try:
                job.terminate()
            except BaseException as cleanup:
                record["cleanup_error"] = repr(cleanup)
        if child:
            try:
                if child.poll() is None:
                    child.kill()  # Popen's retained handle: exact owned process, even if Job assignment failed
                child.wait(timeout=10)
            except Exception as cleanup:
                record["worker_exit_unconfirmed"] = True
                record["owned_child_cleanup_error"] = repr(cleanup)
    finally:
        if job:
            job.close()
        if hardware:
            hardware.close()
        completed_sources = source_snapshot()
        source_unchanged = admitted_sources == completed_sources
        if not source_unchanged:
            record["prior_status_before_source_drift"] = record["status"]
            record["status"] = "STOP_SOURCE_DRIFT"
        record.update({"utc_end": utc(), "elapsed_seconds": time.monotonic() - start,
                       "peak_tree_rss_bytes": peak_rss, "peak_output_bytes": peak_output,
                       "minimum_available_host_bytes": minimum_available,
                       "hardware_handle_closed": hardware is not None,
                       "source_unchanged_during_worker": source_unchanged,
                       "source_completion_sha256": completed_sources,
                       "source_sha256": completed_sources})
        write(root / "OFFLINE_SUPERVISOR_RECEIPT.json", record)
    print(json.dumps({k: record.get(k) for k in ("stage", "status", "error", "elapsed_seconds", "peak_tree_rss_bytes")}, indent=2))
    return 0 if record["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("containment", "tests", "prepare", "inspect"))
    parser.add_argument("--payload", choices=("containment", "tests", "prepare", "inspect"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if args.payload:
        return payload(args.payload, args.output_dir, args.gate)
    if not args.stage:
        parser.error("--stage required")
    return supervise(args.stage, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
