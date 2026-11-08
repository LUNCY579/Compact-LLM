"""Source-bound, model-free Windows CPU guardian for the additive sibling.

Both parent and child require the reviewed Hermes interpreter with -I -B -S.
The launcher is suspended until Job assignment; a live duplicated-handle and
PID/create-time/ancestry handshake admits actual CPython before fixture imports.
No native guardian, grant, checkpoint, final bank, GPU query or model is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes as w
import datetime as dt
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL = REPO / "experiments/titan_quality_repair_v1"
SHARED = Path("C:/Users/clclc/BusinessProject/ChatBubble-agent-prompts-20261002")
HARDWARE_LOCK = SHARED / "hardware-worker.lock"
ARTIFACT_ROOT = SHARED / "run-artifacts/titan-quality-repair-v1"
OUTPUT_ROOT = ARTIFACT_ROOT / "20261005-opposite-diagnostics-v1"
REVIEWED = ARTIFACT_ROOT / "20261005-research-v2/NATIVE_QUALIFICATION_REVIEWED.json"
REVIEWED_SHA = "5052bf797cc89af24b767ce51db6738353e523c2addf5139149e56278798c073"
RUNTIME_SITE = Path("C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Lib/site-packages")
INTERPRETER = Path("C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe")
PACKAGES = {"experiments.titan_quality_repair_v1": ORIGINAL,
            "experiments.titan_quality_repair_diagnostics_v1": HERE}
EXPERIMENT = "TITAN-QUALITY-REPAIR-DIAGNOSTICS-V1"
PROFILE = "LIGHTWEIGHT_CPU_NO_MODEL"
MEMORY_CAP, OUTPUT_CAP = 512 * 2**20, 50 * 2**20
HOST_ADMISSION, HOST_RESERVE = 2 * 2**30, 3 * 2**29
WALL_CAP, CLEANUP_CAP = 900, 30
PAYLOAD_CAP = WALL_CAP - CLEANUP_CAP
CONTAINMENT_KILL_SECONDS = 3
FRAMEWORKS = frozenset(("torch", "transformers", "tensorflow", "jax", "cupy"))
STAGES = ("containment", "tests", "cases")
RETAINED_HANDLES = []
RECEIPT = "CPU_SUPERVISOR_RECEIPT.json"


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_new(path, value):
    # Publish only after complete write/fsync. Hard-link creation is exclusive;
    # no observer can consume a partial bootstrap/GO document and no old receipt
    # is replaced. Failed publication has no overwrite/rename fallback.
    path = Path(path)
    staged = path.with_name(path.name + ".staging-" + uuid.uuid4().hex)
    try:
        with staged.open("xb") as stream:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, path)
    finally:
        if staged.exists():
            staged.unlink()


def read_pinned(path, expected):
    source = Path(path).read_bytes()
    if not isinstance(expected, str) or len(expected) != 64 or digest(source) != expected:
        raise ValueError("STOP_SOURCE_BINDING:" + str(path))
    return source


class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FRAMEWORKS:
            raise ImportError("STOP_LIGHTWEIGHT_MODEL_IMPORT:" + fullname)


def require_isolated_startup():
    if any(name.split(".")[0] in FRAMEWORKS for name in sys.modules):
        raise RuntimeError("STOP_ALREADY_LOADED_FRAMEWORK")
    if "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
        raise RuntimeError("STOP_STARTUP_CUSTOMIZATION_ALREADY_LOADED")
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise RuntimeError("STOP_REQUIRE_ISOLATED_I_B_S_STARTUP")
    if sys.implementation.name != "cpython" or Path(sys.executable).resolve() != INTERPRETER.resolve():
        raise RuntimeError("STOP_REQUIRE_REVIEWED_HERMES_CPYTHON")


def reject_reparse_ancestors(path):
    for item in (Path(path), *Path(path).parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("STOP_REPARSE_PATH:" + str(item))


def source_bundle(expected=None):
    """Read bounded source only; original names and bytes must stay reviewed."""
    reviewed = json.loads(read_pinned(REVIEWED, REVIEWED_SHA))
    original_manifest = reviewed.get("source_manifest")
    if not isinstance(original_manifest, dict) or len(original_manifest) != 40:
        raise ValueError("STOP_ORIGINAL_REVIEWED_MANIFEST")
    sources, snapshot = {}, {"reviewed_proposal": REVIEWED_SHA}
    for package, directory in PACKAGES.items():
        reject_reparse_ancestors(directory)
        files = sorted(p for p in directory.iterdir() if p.is_file())
        names = {p.name for p in files}
        if directory == ORIGINAL and names != set(original_manifest):
            raise ValueError("STOP_ORIGINAL_FILE_SET_CHANGED")
        if "__init__.py" not in names or (directory == HERE and "cpu_supervise.py" not in names):
            raise ValueError("STOP_SOURCE_PACKAGE_INCOMPLETE")
        values = {}
        for path in files:
            reject_reparse_ancestors(path)
            if path.suffix.lower() not in {".py", ".md", ".json", ".txt"} or path.stat().st_size > 4 * 2**20:
                raise ValueError("STOP_SOURCE_LARGE_OR_NONCODE_FILE")
            body = path.read_bytes()
            sha = digest(body)
            if directory == ORIGINAL and sha != original_manifest[path.name]:
                raise ValueError("STOP_PENDING_NATIVE_SOURCE_CHANGED:" + path.name)
            values[path.name] = body
            snapshot[package + "/" + path.name] = sha
        sources[package] = values
    if expected is not None and snapshot != expected:
        raise ValueError("STOP_BOTH_PACKAGE_SOURCE_DRIFT")
    return {"sources": sources, "snapshot": snapshot,
            "source_manifest_sha256": digest(json_bytes(snapshot)), "reviewed": reviewed}


def snapshot():
    return source_bundle()["snapshot"]


class PinnedPackages(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Compile admitted bytes, including fixtures; never consult package pyc."""
    def __init__(self, sources):
        self.sources = sources

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FRAMEWORKS:
            raise ImportError("STOP_LIGHTWEIGHT_MODEL_IMPORT:" + fullname)
        if fullname == "experiments":
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        for package, files in self.sources.items():
            if fullname == package:
                return importlib.util.spec_from_loader(fullname, self, is_package=True)
            if fullname.startswith(package + "."):
                leaf = fullname[len(package) + 1:]
                if "." in leaf or leaf + ".py" not in files:
                    raise ImportError("STOP_UNPINNED_PACKAGE_MODULE:" + fullname)
                if package.endswith("titan_quality_repair_v1") and leaf in {
                        "native_qualification", "native_safety", "native_supervise"}:
                    raise ImportError("STOP_NATIVE_MODULE_IN_CPU_DIAGNOSTICS")
                return importlib.util.spec_from_loader(fullname, self)
        if fullname.startswith("experiments."):
            raise ImportError("STOP_OTHER_EXPERIMENT_IMPORT")
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__ == "experiments":
            module.__path__ = []
            return
        package = next(name for name in self.sources
                       if module.__name__ == name or module.__name__.startswith(name + "."))
        is_package = module.__name__ == package
        leaf = "__init__.py" if is_package else module.__name__.rsplit(".", 1)[1] + ".py"
        source = self.sources[package][leaf]
        module.__file__, module.__package__ = str(PACKAGES[package] / leaf), package
        if is_package:
            module.__path__ = []
        module._diagnostics_verified_source_sha256 = digest(source)
        exec(compile(source, module.__file__, "exec"), module.__dict__)


def install_pinned_packages(sources):
    for name in list(sys.modules):
        if name == "experiments" or name.startswith("experiments."):
            del sys.modules[name]
    finder = PinnedPackages(sources)
    sys.meta_path.insert(0, finder)
    return finder


def append_runtime_metadata(reviewed):
    """Metadata discovery precedes explicit append; site/.pth never run."""
    sites = reviewed.get("runtime_site_packages")
    if not isinstance(sites, list) or len(sites) != 1 or Path(sites[0]).resolve() != RUNTIME_SITE.resolve():
        raise ValueError("STOP_REVIEWED_SITE_PATH")
    if Path(reviewed.get("interpreter", "")).resolve() != INTERPRETER.resolve():
        raise ValueError("STOP_REVIEWED_INTERPRETER")
    reject_reparse_ancestors(RUNTIME_SITE)
    distributions = {d.metadata["Name"].lower().replace("-", "_"): d.version
                     for d in importlib.metadata.Distribution.discover(path=[str(RUNTIME_SITE)])}
    expected = reviewed.get("runtime_packages", {}).get("psutil")
    if not isinstance(expected, str) or distributions.get("psutil") != expected:
        raise RuntimeError("STOP_PSUTIL_METADATA_MISSING_OR_CHANGED")
    if str(RUNTIME_SITE) not in sys.path:
        sys.path.append(str(RUNTIME_SITE))
    import psutil
    if psutil.__version__ != expected:
        raise RuntimeError("STOP_PSUTIL_IMPORTED_VERSION_DRIFT")
    return psutil, {"site_packages": str(RUNTIME_SITE), "psutil_version": expected,
                    "site_processing": False, "startup": "-I -B -S"}


def owned_output(path, *, fresh=False):
    raw = Path(os.path.abspath(os.fspath(path)))
    reject_reparse_ancestors(raw)
    reject_reparse_ancestors(OUTPUT_ROOT)
    physical = raw.resolve()
    if (raw == OUTPUT_ROOT or not raw.is_relative_to(OUTPUT_ROOT)
            or not physical.is_relative_to(OUTPUT_ROOT.resolve())
            or ".codex" in [part.lower() for part in physical.parts]
            or any((parent / ".git").exists() for parent in (physical, *physical.parents))):
        raise ValueError("STOP_DIAGNOSTICS_FRESH_EXTERNAL_OUTPUT_ONLY")
    if fresh:
        physical.mkdir(parents=True, exist_ok=False)
    return physical


def output_bytes(root):
    total, pending = 0, [Path(root)]
    while pending:
        directory = pending.pop()
        reject_reparse_ancestors(directory)
        for path in directory.iterdir():
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise RuntimeError("STOP_OUTPUT_REPARSE_DESCENDANT")
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                total += info.st_size
            else:
                raise RuntimeError("STOP_OUTPUT_NOT_REGULAR_FILE")
    return total


def worker_environment(root):
    root = owned_output(root)
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "TMP": str(root / "tmp"), "TEMP": str(root / "tmp"),
                "HF_HOME": str(root / "cache/hf"), "HF_HUB_CACHE": str(root / "cache/hf/hub"),
                "TORCH_HOME": str(root / "cache/torch"), "XDG_CACHE_HOME": str(root / "cache"),
                "CUDA_CACHE_PATH": str(root / "cache/cuda"),
                "TORCH_EXTENSIONS_DIR": str(root / "cache/torch_extensions"),
                "TORCHINDUCTOR_CACHE_DIR": str(root / "cache/torchinductor"),
                "TRITON_CACHE_DIR": str(root / "cache/triton"), "DIAGNOSTICS_OUTPUT_DIR": str(root)})
    for name in ("tmp", "cache/hf/hub", "cache/torch", "cache/cuda", "cache/torch_extensions",
                 "cache/torchinductor", "cache/triton"):
        owned_output(root / name).mkdir(parents=True, exist_ok=False)
    return env


def kernel():
    if os.name != "nt":
        raise RuntimeError("STOP_WINDOWS_REQUIRED")
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = w.HANDLE
    k.OpenProcess.argtypes, k.OpenProcess.restype = (w.DWORD, w.BOOL, w.DWORD), w.HANDLE
    k.CloseHandle.argtypes, k.CloseHandle.restype = (w.HANDLE,), w.BOOL
    k.DuplicateHandle.argtypes = (w.HANDLE, w.HANDLE, w.HANDLE, ctypes.POINTER(w.HANDLE), w.DWORD, w.BOOL, w.DWORD)
    k.IsProcessInJob.argtypes = (w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL))
    k.QueryInformationJobObject.argtypes = (w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD))
    k.GetFinalPathNameByHandleW.argtypes = (w.HANDLE, w.LPWSTR, w.DWORD, w.DWORD)
    return k


def checked_close(k, handle):
    if handle and not k.CloseHandle(w.HANDLE(int(handle))):
        raise RuntimeError("STOP_OWNED_HANDLE_CLOSE_FAILED")


class HardwareHandle:
    def __init__(self, path, record):
        reject_reparse_ancestors(path)
        self.kernel, self.handle = kernel(), None
        k = self.kernel
        k.CreateFileW.argtypes, k.CreateFileW.restype = (w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, w.HANDLE), w.HANDLE
        handle = k.CreateFileW(str(path), 0xC0000000, 0, None, 4, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise RuntimeError("BLOCKED_HARDWARE_EXCLUSIVE_HANDLE:" + str(ctypes.get_last_error()))
        self.handle = int(handle)
        try:
            k.SetFilePointer.argtypes, k.SetFilePointer.restype = (w.HANDLE, w.LONG, ctypes.c_void_p, w.DWORD), w.DWORD
            k.SetEndOfFile.argtypes = (w.HANDLE,)
            k.WriteFile.argtypes = (w.HANDLE, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD), ctypes.c_void_p)
            k.FlushFileBuffers.argtypes = (w.HANDLE,)
            if k.SetFilePointer(handle, 0, None, 0) == 0xFFFFFFFF or not k.SetEndOfFile(handle):
                raise OSError("STOP_HARDWARE_RECORD_TRUNCATE_FAILED")
            data, written = json_bytes(record), w.DWORD()
            if not k.WriteFile(handle, data, len(data), ctypes.byref(written), None) or written.value != len(data):
                raise OSError("STOP_HARDWARE_RECORD_WRITE_FAILED")
            if not k.FlushFileBuffers(handle):
                raise OSError("STOP_HARDWARE_RECORD_FLUSH_FAILED")
        except BaseException:
            try:
                self.close()
            except BaseException:
                RETAINED_HANDLES.append(self)
            raise

    def close(self):
        if self.handle is not None:
            checked_close(self.kernel, self.handle)
            self.handle = None


class BasicLimit(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", w.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]


class IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount",
                "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class ExtendedLimit(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BasicLimit), ("IoInfo", IOCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class BasicAccounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_longlong) for name in ("TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")] + [
                (name, w.DWORD) for name in ("TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]


class Job:
    def __init__(self, cpu):
        self.kernel, self.handle = kernel(), None
        k = self.kernel
        k.CreateJobObjectW.argtypes, k.CreateJobObjectW.restype = (ctypes.c_void_p, w.LPCWSTR), w.HANDLE
        k.SetInformationJobObject.argtypes = (w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD)
        k.AssignProcessToJobObject.argtypes = (w.HANDLE, w.HANDLE)
        k.TerminateJobObject.argtypes = (w.HANDLE, w.UINT)
        handle = k.CreateJobObjectW(None, None)
        if not handle:
            raise OSError("STOP_JOB_CREATE_FAILED")
        self.handle = int(handle)
        limits = ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x200 | 0x10
        limits.BasicLimitInformation.Affinity = 1 << cpu
        limits.JobMemoryLimit = MEMORY_CAP
        if not k.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            try:
                self.close()
            except BaseException:
                RETAINED_HANDLES.append(self)
            raise OSError("STOP_JOB_LIMIT_FAILED")

    def assign(self, child):
        if not self.kernel.AssignProcessToJobObject(w.HANDLE(self.handle), w.HANDLE(int(child._handle))):
            raise OSError("STOP_SUSPENDED_JOB_ASSIGN_FAILED:" + str(ctypes.get_last_error()))

    def terminate(self):
        if not self.kernel.TerminateJobObject(w.HANDLE(self.handle), 125):
            raise OSError("STOP_OWNED_JOB_TERMINATE_FAILED")

    def accounting(self):
        accounting, limit = BasicAccounting(), ExtendedLimit()
        for category, data in ((1, accounting), (9, limit)):
            if not self.kernel.QueryInformationJobObject(w.HANDLE(self.handle), category, ctypes.byref(data), ctypes.sizeof(data), None):
                raise RuntimeError("STOP_JOB_ACCOUNTING_UNAVAILABLE")
        return {"total_processes": accounting.TotalProcesses, "active_processes": accounting.ActiveProcesses,
                "terminated_processes": accounting.TotalTerminatedProcesses,
                "peak_job_private_commit_bytes": limit.PeakJobMemoryUsed,
                "job_memory_limit_bytes": limit.JobMemoryLimit,
                "job_affinity_mask": limit.BasicLimitInformation.Affinity}

    def close(self):
        if self.handle is not None:
            checked_close(self.kernel, self.handle)
            self.handle = None


def job_members(handle):
    k = kernel()
    size = 8 + 4096 * ctypes.sizeof(ctypes.c_size_t)
    buffer, returned = ctypes.create_string_buffer(size), w.DWORD()
    if not k.QueryInformationJobObject(w.HANDLE(handle), 3, buffer, size, ctypes.byref(returned)):
        raise RuntimeError("STOP_JOB_MEMBERSHIP_UNAVAILABLE")
    assigned, count = ctypes.c_uint32.from_buffer(buffer, 0).value, ctypes.c_uint32.from_buffer(buffer, 4).value
    if assigned > 4096 or count != assigned:
        raise RuntimeError("STOP_JOB_MEMBERSHIP_INCOMPLETE")
    return [ctypes.c_size_t.from_buffer(buffer, 8 + index * ctypes.sizeof(ctypes.c_size_t)).value for index in range(count)]


@contextmanager
def duplicate_parent_handle(parent_pid, handle):
    k, duplicate = kernel(), w.HANDLE()
    parent = k.OpenProcess(0x0040, False, parent_pid)
    if not parent:
        raise RuntimeError("STOP_OWNER_PROCESS_HANDLE_UNAVAILABLE")
    try:
        if not k.DuplicateHandle(parent, w.HANDLE(handle), k.GetCurrentProcess(), ctypes.byref(duplicate), 0, False, 2):
            raise RuntimeError("STOP_OWNER_HANDLE_DUPLICATION_FAILED")
        yield k, duplicate
    finally:
        errors = []
        for owned in (duplicate.value, parent):
            try:
                checked_close(k, owned)
            except BaseException as exc:
                errors.append(repr(exc))
        if errors:
            raise RuntimeError("STOP_DUPLICATE_HANDLE_CLEANUP:" + repr(errors))


def verify_owner_handles(admission):
    with duplicate_parent_handle(admission["supervisor_pid"], admission["hardware_owner_handle"]) as (k, hardware):
        path = ctypes.create_unicode_buffer(32768)
        count = k.GetFinalPathNameByHandleW(hardware, path, len(path), 0)
        if not count or count >= len(path) or Path(path.value.removeprefix("\\\\?\\")).resolve() != HARDWARE_LOCK.resolve():
            raise RuntimeError("STOP_HARDWARE_HANDLE_WRONG_OBJECT")
        k.CreateFileW.argtypes, k.CreateFileW.restype = (w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, w.HANDLE), w.HANDLE
        probe = k.CreateFileW(str(HARDWARE_LOCK), 0xC0000000, 0, None, 3, 0x80, None)
        if probe != ctypes.c_void_p(-1).value:
            checked_close(k, probe)
            raise RuntimeError("STOP_HARDWARE_HANDLE_NOT_EXCLUSIVE")
        if ctypes.get_last_error() != 32:
            raise RuntimeError("STOP_HARDWARE_EXCLUSIVITY_UNOBSERVABLE")
    with duplicate_parent_handle(admission["supervisor_pid"], admission["job_owner_handle"]) as (k, job):
        flag = w.BOOL()
        if not k.IsProcessInJob(k.GetCurrentProcess(), job, ctypes.byref(flag)) or not flag.value:
            raise RuntimeError("STOP_ACTUAL_WORKER_OUTSIDE_OWNED_JOB")
        members = job_members(int(job.value))
        if os.getpid() not in members:
            raise RuntimeError("STOP_ACTUAL_WORKER_JOB_MEMBERSHIP")
    return members


def process_created(psutil, pid):
    try:
        return psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return None
    except psutil.AccessDenied as exc:
        raise RuntimeError("STOP_PROCESS_IDENTITY_UNAVAILABLE") from exc


def resume_suspended(child, psutil):
    threads = psutil.Process(child.pid).threads()
    if len(threads) != 1:
        raise RuntimeError("STOP_SUSPENDED_THREAD_IDENTITY")
    k = kernel()
    k.OpenThread.argtypes, k.OpenThread.restype = (w.DWORD, w.BOOL, w.DWORD), w.HANDLE
    k.ResumeThread.argtypes, k.ResumeThread.restype = (w.HANDLE,), w.DWORD
    thread = k.OpenThread(0x0002, False, threads[0].id)
    if not thread:
        raise RuntimeError("STOP_SUSPENDED_THREAD_HANDLE")
    try:
        if k.ResumeThread(thread) != 1:
            raise RuntimeError("STOP_SUSPEND_COUNT_CHANGED")
    finally:
        checked_close(k, thread)


def limits():
    return {"cpu_threads": 1, "job_private_commit_bytes": MEMORY_CAP, "tree_rss_bytes": MEMORY_CAP,
            "host_admission_bytes": HOST_ADMISSION, "host_reserve_bytes": HOST_RESERVE,
            "new_output_bytes": OUTPUT_CAP, "hard_wall_seconds": WALL_CAP,
            "payload_kill_seconds": PAYLOAD_CAP, "cleanup_seconds": CLEANUP_CAP}


def validate_admission(value, root, stage):
    if (not isinstance(value, dict) or value.get("experiment") != EXPERIMENT or value.get("stage") != stage
            or stage not in STAGES or value.get("profile") != PROFILE
            or value.get("job_assigned_before_payload") is not True
            or value.get("reviewed_proposal_sha256") != REVIEWED_SHA or value.get("limits") != limits()):
        raise ValueError("STOP_CHILD_ADMISSION_SCOPE")
    if Path(value.get("output_dir", "")).resolve() != owned_output(root):
        raise ValueError("STOP_CHILD_ADMISSION_OUTPUT")
    if value.get("hardware_lock") != str(HARDWARE_LOCK):
        raise ValueError("STOP_CHILD_ADMISSION_HARDWARE_PATH")
    for key in ("supervisor_pid", "launcher_pid", "hardware_owner_handle", "job_owner_handle"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError("STOP_CHILD_ADMISSION_IDENTITY_HANDLE")
    for key in ("supervisor_create_time", "launcher_create_time", "started_monotonic"):
        if type(value.get(key)) not in (int, float) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError("STOP_CHILD_ADMISSION_CREATE_TIME")
    if type(value.get("cpu")) is not int or not 0 <= value["cpu"] < ctypes.sizeof(ctypes.c_size_t) * 8:
        raise ValueError("STOP_CHILD_ADMISSION_CPU")
    if not isinstance(value.get("token"), str) or len(value["token"]) != 32:
        raise ValueError("STOP_CHILD_ADMISSION_TOKEN")
    if not isinstance(value.get("source_snapshot"), dict) or not value["source_snapshot"]:
        raise ValueError("STOP_CHILD_ADMISSION_SOURCE")
    return value


def validate_bootstrap(proof, admission, members, identity):
    for key in ("experiment", "stage", "token", "source_manifest_sha256", "supervisor_pid",
                "supervisor_create_time", "launcher_pid", "launcher_create_time", "hardware_owner_handle", "job_owner_handle"):
        if proof.get(key) != admission.get(key):
            raise RuntimeError("STOP_ACTUAL_WORKER_BOOTSTRAP_BINDING:" + key)
    pid, created = proof.get("worker_pid"), proof.get("worker_create_time")
    if (type(pid) is not int or type(created) not in (int, float) or not math.isfinite(created)
            or pid not in members or identity(pid) != created):
        raise RuntimeError("STOP_ACTUAL_WORKER_IDENTITY_OR_JOB")
    if proof.get("owned_handles_verified") is not True or proof.get("worker_cpu_affinity") != [admission["cpu"]]:
        raise RuntimeError("STOP_ACTUAL_WORKER_HANDLE_OR_CPU_PROOF")
    if proof.get("implementation") != "cpython" or Path(proof.get("sys_executable", "")).resolve() != INTERPRETER.resolve():
        raise RuntimeError("STOP_ACTUAL_WORKER_RUNTIME_PROOF")
    ancestors = proof.get("ancestors", [])
    launcher = (admission["launcher_pid"], admission["launcher_create_time"])
    if pid != launcher[0] and launcher not in {(p.get("pid"), p.get("create_time")) for p in ancestors if isinstance(p, dict)}:
        raise RuntimeError("STOP_ACTUAL_WORKER_NOT_OWNED_LAUNCHER_DESCENDANT")
    return {"worker_pid": pid, "worker_create_time": created}


def observe_resources(psutil, root, members, cpu):
    rss = private = 0
    identities = []
    for pid in members:
        try:
            process = psutil.Process(pid)
            created = process.create_time()
            memory, affinity = process.memory_info(), process.cpu_affinity()
            if affinity != [cpu]:
                raise RuntimeError("STOP_JOB_MEMBER_CPU_AFFINITY")
            commit = getattr(memory, "private", None)
            if type(commit) is not int or commit < 0:
                raise RuntimeError("STOP_JOB_MEMBER_PRIVATE_COMMIT_UNAVAILABLE")
            if process.create_time() != created:
                raise RuntimeError("STOP_JOB_MEMBER_IDENTITY_CHANGED")
            rss, private = rss + memory.rss, private + commit
            identities.append({"pid": pid, "create_time": created, "cpu_affinity": affinity})
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise RuntimeError("STOP_JOB_MEMBER_UNOBSERVABLE") from exc
    return {"tree_rss_bytes": rss, "tree_private_commit_bytes": private,
            "host_available_bytes": psutil.virtual_memory().available,
            "new_output_bytes": output_bytes(root), "job_members": identities}


def require_resources(value, *, preload=False):
    for name in ("tree_rss_bytes", "tree_private_commit_bytes", "host_available_bytes", "new_output_bytes"):
        if type(value.get(name)) not in (int, float) or not math.isfinite(value[name]) or value[name] < 0:
            raise RuntimeError("STOP_RESOURCE_OBSERVATION_UNAVAILABLE")
    if (value["tree_rss_bytes"] > MEMORY_CAP or value["tree_private_commit_bytes"] > MEMORY_CAP
            or value["host_available_bytes"] < (HOST_ADMISSION if preload else HOST_RESERVE)
            or value["new_output_bytes"] > OUTPUT_CAP - 2**20):
        raise RuntimeError("STOP_CPU_RESOURCE_ENVELOPE")  # reserve one MiB for complete receipts


def cleanup_owned(job, child, hardware, *, members=job_members, clock=time.monotonic,
                  sleep=time.sleep, deadline, errors=None):
    """Prove Job-zero and exact retained-launcher reap before releasing handles."""
    errors = errors if errors is not None else []
    active = []
    while job is not None:
        try:
            active = members(int(job.handle))
        except BaseException as exc:
            active = None
            errors.append("JOB_MEMBER_ACCOUNTING:" + repr(exc))
            break
        if not active or clock() >= deadline:
            break
        try:
            job.terminate()
        except BaseException as exc:
            errors.append("JOB_TERMINATE:" + repr(exc))
        sleep(min(.05, max(0, deadline - clock())))
    returncode = None
    if child is not None:
        try:
            if child.poll() is None:
                child.kill()  # exact retained Popen handle, including failed assignment
            returncode = child.wait(timeout=max(0, deadline - clock()))
        except BaseException as exc:
            errors.append("EXACT_LAUNCHER_REAP:" + repr(exc))
    reaped = child is None or (returncode is not None and child.poll() is not None)
    if job is not None:
        try:
            active = members(int(job.handle))
        except BaseException as exc:
            active = None
            errors.append("FINAL_JOB_MEMBERSHIP:" + repr(exc))
    dead, accounting = reaped and active == [], None
    if dead and job is not None:
        try:
            accounting = job.accounting()
            if accounting["active_processes"] != 0 or accounting["job_memory_limit_bytes"] != MEMORY_CAP:
                raise RuntimeError("STOP_FINAL_JOB_ACCOUNTING_MISMATCH")
            if accounting["peak_job_private_commit_bytes"] > MEMORY_CAP:
                raise RuntimeError("STOP_FINAL_JOB_PRIVATE_COMMIT_LIMIT")
        except BaseException as exc:
            errors.append("FINAL_JOB_ACCOUNTING:" + repr(exc))
    job_closed = hardware_closed = False
    if dead:
        if job is not None:
            try:
                job.close()
                job_closed = True
            except BaseException as exc:
                errors.append("JOB_CLOSE:" + repr(exc))
                RETAINED_HANDLES.append(job)
        if hardware is not None:
            if job is not None and not job_closed:
                RETAINED_HANDLES.append(hardware)
            else:
                try:
                    hardware.close()
                    hardware_closed = True
                except BaseException as exc:
                    errors.append("HARDWARE_CLOSE:" + repr(exc))
                    RETAINED_HANDLES.append(hardware)
    else:
        RETAINED_HANDLES.extend(item for item in (job, hardware, child) if item is not None)
    return {"whole_job_exit_verified": dead, "exact_launcher_reaped": reaped,
            "launcher_returncode": returncode, "job_active_pids_after_cleanup": active,
            "job_accounting": accounting, "job_handle_closed": job_closed,
            "hardware_handle_closed": hardware_closed,
            "owned_handles_retained_until_supervisor_exit": not dead or bool(errors),
            "cleanup_reconciled": dead and not errors, "cleanup_errors": errors}


def containment_pass(record, cleanup):
    return (record.get("stage") == "containment" and record.get("status") == "SUPERVISOR_STOP"
            and "STOP_CPU_HARD_WALL" in record.get("error", "")
            and record.get("hard_wall_watchdog_fired") is True
            and cleanup.get("launcher_returncode") == 125
            and cleanup.get("whole_job_exit_verified") is True
            and cleanup.get("cleanup_reconciled") is True
            and record.get("source_unchanged_during_worker") is True
            and record.get("containment_fixture", {}).get("forced_owned_job_termination") is True)


def payload(stage, root, admission_path, admission_sha):
    """Child entry; a gate's existence alone cannot admit a payload."""
    require_isolated_startup()
    sys.meta_path.insert(0, NoModelImports())
    root = owned_output(root)
    if admission_path is None or admission_sha is None or Path(admission_path).resolve() != root / "CHILD_ADMISSION.json":
        raise RuntimeError("STOP_DIRECT_PAYLOAD_WITHOUT_LIVE_OWNERSHIP")
    admission = validate_admission(json.loads(read_pinned(admission_path, admission_sha)), root, stage)
    bundle = source_bundle(admission["source_snapshot"])
    if bundle["source_manifest_sha256"] != admission.get("source_manifest_sha256"):
        raise RuntimeError("STOP_CHILD_SOURCE_MANIFEST_BINDING")
    psutil, runtime = append_runtime_metadata(bundle["reviewed"])
    identity = lambda pid: process_created(psutil, pid)
    if identity(admission["supervisor_pid"]) != admission["supervisor_create_time"]:
        raise RuntimeError("STOP_CHILD_STALE_SUPERVISOR")
    members = verify_owner_handles(admission)
    current = psutil.Process(os.getpid())
    proof = {key: admission[key] for key in ("experiment", "stage", "token", "source_manifest_sha256",
                "supervisor_pid", "supervisor_create_time", "launcher_pid", "launcher_create_time",
                "hardware_owner_handle", "job_owner_handle")}
    proof.update({"worker_pid": os.getpid(), "worker_create_time": current.create_time(),
                  "ancestors": [{"pid": p.pid, "create_time": p.create_time()} for p in current.parents()],
                  "worker_cpu_affinity": current.cpu_affinity(), "implementation": sys.implementation.name,
                  "sys_executable": sys.executable, "process_executable": current.exe(),
                  "owned_handles_verified": True, "job_member_pids": members, "runtime": runtime})
    validate_bootstrap(proof, admission, members, identity)
    write_new(root / "WORKER_BOOTSTRAP.json", proof)
    deadline = min(admission["started_monotonic"] + PAYLOAD_CAP, time.monotonic() + 30)
    go_path = root / "WORKER_GO.json"
    while not go_path.exists():
        if time.monotonic() >= deadline or identity(admission["supervisor_pid"]) != admission["supervisor_create_time"]:
            raise RuntimeError("STOP_CHILD_LIVE_GO_TIMEOUT_OR_OWNER_LOST")
        time.sleep(.03)
    go = json.loads(go_path.read_bytes())
    if (go.get("token") != admission["token"] or go.get("worker_pid") != os.getpid()
            or go.get("worker_create_time") != proof["worker_create_time"]
            or go.get("admission_sha256") != admission_sha
            or go.get("bootstrap_sha256") != digest((root / "WORKER_BOOTSTRAP.json").read_bytes())):
        raise RuntimeError("STOP_CHILD_GO_BINDING")
    verify_owner_handles(admission)
    source_bundle(admission["source_snapshot"])
    install_pinned_packages(bundle["sources"])
    return run_stage(stage, root, bundle)


def run_stage(stage, root, bundle):
    import importlib
    sibling = "experiments.titan_quality_repair_diagnostics_v1"
    if stage == "tests":
        import unittest
        modules = sorted(name[:-3] for name in bundle["sources"][sibling]
                         if name.startswith("test_") and name.endswith(".py"))
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(
                    importlib.import_module(sibling + "." + name)) for name in modules)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        write_new(root / "CPU_FIXTURES.json", {"tests_run": result.testsRun, "failures": len(result.failures),
                  "errors": len(result.errors), "passed": result.wasSuccessful(), "profile": PROFILE,
                  "native_model_loads": 0, "native_model_forwards": 0, "capability_evidence": False})
        return 0 if result.wasSuccessful() else 1
    if stage == "cases":
        result = importlib.import_module(sibling + ".cases").audit_prepared_inputs(
                    ARTIFACT_ROOT / "20261005-offline/packet-002", root)
        gains = importlib.import_module(sibling + ".gain_attribution").analyze_gain_attribution()
        if gains.get("status") != "UNMEASURED":
            raise RuntimeError("STOP_CASES_GAIN_ATTRIBUTION_MUST_REMAIN_UNMEASURED")
        write_new(root / "GAIN_ATTRIBUTION_UNMEASURED.json", gains)
        return 0 if result["status"] == "VERIFIED_INPUT_CASE_PATTERN_NO_MODEL_GAIN" else 1
    if stage == "containment":
        script = ("import ctypes,json,os,time,sys; from ctypes import wintypes as w; "
                  "k=ctypes.WinDLL('kernel32',use_last_error=True); k.GetCurrentProcess.restype=w.HANDLE; "
                  "k.IsProcessInJob.argtypes=(w.HANDLE,w.HANDLE,ctypes.POINTER(w.BOOL)); "
                  "flag=w.BOOL(); ok=k.IsProcessInJob(k.GetCurrentProcess(),None,ctypes.byref(flag)); "
                  "from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'in_job':bool(ok and flag.value)})); "
                  "time.sleep(45)")
        child = subprocess.Popen([str(INTERPRETER), "-I", "-B", "-S", "-c", script, str(root / "TINY_DESCENDANT.json")],
                                 stdin=subprocess.DEVNULL, creationflags=0x08000000)
        write_new(root / "CONTAINMENT_FIXTURE.json", {"worker_pid": os.getpid(), "descendant_launcher_pid": child.pid,
                  "fixture_only": True, "native_model_loads": 0, "native_model_forwards": 0})
        time.sleep(45)  # parent must terminate the entire owned Job at three seconds
        return 1       # natural completion cannot pass the forced-kill fixture
    raise ValueError("STOP_UNKNOWN_DIAGNOSTICS_STAGE")


def supervise(stage, path):
    require_isolated_startup()
    if stage not in STAGES:
        raise ValueError("STOP_UNKNOWN_DIAGNOSTICS_STAGE")
    sys.meta_path.insert(0, NoModelImports())
    bundle = source_bundle()
    psutil, runtime = append_runtime_metadata(bundle["reviewed"])
    root = owned_output(path, fresh=True)
    initial = bundle["snapshot"]
    write_new(root / "SOURCE_INITIAL_SNAPSHOT.json", {"source_sha256": initial, "runtime": runtime})
    identity = lambda pid: process_created(psutil, pid)
    started = time.monotonic()
    payload_cap = CONTAINMENT_KILL_SECONDS if stage == "containment" else PAYLOAD_CAP
    record = {"experiment": EXPERIMENT, "stage": stage, "profile": PROFILE, "status": "ADMISSION_PENDING",
              "utc_start": utc(), "worktree": str(REPO), "output_dir": str(root), "runtime": runtime,
              "supervisor_pid": os.getpid(), "supervisor_create_time": identity(os.getpid()),
              "source_initial_sha256": initial, "limits": limits(), "token": uuid.uuid4().hex,
              "native_model_loads": 0, "native_model_forwards": 0, "native_generations": 0,
              "paid_calls": 0, "gpu_queries": 0, "capability_evidence": False}
    record["effective_payload_kill_seconds"] = payload_cap
    hardware = job = child = timer = None
    stop_event, cleanup_errors = threading.Event(), []
    peak_rss = peak_private = peak_output = 0
    minimum_available = None
    known_members = set()
    def hard_kill():
        stop_event.set()
        try:
            if job is not None:
                job.terminate()
        except BaseException as exc:
            cleanup_errors.append("HARD_WALL_JOB_TERMINATE:" + repr(exc))
    def measured():
        nonlocal peak_rss, peak_private, peak_output, minimum_available
        active = job_members(int(job.handle)) if job is not None else []
        known_members.update(active)
        value = observe_resources(psutil, root, active, record["cpu"])
        peak_rss, peak_private = max(peak_rss, value["tree_rss_bytes"]), max(peak_private, value["tree_private_commit_bytes"])
        peak_output = max(peak_output, value["new_output_bytes"])
        minimum_available = value["host_available_bytes"] if minimum_available is None else min(minimum_available, value["host_available_bytes"])
        require_resources(value)
        record["last_resources"] = value
        if stop_event.is_set() or time.monotonic() - started >= payload_cap:
            if not stop_event.is_set():
                hard_kill()  # deterministic even if the timer callback is one scheduler tick late
            raise RuntimeError("STOP_CPU_HARD_WALL")
        return active
    try:
        cpus = psutil.Process().cpu_affinity()
        if not cpus or cpus[0] >= ctypes.sizeof(ctypes.c_size_t) * 8:
            raise RuntimeError("STOP_ONE_CPU_AFFINITY_UNAVAILABLE")
        record["cpu"] = cpus[0]
        if record["supervisor_create_time"] is None:
            raise RuntimeError("STOP_SUPERVISOR_IDENTITY_UNAVAILABLE")
        admission_resources = observe_resources(psutil, root, [], record["cpu"])
        require_resources(admission_resources, preload=True)
        record["preload_resources"] = admission_resources
        minimum_available = admission_resources["host_available_bytes"]
        hardware = HardwareHandle(HARDWARE_LOCK, {**record, "started_monotonic": started})
        source_bundle(initial)
        record["source_admission_sha256"] = initial
        write_new(root / "SOURCE_ADMISSION_SNAPSHOT.json", {"source_sha256": initial})
        # Recheck admission under the live exclusive lease immediately before
        # constructing/launching the contained worker.
        admission_resources = observe_resources(psutil, root, [], record["cpu"])
        require_resources(admission_resources, preload=True)
        record["owned_preload_resources"] = admission_resources
        minimum_available = min(minimum_available, admission_resources["host_available_bytes"])
        job = Job(record["cpu"])
        env = worker_environment(root)
        admission_path = root / "CHILD_ADMISSION.json"
        adapter_sha = initial["experiments.titan_quality_repair_diagnostics_v1/cpu_supervise.py"]
        # This fixed stdlib trampoline is in the command itself. There is no
        # unpinned command file, site startup or package import before live proof.
        trampoline = ("import hashlib,sys,types; from pathlib import Path; "
                      "p=Path(sys.argv[1]); s=p.read_bytes(); "
                      "assert hashlib.sha256(s).hexdigest()==sys.argv[2], 'STOP_ADAPTER_SELF_PIN'; "
                      "m=types.ModuleType('_diagnostics_cpu_bootstrap'); m.__file__=str(p); "
                      "sys.modules[m.__name__]=m; exec(compile(s,str(p),'exec'),m.__dict__); "
                      "a=Path(sys.argv[5]); h=hashlib.sha256(a.read_bytes()).hexdigest(); "
                      "raise SystemExit(m.payload(sys.argv[3],Path(sys.argv[4]),a,h))")
        with (root / "worker.log").open("xb") as log:
            child = subprocess.Popen([str(INTERPRETER), "-I", "-B", "-S", "-c", trampoline,
                                      str(HERE / "cpu_supervise.py"), adapter_sha, stage, str(root), str(admission_path)],
                                     cwd=str(root), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=0x00000004 | 0x08000000 | 0x00000200)
            launcher_created = identity(child.pid)
            if launcher_created is None:
                raise RuntimeError("STOP_LAUNCHER_IDENTITY_UNAVAILABLE")
            job.assign(child)
            psutil.Process(child.pid).cpu_affinity([record["cpu"]])
            admission = {"experiment": EXPERIMENT, "stage": stage, "profile": PROFILE,
                         "token": record["token"], "output_dir": str(root), "limits": limits(),
                         "reviewed_proposal_sha256": REVIEWED_SHA, "source_snapshot": initial,
                         "source_manifest_sha256": bundle["source_manifest_sha256"],
                         "supervisor_pid": os.getpid(), "supervisor_create_time": record["supervisor_create_time"],
                         "launcher_pid": child.pid, "launcher_create_time": launcher_created,
                         "hardware_lock": str(HARDWARE_LOCK), "hardware_owner_handle": int(hardware.handle),
                         "job_owner_handle": int(job.handle), "cpu": record["cpu"], "started_monotonic": started,
                         "job_assigned_before_payload": True}
            write_new(admission_path, admission)
            admission_sha = digest(admission_path.read_bytes())
            record.update({"launcher_pid": child.pid, "launcher_create_time": launcher_created,
                           "job_assigned_before_fixture_imports": True, "child_admission_sha256": admission_sha})
            timer = threading.Timer(max(0, payload_cap - (time.monotonic() - started)), hard_kill)
            timer.daemon = True
            timer.start()
            if child.poll() is not None or identity(child.pid) != launcher_created:
                raise RuntimeError("STOP_LAUNCHER_CHANGED_BEFORE_RESUME")
            resume_suspended(child, psutil)
            bootstrap_path = root / "WORKER_BOOTSTRAP.json"
            deadline = min(started + payload_cap, time.monotonic() + 30)
            while not bootstrap_path.exists():
                active = measured()
                if not active or time.monotonic() >= deadline:
                    raise RuntimeError("STOP_ACTUAL_WORKER_BOOTSTRAP_MISSING")
                time.sleep(.03)
            proof = json.loads(bootstrap_path.read_bytes())
            worker = validate_bootstrap(proof, admission, measured(), identity)
            actual_process = psutil.Process(worker["worker_pid"])
            if actual_process.exe() != proof["process_executable"] or actual_process.create_time() != worker["worker_create_time"]:
                raise RuntimeError("STOP_ACTUAL_HERMES_CPYTHON_PROCESS_CHANGED")
            record["actual_worker"] = worker
            record["source_worker_sha256"] = source_bundle(initial)["snapshot"]
            write_new(root / "WORKER_GO.json", {**worker, "token": record["token"], "admission_sha256": admission_sha,
                      "bootstrap_sha256": digest(bootstrap_path.read_bytes())})
            while measured():
                source_bundle(initial)
                if identity(os.getpid()) != record["supervisor_create_time"]:
                    raise RuntimeError("STOP_SUPERVISOR_IDENTITY_CHANGED")
                time.sleep(.05)
            record["status"] = "WORKER_EXITED"
    except BaseException as exc:
        record["status"] = "BLOCKED" if child is None else "SUPERVISOR_STOP"
        record["error"], record["traceback"] = repr(exc), traceback.format_exc()
        if job is not None:
            try:
                job.terminate()
            except BaseException as cleanup:
                cleanup_errors.append("STOP_JOB_TERMINATE:" + repr(cleanup))
    finally:
        deadline = min(time.monotonic() + CLEANUP_CAP, started + WALL_CAP)
        if timer is not None:
            timer.cancel()
            try:
                timer.join(timeout=max(0, deadline - time.monotonic()))
                if timer.is_alive():
                    raise RuntimeError("STOP_HARD_TIMER_UNREAPED")
            except BaseException as exc:
                cleanup_errors.append("HARD_TIMER_CLEANUP:" + repr(exc))
        cleanup = cleanup_owned(job, child, hardware, deadline=deadline, errors=cleanup_errors)
        record.update(cleanup)
        record["hard_wall_watchdog_fired"] = stop_event.is_set()
        try:
            completed = source_bundle(initial)["snapshot"]
            record["source_unchanged_during_worker"] = True
            record["source_completion_sha256"] = completed
            write_new(root / "SOURCE_COMPLETION_SNAPSHOT.json", {"source_sha256": completed})
        except BaseException as exc:
            record["source_unchanged_during_worker"] = False
            record["source_completion_error"], record["status"] = repr(exc), "STOP_SOURCE_DRIFT"
        try:
            final_size = output_bytes(root)
            peak_output = max(peak_output, final_size)
            if final_size > OUTPUT_CAP:
                raise RuntimeError("STOP_CPU_FINAL_OUTPUT_LIMIT")
            if stage == "containment" and record.get("actual_worker"):
                observed = json.loads((root / "TINY_DESCENDANT.json").read_bytes())
                if observed.get("in_job") is not True or observed.get("pid") not in known_members:
                    raise RuntimeError("STOP_TINY_DESCENDANT_JOB_PROOF_MISSING")
                record["containment_fixture"] = {**observed, "observed_by_job_watchdog": True,
                      "whole_job_exit_verified": cleanup["whole_job_exit_verified"],
                      "forced_owned_job_termination": stop_event.is_set(),
                      "fixture_kill_seconds": CONTAINMENT_KILL_SECONDS,
                      "capability_evidence": False}
        except BaseException as exc:
            record["output_or_fixture_accounting_error"], record["status"] = repr(exc), "STOP_FINAL_ACCOUNTING"
        if (record["status"] == "WORKER_EXITED" and cleanup["launcher_returncode"] == 0
                and cleanup["whole_job_exit_verified"] and cleanup["cleanup_reconciled"]
                and record["source_unchanged_during_worker"] and not stop_event.is_set()):
            record["status"] = "PASS"
        elif record["status"] == "WORKER_EXITED":
            record["status"] = "FIXTURE_FAILURE"
        if containment_pass(record, cleanup):
            record["expected_fixture_stop"] = record["error"]
            record["status"] = "PASS"
        if not cleanup["cleanup_reconciled"]:
            record["prior_status_before_cleanup_failure"], record["status"] = record["status"], "STOP_CLEANUP_UNRECONCILED"
        record.update({"utc_end": utc(), "elapsed_seconds": time.monotonic() - started,
                       "peak_tree_rss_bytes": peak_rss, "peak_tree_private_commit_bytes": peak_private,
                       "peak_output_bytes_before_receipt": peak_output, "minimum_available_host_bytes": minimum_available,
                       "observed_job_member_pids": sorted(known_members), "native_execution_authorized": False})
        if record["elapsed_seconds"] > WALL_CAP:
            record["status"] = "STOP_CPU_TOTAL_WALL_LIMIT"
        try:
            final_bytes = output_bytes(root) + len(json_bytes(record))
            record["final_output_bytes_upper_bound"] = final_bytes + 1024
            if final_bytes + 1024 > OUTPUT_CAP:
                record["status"] = "STOP_CPU_RECEIPT_OUTPUT_LIMIT"
            write_new(root / RECEIPT, record)
        except BaseException:
            record["status"] = "STOP_RECEIPT_ACCOUNTING_OR_WRITE"
            raise
    print(json.dumps({key: record.get(key) for key in ("stage", "status", "error", "elapsed_seconds", "peak_tree_rss_bytes")}))
    return 0 if record["status"] == "PASS" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--payload", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--admission", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--admission-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.payload:
        if args.stage or not args.admission or not args.admission_sha256:
            raise RuntimeError("STOP_DIRECT_PAYLOAD_WITHOUT_LIVE_OWNERSHIP")
        return payload(args.payload, args.output_dir, args.admission, args.admission_sha256)
    if not args.stage or args.admission or args.admission_sha256:
        parser.error("--stage is required for the supervisor")
    return supervise(args.stage, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
