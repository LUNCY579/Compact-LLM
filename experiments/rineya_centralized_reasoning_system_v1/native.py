"""CRS-owned bounded Titan diagnostic, with initially closed efficacy interfaces.

Only the diagnostic CLI can dispatch a model. It requires clean frozen source,
independent review, current exact telemetry, a host-wide lease and Windows Job
containment. Four fixed-shape forwards do not qualify decoding or training.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid

from .operations import OperationsStore, OperationsError, canonical, host_state_path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
CONTRACT = PACKAGE / "contracts/NATIVE_DIAGNOSTIC_V1.json"
GIB = 2 ** 30


class NativeBlocked(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise NativeBlocked(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=unique)
    canonical(value)
    return value


def write_new(path, value):
    path = Path(path)
    raw = canonical(value) + b"\n"
    temporary = path.with_name(".partial-" + path.name + "-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def safe_numbers(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: safe_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_numbers(v) for v in value]
    return value


def committed_snapshot():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    return {"commit": commit, "dirty": bool(dirty), "dirty_status": dirty}


def checked_contract(expected_sha256):
    require(sha(CONTRACT) == expected_sha256, "frozen diagnostic contract drift")
    cfg = read_json(CONTRACT)
    require(cfg["contract_id"] == "CRS-NATIVE-D2-V1", "wrong diagnostic identity")
    require(cfg["limits"] == {"worker_rss_gib": 2.0, "host_reserve_gib": 1.25,
            "preworker_host_gib": 3.3125, "gpu_preload_free_gib": 4.0,
            "gpu_runtime_free_gib": 0.5, "disk_free_gib": 4.0,
            "wall_seconds": 600, "watchdog_poll_seconds": 0.1}, "resource limits changed")
    require(cfg["maximum_forward_calls"] == 4 and cfg["maximum_input_positions"] == 524
            and cfg["generation_tokens"] == cfg["optimizer_steps"] == 0,
            "diagnostic dose changed")
    require(cfg["historical_capture_tolerances"] == {"max_abs": 0.03125, "mean_abs": 0.01},
            "historical numerical limits changed")
    require(cfg["reader_qualified"] is False and cfg["training_qualified"] is False,
            "diagnostic cannot grant reader/training qualification")
    return cfg


def verify_sources(cfg):
    for relative, expected in cfg["source_sha256"].items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == expected,
                "source drift: " + relative)
    for path, expected in cfg["runtime_file_sha256"].items():
        require(sha(path) == expected, "installed runtime source drift: " + path)


def verify_assets(cfg):
    root = Path(cfg["model_dir"]).resolve()
    require(root.is_dir(), "registered external model directory missing")
    for name, expected in cfg["asset_sha256"].items():
        require(Path(name).name == name and sha(root / name) == expected, "asset drift: " + name)


def runtime_metadata():
    return {"executable": str(Path(sys.executable).resolve()), "python": sys.version.split()[0],
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "safetensors", "tokenizers", "psutil")}}


def gpu_snapshot(allowed_worker_pids=(), desktop_identities=()):
    """Real nvidia-smi identity; missing process attribution blocks admission."""
    import psutil
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu",
                                  "--format=csv,noheader,nounits"], text=True, timeout=3).strip()
    gpu_observed_at = time.time()
    fields = [part.strip() for part in raw.split(",")]
    require(len(fields) == 3 and "\n" not in raw, "one unambiguous GPU required")
    free, utilization = float(fields[1]) / 1024, float(fields[2])
    require(math.isfinite(free) and math.isfinite(utilization), "nonfinite GPU telemetry")
    process_raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                         "--format=csv,noheader,nounits"], text=True, timeout=3).strip()
    processes_observed_at = time.time()
    processes, unresolved, conflicts, unclassified = [], [], [], []
    registry = {item["path"].replace("\\", "/").casefold(): item for item in desktop_identities}
    for line in process_raw.splitlines():
        pid_text, name = [part.strip() for part in line.split(",", 1)]
        pid = int(pid_text)
        if "insufficient" in name.lower() or "[n/a]" in name.lower():
            try:
                name = psutil.Process(pid).exe()
            except psutil.NoSuchProcess:
                # Do not infer that the GPU entry disappeared; refresh next poll.
                unresolved.append(pid)
            except (psutil.AccessDenied, OSError):
                try:
                    # Read-only CIM can expose a protected process's name when
                    # executable-path access is denied under WDDM. No escalation.
                    command = ("Get-CimInstance Win32_Process -Filter 'ProcessId = " + str(pid)
                               + "' | Select-Object Name,ExecutablePath | ConvertTo-Json -Compress")
                    info = json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command", command],
                                      text=True, timeout=3).strip())
                    candidate = info.get("ExecutablePath") or info.get("Name")
                    require(isinstance(candidate, str) and candidate.lower().endswith(".exe"),
                            "CIM process identity unavailable")
                    name = candidate
                except (OSError, ValueError, subprocess.SubprocessError, NativeBlocked, AttributeError):
                    unresolved.append(pid)
        identity = {"pid": pid, "name": name}
        if pid not in allowed_worker_pids and pid not in unresolved:
            registered = registry.get(name.replace("\\", "/").casefold())
            if registered is None:
                unclassified.append(pid)
            else:
                try:
                    if sha(name) != registered["sha256"]:
                        unclassified.append(pid)
                    elif any(sha(path) != expected for path, expected in registered.get("dependency_sha256", {}).items()):
                        unclassified.append(pid)
                    else:
                        identity.update(role=registered["role"], executable_sha256=registered["sha256"])
                except OSError:
                    unresolved.append(pid)
        processes.append(identity)
        if pid not in allowed_worker_pids and any(term in name.lower() for term in
                         ("python", "torch", "llama", "ollama", "vllm", "comfy", "train")):
            conflicts.append(pid)
    return {"sampled_at": min(gpu_observed_at, processes_observed_at),
            "gpu_observed_at": gpu_observed_at, "processes_observed_at": processes_observed_at,
            "identity_checks_completed_at": time.time(), "gpu_uuid": fields[0], "gpu_free_gib": free,
            "gpu_utilization_percent": utilization, "processes": processes,
            "unresolved_gpu_pids": unresolved, "competing_gpu_pids": conflicts,
            "unclassified_gpu_pids": unclassified}


def collect_telemetry(output=PACKAGE, allowed_worker_pids=(), desktop_identities=()):
    import psutil
    snapshot = gpu_snapshot(allowed_worker_pids, desktop_identities)
    snapshot.update(host_available_gib=psutil.virtual_memory().available / GIB,
                    host_observed_at=time.time())
    snapshot.update(disk_free_gib=psutil.disk_usage(str(output)).free / GIB,
                    disk_observed_at=time.time(), platform=os.name, process_inventory_available=True)
    return snapshot


def admission_reasons(cfg, telemetry, *, current_time=None, postimport_rss_gib=None):
    """Pure decision boundary; CPU fixtures are never native qualification."""
    now = time.time() if current_time is None else current_time
    reasons = []
    required = {"sampled_at", "host_available_gib", "gpu_free_gib", "disk_free_gib",
                "unresolved_gpu_pids", "competing_gpu_pids", "unclassified_gpu_pids", "process_inventory_available", "gpu_uuid"}
    if not isinstance(telemetry, dict) or not required <= telemetry.keys():
        return ["required current telemetry unavailable"]
    for field in ("sampled_at", "host_available_gib", "gpu_free_gib", "disk_free_gib"):
        value = telemetry[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            reasons.append("invalid telemetry: " + field)
    if reasons:
        return reasons
    if not 0 <= now - telemetry["sampled_at"] <= 5:
        reasons.append("telemetry stale or future-dated")
    if telemetry["gpu_uuid"] != cfg["gpu_uuid"]:
        reasons.append("GPU identity mismatch")
    if telemetry["process_inventory_available"] is not True or telemetry["unresolved_gpu_pids"]:
        reasons.append("GPU process attribution unavailable")
    if telemetry["competing_gpu_pids"]:
        reasons.append("competing native GPU process")
    if telemetry["unclassified_gpu_pids"]:
        reasons.append("unregistered GPU executable identity")
    limits = cfg["limits"]
    host_floor = limits["preworker_host_gib"]
    if postimport_rss_gib is not None:
        if not math.isfinite(postimport_rss_gib) or postimport_rss_gib > limits["worker_rss_gib"]:
            reasons.append("worker RSS over limit")
        host_floor = limits["host_reserve_gib"] + max(0, limits["worker_rss_gib"] - postimport_rss_gib)
    for field, floor in (("host_available_gib", host_floor),
                         ("gpu_free_gib", limits["gpu_preload_free_gib"]),
                         ("disk_free_gib", limits["disk_free_gib"])):
        if telemetry[field] < floor:
            reasons.append(field + " below admitted floor")
    return reasons


def review_reasons(review, cfg_sha256, snapshot, native_sha256):
    expected = {"status": "APPROVED_FOR_BOUNDED_DIAGNOSTIC", "source_commit": snapshot["commit"],
                "contract_sha256": cfg_sha256, "native_source_sha256": native_sha256}
    if not isinstance(review, dict) or any(review.get(k) != v for k, v in expected.items()):
        return ["independent review missing or source/contract binding mismatch"]
    reviewer = review.get("reviewer")
    if (not isinstance(reviewer, str) or not reviewer.strip()
            or reviewer.strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] == "runtime_audit"):
        return ["builder cannot approve native contract"]
    return []


def preflight(*, contract_sha256=None, review_path=None):
    """Read-only report; even READY permits the registered diagnostic only."""
    reasons, observations = [], {}
    try:
        require(contract_sha256 is not None, "frozen contract SHA256 required")
        cfg = checked_contract(contract_sha256)
        snapshot = committed_snapshot(); observations["source"] = snapshot
        if snapshot["dirty"]:
            reasons.append("clean committed source required")
        verify_sources(cfg); verify_assets(cfg)
        observed = runtime_metadata(); observations["runtime"] = observed
        if observed != cfg["runtime"]:
            reasons.append("exact pinned Python/packages required")
        if os.name != "nt":
            reasons.append("registered Windows containment required")
        review = read_json(review_path) if review_path else None
        reasons.extend(review_reasons(review, contract_sha256, snapshot, sha(__file__)))
        observations["host"] = collect_telemetry(desktop_identities=cfg["gpu_process_registry"])
        reasons.extend(admission_reasons(cfg, observations["host"]))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, NativeBlocked,
            importlib.metadata.PackageNotFoundError) as exc:
        reasons.append(str(exc))
    return {"status": "BLOCKED" if reasons else "READY_DIAGNOSTIC_ONLY", "reasons": reasons,
            "observations": observations, "reader_qualified": False, "training_qualified": False,
            "native_launches_this_preflight": 0}


def causal_plan(prompt, generated):
    require(isinstance(prompt, list) and len(prompt) == 128 and generated == [16, 22, 151645],
            "exact failed mechanical trace required")
    require(all(type(t) is int and 0 <= t < 151936 for t in prompt + generated), "invalid tokens")
    full = prompt + generated
    rows = [{"stage": "baseline", "ids": full, "protected_count": 3, "mutation_start": None}]
    for start in (128, 129, 130):
        rows.append({"stage": "mutate_suffix_" + str(start),
                     "ids": full[:start] + [17, 23, 151646][start - 128:],
                     "protected_count": start - 127, "mutation_start": start})
    return rows


def numerical_decision(max_abs, mean_abs, *, bitwise, left_sha256=None, right_sha256=None):
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                for v in (max_abs, mean_abs)), "nonfinite/invalid numerical result")
    raw_equal = (isinstance(left_sha256, str) and len(left_sha256) == 64
                 and left_sha256 == right_sha256)
    return {"passed": bitwise is True and raw_equal and max_abs == mean_abs == 0,
            "fixed_shape_bitwise_required": True, "max_abs": max_abs, "mean_abs": mean_abs,
            "left_raw_sha256": left_sha256, "right_raw_sha256": right_sha256,
            "raw_byte_identity": raw_equal, "capture_qualified": False}


def causal_mask_reference_bytes():
    """Complete registered BF16 little-endian 131x131 triangular mask."""
    return b"".join(struct.pack("<H", 0 if column <= row else 65407)
                    for row in range(131) for column in range(131))


def validate_token_fields(request):
    """Syntax boundary only; this function never licenses a model invocation."""
    require(isinstance(request, dict) and set(request) == {"input_ids", "attention_mask", "max_new_tokens"},
            "native reader accepts prompt token fields only; evaluator/gold fields rejected")
    ids, mask, cap = request["input_ids"], request["attention_mask"], request["max_new_tokens"]
    require(type(ids) is list and 1 <= len(ids) <= 512 and all(type(t) is int and 0 <= t < 151936 for t in ids),
            "reader input outside token contract")
    require(type(mask) is list and mask == [1] * len(ids) and all(type(v) is int for v in mask),
            "reader requires unpadded single sequence")
    require(type(cap) is int and 1 <= cap <= 64 and len(ids) + cap <= 576, "reader output envelope exceeded")
    return {"input_ids": list(ids), "attention_mask": list(mask), "max_new_tokens": cap}


def validate_reader_request(request, qualification=None):
    validate_token_fields(request)
    # A caller-provided status dictionary is not checkpoint qualification. The
    # future reader must validate an independently reviewed immutable evidence
    # bundle and its exact source/model/config/tokenizer/shape/runtime identities.
    raise NativeBlocked("reader512+64 closed: no verified exact 576-token reader evidence/executor exists")


def training_eligibility(*_args, **_kwargs):
    return {"eligible": False, "reason": "native training, update scope, export and complete cross-process resume unqualified"}


def validate_owned_job_flags(flags, member):
    require(type(flags) is int and member is True and flags & 0x2000 and not flags & (0x800 | 0x1000),
            "exact owned worker Job lacks strict kill-on-close/no-breakaway")


def current_windows_job_witness(query_handle, target_pid, target_creation_time):
    """Verify exact supervisor Job, then close its temporary child query handle.

    NULL resolves the process's immediate nested Job, which may be another Job.
    Its flags are retained as observations and never substitute for own identity.
    """
    require(os.name == "nt", "Windows Job containment required")
    require(type(query_handle) is int and 0 < query_handle < 2 ** 64,
            "valid duplicated Job query handle required")
    import psutil
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
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)]
    kernel.IsProcessInJob.restype = w.BOOL
    kernel.QueryInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p]
    kernel.QueryInformationJobObject.restype = w.BOOL
    kernel.GetHandleInformation.argtypes = [w.HANDLE, c.POINTER(w.DWORD)]
    kernel.GetHandleInformation.restype = w.BOOL
    kernel.CloseHandle.argtypes = [w.HANDLE]; kernel.CloseHandle.restype = w.BOOL
    result = None
    try:
        require(type(target_pid) is int and target_pid == os.getpid(),
                "query handle must belong to exact observed worker PID")
        require(psutil.Process().create_time() == target_creation_time, "query handle target creation identity drift")
        handle_flags = w.DWORD()
        require(kernel.GetHandleInformation(query_handle, c.byref(handle_flags))
                and not handle_flags.value & 1, "owned query handle must be noninheritable")
        member = w.BOOL()
        require(kernel.IsProcessInJob(kernel.GetCurrentProcess(), query_handle, c.byref(member)) and member.value,
                "worker not contained in exact supervisor-owned Job")
        limits = Extended()
        require(kernel.QueryInformationJobObject(query_handle, 9, c.byref(limits), c.sizeof(limits), None),
                "exact owned Job limits unavailable")
        flags = int(limits.BasicLimitInformation.LimitFlags)
        validate_owned_job_flags(flags, bool(member.value))
        immediate = Extended()
        immediate_ok = bool(kernel.QueryInformationJobObject(None, 9, c.byref(immediate), c.sizeof(immediate), None))
        result = {"pid": os.getpid(), "creation_time": target_creation_time,
                  "in_exact_supervisor_job": True, "exact_owned_limit_flags": flags,
                  "kill_on_close": True, "breakaway_allowed": False, "query_handle_inheritable": False,
                  "query_handle_access": "JOB_OBJECT_QUERY_ONLY",
                  "immediate_null_query_ok": immediate_ok,
                  "immediate_null_limit_flags": int(immediate.BasicLimitInformation.LimitFlags) if immediate_ok else None,
                  "observation_api": "IsProcessInJob(explicit duplicate)/QueryInformationJobObject(explicit duplicate)"}
    finally:
        require(kernel.CloseHandle(query_handle), "owned query handle failed to close before framework")
    result["query_handle_closed_before_framework"] = True
    return result


def close_unconsumed_query_handle(handle):
    import ctypes as c
    from ctypes import wintypes as w
    require(type(handle) is int and 0 < handle < 2 ** 64, "invalid unconsumed query handle")
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.CloseHandle.argtypes = [w.HANDLE]; kernel.CloseHandle.restype = w.BOOL
    require(kernel.CloseHandle(handle), "unconsumed owned query handle failed to close")


def duplicate_owned_job_query_handle(job, target):
    """Copy QUERY-only, noninheritable access into an identified own runtime."""
    import ctypes as c
    from ctypes import wintypes as w
    import psutil
    process = psutil.Process(target["pid"])
    require(process.create_time() == target["creation_time"] and process.exe() == target["executable"],
            "query-handle target process identity drift")
    require(target["pid"] in job_process_ids(job), "query-handle target not in exact owned Job")
    kernel = job.kernel
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]; kernel.OpenProcess.restype = w.HANDLE
    kernel.DuplicateHandle.argtypes = [w.HANDLE, w.HANDLE, w.HANDLE, c.POINTER(w.HANDLE), w.DWORD, w.BOOL, w.DWORD]
    kernel.DuplicateHandle.restype = w.BOOL
    receiver = kernel.OpenProcess(0x40, False, target["pid"])
    require(receiver, "owned query-handle target OpenProcess failed")
    duplicate = w.HANDLE()
    try:
        require(kernel.DuplicateHandle(kernel.GetCurrentProcess(), job.handle, receiver,
                    c.byref(duplicate), 4, False, 0), "query-only Job handle duplication failed")
    finally:
        kernel.CloseHandle(receiver)
    return {"own_job_query_handle": int(duplicate.value), "query_handle_target_pid": target["pid"],
            "query_handle_target_creation_time": target["creation_time"],
            "duplicate_access_mask": 4, "duplicate_inheritable": False}


def job_process_ids(job):
    import ctypes as c
    from ctypes import wintypes as w
    class Pids(c.Structure):
        _fields_ = [("assigned", w.DWORD), ("returned", w.DWORD), ("pids", c.c_size_t * 128)]
    kernel = job.kernel
    kernel.QueryInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p]
    kernel.QueryInformationJobObject.restype = w.BOOL
    values = Pids()
    require(kernel.QueryInformationJobObject(job.handle, 3, c.byref(values), c.sizeof(values), None),
            "owned complete Job process list unavailable")
    require(values.assigned == values.returned <= 128, "owned Job process list incomplete")
    return [int(values.pids[i]) for i in range(values.returned)]


def job_member_witness(job, cfg):
    import ctypes as c
    from ctypes import wintypes as w
    import psutil
    kernel = job.kernel
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)]
    kernel.IsProcessInJob.restype = w.BOOL
    permitted = {p.replace("\\", "/").casefold(): expected
                 for p, expected in cfg["runtime_file_sha256"].items() if p.lower().endswith(".exe")}
    witnesses = []
    for pid in job_process_ids(job):
        try:
            process = psutil.Process(pid)
            executable = process.exe()
            creation_time = process.create_time()
        except psutil.NoSuchProcess:
            confirm_owned_process_exit(job, pid)
            continue
        expected = permitted.get(executable.replace("\\", "/").casefold())
        require(expected and sha(executable) == expected, "unregistered owned runtime executable")
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle and pid not in job_process_ids(job):
            confirm_owned_process_exit(job, pid, creation_time)
            continue
        require(handle, "owned live process membership handle unavailable")
        try:
            member = w.BOOL()
            require(kernel.IsProcessInJob(handle, job.handle, c.byref(member)) and member.value,
                    "owned runtime escaped exact Job")
        finally:
            kernel.CloseHandle(handle)
        witnesses.append({"pid": pid, "creation_time": creation_time,
                          "executable": executable, "executable_sha256": expected,
                          "exact_owned_job_member": True})
    return witnesses


def confirm_owned_process_exit(job, pid, creation_time=None):
    """A missing process sample requires refreshed exact Job and exit evidence."""
    import psutil
    require(pid not in job_process_ids(job), "vanished runtime still listed in exact owned Job")
    try:
        current = psutil.Process(pid)
        now_creation = current.create_time()
        require(creation_time is not None and now_creation != creation_time,
                "missing owned sample still names a live process; exit not proven")
        reused = True
    except psutil.NoSuchProcess:
        reused = False
    return {"pid": pid, "recorded_creation_time": creation_time,
            "refreshed_exact_job_absent": True, "original_process_exited": True,
            "pid_reused": reused, "observed_at": time.time()}


def owned_job_rss_snapshot(job, cfg):
    """RSS for exact verified members; no parent.children traversal race."""
    import psutil
    members = job_member_witness(job, cfg)
    rss, exits = 0, []
    for witness in members:
        try:
            process = psutil.Process(witness["pid"])
            require(process.create_time() == witness["creation_time"], "owned RSS process identity drift")
            rss += process.memory_info().rss
        except psutil.NoSuchProcess:
            exits.append(confirm_owned_process_exit(job, witness["pid"], witness["creation_time"]))
    return {"members": members, "rss_bytes": rss, "confirmed_exit_races": exits}


def read_event_journal(path):
    path = Path(path)
    require(path.stat().st_size <= 16 * 1024 ** 2, "bounded event journal exceeded")
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), "partial final journal record")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate journal field")
            result[key] = value
        return result
    records = [json.loads(line, object_pairs_hook=unique) for line in raw.splitlines()]
    canonical(records)
    return records


def validate_diagnostic_completion(result, cfg, source_commit, events=None):
    """Externally verify complete evidence; disappearance alone is never success."""
    require(type(result) is dict and result.get("status") == "COMPLETE_FIXED_SHAPE_DIAGNOSTIC_ONLY"
            and result.get("source_commit") == source_commit and not result.get("error"),
            "complete current-source worker result required")
    require(result.get("model_loads") == 1 and result.get("optimizer_steps") == result.get("generation_tokens") == 0
            and all(type(result.get(k)) is int for k in ("model_loads", "optimizer_steps", "generation_tokens"))
            and all(result.get(k) is False for k in ("native_qualified", "capture_qualified", "reader_qualified", "training_qualified")),
            "diagnostic dose or qualification claim mismatch")
    ledger = result.get("call_accounting", {})
    require(type(events) is list and events, "durable operation journal required for completion")
    adapted = []
    previous_time = None
    for record in events:
        require(type(record) is dict and type(record.get("kind")) is str
                and type(record.get("time")) in (int, float) and math.isfinite(record["time"]),
                "invalid durable journal record/time")
        require(previous_time is None or record["time"] >= previous_time, "journal wall time moved backward")
        previous_time = record["time"]
        adapted.append({"event": record["kind"], **{k: v for k, v in record.items() if k not in {"kind", "time"}}})
    from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import verify_call_journal
    try:
        verify_call_journal(adapted, snapshot=ledger, maximum_optimizer_calls=0)
    except (RuntimeError, ValueError, TypeError, KeyError) as error:
        raise NativeBlocked("durable operation journal/result mismatch: " + str(error)) from error
    resources = [{k: v for k, v in event.items() if k not in {"kind", "time"}}
                 for event in events if event["kind"] == "resource"]
    require(resources == result.get("resources") and resources and resources[-1].get("stage") == "complete",
            "complete durable resource evidence/result mismatch")
    require(ledger.get("last_call_id") == 316 and ledger.get("forward_input_positions_attempted") == 524
            and ledger.get("forward_input_positions_returned") == 524 and ledger.get("pending_calls") == [],
            "diagnostic accounting incomplete/pending")
    require(all(type(ledger.get(k)) is int for k in ("last_call_id", "forward_input_positions_attempted", "forward_input_positions_returned")),
            "accounting counters must be integers, not booleans")
    require(all(type(ledger.get(k)) is int and ledger[k] == 0 for k in
                ("max_optimizer_calls", "training_forward_input_positions_attempted", "training_forward_input_positions_returned")),
            "unexpected training/optimizer accounting")
    expected = {"model_construct": 1, "checkpoint_load": 1, "tensor_copy": 310, "forward": 4,
                "backward": 0, "optimizer_step": 0, "checkpoint_save": 0, "checkpoint_reload": 0}
    operations = ledger.get("operations", {})
    require(set(operations) == set(expected), "diagnostic operation schema mismatch")
    for operation, count in expected.items():
        require(type(operations[operation]) is dict and all(type(v) is int for v in operations[operation].values())
                and operations[operation] == {"attempted": count, "returned": count, "failed": 0, "pending": 0},
                "failed/pending or incomplete diagnostic operation: " + operation)
    comparisons = result.get("comparisons", [])
    require(type(comparisons) is list and len(comparisons) == 6, "all six comparisons required")
    expected_comparisons = {(start, kind) for start in (128, 129, 130) for kind in ("logits", "hidden")}
    observed = set()
    for row in comparisons:
        identity = (row.get("mutation_start"), row.get("kind"))
        require(identity in expected_comparisons and identity not in observed, "comparison coverage mismatch")
        observed.add(identity)
        require(row.get("protected_positions") == list(range(127, identity[0]))
                and row.get("passed") is True and row.get("raw_byte_identity") is True
                and type(row.get("left_raw_sha256")) is str and re.fullmatch(r"[0-9a-f]{64}", row["left_raw_sha256"])
                and row["left_raw_sha256"] == row.get("right_raw_sha256")
                and all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) and row[k] == 0
                        for k in ("max_abs", "mean_abs")), "failed/nonfinite raw numerical comparison")
    trials = result.get("trials", [])
    require(type(trials) is list and len(trials) == 4 and trials[0].get("stage") == "baseline", "four raw trial witnesses required")
    measurement_events = [event for event in events if event["kind"] == "measurement"]
    try:
        durable_measurements_match = (canonical([event.get("measurement") for event in measurement_events]) == canonical(trials))
    except (ValueError, TypeError) as error:
        raise NativeBlocked("nonfinite/noncanonical raw measurement payload") from error
    require(len(measurement_events) == 4
            and all(set(event) == {"kind", "time", "measurement"} for event in measurement_events)
            and durable_measurements_match,
            "four durable measurements must exactly bind saved trial payloads")
    require(trials[0].get("logits_sha256") == cfg["reference_bulk_logits_sha256"]
            and trials[0].get("hidden_sha256") == cfg["reference_bulk_hidden_sha256"]
            and trials[0].get("model_logprobs") == cfg["reference_bulk_model_logprobs"], "historical bulk baseline drift")
    mask_sha = hashlib.sha256(causal_mask_reference_bytes()).hexdigest()
    position_sha = hashlib.sha256(struct.pack("<131q", *range(131))).hexdigest()
    expected_representation = {"rows": 131, "columns": 131, "allowed_condition": "column <= row",
          "allowed_uint16": 0, "forbidden_uint16": 65407, "allowed_value": 0.0,
          "forbidden_value": -(2 - 2 ** -7) * 2 ** 127}
    plan = causal_plan(cfg["prompt_ids"], cfg["generated_ids"])
    for index, trial in enumerate(trials):
        registered = plan[index]
        require(type(trial) is dict and set(trial) == {"stage", "logits_sha256", "hidden_sha256", "model_logprobs",
                                                      "mutation_start", "protected_positions", "witnesses"},
                "raw trial schema mismatch")
        require(trial["stage"] == registered["stage"]
                and canonical(trial["mutation_start"]) == canonical(registered["mutation_start"])
                and type(trial["protected_positions"]) is list
                and all(type(v) is int for v in trial["protected_positions"])
                and trial["protected_positions"] == list(range(127, 127 + registered["protected_count"])),
                "trial identity/mutation/protected positions differ from registered plan")
        require(all(type(trial[k]) is str and re.fullmatch(r"[0-9a-f]{64}", trial[k])
                    for k in ("logits_sha256", "hidden_sha256")), "raw full trial SHA256 invalid")
        require(type(trial["model_logprobs"]) is list and len(trial["model_logprobs"]) == 3
                and all(type(v) in (int, float) and math.isfinite(v) and v <= 0 for v in trial["model_logprobs"]),
                "three finite raw model log-probabilities required")
        witnesses = trial.get("witnesses", {})
        counts = witnesses.get("counts", {})
        mask, positions = witnesses.get("mask", {}), witnesses.get("positions", {})
        shape = mask.get("shape", [])
        representation = mask.get("complete_representation", {})
        require(type(counts) is dict and all(type(v) is int for v in counts.values())
                and counts == {"mask": 1, "rotary": 1, "norm": 1}
                and type(shape) is list and all(type(v) is int for v in shape) and shape == [1, 1, 131, 131]
                and mask.get("dtype") == "bfloat16" and mask.get("byte_order") == "little_endian"
                and mask.get("raw_sha256") == mask_sha and mask.get("reference_raw_sha256") == mask_sha
                and mask.get("actual_values_match_representation") is True
                and representation == expected_representation
                and all(type(representation.get(k)) is int for k in ("rows", "columns", "allowed_uint16", "forbidden_uint16"))
                and all(type(representation.get(k)) is float for k in ("allowed_value", "forbidden_value")),
                "complete actual causal-mask witness mismatch")
        pos_ids, pos_shape = positions.get("position_ids"), positions.get("shape", [])
        require(type(pos_ids) is list and len(pos_ids) == 1 and type(pos_ids[0]) is list
                and all(type(v) is int for v in pos_ids[0]) and pos_ids == [list(range(131))]
                and type(pos_shape) is list and all(type(v) is int for v in pos_shape) and pos_shape == [1, 131]
                and positions.get("dtype") == "int64" and positions.get("byte_order") == "little_endian"
                and positions.get("raw_sha256") == position_sha, "complete actual rotary-position witness mismatch")
    for name in ("peak_gpu_allocated_gib", "peak_gpu_reserved_gib", "sampled_min_gpu_free_gib"):
        value = result.get(name)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "allocator/GPU telemetry incomplete")
    require(result["sampled_min_gpu_free_gib"] >= .5, "GPU runtime floor violated")
    return {"complete_worker_accounting_verified": True, "raw_comparisons_verified": 6,
            "trial_witnesses_verified": 4, "durable_journal_verified": True,
            "durable_journal_records": len(events), "capture_qualified": False, "reader_qualified": False}


def resume_suspended_primary_thread(process):
    """Resume only the exact initial thread after its process is contained."""
    import ctypes as c
    from ctypes import wintypes as w
    class Thread(c.Structure):
        _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("tid", w.DWORD), ("pid", w.DWORD),
                    ("priority", w.LONG), ("delta", w.LONG), ("flags", w.DWORD)]
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
    kernel.Thread32First.argtypes = [w.HANDLE, c.POINTER(Thread)]; kernel.Thread32First.restype = w.BOOL
    kernel.Thread32Next.argtypes = [w.HANDLE, c.POINTER(Thread)]; kernel.Thread32Next.restype = w.BOOL
    kernel.OpenThread.argtypes = [w.DWORD, w.BOOL, w.DWORD]; kernel.OpenThread.restype = w.HANDLE
    kernel.ResumeThread.argtypes = [w.HANDLE]; kernel.ResumeThread.restype = w.DWORD
    kernel.CloseHandle.argtypes = [w.HANDLE]; kernel.CloseHandle.restype = w.BOOL
    snapshot = kernel.CreateToolhelp32Snapshot(4, 0)
    require(snapshot and snapshot != c.c_void_p(-1).value, "thread snapshot unavailable")
    tids = []
    try:
        entry = Thread(); entry.size = c.sizeof(entry)
        more = kernel.Thread32First(snapshot, c.byref(entry))
        while more:
            if entry.pid == process.pid:
                tids.append(int(entry.tid))
            more = kernel.Thread32Next(snapshot, c.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    require(len(tids) == 1 and process.poll() is None, "one live suspended initial thread required")
    thread = kernel.OpenThread(0x0002, False, tids[0])
    require(thread, "owned primary thread handle unavailable")
    try:
        previous = int(kernel.ResumeThread(thread))
        require(previous == 1, "primary thread was not exactly suspended once")
    finally:
        kernel.CloseHandle(thread)
    return {"pid": process.pid, "thread_id": tids[0], "prior_suspend_count": previous}


def drain_owned_job(job):
    """Terminate only this owned Job and prove every associated process exited."""
    require(job.kernel.TerminateJobObject(job.handle, 2), "owned Job termination failed")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        active = job_process_ids(job)
        if not active:
            return {"owned_tree_exited": True, "remaining_owned_pids": [], "observed_at": time.time()}
        time.sleep(.02)
    raise NativeBlocked("owned Job still has active descendants; outcome unknown")


def containment_cpu_smoke(cfg):
    """Actual bounded stdlib subprocess tree; no framework or model import."""
    require(os.name == "nt", "Windows containment smoke requires actual Windows")
    import psutil
    from experiments.opallo_native_collector_v1.supervisor import WindowsJob
    child_code = "import time; time.sleep(30)"
    parent_code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-B','-c'," + repr(child_code) + "]); time.sleep(30)"
    process = job = None
    evidence = {"scope": "ACTUAL_WINDOWS_STDLIB_CONTAINMENT_ONLY", "model_loads": 0,
                "framework_imports": 0, "gpu_calls": 0, "worker_executable": sys.executable}
    try:
        process = subprocess.Popen([sys.executable, "-B", "-c", parent_code],
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                  creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x4)
        job = WindowsJob(process)
        evidence["before_resume"] = job_member_witness(job, cfg)
        require(job_process_ids(job) == [process.pid] and not psutil.Process(process.pid).children(recursive=True),
                "suspended parent unexpectedly created a child")
        evidence["resume"] = resume_suspended_primary_thread(process)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            pids = job_process_ids(job)
            if len(pids) >= 4:  # venv launcher + runtime, then child launcher + runtime.
                break
            time.sleep(.02)
        evidence["runtime_tree"] = job_member_witness(job, cfg)
        require(len(evidence["runtime_tree"]) >= 4, "actual descendant smoke tree incomplete")
        evidence["exit"] = drain_owned_job(job)
        process.wait(timeout=5)
        evidence["status"] = "PASS_CONTAINMENT_ONLY"
        return evidence
    finally:
        if job:
            job.close()
        if process is not None and process.poll() is None:
            process.kill(); process.wait(timeout=5)


def exact_job_query_cpu_probe(cfg, output, *, rejection=None):
    """Bounded real child query/close/parent-death control; no framework/model.

    The child emits its receipt before sleeping. Parent closes its last owned Job
    handle and observes every previously verified member exiting automatically.
    """
    require(os.name == "nt" and rejection in {None, "target_pid", "target_creation"}, "registered CPU probe mode")
    import psutil
    from experiments.opallo_native_collector_v1.supervisor import WindowsJob
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    child_receipt = output / "CHILD_QUERY_RECEIPT.json"
    child_code = r'''
import ctypes as c,json,os,subprocess,sys,time
from ctypes import wintypes as w
from experiments.rineya_centralized_reasoning_system_v1.native import current_windows_job_witness,write_new
info=json.loads(sys.stdin.readline());handle=info['own_job_query_handle']
result={'pid':os.getpid(),'framework_imported':'torch' in sys.modules,'model_loads':0,'gpu_calls':0}
try:
 result['witness']=current_windows_job_witness(handle,info['query_handle_target_pid'],info['query_handle_target_creation_time'])
 result['status']='EXACT_OWN_JOB_VERIFIED'
except BaseException as error:
 result['status']='REJECTED_BEFORE_FRAMEWORK';result['error']=type(error).__name__+': '+str(error)
kernel=c.WinDLL('kernel32',use_last_error=True);kernel.GetHandleInformation.argtypes=[w.HANDLE,c.POINTER(w.DWORD)];kernel.GetHandleInformation.restype=w.BOOL
flags=w.DWORD();result['handle_usable_after_witness']=bool(kernel.GetHandleInformation(handle,c.byref(flags)))
result['handle_error_after_witness']=c.get_last_error()
if result['status']=='EXACT_OWN_JOB_VERIFIED':
 subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(30)'])
write_new(info['child_receipt'],result)
time.sleep(30)
'''
    process = job = None
    evidence = {"scope": "ACTUAL_EXACT_JOB_QUERY_CPU_ONLY", "rejection_control": rejection,
                "model_loads": 0, "framework_imports": 0, "gpu_calls": 0}
    try:
        process = subprocess.Popen([sys.executable, "-B", "-c", child_code], cwd=ROOT,
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 4)
        job = WindowsJob(process)
        evidence["before_resume"] = job_member_witness(job, cfg)
        evidence["resume"] = resume_suspended_primary_thread(process)
        deadline = time.monotonic() + 3
        target = []
        while time.monotonic() < deadline:
            members = job_member_witness(job, cfg)
            target = [m for m in members if Path(m["executable"]).resolve() == Path(sys._base_executable).resolve()]
            if len(target) == 1:
                break
            time.sleep(.02)
        require(len(target) == 1, "CPU query probe runtime target unavailable")
        transfer = duplicate_owned_job_query_handle(job, target[0])
        evidence["query_handle_transfer"] = transfer
        sent = {**transfer, "child_receipt": str(child_receipt)}
        if rejection == "target_pid": sent["query_handle_target_pid"] += 1
        if rejection == "target_creation": sent["query_handle_target_creation_time"] -= 1
        process.stdin.write(canonical(sent) + b"\n"); process.stdin.flush()
        deadline = time.monotonic() + 5
        while not child_receipt.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(.02)
        require(child_receipt.exists(), "CPU child query probe did not return bounded receipt")
        observed = read_json(child_receipt)
        evidence["child"] = observed
        require(observed["handle_usable_after_witness"] is False and observed["handle_error_after_witness"] == 6,
                "child query handle remained open before framework boundary")
        expected = "EXACT_OWN_JOB_VERIFIED" if rejection is None else "REJECTED_BEFORE_FRAMEWORK"
        require(observed["status"] == expected and not observed["framework_imported"], "CPU query probe unexpected outcome")
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            verified_members = job_member_witness(job, cfg)
            if rejection is not None or len(verified_members) >= 4:
                break
            time.sleep(.02)
        require(rejection is not None or len(verified_members) >= 4, "post-query grandchild inheritance not observed")
        evidence["members_before_parent_handle_close"] = verified_members
        job.close()  # Last owned Job handle; must kill all members without explicit kill.
        deadline = time.monotonic() + 5
        remaining = []
        while time.monotonic() < deadline:
            remaining = []
            for witness in verified_members:
                try:
                    member = psutil.Process(witness["pid"])
                    if member.create_time() == witness["creation_time"] and member.is_running():
                        remaining.append(member.pid)
                except psutil.NoSuchProcess:
                    pass
            if not remaining:
                break
            time.sleep(.02)
        require(not remaining, "last-handle close did not terminate complete owned tree")
        process.wait(timeout=5)
        evidence.update(status="PASS_EXACT_JOB_QUERY_CPU_ONLY", remaining_owned_pids=[],
                        parent_last_handle_close_terminated_tree=True, child_receipt_sha256=sha(child_receipt))
        write_new(output / "PROBE_RECEIPT.json", evidence)
        return evidence
    finally:
        if job:
            job.close()
        if process is not None and process.poll() is None:
            process.kill(); process.wait(timeout=5)
        if process is not None:
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()


def worker(args, cfg):
    """Contained worker. Requires parent GO, committed review and fenced lease."""
    require(os.name == "nt", "Windows-only worker")
    output = args.output.resolve()
    require(output.is_relative_to(PACKAGE / "results") and args.go_path.resolve().parent == output,
            "owned worker/GO path escape")
    require(Path(args.host_db).resolve() == host_state_path().resolve(), "canonical host database required")
    until = time.monotonic() + 5
    while not args.go_path.exists() and time.monotonic() < until:
        time.sleep(.02)
    require(args.go_path.exists(), "contained GO not received; no framework imported")
    go = read_json(args.go_path)
    query_handle = go.get("own_job_query_handle")
    query_consumed = False
    try:
        require(all(go.get(k) == v for k, v in {"contract_sha256": args.contract_sha256,
                                              "owner": args.owner, "fence": args.fence}.items()),
                "owned contained GO mismatch")
        import psutil
        ancestors = psutil.Process().parents()
        require(go.get("supervisor_pid") in [p.pid for p in ancestors]
                and go.get("worker_tree_pid") in [os.getpid()] + [p.pid for p in ancestors],
                "worker ancestry differs from recorded supervisor/tree")
        supervisor_command = psutil.Process(go["supervisor_pid"]).cmdline()
        require("--supervise" in supervisor_command
                and "experiments.rineya_centralized_reasoning_system_v1.native" in supervisor_command,
                "direct worker invocation unsupported; registered supervisor command required")
        query_consumed = type(query_handle) is int and 0 < query_handle < 2 ** 64
        containment = current_windows_job_witness(query_handle,
                      go.get("query_handle_target_pid"), go.get("query_handle_target_creation_time"))
    finally:
        if not query_consumed and type(query_handle) is int and 0 < query_handle < 2 ** 64:
            close_unconsumed_query_handle(query_handle)
    write_new(output / "CONTAINMENT_RECEIPT.json", {**containment, "supervisor_pid": go["supervisor_pid"],
               "worker_tree_pid": go["worker_tree_pid"], "contract_sha256": args.contract_sha256})
    snapshot = committed_snapshot()
    require(not snapshot["dirty"], "worker source dirty")
    require(not review_reasons(read_json(args.review), args.contract_sha256, snapshot, sha(__file__)),
            "worker review drift")
    verify_sources(cfg); verify_assets(cfg)
    require(runtime_metadata() == cfg["runtime"], "worker runtime drift")
    store = OperationsStore(args.host_db, output / "receipts")
    events = output / "EVENTS.jsonl"
    started = time.monotonic()
    result = {"status": "RUNNING", "native_qualified": False, "capture_qualified": False,
              "reader_qualified": False, "training_qualified": False,
              "model_loads": 0, "optimizer_steps": 0, "generation_tokens": 0,
              "resources": [], "comparisons": [], "trials": [], "source_commit": snapshot["commit"]}
    def event(kind, **detail):
        store._require_lease(args.owner, args.fence)
        value = {"time": time.time(), "kind": kind, **safe_numbers(detail)}
        with events.open("ab") as stream:
            stream.write(canonical(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    accounting = None
    try:
        from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting, tensor_digest
        from experiments.post_training_mimo_centralized_v1.scripts import stream_checkpoint
        accounting = CallAccounting(event, maximum_optimizer_calls=0)
        event("framework_import_attempt")
        import psutil
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                and torch.cuda.device_count() == 1 and torch.version.cuda == "12.6", "CUDA/BF16 identity unavailable")
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
        write_new(output / "RUNTIME_RECEIPT.json", {"runtime": runtime_metadata(), "cuda": torch.version.cuda,
             "gpu_name": torch.cuda.get_device_name(0), "matmul_flags_changed": False,
             "allow_tf32_fixed": False,
             "matmul_flags": {name: {"available": hasattr(torch.backends.cuda.matmul, name),
                                      "value": getattr(torch.backends.cuda.matmul, name, None)}
               for name in ("allow_bf16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction",
                            "allow_bf16_reduced_precision_reduction_split_k", "allow_fp16_reduced_precision_reduction_split_k",
                            "allow_fp16_accumulation")}})
        process = psutil.Process()
        def guard(stage, preload=False):
            store._require_lease(args.owner, args.fence)
            require(committed_snapshot() == snapshot, "worker source commit/dirty drift")
            require(time.monotonic() - started <= 600, "wall ceiling")
            torch.cuda.synchronize(); gpu_free, _ = torch.cuda.mem_get_info()
            sample = {"stage": stage, "rss_gib": process.memory_info().rss / GIB,
                      "host_available_gib": psutil.virtual_memory().available / GIB,
                      "gpu_free_gib": gpu_free / GIB,
                      "gpu_allocated_gib": torch.cuda.memory_allocated() / GIB,
                      "gpu_reserved_gib": torch.cuda.memory_reserved() / GIB,
                      "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
                      "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
                      "disk_free_gib": psutil.disk_usage(str(output)).free / GIB,
                      "elapsed_seconds": time.monotonic() - started}
            result["resources"].append(sample); event("resource", **sample)
            result["sampled_min_gpu_free_gib"] = min(sample["gpu_free_gib"], result.get("sampled_min_gpu_free_gib", float("inf")))
            require(sample["rss_gib"] <= 2 and sample["host_available_gib"] >= 1.25,
                    "worker RAM floor/cap")
            require(sample["disk_free_gib"] >= 4 and sample["gpu_free_gib"] >= (4 if preload else .5),
                    "worker GPU/disk floor")
            if preload:
                require(sample["host_available_gib"] >= 1.25 + max(0, 2 - sample["rss_gib"]),
                        "post-import full growth budget unavailable")
        guard("strict_preload", True)
        torch.cuda.reset_peak_memory_stats()
        verify_sources(cfg); verify_assets(cfg)
        model_cfg = Qwen3Config.from_pretrained(cfg["model_dir"], local_files_only=True)
        model_cfg._attn_implementation = "eager"
        prior = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda:0"):
                model = accounting.call("model_construct", lambda: Qwen3ForCausalLM(model_cfg), stage="construct")
        finally:
            torch.set_default_dtype(prior)
        model.eval().requires_grad_(False)
        hashes, details = accounting.call("checkpoint_load", lambda: stream_checkpoint.load_streamed(
              torch, model, Path(cfg["model_dir"]) / "model.safetensors", accounting, tensor_digest), stage="load")
        result["model_loads"] = 1
        require(model.lm_head.weight is model.model.embed_tokens.weight
                and sum(p.numel() for p in model.parameters()) == 596049920, "loaded tying/count mismatch")
        write_new(output / "MODEL_RECEIPT.json", {"tensor_sha256": hashes, "loader": details,
              "asset_sha256": cfg["asset_sha256"], "dtype": "bfloat16", "trainable_parameters": 0})
        guard("loaded")
        witness = {}
        def mask_hook(_module, _inputs, kwargs):
            mask = kwargs.get("attention_mask")
            require(mask is not None and tuple(mask.shape) == (1, 1, 131, 131)
                    and mask.dtype == torch.bfloat16, "decoder mask identity")
            values = mask[0, 0].float().cpu().tolist()
            minimum = float(torch.finfo(torch.bfloat16).min)
            require(all(v == (0 if col <= row else minimum)
                        for row, vals in enumerate(values) for col, v in enumerate(vals)), "causal mask values")
            require(kwargs.get("past_key_values") is None and kwargs.get("past_key_value") is None, "cache forbidden")
            mask_hash = tensor_digest(torch, mask)
            expected_mask_hash = hashlib.sha256(causal_mask_reference_bytes()).hexdigest()
            require(mask_hash == expected_mask_hash, "raw causal mask differs from replayable representation")
            witness["counts"]["mask"] += 1
            witness["mask"] = {"shape": [1, 1, 131, 131], "dtype": "bfloat16",
                "raw_sha256": mask_hash, "reference_raw_sha256": expected_mask_hash, "byte_order": "little_endian",
                "complete_representation": {"rows": 131, "columns": 131,
                    "allowed_condition": "column <= row", "allowed_uint16": 0, "forbidden_uint16": 65407,
                    "allowed_value": 0.0, "forbidden_value": minimum}, "actual_values_match_representation": True}
        def rotary_hook(_module, inputs, kwargs):
            pos = inputs[1] if len(inputs) > 1 else kwargs.get("position_ids")
            require(pos is not None and pos.dtype == torch.long and pos.cpu().tolist() == [list(range(131))],
                    "rotary positions")
            witness["counts"]["rotary"] += 1
            witness["positions"] = {"shape": [1, 131], "dtype": "int64", "byte_order": "little_endian",
                                    "position_ids": pos.cpu().tolist(), "raw_sha256": tensor_digest(torch, pos)}
        def norm_hook(_module, _inputs, value):
            require(tuple(value.shape) == (1, 131, 1024) and value.dtype == torch.bfloat16, "hidden shape/dtype")
            witness["counts"]["norm"] += 1
            witness["hidden"] = value[0, -4:].detach().clone().contiguous()
        hooks = [model.model.layers[0].register_forward_pre_hook(mask_hook, with_kwargs=True),
                 model.model.rotary_emb.register_forward_pre_hook(rotary_hook, with_kwargs=True),
                 model.model.norm.register_forward_hook(norm_hook)]
        baseline_logits = baseline_hidden = None
        try:
            with torch.no_grad():
                for trial in causal_plan(cfg["prompt_ids"], cfg["generated_ids"]):
                    guard("before_" + trial["stage"]); witness.clear()
                    witness["counts"] = {"mask": 0, "rotary": 0, "norm": 0}
                    require(accounting.snapshot()["operations"]["forward"]["attempted"] < 4, "forward ceiling")
                    ids = torch.tensor([trial["ids"]], dtype=torch.long, device="cuda")
                    logits = accounting.call("forward", lambda: model(ids, use_cache=False, logits_to_keep=4).logits,
                                             stage=trial["stage"], input_positions=131)[0]
                    hidden = witness.pop("hidden")
                    require(witness["counts"] == {"mask": 1, "rotary": 1, "norm": 1}, "hook witness invocation counts")
                    require(tuple(logits.shape) == (4, 151936) and logits.dtype == torch.bfloat16
                            and bool(torch.isfinite(logits).all()) and bool(torch.isfinite(hidden).all()), "NaN/finite shape failure")
                    lp = logits[:3].float().log_softmax(-1).gather(1,
                              torch.tensor(cfg["generated_ids"], device="cuda")[:, None]).squeeze(1).cpu().tolist()
                    measured = {"stage": trial["stage"], "logits_sha256": tensor_digest(torch, logits),
                                "hidden_sha256": tensor_digest(torch, hidden), "model_logprobs": lp,
                                "mutation_start": trial["mutation_start"],
                                "protected_positions": list(range(127, 127 + trial["protected_count"])),
                                "witnesses": dict(witness)}
                    if trial["stage"] == "baseline":
                        require(measured["logits_sha256"] == cfg["reference_bulk_logits_sha256"]
                                and measured["hidden_sha256"] == cfg["reference_bulk_hidden_sha256"]
                                and lp == cfg["reference_bulk_model_logprobs"], "original failed bulk endpoint drift")
                        baseline_logits = logits.detach().clone(); baseline_hidden = hidden.clone()
                    else:
                        count = trial["protected_count"]
                        for label, left, right in (("logits", baseline_logits[:count], logits[:count]),
                                                 ("hidden", baseline_hidden[:count], hidden[:count])):
                            delta = (left.float() - right.float()).abs()
                            left_hash, right_hash = tensor_digest(torch, left), tensor_digest(torch, right)
                            check = numerical_decision(float(delta.max()), float(delta.mean()), bitwise=left_hash == right_hash,
                                                       left_sha256=left_hash, right_sha256=right_hash)
                            result["comparisons"].append({"stage": trial["stage"], "kind": label,
                                "torch_numeric_equal": bool(torch.equal(left, right)),
                                "mutation_start": trial["mutation_start"], "protected_positions": list(range(127, 127 + count)), **check})
                    result["trials"].append(measured)
                    event("measurement", measurement=measured)
                    guard("after_" + trial["stage"])
                    del ids, logits, hidden
        finally:
            for hook in hooks:
                hook.remove()
        require(len(result["comparisons"]) == 6 and all(c["passed"] for c in result["comparisons"]),
                "fixed-shape numerical causality failed")
        for name, parameter in model.named_parameters():
            require(not parameter.requires_grad and parameter.grad is None
                    and parameter.dtype == torch.bfloat16 and tensor_digest(torch, parameter) == hashes[name],
                    "frozen base changed")
        verify_sources(cfg); verify_assets(cfg); guard("complete")
        ledger = accounting.snapshot()
        require(ledger["operations"]["forward"]["returned"] == 4
                and ledger["forward_input_positions_returned"] == 524 and ledger["last_call_id"] == 316,
                "incomplete accounting")
        result["status"] = "COMPLETE_FIXED_SHAPE_DIAGNOSTIC_ONLY"
        result["interpretation"] = "Four trace-local bitwise causality controls; prefix/bulk capture remains failed."
    except BaseException as exc:
        result.update(status="FAILED_DIAGNOSTIC", error=type(exc).__name__ + ": " + str(exc),
                      traceback=traceback.format_exc())
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["call_accounting"] = accounting.snapshot() if accounting else None
        if "torch" in locals():
            try:
                result["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / GIB
                result["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved() / GIB
            except Exception as error:
                result["allocator_peak_telemetry_error"] = str(error)
        write_new(output / "DIAGNOSTIC_RESULT.json", safe_numbers(result))
        store.close()
    return 0 if result["status"] == "COMPLETE_FIXED_SHAPE_DIAGNOSTIC_ONLY" else 2


def supervise(args, cfg):
    """Own one model-worker process and enforce actual external telemetry."""
    import psutil
    require(Path(args.host_db).resolve() == host_state_path().resolve(), "one canonical host-wide database required")
    report = preflight(contract_sha256=args.contract_sha256, review_path=args.review)
    require(report["status"] == "READY_DIAGNOSTIC_ONLY", "preflight blocked: " + repr(report["reasons"]))
    time.sleep(2)
    second = collect_telemetry(desktop_identities=cfg["gpu_process_registry"])
    require(not admission_reasons(cfg, second), "second current pre-worker observation failed")
    output = args.output.resolve()
    require(output.is_relative_to(PACKAGE / "results") and not output.exists(), "new CRS results child required")
    output.mkdir(parents=True)
    store = OperationsStore(args.host_db, output / "receipts")
    owner = "CRS-D2-" + uuid.uuid4().hex
    process = job = None
    tree_exit = None
    lease = None
    started = time.monotonic()
    stopped = threading.Event()
    telemetry_state = {"snapshot": second, "error": None}
    outcome, reason = "FAILED", None
    worker_dispatched = False
    launch_attempted = False
    reservation_admitted = False
    try:
        store.register_job(args.job_key, {"contract_sha256": args.contract_sha256,
                           "source_commit": report["observations"]["source"]["commit"]}, args.epoch)
        store.ready(args.job_key)
        lease = store.acquire_lease(owner, 120)
        store.reserve(args.job_key, owner, lease["fence"], 600)
        reservation_admitted = True
        command = [sys.executable, "-B", "-m", "experiments.rineya_centralized_reasoning_system_v1.native",
                   "--worker", "--contract-sha256", args.contract_sha256, "--review", str(args.review.resolve()),
                   "--output", str(output), "--go-path", str(output / "GO.json"), "--host-db", str(args.host_db),
                   "--owner", owner, "--fence", str(lease["fence"])]
        write_new(output / "LAUNCH_INTENT.json", {"command": command, "preflight": report,
                  "second_observation": second, "job_key": args.job_key, "lease": lease})
        from experiments.opallo_native_collector_v1.supervisor import WindowsJob
        with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            write_new(output / "LAUNCH_ATTEMPT.json", {"job_key": args.job_key, "owner": owner,
                       "fence": lease["fence"], "command_sha256": hashlib.sha256(canonical(command)).hexdigest(),
                       "attempted_at": time.time(), "outcome": "PENDING_SUBPROCESS_CREATION"})
            launch_attempted = True
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                     env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                              CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1"),
                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 0x4)
            worker_dispatched = True
            observed = psutil.Process(process.pid)
            store.mark_running(args.job_key, owner, lease["fence"], {"pid": process.pid,
                "creation_time": str(observed.create_time()), "command_hash": hashlib.sha256(canonical(command)).hexdigest()})
            job = WindowsJob(process)
            before_resume = job_member_witness(job, cfg)
            require(job_process_ids(job) == [process.pid] and not observed.children(recursive=True),
                    "worker must be suspended before any child can form")
            resumed = resume_suspended_primary_thread(process)
            deadline = time.monotonic() + 3
            members = []
            while time.monotonic() < deadline:
                members = job_member_witness(job, cfg)
                if len(members) >= 2:
                    break
                time.sleep(.02)
            require(len(members) >= 2, "venv runtime child membership not observed before GO")
            runtime_targets = [m for m in members if Path(m["executable"]).resolve() == Path(sys._base_executable).resolve()]
            require(len(runtime_targets) == 1, "one exact CPython worker target required for Job query handle")
            query_transfer = duplicate_owned_job_query_handle(job, runtime_targets[0])
            write_new(output / "TREE_CONTAINMENT.json", {"before_resume": before_resume,
                       "resume": resumed, "before_go_members": members, "all_members_in_exact_owned_job": True,
                       "query_handle_transfer": query_transfer})
            write_new(output / "GO.json", {"contract_sha256": args.contract_sha256, "owner": owner, "fence": lease["fence"],
                       "supervisor_pid": os.getpid(), "worker_tree_pid": process.pid, **query_transfer})
            def gpu_watch():
                while not stopped.is_set():
                    querying_live = False
                    try:
                        allowed = job_process_ids(job)
                        if not allowed:
                            if process.poll() is not None:
                                telemetry_state["owned_exit_observed"] = True
                                return
                            stopped.wait(.02)
                            continue
                        querying_live = True
                        telemetry_state["snapshot"] = collect_telemetry(output, allowed, cfg["gpu_process_registry"])
                    except BaseException as exc:
                        if stopped.is_set() and not querying_live:
                            return
                        telemetry_state["error"] = str(exc)
                        break
                    stopped.wait(.5)
            watcher = threading.Thread(target=gpu_watch, daemon=True); watcher.start()
            last_heartbeat, peak_rss = time.monotonic(), 0
            with (output / "WATCHDOG.jsonl").open("xb") as audit:
                while process.poll() is None:
                    store._require_lease(owner, lease["fence"])
                    owned = owned_job_rss_snapshot(job, cfg)
                    members = owned["members"]
                    elapsed = time.monotonic() - started
                    rss = owned["rss_bytes"]
                    peak_rss = max(peak_rss, rss)
                    host_free = psutil.virtual_memory().available / GIB
                    sample = telemetry_state["snapshot"]
                    if elapsed >= 600: reason = "WALL_LIMIT"
                    elif rss > 2 * GIB or host_free < 1.25: reason = "PROCESS_TREE_RAM_LIMIT"
                    elif telemetry_state["error"] or time.time() - sample["sampled_at"] > 5: reason = "TELEMETRY_UNAVAILABLE"
                    elif sample["gpu_free_gib"] < .5 or sample["disk_free_gib"] < 4: reason = "GPU_DISK_FLOOR"
                    elif sample["unresolved_gpu_pids"] or sample["competing_gpu_pids"] or sample["unclassified_gpu_pids"]: reason = "GPU_PROCESS_CONFLICT"
                    elif stdout.tell() + stderr.tell() > 16 * 1024 ** 2: reason = "LOG_LIMIT"
                    audit.write(canonical({"elapsed": elapsed, "rss": rss, "host_free_gib": host_free,
                                "gpu_snapshot": sample, "owned_job_members": members,
                                "confirmed_owned_exit_races": owned["confirmed_exit_races"], "reason": reason}) + b"\n")
                    audit.flush(); os.fsync(audit.fileno())
                    if reason:
                        job.kill(); break
                    if time.monotonic() - last_heartbeat >= 10:
                        store.heartbeat_lease(owner, lease["fence"], 120)
                        last_heartbeat = time.monotonic()
                    stopped.wait(.1)
            code = process.wait(timeout=10)
            require(code == 0 and reason is None, "contained worker failed: " + str(reason or code))
            stopped.set()
            watcher.join(timeout=5)
            require(not watcher.is_alive(), "GPU telemetry worker did not stop")
            require(telemetry_state["error"] is None, "live GPU telemetry failure retained")
            require(job_process_ids(job) == [], "launcher exit left an owned descendant active")
            result = read_json(output / "DIAGNOSTIC_RESULT.json")
            events = read_event_journal(output / "EVENTS.jsonl")
            completion = validate_diagnostic_completion(result, cfg, report["observations"]["source"]["commit"], events)
            completion["events_sha256"] = sha(output / "EVENTS.jsonl")
            write_new(output / "COMPLETION_VERIFICATION.json", completion)
            verify_sources(cfg); verify_assets(cfg)
            require(not committed_snapshot()["dirty"], "source dirty after worker")
            outcome = "SUCCEEDED"
    except BaseException as exc:
        reason = reason or type(exc).__name__ + ": " + str(exc)
    finally:
        stopped.set()
        if job:
            try:
                tree_exit = drain_owned_job(job)
            except BaseException as exc:
                reason = reason or "owned tree exit unverified: " + str(exc)
                outcome = "OUTCOME_UNKNOWN"
            finally:
                job.close()
        if process is not None and process.poll() is None:
            # This process is the caller-owned, recorded worker only.
            process.kill(); process.wait(timeout=10)
        elapsed = time.monotonic() - started
        uncertain_creation = launch_attempted and not worker_dispatched
        if uncertain_creation:
            outcome = "OUTCOME_UNKNOWN"
        receipt = {"status": outcome, "reason": reason, "wall_seconds": elapsed,
                   "launch_attempted": launch_attempted, "worker_dispatched": worker_dispatched,
                   "subprocess_creation_outcome_unknown": uncertain_creation,
                   "owned_tree_exit": tree_exit,
                   "exit_code": process.returncode if process else None,
                   "peak_process_tree_rss_bytes": locals().get("peak_rss"),
                   "watchdog_poll_seconds": .1, "reader_qualified": False, "training_qualified": False}
        try:
            write_new(output / "SUPERVISOR_RESULT.json", receipt)
            if lease is not None and reservation_admitted:
                row = store.get_job(args.job_key)
                if (uncertain_creation or (worker_dispatched and row["worker"] is None)
                        or (worker_dispatched and not tree_exit)):
                    store.mark_unknown(args.job_key, "subprocess creation/identity response unavailable; reconcile before retry")
                    # Popen may have created a worker before losing its response.
                    # No process identity exists to prove exit or zero cost.
                    # Preserve the full reservation and held model lease.
                    print(json.dumps(receipt), flush=True)
                    return 2
                proof = {"observer": "CRS_NATIVE_SUPERVISOR_V1", "request_hash": row["request_hash"],
                         "owner": owner, "fence": lease["fence"],
                         "elapsed_native_seconds": elapsed if worker_dispatched else 0}
                if worker_dispatched:
                    require(process is not None and process.poll() is not None, "worker exit observation missing")
                    proof.update(worker_identity=row["worker"], process_exited=True,
                                 exit_observed_at=time.time(), owned_tree_exited=True,
                                 owned_tree_exit=tree_exit)
                else:
                    require(not launch_attempted, "attempted launch cannot receive zero-cost no-dispatch proof")
                    proof.update(never_dispatched=True, launch_attempted=False)
                proof_ref = store.write_receipt("native_outcome_observation", args.job_key, proof)
                store.reconcile(args.job_key, outcome, elapsed if worker_dispatched else 0,
                                {"supervisor_receipt_sha256": sha(output / "SUPERVISOR_RESULT.json"),
                                 "native_outcome_proof_ref": proof_ref}, owner=owner, fence=lease["fence"])
            if lease is not None:
                store.release_lease(owner, lease["fence"])
        except BaseException:
            # Lost publication/reconciliation leaves reservation conservative.
            if lease is not None and reservation_admitted and store.get_job(args.job_key)["state"] in {"RESERVED", "RUNNING", "OUTCOME_UNKNOWN"}:
                store.mark_unknown(args.job_key, "supervisor publication/reconciliation failed")
            raise
        finally:
            store.close()
    print(json.dumps(receipt), flush=True)
    return 0 if outcome == "SUCCEEDED" else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--supervise", action="store_true")
    mode.add_argument("--worker", action="store_true")
    p.add_argument("--contract-sha256", required=True)
    p.add_argument("--review", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--host-db", type=Path, default=host_state_path())
    p.add_argument("--epoch"); p.add_argument("--job-key")
    p.add_argument("--go-path", type=Path); p.add_argument("--owner"); p.add_argument("--fence", type=int)
    args = p.parse_args()
    if args.preflight:
        print(json.dumps(preflight(contract_sha256=args.contract_sha256, review_path=args.review), indent=2))
        return 0
    require(args.review is not None and args.output is not None, "review/output required")
    cfg = checked_contract(args.contract_sha256)
    if args.worker:
        require(args.go_path is not None and args.owner is not None and args.fence is not None, "owned worker identity required")
        return worker(args, cfg)
    require(args.epoch and args.job_key, "durable epoch and idempotent job key required")
    return supervise(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
