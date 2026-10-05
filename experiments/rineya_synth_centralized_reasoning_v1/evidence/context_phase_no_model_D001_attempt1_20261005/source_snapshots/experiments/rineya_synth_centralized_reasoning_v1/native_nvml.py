"""Read-only GPU0 telemetry through the pinned Windows System32 NVML DLL.

Import is framework/DLL free. Only NVML initialization, device identification,
memory queries and cleanup are exposed. No CUDA context, model or subprocess
is constructed. Native calls run in one bounded daemon thread; an unreturned
call is reported explicitly and never silently replaced by another query.

ABI: https://docs.nvidia.com/deploy/nvml-api/latest/api/index.html
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import os
from pathlib import Path
import re
import threading
import time
import uuid

QUERY_TIMEOUT_SECONDS = 2.0
_STATE_LOCK = threading.Lock()
_CLIENT = None
_ACTIVE = None
_FAULT = None


class NVMLError(RuntimeError):
    pass


class _MemoryInfo(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def library_path():
    """Actual Windows System32 NVML path, for preflight byte pinning."""
    if os.name != "nt":
        raise NVMLError("NVML telemetry requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetSystemDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]
    kernel.GetSystemDirectoryW.restype = wintypes.UINT
    buffer = ctypes.create_unicode_buffer(32768)
    count = kernel.GetSystemDirectoryW(buffer, len(buffer))
    if not 0 < count < len(buffer):
        raise NVMLError("cannot resolve actual Windows system directory")
    candidate = Path(buffer.value) / "nvml.dll"
    if candidate.is_symlink() or not candidate.is_file():
        raise NVMLError("NVML must be a regular System32 DLL")
    path = candidate.resolve()
    if path.parent != Path(buffer.value).resolve():
        raise NVMLError("NVML path leaves actual system directory")
    return path


def library_binding():
    """Return read-only path/hash metadata without loading NVML."""
    path = library_path()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


class _Client:
    def __init__(self):
        self.path = library_path()
        # Search dependencies only beside the absolute DLL and in System32.
        self.api = ctypes.WinDLL(str(self.path), use_last_error=True, winmode=0x100 | 0x800)
        signatures = {
            "nvmlInit_v2": ([], ctypes.c_int),
            "nvmlShutdown": ([], ctypes.c_int),
            "nvmlDeviceGetHandleByIndex_v2": ([ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            "nvmlDeviceGetMemoryInfo": ([ctypes.c_void_p, ctypes.POINTER(_MemoryInfo)], ctypes.c_int),
            "nvmlDeviceGetUUID": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes = arguments
            function.restype = result
        self.initialized = False
        self._check("nvmlInit_v2", self.api.nvmlInit_v2())
        self.initialized = True

    @staticmethod
    def _check(operation, status):
        if status != 0:
            raise NVMLError(operation + " returned NVML status " + str(status))

    def query(self):
        device = ctypes.c_void_p()
        self._check("nvmlDeviceGetHandleByIndex_v2", self.api.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(device)))
        if not device.value:
            raise NVMLError("NVML returned a null GPU0 handle")
        memory = _MemoryInfo()
        self._check("nvmlDeviceGetMemoryInfo", self.api.nvmlDeviceGetMemoryInfo(device, ctypes.byref(memory)))
        identifier = ctypes.create_string_buffer(80)
        self._check("nvmlDeviceGetUUID", self.api.nvmlDeviceGetUUID(device, identifier, len(identifier)))
        try:
            identity = identifier.value.decode("ascii")
        except UnicodeError as error:
            raise NVMLError("NVML GPU UUID is not ASCII") from error
        if re.fullmatch(r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}", identity) is None:
            raise NVMLError("NVML GPU0 UUID is malformed")
        if not 0 < memory.total or memory.free > memory.total:
            raise NVMLError("NVML GPU0 memory observation is invalid")
        return {"GPU_index": 0, "gpu_uuid": identity, "gpu_total_bytes": int(memory.total),
                "gpu_free_bytes": int(memory.free)}

    def close(self):
        if self.initialized:
            self._check("nvmlShutdown", self.api.nvmlShutdown())
            self.initialized = False


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _Client()
    return _CLIENT


def _bounded(operation, function, *, allow_fault=False):
    global _ACTIVE, _FAULT
    started = time.monotonic()
    deadline = started + QUERY_TIMEOUT_SECONDS
    result = {}
    with _STATE_LOCK:
        if _ACTIVE is not None and _ACTIVE["thread"].is_alive():
            raise NVMLError("a previous NVML operation has not returned")
        if _FAULT is not None and not allow_fault:
            raise NVMLError("previous NVML observation failed; no automatic query retry: " + _FAULT)
        def execute():
            try:
                result["value"] = function()
            except BaseException as error:
                result["error"] = error
            finally:
                result["completed"] = time.monotonic()
        thread = threading.Thread(target=execute, daemon=True, name="SYNTH_NVML_" + operation + "_" + uuid.uuid4().hex)
        _ACTIVE = {"thread": thread, "operation": operation, "started_monotonic": started,
                   "deadline_monotonic": deadline}
        thread.start()
    thread.join(timeout=max(0, deadline - time.monotonic()))
    with _STATE_LOCK:
        if thread.is_alive() or result.get("completed", float("inf")) > deadline:
            _FAULT = operation + " exceeded fixed 2s native-call deadline"
            raise TimeoutError(_FAULT)
        _ACTIVE = None
        if "error" in result:
            error = result["error"]
            _FAULT = type(error).__name__ + ": " + str(error)
            raise error
        return result["value"], started, result["completed"]


def gpu_snapshot():
    value, started, completed = _bounded("memory_query", lambda: _client().query())
    return {**value, "gpu_sample_started_monotonic": started, "gpu_sample_completed_monotonic": completed,
            "gpu_observation_id": uuid.uuid4().hex}


def pending_query_threads():
    with _STATE_LOCK:
        if _ACTIVE is None or not _ACTIVE["thread"].is_alive():
            return []
        return [{"name": _ACTIVE["thread"].name, "ident": _ACTIVE["thread"].ident,
                 "operation": _ACTIVE["operation"], "started_monotonic": _ACTIVE["started_monotonic"],
                 "deadline_monotonic": _ACTIVE["deadline_monotonic"]}]


def shutdown():
    """Bound cleanup; refuse concurrent cleanup of a still-running query."""
    global _CLIENT, _FAULT
    with _STATE_LOCK:
        if _ACTIVE is not None and _ACTIVE["thread"].is_alive():
            raise NVMLError("cannot verify NVML cleanup while a native call has not returned")
        if _CLIENT is None:
            return
    _bounded("shutdown", _CLIENT.close, allow_fault=True)
    _CLIENT = None
    _FAULT = None
