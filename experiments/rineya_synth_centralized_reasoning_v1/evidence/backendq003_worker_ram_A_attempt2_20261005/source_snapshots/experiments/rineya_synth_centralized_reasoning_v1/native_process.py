"""Read-only Win32 birth/RSS sampling on one fresh retained handle per PID.

GetProcessTimes, K32GetProcessMemoryInfo and WaitForSingleObject are documented
at learn.microsoft.com/windows/win32/api. QUERY_LIMITED_INFORMATION suffices on
the current qualified Windows host; SYNCHRONIZE permits terminal confirmation.
No psutil broad-system fallback, framework, DLL load or process occurs at import.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import time

QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x100000
READ_RIGHTS = QUERY_LIMITED_INFORMATION | SYNCHRONIZE
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
EPOCH_FILETIME = 116444736000000000
_SYSTEM_SAMPLER = None


class FILETIME(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
        (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
        "PeakPagefileUsage", "PrivateUsage")]


class MEMORY_STATUS_EX(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("memory_load", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in ("total_phys", "available_phys", "total_pagefile", "available_pagefile",
        "total_virtual", "available_virtual", "available_extended_virtual")]


def _ticks(value):
    return (int(value.high) << 32) | int(value.low)


def library_path():
    if os.name != "nt":
        raise RuntimeError("process telemetry requires Windows")
    bootstrap = ctypes.WinDLL("kernel32", use_last_error=True)
    bootstrap.GetSystemDirectoryW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32]
    bootstrap.GetSystemDirectoryW.restype = ctypes.c_uint32
    buffer = ctypes.create_unicode_buffer(32768)
    size = bootstrap.GetSystemDirectoryW(buffer, len(buffer))
    if not 0 < size < len(buffer):
        raise RuntimeError("cannot resolve Windows system directory")
    candidate = Path(buffer.value) / "kernel32.dll"
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError("kernel32 must be a regular System32 DLL")
    path = candidate.resolve()
    if path.parent != Path(buffer.value).resolve():
        raise RuntimeError("kernel32 leaves system directory")
    return path


class Sampler:
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("process telemetry requires Windows")
        self.path = library_path()
        self.api = ctypes.WinDLL(str(self.path), use_last_error=True, winmode=0x800)
        signatures = {
            "OpenProcess": ([ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32], ctypes.c_void_p),
            "GetProcessTimes": ([ctypes.c_void_p] + [ctypes.POINTER(FILETIME)] * 4, ctypes.c_int32),
            "K32GetProcessMemoryInfo": ([ctypes.c_void_p, ctypes.POINTER(MEMORY_COUNTERS_EX), ctypes.c_uint32], ctypes.c_int32),
            "WaitForSingleObject": ([ctypes.c_void_p, ctypes.c_uint32], ctypes.c_uint32),
            "IsProcessInJob": ([ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)], ctypes.c_int32),
            "CloseHandle": ([ctypes.c_void_p], ctypes.c_int32),
            "GlobalMemoryStatusEx": ([ctypes.POINTER(MEMORY_STATUS_EX)], ctypes.c_int32),
        }
        for name, (arguments, result) in signatures.items():
            getattr(self.api, name).argtypes = arguments
            getattr(self.api, name).restype = result

    def _error(self, operation):
        error = ctypes.WinError(ctypes.get_last_error())
        error.add_note(operation)
        return error

    def birth_from_handle(self, handle):
        born, exited, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        if not self.api.GetProcessTimes(handle, ctypes.byref(born), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
            raise self._error("GetProcessTimes")
        return _ticks(born)

    def _terminal(self, handle):
        value = self.api.WaitForSingleObject(handle, 0)
        if value == WAIT_OBJECT_0:
            return True
        if value == WAIT_TIMEOUT:
            return False
        raise self._error("WaitForSingleObject")

    def is_alive(self, identity):
        """Exact raw birth match; neither RSS nor floating conversion used."""
        handle = self.api.OpenProcess(READ_RIGHTS, False, identity["pid"])
        if not handle:
            raise self._error("OpenProcess")
        try:
            if self._terminal(handle):
                return False
            return self.birth_from_handle(handle) == identity["creation_filetime_100ns"]
        finally:
            if not self.api.CloseHandle(handle):
                raise self._error("CloseHandle")

    def memory_status(self):
        value = MEMORY_STATUS_EX()
        value.length = ctypes.sizeof(value)
        if not self.api.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise self._error("GlobalMemoryStatusEx")
        if not 0 < value.total_phys or value.available_phys > value.total_phys:
            raise RuntimeError("invalid available physical memory observation")
        return {"available_bytes": int(value.available_phys), "total_bytes": int(value.total_phys)}


    def process(self, pid, *, job_handle=None):
        if type(pid) is not int or not 0 < pid <= 0xFFFFFFFF:
            raise ValueError("typed positive Windows PID required")
        times = {}
        started = time.monotonic()
        before = time.perf_counter()
        handle = self.api.OpenProcess(READ_RIGHTS, False, pid)
        times["OpenProcess_seconds"] = time.perf_counter() - before
        if not handle:
            # Failure to open never proves terminality. No broad scan or retry.
            raise self._error("OpenProcess")
        result = None
        try:
            before = time.perf_counter()
            if self._terminal(handle):
                result = {"pid": pid, "terminal_confirmed": True, "rss_bytes": 0}
                return result
            times["WaitForSingleObject_seconds"] = time.perf_counter() - before
            if job_handle is not None:
                inside = ctypes.c_int32()
                before = time.perf_counter()
                if not self.api.IsProcessInJob(handle, job_handle, ctypes.byref(inside)):
                    error = self._error("IsProcessInJob")
                    if self._terminal(handle):
                        result = {"pid": pid, "terminal_confirmed": True, "rss_bytes": 0}
                        return result
                    raise error
                if not inside.value:
                    raise RuntimeError("queried PID is outside owned Job")
                times["IsProcessInJob_seconds"] = time.perf_counter() - before
            before = time.perf_counter()
            try:
                birth = self.birth_from_handle(handle)
            except OSError:
                if self._terminal(handle):
                    result = {"pid": pid, "terminal_confirmed": True, "rss_bytes": 0}
                    return result
                raise
            times["GetProcessTimes_seconds"] = time.perf_counter() - before
            memory = MEMORY_COUNTERS_EX()
            memory.cb = ctypes.sizeof(memory)
            before = time.perf_counter()
            if not self.api.K32GetProcessMemoryInfo(handle, ctypes.byref(memory), ctypes.sizeof(memory)):
                error = self._error("K32GetProcessMemoryInfo")
                if self._terminal(handle):
                    result = {"pid": pid, "terminal_confirmed": True, "rss_bytes": 0}
                    return result
                raise error
            times["K32GetProcessMemoryInfo_seconds"] = time.perf_counter() - before
            result = {"pid": pid, "terminal_confirmed": False, "creation_filetime_100ns": birth,
                      "create_time": (birth - EPOCH_FILETIME) / 10000000.0, "rss_bytes": int(memory.WorkingSetSize)}
            return result
        finally:
            before = time.perf_counter()
            if not self.api.CloseHandle(handle):
                raise self._error("CloseHandle")
            times["CloseHandle_seconds"] = time.perf_counter() - before
            if result is not None:
                result["api_latency_seconds"] = times
                result["sample_started_monotonic"] = started
                result["sample_completed_monotonic"] = time.monotonic()


def system_memory_status():
    global _SYSTEM_SAMPLER
    if _SYSTEM_SAMPLER is None:
        _SYSTEM_SAMPLER = Sampler()
    return _SYSTEM_SAMPLER.memory_status()


def owned_snapshot(pids, root_identity, job_handle, sampler):
    identities, samples = [], []
    rss = 0
    for pid in pids:
        value = sampler.process(pid, job_handle=job_handle)
        samples.append(value)
        if value["terminal_confirmed"]:
            continue
        birth = value["creation_filetime_100ns"]
        if pid == root_identity["pid"]:
            if birth != root_identity["creation_filetime_100ns"]:
                raise RuntimeError("owned root raw birth FILETIME changed")
            # Preserve exact legacy psutil identity for lease-history consumers.
            created = root_identity["create_time"]
        else:
            created = value["create_time"]
        identities.append({"pid": pid, "create_time": created, "creation_filetime_100ns": birth})
        rss += value["rss_bytes"]
    return {"identities": identities, "rss_bytes": rss, "process_samples": samples}
