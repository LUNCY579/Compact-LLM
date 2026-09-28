"""Optional read-only Windows memory diagnostics for the calling process.

snapshot() imports ctypes only when called and never changes hostbudget enforcement.
Private bytes are PROCESS_MEMORY_COUNTERS_EX.PrivateUsage (private commit), not
private resident memory. SystemCache includes standby pages plus the system working
set; these counters overlap and must not be added into a physical-memory ledger.
Only current-process memory and aggregate OS cache/pool sizes are returned. No
process inventory, identifiers, hostnames, or other-process records are collected.

Microsoft definitions used for the native layout and units:
https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex
https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information
https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo
https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getperformanceinfo
"""
from __future__ import annotations


def _windows_snapshot() -> dict:
    import ctypes
    from ctypes import wintypes

    size_t = ctypes.c_size_t

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", size_t),
            ("WorkingSetSize", size_t),
            ("QuotaPeakPagedPoolUsage", size_t),
            ("QuotaPagedPoolUsage", size_t),
            ("QuotaPeakNonPagedPoolUsage", size_t),
            ("QuotaNonPagedPoolUsage", size_t),
            ("PagefileUsage", size_t),
            ("PeakPagefileUsage", size_t),
            ("PrivateUsage", size_t),
        ]

    class PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("CommitTotal", size_t),
            ("CommitLimit", size_t),
            ("CommitPeak", size_t),
            ("PhysicalTotal", size_t),
            ("PhysicalAvailable", size_t),
            ("SystemCache", size_t),
            ("KernelTotal", size_t),
            ("KernelPaged", size_t),
            ("KernelNonpaged", size_t),
            ("PageSize", size_t),
            ("HandleCount", wintypes.DWORD),
            ("ProcessCount", wintypes.DWORD),
            ("ThreadCount", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
    current_process = kernel32.GetCurrentProcess
    current_process.argtypes = []
    current_process.restype = wintypes.HANDLE

    process_info = psapi.GetProcessMemoryInfo
    process_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    process_info.restype = wintypes.BOOL
    performance_info = psapi.GetPerformanceInfo
    performance_info.argtypes = [ctypes.POINTER(PERFORMANCE_INFORMATION), wintypes.DWORD]
    performance_info.restype = wintypes.BOOL

    process = PROCESS_MEMORY_COUNTERS_EX()
    process.cb = ctypes.sizeof(process)
    # GetCurrentProcess returns a pseudo-handle; it neither opens another process
    # nor requires CloseHandle (per the documented pseudo-handle contract).
    if not process_info(current_process(), ctypes.byref(process), process.cb):
        raise OSError(f"GetProcessMemoryInfo failed (winerror {ctypes.get_last_error()})")
    system = PERFORMANCE_INFORMATION()
    system.cb = ctypes.sizeof(system)
    if not performance_info(ctypes.byref(system), system.cb):
        raise OSError(f"GetPerformanceInfo failed (winerror {ctypes.get_last_error()})")
    page_bytes = int(system.PageSize)
    if page_bytes <= 0:
        raise ValueError("GetPerformanceInfo returned a nonpositive page size")

    return {
        "available": True,
        "process_working_set_bytes": int(process.WorkingSetSize),
        "process_private_bytes": int(process.PrivateUsage),
        "system_cache_bytes": int(system.SystemCache) * page_bytes,
        "kernel_pool_total_bytes": int(system.KernelTotal) * page_bytes,
        "kernel_paged_pool_bytes": int(system.KernelPaged) * page_bytes,
        "kernel_nonpaged_pool_bytes": int(system.KernelNonpaged) * page_bytes,
        "page_size_bytes": page_bytes,
    }


def snapshot() -> dict:
    """Return current-process/OS byte counts, or {available: False, error: ...}.

    Diagnostics must not interrupt training when unavailable. The two read-only
    calls are consecutive samples, not an atomic measurement of the whole host.
    """
    import sys

    if sys.platform != "win32":
        return {"available": False, "error": "Windows PSAPI diagnostics unavailable on this platform"}
    try:
        return _windows_snapshot()
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    import json

    print(json.dumps(snapshot(), sort_keys=True))
