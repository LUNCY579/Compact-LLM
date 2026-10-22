"""Windows campaign-only process containment and parent-enforced limits.

No model imports. A native bootstrap must wait at its release barrier until the
parent assigns it to this non-inheritable, kill-on-close Windows Job Object.
"""
from __future__ import annotations
import ctypes
from ctypes import wintypes
import os
import subprocess
import time
import psutil


class BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class ThreadEntry(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
                ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG), ("dwFlags", wintypes.DWORD)]


class WindowsJob:
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("this registered supervisor is Windows-only")
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        self.api.SetInformationJobObject.restype = wintypes.BOOL
        self.api.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self.api.AssignProcessToJobObject.restype = wintypes.BOOL
        self.api.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        self.api.TerminateJobObject.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = (wintypes.HANDLE,)
        self.api.CloseHandle.restype = wintypes.BOOL
        self.api.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
        self.api.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        self.api.Thread32First.argtypes = (wintypes.HANDLE, ctypes.POINTER(ThreadEntry))
        self.api.Thread32First.restype = wintypes.BOOL
        self.api.Thread32Next.argtypes = (wintypes.HANDLE, ctypes.POINTER(ThreadEntry))
        self.api.Thread32Next.restype = wintypes.BOOL
        self.api.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        self.api.OpenThread.restype = wintypes.HANDLE
        self.api.ResumeThread.argtypes = (wintypes.HANDLE,)
        self.api.ResumeThread.restype = wintypes.DWORD
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def resume_suspended(self, process):
        """Resume only the initial thread of this just-created suspended child.

        CREATE_SUSPENDED prevents venv redirectors spawning an uncontained real
        interpreter before Job assignment. Snapshot is read-only; only the exact
        child PID's sole thread is operated on. No native imports before release.
        """
        snapshot = self.api.CreateToolhelp32Snapshot(0x4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        ids = []
        entry = ThreadEntry()
        entry.dwSize = ctypes.sizeof(entry)
        try:
            more = self.api.Thread32First(snapshot, ctypes.byref(entry))
            while more:
                if entry.th32OwnerProcessID == process.pid:
                    ids.append(entry.th32ThreadID)
                entry.dwSize = ctypes.sizeof(entry)
                more = self.api.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            self.api.CloseHandle(snapshot)
        if len(ids) != 1:
            raise RuntimeError("expected one suspended initial child thread")
        thread = self.api.OpenThread(0x2, False, ids[0])  # THREAD_SUSPEND_RESUME
        if not thread:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            count = self.api.ResumeThread(thread)
            if count != 1:
                raise RuntimeError(f"unexpected initial suspend count: {count}")
        finally:
            self.api.CloseHandle(thread)

    def terminate(self):
        if self.handle and not self.api.TerminateJobObject(self.handle, 124):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, None
            if not self.api.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())


def resource_sample(process):
    root = psutil.Process(process.pid)
    members = [root, *root.children(recursive=True)]
    rows = []
    for member in members:
        try:
            rows.append({"pid": member.pid, "created": member.create_time(), "rss_bytes": member.memory_info().rss})
        except psutil.NoSuchProcess:
            pass
    return {"elapsed_monotonic": time.monotonic(), "members": rows,
            "rss_bytes": sum(r["rss_bytes"] for r in rows),
            "host_available_bytes": psutil.virtual_memory().available}
