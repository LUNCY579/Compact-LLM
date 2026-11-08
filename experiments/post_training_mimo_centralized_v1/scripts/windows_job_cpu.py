"""Stdlib Windows process/Job witnesses. No model or GPU imports."""
from __future__ import annotations

import ctypes as c
from ctypes import wintypes as w
import os
import shutil
import time

QUERY = 0x1000
SYNC = 0x100000
TERMINATE = 0x0001
WAIT_TIMEOUT = 258


def require(value, message):
    if not value:
        raise RuntimeError(message)


def require_reserves(kernel, directory, host_floor=1.25):
    require(kernel.available() >= host_floor, 'CPU host reserve floor')
    require(shutil.disk_usage(directory).free / 2**30 >= 4, 'CPU ongoing disk reserve floor')


class Basic(c.Structure):
    _fields_ = [("PerProcessUserTimeLimit", c.c_longlong), ("PerJobUserTimeLimit", c.c_longlong),
                ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", c.c_size_t),
                ("MaximumWorkingSetSize", c.c_size_t), ("ActiveProcessLimit", w.DWORD),
                ("Affinity", c.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]


class IO(c.Structure):
    _fields_ = [(n, c.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class Extended(c.Structure):
    _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO), ("ProcessMemoryLimit", c.c_size_t),
                ("JobMemoryLimit", c.c_size_t), ("PeakProcessMemoryUsed", c.c_size_t), ("PeakJobMemoryUsed", c.c_size_t)]


class ProcessEntry(c.Structure):
    _fields_ = [("dwSize", w.DWORD), ("cntUsage", w.DWORD), ("th32ProcessID", w.DWORD),
                ("th32DefaultHeapID", c.c_size_t), ("th32ModuleID", w.DWORD), ("cntThreads", w.DWORD),
                ("th32ParentProcessID", w.DWORD), ("pcPriClassBase", c.c_long), ("dwFlags", w.DWORD),
                ("szExeFile", w.WCHAR * 260)]


class MemoryStatus(c.Structure):
    _fields_ = [("dwLength", w.DWORD), ("dwMemoryLoad", w.DWORD)] + [
        (name, c.c_ulonglong) for name in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                         "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]


class MemoryCounters(c.Structure):
    _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD)] + [
        (name, c.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                      "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]


class JobPids(c.Structure):
    _fields_ = [('assigned', w.DWORD), ('returned', w.DWORD), ('pids', c.c_size_t * 8)]


class JobAccounting(c.Structure):
    _fields_ = [(name, c.c_longlong) for name in (
        'TotalUserTime', 'TotalKernelTime', 'ThisPeriodTotalUserTime', 'ThisPeriodTotalKernelTime')] + [
        (name, w.DWORD) for name in ('TotalPageFaultCount', 'TotalProcesses', 'ActiveProcesses', 'TotalTerminatedProcesses')]


class ThreadEntry(c.Structure):
    _fields_ = [('dwSize', w.DWORD), ('cntUsage', w.DWORD),
                ('th32ThreadID', w.DWORD), ('th32OwnerProcessID', w.DWORD),
                ('tpBasePri', c.c_long), ('tpDeltaPri', c.c_long), ('dwFlags', w.DWORD)]


def membership_result(api_return, last_error, output):
    """A failed query has no valid membership output, even if its buffer is truthy."""
    return {"api_return": int(api_return), "last_error": int(last_error),
            "api_success": bool(api_return), "member": bool(output) if api_return else None}


class Kernel:
    def __init__(self, record=lambda event: None):
        require(os.name == 'nt', 'Windows-only CPU witness')
        self.observer, self.logging_errors = record, []
        self.dll = c.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'GetCurrentProcess': ([], w.HANDLE),
            'OpenProcess': ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            'GetProcessTimes': ([w.HANDLE, *([c.POINTER(w.FILETIME)] * 4)], w.BOOL),
            'GetExitCodeProcess': ([w.HANDLE, c.POINTER(w.DWORD)], w.BOOL),
            'QueryFullProcessImageNameW': ([w.HANDLE, w.DWORD, w.LPWSTR, c.POINTER(w.DWORD)], w.BOOL),
            'CreateToolhelp32Snapshot': ([w.DWORD, w.DWORD], w.HANDLE),
            'Process32FirstW': ([w.HANDLE, c.POINTER(ProcessEntry)], w.BOOL),
            'Process32NextW': ([w.HANDLE, c.POINTER(ProcessEntry)], w.BOOL),
            'Thread32First': ([w.HANDLE, c.POINTER(ThreadEntry)], w.BOOL),
            'Thread32Next': ([w.HANDLE, c.POINTER(ThreadEntry)], w.BOOL),
            'OpenThread': ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            'GetProcessIdOfThread': ([w.HANDLE], w.DWORD),
            'GetThreadTimes': ([w.HANDLE, *([c.POINTER(w.FILETIME)] * 4)], w.BOOL),
            'ResumeThread': ([w.HANDLE], w.DWORD),
            'CreateJobObjectW': ([c.c_void_p, w.LPCWSTR], w.HANDLE),
            'SetInformationJobObject': ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            'QueryInformationJobObject': ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.POINTER(w.DWORD)], w.BOOL),
            'AssignProcessToJobObject': ([w.HANDLE, w.HANDLE], w.BOOL),
            'IsProcessInJob': ([w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)], w.BOOL),
            'DuplicateHandle': ([w.HANDLE, w.HANDLE, w.HANDLE, c.POINTER(w.HANDLE), w.DWORD, w.BOOL, w.DWORD], w.BOOL),
            'TerminateJobObject': ([w.HANDLE, w.UINT], w.BOOL),
            'TerminateProcess': ([w.HANDLE, w.UINT], w.BOOL),
            'WaitForSingleObject': ([w.HANDLE, w.DWORD], w.DWORD),
            'CloseHandle': ([w.HANDLE], w.BOOL),
            'GlobalMemoryStatusEx': ([c.POINTER(MemoryStatus)], w.BOOL),
            'K32GetProcessMemoryInfo': ([w.HANDLE, c.c_void_p, w.DWORD], w.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.dll, name)
            function.argtypes, function.restype = arguments, result

    def record(self, event):
        try:
            self.observer(event)
        except Exception as error:
            # An evidence-writing failure must not interrupt later owned cleanup.
            self.logging_errors.append(type(error).__name__ + ': ' + str(error))

    def call(self, name, *args, context=None):
        c.set_last_error(0)
        started = time.time_ns()
        monotonic_started = time.monotonic_ns()
        result = getattr(self.dll, name)(*args)
        error = c.get_last_error()  # Capture immediately, before logging or another API.
        self.record({'event': 'windows_api', 'api': name, 'call_started_utc_ns': started,
                     'call_started_monotonic_ns': monotonic_started,
                     'call_returned_utc_ns': time.time_ns(), 'call_returned_monotonic_ns': time.monotonic_ns(), 'return': int(result or 0),
                     'last_error': int(error), 'context': context})
        return result, error

    def close(self, handle):
        if handle:
            result, _ = self.call('CloseHandle', handle)
            require(result, 'owned handle close failed')

    def current(self):
        return self.call('GetCurrentProcess')[0]

    def open(self, pid, rights=QUERY | SYNC):
        handle, error = self.call('OpenProcess', rights, False, pid, context={'pid': pid, 'rights': rights})
        require(handle, f'owned process query open failed: {error}')
        return handle

    def identity(self, handle, pid):
        creation, exit_time, kernel, user = [w.FILETIME() for _ in range(4)]
        result, error = self.call('GetProcessTimes', handle, c.byref(creation), c.byref(exit_time), c.byref(kernel), c.byref(user), context={'pid': pid})
        require(result, f'owned creation identity unavailable: {error}')
        size, image = w.DWORD(32768), c.create_unicode_buffer(32768)
        result, error = self.call('QueryFullProcessImageNameW', handle, 0, image, c.byref(size), context={'pid': pid})
        require(result, f'owned image identity unavailable: {error}')
        return {'pid': pid, 'created_filetime': (creation.dwHighDateTime << 32) | creation.dwLowDateTime,
                'image_path': image.value}

    def parent_table(self):
        snapshot, error = self.call('CreateToolhelp32Snapshot', 2, 0)
        require(snapshot and snapshot != c.c_void_p(-1).value, f'lineage snapshot failed: {error}')
        table = {}
        try:
            entry = ProcessEntry()
            entry.dwSize = c.sizeof(entry)
            c.set_last_error(0)
            result = self.dll.Process32FirstW(snapshot, c.byref(entry))
            require(result, f'lineage enumeration failed: {c.get_last_error()}')
            while result:
                table[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                c.set_last_error(0)
                result = self.dll.Process32NextW(snapshot, c.byref(entry))
            last = c.get_last_error()
            require(last == 18, f'lineage enumeration incomplete: {last}')
            self.record({'event': 'lineage_snapshot_complete', 'entries': len(table), 'terminal_error': last})
            return table
        finally:
            self.close(snapshot)

    def lineage(self, pid, stop_pid):
        table, result = self.parent_table(), []
        seen = set()
        while pid and pid not in seen:
            seen.add(pid)
            result.append({'pid': pid, 'parent_pid': table.get(pid)})
            if pid == stop_pid:
                return result
            pid = table.get(pid, 0)
        raise RuntimeError('CPU worker is not a descendant of its owned supervisor')

    def thread_ids(self, owned_pid):
        snapshot, error = self.call('CreateToolhelp32Snapshot', 4, 0)
        require(snapshot and snapshot != c.c_void_p(-1).value, f'owned thread snapshot failed: {error}')
        selected, count = [], 0
        try:
            entry = ThreadEntry()
            entry.dwSize = c.sizeof(entry)
            c.set_last_error(0)
            result = self.dll.Thread32First(snapshot, c.byref(entry))
            require(result, f'owned thread enumeration failed: {c.get_last_error()}')
            while result:
                count += 1
                if entry.th32OwnerProcessID == owned_pid:
                    selected.append(int(entry.th32ThreadID))
                c.set_last_error(0)
                result = self.dll.Thread32Next(snapshot, c.byref(entry))
            last = c.get_last_error()
            require(last == 18, f'owned thread enumeration incomplete: {last}')
            self.record({'event': 'owned_thread_snapshot', 'owned_pid': owned_pid,
                         'owned_thread_ids': selected, 'entries_count': count, 'terminal_error': last})
            return selected
        finally:
            self.close(snapshot)

    def membership(self, process_handle, job_handle, label):
        output = w.BOOL(-1)
        result, error = self.call('IsProcessInJob', process_handle, job_handle, c.byref(output), context=label)
        witness = membership_result(result, error, output.value)
        self.record({'event': 'membership_output', 'label': label, **witness})
        return witness

    def available(self):
        status = MemoryStatus()
        status.dwLength = c.sizeof(status)
        result, _ = self.call('GlobalMemoryStatusEx', c.byref(status))
        require(result, 'host reserve query failed')
        return status.ullAvailPhys / 2**30

    def rss(self, handle):
        info = MemoryCounters()
        info.cb = c.sizeof(info)
        result, error = self.call('K32GetProcessMemoryInfo', handle, c.byref(info), c.sizeof(info))
        require(result, f'CPU process memory query failed: {error}')
        return int(info.WorkingSetSize)

    def job_pids(self, job_handle):
        info, returned = JobPids(), w.DWORD()
        result, error = self.call('QueryInformationJobObject', job_handle, 3, c.byref(info), c.sizeof(info), c.byref(returned))
        require(result and info.returned == info.assigned and info.returned <= 8, f'owned Job process list unavailable: {error}')
        return list(info.pids[:info.returned])

    def job_accounting(self, job_handle):
        info, returned = JobAccounting(), w.DWORD()
        result, error = self.call('QueryInformationJobObject', job_handle, 1,
                                  c.byref(info), c.sizeof(info), c.byref(returned))
        require(result and returned.value == c.sizeof(info), f'owned Job lifetime accounting unavailable: {error}')
        witness = {'total_processes': int(info.TotalProcesses), 'active_processes': int(info.ActiveProcesses),
                   'total_terminated_processes': int(info.TotalTerminatedProcesses)}
        self.record({'event': 'owned_job_lifetime_accounting', **witness})
        return witness

    def exit_code(self, process_handle, label):
        code = w.DWORD()
        result, error = self.call('GetExitCodeProcess', process_handle, c.byref(code), context=label)
        require(result, f'owned process exit code unavailable: {error}')
        self.record({'event': 'owned_process_exit_code', 'label': label, 'exit_code': int(code.value)})
        return int(code.value)


class TracedJob:
    """The original WindowsJob sequence/types/flags, with immediate API witnesses."""
    def __init__(self, kernel, process):
        self.kernel, self.handle = kernel, None
        self.handle, error = kernel.call('CreateJobObjectW', None, None)
        require(self.handle, f'Job creation failed: {error}')
        try:
            limits = Extended()
            limits.BasicLimitInformation.LimitFlags = 0x2000
            result, error = kernel.call('SetInformationJobObject', self.handle, 9, c.byref(limits), c.sizeof(limits))
            require(result, f'kill-on-close setup failed: {error}')
            result, error = kernel.call('AssignProcessToJobObject', self.handle, int(process._handle), context={'target_pid': process.pid})
            require(result, f'launcher assignment failed: {error}')
            queried, returned = Extended(), w.DWORD()
            result, error = kernel.call('QueryInformationJobObject', self.handle, 9, c.byref(queried), c.sizeof(queried), c.byref(returned))
            require(result and queried.BasicLimitInformation.LimitFlags == 0x2000, f'Job limits changed: {error}')
        except BaseException:
            self.close()
            raise

    def terminate(self):
        result, error = self.kernel.call('TerminateJobObject', self.handle, 2)
        require(result, f'owned Job termination failed: {error}')

    def close(self):
        if self.handle:
            self.kernel.close(self.handle)
            self.handle = None


def validate_owned_identity(expected, observed, lineage, supervisor_pid, root_pid):
    require(expected == observed, 'owned CPU worker PID/create-time/image identity mismatch')
    ids = [entry['pid'] for entry in lineage]
    require(ids[0] == expected['pid'] and ids[-1] == supervisor_pid and root_pid in ids,
            'owned CPU worker lineage mismatch')


def cleanup_owned(kernel, job, children, process, supervisor_pid, manual_fallback=True):
    """Attempt every owned cleanup even after an API, logging or identity failure."""
    errors, witnesses, fallbacks = [], [], []
    if job is not None:
        try:
            job.close()
        except Exception as error:
            errors.append('Job close: ' + str(error))
    for item in children.values():
        handle, identity = item['handle'], item['identity']
        try:
            wait, _ = kernel.call('WaitForSingleObject', handle, 2000)
            require(wait in (0, WAIT_TIMEOUT), 'owned process wait query failed')
            if wait == WAIT_TIMEOUT:
                # Cached lineage was verified while the launcher was live. The held
                # handle identifies the same object even after its parent exits.
                try:
                    observed = kernel.identity(handle, identity['pid'])
                except Exception as query_error:
                    exited, _ = kernel.call('WaitForSingleObject', handle, 2000)
                    require(exited == 0, 'unverified live identity after image query failure: ' + str(query_error))
                    witnesses.append({'identity': identity, 'wait_return': exited,
                                      'exit_during_identity_query': str(query_error)})
                    continue
                validate_owned_identity(identity, observed, item['lineage'], supervisor_pid, process.pid)
                require(manual_fallback, 'Job cleanup left an active owned process')
                result, error = kernel.call('TerminateProcess', handle, 2, context=identity)
                require(result, f'owned CPU process termination failed: {error}')
                fallbacks.append(identity['pid'])
            wait, _ = kernel.call('WaitForSingleObject', handle, 2000)
            witnesses.append({'identity': identity, 'wait_return': wait})
            require(wait == 0, 'owned CPU process survived cleanup')
        except Exception as error:
            errors.append('owned child ' + str(identity['pid']) + ': ' + str(error))
        finally:
            try:
                kernel.close(handle)
            except Exception as error:
                errors.append('owned child handle close: ' + str(error))
    if process is not None:
        try:
            if process.poll() is None:
                wait, _ = kernel.call('WaitForSingleObject', int(process._handle), 2000)
                require(wait in (0, WAIT_TIMEOUT), 'owned launcher wait failed')
            else:
                wait = 0
            if wait == WAIT_TIMEOUT:
                require(manual_fallback, 'Job cleanup left an active owned launcher')
                process.kill()  # Original Popen object/handle; never an enumerated PID kill.
                fallbacks.append(process.pid)
            process.wait(timeout=2)
            final_wait, _ = kernel.call('WaitForSingleObject', int(process._handle), 0)
            require(final_wait == 0, 'owned launcher handle not signaled after wait')
            witnesses.append({'role': 'launcher', 'pid': process.pid, 'exit_code': process.returncode,
                              'wait_return': final_wait})
        except Exception as error:
            errors.append('owned launcher cleanup: ' + str(error))
    return {'status': 'OWNED_CPU_PROCESSES_EXITED' if not errors else 'BLOCKED_OWNED_CLEANUP',
            'processes': witnesses, 'errors': errors, 'logging_errors': list(kernel.logging_errors),
            'owned_process_fallback_pids': fallbacks}
