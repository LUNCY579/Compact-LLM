"""CRS-001 cumulative source5: fixed K/O/C/U implementation, NOT_READY live.

K is the existing invocation launcher, sole owner of the outer Job and every
original returned process handle. O performs the unchanged canonical operations;
C invokes the immutable candidate3 library through closed inherited-pipe RPC.
Controlled kernel/transport fixtures execute these same role entrypoints. They
do not qualify Windows, runtime, durability, admission, or a UV execution.
"""
from __future__ import annotations

import argparse
import ctypes as ct
from ctypes import wintypes as wt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import secrets
import struct
import threading
import time

PACKAGE = Path(__file__).resolve().parent
GIB = 2**30
LIB_SHA = '82f211ebd5a49dfa9c2b8cbb098566f459181fab29f9ee19a2c6a917989d9cb0'
OPS_SHA = 'ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30'
AUTHORITY_SHA = 'c36a00c7f1fba55c9e1dc1b8952c29546d9eabe13dfc6b6f93bb34374b3994c9'
UV = 'C:/Users/clclc/.local/bin/uv.exe'
UV_SHA = '5a7ec85884c2ccb1be560cb8fac3eb890df1adf49bfcc070a270ba70401bdd68'
PYTHON = 'C:/Users/clclc/AppData/Roaming/uv/python/cpython-3.11.15-windows-x86_64-none/python.exe'
FRAME_MAX = 8192
LEAF_MAX = 524288
JOB_FLAGS = 0x2000 | 0x8 | 0x200
HANDLE_LIST, JOB_LIST = 0x20002, 0x2000D
RPC_OPERATIONS = frozenset(('CHECK', 'ROW', 'RESERVE', 'MARK', 'UNKNOWN', 'RECEIPT',
    'PORT', 'BIRTH', 'ACK', 'IDENTITY', 'ASSIGN', 'LIMITS', 'PIDS', 'RESUME',
    'OUTPUT', 'EXIT', 'MEMORY', 'TELEMETRY', 'KILL_U', 'CLEANUP', 'RECOVERY',
    'CALLER_DONE', 'FINALIZE', 'OBSERVER_DONE', 'BOOT', 'STOP'))
STORE_OPERATIONS = frozenset(('CHECK', 'ROW', 'RESERVE', 'MARK', 'UNKNOWN', 'RECEIPT'))


class Refused(RuntimeError):
    pass


def need(ok, message):
    if not ok:
        raise Refused(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def exact_module(name, filename, expected):
    path = PACKAGE / filename
    need(file_sha(path) == expected, 'fixed input drift: ' + filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def library():
    return exact_module('crs_immutable_tool3', 'cpu_only_tool_version.py', LIB_SHA)


def operations():
    return exact_module('crs_immutable_operations', 'operations.py', OPS_SHA)


def positive(value, why, zero=False):
    need(type(value) in (int, float) and math.isfinite(value) and
         (value >= 0 if zero else value > 0), why)
    return value


def durable_leaf(directory, name, payload):
    """Role leaf publication; C/O may block, independently bounded by K."""
    raw = canonical(payload)
    need(len(raw) <= LEAF_MAX, 'role leaf cap')
    path = Path(directory) / name
    need(path.parent.resolve() == Path(directory).resolve(), 'leaf escape')
    partial = path.with_name(name + '.partial-' + secrets.token_hex(16))
    with partial.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(partial, path)
    except FileExistsError:
        need(path.read_bytes() == raw, 'immutable leaf conflict')
    partial.unlink()
    need(path.read_bytes() == raw, 'durable role leaf readback')
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def read_leaf(ref, directory):
    path = Path(ref['path'])
    need(path.resolve().parent == Path(directory).resolve(), 'leaf reference escape')
    need(not path.is_symlink() and 0 < ref['bytes'] <= LEAF_MAX, 'leaf path/size')
    raw = path.read_bytes()
    need(len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'leaf hash')
    value = json.loads(raw)
    need(canonical(value) == raw, 'leaf canonical bytes')
    return value


class Codec:
    """Canonical bounded frames bind a channel to actual created role identity."""
    def __init__(self, nonce, role, binding, identity):
        need(isinstance(nonce, str) and len(nonce) == 64, 'channel nonce')
        self.nonce, self.role = nonce, role
        self.binding, self.identity = binding, identity
        self.sent = self.received = 0

    def encode(self, operation, payload):
        need(operation in RPC_OPERATIONS, 'closed operation set')
        raw = canonical({'nonce': self.nonce, 'role': self.role,
            'identity': self.identity, 'binding': self.binding,
            'sequence': self.sent, 'operation': operation, 'payload': payload})
        need(len(raw) <= FRAME_MAX, 'frame bound')
        self.sent += 1
        return struct.pack('<I', len(raw)) + raw

    def decode(self, frame):
        need(len(frame) >= 4, 'truncated frame')
        size = struct.unpack('<I', frame[:4])[0]
        need(0 < size <= FRAME_MAX and len(frame) == size + 4, 'frame count')
        value = json.loads(frame[4:])
        need(canonical(value) == frame[4:] and set(value) == {
            'nonce', 'role', 'identity', 'binding', 'sequence', 'operation', 'payload'}, 'frame schema')
        need(value['nonce'] == self.nonce and value['role'] == self.role and
            value['identity'] == self.identity and value['binding'] == self.binding and
            value['sequence'] == self.received and value['operation'] in RPC_OPERATIONS,
            'nonce/role/actual identity/binding/sequence mismatch')
        self.received += 1
        return value


class Rpc:
    def __init__(self, channel, codec):
        self.channel, self.codec = channel, codec
        self.lock = threading.Lock()

    def call(self, operation, payload=None):
        with self.lock:
            self.channel.send(self.codec.encode(operation, payload or {}))
            frame = self.codec.decode(self.channel.receive())
            need(frame['operation'] == operation, 'RPC response operation mismatch')
            reply = frame['payload']
            if 'error' in reply:
                raise Refused(reply['error'])
            need(set(reply) == {'value'}, 'RPC response schema')
            return reply['value']


class PhaseGuard:
    """Independent K timer permanently latches expiry; never joins role locks."""
    def __init__(self, kernel, clock, seconds, role):
        self.kernel, self.clock, self.role = kernel, clock, role
        self.started = clock()
        self.until = self.started + seconds
        self.expired = threading.Event()
        self.errors = []
        self.closed = False
        self.timer = threading.Timer(seconds, self.trip)
        self.timer.daemon = True
        self.timer.start()

    def trip(self):
        self.expired.set()
        try:
            self.kernel.stop_outer_once()
        except BaseException as exc:
            self.errors.append(type(exc).__name__ + ': ' + str(exc))

    def check(self):
        if self.clock() >= self.until:
            self.trip()
        need(not self.expired.is_set(), self.role + ' phase expired')

    def close(self):
        if self.closed:
            return True
        self.timer.cancel()
        # Cancellation does not erase a timer which already expired.
        if self.timer.is_alive():
            self.timer.join(timeout=0.1)
        self.closed = not self.timer.is_alive()
        return self.closed


class Memory(ct.Structure):
    _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD)] + [(n, ct.c_size_t) for n in
        ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
         'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


class Overlapped(ct.Structure):
    _fields_ = [('Internal', ct.c_size_t), ('InternalHigh', ct.c_size_t),
               ('Offset', wt.DWORD), ('OffsetHigh', wt.DWORD), ('hEvent', wt.HANDLE)]


class MemoryStatus(ct.Structure):
    _fields_ = [('dwLength', wt.DWORD), ('dwMemoryLoad', wt.DWORD)] + [(n, ct.c_ulonglong) for n in
        ('ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile', 'ullAvailPageFile', 'ullTotalVirtual',
         'ullAvailVirtual', 'ullAvailExtendedVirtual')]


class ProcessBasicInfo(ct.Structure):
    _fields_ = [('reserved', ct.c_void_p), ('peb', ct.c_void_p),
                ('reserved2', ct.c_void_p * 2), ('pid', ct.c_size_t), ('parent_pid', ct.c_size_t)]


class Win32Kernel:
    """Concrete typed original-handle API. Construction itself performs no birth."""
    actual = True
    def __init__(self, clock=time.monotonic, custody=None):
        need(os.name == 'nt', 'Windows only')
        if custody is not None:
            custody.kernel = self  # Partial constructor owner precedes fallible setup.
        self.clock = clock
        self.lib = library()
        self.k = ct.WinDLL('C:/Windows/System32/kernel32.dll', use_last_error=True)
        self.ps = ct.WinDLL('C:/Windows/System32/psapi.dll', use_last_error=True)
        self.nt = ct.WinDLL('C:/Windows/System32/ntdll.dll', use_last_error=True)
        self.nt.NtQueryInformationProcess.argtypes = [wt.HANDLE, wt.ULONG, ct.c_void_p, wt.ULONG, ct.POINTER(wt.ULONG)]
        self.nt.NtQueryInformationProcess.restype = wt.LONG
        self.held, self.pipes = {}, {}
        self.outer = self.inner = None
        self.birth_attempts = set()
        self.termination_attempts = set()
        self.outer_stop_requested = False
        self.pending_io = []
        self.pending_attributes = []
        self.parent_copies = {}
        self.birth_lock = threading.RLock()
        specs = [
            ('CreateJobObjectW', [ct.c_void_p, wt.LPCWSTR], wt.HANDLE),
            ('SetInformationJobObject', [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD], wt.BOOL),
            ('QueryInformationJobObject', [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.c_void_p], wt.BOOL),
            ('IsProcessInJob', [wt.HANDLE, wt.HANDLE, ct.POINTER(wt.BOOL)], wt.BOOL),
            ('TerminateJobObject', [wt.HANDLE, wt.UINT], wt.BOOL),
            ('CreatePipe', [ct.POINTER(wt.HANDLE), ct.POINTER(wt.HANDLE), ct.c_void_p, wt.DWORD], wt.BOOL),
            ('SetHandleInformation', [wt.HANDLE, wt.DWORD, wt.DWORD], wt.BOOL),
            ('PeekNamedPipe', [wt.HANDLE, ct.c_void_p, wt.DWORD, ct.c_void_p, ct.POINTER(wt.DWORD), ct.c_void_p], wt.BOOL),
            ('ReadFile', [wt.HANDLE, ct.c_void_p, wt.DWORD, ct.POINTER(wt.DWORD), ct.c_void_p], wt.BOOL),
            ('WriteFile', [wt.HANDLE, ct.c_void_p, wt.DWORD, ct.POINTER(wt.DWORD), ct.c_void_p], wt.BOOL),
            ('InitializeProcThreadAttributeList', [ct.c_void_p, wt.DWORD, wt.DWORD, ct.POINTER(ct.c_size_t)], wt.BOOL),
            ('UpdateProcThreadAttribute', [ct.c_void_p, wt.DWORD, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_void_p], wt.BOOL),
            ('DeleteProcThreadAttributeList', [ct.c_void_p], None),
            ('CreateProcessW', [wt.LPCWSTR, wt.LPWSTR, ct.c_void_p, ct.c_void_p, wt.BOOL, wt.DWORD,
                ct.c_void_p, wt.LPCWSTR, ct.POINTER(self.lib.StartupEx), ct.POINTER(self.lib.ProcessInfo)], wt.BOOL),
            ('ResumeThread', [wt.HANDLE], wt.DWORD),
            ('WaitForSingleObject', [wt.HANDLE, wt.DWORD], wt.DWORD),
            ('GetExitCodeProcess', [wt.HANDLE, ct.POINTER(wt.DWORD)], wt.BOOL),
            ('TerminateProcess', [wt.HANDLE, wt.UINT], wt.BOOL),
            ('GetProcessTimes', [wt.HANDLE] + [ct.POINTER(wt.FILETIME)] * 4, wt.BOOL),
            ('QueryFullProcessImageNameW', [wt.HANDLE, wt.DWORD, wt.LPWSTR, ct.POINTER(wt.DWORD)], wt.BOOL),
            ('GetCurrentProcess', [], wt.HANDLE),
            ('GetCurrentProcessId', [], wt.DWORD),
            ('GetProcessId', [wt.HANDLE], wt.DWORD),
            ('GetThreadId', [wt.HANDLE], wt.DWORD),
            ('DuplicateHandle', [wt.HANDLE, wt.HANDLE, wt.HANDLE, ct.POINTER(wt.HANDLE), wt.DWORD, wt.BOOL, wt.DWORD], wt.BOOL),
            ('CloseHandle', [wt.HANDLE], wt.BOOL),
            ('GlobalMemoryStatusEx', [ct.POINTER(MemoryStatus)], wt.BOOL),
            ('GetDiskFreeSpaceExW', [wt.LPCWSTR, ct.POINTER(ct.c_ulonglong), ct.c_void_p, ct.c_void_p], wt.BOOL),
            ('CreateFileW', [wt.LPCWSTR, wt.DWORD, wt.DWORD, ct.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE], wt.HANDLE),
            ('CreateEventW', [ct.c_void_p, wt.BOOL, wt.BOOL, wt.LPCWSTR], wt.HANDLE),
            ('GetOverlappedResult', [wt.HANDLE, ct.POINTER(Overlapped), ct.POINTER(wt.DWORD), wt.BOOL], wt.BOOL),
            ('CancelIoEx', [wt.HANDLE, ct.POINTER(Overlapped)], wt.BOOL),
        ]
        for name, args, result in specs:
            fn = getattr(self.k, name)
            fn.argtypes, fn.restype = args, result
        self.ps.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ct.c_void_p, wt.DWORD]
        self.ps.GetProcessMemoryInfo.restype = wt.BOOL
        self.khandle = self.k.GetCurrentProcess()

    def new_job(self, count):
        handle = self.k.CreateJobObjectW(None, None)
        need(handle, 'Job creation')
        if count == 3:
            self.outer = handle
        else:
            self.inner = handle
        limits = self.lib.Extended()
        limits.BasicLimitInformation.LimitFlags = JOB_FLAGS
        limits.BasicLimitInformation.ActiveProcessLimit = count
        limits.JobMemoryLimit = 2 * GIB
        need(self.k.SetInformationJobObject(handle, 9, ct.byref(limits), ct.sizeof(limits)), 'Job limit setting')
        self.job_limits(count)
        return handle

    def job_limits(self, count):
        handle = self.outer if count == 3 else self.inner
        x = self.lib.Extended()
        need(handle and self.k.QueryInformationJobObject(handle, 9, ct.byref(x), ct.sizeof(x), None), 'Job limits query')
        need(x.BasicLimitInformation.LimitFlags == JOB_FLAGS and
            x.BasicLimitInformation.ActiveProcessLimit == count and x.JobMemoryLimit == 2 * GIB, 'Job limits drift')
        return {'limit_flags': JOB_FLAGS, 'ActiveProcessLimit': count,
                'JobMemoryLimit': int(x.JobMemoryLimit), 'peak_commit': int(x.PeakJobMemoryUsed)}

    def members(self, count):
        class List(ct.Structure):
            _fields_ = [('assigned', wt.DWORD), ('returned', wt.DWORD), ('ids', ct.c_size_t * 4)]
        out = List()
        handle = self.outer if count == 3 else self.inner
        need(self.k.QueryInformationJobObject(handle, 3, ct.byref(out), ct.sizeof(out), None), 'Job membership query')
        need(out.assigned == out.returned and out.returned <= count, 'complete Job membership')
        return [int(out.ids[i]) for i in range(out.returned)]

    def in_job(self, role, count):
        value = wt.BOOL()
        handle = self.outer if count == 3 else self.inner
        need(self.k.IsProcessInJob(self.held[role].hProcess, handle, ct.byref(value)), 'held membership unavailable')
        need(value.value, 'birth Job membership absent')

    def streams(self, role):
        need(role not in self.pipes, 'duplicate role streams')
        state = self.pipes[role] = {}
        sa = self.lib.Security(ct.sizeof(self.lib.Security), None, True)
        for name in ('stdin', 'stdout', 'stderr'):
            r, w = wt.HANDLE(), wt.HANDLE()
            ok = self.k.CreatePipe(ct.byref(r), ct.byref(w), ct.byref(sa), 32768)
            if r.value:
                state[name + '_read'] = r.value
            if w.value:
                state[name + '_write'] = w.value
            need(ok and r.value and w.value, 'partial pipe allocation')
        for name in ('stdin_write', 'stdout_read', 'stderr_read'):
            need(self.k.SetHandleInformation(state[name], 1, 0), 'noninheritable K pipe')
        return [state['stdin_read'], state['stdout_write'], state['stderr_write']]

    def create(self, role, gate):
        with self.birth_lock:
            need(role in {'O', 'C', 'U'} and role not in self.birth_attempts, 'one role birth only')
            inherited = self.streams(role)
            if role != 'U':
                parent = wt.HANDLE()
                need(self.k.DuplicateHandle(self.khandle, self.khandle, self.khandle, ct.byref(parent),
                     0x1000 | 0x100000, True, 0) and parent.value, 'readonly inheritable exact K identity handle')
                self.parent_copies[role] = parent.value
                inherited.append(parent.value)
            size = ct.c_size_t()
            sized = self.k.InitializeProcThreadAttributeList(None, 2, 0, ct.byref(size))
            need(not sized and ct.get_last_error() == 122 and 0 < size.value <= 65536, 'two-attribute sizing')
            storage = ct.create_string_buffer(size.value)
            attribute = None
            try:
                need(self.k.InitializeProcThreadAttributeList(storage, 2, 0, ct.byref(size)), 'attribute initialization')
                attribute = ct.cast(storage, ct.c_void_p)
                jobs = (wt.HANDLE * (2 if role == 'U' else 1))(*([self.outer, self.inner] if role == 'U' else [self.outer]))
                handles = (wt.HANDLE * len(inherited))(*inherited)
                need(self.k.UpdateProcThreadAttribute(attribute, 0, JOB_LIST, jobs, ct.sizeof(jobs), None, None), 'birth Job list unsupported')
                need(self.k.UpdateProcThreadAttribute(attribute, 0, HANDLE_LIST, handles, ct.sizeof(handles), None, None), 'birth handle list')
                startup = self.lib.StartupEx()
                startup.StartupInfo.cb, startup.StartupInfo.dwFlags = ct.sizeof(startup), 0x100
                startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError = inherited[:3]
                startup.lpAttributeList = attribute
                app = UV if role == 'U' else PYTHON
                args = [app, '--version'] if role == 'U' else [app, '-I', '-S', '-B',
                    str(Path(__file__).resolve()), '--role', role, '--input-handle', str(inherited[0]),
                    '--output-handle', str(inherited[1]), '--parent-handle', str(inherited[3])]
                # Fixed quoting: paths cannot contain quotes; command names are fixed.
                need(all('"' not in a and '\x00' not in a for a in args), 'argv quoting')
                command = ct.create_unicode_buffer(' '.join('"' + a + '"' for a in args))
                env = {'SystemRoot': 'C:/Windows', 'WINDIR': 'C:/Windows', 'PATH': 'C:/Windows/System32',
                       'TEMP': gate['temp'], 'TMP': gate['temp']}
                envbuf = ct.create_unicode_buffer('\x00'.join(k + '=' + v for k, v in sorted(env.items())) + '\x00\x00')
                info = self.lib.ProcessInfo()
                self.birth_attempts.add(role)
                ok = self.k.CreateProcessW(app, command, None, None, True, 4 | 0x400 | 0x80000 | 0x08000000,
                    envbuf, gate['cwd'], ct.byref(startup), ct.byref(info))
                # Retain BOTH partial originals before any fallible witness/check.
                if info.hProcess or info.hThread:
                    self.held[role] = info
                need(ok and info.hProcess and info.hThread and info.dwProcessId > 0, 'birth failed/partial/ambiguous')
                self.in_job(role, 3)
                if role == 'U':
                    self.in_job(role, 1)
                for key in ('stdin_read', 'stdout_write', 'stderr_write'):
                    need(self.k.CloseHandle(self.pipes[role][key]), 'child end close')
                    self.pipes[role][key] = None
                if role == 'U':
                    need(self.k.CloseHandle(self.pipes[role]['stdin_write']), 'U stdin EOF')
                    self.pipes[role]['stdin_write'] = None
                return self.identity(role)
            finally:
                if attribute:
                    try:
                        self.k.DeleteProcThreadAttributeList(attribute)
                    except BaseException:
                        self.pending_attributes.append((storage, attribute))

    def identity(self, role):
        info = self.held[role]
        times = [wt.FILETIME() for _ in range(4)]
        need(self.k.GetProcessTimes(info.hProcess, *(ct.byref(x) for x in times)), 'held process times')
        image = ct.create_unicode_buffer(32768)
        size = wt.DWORD(len(image))
        need(self.k.QueryFullProcessImageNameW(info.hProcess, 0, image, ct.byref(size)), 'held image')
        argv = [UV, '--version'] if role == 'U' else [PYTHON, '-I', '-S', '-B', str(Path(__file__).resolve()), '--role', role]
        return {'pid': int(info.dwProcessId), 'creation_time': str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime),
                'image': image.value.replace('\\', '/'), 'command_hash': digest(argv),
                'command_hash_kind': 'REQUESTED_ARGV_NOT_OBSERVED_OS_COMMAND_LINE'}

    def parent_identity(self, handle=None):
        handle = handle or self.khandle
        times = [wt.FILETIME() for _ in range(4)]
        need(self.k.GetProcessTimes(handle, *(ct.byref(x) for x in times)), 'exact K handle times')
        buf = ct.create_unicode_buffer(32768); size = wt.DWORD(len(buf))
        need(self.k.QueryFullProcessImageNameW(handle, 0, buf, ct.byref(size)), 'exact K held image')
        pbi = ProcessBasicInfo(); length = wt.ULONG()
        need(self.nt.NtQueryInformationProcess(handle, 0, ct.byref(pbi), ct.sizeof(pbi), ct.byref(length)) >= 0 and
             length.value == ct.sizeof(pbi), 'held basic identity unavailable')
        return {'pid': int(pbi.pid), 'creation_time': str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime),
                'image': buf.value.replace('\\', '/')}

    def verify_parent(self, parent_handle, expected):
        need(self.parent_identity(parent_handle) == expected and self.k.WaitForSingleObject(parent_handle, 0) == 258,
             'inherited exact K identity/live signal')
        pbi = ProcessBasicInfo(); length = wt.ULONG()
        need(self.nt.NtQueryInformationProcess(self.khandle, 0, ct.byref(pbi), ct.sizeof(pbi), ct.byref(length)) >= 0 and
             length.value == ct.sizeof(pbi) and pbi.parent_pid == expected['pid'], 'actual role parent differs from K')

    def resume(self, role):
        self.in_job(role, 3)
        if role == 'U':
            self.in_job(role, 1)
        need(self.k.ResumeThread(self.held[role].hThread) == 1, 'primary suspension count')

    def copies_for_c(self):
        need('C' in self.held and 'U' in self.held, 'exact C/U handles')
        result = {}
        for name, handle in (('process', self.held['U'].hProcess), ('thread', self.held['U'].hThread), ('inner', self.inner)):
            out = wt.HANDLE()
            ok = self.k.DuplicateHandle(self.khandle, handle, self.held['C'].hProcess, ct.byref(out), 0, False, 2)
            if out.value:
                result[name] = int(out.value)
            need(ok and out.value, 'partial handle copy; originals retained')
        return result

    def verify_copies(self, returned):
        copies, identity = returned['copies'], returned['identity']
        need(self.k.GetProcessId(copies['process']) == identity['pid'] and
             self.k.GetThreadId(copies['thread']) == returned['thread_id'], 'C received handle type/PID/thread mismatch')
        info = self.lib.ProcessInfo()
        info.hProcess, info.dwProcessId = copies['process'], identity['pid']
        self.held['U'] = info
        need(self.identity('U') == identity, 'C copied process actual identity')
        limits = self.lib.Extended()
        need(self.k.QueryInformationJobObject(copies['inner'], 9, ct.byref(limits), ct.sizeof(limits), None) and
             limits.BasicLimitInformation.LimitFlags == JOB_FLAGS and
             limits.BasicLimitInformation.ActiveProcessLimit == 1 and limits.JobMemoryLimit == 2 * GIB,
             'C actual received inner Job')
        member = wt.BOOL()
        need(self.k.IsProcessInJob(copies['process'], copies['inner'], ct.byref(member)) and member.value,
             'C copied handle actual inner membership')

    def snapshot(self, role):
        handle = self.khandle if role == 'K' else self.held[role].hProcess
        memory = Memory(); memory.cb = ct.sizeof(memory)
        need(self.ps.GetProcessMemoryInfo(handle, ct.byref(memory), ct.sizeof(memory)), 'held RSS/peak/commit')
        times = [wt.FILETIME() for _ in range(4)]
        need(self.k.GetProcessTimes(handle, *(ct.byref(x) for x in times)), 'held CPU times')
        cpu = sum((x.dwHighDateTime << 32) | x.dwLowDateTime for x in times[2:]) / 10000000
        state = 258 if role == 'K' else self.k.WaitForSingleObject(handle, 0)
        need(state in (0, 258), 'held signal unknown')
        code = None
        if state == 0:
            value = wt.DWORD()
            need(self.k.GetExitCodeProcess(handle, ct.byref(value)) and value.value != 259, 'exit code unknown')
            code = int(value.value)
        return {'rss_bytes': int(memory.WorkingSetSize), 'peak_rss_bytes': int(memory.PeakWorkingSetSize),
                'private_commit_bytes': int(memory.PagefileUsage), 'peak_private_commit_bytes': int(memory.PeakPagefileUsage),
                'CPU_seconds': cpu, 'signal': state == 0, 'exit_code': code, 'sampled_at': self.clock()}

    def read(self, role, name, maximum):
        handle = self.pipes[role][name + '_read']
        available = wt.DWORD()
        ok = self.k.PeekNamedPipe(handle, None, 0, None, ct.byref(available), None)
        if not ok:
            need(ct.get_last_error() == 109, 'pipe EOF unknown')
            return b'', True
        if not available.value:
            return b'', False
        n = min(int(available.value), maximum)
        need(n > 0, 'pipe capture cap')
        buffer = ct.create_string_buffer(n); count = wt.DWORD()
        need(self.k.ReadFile(handle, buffer, n, ct.byref(count), None), 'pipe read')
        need(0 < count.value <= n, 'pipe read count')
        return buffer.raw[:count.value], False

    def write(self, role, raw):
        """May block; K invokes this only through its independent writer thread."""
        buffer = ct.create_string_buffer(raw); count = wt.DWORD()
        need(self.k.WriteFile(self.pipes[role]['stdin_write'], buffer, len(raw), ct.byref(count), None)
             and count.value == len(raw), 'pipe write completion')

    def kill_u(self):
        need(self.inner and self.k.TerminateJobObject(self.inner, 2), 'inner Job stop')

    def stop_outer_once(self):
        if not self.outer_stop_requested:
            self.outer_stop_requested = True
            if self.outer:
                need(self.k.TerminateJobObject(self.outer, 2), 'outer Job stop failed; originals retained')

    def cleanup_u(self):
        errors = []
        try:
            self.kill_u()
        except BaseException as exc:
            errors.append(str(exc))
        if 'U' in self.held:
            info = self.held['U']
            if self.k.WaitForSingleObject(info.hProcess, 0) == 258:
                if 'U' not in self.termination_attempts:
                    self.termination_attempts.add('U')
                    if not self.k.TerminateProcess(info.hProcess, 2):
                        errors.append('genuinely live exact termination failed')
            if self.k.WaitForSingleObject(info.hProcess, 0) != 0:
                errors.append('held U signal unproved')
        if self.members(1):
            errors.append('inner membership not empty')
        return errors

    def telemetry(self):
        memory = MemoryStatus(); memory.dwLength = ct.sizeof(memory)
        free = ct.c_ulonglong()
        need(self.k.GlobalMemoryStatusEx(ct.byref(memory)), 'host RAM unavailable')
        need(self.k.GetDiskFreeSpaceExW(str(PACKAGE), ct.byref(free), None, None), 'disk unavailable')
        return {'age_seconds': 0.0, 'sampled_at': self.clock(), 'host_available_bytes': int(memory.ullAvailPhys),
                'disk_free_bytes': int(free.value), 'source': 'GlobalMemoryStatusEx/GetDiskFreeSpaceExW'}

    def release(self):
        need(not self.pending_io and not self.pending_attributes, 'pending exact I/O/attribute owners retained')
        need(self.members(3) == [] and self.members(1) == [], 'release requires complete empty Jobs')
        for role in self.held:
            need(self.snapshot(role)['signal'], 'release requires each original signal')
        for info in self.held.values():
            for handle in (info.hThread, info.hProcess):
                if handle:
                    need(self.k.CloseHandle(handle), 'original handle close')
        for values in self.pipes.values():
            for handle in values.values():
                if handle:
                    need(self.k.CloseHandle(handle), 'pipe close')
        for handle in self.parent_copies.values():
            need(self.k.CloseHandle(handle), 'K parent identity copy close')
        for name in ('inner', 'outer'):
            value = getattr(self, name)
            if value:
                need(self.k.CloseHandle(value), 'Job close')
                setattr(self, name, None)

    def publish_whole(self, directory, name, payload, phase):
        return Win32Publication(self, directory, name, payload, phase).publish()


class Win32Publication:
    """CREATE_NEW write-through overlapped publication; no unproved close/retry.

    Overlapped WriteFile can itself block on Windows. The independent phase timer
    latches expiry regardless; a late return is never ACCEPTED. Cancellation is
    only a request. Exact file/event/buffer/OVERLAPPED owners remain reachable
    until completed I/O is positively witnessed. Hardware durability is gated.
    """
    def __init__(self, kernel, directory, name, payload, phase):
        self.kernel, self.k, self.phase = kernel, kernel.k, phase
        self.path = Path(directory) / name
        self.raw = canonical(payload)
        need(0 < len(self.raw) <= LEAF_MAX and self.path.parent.resolve() == Path(directory).resolve(), 'whole payload/path')
        self.handle = self.event = None
        self.buffer = self.overlapped = None
        self.pending = False
        self.completed = False
        kernel.pending_io.append(self)

    def transfer(self, writing, length):
        self.phase.check()
        self.event = self.k.CreateEventW(None, True, False, None)
        need(self.event, 'publication event')
        self.overlapped = Overlapped(); self.overlapped.hEvent = self.event
        self.buffer = ct.create_string_buffer(self.raw if writing else length)
        count = wt.DWORD()
        self.pending = True
        ok = (self.k.WriteFile if writing else self.k.ReadFile)(self.handle, self.buffer, length, ct.byref(count), ct.byref(self.overlapped))
        error = 0 if ok else ct.get_last_error()
        need(ok or error == 997, 'publication I/O refused; owner retained')
        while True:
            done = wt.DWORD()
            if self.k.GetOverlappedResult(self.handle, ct.byref(self.overlapped), ct.byref(done), False):
                self.pending = False
                need(done.value == length, 'publication exact byte completion')
                data = self.buffer.raw[:done.value]
                need(self.k.CloseHandle(self.event), 'publication event close')
                self.event = None
                self.phase.check()
                return data
            need(ct.get_last_error() == 996, 'publication completion unknown; exact owner retained')
            if self.phase.expired.is_set() or self.kernel.clock() >= self.phase.until:
                self.phase.trip()
                self.k.CancelIoEx(self.handle, ct.byref(self.overlapped))
                # No completion is inferred from CancelIoEx or ERROR_NOT_FOUND.
                raise Refused('publication pending at deadline; file/event/buffer/OVERLAPPED retained')
            time.sleep(0.001)

    def publish(self):
        flags = 0x40000000 | 0x80000000  # OVERLAPPED + WRITE_THROUGH
        self.handle = self.k.CreateFileW(str(self.path), 0xC0000000, 0, None, 1, flags, None)
        need(self.handle not in (None, 0, ct.c_void_p(-1).value), 'CREATE_NEW immutable publication refused')
        self.transfer(True, len(self.raw))
        need(self.transfer(False, len(self.raw)) == self.raw, 'whole exact readback mismatch')
        self.phase.check()
        need(self.k.CloseHandle(self.handle), 'publication file close')
        self.handle = None
        self.completed = True
        self.kernel.pending_io.remove(self)
        return {'path': str(self.path), 'sha256': hashlib.sha256(self.raw).hexdigest(), 'bytes': len(self.raw),
                'interface': 'CREATE_NEW/WRITE_THROUGH/completed OVERLAPPED write/readback',
                'hardware_durability_qualified_by_fixture': False}


def physical(path, expected=None):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path.absolute(), 'noncanonical physical path')
    for part in (path, *path.parents):
        need(not part.is_symlink() and not (getattr(part.stat(), 'st_file_attributes', 0) & 0x400), 'reparse path refused')
    if expected:
        need(set(expected) == {'path', 'bytes', 'sha256'} and str(path) == expected['path'], 'physical pin schema')
        need(path.stat().st_size == expected['bytes'] and file_sha(path) == expected['sha256'], 'physical byte pin')
    return str(path)


def read_bounded_json(path):
    physical(path)
    need(0 < Path(path).stat().st_size <= LEAF_MAX, 'qualification record cap')
    return json.loads(Path(path).read_text(encoding='utf-8'))


class ActualAdmission:
    """Evidence-specific verified producers; library booleans are derived last.

    The invocation grant must pin the complete capsule digest and exactly one
    key. Source review/synthetic capsules are explicitly not actual launch grants.
    No current actual grant or capsule is supplied by integration014.
    """
    def __init__(self, capsule, grant, grant_sha, clock=time.monotonic):
        self.capsule_path, self.grant_path, self.grant_sha = capsule, grant, grant_sha
        self.clock = clock
        self.value = None

    def verify(self, kernel):
        need(kernel.actual, 'actual admission cannot use controlled kernel')
        # Trust root is the separately adopted physical role contract, not CLI
        # strings or a supplied accepted=True field. Registry is empty in014.
        authority = PACKAGE / 'authorization/OWNER_CALLER_OBSERVER_RESUMPTION_2026-10-10.json'
        need(file_sha(authority) == AUTHORITY_SHA, 'actual retained human authority drift')
        policy = read_bounded_json(PACKAGE / 'contracts/CALLER_OBSERVER_INTEGRATION_V1.json')
        grants = policy['actual_grant_registry']
        need(type(grants) is list and len(grants) <= 1 and policy['actual_actions_allowed'] == 1 and
             self.grant_sha in grants, 'no independently adopted actual grant; source scope only')
        physical(self.grant_path)
        need(Path(self.grant_path).resolve().parent == (PACKAGE / 'authorization').resolve(),
             'actual grant outside authoritative registry directory')
        need(file_sha(self.grant_path) == self.grant_sha, 'exact grant hash')
        grant = read_bounded_json(self.grant_path)
        value = read_bounded_json(self.capsule_path)
        need(grant.get('kind') == 'ONE_EXACT_ACTUAL_METADATA_LAUNCH' and
             grant.get('attempts_max') == 1 and grant.get('retry') is False and
             grant.get('capsule_sha256') == digest(value), 'absent actual evidence-specific grant')
        need(value.get('status') == 'ACTUAL_RUNTIME_TRUST_AND_ADMISSION_QUALIFIED' and
             value.get('synthetic') is False, 'synthetic/source proof cannot qualify actual runtime')
        g = value['gate']
        need(grant['key'] == g['key'] and grant['request_hash'] == g['request_hash'], 'grant exact request')
        pins = value['physical_files']
        need(0 < len(pins) <= 64, 'finite physical pins')
        by_path = {}
        for row in pins:
            physical(row['path'], row)
            by_path[row['path'].replace('\\', '/').casefold()] = row['sha256']
        required = {str(PACKAGE / 'cpu_only_tool_version.py').replace('\\', '/'): LIB_SHA,
                    str(PACKAGE / 'operations.py').replace('\\', '/'): OPS_SHA,
                    UV: UV_SHA}
        for path, sha in required.items():
            need(by_path.get(path.casefold()) == sha, 'missing exact physical input')
        for path in (PYTHON, 'C:/Windows/System32/kernel32.dll', 'C:/Windows/System32/psapi.dll', 'C:/Windows/System32/ntdll.dll',
                     str(Path(__file__).resolve()).replace('\\', '/'),
                     str(PACKAGE / 'contracts/CALLER_OBSERVER_INTEGRATION_V1.json').replace('\\', '/')):
            need(path.casefold() in by_path, 'unbound runtime/controller source/DLL')
        need(value['python_version'] == '3.11.15' and value['python_path'] == PYTHON,
             'exact Python runtime')
        source_sha = by_path[str(Path(__file__).resolve()).replace('\\', '/').casefold()]
        contract_sha = by_path[str(PACKAGE / 'contracts/CALLER_OBSERVER_INTEGRATION_V1.json').replace('\\', '/').casefold()]
        role_ids = set()
        need(len(value['independent_reviews']) == 2, 'two complete reviews')
        for item in value['independent_reviews']:
            physical(item['path'], item)
            record = read_bounded_json(item['path'])
            need(record['role'] in {'SCI', 'OPS'} and record['role'] not in role_ids and
                record['reviewer'] != value['builder'] and record['source_sha256'] == source_sha and
                record['contract_sha256'] == contract_sha and record['source_commit'] == g['source_commit'] and
                record['verdict'] == 'ACCEPT_ACTUAL_EXACT_COMPLETE_PATH', 'nonbuilder exact review proof absent')
            role_ids.add(record['role'])
        # Pinned qualification evidence identifies what was measured, not booleans.
        for name in ('Windows_birth_pipe_peak', 'runtime_DLL_search', 'trusted_metadata_boundary',
                     'whole_publication_volume_durability', 'host_process_and_unknown_reconciliation'):
            item = value['qualification_records'][name]
            physical(item['path'], item)
            proof = read_bounded_json(item['path'])
            need(proof['kind'] == name and proof['status'] == 'QUALIFIED_ACTUAL' and
                 proof['source_sha256'] == source_sha and proof['runtime_pin'] == g['runtime_pin'] and
                 proof.get('synthetic') is False and proof.get('complete_evidence_refs'), 'missing actual ' + name)
            refs = proof['complete_evidence_refs']
            need(type(refs) is list and 0 < len(refs) <= 16, 'finite concrete qualification refs')
            for ref in refs:
                physical(ref['path'], ref)
                facts = read_bounded_json(ref['path'])
                need(facts['kind'] == name + '_measurement' and facts['runtime_pin'] == g['runtime_pin'] and
                     facts['source_sha256'] == source_sha and facts['tool_path'] == UV and
                     facts['tool_sha256'] == UV_SHA and facts['synthetic'] is False and
                     facts['measured_process_identity']['creation_time'] and
                     facts['measured_process_identity']['image'] and
                     type(facts['measured_process_identity']['pid']) is int and
                     facts['measurement_method'] in proof['accepted_measurement_methods'] and
                     facts['raw_observation_ref']['sha256'] in proof['accepted_raw_sha256'],
                     'qualification measured identity/raw/method mismatch')
                physical(facts['raw_observation_ref']['path'], facts['raw_observation_ref'])
            if name == 'host_process_and_unknown_reconciliation':
                need(0 <= self.clock() - proof['sampled_monotonic'] <= 5 and
                     proof['canonical_DB_path'] == g['db_path'] and
                     proof['actual_owned_processes'] == [] and proof['unknown_outcomes'] == [],
                     'fresh process/unknown/DB binding')
        # Git is read directly; no child process or shell is part of admission.
        gitfile = PACKAGE.parent.parent / '.git'
        if gitfile.is_file():
            text = gitfile.read_text().strip()
            need(text.startswith('gitdir: '), 'Git directory identity')
            gitdir = Path(text[8:]).resolve()
        else:
            gitdir = gitfile
        head = (gitdir / 'HEAD').read_text().strip()
        if head.startswith('ref: '):
            ref = head[5:]
            local = gitdir / ref
            if not local.exists() and (gitdir / 'commondir').exists():
                local = (gitdir / (gitdir / 'commondir').read_text().strip()).resolve() / ref
            head = local.read_text().strip()
        need(head == g['source_commit'], 'source commit drift')
        clean = value['clean_commit_record']
        physical(clean['path'], clean)
        clean_record = read_bounded_json(clean['path'])
        need(clean_record['commit'] == head and clean_record['tracked_manifest_sha256'] == digest(pins) and
             clean_record['independent_verified_git_index_sha256'] == file_sha(gitdir / 'index'), 'clean commit/index binding')
        # The exact tree blob bindings are mandatory in addition to an index hash.
        # Only loose canonical Git objects are supported; packed-only refuses.
        import zlib
        common = gitdir
        if (gitdir / 'commondir').exists():
            common = (gitdir / (gitdir / 'commondir').read_text().strip()).resolve()
        def object_bytes(oid, kind):
            need(isinstance(oid, str) and len(oid) == 40 and all(c in '0123456789abcdef' for c in oid), 'Git object identity')
            stored = common / 'objects' / oid[:2] / oid[2:]
            need(stored.exists() and stored.stat().st_size <= 4 * LEAF_MAX, 'unavailable/bounded loose Git object')
            raw = zlib.decompress(stored.read_bytes())
            need(hashlib.sha1(raw).hexdigest() == oid, 'Git object hash')
            header, body = raw.split(b'\0', 1)
            need(header == (kind + ' ' + str(len(body))).encode(), 'Git object kind/size')
            return body
        commit = object_bytes(head, 'commit')
        tree = commit.splitlines()[0]
        need(tree.startswith(b'tree ') and tree[5:].decode() == clean_record['tree_oid'], 'commit tree binding')
        root_tree = clean_record['tree_oid']
        for item in clean_record['tracked_blobs']:
            parts = item['repository_path'].split('/')
            oid = root_tree
            for index, part in enumerate(parts):
                body = object_bytes(oid, 'tree')
                entries = {}
                while body:
                    label, tail = body.split(b'\0', 1)
                    mode, name = label.split(b' ', 1)
                    entries[name.decode()] = (mode, tail[:20].hex())
                    body = tail[20:]
                need(part in entries, 'tracked tree path absent')
                mode, oid = entries[part]
                need(mode == (b'40000' if index < len(parts) - 1 else b'100644'), 'tracked mode/symlink refused')
            blob = object_bytes(oid, 'blob')
            need(oid == item['blob_oid'] and hashlib.sha256(blob).hexdigest() == item['sha256'] and
                 file_sha(item['physical_path']) == item['sha256'], 'working bytes differ from frozen Git tree')
        for name in ('cwd', 'temp', 'leaf_dir'):
            physical(g[name])
            need(Path(g[name]).is_relative_to(PACKAGE / '.venv/tool-version-007'), 'actual scoped roots')
            need(not any(Path(g[name]).iterdir()), 'output/temp root not empty')
        physical(g['db_path'])
        # Read-only gate first: instantiate the canonical store only inside O.
        import sqlite3
        with sqlite3.connect('file:' + Path(g['db_path']).as_posix() + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            row = db.execute('SELECT * FROM jobs WHERE key=?', (g['key'],)).fetchone()
            lease = db.execute("SELECT * FROM leases WHERE kind='model'").fetchone()
            need(row and row['state'] == 'READY' and row['request_hash'] == g['request_hash'] and row['epoch'] == g['epoch'], 'registered READY exact key')
            payload = json.loads(row['payload'])
            need(payload == value['canonical_job_payload'] and payload['argv'] == [UV, '--version'] and
                 payload['source_sha256'] == source_sha and payload['contract_sha256'] == contract_sha and
                 payload['runtime_pin'] == g['runtime_pin'] and payload['source_commit'] == head,
                 'authoritative registered payload differs from exact source/runtime')
            need(lease and lease['status'] == 'LIVE' and lease['owner'] == g['owner'] and lease['fence'] == g['fence'] and
                 lease['expires'] > time.time(), 'fresh canonical lease')
            need(not db.execute("SELECT 1 FROM jobs WHERE state IN ('RESERVED','RUNNING','OUTCOME_UNKNOWN')").fetchone(), 'unresolved global worker')
            charged = db.execute("SELECT epoch,state,reserved_seconds,actual_seconds,finished FROM jobs WHERE reserved IS NOT NULL").fetchall()
            need(len(charged) <= 128, 'finite canonical budget rows')
            costs = [(r, r['reserved_seconds'] if r['actual_seconds'] is None else r['actual_seconds']) for r in charged]
            for r, cost in costs:
                positive(cost, 'canonical finite historical cost', True)
            epoch_cost = sum(cost for r, cost in costs if r['epoch'] == g['epoch'])
            rolling_cost = sum(cost for r, cost in costs if r['state'] in {'RESERVED', 'RUNNING', 'OUTCOME_UNKNOWN'} or
                               (r['finished'] is not None and r['finished'] >= time.time() - 86400))
            need(epoch_cost + 10 <= 7200 and rolling_cost + 10 <= 14400, 'prebirth cumulative native budget')
            need(sum(cost for r, cost in costs if r['actual_seconds'] is not None) >= 47.421,
                 'historical47.421 accounting not retained')
        measured = kernel.telemetry()
        need(measured['host_available_bytes'] >= 3.3125 * GIB and measured['disk_free_bytes'] >= 4 * GIB,
             'preworker host/disk admission')
        need(0 <= self.clock() - value['admission_sampled_monotonic'] <= 5, 'admission freshness')
        self.value = value
        gate = dict(g)
        gate.update(separately_registered=True, source_roles_commit_accepted=True, loader_documentary_accepted=True,
                    trusted_metadata_boundary_accepted=True, physical_paths_verified=True, empty_scoped_paths_verified=True,
                    canonical_operations_sha256=OPS_SHA, tool_path=UV, tool_sha256=UV_SHA, argv=['--version'],
                    telemetry_age_seconds=0, preworker_available_bytes=measured['host_available_bytes'],
                    disk_free_bytes=measured['disk_free_bytes'], host_reserve_bytes=int(1.25 * GIB),
                    gpu_status='UNMEASURED_NOT_QUALIFIED')
        return gate


class ChildPipe:
    def __init__(self, kernel, incoming, outgoing):
        self.k, self.incoming, self.outgoing = kernel.k, incoming, outgoing

    def send(self, raw):
        need(len(raw) <= FRAME_MAX + 4, 'child frame cap')
        buffer = ct.create_string_buffer(raw); count = wt.DWORD()
        need(self.k.WriteFile(self.outgoing, buffer, len(raw), ct.byref(count), None) and count.value == len(raw), 'child frame write')

    def _read(self, n):
        raw = bytearray()
        while len(raw) < n:
            buffer = ct.create_string_buffer(n - len(raw)); count = wt.DWORD()
            need(self.k.ReadFile(self.incoming, buffer, len(buffer), ct.byref(count), None) and count.value,
                 'child inherited channel closed')
            raw.extend(buffer.raw[:count.value])
        return bytes(raw)

    def receive(self):
        head = self._read(4)
        size = struct.unpack('<I', head)[0]
        need(0 < size <= FRAME_MAX, 'child frame cap')
        return head + self._read(size)


class KWriter:
    """Writes may block in a thread; K monitoring/timer never waits on its lock."""
    def __init__(self, kernel, role):
        self.kernel, self.role = kernel, role
        self.queue = queue.Queue(maxsize=8)
        self.errors = []
        self.pending = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while True:
            raw = self.queue.get()
            if raw is None:
                return
            self.pending.set()
            try:
                self.kernel.write(self.role, raw)
            except BaseException as exc:
                self.errors.append(type(exc).__name__ + ': ' + str(exc))
            finally:
                self.pending.clear()

    def send(self, raw):
        self.queue.put_nowait(raw)

    def close(self):
        need(not self.pending.is_set() and self.queue.empty(), 'uncompleted writer retains owner')
        self.queue.put_nowait(None)
        self.thread.join(timeout=0.1)
        need(not self.thread.is_alive(), 'writer completion unknown')


class StoreProxy:
    def __init__(self, rpc):
        self.rpc = rpc
        self.clock = time.time

    def _require_lease(self, owner, fence):
        return self.rpc.call('CHECK', {'owner': owner, 'fence': fence})

    def get_job(self, key):
        return self.rpc.call('ROW', {'key': key})

    def reserve(self, key, owner, fence, seconds):
        return self.rpc.call('RESERVE', {'key': key, 'owner': owner, 'fence': fence, 'seconds': seconds})

    def mark_running(self, key, owner, fence, identity):
        return self.rpc.call('MARK', {'key': key, 'owner': owner, 'fence': fence, 'identity': identity})

    def mark_unknown(self, key, reason):
        return self.rpc.call('UNKNOWN', {'key': key, 'reason': reason})

    def write_receipt(self, kind, key, payload):
        return self.rpc.call('RECEIPT', {'kind': kind, 'key': key, 'payload': payload})


class JobProxy:
    def __init__(self, rpc, guard):
        self.rpc, self.guard, self.handle = rpc, guard, 1
        guard.own_job(self)

    def assign(self, process):
        return self.rpc.call('ASSIGN', {'pid': int(process.dwProcessId)})

    def verify_limits(self):
        return self.rpc.call('LIMITS')

    def pids(self):
        return self.rpc.call('PIDS')

    def kill(self):
        return self.rpc.call('KILL_U')

    def close(self):
        self.handle = None  # C closes its proxy; K retains original inner Job.


class PortProxy:
    def __init__(self, rpc, guard, lib, copy_verifier):
        self.rpc, self.guard, self.lib = rpc, guard, lib
        self.copy_verifier = copy_verifier
        self.process = None
        self.cleanup_exit_proved = self.cleanup_job_empty = False
        self.cleanup_exit_at = None
        guard.own_port(self)
        rpc.call('PORT')
        self.job = JobProxy(rpc, guard)

    def spawn_suspended(self, cwd, env):
        returned = self.rpc.call('BIRTH', {'cwd': cwd, 'env': env})
        copies = returned['copies']
        self.job.handle = copies['inner']
        p = self.lib.ProcessInfo()
        p.hProcess, p.hThread = copies['process'], copies['thread']
        p.dwProcessId, p.dwThreadId = returned['identity']['pid'], returned['thread_id']
        self.process = p
        self.copy_verifier(returned)
        self.rpc.call('ACK', {'copies': copies, 'token_sha256': returned['token_sha256'],
                             'birth_response_sha256': digest(returned)})
        return p

    def identity(self, process):
        return self.rpc.call('IDENTITY', {'pid': int(process.dwProcessId)})

    def resume(self, process):
        return self.rpc.call('RESUME', {'pid': int(process.dwProcessId)})

    def output(self, remaining):
        value = self.rpc.call('OUTPUT', {'remaining': remaining})
        value['chunks'] = [(name, bytes.fromhex(raw)) for name, raw in value['chunks']]
        return value

    def exit_code(self, process):
        return self.rpc.call('EXIT', {'pid': int(process.dwProcessId)})

    def memory(self, process):
        return self.rpc.call('MEMORY', {'pid': int(process.dwProcessId)})

    def terminate_created(self):
        return not self.rpc.call('CLEANUP')['errors']

    def cleanup(self):
        quiescent = self.guard.quiesce()
        if not quiescent:
            return ['C proxy timer unquiescent; K exact originals retained']
        value = self.rpc.call('CLEANUP')
        self.cleanup_exit_proved, self.cleanup_job_empty = value['signal'], value['empty']
        self.cleanup_exit_at = value['at']
        if self.cleanup_exit_proved and self.cleanup_job_empty:
            self.job.close()
        return value['errors']

    def recovery_state(self):
        return self.rpc.call('RECOVERY')


def caller_entry(channel, codec, gate, lib=None, clock=time.monotonic, sleep=time.sleep, guard_factory=None, wall_clock=time.time, copy_verifier=None):
    """The actual C source calls immutable observe_version, never a mock result."""
    lib = lib or library()
    need(callable(copy_verifier), 'concrete received-handle verifier absent')
    rpc = Rpc(channel, codec)
    store = StoreProxy(rpc)
    store.clock = wall_clock
    result = lib.observe_version(gate, store, lambda guard: PortProxy(rpc, guard, lib, copy_verifier),
        lambda: rpc.call('TELEMETRY'), clock=clock, sleep=sleep, guard_factory=guard_factory)
    serial = dict(result)
    owner = serial.pop('recovery_owner', None)
    serial['C_in_memory_guard_retained'] = owner is not None
    # K retains originals independently, even when C cannot serialize its guard.
    leaf = durable_leaf(gate['leaf_dir'], 'caller.json', {'binding': codec.binding, 'C_identity': codec.identity, 'result': serial})
    rpc.call('CALLER_DONE', {'leaf': leaf})
    return serial


class CanonicalDelegate:
    def __init__(self, store, gate):
        self.store, self.gate = store, gate

    def check(self, states=None, identity=None):
        g = self.gate
        self.store._require_lease(g['owner'], g['fence'])
        row = self.store.get_job(g['key'])
        need(row['request_hash'] == g['request_hash'] and row['epoch'] == g['epoch'], 'canonical request/epoch drift')
        if row['state'] != 'READY':
            need(row['owner'] == g['owner'] and row['fence'] == g['fence'] and row['reserved_seconds'] == 10, 'canonical reservation binding')
        if states:
            need(row['state'] in states, 'canonical state not permitted')
        if identity is not None:
            need(row['worker'] == identity, 'canonical held-worker identity')
        return row

    def dispatch(self, op, p):
        g = self.gate
        need(op in STORE_OPERATIONS, 'closed store API')
        if 'key' in p:
            need(p['key'] == g['key'], 'RPC key mismatch')
        if 'owner' in p:
            need(p['owner'] == g['owner'] and p['fence'] == g['fence'], 'RPC owner/fence mismatch')
        self.check()
        if op == 'CHECK':
            lease = self.store._require_lease(g['owner'], g['fence'])
            value = {'lease': 'LIVE', 'owner': g['owner'], 'fence': g['fence'],
                     'expires': lease['expires'], 'row': self.check()}
        elif op == 'ROW':
            value = self.check()
        elif op == 'RESERVE':
            need(p['seconds'] == 10, 'fixed reservation')
            value = self.store.reserve(g['key'], g['owner'], g['fence'], 10)
        elif op == 'MARK':
            identity = p['identity']
            row = self.check({'RESERVED', 'RUNNING'})
            if row['state'] == 'RUNNING':
                need(row['worker'] == identity, 'idempotent worker drift')
            else:
                try:
                    self.store.mark_running(g['key'], g['owner'], g['fence'], identity)
                except BaseException:
                    # Only the same created identity may be recovered after COMMIT.
                    row = self.check({'RESERVED', 'RUNNING'})
                    if row['state'] != 'RUNNING' or row['worker'] != identity:
                        raise
            value = self.check({'RUNNING'}, identity)
        elif op == 'UNKNOWN':
            row = self.check()
            need(row['worker'] is not None or p.get('creation_attempted') is True,
                 'pristine RESERVED UNKNOWN refused; actual state retained')
            self.store.mark_unknown(g['key'], p['reason'])
            value = self.check()
        else:
            need(p['kind'] == 'native_outcome_observation', 'library receipt kind')
            value = self.store.write_receipt(p['kind'], g['key'], p['payload'])
        # Renew on EVERY return, including RUNNING fast and response-loss reread.
        self.check()
        return value

    def finalize(self, caller, witness):
        g = self.gate
        result = caller['result']
        identity = witness['U_identity']
        self.check({'OUTCOME_UNKNOWN', 'RUNNING'}, identity)
        ref = result['outcome_observation_ref']
        receipt = self.store.verify_receipt(ref)
        proof = receipt['payload']
        native = positive(result['native_elapsed_seconds'], 'unknown native interval')
        need(receipt['kind'] == 'native_outcome_observation' and receipt['key'] == g['key'] and
             proof['worker_identity'] == identity and proof['request_hash'] == g['request_hash'] and
             proof['elapsed_native_seconds'] == native and proof['process_exited'] is True and
             proof['never_dispatched'] is False and witness['U_signal'] and witness['inner_empty'] and
             witness['U_EOF'] == {'stdout': True, 'stderr': True}, 'native immutable observation mismatch')
        need(result['native_interval_ended'] - result['native_interval_started'] == native and
             result['native_interval_started'] <= witness['U_born_at'] <= result['native_interval_ended'], 'exact native interval/birth')
        okay = (result['status'] == 'METADATA_OBSERVED_PENDING_EXTERNAL_FINALIZATION' and
                not result['cleanup_errors'] if 'cleanup_errors' in result else
                result['status'] == 'METADATA_OBSERVED_PENDING_EXTERNAL_FINALIZATION')
        outcome = 'SUCCEEDED' if okay and witness['U_exit_code'] == 0 else 'FAILED'
        evidence = {'native_outcome_proof_ref': ref, 'caller_leaf_ref': witness['caller_leaf'],
                    'K_TC_witness_sha256': digest(witness), 'native_success_is_not_whole_acceptance': True}
        try:
            self.store.reconcile(g['key'], outcome, native, evidence, owner=g['owner'], fence=g['fence'])
        except BaseException:
            row = self.check()
            expected = {'evidence': evidence, 'outcome': outcome, 'actual_native_seconds': native,
                        'reservation_exceeded': native > 10}
            need(row['state'] == outcome and row['worker'] == identity and row['result'] == expected,
                 'canonical reconciliation response unknown; no retry/reset')
        row = self.check({outcome}, identity)
        leaf = durable_leaf(g['leaf_dir'], 'observer.json', {'binding': witness['binding'],
            'native_ref': ref, 'canonical_state': row['state'], 'canonical_result': row['result'],
            'caller_leaf': witness['caller_leaf'], 'TC_witness_sha256': digest(witness)})
        self.check({outcome}, identity)
        return {'leaf': leaf, 'native_state': row['state'], 'native_seconds': native}


def observer_entry(channel, codec, gate, store=None):
    """O canonical server then one retrospective native/observer publication."""
    own = store is None
    store = store or operations().OperationsStore(gate['db_path'], gate['receipt_dir'])
    delegate = CanonicalDelegate(store, gate)
    try:
        while True:
            frame = codec.decode(channel.receive())
            op, p = frame['operation'], frame['payload']
            if op == 'STOP':
                return
            try:
                if op == 'FINALIZE':
                    caller = read_leaf(p['caller_leaf'], gate['leaf_dir'])
                    value = delegate.finalize(caller, p['witness'])
                else:
                    value = delegate.dispatch(op, p)
                channel.send(codec.encode(op, {'value': value}))
                if op == 'FINALIZE':
                    return
            except BaseException as exc:
                channel.send(codec.encode(op, {'error': type(exc).__name__ + ': ' + str(exc)}))
                if op == 'FINALIZE':
                    return
    finally:
        if own:
            store.close()


class Integration:
    """One complete K orchestration; no birth is possible before admission."""
    def __init__(self, kernel, admission, clock=time.monotonic, sleep=time.sleep, phase=None, initial_cpu=None):
        self.kernel, self.admission, self.clock, self.sleep = kernel, admission, clock, sleep
        self.gate = None
        self.codecs, self.writers, self.buffers = {}, {}, {}
        self.identities, self.eof = {}, {}
        self.pending = None
        self.C_request = None
        self.copy_reply = None
        self.acknowledged = False
        self.caller_ref = self.observer_ref = None
        self.caller_result = self.observer_result = None
        self.native_terminal = None
        self.U_output = {'stdout': bytearray(), 'stderr': bytearray()}
        self.U_EOF = {'stdout': False, 'stderr': False}
        self.U_born_at = None
        self.failures, self.samples = [], []
        self.caller_phase, self.observer_phase = phase, None
        self.initial_cpu = initial_cpu
        self.cpu_origin = self.cpu_at_TC = None
        self.binding = None
        self.TC_witness = None
        self.completed = False

    def boot(self, role):
        identity = self.kernel.create(role, self.gate)
        expected = PYTHON.casefold()
        need(identity['image'].casefold() == expected, 'created role executable drift')
        self.identities[role] = identity
        codec = self.codecs[role] = Codec(secrets.token_hex(32), role, self.binding, identity)
        self.buffers[role] = bytearray()
        self.eof[role] = {'stdout': False, 'stderr': False}
        self.writers[role] = KWriter(self.kernel, role)
        bootstrap = {'role': role, 'nonce': codec.nonce, 'binding': self.binding,
                     'identity': identity, 'K_identity': self.kernel.parent_identity(), 'gate': self.gate}
        raw = canonical(bootstrap)
        need(len(raw) <= FRAME_MAX, 'bootstrap cap')
        self.writers[role].send(struct.pack('<I', len(raw)) + raw)
        self.kernel.resume(role)

    def send(self, role, op, payload):
        self.writers[role].send(self.codecs[role].encode(op, payload))

    def o_call(self, op, payload, continuation):
        need(self.pending is None, 'one outstanding O transaction')
        self.pending = (op, continuation)
        self.send('O', op, payload)

    def renew(self, continuation, states=None, identity=False):
        def done(value):
            g = self.gate
            need(value['owner'] == g['owner'] and value['fence'] == g['fence'] and
                 value['expires'] > self.wall(), 'live canonical lease at return')
            row = value['row']
            need(row['request_hash'] == g['request_hash'] and row['epoch'] == g['epoch'], 'request/epoch at return')
            if row['state'] != 'READY':
                need(row['owner'] == g['owner'] and row['fence'] == g['fence'] and row['reserved_seconds'] == 10,
                     'reservation at return')
            if states:
                need(row['state'] in states, 'canonical state at return')
            if identity:
                actual = self.kernel.identity('U')
                need(actual == self.identities['U'] and row['worker'] == actual, 'held/canonical identity at return')
            self.phase().check()
            continuation(row)
        self.o_call('CHECK', {'owner': self.gate['owner'], 'fence': self.gate['fence']}, done)

    def wall(self):
        return self.kernel.wall() if hasattr(self.kernel, 'wall') else time.time()

    def phase(self):
        return self.observer_phase or self.caller_phase

    def c_reply(self, value, states=None, identity=None):
        operation = self.C_request['operation']
        if identity is None:
            identity = 'U' in self.identities
        def complete(row):
            self.send('C', operation, {'value': value})
            self.C_request = None
        self.renew(complete, states, identity)

    def c_error(self, operation, exc):
        self.send('C', operation, {'error': type(exc).__name__ + ': ' + str(exc)})
        self.C_request = None

    def resource_sample(self):
        roles = ['K'] + [r for r in ('O', 'C', 'U') if r in self.kernel.held]
        samples = {r: self.kernel.snapshot(r) for r in roles}
        for role, sample in samples.items():
            positive(sample['CPU_seconds'], 'CPU sample unavailable', True)
            need(0 <= sample['rss_bytes'] <= sample['peak_rss_bytes'], 'RSS sample invalid')
            if role != 'K' and role in self.identities:
                need(self.kernel.identity(role) == self.identities[role], 'held graph identity drift')
        peak = sum(x['peak_rss_bytes'] for x in samples.values())
        need(peak <= 2 * GIB, 'combined K/O/C/U conservative peak RSS excess')
        outer = self.kernel.job_limits(3)
        inner = self.kernel.job_limits(1)
        commit = samples['K']['peak_private_commit_bytes'] + outer['peak_commit']
        need(outer['peak_commit'] <= 2 * GIB and commit <= 2 * GIB, 'separate Job/aggregate K commit excess')
        members = self.kernel.members(3)
        expected = {self.identities[r]['pid'] for r in self.identities}
        need(set(members) <= expected, 'outer unexpected/unidentified membership')
        if 'U' in self.identities:
            need(self.kernel.members(1) in ([], [self.identities['U']['pid']]), 'inner U descendant')
        telemetry = self.kernel.telemetry()
        need(0 <= telemetry['age_seconds'] <= 5 and telemetry['host_available_bytes'] >= 1.25 * GIB and
             telemetry['disk_free_bytes'] >= 4 * GIB, 'current host/disk freshness')
        sample = {'at': self.clock(), 'roles': samples, 'outer': outer, 'inner': inner,
                  'peak_RSS_sum': peak, 'aggregate_commit': commit, 'telemetry': telemetry}
        self.samples.append(sample)
        need(len(self.samples) <= 256, 'complete resource journal cap')
        return sample

    def cpu_sum(self, sample):
        return sum(x['CPU_seconds'] for x in sample['roles'].values())

    def dispatch_c(self, frame):
        need(self.C_request is None and self.pending is None, 'overlapping C RPC')
        self.C_request = frame
        op, p = frame['operation'], frame['payload']
        self.caller_phase.check()
        if op in STORE_OPERATIONS:
            if op == 'MARK':
                need(p['identity'] == self.identities.get('U') == self.kernel.identity('U'), 'C mark actual created identity')
            if op == 'UNKNOWN':
                p = dict(p, creation_attempted='U' in self.kernel.birth_attempts)
            self.o_call(op, p, lambda value: self.c_reply(value))
        elif op == 'BIRTH':
            env = {'SystemRoot': 'C:/Windows', 'WINDIR': 'C:/Windows', 'PATH': 'C:/Windows/System32',
                   'TEMP': self.gate['temp'], 'TMP': self.gate['temp']}
            need(p == {'cwd': self.gate['cwd'], 'env': env}, 'exact immutable caller creation arguments')
            def create(row):
                need(row['worker'] is None, 'RESERVED birth cannot use a future worker')
                self.U_born_at = self.clock()
                identity = self.kernel.create('U', self.gate)
                self.identities['U'] = identity
                need(identity['image'].casefold() == UV.casefold(), 'actual U image drift')
                def marked(value):
                    def copied(row):
                        copies = self.kernel.copies_for_c()
                        token = {'copies': copies, 'C_identity': self.identities['C'], 'U_identity': identity,
                                 'nonce': self.codecs['C'].nonce, 'binding': self.binding}
                        info = self.kernel.held['U']
                        self.copy_reply = {'copies': copies, 'identity': identity,
                                           'thread_id': int(info.dwThreadId), 'token_sha256': digest(token)}
                        self.c_reply(self.copy_reply, {'RUNNING'}, True)
                    self.renew(copied, {'RUNNING'}, True)
                self.o_call('MARK', {'key': self.gate['key'], 'owner': self.gate['owner'],
                    'fence': self.gate['fence'], 'identity': identity}, marked)
            self.renew(create, {'RESERVED'}, False)
        elif op == 'ACK':
            need(self.copy_reply and p == {'copies': self.copy_reply['copies'],
                 'token_sha256': self.copy_reply['token_sha256'], 'birth_response_sha256': digest(self.copy_reply)},
                 'exact C channel/token/frame ACK mismatch')
            # frame was decoded on the actual C nonce/creation-identity channel.
            self.acknowledged = True
            self.c_reply({'acknowledged': digest(frame)}, {'RUNNING'}, True)
        elif op == 'RESUME':
            need(self.acknowledged and p['pid'] == self.identities['U']['pid'], 'resume ACK/PID')
            def resume(row):
                self.kernel.resume('U')
                self.c_reply({'resumed': True}, {'RUNNING'}, True)
            self.renew(resume, {'RUNNING'}, True)
        elif op == 'ASSIGN':
            need(p['pid'] == self.identities['U']['pid'], 'assign exact U')
            self.kernel.in_job('U', 1)
            self.kernel.in_job('U', 3)
            # Already born in both Jobs; no post-birth assignment call exists.
            self.c_reply({'birth_membership_verified': True}, {'RUNNING'}, True)
        elif op == 'IDENTITY':
            need(p['pid'] == self.identities['U']['pid'], 'identity exact U')
            self.c_reply(self.kernel.identity('U'))
        elif op == 'LIMITS':
            self.c_reply(self.kernel.job_limits(1))
        elif op == 'PIDS':
            self.c_reply(self.kernel.members(1))
        elif op in {'MEMORY', 'EXIT'}:
            need(p['pid'] == self.identities['U']['pid'], 'held exact U token')
            sample = self.kernel.snapshot('U')
            if op == 'EXIT':
                value = sample['exit_code'] if sample['signal'] else None
            else:
                value = {'rss_bytes': sample['rss_bytes'], 'peak_rss_bytes': sample['peak_rss_bytes'],
                         'source': 'K original GetProcessMemoryInfo', 'measured_at': sample['sampled_at']}
            self.c_reply(value)
        elif op == 'OUTPUT':
            remaining = p['remaining']
            need(type(remaining) is int and 0 <= remaining <= 8192, 'library combined output remaining')
            chunks = []
            frame_budget = 1536  # Hex + full binding/identity stays below8192.
            for name in ('stdout', 'stderr'):
                raw, eof = self.kernel.read('U', name, min(remaining + 1, frame_budget))
                self.U_output[name].extend(raw)
                self.U_EOF[name] = eof
                if raw:
                    chunks.append([name, raw.hex()]); remaining -= len(raw); frame_budget -= len(raw)
                    if remaining < 0:
                        break
            need(sum(len(x) for x in self.U_output.values()) <= 8193, 'single overflow sentinel cap')
            self.c_reply({'chunks': chunks, 'EOF': dict(self.U_EOF)})
        elif op == 'TELEMETRY':
            sample = self.resource_sample()
            self.c_reply(sample['telemetry'])
        elif op == 'PORT':
            self.c_reply({'K_owns_originals': True}, {'RESERVED'}, False)
        elif op == 'KILL_U':
            self.kernel.kill_u()
            self.c_reply({'owned_request': True})
        elif op == 'CLEANUP':
            errors = self.kernel.cleanup_u()
            signal = 'U' in self.kernel.held and self.kernel.snapshot('U')['signal']
            empty = self.kernel.members(1) == []
            self.c_reply({'errors': errors, 'signal': signal, 'empty': empty, 'at': self.clock()})
        elif op == 'RECOVERY':
            signal = 'U' in self.kernel.held and self.kernel.snapshot('U')['signal']
            empty = self.kernel.members(1) == []
            self.c_reply({'process_handle': None if signal and empty else 'K_HELD_ORIGINAL',
                'Job_handle': None if signal and empty else 'K_HELD_ORIGINAL_INNER',
                'other_handles': {}, 'attribute_retained': bool(self.kernel.pending_attributes),
                'exact_process_signal_proved': signal, 'owned_Job_empty_proved': empty,
                'timer_quiescent_at_cleanup': True, 'K_original_owner_reachable': True,
                'in_memory_only_not_durable_or_transferred': True})
        elif op == 'CALLER_DONE':
            need(self.caller_ref is None, 'one caller leaf')
            caller = read_leaf(p['leaf'], self.gate['leaf_dir'])
            need(caller['binding'] == self.binding and caller['C_identity'] == self.identities['C'], 'caller leaf binding')
            self.caller_ref, self.caller_result = p['leaf'], caller
            self.c_reply({'leaf_verified': True})
        else:
            raise Refused('operation invalid on C channel')

    def read_roles(self):
        for role in ('O', 'C'):
            raw, eof = self.kernel.read(role, 'stdout', FRAME_MAX + 4)
            self.eof[role]['stdout'] = eof
            err, err_eof = self.kernel.read(role, 'stderr', 8193)
            self.eof[role]['stderr'] = err_eof
            need(not err, 'unexpected ' + role + ' stderr')
            self.buffers[role].extend(raw)
            need(len(self.buffers[role]) <= 2 * (FRAME_MAX + 4), 'role receive buffer cap')
            if len(self.buffers[role]) < 4:
                continue
            size = struct.unpack('<I', self.buffers[role][:4])[0]
            need(0 < size <= FRAME_MAX, 'role frame count')
            if len(self.buffers[role]) < size + 4:
                continue
            frame_bytes = bytes(self.buffers[role][:size + 4])
            del self.buffers[role][:size + 4]
            frame = self.codecs[role].decode(frame_bytes)
            if role == 'C':
                try:
                    self.dispatch_c(frame)
                except BaseException as exc:
                    self.c_error(frame['operation'], exc)
            else:
                need(self.pending and frame['operation'] == self.pending[0], 'O response sequence/operation')
                op, continuation = self.pending
                self.pending = None
                if 'error' in frame['payload']:
                    if self.C_request:
                        self.c_error(self.C_request['operation'], Refused(frame['payload']['error']))
                    else:
                        raise Refused(frame['payload']['error'])
                else:
                    continuation(frame['payload']['value'])

    def witness_caller(self):
        need(self.caller_ref and self.pending is None and self.C_request is None, 'completed caller leaf/RPC')
        sample = self.resource_sample()
        c = sample['roles']['C']; u = sample['roles'].get('U')
        need(c['signal'] and c['exit_code'] == 0 and self.eof['C'] == {'stdout': True, 'stderr': True} and
             not self.buffers['C'], 'actual C exit/closed streams')
        need(u and u['signal'] and self.kernel.members(1) == [] and self.U_EOF == {'stdout': True, 'stderr': True}, 'complete U exit/EOF/empty')
        need(self.kernel.identity('U') == self.identities['U'], 'post-exit U identity')
        result = self.caller_result['result']
        need(result['worker_identity'] == self.identities['U'] and result['worker_exit_proved'] and
             result['stream_EOF'] == self.U_EOF and result['raw_stdout_hex'] == self.U_output['stdout'].hex() and
             result['raw_stderr_hex'] == self.U_output['stderr'].hex() and result['timer_quiescent_at_return'], 'caller/native exact facts')
        self.caller_phase.check()
        TC = self.clock()
        self.cpu_at_TC = self.cpu_sum(sample)
        need(0 <= self.cpu_at_TC - self.cpu_origin <= 10, 'caller K/O/C/U CPU excess')
        self.TC_witness = {'binding': self.binding, 'T0': self.caller_phase.started, 'TC': TC,
            'caller_leaf': self.caller_ref, 'C_identity': self.identities['C'], 'C_signal': True,
            'C_EOF': dict(self.eof['C']), 'U_identity': self.identities['U'], 'U_signal': u['signal'],
            'U_exit_code': u['exit_code'], 'U_EOF': dict(self.U_EOF), 'inner_empty': True,
            'U_born_at': self.U_born_at, 'post_exit_resources': sample,
            'caller_phase_CPU_seconds': self.cpu_at_TC - self.cpu_origin}
        need(self.caller_phase.close(), 'caller K timer not quiescent')
        self.observer_phase = PhaseGuard(self.kernel, self.clock, 2, 'observer')
        # Start immediately at TC, not after queueing finalizer or validation.
        self.observer_phase.started = TC
        self.observer_phase.until = TC + 2
        def finalized(value):
            self.observer_ref = value['leaf']
            self.observer_result = read_leaf(value['leaf'], self.gate['leaf_dir'])
            self.native_terminal = value['native_state']
            need(value['native_seconds'] == result['native_elapsed_seconds'], 'native cost finalizer mismatch')
        self.o_call('FINALIZE', {'caller_leaf': self.caller_ref, 'witness': self.TC_witness}, finalized)

    def complete_whole(self):
        sample = self.resource_sample()
        need(self.observer_ref and self.pending is None and sample['roles']['O']['signal'] and
             sample['roles']['O']['exit_code'] == 0 and self.eof['O'] == {'stdout': True, 'stderr': True} and
             not self.buffers['O'] and self.kernel.members(3) == [], 'actual O/graph completion')
        need(self.observer_result['TC_witness_sha256'] == digest(self.TC_witness) and
             self.observer_result['caller_leaf'] == self.caller_ref and
             self.observer_result['binding'] == self.binding, 'observer leaf exact TC/binding')
        self.observer_phase.check()
        payload = {'binding': self.binding, 'TC_witness': self.TC_witness,
            'caller_leaf': self.caller_ref, 'observer_leaf': self.observer_ref,
            'native_state': self.native_terminal, 'native_seconds': self.caller_result['result']['native_elapsed_seconds'],
            'O_identity': self.identities['O'], 'O_signal': True, 'O_EOF': dict(self.eof['O']),
            'post_exit_resources': sample, 'status': 'COMPLETED_LEAF_AWAITING_EXTERNAL_RETURN_WITNESS',
            'OE_or_final_cost_not_predicted_in_durable_leaf': True}
        whole_ref = self.kernel.publish_whole(self.gate['leaf_dir'], 'whole.json', payload, self.observer_phase)
        for writer in self.writers.values():
            writer.close()
        self.kernel.release()
        # Serialize the completed body BEFORE the endpoint. The body intentionally
        # has no future OE/cost claim; the selected return interface pairs it
        # with a separately sampled completed-return witness below.
        body = {'status': 'COMPLETED_BODY_REQUIRES_RETURN_WITNESS', 'whole_leaf': whole_ref,
                'TC': self.TC_witness['TC'], 'native_state': self.native_terminal,
                'native_seconds': payload['native_seconds'], 'binding': self.binding}
        serialized = canonical(body).decode('utf-8')
        need(self.observer_phase.close(), 'observer K timer unquiescent')
        final_K = self.kernel.snapshot('K')
        observer_cpu = (final_K['CPU_seconds'] + sum(sample['roles'][r]['CPU_seconds'] for r in ('O', 'C', 'U'))) - self.cpu_at_TC
        self.observer_phase.check()
        need(0 <= observer_cpu <= 2, 'whole observer CPU excess')
        need(final_K['peak_rss_bytes'] + sum(sample['roles'][r]['peak_rss_bytes'] for r in ('O', 'C', 'U')) <= 2 * GIB and
             final_K['peak_private_commit_bytes'] + sample['outer']['peak_commit'] <= 2 * GIB, 'post-publication K resource excess')
        # Selected concrete interface is this post-publication RETURN witness.
        # It binds immutable completed leaves; no recursive/future-cost receipt.
        result = {'status': 'ACCEPTED_SOURCE_PATH_SYNTHETIC' if not self.kernel.actual else 'ACCEPTED_WHOLE_OBSERVATION',
            'whole_leaf': whole_ref, 'TC': self.TC_witness['TC'], 'OE': None,
            'caller_wall_seconds': self.TC_witness['TC'] - self.caller_phase.started,
            'caller_CPU_seconds': self.cpu_at_TC - self.cpu_origin,
            'observer_wall_seconds': None, 'observer_CPU_seconds': observer_cpu,
            'native_state': self.native_terminal, 'native_seconds': payload['native_seconds'],
            'actual_qualification_from_fixture': False, 'ownership_released_after_signal_empty_EOF': True,
            'serialized_completed_body': serialized,
            'completion_interface': 'prebuilt completed body paired with post-quiescence endpoint fields in returned object; later CLI sink is not this witness'}
        OE = self.clock()
        need(not self.observer_phase.expired.is_set() and OE <= self.observer_phase.until, 'completed return deadline')
        result['OE'] = OE
        result['observer_wall_seconds'] = OE - self.TC_witness['TC']
        self.completed = True
        return result

    def run(self):
        self.caller_phase = self.caller_phase or PhaseGuard(self.kernel, self.clock, 10, 'caller')
        try:
            self.gate = self.admission.verify(self.kernel)
            self.caller_phase.check()
            self.binding = {k: self.gate[k] for k in ('key', 'request_hash', 'epoch', 'owner', 'fence')}
            self.cpu_origin = self.initial_cpu if self.initial_cpu is not None else self.kernel.snapshot('K')['CPU_seconds']
            self.kernel.new_job(3)
            self.kernel.new_job(1)
            self.boot('O'); self.boot('C')
            last_sample = self.clock()
            while True:
                self.phase().check()
                self.read_roles()
                if self.clock() - last_sample >= 0.1:
                    sample = self.resource_sample(); last_sample = self.clock()
                    current_cpu = self.cpu_sum(sample)
                    need(current_cpu - (self.cpu_at_TC if self.observer_phase else self.cpu_origin) <=
                         (2 if self.observer_phase else 10), 'independent phase CPU stop')
                for role in ('O', 'C'):
                    state = self.kernel.snapshot(role)
                    if state['signal'] and not (self.caller_ref if role == 'C' else self.observer_ref):
                        # Drain buffered final reply first; otherwise loss stops graph.
                        if self.eof[role]['stdout'] and not self.buffers[role]:
                            raise Refused(role + ' lost before completed leaf')
                if (not self.observer_phase and self.caller_ref and self.kernel.snapshot('C')['signal'] and
                        self.eof['C'] == {'stdout': True, 'stderr': True}):
                    self.witness_caller()
                if (self.observer_phase and self.observer_ref and self.kernel.snapshot('O')['signal'] and
                        self.eof['O'] == {'stdout': True, 'stderr': True}):
                    return self.complete_whole()
                self.sleep(0.001)
        except BaseException as exc:
            self.failures.append(type(exc).__name__ + ': ' + str(exc))
            self.kernel.stop_outer_once()
            # Never rewrite immutable native terminal outcome after whole failure.
            return {'status': 'LATE_OR_UNKNOWN_UNACCEPTED', 'errors': self.failures,
                'caller_wall_seconds': self.clock() - self.caller_phase.started,
                'observer_wall_seconds': self.clock() - self.observer_phase.started if self.observer_phase else None,
                'native_terminal_already_published': self.native_terminal,
                'full_reservation_required_if_native_unresolved': self.native_terminal is None,
                'retry_allowed': False, 'recovery_owner': self,
                'K_crash_durable_recovery_claimed': False}
        finally:
            if not self.caller_phase.closed:
                self.caller_phase.close()
            if self.observer_phase and not self.observer_phase.closed:
                self.observer_phase.close()


def role_main(role, incoming, outgoing, parent_handle):
    kernel = Win32Kernel()
    pipe = ChildPipe(kernel, incoming, outgoing)
    raw = pipe.receive()
    bootstrap = json.loads(raw[4:])
    need(canonical(bootstrap) == raw[4:] and bootstrap['role'] == role, 'canonical role bootstrap')
    identity = bootstrap['identity']
    # Check actual process PID/creation/image through own current process handle.
    info = kernel.lib.ProcessInfo()
    info.hProcess = kernel.khandle
    info.dwProcessId = kernel.k.GetCurrentProcessId()
    kernel.held[role] = info
    need(kernel.identity(role) == identity, 'bootstrap actual role identity mismatch')
    kernel.verify_parent(parent_handle, bootstrap['K_identity'])
    codec = Codec(bootstrap['nonce'], role, bootstrap['binding'], identity)
    if role == 'C':
        caller_entry(pipe, codec, bootstrap['gate'], copy_verifier=kernel.verify_copies)
    else:
        observer_entry(pipe, codec, bootstrap['gate'])


class ConstructionCustody:
    def __init__(self):
        self.kernel = None
    def stop_outer_once(self):
        if self.kernel is not None and getattr(self.kernel, 'outer', None):
            self.kernel.stop_outer_once()


def actual_dispatch(capsule, grant, grant_sha):
    custody = ConstructionCustody()
    # T0 and CPU begin BEFORE runtime source import/API/path/qualification setup.
    phase = PhaseGuard(custody, time.monotonic, 10, 'caller')
    initial_cpu = time.process_time()
    try:
        kernel = Win32Kernel(custody=custody)
        phase.check()
        return Integration(kernel, ActualAdmission(capsule, grant, grant_sha),
                           phase=phase, initial_cpu=initial_cpu).run()
    except BaseException as exc:
        custody.stop_outer_once()
        return {'status': 'LATE_OR_UNKNOWN_UNACCEPTED', 'errors': [type(exc).__name__ + ': ' + str(exc)],
                'caller_wall_seconds': time.monotonic() - phase.started,
                'caller_CPU_seconds': time.process_time() - initial_cpu,
                'retry_allowed': False, 'recovery_owner': custody}
    finally:
        if not phase.closed:
            phase.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('C', 'O'))
    parser.add_argument('--input-handle', type=int)
    parser.add_argument('--output-handle', type=int)
    parser.add_argument('--parent-handle', type=int)
    parser.add_argument('--capsule')
    parser.add_argument('--grant')
    parser.add_argument('--grant-sha256')
    args = parser.parse_args()
    if args.role:
        need(args.input_handle and args.output_handle and args.parent_handle and not args.capsule and not args.grant,
             'role exact inherited entry')
        role_main(args.role, args.input_handle, args.output_handle, args.parent_handle)
        return 0
    need(args.capsule and args.grant and args.grant_sha256, 'actual capsule/grant absent; NOT_READY before birth')
    result = actual_dispatch(args.capsule, args.grant, args.grant_sha256)
    result.pop('recovery_owner', None)
    # Console serialization/pipe drain is a distinct sink and is not claimed as
    # the selected completed-return witness. No printed pre-OE ACCEPTED receipt.
    print(result['serialized_completed_body'] if 'serialized_completed_body' in result else canonical(result).decode())
    return 0 if result['status'] == 'ACCEPTED_WHOLE_OBSERVATION' else 2


if __name__ == '__main__':
    raise SystemExit(main())
