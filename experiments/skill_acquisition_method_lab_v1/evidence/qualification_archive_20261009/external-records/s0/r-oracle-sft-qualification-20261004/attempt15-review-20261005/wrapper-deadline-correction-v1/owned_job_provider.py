"""Review candidate: exact retained-handle Windows Job containment.

No DLL, OS API, process, timer or file is opened at import. The real backend is
created only by an explicitly admitted launcher. This module grants no scope.
"""
from __future__ import annotations

import ctypes as c
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

KILL_ON_JOB_CLOSE = 0x2000
CREATE_SUSPENDED = 0x4
CREATE_NO_WINDOW = 0x08000000
EXTENDED_STARTUPINFO_PRESENT = 0x00080000
CREATE_UNICODE_ENVIRONMENT = 0x400
JOB_LIST_ATTRIBUTE = 0x0002000D
HANDLE_LIST_ATTRIBUTE = 0x00020002
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
TERMINATED = 124
MAX_POLL_COUNT = 200000
MAX_LOG_OPERATIONS = 65536


def require(condition, message):
    if condition is not True:
        raise RuntimeError(message)


class SecurityAttributes(c.Structure):
    _fields_ = [('length', c.c_uint32), ('descriptor', c.c_void_p), ('inherit', c.c_int)]


class FileTime(c.Structure):
    _fields_ = [('low', c.c_uint32), ('high', c.c_uint32)]


class IOCounters(c.Structure):
    _fields_ = [(name, c.c_uint64) for name in (
        'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
        'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class BasicLimits(c.Structure):
    _fields_ = [('per_process_user', c.c_int64), ('per_job_user', c.c_int64),
               ('flags', c.c_uint32), ('min_working_set', c.c_size_t),
               ('max_working_set', c.c_size_t), ('active_limit', c.c_uint32),
               ('affinity', c.c_size_t), ('priority', c.c_uint32), ('scheduling', c.c_uint32)]


class ExtendedLimits(c.Structure):
    _fields_ = [('basic', BasicLimits), ('reserved_io', IOCounters),
               ('process_memory', c.c_size_t), ('job_memory', c.c_size_t),
               ('peak_process_memory', c.c_size_t), ('peak_job_memory', c.c_size_t)]


class BasicAccounting(c.Structure):
    _fields_ = [('user', c.c_int64), ('kernel', c.c_int64),
               ('period_user', c.c_int64), ('period_kernel', c.c_int64),
               ('page_faults', c.c_uint32), ('total', c.c_uint32),
               ('active', c.c_uint32), ('terminated', c.c_uint32)]


class JobAccounting(c.Structure):
    _fields_ = [('basic', BasicAccounting), ('io', IOCounters)]


class StartupInfo(c.Structure):
    _fields_ = [('cb', c.c_uint32), ('reserved', c.c_wchar_p), ('desktop', c.c_wchar_p),
               ('title', c.c_wchar_p), ('x', c.c_uint32), ('y', c.c_uint32),
               ('x_size', c.c_uint32), ('y_size', c.c_uint32),
               ('x_chars', c.c_uint32), ('y_chars', c.c_uint32), ('fill', c.c_uint32),
               ('flags', c.c_uint32), ('show', c.c_uint16), ('reserved_size', c.c_uint16),
               ('reserved_bytes', c.c_void_p), ('stdin', c.c_void_p),
               ('stdout', c.c_void_p), ('stderr', c.c_void_p)]


class StartupInfoEx(c.Structure):
    _fields_ = [('startup', StartupInfo), ('attributes', c.c_void_p)]


class ProcessInformation(c.Structure):
    _fields_ = [('process', c.c_void_p), ('thread', c.c_void_p),
               ('pid', c.c_uint32), ('tid', c.c_uint32)]


def counter_dict(value):
    return {name: int(getattr(value, name)) for name, _ in IOCounters._fields_}


def filetime(value):
    return int(value.low) | int(value.high) << 32


class Win32Backend:
    """Actual APIs only behind runtime construction; never reopen a PID.

    JOB_LIST assigns the suspended process at birth. This is required because
    a separate post-CreateProcess assignment leaves an orphan race if the
    outer watchdog ends the wrapper during that interval. No permissive
    fallback exists. Windows 10+ and successful nested Job support are gates.
    """
    def __init__(self, *, clock=time.monotonic):
        require(os.name == 'nt', 'Windows native backend only')
        self.clock = clock
        self.calls = 0
        self.counter_lock = threading.Lock()
        self.closed = False
        self.job = self.process = self.thread = None
        self.pi = None
        self.identity = None
        self.abort_evidence = None
        self.attribute_buffer = self.attribute_pointer = None
        self.attributes_initialized = False
        self.attribute_arrays = None
        self.read_fds = []
        self.write_fds = []
        self.stdin_fd = None
        self.kernel = c.WinDLL('kernel32', use_last_error=True)
        self._declare()

    def _declare(self):
        k = self.kernel
        signatures = {
            'CreateJobObjectW': ([c.c_void_p, c.c_wchar_p], c.c_void_p),
            'SetInformationJobObject': ([c.c_void_p, c.c_int, c.c_void_p, c.c_uint32], c.c_int),
            'QueryInformationJobObject': ([c.c_void_p, c.c_int, c.c_void_p, c.c_uint32, c.c_void_p], c.c_int),
            'TerminateJobObject': ([c.c_void_p, c.c_uint32], c.c_int),
            'IsProcessInJob': ([c.c_void_p, c.c_void_p, c.POINTER(c.c_int)], c.c_int),
            'InitializeProcThreadAttributeList': ([c.c_void_p, c.c_uint32, c.c_uint32, c.POINTER(c.c_size_t)], c.c_int),
            'UpdateProcThreadAttribute': ([c.c_void_p, c.c_uint32, c.c_size_t, c.c_void_p, c.c_size_t, c.c_void_p, c.c_void_p], c.c_int),
            'DeleteProcThreadAttributeList': ([c.c_void_p], None),
            'CreateProcessW': ([c.c_wchar_p, c.c_wchar_p, c.c_void_p, c.c_void_p, c.c_int,
                                c.c_uint32, c.c_void_p, c.c_wchar_p, c.c_void_p,
                                c.POINTER(ProcessInformation)], c.c_int),
            'ResumeThread': ([c.c_void_p], c.c_uint32),
            'WaitForSingleObject': ([c.c_void_p, c.c_uint32], c.c_uint32),
            'GetExitCodeProcess': ([c.c_void_p, c.POINTER(c.c_uint32)], c.c_int),
            'GetProcessTimes': ([c.c_void_p, c.POINTER(FileTime), c.POINTER(FileTime),
                                 c.POINTER(FileTime), c.POINTER(FileTime)], c.c_int),
            'QueryFullProcessImageNameW': ([c.c_void_p, c.c_uint32, c.c_wchar_p, c.POINTER(c.c_uint32)], c.c_int),
            'GetProcessId': ([c.c_void_p], c.c_uint32),
            'GetCurrentProcess': ([], c.c_void_p),
            'GetProcessIoCounters': ([c.c_void_p, c.POINTER(IOCounters)], c.c_int),
            'CloseHandle': ([c.c_void_p], c.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(k, name)
            function.argtypes, function.restype = args, result

    def call(self, name, *args, allow_false=False):
        with self.counter_lock:
            self.calls += 1
            require(self.calls <= 1000000, 'finite native API callback budget')
        value = getattr(self.kernel, name)(*args)
        if not allow_false:
            require(bool(value), 'owned native API failed: ' + name + ' / ' + str(c.get_last_error()))
        return value

    def own_io(self):
        counters = IOCounters()
        self.call('GetProcessIoCounters', self.call('GetCurrentProcess'), c.byref(counters))
        return counter_dict(counters)

    def boot_identity(self):
        # Exact existing runner.windows_boot_identity semantics, copied here
        # without importing model/scientific modules. It is re-reviewed source.
        class BootEnvironment(c.Structure):
            _fields_ = [('identifier', c.c_ubyte * 16), ('firmware', c.c_uint32), ('flags', c.c_uint64)]
        information = BootEnvironment()
        returned = c.c_uint32()
        query = c.WinDLL('ntdll', use_last_error=True).NtQuerySystemInformation
        query.argtypes = [c.c_uint32, c.c_void_p, c.c_uint32, c.POINTER(c.c_uint32)]
        query.restype = c.c_long
        with self.counter_lock:
            self.calls += 1
        status = query(90, c.byref(information), c.sizeof(information), c.byref(returned))
        require(status == 0 and returned.value == c.sizeof(information), 'actual boot GUID query unavailable')
        identifier = uuid.UUID(bytes_le=bytes(information.identifier))
        require(identifier.int != 0 and information.firmware in (1, 2) and not information.flags & 2,
                'ambiguous boot/hibernation epoch refused')
        return 'windows-boot-guid-' + str(identifier)

    def create_job(self):
        require(self.job is None, 'one fresh unnamed owned Job')
        self.job = self.call('CreateJobObjectW', None, None)
        limits = ExtendedLimits()
        limits.basic.flags = KILL_ON_JOB_CLOSE
        self.call('SetInformationJobObject', self.job, 9, c.byref(limits), c.sizeof(limits))
        # No breakaway, silent breakaway, CPU-time replacement, memory-limit
        # weakening, active-process cap change, or inherited Job handle.

    def _pipes(self):
        import msvcrt
        self.stdin_fd = os.open('NUL', os.O_RDONLY | os.O_BINARY)
        os.set_inheritable(self.stdin_fd, True)
        for _ in range(2):
            read_fd, write_fd = os.pipe()
            self.read_fds.append(read_fd)
            self.write_fds.append(write_fd)
            os.set_inheritable(read_fd, False)
            os.set_inheritable(write_fd, True)
        return [msvcrt.get_osfhandle(fd) for fd in [self.stdin_fd, *self.write_fds]]

    def create_suspended(self, argv, cwd, environment):
        require(self.job is not None and self.process is None, 'fresh owned Job precedes creation')
        handles = self._pipes()
        size = c.c_size_t()
        self.call('InitializeProcThreadAttributeList', None, 2, 0, c.byref(size), allow_false=True)
        require(0 < size.value <= 1024**2, 'bounded Job/handle attribute allocation')
        self.attribute_buffer = c.create_string_buffer(size.value)
        self.attribute_pointer = c.cast(self.attribute_buffer, c.c_void_p)
        self.call('InitializeProcThreadAttributeList', self.attribute_pointer, 2, 0, c.byref(size))
        self.attributes_initialized = True
        job_array = (c.c_void_p * 1)(self.job)
        handle_array = (c.c_void_p * 3)(*handles)
        self.attribute_arrays = (job_array, handle_array)
        self.call('UpdateProcThreadAttribute', self.attribute_pointer, 0, JOB_LIST_ATTRIBUTE,
                  c.cast(job_array, c.c_void_p), c.sizeof(job_array), None, None)
        self.call('UpdateProcThreadAttribute', self.attribute_pointer, 0, HANDLE_LIST_ATTRIBUTE,
                  c.cast(handle_array, c.c_void_p), c.sizeof(handle_array), None, None)
        startup = StartupInfoEx()
        startup.startup.cb = c.sizeof(startup)
        startup.startup.flags = 0x100
        startup.startup.stdin, startup.startup.stdout, startup.startup.stderr = handles
        startup.attributes = self.attribute_pointer
        pi = ProcessInformation()
        command = c.create_unicode_buffer(subprocess.list2cmdline(list(argv)))
        require(type(environment) is dict and all(type(k) is str and type(v) is str
                and '\x00' not in k + v and '=' not in k for k, v in environment.items()),
                'finite explicit environment')
        env_text = '\x00'.join(k + '=' + v for k, v in sorted(environment.items(), key=lambda pair: pair[0].upper())) + '\x00\x00'
        require(len(env_text) <= 32767, 'bounded Unicode environment')
        env = c.create_unicode_buffer(env_text)
        flags = CREATE_SUSPENDED | CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT
        try:
            self.call('CreateProcessW', argv[0], command, None, None, True, flags,
                      c.cast(env, c.c_void_p), str(cwd), c.byref(startup), c.byref(pi))
            # Birth Job membership is atomic with CreateProcessW. If this
            # wrapper dies before these assignments, KILL_ON_JOB_CLOSE stops
            # the suspended member through the already owned Job handle.
            self.pi, self.process, self.thread = pi, pi.process, pi.thread
            inside = c.c_int()
            self.call('IsProcessInJob', self.process, self.job, c.byref(inside))
            require(inside.value == 1, 'exact new suspended process belongs to original Job')
            birth, exit_time, kernel, user = FileTime(), FileTime(), FileTime(), FileTime()
            self.call('GetProcessTimes', self.process, c.byref(birth), c.byref(exit_time), c.byref(kernel), c.byref(user))
            image = c.create_unicode_buffer(32768)
            image_size = c.c_uint32(len(image))
            self.call('QueryFullProcessImageNameW', self.process, 0, image, c.byref(image_size))
            require(self.call('GetProcessId', self.process) == pi.pid
                    and Path(image.value).resolve() == Path(argv[0]).resolve(), 'retained exact supervisor image/PID')
            self.identity = {'pid': int(pi.pid), 'creation_filetime_100ns': filetime(birth),
                             'image_path': image.value, 'argv': list(argv)}
            require(self.identity['creation_filetime_100ns'] > 0, 'actual retained supervisor birth')
            return dict(self.identity)
        finally:
            failures = []
            if self.attributes_initialized:
                try:
                    self.call('DeleteProcThreadAttributeList', self.attribute_pointer, allow_false=True)
                    self.attributes_initialized = False
                    self.attribute_pointer = None
                except BaseException as error:
                    failures.append('attributes: ' + str(error))
            remaining = []
            for fd in self.write_fds:
                try:
                    os.close(fd)
                except OSError as error:
                    remaining.append(fd)
                    failures.append('writer fd: ' + str(error))
            self.write_fds = remaining
            if self.stdin_fd is not None:
                try:
                    os.close(self.stdin_fd)
                    self.stdin_fd = None
                except OSError as error:
                    failures.append('stdin fd: ' + str(error))
            require(not failures, 'owned creation ancillary cleanup failed: ' + '; '.join(failures))

    def resume(self):
        require(self.thread is not None, 'original retained primary thread')
        value = self.call('ResumeThread', self.thread, allow_false=True)
        require(value == 1, 'exact one original suspend count; no blind repeated resumes')

    def poll(self):
        require(self.process is not None and self.job is not None, 'owned retained process and Job')
        waited = self.call('WaitForSingleObject', self.process, 0, allow_false=True)
        require(waited in (WAIT_OBJECT_0, WAIT_TIMEOUT), 'owned process wait failed')
        accounted = JobAccounting()
        self.call('QueryInformationJobObject', self.job, 8, c.byref(accounted), c.sizeof(accounted), None)
        code = None
        if waited == WAIT_OBJECT_0:
            result = c.c_uint32()
            self.call('GetExitCodeProcess', self.process, c.byref(result))
            code = int(result.value)
        return {'supervisor_signaled': waited == WAIT_OBJECT_0,
                'actual_supervisor_returncode': code, 'job_active_processes': int(accounted.basic.active),
                'job_total_processes': int(accounted.basic.total), 'job_cumulative_io': counter_dict(accounted.io)}

    def stop(self):
        if self.job is not None:
            self.call('TerminateJobObject', self.job, TERMINATED)

    def close(self):
        failures = []
        # Always attempt every original resource, even after failed accounting,
        # failed publication or an earlier CloseHandle. No PID reopen or kill.
        if self.attributes_initialized:
            try:
                self.call('DeleteProcThreadAttributeList', self.attribute_pointer, allow_false=True)
                self.attributes_initialized = False
                self.attribute_pointer = None
            except BaseException as error:
                failures.append('attributes: ' + str(error))
        for name in ('thread', 'process', 'job'):
            handle = getattr(self, name)
            if handle is not None:
                try:
                    self.call('CloseHandle', handle)
                    setattr(self, name, None)
                except BaseException as error:
                    failures.append(name + ': ' + str(error))
        failed_fds = []
        for fd in [*self.read_fds, *self.write_fds, *([self.stdin_fd] if self.stdin_fd is not None else [])]:
            try:
                os.close(fd)
            except OSError as error:
                failed_fds.append(fd)
                failures.append('fd: ' + str(error))
        self.read_fds = failed_fds
        self.write_fds = []
        self.stdin_fd = None
        self.closed = not failures
        require(not failures, 'owned handle cleanup incomplete: ' + '; '.join(failures))


class BoundedLogs:
    def __init__(self, backend, directory, maximum, maximum_operations=MAX_LOG_OPERATIONS):
        require(type(maximum) is int and 0 < maximum <= 32 * 1024**2, 'fixed finite aggregate log limit')
        self.backend, self.directory, self.maximum = backend, Path(directory), maximum
        require(type(maximum_operations) is int and 16 <= maximum_operations <= MAX_LOG_OPERATIONS,
                'finite selected log call ceiling')
        self.maximum_operations = maximum_operations
        self.persisted = self.read = self.calls = 0
        self.lock = threading.Lock()
        self.errors = []
        self.threads = []

    def _read(self, fd, name):
        try:
            with self.lock:
                self.calls += 1
                require(self.calls <= self.maximum_operations - 16, 'bounded log open operation ceiling')
            with (self.directory / name).open('xb') as stream:
                while True:
                    with self.lock:
                        # Reserve before every read. A short read is allowed;
                        # a large requested chunk never proves few calls.
                        require(self.calls + 1 <= self.maximum_operations - 16, 'bounded log read operation ceiling')
                        self.calls += 1
                    block = os.read(fd, 4096)
                    with self.lock:
                        self.read += len(block)
                    if not block:
                        break
                    with self.lock:
                        require(self.persisted + len(block) <= self.maximum, 'supervisor aggregate log cap exceeded')
                        require(self.calls + 1 <= self.maximum_operations - 16, 'bounded log write operation ceiling')
                        self.calls += 1
                        require(stream.write(block) == len(block), 'short bounded log write')
                        self.persisted += len(block)
                with self.lock:
                    require(self.calls + 3 <= self.maximum_operations, 'bounded log flush/fsync/close operation ceiling')
                    self.calls += 3
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException as error:
            with self.lock:
                self.errors.append(str(error)[:1000])
            try:
                self.backend.stop()
            except BaseException as secondary:
                with self.lock:
                    self.errors.append(str(secondary)[:1000])

    def start(self):
        require(len(self.backend.read_fds) == 2, 'two original supervisor output pipe handles')
        for fd, name in zip(self.backend.read_fds, ('supervisor.stdout.log', 'supervisor.stderr.log')):
            thread = threading.Thread(target=self._read, args=(fd, name), daemon=True)
            self.threads.append(thread)
            thread.start()

    def finish(self, timeout):
        end = time.monotonic() + timeout
        for thread in self.threads:
            thread.join(max(0, end - time.monotonic()))
        require(all(not thread.is_alive() for thread in self.threads), 'bounded owned log readers not terminal')
        require(not self.errors, 'owned logs failed: ' + '; '.join(self.errors))
        return {'read_bytes': self.read, 'write_bytes': self.persisted, 'operation_attempts': self.calls}


def run_owned(backend, *, argv, cwd, environment, directory, deadline,
              maximum_logs, clock=time.monotonic, sleep=time.sleep,
              log_factory=BoundedLogs, on_birth=lambda _: None, close_on_return=True,
              maximum_polls=MAX_POLL_COUNT, cleanup_deadline=None):
    """Reviewable protocol shared by real API and pure fake backends.

    Caller prepays whole operation plus failure cleanup before entry. Caller
    retains a separate independent outer watchdog through final publication.
    This operation never cancels that watchdog or infers native qualification.
    """
    require(type(deadline) in (int, float) and math.isfinite(deadline) and deadline > clock(),
            'physical deadline already exhausted')
    require(type(maximum_polls) is int and 1 <= maximum_polls <= MAX_POLL_COUNT, 'finite paid poll ceiling')
    cleanup_deadline = deadline if cleanup_deadline is None else cleanup_deadline
    require(type(cleanup_deadline) in (int, float) and math.isfinite(cleanup_deadline)
            and deadline <= cleanup_deadline <= deadline + 1,
            'source-selected cleanup clock can add at most one fixture second; outer watchdog never changes')
    logs = None
    result = primary = None
    started = clock()
    try:
        backend.create_job()
        require(clock() < deadline, 'outer deadline before supervisor birth')
        identity = backend.create_suspended(argv, cwd, environment)
        require(clock() < deadline, 'outer deadline before supervisor resume')
        on_birth(identity)
        require(clock() < deadline, 'birth publication consumed remaining physical allowance')
        logs = log_factory(backend, directory, maximum_logs)
        logs.start()
        backend.resume()
        for _ in range(maximum_polls):
            require(clock() < deadline, 'outer physical deadline exhausted')
            observation = backend.poll()
            if observation['supervisor_signaled'] and observation['job_active_processes'] == 0:
                result = observation
                break
            sleep(min(.1, max(0, deadline - clock())))
        require(result is not None, 'finite owned process poll cap exceeded')
        output = logs.finish(min(10, max(0, deadline - clock())))
        require(clock() <= deadline, 'owned process/log terminal observation after physical deadline')
        result.update(same_retained_supervisor_birth=True, retained_supervisor_identity=identity,
                      terminal_monotonic=clock(), log_accounting=output,
                      native_qualified=False, execution_verified=False)
        return result
    except BaseException as error:
        primary = error
        try:
            backend.stop()
            # Main qualification uses the SAME deadline; no added physical
            # time. Tiny separate fixture cases may price <=1s cleanup inside
            # their one120s enclosing deadline to observe termination evidence.
            if getattr(backend, 'process', None) is not None:
                for _ in range(100):
                    if clock() >= cleanup_deadline:
                        break
                    aborted = backend.poll()
                    if aborted['supervisor_signaled'] and aborted['job_active_processes'] == 0:
                        backend.abort_evidence = aborted
                        break
                    sleep(min(.01, max(0, cleanup_deadline - clock())))
        except BaseException as stop_error:
            error.add_note('Original owned Job termination failed: ' + str(stop_error)[:1000])
        raise
    finally:
        if primary is not None or close_on_return:
            try:
                backend.close()
            except BaseException as cleanup:
                if primary is not None:
                    primary.add_note('Original owned handle cleanup failed: ' + str(cleanup)[:1000])
                    raise primary from cleanup
                raise
        if result is not None:
            result['all_owned_handles_closed'] = backend.closed
            result['provider_wall_seconds'] = clock() - started


def explicit_environment():
    """Whitelist preserves deterministic runtime; no argv/scope comes from env."""
    names = ('SystemRoot', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'LOCALAPPDATA', 'APPDATA', 'USERPROFILE')
    env = {key: os.environ[key] for key in names if key in os.environ}
    env.update(PYTHONHASHSEED='0', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return env
