"""Read-only Windows worker provenance; no spawning, model or native grant.

Registered observation authority remains absent. Native proof paths require a
source-issued metadata permit before OS/filesystem work. Optional current-PID
probes are explicitly non-native. Process termination uses a retained OS handle,
GetProcessTimes and a signaled wait plus GetExitCodeProcess, never a status file.

API sources: Microsoft Learn GetProcessTimes/GetExitCodeProcess/
QueryFullProcessImageNameW/CommandLineToArgvW/WaitForSingleObject; System Informer
phnt ntpsapi.h declares ProcessCommandLineInformation=60 (UNICODE_STRING).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

SCOPE = 'CE8-WORKER-OBSERVATION-SCOPE-1'
PROBE = 'CE8-READONLY-WORKER-PROBE-SCOPE-1'
PROOF = 'CE8-QUALIFICATION-WORKER-PROOF-1'
IDENTITY = 'CE8-WINDOWS-PROCESS-IDENTITY-1'
TERMINAL = 'CE8-WINDOWS-PROCESS-TERMINAL-1'
_REGISTERED_METADATA_VERIFIERS = frozenset()
_PERMIT_SEAL = object()
_HANDLE_SEAL = object()
CHUNK = 1024 ** 2


class WorkerEvidenceError(ValueError): pass


def require(condition, message):
    if not condition: raise WorkerEvidenceError(message)


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def _hex(value): return type(value) is str and len(value) == 64 and not set(value) - set('0123456789abcdef')


def validate_spec(spec, *, probe=False):
    require(type(spec) is dict and spec.get('schema') == (PROBE if probe else SCOPE), 'explicit worker observation scope')
    require(spec.get('mode') == ('readonly_current_probe' if probe else 'registered_metadata'), 'worker observation mode')
    require(all(_hex(spec.get(key)) for key in ('request_sha256', 'scope_sha256')), 'exact worker request/scope binding')
    for key in ('expected_pid', 'expected_creation_filetime_100ns', 'launch_not_before_filetime_100ns', 'launch_not_after_filetime_100ns'):
        require(type(spec.get(key)) is int and spec[key] > 0, 'explicit positive OS identity/window: ' + key)
    require(spec['launch_not_before_filetime_100ns'] <= spec['expected_creation_filetime_100ns']
            <= spec['launch_not_after_filetime_100ns'], 'expected worker creation outside launch window')
    require(type(spec.get('expected_boot_identity')) is str and spec['expected_boot_identity'].startswith('windows-boot-guid-'),
            'exact Windows boot GUID required')
    require(type(spec.get('source_checkout')) is str and Path(spec['source_checkout']).is_absolute(), 'absolute source checkout required')
    require(type(spec.get('argv')) is list and spec['argv'] and all(type(arg) is str and arg for arg in spec['argv']), 'explicit whole OS argv required')
    refs = spec.get('source_files')
    require(type(refs) is dict and refs and all(type(name) is str and name for name in refs), 'complete source file/hash map required')
    approved = spec.get('approved_code_sha256')
    require(type(approved) is dict and set(approved) == set(refs)
            and all(_hex(value) and refs[name].get('sha256') == value for name, value in approved.items()),
            'worker source bundle must equal complete approved source map')
    references = [spec.get('python_ref'), spec.get('process_image_ref'), spec.get('request_ref'), *refs.values()]
    if probe and spec.get('entry_script_ref') is not None: references.append(spec['entry_script_ref'])
    for reference in references:
        require(type(reference) is dict and set(reference) == {'path', 'sha256', 'bytes'}
                and type(reference['path']) is str and Path(reference['path']).is_absolute()
                and _hex(reference['sha256']) and type(reference['bytes']) is int and reference['bytes'] > 0,
                'exact explicit file reference required')
    require(Path(spec['argv'][0]) == Path(spec['process_image_ref']['path']), 'whole OS argv image differs from pinned process image')
    if probe and spec.get('entry_script_ref') is not None:
        require(type(spec['entry_script_ref']) is dict, 'explicit raw-hash probe script reference required')
    else:
        require(type(spec.get('entry_module')) is str and spec['entry_module']
                and all(part.isidentifier() for part in spec['entry_module'].split('.')), 'source-bound worker module required')
    require(type(spec.get('bounds')) is dict and set(spec['bounds']) == {'operation_attempts_upper', 'io_read_bytes_upper',
            'identity_hash_attempts_upper', 'identity_hash_bytes_upper', 'wall_seconds_upper'}
            and all(type(value) is int and value > 0 for value in spec['bounds'].values()), 'finite prepaid OS/source observation bounds')
    canonical(spec)
    return spec


class WorkerObservationPermit:
    """Metadata-only pre-model capability; never a model/training permission."""
    __slots__ = ('_seal', '_spec', '_sha', '_issuer', '_verifier', '_observer')
    def __init__(self, spec, issuer, verifier, observer):
        require(any(verifier is item for item in _REGISTERED_METADATA_VERIFIERS), 'no source-registered worker metadata authority')
        validate_spec(spec)
        require(issuer is not None and not isinstance(issuer, (dict, list, str, int, float, bool)) and callable(observer), 'opaque issuer/bound preflight observer required')
        hashed = digest(spec)
        require(verifier(issuer, hashed, spec['request_sha256'], spec['scope_sha256']) is True, 'worker metadata authority failed')
        for key, value in (('_seal', _PERMIT_SEAL), ('_spec', copy.deepcopy(spec)), ('_sha', hashed),
                ('_issuer', issuer), ('_verifier', verifier), ('_observer', observer)):
            object.__setattr__(self, key, value)
    def __setattr__(self, name, value): raise WorkerEvidenceError('worker observation permit is immutable')
    def require(self, spec, observer):
        require(getattr(self, '_seal', None) is _PERMIT_SEAL
                and any(getattr(self, '_verifier', None) is item for item in _REGISTERED_METADATA_VERIFIERS), 'source-issued worker metadata permit required')
        require(observer is self._observer and digest(self._spec) == self._sha == digest(spec), 'worker metadata scope/callback drift')
        require(self._verifier(self._issuer, self._sha, spec['request_sha256'], spec['scope_sha256']) is True, 'worker metadata authority no longer valid')


def _admit(spec, permit, observer):
    require(type(permit) is WorkerObservationPermit, 'opaque worker observation permit required before OS/IO')
    permit.require(spec, observer)
    validate_spec(spec)


class _Budget:
    def __init__(self, spec, observer): self.remaining, self.observer, self.start = dict(spec['bounds']), observer, time.perf_counter()
    def run(self, kind, work, *, read=0, hashes=0, hashed=0):
        wanted = {'operation_attempts_upper': 1, 'io_read_bytes_upper': read,
                  'identity_hash_attempts_upper': hashes, 'identity_hash_bytes_upper': hashed}
        require(time.perf_counter() - self.start <= self.remaining['wall_seconds_upper'], 'worker observation wall limit')
        require(all(self.remaining[key] >= value for key, value in wanted.items()), 'worker observation allowance exhausted before work')
        for key, value in wanted.items(): self.remaining[key] -= value
        identifier, started = uuid.uuid4().hex, time.perf_counter()
        self.observer(kind, 'started', operation_id=identifier, **wanted)
        try:
            result = work()
            require(time.perf_counter() - self.start <= self.remaining['wall_seconds_upper'], 'worker observation wall limit exceeded after work')
        except BaseException as error:
            try: self.observer(kind, 'failed', operation_id=identifier, wall_s=time.perf_counter() - started,
                               error_type=type(error).__name__, error=str(error)[:1000])
            except BaseException as record_error: error.worker_accounting_failure = str(record_error)[:1000]
            raise
        self.observer(kind, 'returned', operation_id=identifier, wall_s=time.perf_counter() - started)
        return result


def _file(reference, budget):
    def read():
        path = Path(reference['path'])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == reference['bytes'], 'worker bound file path/size changed')
        hashed, count = hashlib.sha256(), 0
        with path.open('rb') as stream:
            while block := stream.read(min(CHUNK, reference['bytes'] + 1 - count)):
                count += len(block); require(count <= reference['bytes'], 'worker bound file grew'); hashed.update(block)
        require(count == reference['bytes'] and hashed.hexdigest() == reference['sha256'], 'worker source/Python/request raw hash drift')
        return copy.deepcopy(reference)
    return budget.run('worker_source_verification', read, read=reference['bytes'] + 1, hashes=1, hashed=reference['bytes'] + 1)


class _WindowsOS:
    """Actual read-only Windows APIs, loaded only after observation admission."""
    def __init__(self):
        require(os.name == 'nt', 'Windows OS process evidence required')
        import ctypes
        self.ctypes = ctypes
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.ntdll = ctypes.WinDLL('ntdll')
        self.shell = ctypes.WinDLL('shell32', use_last_error=True)
        signatures = {'OpenProcess': ([ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32], ctypes.c_void_p),
            'CloseHandle': ([ctypes.c_void_p], ctypes.c_int), 'GetProcessId': ([ctypes.c_void_p], ctypes.c_uint32),
            'GetProcessTimes': ([ctypes.c_void_p] + [ctypes.c_void_p] * 4, ctypes.c_int),
            'GetExitCodeProcess': ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)], ctypes.c_int),
            'WaitForSingleObject': ([ctypes.c_void_p, ctypes.c_uint32], ctypes.c_uint32),
            'QueryFullProcessImageNameW': ([ctypes.c_void_p, ctypes.c_uint32, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32)], ctypes.c_int),
            'LocalFree': ([ctypes.c_void_p], ctypes.c_void_p)}
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name); function.argtypes, function.restype = arguments, result
        self.shell.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
        self.shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
        self.ntdll.NtQueryInformationProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        self.ntdll.NtQueryInformationProcess.restype = ctypes.c_long
    def open(self, pid):
        handle = self.kernel.OpenProcess(0x1000 | 0x100000, False, pid)
        require(bool(handle), 'worker process unavailable; identity cannot be observed')
        return handle
    def close(self, handle): require(bool(self.kernel.CloseHandle(handle)), 'worker handle close failed')
    def boot(self):
        from .runner import windows_boot_identity
        return windows_boot_identity()
    def command(self, handle):
        c = self.ctypes
        needed = c.c_uint32()
        status = self.ntdll.NtQueryInformationProcess(handle, 60, None, 0, c.byref(needed))
        require((status & 0xffffffff) in (0xc0000004, 0xc0000023) and 0 < needed.value <= 131072, 'bounded OS command line query unavailable')
        buffer = c.create_string_buffer(needed.value)
        require(self.ntdll.NtQueryInformationProcess(handle, 60, buffer, len(buffer), c.byref(needed)) == 0, 'OS command line read failed')
        class Unicode(c.Structure): _fields_ = [('length', c.c_uint16), ('maximum', c.c_uint16), ('pointer', c.c_void_p)]
        value = Unicode.from_buffer(buffer); address = c.addressof(buffer)
        require(value.length > 0 and value.length % 2 == 0 and value.length <= value.maximum
                and value.pointer is not None and address + c.sizeof(Unicode) <= value.pointer
                and value.pointer + value.length <= address + len(buffer), 'OS command line pointer/length ambiguity')
        text = c.wstring_at(value.pointer, value.length // 2)
        count = c.c_int(); argv = self.shell.CommandLineToArgvW(text, c.byref(count))
        require(bool(argv) and 0 < count.value <= 4096, 'OS argv parsing failed')
        try: return [argv[index] for index in range(count.value)]
        finally: self.kernel.LocalFree(c.cast(argv, c.c_void_p))
    def snapshot(self, handle, *, command=True):
        c = self.ctypes
        class FileTime(c.Structure): _fields_ = [('low', c.c_uint32), ('high', c.c_uint32)]
        creation, exited, kernel, user = FileTime(), FileTime(), FileTime(), FileTime()
        require(bool(self.kernel.GetProcessTimes(handle, c.byref(creation), c.byref(exited), c.byref(kernel), c.byref(user))), 'OS process times unavailable')
        pid = int(self.kernel.GetProcessId(handle)); require(pid > 0, 'OS PID unavailable')
        name = c.create_unicode_buffer(32768); length = c.c_uint32(len(name))
        require(bool(self.kernel.QueryFullProcessImageNameW(handle, 0, name, c.byref(length))), 'OS executable image unavailable')
        wait = int(self.kernel.WaitForSingleObject(handle, 0)); require(wait in (0, 258), 'OS process wait failed/ambiguous')
        code = c.c_uint32(); require(bool(self.kernel.GetExitCodeProcess(handle, c.byref(code))), 'OS process exit code unavailable')
        result = {'pid': pid, 'creation_filetime_100ns': (int(creation.high) << 32) | int(creation.low),
            'image_path': name.value, 'signaled': wait == 0, 'exit_code': int(code.value),
            'exit_filetime_100ns': ((int(exited.high) << 32) | int(exited.low)) if wait == 0 else None}
        if command: result['argv'] = self.command(handle)
        return result


def compare_identity(spec, facts, boot):
    """Pure comparison of already observed OS facts; no Boolean freshness input."""
    require(type(facts) is dict and type(facts.get('pid')) is int and facts['pid'] == spec['expected_pid'], 'foreign/reused worker PID')
    require(type(facts.get('creation_filetime_100ns')) is int
            and facts['creation_filetime_100ns'] == spec['expected_creation_filetime_100ns']
            and spec['launch_not_before_filetime_100ns'] <= facts['creation_filetime_100ns'] <= spec['launch_not_after_filetime_100ns'],
            'worker creation time/reuse/launch-window mismatch')
    require(boot == spec['expected_boot_identity'], 'worker boot epoch mismatch')
    require(Path(facts['image_path']) == Path(spec['process_image_ref']['path']), 'worker executable image mismatch')
    if 'argv' in facts: require(facts['argv'] == spec['argv'], 'whole OS worker argv mismatch')
    return {'schema': IDENTITY, 'pid': facts['pid'], 'creation_filetime_100ns': facts['creation_filetime_100ns'],
        'boot_identity': boot, 'image_path': facts['image_path'], 'argv': list(spec['argv'])}


class WinWorkerHandle:
    """One retained actual process handle, avoiding reopening a recycled PID."""
    def __init__(self, spec, permit, observer):
        _admit(spec, permit, observer)
        self._permit = permit
        self._open(spec, observer, native=True)
    def _open(self, spec, observer, *, native):
        if native: _admit(spec, getattr(self, '_permit', None), observer)
        self.spec, self.observer, self.native = copy.deepcopy(spec), observer, native
        self.budget = _Budget(spec, observer)
        self.os = self.budget.run('worker_os_binding', _WindowsOS)
        self.handle = self.budget.run('worker_os_binding', lambda: self.os.open(spec['expected_pid']))
        self._seal, self.closed = _HANDLE_SEAL, False
        try:
            boot = self.budget.run('worker_os_binding', self.os.boot)
            facts = self.budget.run('worker_os_binding', lambda: self.os.snapshot(self.handle))
            self.identity = compare_identity(spec, facts, boot)
            require(facts['signaled'] is False and facts['exit_code'] == 259, 'worker already terminal at admission')
            self.sources = self._sources()
            again = self.budget.run('worker_os_binding', lambda: self.os.snapshot(self.handle))
            require(compare_identity(spec, again, self.budget.run('worker_os_binding', self.os.boot)) == self.identity
                    and again['signaled'] is False, 'worker identity changed during source verification')
            self.identity_sha256 = digest({'os': self.identity, 'source': self.sources,
                'request_sha256': spec['request_sha256'], 'scope_sha256': spec['scope_sha256']})
        except BaseException as error:
            try: self.close()
            except BaseException as close_error:
                error.worker_close_failure = {'type': type(close_error).__name__, 'message': str(close_error)[:1000]}
                raise error from close_error
            raise
    def _sources(self):
        python = _file(self.spec['python_ref'], self.budget)
        image = _file(self.spec['process_image_ref'], self.budget)
        request = _file(self.spec['request_ref'], self.budget)
        runtime_launcher = None
        if self.spec['expected_pid'] == os.getpid():
            def observe_launcher():
                require(Path(sys.executable) == Path(self.spec['python_ref']['path']), 'runtime Python launcher differs from its pinned reference')
                return {'path': sys.executable, 'source': 'current worker sys.executable, separate from OS image/argv'}
            runtime_launcher = self.budget.run('worker_runtime_binding', observe_launcher)
        root = Path(self.spec['source_checkout'])
        sources = {}
        for name, ref in self.spec['source_files'].items():
            def verify_path():
                require(Path(ref['path']).resolve().is_relative_to(root.resolve()), 'worker source file outside bound checkout')
                require(Path(ref['path']).resolve() == (root / name).resolve(), 'worker source logical path mismatch')
            self.budget.run('worker_source_verification', verify_path)
            sources[name] = _file(ref, self.budget)
        entry_script = None
        if not self.native and self.spec.get('entry_script_ref') is not None:
            entry_script = _file(self.spec['entry_script_ref'], self.budget)
            entry = entry_script['path']
            require(any(Path(arg) == Path(entry) for arg in self.spec['argv'][1:]), 'probe OS argv/script source mismatch')
        else:
            entry = self.spec['entry_module'].replace('.', '/') + '.py'
            require(entry in sources, 'worker entry module absent from approved source bundle')
            args = self.spec['argv']; module_positions = [index for index, arg in enumerate(args) if arg == '-m']
            require(len(module_positions) == 1 and module_positions[0] + 1 < len(args)
                    and args[module_positions[0] + 1] == self.spec['entry_module'], 'OS argv does not invoke bound source module')
        return {'python_ref': python, 'process_image_ref': image, 'request_ref': request, 'source_files': sources,
                'entry': entry, 'entry_script_ref': entry_script, 'runtime_launcher': runtime_launcher,
                'launcher_provenance': 'current-runtime path plus raw hash' if runtime_launcher is not None
                    else 'launcher raw hash only; parent/issuer launch record must establish invocation provenance',
                'source_observation': 'raw file hashes at initial handle admission; not current files or executed-code proof'}
    def proof(self):
        require(self._seal is _HANDLE_SEAL and not self.closed, 'live retained OS worker handle required')
        if self.native: _admit(self.spec, self._permit, self.observer)
        boot = self.budget.run('worker_os_binding', self.os.boot)
        facts = self.budget.run('worker_os_binding', lambda: self.os.snapshot(self.handle))
        require(compare_identity(self.spec, facts, boot) == self.identity and facts['signaled'] is False
                and facts['exit_code'] == 259, 'retained worker is no longer live with the same OS identity')
        return {'schema': PROOF, 'worker_id': 'win-worker-' + self.identity_sha256,
            'worker_identity_sha256': self.identity_sha256, 'boot_identity_sha256': digest(self.identity['boot_identity']),
            'fresh_process': self.native, 'boot_verified': True, 'source_verified': self.native,
            'raw_source_hashes_verified': True,
            'request_sha256': self.spec['request_sha256'], 'scope_sha256': self.spec['scope_sha256'],
            'os_identity': copy.deepcopy(self.identity), 'sources': copy.deepcopy(self.sources),
            'native_qualified': False, 'execution_verified': False,
            'classification': 'registered read-only OS/source provenance' if self.native else 'read-only current-PID probe; no native provenance authority'}
    def terminal_evidence(self):
        require(self._seal is _HANDLE_SEAL and not self.closed, 'retained worker handle required for terminal evidence')
        if self.native: _admit(self.spec, self._permit, self.observer)
        boot = self.budget.run('worker_os_binding', self.os.boot)
        facts = self.budget.run('worker_os_binding', lambda: self.os.snapshot(self.handle, command=False))
        identity = compare_identity(self.spec, facts, boot)
        require(identity == self.identity and facts['signaled'] is True, 'same retained worker is not OS-terminal')
        require(type(facts['exit_code']) is int and type(facts['exit_filetime_100ns']) is int
                and facts['exit_filetime_100ns'] >= facts['creation_filetime_100ns'], 'actual OS terminal time/code required')
        return {'schema': TERMINAL, 'worker_identity_sha256': self.identity_sha256, 'os_identity': identity,
            'exit_code': facts['exit_code'], 'exit_filetime_100ns': facts['exit_filetime_100ns'],
            'request_sha256': self.spec['request_sha256'], 'scope_sha256': self.spec['scope_sha256'],
            'status_file_used': False, 'native_qualified': False}
    def close(self):
        if not getattr(self, 'closed', True):
            entered = False
            def release():
                nonlocal entered
                entered = True; self.os.close(self.handle)
            try: self.budget.run('worker_handle_close', release)
            finally:
                # Closing this retained read-only handle remains authorized if
                # its preflight receipt fails or the metadata quota is exhausted.
                if not entered: self.os.close(self.handle)
                self.closed = True


def open_worker(spec, permit, observer):
    _admit(spec, permit, observer)
    return WinWorkerHandle(spec, permit, observer)


def current_worker_proof(spec, permit, observer):
    _admit(spec, permit, observer)
    require(spec['expected_pid'] == os.getpid(), 'current worker proof cannot certify a foreign PID')
    worker = WinWorkerHandle(spec, permit, observer)
    try: return worker.proof()
    finally: worker.close()


def readonly_current_worker_probe(spec, observer):
    validate_spec(spec, probe=True)
    require(callable(observer) and spec['expected_pid'] == os.getpid(), 'explicit prepaid current-PID-only probe required')
    worker = object.__new__(WinWorkerHandle)
    worker._permit = None
    worker._open(spec, observer, native=False)
    try: return worker.proof()
    finally: worker.close()


def build_current_probe_spec(base, observer):
    """Observe current OS facts under an explicit prepaid metadata bootstrap.

    ``base`` supplies request/scope/source/Python/file bindings and finite bounds;
    only expected PID, creation, boot and launch window are filled from the OS.
    The resulting scope is strictly read-only and cannot assert native freshness.
    No child is created, no files are hashed until the subsequent probe.
    """
    require(type(base) is dict and base.get('schema') == PROBE and base.get('mode') == 'readonly_current_probe'
            and callable(observer), 'explicit prepaid non-native current-probe bootstrap required')
    budget = _Budget(base, observer)
    os_api = budget.run('worker_os_binding', _WindowsOS)
    pid = budget.run('worker_os_binding', os.getpid)
    handle = budget.run('worker_os_binding', lambda: os_api.open(pid))
    primary = None
    try:
        facts = budget.run('worker_os_binding', lambda: os_api.snapshot(handle))
        boot = budget.run('worker_os_binding', os_api.boot)
        require(facts['pid'] == pid and facts['signaled'] is False, 'current process must be OS-live')
    except BaseException as error:
        primary = error; raise
    finally:
        entered = False
        def release():
            nonlocal entered
            entered = True; os_api.close(handle)
        try: budget.run('worker_handle_close', release)
        except BaseException as close_error:
            if not entered:
                try: os_api.close(handle)
                except BaseException as physical_close_error:
                    close_error.physical_close_failure = {'type': type(physical_close_error).__name__, 'message': str(physical_close_error)[:1000]}
            if primary is not None:
                primary.worker_close_failure = {'type': type(close_error).__name__, 'message': str(close_error)[:1000]}
                raise primary from close_error
            raise
    result = copy.deepcopy(base)
    result.update(expected_pid=pid, expected_creation_filetime_100ns=facts['creation_filetime_100ns'],
        launch_not_before_filetime_100ns=facts['creation_filetime_100ns'],
        launch_not_after_filetime_100ns=facts['creation_filetime_100ns'], expected_boot_identity=boot,
        argv=facts['argv'])
    result['observed_process_image'] = {'path': facts['image_path'], 'pid': pid,
        'creation_filetime_100ns': facts['creation_filetime_100ns'], 'boot_identity': boot,
        'classification': 'prepaid read-only OS observation; raw image hash not yet verified'}
    if result.get('process_image_ref') is not None:
        validate_spec(result, probe=True)
    else:
        result['scope_status'] = 'INCOMPLETE_REQUIRES_EXPLICIT_PROCESS_IMAGE_RAW_REFERENCE'
    return result


def verify_worker_handle(worker, expected, *, require_native=True, terminal=False):
    require(type(worker) is WinWorkerHandle and getattr(worker, '_seal', None) is _HANDLE_SEAL
            and not getattr(worker, 'closed', True), 'source-owned retained OS handle required')
    require(type(require_native) is bool and type(terminal) is bool, 'strict verification mode flags')
    require(not require_native or worker.native is True, 'read-only fixture probe cannot establish native worker provenance')
    if require_native: _admit(worker.spec, getattr(worker, '_permit', None), worker.observer)
    require(type(expected) is dict and all(expected.get(key) == worker.spec[key] for key in ('request_sha256', 'scope_sha256'))
            and expected.get('worker_identity_sha256') == worker.identity_sha256, 'worker/request/scope binding mismatch')
    return worker.terminal_evidence() if terminal else worker.proof()
