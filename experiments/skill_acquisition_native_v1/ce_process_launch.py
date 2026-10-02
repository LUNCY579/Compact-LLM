"""Owned hidden Windows launch and bounded RPC; native registries remain empty.

No worker/module/environment defaults, shell, model imports or launch occur at
import. Native launches require a source-issued launch capability before IO or
process creation. The explicit fixture path permits only the bundled metadata
fixture after resource clearance. Messages carry no executable authority.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid

from . import ce_worker_evidence as worker

SPEC = 'CE8-PROCESS-LAUNCH-SPEC-1'
RPC = 'CE8-BOUNDED-RPC-1'
FIXTURE_MODULE = 'experiments.skill_acquisition_native_v1.ce_process_fixture'
_REGISTERED_LAUNCH_VERIFIERS = frozenset()
_REGISTERED_OBSERVATION_FACTORIES = frozenset()
_LAUNCH_SEAL = object()
_OWNED_SEAL = object()
_FIXTURE_CLEARANCE_SEAL = object()
require, digest, canonical = worker.require, worker.digest, worker.canonical


def validate_spec(spec, *, fixture=False):
    require(type(spec) is dict and spec.get('schema') == SPEC
            and spec.get('mode') == ('fixture_only' if fixture else 'registered_native_worker'), 'explicit launch scope/mode')
    require(all(worker._hex(spec.get(key)) for key in ('request_sha256', 'scope_sha256')), 'launch request/scope binding')
    for key in ('cwd', 'source_checkout'):
        require(type(spec.get(key)) is str and Path(spec[key]).is_absolute(), 'explicit absolute launch ' + key)
    require(Path(spec['cwd']) == Path(spec['source_checkout']), 'worker cwd must be bound source checkout')
    require(type(spec.get('worker_module')) is str and spec['worker_module']
            and all(part.isidentifier() for part in spec['worker_module'].split('.')), 'exact registered worker module')
    require(not fixture or spec['worker_module'] == FIXTURE_MODULE, 'fixture cannot select another worker module')
    require(type(spec.get('argv')) is list and spec['argv'] and all(type(arg) is str and arg for arg in spec['argv']), 'explicit launcher argv list')
    refs = spec.get('source_files'); approved = spec.get('approved_code_sha256')
    require(type(refs) is dict and refs and type(approved) is dict and set(refs) == set(approved)
            and all(worker._hex(approved[name]) and refs[name].get('sha256') == approved[name] for name in refs), 'complete approved launch source map')
    for ref in (spec.get('python_ref'), spec.get('process_image_ref'), spec.get('request_ref'), *refs.values()):
        require(type(ref) is dict and set(ref) == {'path', 'sha256', 'bytes'} and type(ref['path']) is str
                and Path(ref['path']).is_absolute() and worker._hex(ref['sha256'])
                and type(ref['bytes']) is int and ref['bytes'] > 0, 'exact launch file reference')
    args = spec['argv']
    require(Path(args[0]) == Path(spec['python_ref']['path']), 'launch argv must invoke exact pinned launcher')
    require(len(args) >= 5 and args[1:3] == ['-m', spec['worker_module']]
            and args.count('--request') == 1 and args.index('--request') + 1 < len(args)
            and Path(args[args.index('--request') + 1]) == Path(spec['request_ref']['path']), 'launch module/request arguments drift')
    require(spec['worker_module'].replace('.', '/') + '.py' in refs, 'launch worker module absent from approved source map')
    require(spec.get('worker_topology') in ('direct', 'venv_redirector'), 'explicit launcher/worker OS topology required')
    env = spec.get('environment')
    require(type(env) is dict and env and all(type(key) is str and key and type(value) is str for key, value in env.items()), 'explicit full worker environment required')
    require('CE8_RPC_BINDING' not in env, 'caller cannot override RPC binding environment')
    for key in ('deadline_seconds', 'handshake_seconds', 'rpc_seconds', 'terminate_grace_seconds'):
        require(type(spec.get(key)) in (int, float) and 0 < spec[key] < float('inf'), 'finite positive launch deadline: ' + key)
    for key in ('max_message_bytes', 'max_messages', 'max_stderr_bytes'):
        require(type(spec.get(key)) is int and spec[key] > 0, 'finite positive launch/RPC cap: ' + key)
    require(spec['max_message_bytes'] <= 1024 ** 2, 'bounded RPC message ceiling at most1MiB')
    require(type(spec.get('parent_identity')) is dict and type(spec['parent_identity'].get('pid')) is int
            and spec['parent_identity']['pid'] > 0 and type(spec['parent_identity'].get('creation_filetime_100ns')) is int
            and spec['parent_identity']['creation_filetime_100ns'] > 0 and type(spec['parent_identity'].get('boot_identity')) is str,
            'explicit exact parent OS identity required')
    require(type(spec.get('bounds')) is dict and set(spec['bounds']) == {'operation_attempts_upper', 'io_read_bytes_upper',
        'identity_hash_attempts_upper', 'identity_hash_bytes_upper', 'wall_seconds_upper'}
        and all(type(value) is int and value > 0 for value in spec['bounds'].values()), 'prepaid launch metadata bounds')
    canonical(spec)
    return spec


class NativeLaunchPermit:
    """Source-issued launch-only authority, separate from model execution."""
    __slots__ = ('_seal', '_spec', '_sha', '_issuer', '_verifier', '_observer')
    def __init__(self, spec, issuer, verifier, observer):
        require(any(verifier is item for item in _REGISTERED_LAUNCH_VERIFIERS), 'native launch authority not registered')
        validate_spec(spec)
        require(issuer is not None and not isinstance(issuer, (dict, list, str, int, float, bool)) and callable(observer), 'opaque launch issuer and bound observer required')
        hashed = digest(spec)
        require(verifier(issuer, hashed, spec['request_sha256'], spec['scope_sha256']) is True, 'native launch authority failed')
        for key, value in (('_seal', _LAUNCH_SEAL), ('_spec', copy.deepcopy(spec)), ('_sha', hashed),
                ('_issuer', issuer), ('_verifier', verifier), ('_observer', observer)):
            object.__setattr__(self, key, value)
    def __setattr__(self, key, value): raise worker.WorkerEvidenceError('native launch permit immutable')
    def require(self, spec, observer):
        require(getattr(self, '_seal', None) is _LAUNCH_SEAL
                and any(getattr(self, '_verifier', None) is item for item in _REGISTERED_LAUNCH_VERIFIERS), 'source-issued native launch permit required')
        require(observer is self._observer and digest(self._spec) == self._sha == digest(spec), 'native launch scope/observer changed')
        require(self._verifier(self._issuer, self._sha, spec['request_sha256'], spec['scope_sha256']) is True, 'native launch authority no longer valid')


def _admit(spec, permit, observer):
    require(type(permit) is NativeLaunchPermit, 'opaque native launch permit required before IO/process creation')
    permit.require(spec, observer); validate_spec(spec)


def _observation_factory(spec):
    descriptor = spec.get('observation_factory_descriptor')
    require(type(descriptor) is dict and set(descriptor) == {'module', 'symbol', 'source_sha256'}
            and type(descriptor['module']) is str and type(descriptor['symbol']) is str
            and worker._hex(descriptor['source_sha256']), 'explicit native observation factory descriptor')
    name = descriptor['module'].replace('.', '/') + '.py'
    require(spec['approved_code_sha256'].get(name) == descriptor['source_sha256'], 'post-launch observation factory source drift')
    actual = [item for item in _REGISTERED_OBSERVATION_FACTORIES if getattr(item, '__module__', None) == descriptor['module']
              and getattr(item, '__name__', None) == descriptor['symbol']]
    require(len(actual) == 1, 'post-launch native observation factory registry remains empty/unqualified')
    return actual[0]


def validate_message(message, binding, *, expected_kind, expected_sequence, expected_pid):
    require(type(message) is dict and set(message) == {'schema', 'session_id', 'request_sha256', 'scope_sha256',
            'kind', 'sequence', 'worker_pid', 'payload'} and message['schema'] == RPC, 'exact bounded RPC envelope')
    require(all(message.get(key) == binding[key] for key in ('session_id', 'request_sha256', 'scope_sha256')),
            'foreign RPC session/request/scope')
    require(message['kind'] == expected_kind and type(message['sequence']) is int and message['sequence'] == expected_sequence
            and type(message['worker_pid']) is int and message['worker_pid'] == expected_pid, 'RPC phase/sequence/worker mismatch')
    require(type(message['payload']) is dict and not any(key in message['payload'] for key in
            ('native_permit', 'launch_permit', 'approved', 'training_authorized', 'qualification_authorized')), 'RPC cannot convey authority flags')
    canonical(message)
    return message['payload']


def _parent_pid(os_api, handle):
    c = os_api.ctypes
    class Basic(c.Structure):
        _fields_ = [('exit', c.c_long), ('peb', c.c_void_p), ('affinity', c.c_size_t),
                    ('priority', c.c_long), ('pid', c.c_size_t), ('parent', c.c_size_t)]
    value, returned = Basic(), c.c_uint32()
    require(os_api.ntdll.NtQueryInformationProcess(handle, 0, c.byref(value), c.sizeof(value), c.byref(returned)) == 0
            and returned.value == c.sizeof(value) and value.pid == os_api.kernel.GetProcessId(handle)
            and value.parent > 0, 'OS process parent identity unavailable')
    return int(value.parent)


def _child_pids(os_api, parent_pid):
    """Read-only Toolhelp process snapshot; returned IDs still need handle proof."""
    c, kernel = os_api.ctypes, os_api.kernel
    class Entry(c.Structure):
        _fields_ = [('size', c.c_uint32), ('usage', c.c_uint32), ('pid', c.c_uint32), ('heap', c.c_size_t),
                    ('module', c.c_uint32), ('threads', c.c_uint32), ('parent', c.c_uint32),
                    ('priority', c.c_long), ('flags', c.c_uint32), ('exe', c.c_wchar * 260)]
    kernel.CreateToolhelp32Snapshot.argtypes, kernel.CreateToolhelp32Snapshot.restype = [c.c_uint32, c.c_uint32], c.c_void_p
    for name in ('Process32FirstW', 'Process32NextW'):
        function = getattr(kernel, name); function.argtypes, function.restype = [c.c_void_p, c.POINTER(Entry)], c.c_int
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    require(snapshot not in (None, c.c_void_p(-1).value), 'OS process census unavailable')
    result, count, entry = [], 0, Entry(); entry.size = c.sizeof(entry)
    try:
        available = kernel.Process32FirstW(snapshot, c.byref(entry))
        while available:
            count += 1; require(count <= 4096, 'bounded OS process census exceeded')
            if entry.parent == parent_pid and entry.pid > 0: result.append(int(entry.pid))
            available = kernel.Process32NextW(snapshot, c.byref(entry))
    finally: os_api.close(snapshot)
    return result


def validate_worker_lineage(spec, launcher, actual, actual_parent_pid, boot):
    expected = list(spec['argv']); expected[0] = spec['process_image_ref']['path']
    require(type(actual_parent_pid) is int and actual_parent_pid == launcher['pid'], 'actual worker has foreign OS parent')
    require(type(actual['pid']) is int and actual['pid'] != launcher['pid']
            and actual['creation_filetime_100ns'] >= launcher['creation_filetime_100ns'], 'worker PID/creation lineage mismatch')
    require(Path(actual['image_path']) == Path(spec['process_image_ref']['path']) and actual['argv'] == expected
            and boot == spec['parent_identity']['boot_identity'] and actual['signaled'] is False,
            'actual worker image/whole argv/boot/liveness mismatch')
    return actual


class OwnedProcess:
    """One owned Popen and retained OS handle, with bounded RPC/deadline."""
    def __init__(self, spec, observer, *, native, permit=None, _fixture_seal=None):
        if native:
            _admit(spec, permit, observer)
            self.observation_factory = _observation_factory(spec)  # completeness before process creation
        else:
            require(_fixture_seal is _FIXTURE_CLEARANCE_SEAL, 'root resource-cleared fixture launch entry required')
            validate_spec(spec, fixture=True)
        self.spec, self.observer, self.native, self.permit = copy.deepcopy(spec), observer, native, permit
        self.budget = worker._Budget(spec, observer)
        self.binding = {'session_id': uuid.uuid4().hex, 'request_sha256': spec['request_sha256'], 'scope_sha256': spec['scope_sha256']}
        self.sequence = self.sent = self.received = self.stderr_bytes = 0
        self.stdout, self.errors = queue.Queue(maxsize=spec['max_messages']), []
        self.stdout_thread=self.stderr_thread=None
        self.stderr_data=bytearray();self.stderr_lock=threading.Lock()
        self.started = time.perf_counter(); self.deadline = self.started + spec['deadline_seconds']
        self._seal, self.closed = _OWNED_SEAL, False
        self.process = self.os = self.monitor = None
        self.handle=self.worker_handle = None
        self.stop_reservation_id = None
        self.stop_finished = False
        self._launch()

    def _launch(self):
        def verify_sources():
            root = Path(self.spec['source_checkout']).resolve()
            for name, ref in self.spec['source_files'].items():
                require(Path(ref['path']).resolve() == (root / name).resolve()
                        and Path(ref['path']).resolve().is_relative_to(root), 'launch source outside approved checkout')
        self.budget.run('process_launch_preflight', verify_sources)
        for ref in (self.spec['python_ref'], self.spec['process_image_ref'], self.spec['request_ref'], *self.spec['source_files'].values()):
            worker._file(ref, self.budget)
        self.os = self.budget.run('process_launch_preflight', worker._WindowsOS)
        parent=None;parent_error=None
        try:
            def open_parent():
                nonlocal parent
                parent=self.os.open(os.getpid());return parent
            self.budget.run('process_launch_parent',open_parent)
            facts = self.budget.run('process_launch_parent', lambda: self.os.snapshot(parent, command=False))
            boot = self.budget.run('process_launch_parent', self.os.boot)
            require({key: facts[key] for key in ('pid', 'creation_filetime_100ns')} == {
                    key: self.spec['parent_identity'][key] for key in ('pid', 'creation_filetime_100ns')}
                    and boot == self.spec['parent_identity']['boot_identity'], 'launch parent OS identity changed')
        except BaseException as error:parent_error=error;raise
        finally:
            if parent is not None:
                try:self.os.close(parent)
                except BaseException as closing:
                    if parent_error is not None:
                        parent_error.owned_parent_close_failure={'type':type(closing).__name__,'message':str(closing)[:1000]}
                        raise parent_error from closing
                    raise
        def spawn():
            require(os.name == 'nt', 'hidden Windows worker launch only')
            import subprocess
            env = dict(self.spec['environment']); env['CE8_RPC_BINDING'] = canonical(self.binding).decode('ascii')
            self.process = subprocess.Popen(self.spec['argv'], shell=False, cwd=self.spec['cwd'], env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                creationflags=subprocess.CREATE_NO_WINDOW)
            # Capture bounded diagnostics before any identity query or returned
            # launch observer can reject an already-created owned process.
            self._start_pipe_readers()
            return self.process
        # Permanent cleanup prepayment precedes process creation. It remains
        # valid for only the Popen returned below if later observers refuse.
        self.stop_reservation_id = uuid.uuid4().hex
        self.observer('process_owned_stop', 'started', operation_id=self.stop_reservation_id,
                      deferred_owned_cleanup=True, launch_spec_sha256=digest(self.spec))
        try:
            self.budget.run('process_launch', spawn)
            self.handle = int(self.process._handle)  # retained owned Popen handle; never reopen PID
            facts = self.budget.run('process_launch_child', lambda: self.os.snapshot(self.handle))
            self.boot = self.budget.run('process_launch_child', self.os.boot)
            require(facts['pid'] == self.process.pid and self.boot == self.spec['parent_identity']['boot_identity']
                    and facts['signaled'] is False, 'owned launcher PID/boot/liveness mismatch')
            self.launcher_identity = dict(facts, boot_identity=self.boot)
            if self.spec['worker_topology'] == 'direct':
                expected_argv = list(self.spec['argv']); expected_argv[0] = self.spec['process_image_ref']['path']
                require(Path(facts['image_path']) == Path(self.spec['process_image_ref']['path']) and facts['argv'] == expected_argv,
                        'direct owned worker OS image/argv mismatch')
                self.worker_handle = self.handle
            else:
                require(Path(facts['image_path']) == Path(self.spec['python_ref']['path']) and facts['argv'] == self.spec['argv'],
                        'venv redirector launcher OS image/argv mismatch')
                self.worker_handle, facts = self._find_worker(self.launcher_identity)
            self.worker_pid = facts['pid']
            self.identity = {'pid': facts['pid'], 'creation_filetime_100ns': facts['creation_filetime_100ns'],
                'image_path': facts['image_path'], 'argv': facts['argv'], 'boot_identity': self.boot}
            self.worker_identity_sha256 = digest({'os': self.identity, 'launch_spec_sha256': digest(self.spec), 'parent_identity': self.spec['parent_identity']})
            if self.native:
                observation_spec, observation_permit = self.observation_factory(
                    self.spec, self.permit, self.identity, self.spec['parent_identity'], self.observer)
                require(type(observation_permit) is worker.WorkerObservationPermit, 'source-issued child observation permit required')
                self.monitor = worker.open_worker(observation_spec, observation_permit, self.observer)
                proof = self.monitor.proof()
                require(proof['os_identity']['pid'] == self.worker_pid
                        and proof['os_identity']['creation_filetime_100ns'] == self.identity['creation_filetime_100ns'],
                        'post-launch child observation changed owned process identity')
                self.worker_identity_sha256 = proof['worker_identity_sha256']
            retained = {'observed_worker_identity': self.identity, 'observed_launcher_identity': self.launcher_identity}
            if self.native:retained['observed_worker_proof'] = proof
            # The original paid launch owner retains these real observations
            # before any ready/control bytes are accepted from the child.
            self.budget.run('process_launch_child', lambda: None, retained_result_fields=retained)
        except BaseException as error:
            # Only this successfully-created owned child is eligible for stop.
            if self.process is not None:
                try: self._stop_owned('launch_identity_or_adapter_failure', error)
                except BaseException as stop_error:
                    error.owned_stop_failure = {'type': type(stop_error).__name__, 'message': str(stop_error)[:1000]}
                if self.process.poll() is not None:
                    try:
                        self._attach_owned_stderr(error)
                    except BaseException as diagnostic:
                        error.owned_stderr_failure={'type':type(diagnostic).__name__,'message':str(diagnostic)[:1000]}
                    try:
                        self.close()
                    except BaseException as closing:
                        error.owned_launch_cleanup_failure={'type':type(closing).__name__,'message':str(closing)[:1000]}
                        error.add_note('Owned startup cleanup failure: '+type(closing).__name__+': '+str(closing)[:1000])
                else:error.add_note('Owned startup cleanup could not establish launcher quiescence; resources remain owned.')
            raise

    def _start_pipe_readers(self):
        self.stderr_thread=threading.Thread(target=self._read_stderr,daemon=True)
        self.stderr_thread.start()
        self.stdout_thread=threading.Thread(target=self._read_stdout,daemon=True)
        self.stdout_thread.start()

    def _attach_owned_stderr(self,error):
        thread=getattr(self,'stderr_thread',None)
        if thread is not None:
            thread.join(timeout=min(5,self.spec['terminate_grace_seconds']))
        lock=getattr(self,'stderr_lock',None)
        if lock is not None:
            with lock:blob=bytes(self.stderr_data[-min(self.spec['max_stderr_bytes'],65536):])
        else:blob=b''
        if blob:
            diagnostic=blob.decode('utf8',errors='replace')
            error.owned_child_stderr=diagnostic
            error.add_note('Owned child stderr (bounded, untrusted diagnostic):\n'+diagnostic)
        if thread is not None and thread.is_alive():error.add_note('Owned child stderr reader did not finish within the bounded cleanup join.')

    def _attach_rpc_diagnostics(self,error,raw):
        if type(raw) is bytes:
            retained=raw[:min(self.spec['max_message_bytes'],65536)]
            error.owned_child_rpc_line=retained.decode('utf8',errors='replace')
            error.owned_child_rpc_line_bytes=len(retained)
            error.owned_child_rpc_line_total_bytes=len(raw)
            error.add_note('Owned child RPC line (bounded, untrusted diagnostic):\n'+error.owned_child_rpc_line)
        try:self._attach_owned_stderr(error)
        except BaseException as diagnostic:
            error.owned_stderr_failure={'type':type(diagnostic).__name__,'message':str(diagnostic)[:1000]}

    def _find_worker(self, launcher):
        finish = time.perf_counter() + self._remaining(self.spec['handshake_seconds'])
        while time.perf_counter() < finish:
            candidates = self.budget.run('process_launch_child_census', lambda: _child_pids(self.os, launcher['pid']))
            matches = []
            for pid in candidates:
                handle = None
                try:
                    def open_owned():
                        nonlocal handle
                        # Only this actual owned worker receives DUP_HANDLE.
                        # Ordinary read-only observation handles keep their rights.
                        handle = self.os.kernel.OpenProcess(0x1000 | 0x100000 | 1 | 0x40, False, pid)
                        require(bool(handle), 'owned worker process unavailable'); return handle
                    self.budget.run('process_launch_child', open_owned)
                    facts = self.budget.run('process_launch_child', lambda: self.os.snapshot(handle))
                    parent = self.budget.run('process_launch_child', lambda: _parent_pid(self.os, handle))
                    boot = self.budget.run('process_launch_child', self.os.boot)
                    try: validate_worker_lineage(self.spec, launcher, facts, parent, boot)
                    except worker.WorkerEvidenceError: continue
                    matches.append((handle, facts)); handle = None
                finally:
                    if handle is not None: self.os.close(handle)
            if len(matches) > 1:
                for handle, _ in matches: self.os.close(handle)
                raise worker.WorkerEvidenceError('multiple OS workers match one owned launcher; fail closed')
            if matches: return matches[0]
            time.sleep(.025)
        raise worker.WorkerEvidenceError('owned redirector worker was not identified before deadline')

    def _read_stdout(self):
        try:
            while True:
                raw = self.process.stdout.readline(self.spec['max_message_bytes'] + 1)
                if not raw: self.stdout.put_nowait(None); return
                require(len(raw) <= self.spec['max_message_bytes'] and raw.endswith(b'\n'), 'oversized/partial RPC message')
                self.stdout.put_nowait(raw)
        except BaseException as error: self.errors.append(error)
    def _read_stderr(self):
        try:
            while block := self.process.stderr.read(min(4096, self.spec['max_stderr_bytes'] + 1 - self.stderr_bytes)):
                self.stderr_bytes += len(block)
                with self.stderr_lock:
                    self.stderr_data.extend(block[:max(0,self.spec['max_stderr_bytes']-len(self.stderr_data))])
                require(self.stderr_bytes <= self.spec['max_stderr_bytes'], 'worker stderr cap exceeded')
        except BaseException as error: self.errors.append(error)

    def _remaining(self, cap):
        remaining = min(cap, self.deadline - time.perf_counter())
        require(remaining > 0, 'owned worker recorded deadline exhausted')
        return remaining
    def _receive(self, kind, sequence, cap):
        def read():
            try: raw = self.stdout.get(timeout=self._remaining(cap))
            except queue.Empty:
                self._stop_owned('rpc_deadline'); raise worker.WorkerEvidenceError('owned RPC deadline expired')
            try:
                require(not self.errors, 'worker pipe error: ' + str(self.errors[0]) if self.errors else 'worker pipe error')
                require(raw is not None, 'worker exited without expected bounded RPC message')
                self.received += 1; require(self.received <= self.spec['max_messages'], 'RPC message count exceeded')
                def unique(pairs):
                    value = {}
                    for key, item in pairs:
                        require(key not in value, 'duplicate RPC JSON key'); value[key] = item
                    return value
                message = json.loads(raw, object_pairs_hook=unique,
                    parse_constant=lambda _: (_ for _ in ()).throw(worker.WorkerEvidenceError('nonfinite RPC JSON')))
                return validate_message(message, self.binding, expected_kind=kind, expected_sequence=sequence, expected_pid=self.worker_pid)
            except BaseException as error:
                self._attach_rpc_diagnostics(error,raw)
                raise
        return self.budget.run('process_rpc_receive', read, read=self.spec['max_message_bytes'])
    def handshake(self): return self._receive('ready', 0, self.spec['handshake_seconds'])
    def request(self, command, payload):
        require(self._seal is _OWNED_SEAL and not self.closed and type(command) is str and command, 'owned worker/control command required')
        require(type(payload) is dict and not any(key in payload for key in ('approved', 'native_permit', 'training_authorized', 'qualification_authorized')),
                'control payload cannot issue native authority')
        self.sequence += 1; self.sent += 1; require(self.sent <= self.spec['max_messages'], 'RPC send count exceeded')
        message = {'schema': RPC, **self.binding, 'kind': 'control', 'sequence': self.sequence,
                   'worker_pid': self.worker_pid, 'payload': {'command': command, 'data': payload}}
        raw = canonical(message) + b'\n'; require(len(raw) <= self.spec['max_message_bytes'], 'bounded RPC control bytes')
        def write():
            offset = 0
            while offset < len(raw):
                self._remaining(self.spec['rpc_seconds'])
                amount = self.process.stdin.write(raw[offset:])
                require(type(amount) is int and 0 < amount <= len(raw) - offset, 'owned RPC pipe short/failed write')
                offset += amount
            self.process.stdin.flush()
        def bounded_write():
            completion = queue.Queue(maxsize=1)
            def run():
                try: write(); completion.put_nowait(None)
                except BaseException as error: completion.put_nowait(error)
            threading.Thread(target=run, daemon=True).start()
            try: error = completion.get(timeout=self._remaining(self.spec['rpc_seconds']))
            except queue.Empty:
                self._stop_owned('rpc_send_deadline'); raise worker.WorkerEvidenceError('owned RPC write deadline expired')
            if error is not None: raise error
        self.budget.run('process_rpc_send', bounded_write)
        return self._receive('response', self.sequence, self.spec['rpc_seconds'])
    def terminal(self, timeout_seconds):
        def wait():
            import subprocess
            try: self.process.wait(timeout=self._remaining(timeout_seconds))
            except subprocess.TimeoutExpired:
                self._stop_owned('terminal_deadline'); raise worker.WorkerEvidenceError('owned terminal deadline expired')
            facts = self.os.snapshot(self.worker_handle, command=False, terminal=True)
            image = self.identity['image_path'] if facts.get('terminal_identity_only') is True and facts['signaled'] is True else facts['image_path']
            require(facts['pid'] == self.identity['pid'] and facts['creation_filetime_100ns'] == self.identity['creation_filetime_100ns']
                    and Path(image) == Path(self.identity['image_path']) and self.os.boot() == self.boot
                    and facts['signaled'] is True and facts['exit_code'] == self.process.returncode,
                    'terminal evidence must derive from same retained owned OS process')
            return {'schema': 'CE8-OWNED-PROCESS-TERMINAL-1', 'worker_identity_sha256': self.worker_identity_sha256,
                'os_identity': self.identity, 'exit_code': facts['exit_code'], 'exit_filetime_100ns': facts['exit_filetime_100ns'],
                'launcher_identity': self.launcher_identity,
                'request_sha256': self.spec['request_sha256'], 'scope_sha256': self.spec['scope_sha256'],
                'native_qualified': False, 'status_file_used': False}
        return self.budget.run('process_terminal', wait)
    def _stop_owned(self, reason, primary=None):
        require(self._seal is _OWNED_SEAL and self.process is not None, 'only this owned child may be stopped')
        require(self.stop_reservation_id is not None, 'owned stop must be permanently prepaid before launch')
        if self.stop_finished: return
        identifier = self.stop_reservation_id
        failure = None
        try:
            actual_handle = getattr(self, 'worker_handle', None)
            if actual_handle is not None and actual_handle != getattr(self, 'handle', None):
                facts = self.os.snapshot(actual_handle, command=False, terminal=True)
                require(facts['pid'] == self.identity['pid'] and facts['creation_filetime_100ns'] == self.identity['creation_filetime_100ns'],
                        'owned stop worker identity drift')
                if not facts['signaled']:
                    c = self.os.ctypes
                    self.os.kernel.TerminateProcess.argtypes = [c.c_void_p, c.c_uint32]
                    self.os.kernel.TerminateProcess.restype = c.c_int
                    require(bool(self.os.kernel.TerminateProcess(actual_handle, 124)), 'owned actual worker stop failed')
            if self.process.poll() is None:
                self.process.terminate(); self.process.wait(timeout=self.spec['terminate_grace_seconds'])
        except BaseException as error:
            failure = error
        try:
            self.observer('process_owned_stop', 'failed' if failure else 'returned', operation_id=identifier,
                          pid=self.process.pid, reason=reason, error=str(failure)[:1000] if failure else None)
        except BaseException as record_error:
            if failure is not None:
                failure.owned_stop_accounting_failure = {'type': type(record_error).__name__, 'message': str(record_error)[:1000]}
                raise failure from record_error
            failure = record_error
        self.stop_finished = True
        if failure is not None: raise failure
    def close(self):
        require(self._seal is _OWNED_SEAL, 'owned process close only')
        if self.closed: return
        require(self.process.poll() is not None, 'live owned child requires recorded deadline/terminal handling before close')
        primary, extras = None, []
        try:
            if not self.stop_finished: self._stop_owned('normal_terminal_cleanup')
        except BaseException as error: primary = error
        finally:
            releases = [stream.close for stream in (getattr(self.process,'stdin',None),getattr(self.process,'stdout',None),
                getattr(self.process,'stderr',None)) if stream is not None]
            if self.monitor is not None: releases.append(self.monitor.close)
            if self.worker_handle is not None and self.worker_handle != getattr(self,'handle',None):
                releases.append(lambda: self.os.close(self.worker_handle))
            close_popen = getattr(getattr(self.process, '_handle', None), 'Close', None)
            if callable(close_popen): releases.append(close_popen)
            for release in releases:
                try: release()
                except BaseException as error:
                    if primary is None: primary = error
                    else: extras.append({'type': type(error).__name__, 'message': str(error)[:1000]})
            self.closed = True
        if primary is not None:
            primary.owned_close_secondary_failures = extras
            raise primary


def launch_native(spec, permit, observer):
    _admit(spec, permit, observer)
    return OwnedProcess(spec, observer, native=True, permit=permit)


def launch_fixture(spec, observer, resource_clearance):
    validate_spec(spec, fixture=True)
    require(callable(observer) and callable(resource_clearance), 'explicit prepaid observer and root resource clearance required')
    require(resource_clearance() is True, 'root fixture resource clearance refused')
    return OwnedProcess(spec, observer, native=False, _fixture_seal=_FIXTURE_CLEARANCE_SEAL)
