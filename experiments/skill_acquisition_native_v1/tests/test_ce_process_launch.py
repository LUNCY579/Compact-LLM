"""Pure launch/RPC/owned cleanup tests. No real Popen or child fixture run."""
import ast
import copy
import inspect
import io
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_process_launch as p


class Trap:
    def __getattribute__(self, name): raise AssertionError('work before launch admission')
    def __call__(self, *args, **kwargs): raise AssertionError('work before launch admission')


class OwnedDuplicateRightsTests(unittest.TestCase):
    def test_owned_redirector_target_retains_open_before_return_receipt_failure(self):
        import time
        current = object.__new__(p.OwnedProcess); events = []
        original = RuntimeError('owned target open receipt failed')
        class Kernel:
            def OpenProcess(self, rights, inherit, pid):
                events.append(('open', rights, inherit, pid)); return 501
        class OS:
            kernel = Kernel()
            def close(self, handle): events.append(('close', handle))
        class Budget:
            def run(self, kind, work):
                result = work()
                if kind == 'process_launch_child': raise original
                return result
        current.os, current.budget, current.spec = OS(), Budget(), {'handshake_seconds': 1}
        current.deadline = time.perf_counter() + 2
        with patch.object(p, '_child_pids', return_value=[41]):
            with self.assertRaises(RuntimeError) as caught: current._find_worker({'pid': 40})
        self.assertIs(caught.exception, original)
        self.assertEqual(events, [('open', 0x1000 | 0x100000 | 1 | 0x40, False, 41), ('close', 501)])


def spec():
    root = Path(tempfile.gettempdir()).resolve()
    ref = lambda name: {'path': str(root / name), 'sha256': p.digest(name), 'bytes': 10}
    name = p.FIXTURE_MODULE.replace('.', '/') + '.py'
    return {'schema': p.SPEC, 'mode': 'fixture_only', 'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64,
        'cwd': str(root), 'source_checkout': str(root), 'python_ref': ref('launcher.exe'),
        'process_image_ref': ref('image.exe'), 'request_ref': ref('request.json'),
        'source_files': {name: ref(name)}, 'approved_code_sha256': {name: ref(name)['sha256']},
        'worker_module': p.FIXTURE_MODULE, 'argv': [str(root / 'launcher.exe'), '-m', p.FIXTURE_MODULE, '--request', str(root / 'request.json')],
        'worker_topology': 'venv_redirector',
        'environment': {'PATH': 'fixture'}, 'parent_identity': {'pid': 42, 'creation_filetime_100ns': 100,
            'boot_identity': 'windows-boot-guid-fixture'}, 'deadline_seconds': 30, 'handshake_seconds': 5,
        'rpc_seconds': 5, 'terminate_grace_seconds': 5, 'max_message_bytes': 4096, 'max_messages': 10,
        'max_stderr_bytes': 4096, 'bounds': {'operation_attempts_upper': 100, 'io_read_bytes_upper': 65536,
            'identity_hash_attempts_upper': 100, 'identity_hash_bytes_upper': 65536, 'wall_seconds_upper': 30}}


class ProcessLaunchTests(unittest.TestCase):
    def test_framework_free_and_all_native_authorities_closed(self):
        self.assertNotIn('torch', sys.modules); self.assertNotIn('transformers', sys.modules)
        self.assertFalse(p._REGISTERED_LAUNCH_VERIFIERS)
        self.assertFalse(p._REGISTERED_OBSERVATION_FACTORIES)

    def test_native_flags_or_callbacks_cannot_spawn_or_open_metadata(self):
        with self.assertRaisesRegex(p.worker.WorkerEvidenceError, 'not registered'):
            p.NativeLaunchPermit(Trap(), object(), lambda *args: True, Trap())
        for permit in (None, True, {'approved': True}, object.__new__(p.NativeLaunchPermit)):
            with self.assertRaises(p.worker.WorkerEvidenceError): p.launch_native(Trap(), permit, Trap())

    def test_fixture_requires_resource_clearance_and_cannot_select_native_worker(self):
        value = spec(); p.validate_spec(value, fixture=True)
        with self.assertRaisesRegex(p.worker.WorkerEvidenceError, 'resource clearance'):
            p.launch_fixture(value, Trap(), lambda: False)
        with self.assertRaisesRegex(p.worker.WorkerEvidenceError, 'resource-cleared'):
            p.OwnedProcess(value, Trap(), native=False)
        value['worker_module'] = 'native_unapproved'
        with self.assertRaises(p.worker.WorkerEvidenceError): p.validate_spec(value, fixture=True)

    def test_source_subset_env_callback_or_request_argument_drift_refuses(self):
        for mutate in (lambda v: v['approved_code_sha256'].update(other='f' * 64),
                       lambda v: v['environment'].update(CE8_RPC_BINDING='forged'),
                       lambda v: v['argv'].append('--request'),
                       lambda v: v['argv'].__setitem__(0, v['process_image_ref']['path']),
                       lambda v: v.update(max_messages=True)):
            value = spec(); mutate(value)
            with self.assertRaises(p.worker.WorkerEvidenceError): p.validate_spec(value, fixture=True)

    def test_rpc_session_sequence_pid_and_authority_flags_refuse(self):
        binding = {'session_id': 'session', 'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64}
        message = {'schema': p.RPC, **binding, 'kind': 'response', 'sequence': 1, 'worker_pid': 42, 'payload': {'fixture': True}}
        self.assertEqual(p.validate_message(message, binding, expected_kind='response', expected_sequence=1, expected_pid=42), {'fixture': True})
        for mutate in (lambda m: m.update(session_id='foreign'), lambda m: m.update(sequence=True),
                       lambda m: m.update(worker_pid=43), lambda m: m['payload'].update(native_permit={'approved': True})):
            value = copy.deepcopy(message); mutate(value)
            with self.assertRaises(p.worker.WorkerEvidenceError):
                p.validate_message(value, binding, expected_kind='response', expected_sequence=1, expected_pid=42)

    def test_owned_stop_prepaid_release_survives_return_observer_denial(self):
        value = object.__new__(p.OwnedProcess); events = []
        value._seal, value.stop_reservation_id, value.stop_finished = p._OWNED_SEAL, 'prepaid-ticket', False
        value.spec = {'terminate_grace_seconds': 5}
        value.process = SimpleNamespace(pid=42, poll=lambda: None, terminate=lambda: events.append('terminated'),
                                        wait=lambda **kwargs: events.append('waited'))
        def observer(kind, phase, **fields):
            events.append(phase)
            if phase == 'returned': raise RuntimeError('receipt refused')
        value.observer = observer
        with self.assertRaisesRegex(RuntimeError, 'receipt refused'): value._stop_owned('recorded_deadline')
        self.assertEqual(events, ['terminated', 'waited', 'returned'])
        self.assertTrue(value.stop_finished)
        unowned = object.__new__(p.OwnedProcess); unowned._seal = None
        with self.assertRaises(p.worker.WorkerEvidenceError): unowned._stop_owned('forged')

    def test_rpc_short_writes_are_completed_before_response(self):
        value = object.__new__(p.OwnedProcess); events, written = [], bytearray()
        value._seal, value.closed, value.sequence, value.sent = p._OWNED_SEAL, False, 0, 0
        value.spec = {'max_messages': 10, 'max_message_bytes': 4096, 'rpc_seconds': 5}
        value.binding = {'session_id': 's', 'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64}
        def write(raw):
            count = min(3, len(raw)); written.extend(raw[:count]); return count
        value.process = SimpleNamespace(pid=42, stdin=SimpleNamespace(write=write, flush=lambda: events.append('flushed')))
        value.worker_pid = 42
        value._remaining = lambda cap: cap
        value._receive = lambda *args: events.append('response') or {'fixture': True}
        value.budget = SimpleNamespace(run=lambda kind, callback, **kwargs: callback())
        result = value.request('ping', {})
        self.assertEqual(result, {'fixture': True}); self.assertTrue(written.endswith(b'\n'))
        self.assertEqual(events, ['flushed', 'response'])

    def test_source_popen_is_hidden_shell_free_and_terminal_os_authoritative(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(p.OwnedProcess)))
        popens = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == 'Popen']
        self.assertEqual(len(popens), 1)
        keywords = {arg.arg: arg.value for arg in popens[0].keywords}
        self.assertIsInstance(keywords['shell'], ast.Constant); self.assertFalse(keywords['shell'].value)
        self.assertEqual(keywords['creationflags'].attr, 'CREATE_NO_WINDOW')
        terminal = inspect.getsource(p.OwnedProcess.terminal)
        self.assertIn('signaled', terminal); self.assertIn('creation_filetime_100ns', terminal)
        self.assertIn('exit_code', terminal); self.assertIn('status_file_used', terminal)

    def test_actual_worker_lineage_requires_os_parent_creation_image_and_exact_argv(self):
        value = spec(); boot = value['parent_identity']['boot_identity']
        launcher = {'pid': 42, 'creation_filetime_100ns': 100}
        args = list(value['argv']); args[0] = value['process_image_ref']['path']
        actual = {'pid': 43, 'creation_filetime_100ns': 101, 'image_path': value['process_image_ref']['path'],
                  'argv': args, 'signaled': False}
        self.assertIs(p.validate_worker_lineage(value, launcher, actual, 42, boot), actual)
        for mutate in (lambda a: a.update(pid=42), lambda a: a.update(creation_filetime_100ns=99),
                       lambda a: a.update(image_path=value['python_ref']['path']),
                       lambda a: a['argv'].append('--foreign'), lambda a: a.update(signaled=True)):
            changed = copy.deepcopy(actual); mutate(changed)
            with self.assertRaises(p.worker.WorkerEvidenceError): p.validate_worker_lineage(value, launcher, changed, 42, boot)
        with self.assertRaises(p.worker.WorkerEvidenceError): p.validate_worker_lineage(value, launcher, actual, 99, boot)
        with self.assertRaises(p.worker.WorkerEvidenceError): p.validate_worker_lineage(value, launcher, actual, 42, boot + 'foreign')

    def test_terminal_worker_uses_retained_identity_when_dead_image_query_unavailable(self):
        process = object.__new__(p.OwnedProcess)
        process._seal, process.closed, process.native = p._OWNED_SEAL, False, False
        process.identity = {'pid': 43, 'creation_filetime_100ns': 101, 'image_path': spec()['process_image_ref']['path']}
        process.launcher_identity, process.worker_identity_sha256 = {'pid': 42}, 'c' * 64
        process.boot, process.spec = spec()['parent_identity']['boot_identity'], spec()
        process.worker_handle = object()
        process.process = SimpleNamespace(returncode=0, wait=lambda **kwargs: 0)
        process._remaining = lambda cap: cap
        process.budget = SimpleNamespace(run=lambda kind, callback, **kwargs: callback())
        facts = {'pid': 43, 'creation_filetime_100ns': 101, 'image_path': None, 'terminal_identity_only': True,
                 'signaled': True, 'exit_code': 0, 'exit_filetime_100ns': 102}
        process.os = SimpleNamespace(snapshot=lambda *args, **kwargs: dict(facts), boot=lambda: process.boot)
        self.assertEqual(process.terminal(5)['exit_code'], 0)
        facts['creation_filetime_100ns'] += 1
        with self.assertRaises(p.worker.WorkerEvidenceError): process.terminal(5)

    def test_close_releases_streams_and_child_handle_even_after_stop_query_failure(self):
        process = object.__new__(p.OwnedProcess); events = []
        process._seal, process.closed, process.stop_finished = p._OWNED_SEAL, False, False
        process.handle, process.worker_handle, process.monitor = object(), object(), None
        process.process = SimpleNamespace(poll=lambda: 0,
            stdin=SimpleNamespace(close=lambda: events.append('stdin')),
            stdout=SimpleNamespace(close=lambda: events.append('stdout')),
            stderr=SimpleNamespace(close=lambda: events.append('stderr')))
        process.os = SimpleNamespace(close=lambda handle: events.append('worker_handle'))
        def refuse(reason): raise RuntimeError('primary OS query failure')
        process._stop_owned = refuse
        with self.assertRaisesRegex(RuntimeError, 'primary OS query failure'): process.close()
        self.assertTrue(process.closed)
        self.assertEqual(events, ['stdin', 'stdout', 'stderr', 'worker_handle'])

    def startup_failure(self,returned_observer=False):
        import subprocess
        value=spec();value['worker_topology']='direct'
        events=[];original=RuntimeError('launch returned observer refused') if returned_observer else OSError('QueryFullProcessImageNameW child unavailable')
        class Handle(int):
            def Close(self):events.append('popen_handle_closed')
        class Stream(io.BytesIO):
            def __init__(self,label,blob=b''):super().__init__(blob);self.label=label
            def close(self):events.append(self.label+'_closed');super().close()
        class Process:
            def __init__(self):
                self.pid=43;self._handle=Handle(99);self.returncode=None
                self.stdin=Stream('stdin');self.stdout=Stream('stdout')
                self.stderr=Stream('stderr',b'Worker primary: namespace package origin unavailable: experiments.example\n')
            def poll(self):return self.returncode
            def terminate(self):events.append('owned_terminated');self.returncode=1
            def wait(self,**kwargs):events.append('owned_waited');return self.returncode
        process=Process()
        class OS:
            def open(self,pid):events.append(('parent_open',pid));return 501
            def close(self,handle):events.append(('os_closed',handle))
            def boot(self):return value['parent_identity']['boot_identity']
            def snapshot(self,handle,**kwargs):
                if handle==99:
                    self_ref=current[0]
                    self.assert_readers_started(self_ref)
                    raise original
                return {'pid':42,'creation_filetime_100ns':100}
            def assert_readers_started(self,current):
                assert current.stderr_thread is not None and current.stdout_thread is not None
                events.append('readers_started_before_identity')
        def observer(kind,phase,**fields):
            events.append((kind,phase))
            if returned_observer and kind=='process_launch' and phase=='returned':raise original
        current=[None];owned=object.__new__(p.OwnedProcess);current[0]=owned
        with (patch.object(p.worker,'_file',return_value=None),patch.object(p.worker,'_WindowsOS',return_value=OS()),
              patch.object(p.os,'getpid',return_value=42),patch.object(subprocess,'Popen',return_value=process)):
            with self.assertRaises(type(original)) as caught:
                p.OwnedProcess.__init__(owned,value,observer,native=False,_fixture_seal=p._FIXTURE_CLEARANCE_SEAL)
        self.assertIs(caught.exception,original)
        self.assertIn('namespace package origin unavailable',original.owned_child_stderr)
        self.assertIn('bounded, untrusted diagnostic','\n'.join(original.__notes__))
        self.assertTrue(owned.closed);self.assertFalse(owned.stderr_thread.is_alive());self.assertFalse(owned.stdout_thread.is_alive())
        self.assertIn(('os_closed',501),events)
        for closed in ('stdin_closed','stdout_closed','stderr_closed','popen_handle_closed'):self.assertIn(closed,events)
        self.assertEqual(events.count('owned_terminated'),1)
        self.assertTrue(owned.stop_finished)
        return events

    def test_startup_identity_failure_keeps_bounded_child_stderr_and_closes_owned_partial_resources(self):
        events=self.startup_failure()
        self.assertIn('readers_started_before_identity',events)

    def test_postspawn_return_observer_failure_still_captures_stderr_and_closes_owned_process(self):
        events=self.startup_failure(returned_observer=True)
        self.assertNotIn('readers_started_before_identity',events)

    def test_partial_monitor_worker_handle_and_pipes_close_even_if_monitor_close_fails(self):
        current=object.__new__(p.OwnedProcess);events=[];error=RuntimeError('monitor close receipt failed')
        current._seal,current.closed,current.stop_finished=p._OWNED_SEAL,False,True
        current.handle,current.worker_handle=99,100
        class Handle:
            def Close(self):events.append('popen_handle')
        current.process=SimpleNamespace(poll=lambda:1,_handle=Handle(),stdin=None,
            stdout=SimpleNamespace(close=lambda:events.append('stdout')),stderr=SimpleNamespace(close=lambda:events.append('stderr')))
        def close_monitor():events.append('monitor');raise error
        current.monitor=SimpleNamespace(close=close_monitor);current.os=SimpleNamespace(close=lambda handle:events.append(('worker',handle)))
        with self.assertRaises(RuntimeError) as caught:current.close()
        self.assertIs(caught.exception,error);self.assertTrue(current.closed)
        self.assertEqual(events,['stdout','stderr','monitor',('worker',100),'popen_handle'])

    def test_rpc_eof_preserves_primary_and_attaches_owned_bounded_stderr(self):
        import queue
        import threading
        current=object.__new__(p.OwnedProcess);joins=[]
        current.spec=spec();current.stdout=queue.Queue();current.stdout.put(None);current.errors=[]
        current.stderr_data=bytearray(b'Actual worker failure: NVMLUnknownError before model allocation\n')
        current.stderr_lock=threading.Lock()
        current.stderr_thread=SimpleNamespace(join=lambda timeout:joins.append(timeout),is_alive=lambda:False)
        current._remaining=lambda cap:cap
        current.budget=SimpleNamespace(run=lambda kind,callback,**fields:callback())
        with self.assertRaisesRegex(p.worker.WorkerEvidenceError,'worker exited without expected bounded RPC message') as caught:
            current._receive('response',1,5)
        self.assertIn('NVMLUnknownError',caught.exception.owned_child_stderr)
        self.assertIn('bounded, untrusted diagnostic','\n'.join(caught.exception.__notes__))
        self.assertEqual(joins,[5])


if __name__ == '__main__': unittest.main()
