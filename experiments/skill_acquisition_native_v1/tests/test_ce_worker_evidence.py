"""Pure OS-fact comparisons and closed worker routes; no actual process probe."""
import copy
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import unittest

from experiments.skill_acquisition_native_v1 import ce_worker_evidence as w


class Trap:
    def __getattribute__(self, name): raise AssertionError('OS/IO before metadata admission')
    def __call__(self, *args, **kwargs): raise AssertionError('OS/IO before metadata admission')


def spec():
    root = Path(tempfile.gettempdir()).resolve()
    ref = lambda name: {'path': str(root / name), 'sha256': w.digest(name), 'bytes': 10}
    return {'schema': w.SCOPE, 'mode': 'registered_metadata', 'request_sha256': 'a' * 64,
        'scope_sha256': 'b' * 64, 'expected_pid': 1234, 'expected_creation_filetime_100ns': 130000000000000000,
        'launch_not_before_filetime_100ns': 129999999999999999, 'launch_not_after_filetime_100ns': 130000000000000001,
        'expected_boot_identity': 'windows-boot-guid-00000000-0000-0000-0000-000000000001',
        'source_checkout': str(root), 'python_ref': ref('launcher-python.exe'), 'process_image_ref': ref('base-python.exe'), 'request_ref': ref('request.json'),
        'source_files': {'worker.py': ref('worker.py')}, 'approved_code_sha256': {'worker.py': ref('worker.py')['sha256']}, 'entry_module': 'worker',
        'argv': [str(root / 'base-python.exe'), '-m', 'worker', '--request', str(root / 'request.json')],
        'bounds': {'operation_attempts_upper': 50, 'io_read_bytes_upper': 1000,
                   'identity_hash_attempts_upper': 50, 'identity_hash_bytes_upper': 1000, 'wall_seconds_upper': 30}}


def facts(value):
    return {'pid': value['expected_pid'], 'creation_filetime_100ns': value['expected_creation_filetime_100ns'],
        'image_path': value['process_image_ref']['path'], 'argv': list(value['argv']), 'signaled': False, 'exit_code': 259,
        'exit_filetime_100ns': None}


class WorkerContractTests(unittest.TestCase):
    def terminal_fixture(self, fail_duplicate_return=False):
        value = spec(); actual = facts(value)
        actual.update(signaled=True, exit_code=0, exit_filetime_100ns=actual['creation_filetime_100ns'] + 100)
        closed = []
        api = SimpleNamespace(boot=lambda: value['expected_boot_identity'],
            snapshot=lambda handle, **kwargs: dict(actual), duplicate=lambda handle: 222,
            close=lambda handle: closed.append(handle))
        def observed(kind, phase, **fields):
            if fail_duplicate_return and kind == 'worker_handle_duplicate' and phase == 'returned':
                raise RuntimeError('duplicate receipt failed')
        result = object.__new__(w.WinWorkerHandle)
        result.spec, result.observer, result.native, result._permit = value, observed, False, None
        result.budget, result.os = w._Budget(value, observed), api
        result._seal, result.closed, result.handle = w._HANDLE_SEAL, False, 111
        result.identity = w.compare_identity(value, actual, value['expected_boot_identity'])
        result.sources, result.identity_sha256 = {'fixture': True}, 'f' * 64
        return result, closed

    def test_terminal_clone_keeps_original_os_object_after_source_close(self):
        source, closed = self.terminal_fixture()
        expected = source.terminal_evidence(); clone = source.retain_terminal()
        self.assertIs(clone.budget, source.budget)
        source.close()
        self.assertEqual(clone.terminal_evidence(), expected)
        self.assertEqual(closed, [111]); clone.close(); self.assertEqual(closed, [111, 222])
        with self.assertRaisesRegex(ValueError, 'original retained'):
            source.retain_terminal()

    def test_post_duplicate_receipt_failure_releases_new_handle(self):
        source, closed = self.terminal_fixture(fail_duplicate_return=True)
        with self.assertRaisesRegex(RuntimeError, 'duplicate receipt failed'):
            source.retain_terminal()
        self.assertEqual(closed, [222]); self.assertFalse(source.closed)

    def test_import_framework_free_and_metadata_authority_closed(self):
        self.assertFalse(w._REGISTERED_METADATA_VERIFIERS)
        self.assertNotIn('torch', sys.modules); self.assertNotIn('transformers', sys.modules)

    def test_permit_flags_and_always_true_callbacks_cannot_open_worker(self):
        with self.assertRaisesRegex(w.WorkerEvidenceError, 'no source-registered'):
            w.WorkerObservationPermit(Trap(), object(), lambda *args: True, Trap())
        for permit in (None, {'approved': True}, True, object.__new__(w.WorkerObservationPermit)):
            for invoke in (lambda: w.open_worker(Trap(), permit, Trap()), lambda: w.current_worker_proof(Trap(), permit, Trap())):
                with self.assertRaises(w.WorkerEvidenceError): invoke()

    def test_pid_creation_boot_executable_and_argv_drift_fail_closed(self):
        value = spec(); actual = facts(value)
        identity = w.compare_identity(value, actual, value['expected_boot_identity'])
        self.assertEqual(identity['creation_filetime_100ns'], 130000000000000000)
        for mutate in (lambda a: a.update(pid=1235), lambda a: a.update(pid=True),
                       lambda a: a.update(creation_filetime_100ns=a['creation_filetime_100ns'] + 1),
                       lambda a: a.update(image_path=str(Path(a['image_path']).with_name('foreign.exe'))),
                       lambda a: a['argv'].append('--forged')):
            altered = copy.deepcopy(actual); mutate(altered)
            with self.assertRaises(w.WorkerEvidenceError): w.compare_identity(value, altered, value['expected_boot_identity'])
        with self.assertRaises(w.WorkerEvidenceError): w.compare_identity(value, actual, value['expected_boot_identity'] + 'changed')

    def test_creation_window_and_file_bounds_do_not_accept_booleans(self):
        w.validate_spec(spec())
        for mutate in (lambda v: v.update(expected_creation_filetime_100ns=True),
                       lambda v: v.update(launch_not_before_filetime_100ns=v['expected_creation_filetime_100ns'] + 1),
                       lambda v: v['python_ref'].update(bytes=True),
                       lambda v: v['bounds'].update(operation_attempts_upper=True)):
            value = spec(); mutate(value)
            with self.assertRaises(w.WorkerEvidenceError): w.validate_spec(value)

    def test_source_subset_or_hash_drift_cannot_match_full_approved_bundle(self):
        for mutate in (lambda v: v['approved_code_sha256'].update(other='f' * 64),
                       lambda v: v['source_files']['worker.py'].update(sha256='f' * 64)):
            value = spec(); mutate(value)
            with self.assertRaises(w.WorkerEvidenceError): w.validate_spec(value)

    def test_launcher_and_process_image_are_independent_explicit_bindings(self):
        value = spec(); actual = facts(value)
        self.assertNotEqual(value['python_ref']['path'], value['process_image_ref']['path'])
        w.validate_spec(value)
        w.compare_identity(value, actual, value['expected_boot_identity'])
        value['argv'][0] = value['python_ref']['path']
        with self.assertRaisesRegex(w.WorkerEvidenceError, 'image'): w.validate_spec(value)
        value = spec(); actual['image_path'] = value['python_ref']['path']
        with self.assertRaisesRegex(w.WorkerEvidenceError, 'image'): w.compare_identity(value, actual, value['expected_boot_identity'])
        value = spec(); value.pop('process_image_ref')
        with self.assertRaises(w.WorkerEvidenceError): w.validate_spec(value)

    def test_same_worker_identity_does_not_depend_on_current_or_supervisor_view(self):
        value=spec(); identity=w.compare_identity(value,facts(value),value['expected_boot_identity'])
        core={key:copy.deepcopy(value[key]) for key in ('python_ref','process_image_ref','request_ref','source_files')}
        core.update(entry='worker.py',entry_script_ref=None)
        parent=dict(core,runtime_launcher=None,launcher_provenance='supervisor launch record required',
                    source_observation='initial parent observation')
        child=dict(core,runtime_launcher={'path':value['python_ref']['path'],'source':'current worker sys.executable'},
                   launcher_provenance='current-runtime path plus raw hash',source_observation='initial child observation')
        args=(value['request_sha256'],value['scope_sha256'])
        parent_preimage=w.worker_identity_preimage(identity,parent,*args)
        child_preimage=w.worker_identity_preimage(identity,child,*args)
        self.assertEqual(w.digest(parent_preimage),w.digest(child_preimage))
        self.assertNotEqual(parent,child)  # descriptive evidence remains in each proof
        for key in ('runtime_launcher','launcher_provenance','source_observation'):
            self.assertNotIn(key,parent_preimage['source'])
        child_preimage['source']['python_ref']['path']='mutated returned preimage'
        self.assertEqual(child['python_ref']['path'],value['python_ref']['path'])

    def test_worker_identity_changes_with_os_request_scope_or_any_bound_source(self):
        value=spec(); identity=w.compare_identity(value,facts(value),value['expected_boot_identity'])
        sources={key:copy.deepcopy(value[key]) for key in ('python_ref','process_image_ref','request_ref','source_files')}
        sources.update(entry='worker.py',entry_script_ref=None)
        baseline=w.digest(w.worker_identity_preimage(identity,sources,value['request_sha256'],value['scope_sha256']))
        for key in ('pid','creation_filetime_100ns','boot_identity','image_path','argv'):
            changed=copy.deepcopy(identity)
            changed[key]=changed[key]+1 if type(changed[key]) is int else str(changed[key])+'changed'
            self.assertNotEqual(w.digest(w.worker_identity_preimage(changed,sources,value['request_sha256'],value['scope_sha256'])),baseline)
        for key in ('python_ref','process_image_ref','request_ref'):
            changed=copy.deepcopy(sources); changed[key]['sha256']='f'*64
            self.assertNotEqual(w.digest(w.worker_identity_preimage(identity,changed,value['request_sha256'],value['scope_sha256'])),baseline)
        changed=copy.deepcopy(sources); changed['source_files']['worker.py']['sha256']='f'*64
        self.assertNotEqual(w.digest(w.worker_identity_preimage(identity,changed,value['request_sha256'],value['scope_sha256'])),baseline)
        for args in (('f'*64,value['scope_sha256']),(value['request_sha256'],'f'*64)):
            self.assertNotEqual(w.digest(w.worker_identity_preimage(identity,sources,*args)),baseline)

    def test_status_file_or_probe_flags_cannot_establish_native_process_handle(self):
        for worker in ({'terminal': True, 'exit_code': 0}, True, object(), object.__new__(w.WinWorkerHandle)):
            with self.assertRaises(w.WorkerEvidenceError): w.verify_worker_handle(worker, {'worker_identity_sha256': 'c' * 64})
        worker = object.__new__(w.WinWorkerHandle)
        worker._seal, worker.closed, worker.native = w._HANDLE_SEAL, False, False
        with self.assertRaisesRegex(w.WorkerEvidenceError, 'probe'):
            w.verify_worker_handle(worker, {})

    def test_terminal_requires_signaled_same_retained_identity_and_real_exit_time(self):
        # Pure snapshot spies do not obtain an OS handle or native authority.
        value = spec(); actual = facts(value)
        class Budget:
            def run(self, kind, callback, **kwargs): return callback()
        class OS:
            def boot(self): return value['expected_boot_identity']
            def snapshot(self, handle, **kwargs): return copy.deepcopy(actual)
        worker = object.__new__(w.WinWorkerHandle)
        worker._seal, worker.closed, worker.native = w._HANDLE_SEAL, False, False
        worker.spec, worker.budget, worker.os, worker.handle = value, Budget(), OS(), object()
        worker.identity = w.compare_identity(value, actual, value['expected_boot_identity'])
        worker.identity_sha256 = w.digest(worker.identity)
        with self.assertRaisesRegex(w.WorkerEvidenceError, 'not OS-terminal'): worker.terminal_evidence()
        actual.update(signaled=True, exit_code=259, exit_filetime_100ns=actual['creation_filetime_100ns'] + 10)
        terminal = worker.terminal_evidence()
        self.assertEqual(terminal['exit_code'], 259)  # signaled process may actually exit with259
        self.assertFalse(terminal['status_file_used'])
        actual.update(image_path=None, terminal_identity_only=True)
        terminal = worker.terminal_evidence()
        self.assertEqual(terminal['os_identity']['image_path'], value['process_image_ref']['path'])
        actual['creation_filetime_100ns'] += 1
        with self.assertRaises(w.WorkerEvidenceError): worker.terminal_evidence()

    def test_reservation_denial_precedes_every_source_or_os_callback(self):
        value = spec(); seen = []
        def observer(kind, phase, **fields):
            seen.append(phase)
            if phase == 'started': raise RuntimeError('preflight denied')
        budget = w._Budget(value, observer)
        with self.assertRaisesRegex(RuntimeError, 'preflight denied'):
            budget.run('worker_os_binding', Trap())
        self.assertEqual(seen, ['started'])
        self.assertEqual(budget.remaining['operation_attempts_upper'], 49)

    def test_slow_first_query_cannot_return_past_wall_limit(self):
        events = []
        with patch.object(w.time, 'perf_counter', side_effect=[0, 0, 0, 31, 31]):
            budget = w._Budget(spec(), lambda kind, phase, **fields: events.append(phase))
            with self.assertRaisesRegex(w.WorkerEvidenceError, 'after work'):
                budget.run('worker_os_binding', lambda: 'fixture')
        self.assertEqual(events, ['started', 'failed'])

    def test_live_proof_reobserves_retained_handle_and_rejects_new_terminal_state(self):
        value = spec(); actual = facts(value)
        class Budget:
            def run(self, kind, callback, **kwargs): return callback()
        class OS:
            def boot(self): return value['expected_boot_identity']
            def snapshot(self, handle, **kwargs): return copy.deepcopy(actual)
        worker = object.__new__(w.WinWorkerHandle)
        worker._seal, worker.closed, worker.native = w._HANDLE_SEAL, False, False
        worker.spec, worker.budget, worker.os, worker.handle = value, Budget(), OS(), object()
        worker.identity = w.compare_identity(value, actual, value['expected_boot_identity'])
        worker.identity_sha256, worker.sources = w.digest(worker.identity), {}
        proof = worker.proof()
        self.assertFalse(proof['fresh_process']); self.assertFalse(proof['source_verified'])
        self.assertTrue(proof['raw_source_hashes_verified'])
        actual.update(signaled=True, exit_code=0)
        with self.assertRaisesRegex(w.WorkerEvidenceError, 'no longer live'): worker.proof()

    def test_probe_bootstrap_releases_handle_even_when_close_preflight_refuses(self):
        value = spec(); value.update(schema=w.PROBE, mode='readonly_current_probe')
        events = []
        class OS:
            def open(self, pid): events.append('opened'); return object()
            def snapshot(self, handle):
                result = facts(value); result['pid'] = 42; return result
            def boot(self): return value['expected_boot_identity']
            def close(self, handle): events.append('closed')
        def observe(kind, phase, **fields):
            if kind == 'worker_handle_close' and phase == 'started': raise RuntimeError('close preflight refused')
        with patch.object(w, '_WindowsOS', return_value=OS()), patch.object(w.os, 'getpid', return_value=42):
            with self.assertRaisesRegex(RuntimeError, 'close preflight refused'): w.build_current_probe_spec(value, observe)
        self.assertEqual(events, ['opened', 'closed'])

    def test_open_handle_is_owned_before_returned_observer_can_fail(self):
        value = spec(); handle, closed, just_opened = object(), [], []
        original = RuntimeError('open returned observer failed')
        class OS:
            def open(self, pid): just_opened.append(True); return handle
            def close(self, current): closed.append(current)
        def observe(kind, phase, **fields):
            if phase == 'returned' and just_opened:
                just_opened.clear(); raise original
        current = object.__new__(w.WinWorkerHandle)
        with patch.object(w, '_WindowsOS', return_value=OS()):
            with self.assertRaises(RuntimeError) as caught: current._open(value, observe, native=False)
        self.assertIs(caught.exception, original)
        self.assertEqual(closed, [handle]); self.assertTrue(current.closed)


if __name__ == '__main__': unittest.main()
