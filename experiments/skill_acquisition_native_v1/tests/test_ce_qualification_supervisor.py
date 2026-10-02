"""Admission and retained-handle cleanup controls; no process/native run."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_qualification_supervisor as supervisor
from experiments.skill_acquisition_native_v1 import ce_native_services as services
from .test_ce_native_services import Trap


class SupervisionTests(unittest.TestCase):
    def test_denies_counterfeit_service_before_launch_or_paths(self):
        with patch.object(supervisor.launch, 'launch_native', side_effect=AssertionError('launch before admission')):
            with self.assertRaises(services.worker.WorkerEvidenceError):
                supervisor.supervise_qualification(Trap(), Trap(), Trap())

    def test_actual_service_with_missing_permit_denies_before_launch(self):
        service = object.__new__(services.NativeControllerServices)
        service.permit, service.request, service.spec = None, Trap(), Trap()
        with patch.object(supervisor.launch, 'launch_native', side_effect=AssertionError('launch before admission')):
            with self.assertRaises(services.worker.WorkerEvidenceError):
                supervisor.supervise_qualification(service, Trap(), Trap())

    def test_missing_publication_envelope_budget_denies_before_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'uncreated'
            service = object.__new__(services.NativeControllerServices)
            service.spec = {'bootstrap': {'operation_profiles': {'terminal_publication': {
                'read_bytes_upper': 1, 'write_bytes_upper': 1, 'identity_hash_attempts_upper': 1,
                'identity_hash_bytes_upper': 1, 'wall_s_upper': 1}}}}
            with self.assertRaisesRegex(services.worker.WorkerEvidenceError, 'reserved before paths'):
                supervisor._retain(service, {'owned_directory': str(root), 'max_json_bytes': 1000}, {})
            self.assertFalse(root.exists())

    def test_repeated_entry_admission_reuses_context_and_refreshes_same_worker(self):
        service = object.__new__(services.NativeControllerServices)
        service.request, service.context = {'phase': 'qualification'}, {'retained': 'fixture'}
        events = []; service._admit = lambda: events.append('service')
        service.verify_worker = lambda ctx: events.append(('worker', ctx))
        self.assertIs(service.admit_request(dict(service.request)), service.context)
        self.assertEqual(events, ['service', ('worker', service.context)])
        with self.assertRaisesRegex(services.worker.WorkerEvidenceError, 'request changed'):
            service.admit_request({'phase': 'training'})

    def test_publication_failure_closes_owned_process_and_terminal_clone_preserving_primary(self):
        from types import SimpleNamespace
        events = []; identity = {'pid': 123, 'creation_filetime_100ns': 1, 'boot_identity': 'fixture',
                                 'image_path': 'fixture-python', 'argv': ['fixture-python']}
        live = {'worker_identity_sha256': 'c' * 64}
        terminal = {'exit_code': 0, 'os_identity': identity, 'exit_filetime_100ns': 5}
        retained = SimpleNamespace(close=lambda: events.append('clone_closed'))
        class Monitor:
            def proof(self): return live
            def retain_terminal(self): events.append('retained'); return retained
        class Owned:
            def __init__(self): self.monitor = Monitor()
            def handshake(self): return {'worker_proof': live}
            def request(self, command, payload):
                events.append(command)
                if command == 'finish': return {'finished': True, 'native_qualified': False}
                return {'body_completed': True, 'os_terminal_verified': False, 'native_qualified': False,
                        'raw_artifact': {'path': str(Path(tempfile.gettempdir()) / 'fixture.json'), 'sha256': 'd' * 64, 'bytes': 10}}
            def terminal(self, seconds): events.append('actual_terminal'); return terminal
            def _stop_owned(self, reason, primary): events.append('prepaid_stop')
            def close(self): events.append('owned_closed'); raise RuntimeError('secondary close receipt')
        service = object.__new__(services.NativeControllerServices)
        service.request = {'scope_ref': {'sha256': 'b' * 64}}
        service.spec = {'approved_code_sha256': {'source.py': 'a' * 64}, 'supervision': {
            'schema': supervisor.SPEC, 'owned_directory': str(Path(tempfile.gettempdir())), 'max_json_bytes': 1000}}
        service._admit = lambda: None; service.operation = object(); service.gate = SimpleNamespace(launch=lambda: events.append('launch_floor'))
        service.bootstrap = SimpleNamespace(snapshot=lambda: {'prefix': 'fixture'})
        service.constructor_prepayment = SimpleNamespace(reference={'path': 'fixture'})
        spec = {'worker_module': supervisor.ENTRY_MODULE, 'request_sha256': services.digest(services.controller.request_preimage(service.request)),
                'scope_sha256': 'b' * 64, 'approved_code_sha256': service.spec['approved_code_sha256'], 'deadline_seconds': 1}
        primary = RuntimeError('primary publication failure')
        with patch.object(supervisor.launch, '_admit', return_value=None), \
             patch.object(supervisor.launch, 'OwnedProcess', Owned), \
             patch.object(supervisor.launch, 'launch_native', return_value=Owned()), \
             patch.object(supervisor.worker, 'WinWorkerHandle', Monitor), \
             patch.object(supervisor.worker, 'verify_worker_handle', return_value=terminal), \
             patch.object(supervisor.local, 'validate_parent_view', return_value=None), \
             patch.object(supervisor, '_compare_ready', return_value=None), \
             patch.object(supervisor, '_read_body', return_value={}), \
             patch.object(supervisor, '_retain', side_effect=primary):
            with self.assertRaisesRegex(RuntimeError, 'primary publication') as caught:
                supervisor.supervise_qualification(service, spec, object())
        self.assertIs(caught.exception, primary)
        self.assertEqual(events, ['launch_floor', 'qualify', 'finish', 'actual_terminal', 'retained',
                                  'prepaid_stop', 'owned_closed', 'clone_closed'])

    def test_malformed_ready_stops_still_live_owned_worker_before_close(self):
        from types import SimpleNamespace
        events = []
        class Monitor: pass
        class Owned:
            def __init__(self): self.monitor = Monitor(); self.running = True
            def handshake(self): return {'native_qualified': True}
            def _stop_owned(self, reason, primary): events.append('prepaid_stop'); self.running = False
            def close(self):
                if self.running: raise AssertionError('live worker leaked')
                events.append('closed')
        service = object.__new__(services.NativeControllerServices)
        service.request = {'scope_ref': {'sha256': 'b' * 64}}
        service.spec = {'approved_code_sha256': {'source.py': 'a' * 64}, 'supervision': {
            'schema': supervisor.SPEC, 'owned_directory': str(Path(tempfile.gettempdir())), 'max_json_bytes': 1000}}
        service._admit = lambda: None; service.operation = object(); service.gate = SimpleNamespace(launch=lambda: None)
        spec = {'worker_module': supervisor.ENTRY_MODULE, 'request_sha256': services.digest(services.controller.request_preimage(service.request)),
                'scope_sha256': 'b' * 64, 'approved_code_sha256': service.spec['approved_code_sha256']}
        owned = Owned()
        with patch.object(supervisor.launch, '_admit', return_value=None), \
             patch.object(supervisor.launch, 'OwnedProcess', Owned), \
             patch.object(supervisor.launch, 'launch_native', return_value=owned), \
             patch.object(supervisor.worker, 'WinWorkerHandle', Monitor):
            with self.assertRaisesRegex(services.worker.WorkerEvidenceError, 'ready protocol'):
                supervisor.supervise_qualification(service, spec, object())
        self.assertFalse(owned.running); self.assertEqual(events, ['prepaid_stop', 'closed'])
