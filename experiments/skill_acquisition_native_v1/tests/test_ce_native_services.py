"""Pure service dispatch/resource graph tests; no OS/native/framework execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_native_services as s


class Trap:
    def __getattribute__(self, name): raise AssertionError('work before service authority')
    def __call__(self, *args, **kwargs): raise AssertionError('work before service authority')


def bootstrap_spec(root):
    profile = {'read_bytes_upper': 1000, 'write_bytes_upper': 1000, 'identity_hash_attempts_upper': 10,
               'identity_hash_bytes_upper': 1000, 'wall_s_upper': 1}
    return {'owned_directory': str(root), 'caps': dict(read_bytes=10000, write_bytes=100000,
        hash_attempts=1000, hash_bytes=100000, metadata_bytes=100000, operation_attempts=100, wall_seconds=100),
        'setup_bounds': dict(profile), 'operation_profiles': {'fixture': dict(profile)}}


def resource_spec():
    from experiments.skill_acquisition_native_v1 import ce_resource_evidence as r
    root = Path(tempfile.gettempdir()).resolve()
    return {'limits': {'launch_host_gib': 5.5, 'ongoing_host_gib': 1, 'max_worker_gib': 4.5,
        'gpu_free_gib': .5, 'allocator_fraction': .78, 'reserved_fraction_limit': .95,
        'disk_floor_bytes': 14 * 1024 ** 3, 'launch_disk_bytes': 32 * 1024 ** 3, 'active_storage_reserve_bytes': 0},
        'host_transient_bytes': {'configured_runtime': 0}, 'resource_probe': {'schema': r.SPEC,
            'nvidia_smi_ref': {'path': str(root / 'nvidia-smi.exe'), 'sha256': 'a' * 64, 'bytes': 100},
            'output_root': str(root), 'gpu_uuid': 'GPU-' + '0' * 8 + '-' + '0' * 4 + '-' + '0' * 4 + '-' + '0' * 4 + '-' + '0' * 12,
            'cuda_visible_devices': None, 'query_timeout_seconds': 1,
            'probe_bounds': {'read_bytes_upper': 101, 'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': 101, 'wall_s_upper': 1}}}


class NativeServicesTests(unittest.TestCase):
    def test_closed_service_roles_and_constructor_before_io(self):
        self.assertFalse(s._REGISTERED_SERVICE_VERIFIERS); self.assertFalse(s._REGISTERED_ROLE_FACTORIES)
        self.assertFalse(s._REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES)
        with self.assertRaises(s.worker.WorkerEvidenceError): s.NativeControllerServices(Trap(), None, Trap())
        with self.assertRaises(s.worker.WorkerEvidenceError): s.ControllerServicePermit(Trap(), Trap(), object(), lambda *args: True)
        with self.assertRaises(s.worker.WorkerEvidenceError): s.BootstrapLedger(Trap(), 'a' * 64)
        self.assertNotIn('torch', sys.modules); self.assertNotIn('transformers', sys.modules)

    def test_role_descriptors_cannot_resolve_unregistered_callbacks_or_changed_source(self):
        descriptor = {'module': 'worker', 'symbol': 'factory', 'source_sha256': 'a' * 64}
        with self.assertRaises(s.MissingServiceRole): s.resolve_role(descriptor, {'worker.py': 'a' * 64})
        with self.assertRaises(s.worker.WorkerEvidenceError): s.resolve_role(descriptor, {'worker.py': 'b' * 64})
        with self.assertRaises(s.worker.WorkerEvidenceError): s.resolve_role(lambda *args: True, {})

    def test_dispatch_callback_identity_stable_and_prior_costs_not_migrated(self):
        events = []
        bootstrap = SimpleNamespace(operation=lambda *args, **kwargs: events.append(('bootstrap', args)), snapshot=lambda: {'costs': 3})
        dispatcher = s.StableDispatcher(bootstrap); original = dispatcher
        dispatcher('fixture', 'started', operation_id='before')
        ledger = SimpleNamespace(scope_role='qualification_only', operation=lambda *args, **kwargs: events.append(('qualification', args)), snapshot=lambda: {'costs': 7})
        dispatcher.attach_qualification_accounting(ledger)
        dispatcher('loss_forward', 'started', operation_id='after')
        self.assertIs(dispatcher, original); self.assertIs(dispatcher.bootstrap, bootstrap)
        self.assertEqual([name for name, _ in events], ['bootstrap', 'qualification'])
        self.assertEqual(bootstrap.snapshot(), {'costs': 3})
        dispatcher('fixture', 'returned', operation_id='before')
        dispatcher('loss_forward', 'returned', operation_id='after')
        self.assertEqual([name for name, _ in events], ['bootstrap', 'qualification', 'bootstrap', 'qualification'])
        with self.assertRaisesRegex(s.worker.WorkerEvidenceError, 'identity reused'):
            dispatcher('fixture', 'started', operation_id='before')
        with self.assertRaises(s.worker.WorkerEvidenceError): dispatcher.attach_qualification_accounting(ledger)

    def test_actual_work_overrun_is_fatal_keeps_pending_reservation(self):
        for field, value in (('wall_s', 2), ('bytes_read', 1001), ('bytes_written', 1001),
                ('identity_hash_attempts_actual', 11), ('identity_hash_bytes_actual', 1001), ('wall_s', True)):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                ledger = s.BootstrapLedger(bootstrap_spec(Path(directory)), 'a' * 64, synthetic_fixture=True)
                ledger.operation('fixture', 'started', operation_id='overrun')
                charged = dict(ledger.counters)
                with self.assertRaisesRegex(s.worker.WorkerEvidenceError, 'actual work exceeds'):
                    ledger.operation('fixture', 'returned', operation_id='overrun', **{field: value})
                self.assertTrue(ledger.fatal)
                self.assertEqual(ledger.operations['overrun']['phase'], 'started')
                self.assertEqual(ledger.counters, charged)
                self.assertEqual(ledger.snapshot()['pending'], ['overrun'])

    def test_finalize_worker_failure_closes_local_handle_preserves_primary(self):
        services = object.__new__(s.NativeControllerServices)
        services._admit = lambda: None
        services.operation = lambda *args, **kwargs: None
        services._role = lambda *args: (lambda *args: ({}, object()))
        calls = []; primary = RuntimeError('OS proof failed')
        def fail(): raise primary
        def close(): calls.append('close'); raise RuntimeError('close failed')
        handle = SimpleNamespace(proof=fail, close=close)
        with patch.object(s.worker, 'open_worker', return_value=handle):
            with self.assertRaisesRegex(RuntimeError, 'OS proof failed') as caught:
                services.finalize_worker_context({'fixture': True})
        self.assertIs(caught.exception, primary)
        self.assertEqual(calls, ['close'])
        self.assertEqual(primary.worker_close_failure['message'], 'close failed')

    def test_bootstrap_failed_attempt_remains_charged_and_hash_chain_is_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = s.BootstrapLedger(bootstrap_spec(Path(directory)), 'a' * 64, synthetic_fixture=True)
            before = ledger.snapshot()['cumulative']
            ledger.operation('fixture', 'started', operation_id='one', read_bytes_upper=10)
            charged = ledger.snapshot()['cumulative']
            ledger.operation('fixture', 'failed', operation_id='one', error='fixture')
            self.assertGreater(charged['read_bytes'], before['read_bytes'])
            self.assertEqual(ledger.snapshot()['cumulative']['read_bytes'], charged['read_bytes'])
            rows = [json.loads(raw) for raw in ledger.path.read_bytes().splitlines()]
            previous = None
            for index, row in enumerate(rows):
                body = {key: value for key, value in row.items() if key != 'sha256'}
                self.assertEqual(row['previous'], previous); self.assertEqual(row['seq'], index)
                self.assertEqual(s.digest(body), row['sha256']); previous = row['sha256']

    def test_bootstrap_strict_caps_and_conflicting_aliases_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = bootstrap_spec(Path(directory)); spec['caps']['read_bytes'] = True
            with self.assertRaises(s.worker.WorkerEvidenceError): s.BootstrapLedger(spec, 'a' * 64, synthetic_fixture=True)
        with tempfile.TemporaryDirectory() as directory:
            ledger = s.BootstrapLedger(bootstrap_spec(Path(directory)), 'a' * 64, synthetic_fixture=True)
            with self.assertRaisesRegex(s.worker.WorkerEvidenceError, 'aliases'):
                ledger.operation('fixture', 'started', operation_id='one', read_bytes_upper=10, io_read_bytes_upper=11)
            self.assertNotIn('one', ledger.operations)
            before = dict(ledger.counters)
            with self.assertRaisesRegex(s.worker.WorkerEvidenceError, 'declared wall work'):
                ledger.operation('fixture', 'started', operation_id='wall', wall_s_upper=2)
            self.assertEqual(ledger.counters, before); self.assertNotIn('wall', ledger.operations)

    def test_short_bootstrap_append_is_fatal_and_keeps_prepaid_charges(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = s.BootstrapLedger(bootstrap_spec(Path(directory)), 'a' * 64, synthetic_fixture=True)
            before = dict(ledger.counters)
            class Stream:
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def write(self, raw): return len(raw) - 1
            with patch.object(Path, 'open', return_value=Stream()):
                with self.assertRaisesRegex(s.worker.WorkerEvidenceError, 'short'):
                    ledger.operation('fixture', 'started', operation_id='short')
            self.assertTrue(ledger.fatal)
            self.assertGreater(ledger.counters['write_bytes'], before['write_bytes'])
            with self.assertRaisesRegex(s.worker.WorkerEvidenceError, 'fatal'):
                ledger.operation('fixture', 'started', operation_id='again')

    def test_resource_gate_never_queries_uninitialized_cuda(self):
        from experiments.skill_acquisition_native_v1 import ce_resource_evidence as r
        events = []
        gate = s.NativeResourceGate(resource_spec(), lambda *args, **kwargs: events.append(args))
        gate.framework = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False, mem_get_info=Trap(), max_memory_reserved=Trap()))
        sample = {'schema': r.SCHEMA, 'host_free_bytes': 8 * 1024 ** 3, 'worker_peak_working_set_bytes': 1024 ** 3,
                  'gpu_free_bytes': 1024 ** 3, 'disk_free_bytes': 64 * 1024 ** 3}
        with patch.object(r, 'readonly_sample', return_value=sample): gate('configured_runtime')
        self.assertEqual(events, [])

    def test_resource_barriers_cannot_be_lowered_before_sampling(self):
        changes = {'launch_host_gib': 5, 'ongoing_host_gib': .5, 'max_worker_gib': 5,
                   'gpu_free_gib': .4, 'allocator_fraction': .8, 'reserved_fraction_limit': 1,
                   'disk_floor_bytes': 1, 'launch_disk_bytes': 1}
        for key, value in changes.items():
            spec = resource_spec(); spec['limits'][key] = value
            with self.assertRaises(s.worker.WorkerEvidenceError): s.NativeResourceGate(spec, Trap())

    def test_verify_worker_compares_frozen_context_without_mutation(self):
        services = object.__new__(s.NativeControllerServices)
        services._admit = lambda: None
        proof = {'fresh_process': True, 'source_verified': True, 'worker_id': 'fixture-worker'}
        context = {'worker_proof_sha256': s.digest(proof)}; before = copy.deepcopy(context)
        services.worker_handle = SimpleNamespace(proof=lambda: copy.deepcopy(proof))
        self.assertEqual(services.verify_worker(context), proof); self.assertEqual(context, before)
        proof['worker_id'] = 'changed'
        with self.assertRaises(s.worker.WorkerEvidenceError): services.verify_worker(context)
        self.assertEqual(context, before)


if __name__ == '__main__': unittest.main()
