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
    def test_sample_attempt_ceiling_refuses_before_another_resource_query(self):
        from experiments.skill_acquisition_native_v1 import ce_resource_evidence as resource
        spec = resource_spec(); spec['sample_attempts_upper'] = 1
        gate = s.NativeResourceGate(spec, lambda *args, **fields: None)
        sample = {'schema': resource.SCHEMA, 'host_free_bytes': 8 * resource.GIB,
            'worker_peak_working_set_bytes': resource.GIB, 'gpu_free_bytes': resource.GIB,
            'disk_free_bytes': 64 * resource.GIB}
        with patch.object(resource, 'readonly_sample', return_value=sample) as probe:
            gate('configured_runtime')
            with self.assertRaisesRegex(ValueError, 'attempt ceiling'):
                gate('configured_runtime')
            self.assertEqual(probe.call_count, 1)
        self.assertEqual(gate.sample_attempts, 1)

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

    def test_row_outer_reservation_keeps_original_owner_during_q_stage(self):
        events=[]
        bootstrap=SimpleNamespace(operation=lambda kind,phase,**fields: events.append(('bootstrap',kind,phase)))
        def qualification(kind,phase,**fields):
            if kind!='loss_forward': raise AssertionError('metadata reached active Q ledger')
            events.append(('qualification',kind,phase))
        active=SimpleNamespace(scope_role='qualification_only',active=object(),worker_open=True,operation=qualification)
        dispatcher=s.StableDispatcher(bootstrap); dispatcher.attach_qualification_accounting(active)
        dispatcher('outer_reservation','started',operation_id='row-source')
        dispatcher('loss_forward','started',operation_id='native-work')
        active.active=None
        dispatcher('outer_reservation','returned',operation_id='row-source')
        dispatcher('loss_forward','returned',operation_id='native-work')
        self.assertEqual(events,[('bootstrap','outer_reservation','started'),
            ('qualification','loss_forward','started'),('bootstrap','outer_reservation','returned'),
            ('qualification','loss_forward','returned')])
        self.assertNotIn('torch',sys.modules); self.assertNotIn('transformers',sys.modules)

    def test_supervisor_guards_and_release_keep_bootstrap_owner_after_q_end(self):
        from .test_ce_checkpoint_lease import spec,proof,FixtureModule
        from experiments.skill_acquisition_native_v1 import ce_checkpoint_lease as lease
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); bootstrap=s.BootstrapLedger(bootstrap_spec(root/'bootstrap'),'a'*64,synthetic_fixture=True)
            bound=spec(root)
            bootstrap.spec['operation_profiles']['supervisor_lease']=dict(bound['operation_bounds']['transfer'])
            bootstrap.spec['operation_profiles']['supervisor_lease_acquire']=dict(bound['operation_bounds']['acquire'])
            bootstrap.spec['operation_profiles']['supervisor_lease_release']=dict(bound['operation_bounds']['release'])
            # Explicit generous fixture budgets, never native measured prices.
            bootstrap.spec['caps'].update(read_bytes=10**8,write_bytes=10**8,hash_bytes=10**8,metadata_bytes=10**8,hash_attempts=10000,wall_seconds=10000)
            dispatch=s.StableDispatcher(bootstrap)
            context={'schema':lease.FIXTURE,'mode':'synthetic_fixture','approved':False,'training_authorized':False,
                'request_sha256':'a'*64,'scope_sha256':'b'*64,'bindings':{'scope':{'checkpoint_lease':bound}}}
            owner=lease.fixture_supervisor(context,None,proof(111,'c'*64),FixtureModule(root/'rineya-gpu-0.lock'),dispatch)
            owner.acquire(); original_owner=dispatch.owners[owner.release_id][1]
            ended=SimpleNamespace(scope_role='qualification_only',operation=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('Q already ended')))
            dispatch.attach_qualification_accounting(ended)
            owner.assert_parent_use(); owner.close()
            self.assertIs(original_owner,bootstrap); self.assertIs(dispatch.owners[owner.release_id][1],bootstrap)
            self.assertEqual(bootstrap.operations[owner.release_id]['phase'],'returned')
            self.assertFalse(owner.module.LOCK.exists())
            with self.assertRaises(AssertionError): dispatch('unknown_kind','started',operation_id='unknown')

    def test_service_lease_uses_one_lazy_owner_and_preserves_primary_after_release(self):
        from experiments.skill_acquisition_native_v1 import ce_checkpoint_lease as lease
        services=object.__new__(s.NativeControllerServices); services._admit=lambda:None
        services.context={'phase':'qualification','native_execution_permit':object(),'native_services':{}}
        services.operation=SimpleNamespace(ledger=None); services.gate=SimpleNamespace(supervisor=None)
        services.worker_handle=object(); services.supervisor_lease=services.lease_token=None
        calls=[]
        class Owner:
            def __init__(self,*args):
                self.token='one'; self.release_seal=None; self.closed=False; self.backend=None
                self.native=True; self.context=args[0]; self.permit=args[1]; self.operation=args[4]; self.gate=args[5]
                calls.append(('construct',args[3]))
            def acquire(self): self.release_seal=object(); calls.append('acquire')
            def close(self): self.closed=True; calls.append('close'); raise RuntimeError('secondary lease receipt failure')
        original=RuntimeError('primary preparation failure')
        with patch.object(s.authority,'require_execution_permit',return_value=None),patch.object(lease,'SupervisorGpuLease',Owner):
            with self.assertRaisesRegex(RuntimeError,'primary preparation') as caught:
                with services.lease(services.context):
                    self.assertIs(services.context['native_services']['checkpoint_qualification']['handoff'],services.supervisor_lease)
                    raise original
        self.assertIs(caught.exception,original); self.assertEqual(calls,[('construct',None),'acquire','close'])
        self.assertEqual(original.supervisor_lease_cleanup_failure['message'],'secondary lease receipt failure')
        self.assertIsNone(services.lease_token); self.assertIsNone(services.gate.supervisor)

    def test_metadata_backend_cleanup_closes_even_after_service_permission_failure(self):
        from experiments.skill_acquisition_native_v1.ce_native_backend import NativeCEBackend
        services=object.__new__(s.NativeControllerServices); context={}; services.operation=object(); services.gate=object()
        failure=RuntimeError('service expired')
        def refused(): raise failure
        services._admit=refused
        backend=object.__new__(NativeCEBackend); backend.context=context; backend.operation=services.operation; backend.gate=services.gate
        backend.model=backend.optimizer=backend.tokenizer=backend.torch=None; backend.closed=False
        with self.assertRaisesRegex(RuntimeError,'expired') as caught: services.cleanup(context,backend)
        self.assertIs(caught.exception,failure); self.assertTrue(backend.closed)

    def test_worker_os_refresh_keeps_exact_bootstrap_owner_after_q_end(self):
        events=[]
        bootstrap=SimpleNamespace(operation=lambda *args,**kwargs:events.append(('bootstrap',args)))
        dispatch=s.StableDispatcher(bootstrap)
        ended=SimpleNamespace(scope_role='qualification_only',operation=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('ended Q')))
        dispatch.attach_qualification_accounting(ended)
        dispatch('worker_os_binding','started',operation_id='refresh')
        dispatch('worker_os_binding','returned',operation_id='refresh')
        self.assertEqual([owner for owner,_ in events],['bootstrap','bootstrap'])
        self.assertIs(dispatch.owners['refresh'][1],bootstrap)
        for kind in ('worker_handle_close','worker_handle_duplicate'):
            dispatch(kind,'started',operation_id=kind)
            dispatch(kind,'returned',operation_id=kind)
            self.assertIs(dispatch.owners[kind][1],bootstrap)
        self.assertEqual([owner for owner,_ in events],['bootstrap']*6)

    def test_resource_gate_checks_suspended_parent_before_resource_io(self):
        from experiments.skill_acquisition_native_v1 import ce_resource_evidence as r
        gate=s.NativeResourceGate(resource_spec(),Trap())
        def denied(): raise RuntimeError('parent GPU use suspended')
        gate.supervisor=SimpleNamespace(assert_parent_use=denied)
        with patch.object(r,'readonly_sample',side_effect=AssertionError('IO before owner check')):
            with self.assertRaisesRegex(RuntimeError,'suspended'): gate('configured_runtime')

    def test_actual_work_overrun_is_fatal_keeps_pending_reservation(self):
        for field, value in (('wall_s', 2), ('bytes_read', 1001), ('bytes_written', 1001),
                ('identity_hash_attempts_actual', 11), ('identity_hash_bytes_actual', 1001), ('wall_s', True)):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                ledger = s.BootstrapLedger(bootstrap_spec(Path(directory)), 'a' * 64, synthetic_fixture=True)
                ledger.operation('fixture', 'started', operation_id='overrun')
                charged = dict(ledger.counters)
                # Source contract wall validation is a pure ValueError; raw
                # byte/hash checks retain the worker evidence exception.
                refused = ValueError if field == 'wall_s' else s.worker.WorkerEvidenceError
                with self.assertRaisesRegex(refused, 'actual work exceeds'):
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

    def test_resource_samples_keep_bootstrap_active_q_and_between_stage_owners_with_real_guards(self):
        from . import test_ce_qualification_accounting as fixture
        from experiments.skill_acquisition_native_v1 import ce_resource_evidence as resource
        from experiments.skill_acquisition_native_v1 import ce_qualification_accounting as accounting
        with tempfile.TemporaryDirectory(prefix='resource-owner-pure-') as directory:
            root = Path(directory); context, proof, receipt = fixture.fixture(root)
            ledger = accounting.QualificationAccounting(context, None, proof, receipt)
            bootstrap = bootstrap_spec(root / 'service-bootstrap')
            profile = {'read_bytes_upper': 101, 'write_bytes_upper': 0, 'identity_hash_attempts_upper': 1,
                'identity_hash_bytes_upper': 101, 'wall_s_upper': 10}
            bootstrap['operation_profiles']['resource_probe'] = profile
            bootstrap['caps'].update(read_bytes=10**8, write_bytes=10**8, hash_attempts=10**6,
                hash_bytes=10**8, metadata_bytes=10**8, operation_attempts=10000, wall_seconds=10000)
            owner = s.BootstrapLedger(bootstrap, context['request_sha256'], synthetic_fixture=True)
            dispatch = s.StableDispatcher(owner)
            gate_spec = resource_spec(); gate_spec['host_transient_bytes'].update(before_framework=0, after_update=0, final=0)
            gate = s.NativeResourceGate(gate_spec, dispatch)
            serial = []
            def observed(spec, operation, *, barrier=None):
                identifier = 'sample-' + str(len(serial)); serial.append(identifier)
                operation('resource_probe', 'started', operation_id=identifier, barrier=barrier, **spec['probe_bounds'])
                value = {'schema': resource.SCHEMA, 'barrier': barrier, 'resource_probe_operation': identifier,
                    'host_free_bytes': 8 * resource.GIB, 'worker_peak_working_set_bytes': resource.GIB,
                    'gpu_free_bytes': resource.GIB, 'disk_free_bytes': 64 * resource.GIB, 'native_qualified': False,
                    'classification': 'synthetic source sample; no OS measurement'}
                operation('resource_probe', 'returned', operation_id=identifier, barrier=barrier, wall_s=0,
                    bytes_read=0, bytes_written=0, identity_hash_attempts_actual=0, identity_hash_bytes_actual=0,
                    resource_observation=value)
                return value
            with patch.object(resource, 'readonly_sample', side_effect=observed):
                gate.launch(); gate('before_framework')
                recorder = resource.NativeResourceRecorder.fixture_recorder(gate_spec['resource_probe'], dispatch, gate.samples)
                gate.recorder = recorder  # explicit non-native composition fixture
                dispatch.attach_qualification_accounting(ledger)
                gate('configured_runtime')  # between stages: stays bootstrap
                ledger.begin_stage('runtime_anchor', 'fixture/runtime')
                gate('after_update')
                raw_path = root / 'fixture-result.json'; raw_path.write_bytes(b'{"fixture":true}\n')
                import hashlib
                ledger.finish_stage([{'path': str(raw_path), 'sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(), 'bytes': raw_path.stat().st_size}])
                gate('final'); ledger.end(); gate('final')
            self.assertIs(recorder.samples, gate.samples)
            targets = [dispatch.owners[identifier][1] for identifier in serial]
            self.assertEqual(targets, [owner, owner, owner, ledger, owner, owner])
            self.assertEqual([row['barrier'] for row in recorder.samples],
                ['launch', 'before_framework', 'configured_runtime', 'after_update', 'final', 'final'])
            self.assertEqual(recorder.samples[3]['operation_links'][0]['owner_schema'], accounting.SCHEMA)
            self.assertEqual(recorder.samples[-1]['operation_links'][0]['owner_schema'], s.BOOTSTRAP)
            self.assertNotIn(serial[-1], ledger.operations)
            self.assertEqual(ledger.operations[serial[3]]['phase'], 'returned')

    def test_missing_barrier_transient_refuses_before_resource_query(self):
        from experiments.skill_acquisition_native_v1 import ce_resource_evidence as resource
        gate = s.NativeResourceGate(resource_spec(), Trap())
        with patch.object(resource, 'readonly_sample', side_effect=AssertionError('unpriced barrier queried')):
            with self.assertRaisesRegex(ValueError, 'unpriced service host transient'): gate('missing')

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
