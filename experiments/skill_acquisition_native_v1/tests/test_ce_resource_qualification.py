"""Pure retained-data controls and isolated JSON writer; no actual sampling/ML."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_native_authority as a
from experiments.skill_acquisition_native_v1 import ce_resource_evidence as r
from experiments.skill_acquisition_native_v1 import ce_resource_qualification as q

GPU = 'GPU-12345678-1234-1234-1234-123456789abc'
WORKER = 'a' * 64


def limits():
    return {'launch_host_gib': 5.5, 'ongoing_host_gib': 1, 'max_worker_gib': 4.5,
        'gpu_free_gib': .5, 'allocator_fraction': .78, 'reserved_fraction_limit': .95,
        'disk_floor_bytes': 14*r.GIB, 'launch_disk_bytes': 32*r.GIB, 'active_storage_reserve_bytes': 0}


def samples():
    initial = {'schema': r.SCHEMA, 'stage': 'launch', 'worker_id': 1234, 'gpu_uuid': GPU,
        'perf': 1, 'host_free_bytes': 6*r.GIB, 'worker_peak_working_set_bytes': 2*r.GIB,
        'gpu_total_bytes': 8*r.GIB-4*1024**2, 'gpu_free_bytes': 4*r.GIB, 'disk_free_bytes': 40*r.GIB}
    cuda = dict(initial, stage='resource_io_measurement', perf=2, worker_peak_working_set_bytes=3*r.GIB,
        cuda_total_bytes=8*r.GIB, cuda_peak_reserved_bytes=2*r.GIB, cuda_free_bytes=3*r.GIB)
    return [initial, cuda]


def operation():
    return {'kind': 'resource_probe', 'phase': 'returned', 'worker_id': WORKER,
        'started_seq': 1, 'returned_seq': 2, 'stage': 'runtime_anchor', 'attempt_id': 'anchor-1',
        'profile': {'read_bytes_upper': 99999, 'write_bytes_upper': 99999,
                    'identity_hash_attempts_upper': 99999, 'identity_hash_bytes_upper': 99999},
        'result': {'wall_s': .25, 'bytes_read': 100, 'bytes_written': 0,
                   'identity_hash_attempts_actual': 1, 'identity_hash_bytes_actual': 100}}


def binding():
    return {'stage': 'resource_io_measurement', 'request_sha256': 'b'*64, 'scope_sha256': 'c'*64,
        'static_recipe_sha256': 'd'*64, 'code_sha256': {'resource.py': 'e'*64},
        'worker_identity_sha256': WORKER, 'attempt_id': 'resource-1'}


class ReducerControls(unittest.TestCase):
    def test_exact_cuda_total_is_separate_from_cli_rounding(self):
        observed = q.reduce_samples(samples(), ['launch', 'resource_io_measurement'],
            worker_pid=1234, gpu_uuid=GPU, limits=limits())
        self.assertEqual(observed['gpu_total_gib'], 8)
        self.assertEqual(observed['displayed_gpu_total_bytes'], 8*r.GIB-4*1024**2)
        self.assertEqual(observed['peak_reserved_gib'], 2)
        self.assertEqual(observed['peak_worker_rss_gib'], 3)
        self.assertEqual(observed['min_gpu_free_gib'], 3)
        self.assertIn('no continuous', observed['gpu_observation_scope'])

    def test_missing_barriers_and_foreign_or_regressing_samples_reject(self):
        controls = (lambda value: value.pop(), lambda value: value[1].update(worker_id=1235),
            lambda value: value[1].update(gpu_uuid='GPU-foreign'), lambda value: value[1].update(perf=.5),
            lambda value: value[1].update(perf=True), lambda value: value[1].update(gpu_total_bytes=[]),
            lambda value: value.__setitem__(1, None))
        for mutate in controls:
            value=samples(); mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                q.reduce_samples(value, ['launch', 'resource_io_measurement'], worker_pid=1234, gpu_uuid=GPU, limits=limits())

    def test_inconsistent_cuda_inventory_and_lowered_resource_barriers_reject(self):
        value=samples(); value.append(dict(value[1], perf=3, cuda_total_bytes=9*r.GIB))
        with self.assertRaisesRegex(ValueError, 'CUDA-phase'):
            q.reduce_samples(value, ['launch'], worker_pid=1234, gpu_uuid=GPU, limits=limits())
        for change in ({'launch_host_gib': 5.4}, {'gpu_free_gib': .49}, {'allocator_fraction': .8}):
            bound=limits(); bound.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                q.reduce_samples(samples(), ['launch'], worker_pid=1234, gpu_uuid=GPU, limits=bound)

    def test_actual_counters_are_not_reserved_prices(self):
        value=q.reduce_operations({'probe-1':operation()}, ['resource_probe'], worker_identity_sha256=WORKER)
        row=value['operation_measurements'][0]
        self.assertEqual(row['io_read_bytes'], 100)
        self.assertEqual(row['io_write_bytes'], 0)
        self.assertEqual(row['identity_hash_attempts'], 1)
        self.assertEqual(row['elapsed_seconds'], .25)
        self.assertEqual(row['io_observation_scope'], 'tracked returned counters only')
        scoped=operation(); scoped['result']['io_observation_scope']='Python file IO and hashes; external utility OS IO unmeasured'
        retained=q.reduce_operations({'probe-1':scoped}, ['resource_probe'], worker_identity_sha256=WORKER)
        self.assertEqual(retained['operation_measurements'][0]['io_observation_scope'], scoped['result']['io_observation_scope'])
        self.assertFalse(value['native_qualified']); self.assertFalse(value['execution_verified'])
        for key in operation()['result']:
            op=operation(); del op['result'][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'actual operation'):
                q.reduce_operations({'probe-1':op}, ['resource_probe'], worker_identity_sha256=WORKER)

    def test_failed_pending_foreign_and_inconsistent_aliases_cannot_measure(self):
        controls=(lambda op:op.update(phase='failed'), lambda op:op.update(phase='started'),
            lambda op:op.update(worker_id='f'*64), lambda op:op.update(returned_seq=1),
            lambda op:op['result'].update(io_read_bytes=101), lambda op:op['result'].update(bytes_read=True),
            lambda op:op['result'].update(wall_s=float('nan')))
        for mutate in controls:
            op=operation(); mutate(op)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                q.reduce_operations({'probe-1':op}, ['resource_probe'], worker_identity_sha256=WORKER)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            q.reduce_operations({'probe-1':operation()}, ['resource_probe','model_load'], worker_identity_sha256=WORKER)

    def test_malformed_operations_are_rejected_and_unselected_failures_retained(self):
        for rows,kinds in (({'x':None},['resource_probe']), ({'x':operation()},[{}]),
                           ({'x':operation()},['resource_probe','resource_probe'])):
            with self.subTest(rows=rows,kinds=kinds), self.assertRaises(ValueError):
                q.reduce_operations(rows,kinds,worker_identity_sha256=WORKER)
        rows={'probe-1':operation(),'unknown-1':{'kind':'other', 'phase':'failed'}}
        retained=copy.deepcopy(rows)
        value=q.reduce_operations(rows,['resource_probe'],worker_identity_sha256=WORKER)
        self.assertEqual(rows,retained); self.assertEqual(len(value['operation_measurements']),1)

    def test_binding_checks_full_context_and_active_attempt(self):
        bind=binding()
        context={key:bind[key] for key in ('request_sha256','scope_sha256','static_recipe_sha256')}
        context['bindings']={'source':{'code_sha256':bind['code_sha256']}}
        ledger=SimpleNamespace(active={'stage':bind['stage'],'attempt_id':bind['attempt_id'],'status':'STARTED','extra':'durable'})
        self.assertIs(q.validate_binding(context,{'worker_identity_sha256':WORKER},ledger,bind),bind)
        for key in ('request_sha256','scope_sha256','static_recipe_sha256','worker_identity_sha256'):
            changed=copy.deepcopy(bind); changed[key]='f'*64
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.validate_binding(context,{'worker_identity_sha256':WORKER},ledger,changed)
        ledger.active['status']='RETURNED'
        with self.assertRaises(ValueError): q.validate_binding(context,{'worker_identity_sha256':WORKER},ledger,bind)


class Trap:
    def __getattribute__(self,name): raise AssertionError('native work before authority: '+name)


class SpyDispatcher:
    def __init__(self,ledger): self.ledger=ledger; self.events=[]; self.denied_kind=None
    def __call__(self,kind,phase,**fields):
        self.events.append((kind,phase,fields))
        if kind==self.denied_kind and phase=='started': raise RuntimeError('prepayment denied')


def isolated_writer(root):
    # These exact source objects have only synthetic retained data. Admission is
    # patched locally by writer tests; no native verifier/registry is populated.
    from experiments.skill_acquisition_native_v1.ce_native_backend import NativeCEBackend
    from experiments.skill_acquisition_native_v1.ce_qualification_accounting import QualificationAccounting
    token,bind=object(),binding()
    context={key:bind[key] for key in ('request_sha256','scope_sha256','static_recipe_sha256')}
    spec={'schema':q.SCHEMA,'required_barriers':['launch','resource_io_measurement'],
        'measured_operation_kinds':['resource_probe'],'limits':limits(), 'owned_directory':str(root/'raw'),
        'json_bytes_upper':1024**2,'reducer_work_bytes_upper':4*1024**2}
    context['bindings']={'source':{'code_sha256':bind['code_sha256']},'anchors':{},
        'scope':{'resource_measurement':spec,'native_backend':{'output_directory':str(root)}}}
    ledger=object.__new__(QualificationAccounting)
    ledger.scope_role='qualification_only'; ledger.request_sha256=context['request_sha256']; ledger.scope_sha256=context['scope_sha256']
    ledger.active={'stage':bind['stage'],'attempt_id':bind['attempt_id'],'status':'STARTED'}
    ledger.operations={'probe-1':operation()}
    ledger.fixture=True; ledger.binding_sha256='f'*64; ledger.events=[]; ledger.proof={'fixture':True}
    ledger.cumulative={}; ledger.by_stage={}; ledger.reducer_reserved_bytes=0; ledger.reducer_by_stage={}
    ledger.attempts=[]; ledger.fatal=False; ledger.unknown_wall=False; ledger.worker_open=True
    ledger.rejection_policy=None; ledger.rejection_ticket=None; ledger.rejection_used=False
    ledger.rejection_record=None; ledger.rejection_unknown=None
    ledger.actual_wall_seconds=0; ledger.last_clock=0; ledger.native_activation_gaps=[]
    ledger.bootstrap_handle=SimpleNamespace(evidence={'fixture':True})
    ledger._head=lambda:None; ledger.progress_record=lambda:{'C':0,'U':0}
    ledger.reserve_metadata=lambda *args,**kwargs:None
    dispatcher=SpyDispatcher(ledger)
    backend=object.__new__(NativeCEBackend)
    backend.permit=token; backend.operation=dispatcher
    backend.torch=Trap()
    recorder=object.__new__(r.NativeResourceRecorder)
    recorder.permit=token; recorder.context=context; recorder.operation=dispatcher
    recorder.spec={'gpu_uuid':GPU}; recorder.samples=samples()
    recorder.sample=lambda stage:None
    def fake_gate(stage):
        if stage=='resource_io_measurement':recorder.sample(stage)
    fake_gate.spec={'host_transient_bytes':{'qualification_reducer_overhead':64*r.CHUNK,
        'resource_io_measurement':r.REPORT_TRANSIENT_BYTES,'qualification_artifact':r.REPORT_TRANSIENT_BYTES}}
    backend.gate=fake_gate
    context['native_services']={'resource_recorder':recorder}
    proof={'worker_identity_sha256':WORKER,'os_identity':{'pid':1234},'fresh_process':False,'source_verified':False}
    return context,token,SimpleNamespace(receipt={}),backend,ledger,proof,bind


class AdapterControls(unittest.TestCase):
    def test_counterfeit_permits_refuse_before_any_objects(self):
        for permit in (None,True,{'approved':True},object.__new__(a.NativeExecutionPermit)):
            with self.assertRaises(a.NativeAdmissionError): q.resource_adapter(Trap(),permit,Trap(),Trap(),Trap(),Trap(),Trap())
        self.assertFalse(a._REGISTERED_NATIVE_AUTHORITY_VERIFIERS)
        self.assertFalse(q._REGISTERED_RESOURCE_ADAPTERS)
        self.assertNotIn('torch',sys.modules); self.assertNotIn('transformers',sys.modules)

    def test_artifact_write_readback_is_paid_and_retains_exact_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            args=isolated_writer(Path(temp))
            with patch.object(a,'require_execution_permit',return_value=None): result=q.resource_adapter(*args)
            ref=result['raw_artifacts'][0]; raw=Path(ref['path']).read_bytes(); envelope=json.loads(raw)
            self.assertEqual(ref['sha256'],hashlib.sha256(raw).hexdigest())
            self.assertEqual(ref['binding_sha256'],a.digest(args[-1]))
            self.assertEqual(ref['bytes'],len(raw)); self.assertEqual(envelope['evidence_kind'],'resource_io')
            self.assertEqual(envelope['payload']['raw_operations'],args[4].operations)
            self.assertEqual(envelope['payload']['resource_samples'],args[0]['native_services']['resource_recorder'].samples)
            self.assertFalse(result['native_qualified']); self.assertFalse(result['execution_verified'])
            events=args[3].operation.events
            self.assertEqual([(kind,phase) for kind,phase,_ in events],[
                ('qualification_reducer_overhead','started'),('qualification_reducer_overhead','returned'),
                ('qualification_artifact','started'),('qualification_artifact','returned')])
            self.assertEqual(events[-1][2]['bytes_read'],len(raw)); self.assertEqual(events[-1][2]['bytes_written'],len(raw))
            self.assertEqual(events[-1][2]['identity_hash_attempts_actual'],2)

    def test_final_barrier_is_sampled_before_resource_reduction_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            args = isolated_writer(Path(temp)); labels = []
            args[0]['native_services']['resource_recorder'].sample = labels.append
            with patch.object(a, 'require_execution_permit', return_value=None): q.resource_adapter(*args)
            self.assertEqual(labels, ['final', 'resource_io_measurement'])

    def test_denied_publication_precedes_all_filesystem_work(self):
        with tempfile.TemporaryDirectory() as temp:
            args=isolated_writer(Path(temp)); args[3].operation.denied_kind='qualification_artifact'
            with (patch.object(a,'require_execution_permit',return_value=None),
                  patch.object(Path,'mkdir',side_effect=AssertionError('filesystem before reservation')),
                  patch.object(Path,'resolve',side_effect=AssertionError('filesystem before reservation'))):
                with self.assertRaisesRegex(RuntimeError,'prepayment denied'): q.resource_adapter(*args)
            self.assertEqual(args[3].operation.events[-1][:2],('qualification_artifact','started'))

    def test_short_write_fails_and_retains_prepaid_failure(self):
        class ShortWrite(io.BytesIO):
            def write(self,data): return super().write(data[:-1])
        with tempfile.TemporaryDirectory() as temp:
            args=isolated_writer(Path(temp))
            with (patch.object(a,'require_execution_permit',return_value=None),patch.object(Path,'open',return_value=ShortWrite())):
                with self.assertRaisesRegex(ValueError,'short resource'): q.resource_adapter(*args)
            events=args[3].operation.events
            self.assertEqual(events[-1][:2],('qualification_artifact','failed'))
            self.assertEqual(events[-2][2]['write_bytes_upper'],1024**2)


if __name__=='__main__': unittest.main()
