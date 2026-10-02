"""Finite source entry/ownership controls; no native OS, CUDA or model work."""
import copy
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, MappingProxyType
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_dev_entry as e, ce_resource_evidence as resource
from experiments.skill_acquisition_native_v1.tests.test_ce_dev_runtime import context_fixture


def profile():
    return dict(read_bytes_upper=128*e.raw.CHUNK,write_bytes_upper=128*e.raw.CHUNK,
        identity_hash_attempts_upper=10000,identity_hash_bytes_upper=256*e.raw.CHUNK,wall_s_upper=120)


def metadata_owner(directory):
    owner=object.__new__(e.DevEntryOwner)
    owner.role=e.RUN_IDS[0];owner.dev_lease_owner=None;owner.dev_worker_handle=SimpleNamespace(identity={'pid':42})
    owner._admit=lambda:None
    owner.operation=lambda *args,**fields:None
    owner.dev_verify_profile=profile()
    owner.spec={'lease':{'operation_bounds':{name:profile() for name in ('acquire','release','assert')},
        'label':'comp-rs-001-dev-supa-read-1','timeout_seconds':180,'lock_path':str(directory/'lock'),
        'lock_bytes_upper':4096,'helper_ref':{'path':str(directory/'helper.py'),'sha256':'a'*64,'bytes':1}},
        'bootstrap':{'operation_profiles':{'dev_entry_source':profile()}}}
    return owner


class Tests(unittest.TestCase):
    def test_request_argument_cannot_select_another_metadata_path(self):
        role=e.RUN_IDS[0];path=str(Path(tempfile.gettempdir())/'fixed-context.json')
        config={'context_refs':{role:{'path':path}}}
        args=['--request',path,'--read-id',role]
        self.assertEqual(e._fixed_read_id(args),role);e._fixed_request(args,config,role)
        with self.assertRaisesRegex(ValueError,'source-selected'):e._fixed_request(['--request',path+'.other','--read-id',role],config,role)
    def test_default_closed_before_files_and_false_historical_pin(self):
        before=set(sys.modules)&{'torch','numpy','transformers','safetensors'}
        with patch.object(Path,'open',side_effect=AssertionError('unadmitted IO')):
            with self.assertRaisesRegex(ValueError,'entry closed'):e._configuration()
            with self.assertRaisesRegex(ValueError,'type remains closed'):e.DevConstructorPrepayment(object())
        declaration={'configuration_path':str(Path(tempfile.gettempdir())/'config.json'),
            'output_root':tempfile.gettempdir(),'source_checkout':tempfile.gettempdir(),
            'training_selection_sha256':'a'*64,'intake_bounds':dict.fromkeys(e.services.COUNTERS,1)}
        with patch.object(e,'_SOURCE_DECLARATION',MappingProxyType(declaration)),patch.object(e,'_SOURCE_DECLARATION_SHA256',e.digest(declaration)):
            with self.assertRaisesRegex(ValueError,'whole configuration'):e._source_pin()
        self.assertEqual(before,set(sys.modules)&{'torch','numpy','transformers','safetensors'})

    def test_constructor_permanent_receipt_precedes_input_and_is_one_use(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);context=context_fixture();blob=e.canonical(context)+b'\n';path=root/'context.json';path.write_bytes(blob)
            owner=object.__new__(e.DevEntryOwner);owner.role=e.RUN_IDS[0];owner._admit=lambda:None
            ref={'path':str(path),'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}
            owner.config={'context_refs':{owner.role:ref}}
            bounds=dict.fromkeys(e.services.COUNTERS,1024*e.raw.CHUNK);bounds.update(hash_attempts=20000,wall_seconds=120)
            owner.spec={'constructor_bounds':bounds,'bootstrap':{'owned_directory':str(root/'bootstrap'),'setup_bounds':profile()}}
            with patch.object(e.services,'_REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES',frozenset((e.DevConstructorPrepayment,))), \
                    patch.object(e,'_source_pin',return_value=({'output_root':str(root)},None)):
                ticket=e.DevConstructorPrepayment(owner)
                self.assertIsNone(ticket.request_sha)
                self.assertTrue(Path(ticket.reference['path']).is_file())
                receipt=e.raw._json(Path(ticket.reference['path']).read_bytes())
                self.assertEqual(receipt['status'],'PERMANENTLY_RESERVED');self.assertFalse(receipt['model_authority'])
                self.assertEqual(ticket.read_context(),context)
                ticket.require_constructor(e.digest(owner.spec['bootstrap']),context['request_sha256'])
                with self.assertRaisesRegex(ValueError,'one source'):ticket.require_constructor(ticket.spec_sha,ticket.request_sha)
                with self.assertRaises(FileExistsError):e.DevConstructorPrepayment(owner)

    def test_owned_lock_detects_token_pid_marker_and_pointer_changes(self):
        owner=metadata_owner(Path(tempfile.gettempdir()));events=[]
        owner.operation=lambda kind,phase,**fields:events.append((kind,phase))
        record={'pid':42,'token':'own','created':1.25,'label':owner.spec['lease']['label'],'protocol':'GPU-LEASE','worker_pids':[]}
        lease=e._LeaseOwner(owner,None,None,'own',record);owner.dev_lease_owner=lease
        for changed in (dict(record,token='foreign'),dict(record,pid=43),dict(record,created=2.25),dict(record,worker_pids=[99])):
            with patch.object(e.locks,'_read_lock',return_value=(changed,{})):
                with self.assertRaisesRegex(ValueError,'token/PID/create'):lease.assert_owned()
        with patch.object(e.locks,'_read_lock',return_value=(record,{})):lease.assert_owned()
        owner.dev_lease_owner=object()
        with self.assertRaisesRegex(ValueError,'lease pointer'):lease.assert_owned()
        self.assertIn(('dev_entry_lease_assert','failed'),events)

    def test_runtime_lease_entry_failure_never_exits_unentered_manager(self):
        from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as rt
        owner=metadata_owner(Path(tempfile.gettempdir()));events=[]
        owner.operation=lambda kind,phase,**fields:events.append((kind,phase))
        owner.spec['lease']['helper_ref']={'path':str(rt.GPU_LEASE_PATH),'sha256':rt.GPU_LEASE_SHA,'bytes':1}
        module=SimpleNamespace(LOCK=owner.spec['lease']['lock_path']);manager=SimpleNamespace()
        manager.__enter__=lambda:(_ for _ in ()).throw(RuntimeError('acquire denied'))
        manager.__exit__=lambda *args:(_ for _ in ()).throw(AssertionError('unentered exit'))
        with patch.object(e,'_read'),patch.object(rt,'gpu_lease',return_value=module),patch.object(rt,'lease',return_value=manager):
            with self.assertRaisesRegex(RuntimeError,'acquire denied'):
                with owner.lease():pass
        self.assertIn(('dev_entry_lease_release','failed'),events)
        self.assertIsNone(owner.dev_lease_owner)

    def test_release_after_post_acquire_accounting_denial_preserves_primary(self):
        from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as rt
        owner=metadata_owner(Path(tempfile.gettempdir()));closed=[];events=[]
        owner.spec['lease']['helper_ref']={'path':str(rt.GPU_LEASE_PATH),'sha256':rt.GPU_LEASE_SHA,'bytes':1}
        module=SimpleNamespace(LOCK=owner.spec['lease']['lock_path'])
        manager=SimpleNamespace(__enter__=lambda:'own',__exit__=lambda *args:closed.append(args))
        record={'pid':42,'token':'own','created':1.25,'label':owner.spec['lease']['label'],'protocol':'GPU-LEASE','worker_pids':[]}
        def operation(kind,phase,**fields):
            events.append((kind,phase))
            if kind=='dev_entry_lease' and phase=='returned':raise RuntimeError('acquire receipt denied')
        owner.operation=operation
        with patch.object(e,'_read'),patch.object(rt,'gpu_lease',return_value=module),patch.object(rt,'lease',return_value=manager), \
                patch.object(e.locks,'_read_lock',return_value=(record,{})):
            with self.assertRaisesRegex(RuntimeError,'acquire receipt denied'):
                with owner.lease():self.fail('body must not start')
        self.assertEqual(len(closed),1);self.assertIsNone(owner.dev_lease_owner)
        self.assertIn(('dev_entry_lease_release','failed'),events)

    def test_actual_initialized_cuda_peak_is_enforced_and_uninitialized_untouched(self):
        owner=metadata_owner(Path(tempfile.gettempdir()));events=[]
        owner.operation=lambda kind,phase,**fields:events.append((kind,phase,fields))
        limits=dict(launch_host_gib=5.5,ongoing_host_gib=1,max_worker_gib=4.5,gpu_free_gib=.5,
            allocator_fraction=.78,reserved_fraction_limit=.95,disk_floor_bytes=14*resource.GIB,
            launch_disk_bytes=32*resource.GIB,active_storage_reserve_bytes=0)
        sample={'schema':resource.SCHEMA,'host_free_bytes':8*resource.GIB,'worker_peak_working_set_bytes':resource.GIB,
            'gpu_free_bytes':7*resource.GIB,'disk_free_bytes':40*resource.GIB}
        class Gate:
            def __init__(self):self.samples=[]
            def __call__(self,stage):self.samples.append(dict(sample));return dict(sample)
        owner.resource_gate=Gate();owner.spec['resource']={'limits':limits,'host_transient_bytes':{'dev_decode':0}}
        owner.spec['bootstrap']['operation_profiles']['resource_cuda_probe']=profile()
        cuda=SimpleNamespace(is_initialized=lambda:True,mem_get_info=lambda index:(7*resource.GIB,8*resource.GIB),
            device_count=lambda:1,memory_reserved=lambda index:resource.GIB,max_memory_reserved=lambda index:7*resource.GIB)
        with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=cuda)}):
            with self.assertRaisesRegex(ValueError,'allocator peak'):owner._gate('dev_decode')
            cuda.max_memory_reserved=lambda index:resource.GIB
            returned=owner._gate('dev_decode');self.assertEqual(returned['cuda_total_bytes'],8*resource.GIB)
            cuda.is_initialized=lambda:False
            cuda.mem_get_info=lambda index:(_ for _ in ()).throw(AssertionError('early CUDA query'))
            self.assertNotIn('cuda_total_bytes',owner._gate('dev_decode'))
        self.assertIn(('resource_cuda_probe','failed'),[(kind,phase) for kind,phase,_ in events])

    def test_rpc_helper_stdout_is_stderr_and_only_fixed_protocol_messages(self):
        class Pipe:
            def __init__(self):self.messages=[]
            def emit(self,kind,index,data):self.messages.append((kind,index,data))
            def receive(self,index):return {'command':('read' if index==1 else 'finish'),'data':{}}
        pipe=Pipe();stdout=io.StringIO();stderr=io.StringIO()
        def body():
            print('lease helper noise')
            return {'rows':1950,'native_qualified':False,'body_ref':{'path':str(Path(tempfile.gettempdir())/'body.json'),
                'sha256':'a'*64,'bytes':100}}
        with patch.object(sys,'stdout',stdout),patch.object(sys,'stderr',stderr):self.assertEqual(e.serve_read(pipe,{},body),0)
        self.assertEqual(stdout.getvalue(),'');self.assertEqual(stderr.getvalue(),'lease helper noise\n')
        self.assertEqual([(kind,index) for kind,index,_ in pipe.messages],[('ready',0),('response',1),('response',2)])
        self.assertFalse(pipe.messages[1][2]['os_terminal_verified'])

    def test_owned_worker_release_survives_denied_metadata_reservation(self):
        owner=metadata_owner(Path(tempfile.gettempdir()));closed=[]
        class Handle:
            def close(self):closed.append('owned handle closed')
        owner.dev_worker_handle=Handle();owner.spec['bootstrap']['operation_profiles']['dev_entry_metadata']=profile()
        def operation(*args,**fields):raise RuntimeError('metadata allowance exhausted')
        owner.operation=operation
        with patch.object(e.worker,'WinWorkerHandle',Handle):
            with self.assertRaisesRegex(RuntimeError,'allowance exhausted'):owner.close_worker()
        self.assertEqual(closed,['owned handle closed'])


if __name__=='__main__':unittest.main()
