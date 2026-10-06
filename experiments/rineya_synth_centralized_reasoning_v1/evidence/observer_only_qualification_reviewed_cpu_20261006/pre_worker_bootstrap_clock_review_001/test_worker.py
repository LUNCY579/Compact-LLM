"""Actual separate observer-only worker body with inert selected bytes/APIs."""
from pathlib import Path
from copy import deepcopy
import builtins, hashlib, tempfile, unittest
from unittest.mock import patch
import worker as W

HERE=Path(__file__).resolve().parent
def descriptor(raw,role):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':role}
def u16(text):
    raw=text.encode('utf-16-le');return [int.from_bytes(raw[i:i+2],'little') for i in range(0,len(raw),2)]

class Transport:
    def __init__(self,binding,fixture):
        self.binding=deepcopy(binding);self.fixture=fixture;self.rows=[];self.closed=False;self.calls=[]
        self.start_status=0;self.stop_status=0;self.poison={};self.rewrite=False
    def abi_binding(self):self.calls.append('binding');return deepcopy(self.binding)
    def snapshot(self):
        self.calls.append('snapshot');n=len(self.rows)
        row={'abi_version':1,'row_bytes':2600,'snapshot_bytes':72,'row_capacity':4096,'full_capacity':1024,'base_capacity':256,
            'phase':5 if self.closed else 2,'faults':0,'attempted':n,'reserved':n,'published_prefix':n,'inflight':0,
            'cookie_live':0 if self.closed else 1,'register_status':0,'unregister_status':0 if self.closed else -2147483648,
            'drained':int(self.closed),'complete':int(self.closed),'dropped':0}
        row.update(self.poison);return row
    def start(self):self.calls.append('start');return self.start_status
    def stop(self):self.calls.append('stop');self.closed=True;return self.stop_status,self.snapshot()
    def copy_rows(self,count):
        self.calls.append('copy');rows=deepcopy(self.rows[:count])
        if self.rewrite:rows[0]['base_address']+=1
        return rows
    def append(self,reason,handle,path=None):
        full=u16(self.fixture if path is None else path);base=u16(Path(self.fixture if path is None else path).name)
        self.rows.append({'sequence':len(self.rows)+1,'base_address':handle,'reason':reason,'image_size':4096,
            'full_units':len(full),'base_units':len(base),'flags':0,'ready':1,'full_name':full,'base_name':base})

class Provider:
    def __init__(self,baseline,transport):
        self.rows=deepcopy(baseline);self.transport=transport;self.calls=[];self.unload_result=True;self.extra=None;self.load_error=None
    def owned_identity(self):self.calls.append('identity');return {'pid':12345,'creation_filetime_100ns':134000000000000001}
    def modules(self):
        self.calls.append('modules');rows=deepcopy(self.rows)
        if self.extra:rows[self.extra]=999
        return {'kind':'OBSERVED_PROCESS_LIBRARY_SNAPSHOT','complete':True,'libraries':[{'path':p,'module_handle':h} for p,h in sorted(rows.items())]}
    def load_fixture(self,path):
        self.calls.append('load')
        if self.load_error:raise self.load_error
        self.rows[path]=303;self.transport.append(1,303);return 303
    def unload_fixture(self,handle):
        self.calls.append('unload')
        if self.unload_result:self.transport.append(2,handle);del self.rows[self.transport.fixture]
        return self.unload_result

def make_cpu_fixture(root):
    root=Path(root).resolve();package=root/'rineya_synth_centralized_reasoning_v1';package.mkdir()
    sources={};binaries={}
    for name in ('worker.py','contract.py'):
        p=package/name;raw=(HERE/name).read_bytes();p.write_bytes(raw);sources[str(p)]=descriptor(raw,'python_source')
    for name,role in (('native_library_observer.dll','observer_binary'),('observer_only_fixture.dll','observer_fixture_binary'),('CPU_python.exe','interpreter_binary')):
        p=package/name;raw=('INERT selected '+name+'\n').encode('ascii');p.write_bytes(raw);binaries[str(p)]=descriptor(raw,role)
    observer=str(package/'native_library_observer.dll');fixture=str(package/'observer_only_fixture.dll');interpreter=str(package/'CPU_python.exe')
    c=W.C;registration={'binary':{'path':observer,'descriptor':binaries[observer]},'abi':deepcopy(c.ABI),
        'adapter_source':{'path':str(package/'worker.py'),'descriptor':sources[str(package/'worker.py')]}}
    config=c.seal({'kind':'REGISTERED_OBSERVER_ONLY_QUALIFICATION_V1','scope':c.SCOPE,'permission':deepcopy(c.PERMISSION),
        'resources':deepcopy(c.RESOURCES),'source_commit':'a'*40,'source_files':sources,'runtime_files':binaries,
        'observer_registration':registration,'binary_fixture':{'path':fixture,'descriptor':binaries[fixture]},
        'worker_path':str(package/'worker.py'),'offline_environment':{'CPU_INERT_OFFLINE':'1'},'native_launch_authorized':False})
    runtime=c.seal({'source_commit':'a'*40,'files':binaries,'library_event_observer':registration,'initial_loaded_libraries':[interpreter,observer]})
    value={'config':config,'runtime':runtime,'registration':{'profile_sha256':'b'*64},'expected_head':'a'*40,
        'config_path':str(package/'configs/CPU_OBSERVER.json'),'anchors':{'expected_config_file':'c'*64},
        'fixture_only':True,'native_PASS':False,'native_authority_issued':False}
    cap=c._CPU_capability();context=c._CPU_context_value_fixture(value,_capability=cap)
    binding={'path':observer,'descriptor':binaries[observer],'abi':deepcopy(c.ABI),'dll_handle':202}
    transport=Transport(binding,fixture);provider=Provider({interpreter:101,observer:202},transport)
    output=root/'output';output.mkdir();return c,context,value,transport,provider,output,cap

class WorkerControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_observer_worker_',dir=HERE);self.addCleanup(self.temp.cleanup)
        self.c,self.context,self.value,self.transport,self.provider,self.output,self.cap=make_cpu_fixture(self.temp.name)
        self.clock_value=100.;self.clock_calls=0
        def clock():self.clock_calls+=1;self.clock_value+=.001;return self.clock_value
        self.clock=clock;self.imports=[];original=builtins.__import__
        def importing(name,*args,**kwargs):
            if name.split('.',1)[0] in {'ctypes','torch','numpy','psutil','safetensors','transformers','native_worker','native_process','native_nvml'}:
                self.imports.append(name);raise AssertionError('native/model imports forbidden')
            return original(name,*args,**kwargs)
        trap=patch.object(builtins,'__import__',side_effect=importing);trap.start();self.addCleanup(trap.stop)
        self.addCleanup(lambda:self.assertEqual(self.imports,[]))
    def run_worker(self):
        return W._CPU_run_fixture(self.context,self.c,self.transport,self.provider,self.output,_capability=self.cap,clock=self.clock)
    def phases(self):return [self.c.parse(line) for line in (self.output/'OBSERVER_PHASES.jsonl').read_bytes().splitlines()]
    def test_actual_twelve_operations_and_complete_exclusive_raw_artifacts(self):
        result=self.run_worker();self.assertEqual(result['status'],'OBSERVER_OPERATIONS_RETURNED_PENDING_TERMINAL_AUDIT')
        self.assertFalse(result['native_PASS']);self.assertFalse(result['native_authority_issued']);self.assertTrue(result['fixture_only'])
        rows=self.phases();self.assertEqual(len(rows),24)
        self.assertEqual([row['operation'] for row in rows[::2]],list(W.OPS))
        self.assertEqual([row['event'] for row in rows],['started','returned']*12)
        self.assertTrue(all(count=={'attempted':1,'returned':1,'failed':0,'pending':0} for count in result['phase_counts'].values()))
        self.assertEqual(result['events']['events'],[{'sequence':1,'event':'load','path':self.transport.fixture,'module_handle':303},
            {'sequence':2,'event':'unload','path':self.transport.fixture,'module_handle':303}])
        self.assertEqual(result['final_collector_snapshot']['phase'],5);self.assertTrue(self.transport.closed)
        self.assertEqual(self.c.parse((self.output/'OBSERVER_WORKER_RESULT.json').read_bytes()),result)
        expected=sum(d['bytes'] for m in (self.value['config']['source_files'],self.value['runtime']['files']) for d in m.values())
        self.assertEqual(rows[1]['data']['verified_bytes'],expected)
        self.assertEqual(rows[-1]['data'],rows[1]['data'])
    def test_forged_capability_and_public_native_gate_refuse_before_providers(self):
        with self.assertRaisesRegex(ValueError,'private observer-only CPU capability'):
            W._CPU_run_fixture(self.context,self.c,self.transport,self.provider,self.output,_capability=object(),clock=self.clock)
        with patch.object(W,'_native_transport',side_effect=AssertionError('must not load observer')):
            with self.assertRaisesRegex(ValueError,'no native authority'):W.main()
        self.assertEqual(self.provider.calls,[]);self.assertEqual(self.transport.calls,[])
    def test_actual_selected_fixture_byte_drift_stops_before_native_load(self):
        Path(self.transport.fixture).write_bytes(b'drift')
        result=self.run_worker();self.assertEqual(result['status'],'OBSERVER_OPERATIONS_FAILED')
        self.assertEqual(self.provider.calls,[]);self.assertEqual(self.transport.calls,[])
        self.assertEqual(result['phase_counts']['source_preflight']['failed'],1)
    def test_incomplete_or_extra_initial_modules_stop_new_work_and_unregister(self):
        self.provider.extra=str(Path(self.transport.fixture).with_name('unregistered.dll'))
        result=self.run_worker();self.assertEqual(result['status'],'OBSERVER_OPERATIONS_FAILED')
        self.assertNotIn('load',self.provider.calls);self.assertEqual(self.transport.calls.count('stop'),1)
        self.assertEqual(result['phase_counts']['initial_module_baseline']['failed'],1)
    def test_failed_fixture_unload_is_not_retried_and_prevents_success(self):
        self.provider.unload_result=False;result=self.run_worker()
        self.assertEqual(result['status'],'OBSERVER_OPERATIONS_FAILED');self.assertEqual(self.provider.calls.count('unload'),1)
        self.assertEqual(self.transport.calls.count('stop'),1);self.assertIsNone(result['events'])
    def test_foreign_handle_rewrite_and_late_callback_poison_cannot_complete(self):
        self.transport.rewrite=True;result=self.run_worker()
        self.assertEqual(result['status'],'OBSERVER_OPERATIONS_FAILED');self.assertEqual(result['phase_counts']['final_event_drain']['failed'],1)
    def test_start_failure_still_attempts_captured_transport_stop_once(self):
        self.transport.start_status=-7;result=self.run_worker()
        self.assertEqual(result['status'],'OBSERVER_OPERATIONS_FAILED');self.assertEqual(self.transport.calls.count('start'),1)
        self.assertEqual(self.transport.calls.count('stop'),1);self.assertNotIn('load',self.provider.calls)
    def test_clock_failure_after_start_cannot_prevent_actual_unregister_cleanup(self):
        def clock():
            self.clock_calls+=1
            if self.clock_calls>=9:raise RuntimeError('inert clock failure')
            return 100.+self.clock_calls*.001
        self.clock=clock;result=self.run_worker()
        self.assertEqual(result['status'],'OBSERVER_OPERATIONS_FAILED');self.assertTrue(result['errors'])
        self.assertEqual(self.transport.calls.count('start'),1);self.assertEqual(self.transport.calls.count('stop'),1)
        self.assertTrue(self.transport.closed);self.assertEqual(result['phase_counts']['observer_stop']['returned'],0)
    def test_preexisting_phase_or_result_files_are_never_overwritten(self):
        prior=b'prior immutable phase bytes\n';path=self.output/'OBSERVER_PHASES.jsonl';path.write_bytes(prior)
        with self.assertRaises(FileExistsError):self.run_worker()
        self.assertEqual(path.read_bytes(),prior);self.assertEqual(self.provider.calls,[])
    def test_unsealed_or_faulted_stop_snapshot_blocks_final_source_success(self):
        original=self.transport.stop
        def stop():status,snap=original();snap['faults']=128;return status,snap
        self.transport.stop=stop;result=self.run_worker()
        self.assertEqual(result['status'],'OBSERVER_OPERATIONS_FAILED');self.assertEqual(result['phase_counts']['observer_stop']['failed'],1)
        self.assertEqual(result['phase_counts']['final_source_preflight']['attempted'],0)

if __name__=='__main__':unittest.main()
