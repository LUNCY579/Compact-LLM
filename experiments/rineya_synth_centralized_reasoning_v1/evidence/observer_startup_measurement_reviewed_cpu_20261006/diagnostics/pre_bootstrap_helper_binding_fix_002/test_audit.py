"""Actual worker observer-only output -> complete independent terminal audit.

Transport/provider/OS/SUP/resources/lease/death facts are explicitly inert CPU
suppliers. Inert DLL bytes are hashed but never loaded. No native qualification.
"""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
from datetime import datetime,timezone
import builtins,hashlib,json,tempfile,time,unittest
import audit as A

HERE=Path(__file__).resolve().parent

def source(path):
    raw=path.read_bytes();module=ModuleType('CPU_observer_'+path.stem);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__);return module
def desc(raw,role=None):
    d={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    if role is not None:d['role']=role
    return d
def sealed(v):return {**v,'sha256':A.digest(v)}

class Transport:
    def __init__(self,binding,fixture):self.binding=binding;self.fixture=fixture;self.rows=[];self.closed=False
    def abi_binding(self):return deepcopy(self.binding)
    def snapshot(self):
        return {**A.ABI_FIXED,'phase':5 if self.closed else 2,'faults':0,'attempted':len(self.rows),'reserved':len(self.rows),'published_prefix':len(self.rows),'inflight':0,'cookie_live':0 if self.closed else 1,'register_status':0,'unregister_status':0 if self.closed else -2**31,'drained':1 if self.closed else 0,'complete':1 if self.closed else 0,'dropped':0}
    def start(self):return 0
    def stop(self):self.closed=True;return 0,self.snapshot()
    def copy_rows(self,count):return deepcopy(self.rows[:count])
    def append(self,reason,handle):
        def u16(s):b=s.encode('utf-16-le');return [int.from_bytes(b[i:i+2],'little') for i in range(0,len(b),2)]
        full=u16(self.fixture);base=u16(Path(self.fixture).name)
        self.rows.append({'sequence':len(self.rows)+1,'base_address':handle,'reason':reason,'image_size':4096,'full_units':len(full),'base_units':len(base),'flags':0,'ready':1,'full_name':full,'base_name':base})

class Provider:
    def __init__(self,baseline,transport,birth):self.rows=deepcopy(baseline);self.transport=transport;self.birth=birth
    def owned_identity(self):return deepcopy(self.birth)
    def modules(self):return {'kind':'OBSERVED_PROCESS_LIBRARY_SNAPSHOT','complete':True,'libraries':[{'path':p,'module_handle':h} for p,h in self.rows.items()]}
    def load_fixture(self,path):self.rows[path]=303;self.transport.append(1,303);return 303
    def unload_fixture(self,handle):self.transport.append(2,handle);del self.rows[self.transport.fixture];return True

class ObserverAuditControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_observer_audit_',dir=HERE);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.package=self.root/'rineya_synth_centralized_reasoning_v1';self.package.mkdir()
        self.raw=self.package/'artifacts/observer_only_CPU';self.raw.mkdir(parents=True);self.preserved=self.package/'evidence/observer_only_CPU';self.preserved.parent.mkdir(parents=True)
        self.sources={}
        for name in ('contract.py','worker.py','audit.py'):
            p=self.package/'observer_startup_measurement_v1'/name;p.parent.mkdir(exist_ok=True)
            raw=(HERE/name).read_bytes();p.write_bytes(raw);self.sources[str(p)]=desc(raw,'python_source')
        self.C=source(HERE/'contract.py');self.W=source(HERE/'worker.py')
        native_supervisor=HERE.parents[1]/'native_supervisor.py';p=self.package/'native_supervisor.py';p.write_bytes(native_supervisor.read_bytes());self.sources[str(p)]=desc(p.read_bytes(),'python_source')
        self.abi=deepcopy(self.C.ABI);self.binaries={}
        for name,role in (('native_library_observer.dll','observer_binary'),('observer_only_fixture.dll','observer_fixture_binary'),('CPU_python.exe','interpreter_binary')):
            p=self.package/name;p.write_bytes(('INERT CPU '+name).encode());self.binaries[str(p)]=desc(p.read_bytes(),role)
        self.observer=str(self.package/'native_library_observer.dll');self.fixture=str(self.package/'observer_only_fixture.dll');self.interpreter=str(self.package/'CPU_python.exe')
        self.birth={'pid':12345,'creation_filetime_100ns':134000000000000001};self.alive=False
        bp=self.package/'CPU_BUILD.json';build=sealed({'kind':'CPU_INJECTED_BUILD_RECEIPT','source_commit':'a'*40,'abi':self.abi});self.write(bp,build)
        registration_observer={'kind':'ROOT_REGISTERED_NATIVE_LIBRARY_OBSERVER_V1','adapter_source':{'path':next(iter(self.sources)),'descriptor':next(iter(self.sources.values()))},'native_source_files':{},'binary':{'path':self.observer,'descriptor':self.binaries[self.observer]},'abi':self.abi,'build_receipt':{'path':str(bp),'file_sha256':hashlib.sha256(bp.read_bytes()).hexdigest(),'body_sha256':build['sha256']}}
        config=sealed({'scope':A.SCOPE,'permission':deepcopy(self.C.PERMISSION),'resources':deepcopy(self.C.RESOURCES),'implementation_commit':'e'*40,'source_files':self.sources,'runtime_files':self.binaries,'observer_registration':registration_observer,'binary_fixture':{'path':self.fixture,'descriptor':self.binaries[self.fixture]},'worker_path':str(self.package/'observer_startup_measurement_v1/worker.py'),'native_launch_authorized':False})
        runtime=sealed({'implementation_commit':'e'*40,'files':self.binaries,'library_event_observer':registration_observer,'allowed_initial_loaded_libraries':[self.interpreter,self.observer,self.fixture]})
        reg=sealed({'implementation_commit':'e'*40,'config_body_sha256':config['sha256'],'runtime_body_sha256':runtime['sha256'],'source_files':self.sources,'runtime_files':self.binaries,'profile_sha256':'b'*64})
        self.value={'config':config,'runtime':runtime,'registration':reg,'expected_head':'a'*40,'config_path':str(self.package/'configs/CPU.json'),'anchors':{'expected_config_file':'c'*64,'expected_config_body':config['sha256'],'expected_runtime_body':runtime['sha256'],'expected_registration_body':reg['sha256']},'fixture_only':True,'native_PASS':False,'native_authority_issued':False}
        self.context=self.C._CPU_context_value_fixture(self.value,_capability=self.C._CPU_capability())
        self.binding={'path':self.observer,'descriptor':self.binaries[self.observer],'abi':self.abi,'dll_handle':202}
        self.transport=Transport(self.binding,self.fixture);self.provider=Provider({self.interpreter:101,self.observer:202},self.transport,self.birth)
        self.attempts=[];old=builtins.__import__
        def imports(name,*args,**kwargs):
            if name.split('.',1)[0] in {'torch','ctypes','numpy','psutil','native_worker','native_process','native_nvml','transformers','safetensors'}:
                self.attempts.append(name);raise AssertionError('native/framework import prohibited: '+name)
            return old(name,*args,**kwargs)
        from unittest.mock import patch
        self.trap=patch.object(builtins,'__import__',side_effect=imports);self.trap.start();self.addCleanup(self.trap.stop);self.addCleanup(lambda:self.assertEqual(self.attempts,[]))

    def write(self,p,value):p.write_bytes((A.canonical(value)+'\n').encode('ascii'))
    def build(self):
        self.result=self.W._CPU_run_fixture(self.context,self.C,self.transport,self.provider,self.raw,_capability=self.C._CPU_capability(),clock=time.monotonic)
        self.result=A.parse((self.raw/'OBSERVER_WORKER_RESULT.json').read_bytes())
        phases=[A.parse(line) for line in (self.raw/'OBSERVER_PHASES.jsonl').read_bytes().splitlines()];first=phases[0]['monotonic'];last=phases[-1]['monotonic'];resume=first-.01
        owner={'worker_pid':self.birth['pid'],'root_worker_pid':self.birth['pid'],'supervisor_pid':12344,'supervisor_create_time':1000.0,'job_name':'Local\\SYNTH_NQ_JOB_'+'1'*32,'config_sha256':'c'*64}
        full={**self.birth,'job_name':owner['job_name'],'assigned_before_resume':True,'job_kill_on_close':True,'breakaway_allowed':False}
        self.write(self.raw/'supervisor_ownership.json',owner);self.write(self.raw/'OWNED_EXTERNAL_ANCHORS.json',{'config_path':self.value['config_path'],'anchors':self.value['anchors']})
        samples=[];clock=resume
        while clock<last:
            samples.extend([{'kind':'CPU','sample_started_monotonic':clock,'host_available_bytes':8*1024**3,'worker_tree_rss_bytes':512*1024**2,'owned_process_identities':[self.birth]}, {'kind':'GPU','gpu_sample_started_monotonic':clock,'gpu_free_bytes':16*1024**3,'gpu_total_bytes':24*1024**3,'gpu_uuid':'CPU_GPU0'}]);clock+=.1
        samples.extend([{'kind':'CPU','sample_started_monotonic':last,'host_available_bytes':8*1024**3,'worker_tree_rss_bytes':512*1024**2,'owned_process_identities':[self.birth]}, {'kind':'GPU','gpu_sample_started_monotonic':last,'gpu_free_bytes':16*1024**3,'gpu_total_bytes':24*1024**3,'gpu_uuid':'CPU_GPU0'}])
        self.write(self.raw/'RESOURCE_OBSERVATIONS.json',samples)
        (self.raw/'worker.stdout.log').write_bytes(b'');(self.raw/'worker.stderr.log').write_bytes(b'');(self.raw/'supervisor_heartbeat.jsonl').write_bytes(b'{}\n')
        lease=self.package/'artifacts/_native_gpu_leases/CPU_OBSERVER';lease.mkdir(parents=True)
        def admission(clock,index):return {'sample_started_monotonic':clock,'sample_completed_monotonic':clock,'host_available_bytes':8*1024**3,'worker_tree_rss_bytes':512*1024**2,'disk_free_bytes':64*1024**3,'gpu_sample_started_monotonic':clock,'gpu_sample_completed_monotonic':clock,'gpu_free_bytes':16*1024**3,'gpu_total_bytes':24*1024**3,'GPU_index':0,'gpu_uuid':'CPU_GPU0','gpu_observation_id':str(index)}
        sup={'status':'SUPERVISED_WORKER_EXITED_ZERO','returncode':0,'worker_attempted':True,'worker_launched':True,'worker_resumed':True,'preflight_revalidated_before_current_observations':True,'job_closed_kill_on_close':True,'sampler_threads_terminal_verified':True,'gpu_telemetry_cleanup_verified':True,'partial_outputs_frozen':True,'owned_worker_identity_currently_alive':False,'unjoined_sampler_threads':[],'worker_identity':full,'owned_process_identities':[self.birth],'output_dir':str(self.raw),'config_sha256':'c'*64,'preflight':{'source_commit':'a'*40,'authorization':{'scope':A.SCOPE},'profile_sha256':'b'*64,'resources':self.C.RESOURCES},'worker_resume_monotonic':resume,'deadline_monotonic':resume+600,'lease_path':str(lease),'admission_observations':[admission(resume-2.04,0),admission(resume-.03,1)],'dispatch_observation':admission(resume-.005,2)}
        sup['partial_outputs']=[{'path':p.name,**desc(p.read_bytes())} for p in sorted(self.raw.iterdir())];self.write(self.raw/'SUPERVISOR_RECEIPT.json',sup)
        leases={'worker.json':full,'reservation.json':{'config_sha256':'c'*64},'resume.json':{'worker_identity':full,'external_monitor_ready_before_resume':True,'resumed_monotonic':resume},'terminal.json':sup}
        for n,v in leases.items():self.write(lease/n,v)
        self.preserve(lease)

    def preserve(self,lease):
        from unittest.mock import patch
        P=source(HERE/'preserve.py')
        with patch.object(self.C,'PACKAGE',self.package):
            self.pres=P.preserve(self.raw,self.preserved,contract=self.C,context=self.context,
                revalidate=lambda:self.C._CPU_revalidate_context(self.context,_capability=self.C._CPU_capability()),fresh_death=lambda _:False)

    def audit(self):
        raw=(self.preserved/'PRESERVATION.json').read_bytes()
        return A._CPU_audit_fixture(self.preserved,original_directory=self.raw,context=self.context,contract=self.C,_capability=self.C._CPU_capability(),expected_preservation_file_sha256=hashlib.sha256(raw).hexdigest(),expected_preservation_body_sha256=self.pres['sha256'],fresh_death=lambda _:self.alive)

    def reseal_result(self,change):
        result=deepcopy(self.result);change(result);result['sha256']=A.digest({k:v for k,v in result.items() if k!='sha256'});return result

    def replay(self,result):return A.replay_observer_records(result,library_files=self.binaries,observer_binding=self.binding,fixture_path=self.fixture,owned_birth=self.birth,source_commit='a'*40,config_sha256='c'*64)

    def test_actual_worker_outputs_flow_full_original_copy_terminal_audit(self):
        self.build();r=self.audit();self.assertTrue(r['observer_mechanics_recorded']);self.assertFalse(r['native_PASS']);self.assertFalse(r['native_observer_qualified']);self.assertFalse(r['tensor_authority'])
        self.assertNotIn(self.fixture,{row['path'] for row in self.result['initial_modules']['libraries']})
    def test_startup_replay_uses_original_baseline_not_prospective_admission_as_truth(self):
        self.build();packet=A.parse((self.raw/'OBSERVER_STARTUP_INVENTORY.json').read_bytes())
        r=A.replay_startup(packet,result=self.result,selected_context=self.value)
        self.assertEqual(r['initial_libraries'],{self.interpreter:101,self.observer:202})
        self.assertNotIn(self.fixture,r['initial_libraries'])
        packet['initial_module_snapshots'][0]['libraries'].append({'path':self.fixture,'module_handle':303})
        packet=sealed({k:v for k,v in packet.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'first complete baseline retained'):
            A.replay_startup(packet,result=self.result,selected_context=self.value)
    def test_missing_or_extra_baseline_library_rejected(self):
        self.build();r=self.reseal_result(lambda r:r['initial_modules']['libraries'].pop())
        with self.assertRaises(ValueError):self.replay(r)
    def test_reordered_or_rewritten_load_unload_events_rejected(self):
        self.build();r=self.reseal_result(lambda r:r['events']['events'].reverse())
        with self.assertRaises(ValueError):self.replay(r)
    def test_inflight_drop_latefault_or_unsealed_cleanup_rejected(self):
        self.build()
        for field in ('inflight','dropped','faults','cookie_live'):
            r=self.reseal_result(lambda r:r['final_collector_snapshot'].__setitem__(field,1))
            with self.subTest(field=field),self.assertRaises(ValueError):self.replay(r)
    def test_phase_cost_or_order_rewrite_rejected(self):
        self.build();r=self.reseal_result(lambda r:r['phase_counts']['fixture_load'].__setitem__('returned',2))
        with self.assertRaises(ValueError):A.replay_phases((self.raw/'OBSERVER_PHASES.jsonl').read_bytes(),result=r,selected_context=self.value)
    def test_original_copy_or_whole_roster_drift_rejected(self):
        self.build();p=self.raw/'OBSERVER_WORKER_RESULT.json';p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaises(ValueError):self.audit()
    def test_source_or_binary_pin_drift_rejected(self):
        self.build();p=Path(self.observer);p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaises(ValueError):self.audit()
    def test_owned_birth_live_or_substituted_rejected(self):
        self.build();self.alive=True
        with self.assertRaises(ValueError):self.audit()
        self.alive=False;r=self.reseal_result(lambda r:r['owned_identity'].__setitem__('creation_filetime_100ns',123))
        with self.assertRaises(ValueError):self.replay(r)


if __name__=='__main__':unittest.main()
