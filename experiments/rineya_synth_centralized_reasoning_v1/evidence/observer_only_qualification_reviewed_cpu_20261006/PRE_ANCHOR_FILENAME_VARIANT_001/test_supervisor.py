"""Exact original supervisor body with explicit inert lifecycle facade only."""
from pathlib import Path
from types import ModuleType
import builtins,hashlib,tempfile,unittest
from unittest.mock import patch
import supervisor as S

def contract():
    path=Path(__file__).with_name('contract.py');module=ModuleType('CPU_observer_only_C');module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__);return module

class Host:
    class LaunchFailure(RuntimeError):pass
    def __init__(self,root):
        self.root=Path(root);self.now=100.;self.calls=[];self.writes={};self.dead=False;self.closed=False
        self.nvml_closed=False;self.joined=False;self.fail=None;self.low=False;self.stale=False
        self.pre_resume_failure=None;self.returncode=0;self.pending=[];self.alive_after=False
        self.identity={'pid':101,'creation_filetime_100ns':12345678};self.assigned=False;self.resumed=False
    def clock(self):return self.now
    def sleep(self,seconds):self.now+=seconds
    def utc(self):return 'CPU_INERT_TIME'
    def new_output(self,output):self.calls.append('new_output');return Path(output)
    def write_new(self,path,value):
        name=str(path);assert name not in self.writes;self.writes[name]=value
    def acquire_lease(self,output,pin):self.calls.append('lease');return self.root/'lease'
    def release_lease(self,*args):self.calls.append('lease_release')
    def observation(self,output):
        self.calls.append('admission');t=self.now-(1 if self.stale else 0)
        return {'host_available_bytes':int(4*1024**3),'disk_free_bytes':4*1024**3,'worker_tree_rss_bytes':0,
          'sample_started_monotonic':t,'sample_completed_monotonic':t,
          'GPU_index':0,'gpu_total_bytes':8*1024**3,'gpu_free_bytes':(1 if self.low else 4)*1024**3,
          'gpu_sample_started_monotonic':self.now,'gpu_sample_completed_monotonic':self.now,
          'gpu_uuid':'CPU_INERT_GPU','gpu_observation_id':'CPU_'+str(self.now)}
    def cpu_observation(self,output):return {key:value for key,value in self.observation(output).items() if not key.startswith(('gpu_','GPU_'))}
    def worker(self,argv,output,**kwargs):
        self.calls.append('created_suspended');self.argv=argv
        host=self
        class Worker:
            identity=host.identity
            def is_alive(self):return not host.dead
            def resume(self):
                assert host.assigned;host.calls.append('resume');host.resumed=True
            def poll(self):return host.returncode
            def process_identities(self):return []
            def terminate(self):host.calls.append('terminate_owned');host.dead=True
            def close(self):host.calls.append('Job_kill_on_close');host.dead=True;host.closed=True
        self.assigned=True;self.calls.append('Job_assigned');return Worker()
    def monitor(self,worker,output,deadline,observation):
        host=self
        class Monitor:
            unjoined_threads=[]
            def start(self):host.calls.append('samplers_started')
            def pre_resume_check(self):return host.pre_resume_failure
            def check(self):return host.fail
            def _abort(self,reason):host.calls.append('abort_owned');worker.terminate()
            def close(self):host.calls.append('samplers_joined');host.joined=True;return {'CPU':[],'GPU':[],'fixture_only':True}
        return Monitor()
    def close_NVML(self):self.calls.append('NVML_closed');self.nvml_closed=True
    def pending_threads(self):return self.pending
    def current_alive(self,birth):assert birth==self.identity;return self.alive_after
    def inventory(self,output):return []

class ObserverSupervisorControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_observer_supervisor_');self.addCleanup(self.temp.cleanup)
        self.C=contract();self.host=Host(self.temp.name);self.output=Path(self.temp.name)/'output';self.output.mkdir()
        self.value={'config_path':str(Path(self.temp.name)/'config.json'),'expected_head':'a'*40,
          'config':{'kind':'REGISTERED_OBSERVER_ONLY_QUALIFICATION_V1','worker_path':str(self.C.PACKAGE/S.WORKER_RELATIVE),
                    'resources':dict(self.C.RESOURCES),'source_files':{}},
          'runtime':{'files':{}},'registration':{'profile_sha256':'b'*64},
          'anchors':{'expected_config_file':'c'*64},'fixture_only':True,'native_PASS':False,'native_authority_issued':False}
        self.context=self.C._CPU_context_value_fixture(self.value,_capability=self.C._CPU_capability())
        self.imports=[];original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.',1)[0] in {'ctypes','torch','psutil','native_worker','native_supervisor','native_process','native_nvml'}:
                self.imports.append(name);raise AssertionError('actual native import forbidden')
            return original(name,*args,**kwargs)
        p=patch.object(builtins,'__import__',side_effect=guarded);p.start();self.addCleanup(p.stop)
        self.addCleanup(lambda:self.assertEqual(self.imports,[]))
    def run_body(self,**kwargs):
        return S._CPU_supervise_fixture(self.context,self.host,self.output,contract=self.C,_capability=S._CPU_capability(),**kwargs)
    def test_exact_original_shared_lifecycle_scope_three_changes_and_cleanup(self):
        result=self.run_body();receipt=result['receipt']
        self.assertEqual(receipt['status'],'SUPERVISED_WORKER_EXITED_ZERO')
        self.assertEqual(result['source_adaptation']['mutation_count'],3)
        self.assertEqual(receipt['bounds'],S.BOUNDS);self.assertEqual(receipt['deadline_monotonic'],700.)
        self.assertLess(self.host.calls.index('Job_assigned'),self.host.calls.index('resume'))
        self.assertLess(self.host.calls.index('samplers_started'),self.host.calls.index('resume'))
        self.assertTrue(receipt['sampler_threads_terminal_verified']);self.assertTrue(receipt['gpu_telemetry_cleanup_verified'])
        self.assertTrue(receipt['job_closed_kill_on_close']);self.assertFalse(receipt['owned_worker_identity_currently_alive'])
        self.assertEqual(receipt,self.host.writes[str(self.output/'SUPERVISOR_RECEIPT.json')])
        self.assertEqual(receipt,self.host.writes[str(self.host.root/'lease/terminal.json')])
        self.assertEqual(self.host.writes[str(self.output/'OBSERVER_EXTERNAL_ANCHORS.json')],
                         {'config_path':self.value['config_path'],'anchors':self.value['anchors']})
        self.assertFalse(result['native_PASS']);self.assertTrue(result['fixture_only'])
    def test_admission_denied_floor_no_worker_no_resume(self):
        self.host.low=True;result=self.run_body()
        self.assertEqual(result['receipt']['status'],'ADMISSION_DENIED')
        self.assertFalse(result['receipt']['worker_attempted']);self.assertNotIn('created_suspended',self.host.calls)
        self.assertTrue(self.host.nvml_closed)
    def test_stale_CPU_admission_no_worker(self):
        self.host.stale=True;result=self.run_body()
        self.assertEqual(result['receipt']['status'],'ADMISSION_DENIED');self.assertNotIn('resume',self.host.calls)
    def test_pre_resume_failure_still_kills_only_owned_Job_and_joins(self):
        self.host.pre_resume_failure='CPU_current_floor';result=self.run_body()
        self.assertEqual(result['receipt']['status'],'OWNED_WORKER_ABORTED');self.assertNotIn('resume',self.host.calls)
        self.assertIn('Job_kill_on_close',self.host.calls);self.assertTrue(self.host.joined)
    def test_worker_failure_status_kept_no_success(self):
        self.host.returncode=9;result=self.run_body()
        self.assertEqual(result['receipt']['status'],'SUPERVISED_WORKER_FAILED')
        self.assertEqual(result['receipt']['returncode'],9)
    def test_unjoined_native_call_refuses_frozen_terminal(self):
        self.host.pending=[{'name':'CPU_fake_pending','ident':1,'operation':'CPU_fake'}]
        result=self.run_body();self.assertEqual(result['receipt']['status'],'SUPERVISOR_SHUTDOWN_UNVERIFIED')
        self.assertFalse(result['receipt']['partial_outputs_frozen'])
    def test_original_source_hash_mutation_rejected_before_compile(self):
        raw=S.read_source(S.PACKAGE/'native_supervisor.py')
        with self.assertRaisesRegex(ValueError,'original supervisor source SHA'):S.adapted_body(b'#'+raw[1:],{})
    def test_public_native_and_forged_CPU_capability_refuse_before_lifecycle(self):
        with self.assertRaisesRegex(ValueError,'no native authority'):S.supervise('unused','unused')
        with self.assertRaisesRegex(ValueError,'private CPU supervisor capability'):
            S._CPU_supervise_fixture(self.context,self.host,self.output,contract=self.C,_capability=object())
        self.assertEqual(self.host.calls,[])

if __name__=='__main__':unittest.main()
