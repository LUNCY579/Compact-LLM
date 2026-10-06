"""Original SUP admission denial -> actual driver failure -> exact lease copies.

Host/resource/clock/lease suppliers are inert CPU fixtures. No native helpers,
WinAPI, framework, model, binary loader or process launch is imported/executed.
"""
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from copy import deepcopy
import ast,builtins,hashlib,sys,tempfile,unittest

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
PUBLIC=PACKAGE/'observer_startup_measurement_v1'

def source(path,name):
    module=ModuleType(name);module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__);return module

class DeniedControls(unittest.TestCase):
    def setUp(self):
        self.imports=[];original=builtins.__import__
        def imports(name,*args,**kwargs):
            if name.split('.',1)[0] in {'ctypes','torch','psutil','native_worker','native_supervisor','native_process','native_nvml','numpy','transformers','safetensors'}:
                self.imports.append(name);raise AssertionError('actual native/framework import forbidden: '+name)
            return original(name,*args,**kwargs)
        trap=patch.object(builtins,'__import__',side_effect=imports);trap.start();self.addCleanup(trap.stop)
        self.addCleanup(lambda:self.assertEqual(self.imports,[]))
        self.C=source(PUBLIC/'contract.py','denied_CPU_C');self.S=source(PUBLIC/'supervisor.py','denied_CPU_S')
        self.P=source(PUBLIC/'preserve.py','denied_CPU_P');self.A=source(PUBLIC/'audit.py','denied_CPU_A')
        self.D=source(HERE/'launch_driver.py','denied_CPU_D')
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_denied_',dir=HERE);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.raw=self.root/'attempt';self.target=self.root/'attempt_preserved';self.receipt=self.root/'driver.json'
        self.source_path=self.root/'selected_inert_source.py';raw=b'value="inert selected source"\n';self.source_path.write_bytes(raw)
        descriptors={str(self.source_path):{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}}
        self.value={'config_path':str(self.root/'config.json'),'expected_head':'a'*40,'config':{'kind':'REGISTERED_OBSERVER_STARTUP_MEASUREMENT_V1','scope':self.C.SCOPE,'worker_path':str(self.C.PACKAGE/self.S.WORKER_RELATIVE),'resources':dict(self.C.RESOURCES),'source_files':descriptors},'runtime':{'files':{}},'registration':{'profile_sha256':'b'*64},'anchors':{'expected_config_file':'c'*64},'fixture_only':True,'native_PASS':False,'native_authority_issued':False}
        self.context=self.C._CPU_context_value_fixture(self.value,_capability=self.C._CPU_capability())
        self.host=self.host_fixture();self.deaths=[]
        def death(birth):self.deaths.append(birth);raise AssertionError('denied run has no owned birth to verify')
        self.death=death
    def host_fixture(self):
        outer=self;c=self.C
        class Host:
            class LaunchFailure(RuntimeError):pass
            def __init__(self):self.now=100.;self.calls=[];self.nvml_closed=False;self.lease=None
            def clock(self):return self.now
            def sleep(self,seconds):self.now+=seconds
            def utc(self):return 'CPU_INERT_TIME'
            def new_output(self,output):Path(output).mkdir();self.calls.append('new_output');return Path(output)
            def write_new(self,path,value):
                with path.open('xb') as stream:stream.write((c.canonical(value)+'\n').encode('ascii'))
            def acquire_lease(self,output,pin):
                self.calls.append('original_lease_supplier');self.lease=output.parent/'_native_gpu_leases/CPU_DENIED';self.lease.mkdir(parents=True)
                self.write_new(self.lease/'reservation.json',{'schema_version':1,'lease_id':'CPU_DENIED','supervisor':{'pid':4242,'create_time':1000.},'config_sha256':pin,'output_dir':str(output)})
                return self.lease
            def release_lease(self,*args):self.calls.append('lease_release')
            def observation(self,output):
                self.calls.append('inert_admission_observation')
                return {'host_available_bytes':8*1024**3,'disk_free_bytes':8*1024**3,'worker_tree_rss_bytes':0,'sample_started_monotonic':self.now,'sample_completed_monotonic':self.now,'GPU_index':0,'gpu_total_bytes':8*1024**3,'gpu_free_bytes':1*1024**3,'gpu_sample_started_monotonic':self.now,'gpu_sample_completed_monotonic':self.now,'gpu_uuid':'CPU_INERT_GPU','gpu_observation_id':'CPU_INERT_ONE'}
            def cpu_observation(self,output):raise AssertionError('denied before dispatch CPU sample')
            def worker(self,*args,**kwargs):raise AssertionError('denied before any worker construction')
            def monitor(self,*args,**kwargs):raise AssertionError('denied before any monitor construction')
            def close_NVML(self):self.calls.append('inert_NVML_close');self.nvml_closed=True
            def pending_threads(self):return []
            def current_alive(self,*args):raise AssertionError('no worker birth in denied run')
            def inventory(self,output):return [{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.iterdir())]
        return Host()
    def supervise(self):return self.S._CPU_supervise_fixture(self.context,self.host,self.raw,contract=self.C,_capability=self.S._CPU_capability())
    def run_driver(self,supervise=None):
        return self.D._CPU_run_fixture(self.context,self.C,self.raw,self.target,self.receipt,_capability=self.C._CPU_capability(),supervise=supervise or self.supervise,preservation_module=self.P,audit_module=self.A,death=self.death)
    def test_original_SUP_denied_body_actual_driver_failure_copies_actual_lease(self):
        with self.assertRaisesRegex(ValueError,'exact original terminal lease population'):self.run_driver()
        parent=self.C.parse(self.receipt.read_bytes());snapshot=self.C.parse((self.target/'DRIVER_FAILURE_SNAPSHOT.json').read_bytes())
        sup=self.C.parse((self.raw/'SUPERVISOR_RECEIPT.json').read_bytes());self.assertEqual(sup['status'],'ADMISSION_DENIED')
        self.assertEqual(sup['bounds'],self.S.BOUNDS);self.assertEqual(sup['deadline_monotonic'],700.)
        self.assertIn('GPU free',sup['failure']);self.assertTrue(self.host.nvml_closed)
        self.assertEqual(self.host.calls,['new_output','original_lease_supplier','inert_admission_observation','inert_NVML_close'])
        self.assertEqual([e['stage'] for e in parent['errors']],['preservation','terminal_audit'])
        self.assertIsNone(parent['preservation']);self.assertIsNone(parent['terminal_audit']);self.assertEqual(parent['status'],'OBSERVER_DRIVER_FAILED')
        self.assertEqual(snapshot,parent['incomplete_preservation']);self.assertEqual(snapshot['owned_birth_status'],'UNAVAILABLE');self.assertEqual(snapshot['owned_death_observations'],[]);self.assertEqual(self.deaths,[])
        self.assertEqual(snapshot['admission_denial'],{'status':'ADMISSION_DENIED','worker_attempted':False,'worker_launched':False,'worker_resumed':False})
        self.assertEqual(snapshot['external_anchors'],{'config_path':self.value['config_path'],'anchors':self.value['anchors']})
        self.assertEqual(set(snapshot['lease_files']),{'reservation.json','terminal.json'})
        for name,d in snapshot['lease_files'].items():
            actual=(self.host.lease/name).read_bytes();self.assertEqual(actual,(self.target/d['path']).read_bytes());self.assertEqual(d['sha256'],hashlib.sha256(actual).hexdigest())
        self.assertEqual((self.raw/'SUPERVISOR_RECEIPT.json').read_bytes(),(self.host.lease/'terminal.json').read_bytes())
        for name,d in snapshot['raw_files'].items():self.assertEqual((self.raw/name).read_bytes(),(self.target/name).read_bytes())
        for source,d in snapshot['source_snapshot_files'].items():self.assertEqual(Path(source).read_bytes(),(self.target/d['path']).read_bytes())
        self.assertFalse((self.target/'PRESERVATION.json').exists());self.assertFalse((self.host.lease/'worker.json').exists());self.assertFalse((self.host.lease/'resume.json').exists())
        self.assertFalse(snapshot['complete_original_preservation'])
        for record in (parent,snapshot):
            self.assertFalse(record['native_PASS']);self.assertFalse(record['native_authority_issued']);self.assertTrue(record['fixture_only'])
            self.assertEqual(record['sha256'],self.C.digest({k:v for k,v in record.items() if k!='sha256'}))
    def test_no_SUP_early_failure_retains_original_incomplete_path(self):
        primary=RuntimeError('inert early failure')
        def fail():raise primary
        with self.assertRaises(RuntimeError) as caught:self.run_driver(fail)
        self.assertIs(caught.exception,primary);parent=self.C.parse(self.receipt.read_bytes());snapshot=parent['incomplete_preservation']
        self.assertNotIn('admission_denial',snapshot);self.assertNotIn('lease_files',snapshot);self.assertEqual(snapshot['owned_death_observations'],[])
        self.assertEqual(snapshot,self.C.parse((self.target/'DRIVER_FAILURE_SNAPSHOT.json').read_bytes()))
    def test_extra_worker_lease_refuses_denial_preservation(self):
        def supervise():
            result=self.supervise();(self.host.lease/'worker.json').write_bytes(b'{}\n');return result
        with self.assertRaisesRegex(ValueError,'exact original terminal lease population'):self.run_driver(supervise)
        parent=self.C.parse(self.receipt.read_bytes());self.assertIsNone(parent['incomplete_preservation'])
        self.assertIn('failure_snapshot',[e['stage'] for e in parent['errors']]);self.assertFalse(parent['native_PASS'])
    def test_external_lease_escape_refused_without_copying_foreign_bytes(self):
        def supervise():
            result=self.supervise();sup=result['receipt'];sup['lease_path']=str(self.root/'foreign')
            (self.raw/'SUPERVISOR_RECEIPT.json').write_bytes((self.C.canonical(sup)+'\n').encode('ascii'));return result
        with self.assertRaises(ValueError):self.run_driver(supervise)
        parent=self.C.parse(self.receipt.read_bytes());self.assertIsNone(parent['incomplete_preservation'])
        errors=[e for e in parent['errors'] if e['stage']=='failure_snapshot'];self.assertEqual(len(errors),1)
        self.assertIn('selected external output root',errors[0]['error']['message'])
        self.assertFalse((self.target/'lease_reservation.json').exists())
    def test_original_full_preservation_source_unchanged(self):
        tree=ast.parse((PUBLIC/'preserve.py').read_bytes());text=ast.unparse(tree)
        self.assertIn("('reservation.json', 'worker.json', 'resume.json', 'terminal.json')",text)
        self.assertIn('exact original terminal lease population',text)
        self.assertIn('fresh externally bound every owned birth dead',text)

if __name__=='__main__':unittest.main()
