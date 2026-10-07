"""Original supervisor body with inert owned host; no native imports/launches."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import ast,hashlib,json,math,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent;PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
S=ModuleType('CPU_import_supervisor_adapter');S.__file__=str(HERE/'import_observation_supervisor.py');exec(compile(Path(S.__file__).read_bytes(),S.__file__,'exec'),S.__dict__)
SOURCE=PACKAGE/'native_supervisor.py';RAW=SOURCE.read_bytes();TREE=ast.parse(RAW,str(SOURCE))
fixture_source=PACKAGE/'evidence/observer_startup_measurement_reviewed_cpu_20261006/test_supervisor.py';hostnode=next(n for n in ast.parse(fixture_source.read_bytes()).body if isinstance(n,ast.ClassDef) and n.name=='Host');env={'Path':Path};exec(compile(ast.Module(body=[hostnode],type_ignores=[]),str(fixture_source),'exec'),env);Host=env['Host']
FIXED={}
for node in TREE.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('GIB','BOUNDS','CPU_INTERVAL','CPU_FRESHNESS','GPU_INTERVAL','GPU_TIMEOUT','GPU_FRESHNESS','ADMISSION_INTERVAL','ADMISSION_DEADLINE') for t in node.targets):exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),FIXED)
class Controls(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='CPU_import_SUP_');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.host=Host(self.root);self.output=self.root/'output';self.output.mkdir();self.worker=PACKAGE/'import_observation_worker.py';self.config=str(self.root/'config.json');self.refs={'schema_version':2,'config_path':self.config,'anchors':{'expected_config_file':'a'*64,'expected_config_body':'b'*64,'registration_path':str(self.root/'reg.json'),'expected_registration_file':'c'*64,'expected_registration_body':'d'*64,'runtime_path':str(self.root/'runtime.json'),'expected_runtime_file':'e'*64,'expected_runtime_body':'f'*64,'expected_head':'1'*40},'grant_reference':{'path':str(self.root/'grant.json'),'file_sha256':'2'*64,'body_sha256':'3'*64},'owner_receipt_reference':{'path':str(self.root/'receipt.json'),'file_sha256':'4'*64,'body_sha256':'5'*64}}
    def run_body(self):
        host=self.host
        def new_output(request):output=host.new_output(request);host.write_new(output/'OWNED_EXTERNAL_ANCHORS.json',self.refs);return output
        namespace={**FIXED,'Path':Path,'math':math,'PACKAGE':PACKAGE,'sys':SimpleNamespace(executable='CPU_PINNED_PYTHON'),'time':SimpleNamespace(monotonic=host.clock,sleep=host.sleep),'_utc':host.utc,'_json_safe':lambda v:v,'_new_output':new_output,'_json_new':host.write_new,'_acquire_lease':host.acquire_lease,'_release_lease':host.release_lease,'resource_snapshot':host.observation,'_cpu_snapshot':host.cpu_observation,'_WindowsOwnedWorker':host.worker,'_Monitor':host.monitor,'_shutdown_gpu_probe':host.close_NVML,'_gpu_pending_threads':host.pending_threads,'_pid_alive':host.current_alive,'_inventory':host.inventory,'_LaunchFailure':host.LaunchFailure}
        functions=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in ('_bounds_valid','_fresh_cpu','_fresh_gpu','_admission_failure')];exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),namespace)
        resources=json.loads((PACKAGE/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json').read_bytes())['resources']
        namespace['_preflight']=lambda request:{'config_sha256':'a'*64,'config_path':self.config,'worker_path':str(self.worker),'authorization':{'scope':S.SCOPE,'production_training':False,'protected_bank_access':False,'paid_services':False},'resources':resources}
        tree,proof=S._adapt_supervise(RAW,SOURCE,self.worker);exec(compile(tree,str(SOURCE),'exec'),namespace);return namespace['supervise'](self.config,self.output),proof
    def test_actual_supervisor_argv_V2_scope_Job_and_cleanup(self):
        result,proof=self.run_body();self.assertEqual(result['status'],'SUPERVISED_WORKER_EXITED_ZERO');self.assertEqual(proof['total_changes'],6);self.assertEqual(self.host.argv,['CPU_PINNED_PYTHON','-I','-S','-B',str(self.worker),'--config',self.config,'--output-dir',str(self.output)]);self.assertEqual(self.host.writes[str(self.output/'OWNED_EXTERNAL_ANCHORS.json')],self.refs);self.assertLess(self.host.calls.index('Job_assigned'),self.host.calls.index('resume'));self.assertTrue(result['partial_outputs_frozen']);self.assertTrue(result['sampler_threads_terminal_verified']);self.assertTrue(result['job_closed_kill_on_close'])
    def test_original_low_floor_refuses_before_worker(self):
        self.host.low=True;result,proof=self.run_body();self.assertEqual(result['status'],'ADMISSION_DENIED');self.assertFalse(result['worker_attempted']);self.assertNotIn('resume',self.host.calls)
    def test_original_failed_worker_and_unjoined_sampler_never_pass(self):
        self.host.returncode=2;result,proof=self.run_body();self.assertEqual(result['status'],'SUPERVISED_WORKER_FAILED');self.assertEqual(result['returncode'],2)
    def test_public_entry_closed_before_source_or_worker_access(self):
        class Closed:
            SCOPE=S.SCOPE
            def enroll_native_gate(self,*a):raise ValueError('no separate owner grant')
        with self.assertRaisesRegex(ValueError,'no separate owner grant'):S.supervise_registered('not-read','not-created',contract=Closed())
        self.assertEqual(self.host.calls,[])
    def test_actual_partial_preservation_without_worker_result(self):
        package=self.root/'package';output=package/'artifacts/run';output.mkdir(parents=True);(package/'evidence').mkdir();lease=package/'artifacts/_native_gpu_leases/owned';lease.mkdir(parents=True);source=package/'source.py';source.write_bytes(b'# inert selected source\n')
        canon=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'));digest=lambda v:hashlib.sha256(canon(v).encode()).hexdigest()
        def write(path,value):path.write_text(canon(value)+'\n')
        owner={'pid':7,'creation_filetime_100ns':123};resources=[{'owned_process_identities':[owner]}];write(output/'RESOURCE_OBSERVATIONS.json',resources);write(output/'RAW_MEASUREMENT_FAILED.json',{'recording_complete':False,'fixture_only':True});write(output/'OWNED_EXTERNAL_ANCHORS.json',self.refs)
        partial=[{'path':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in output.iterdir()]
        sup={'config_sha256':'a'*64,'output_dir':str(output),'job_closed_kill_on_close':True,'sampler_threads_terminal_verified':True,'gpu_telemetry_cleanup_verified':True,'partial_outputs_frozen':True,'owned_worker_identity_currently_alive':False,'unjoined_sampler_threads':[],'worker_identity':owner,'lease_path':str(lease),'partial_outputs':partial,'returncode':2};write(output/'SUPERVISOR_RECEIPT.json',sup)
        for name,value in {'reservation.json':{'config_sha256':'a'*64},'worker.json':owner,'resume.json':{'worker_identity':owner},'terminal.json':sup}.items():write(lease/name,value)
        d={'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'role':'python_source'};value={'fixture_only':True,'config':{'sha256':'b'*64,'source_files':{str(source):d}},'anchors':{'expected_config_file':'a'*64}}
        def fd(path,expected):self.assertEqual(Path(path).stat().st_size,expected['bytes']);self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),expected['sha256'])
        cap=object();contract=SimpleNamespace(PACKAGE=package,revalidate_context=lambda context:value,read_bytes=lambda path:Path(path).read_bytes(),parse=json.loads,canonical=canon,digest=digest,file_descriptor=fd,_check_CPU=lambda candidate:self.assertIs(candidate,cap))
        seen=[];target=package/'evidence/preserved';result=S.preserve_terminal(output,target,contract=contract,context=object(),death_verifier=lambda birth:seen.append(birth) or False,_CPU_capability=cap)
        self.assertFalse(result['worker_result_present']);self.assertFalse(result['native_PASS']);self.assertEqual(seen,[owner]);self.assertEqual(set(result['raw_files']),{p.name for p in output.iterdir()});self.assertEqual((target/'RAW_MEASUREMENT_FAILED.json').read_bytes(),(output/'RAW_MEASUREMENT_FAILED.json').read_bytes())
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_IMPORT_ONLY_ORIGINAL_SUPERVISOR_ADAPTER_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_original_supervise_body_executed':True,'inert_host_only':True,'native_API_framework_or_process_called':False,'native_launch_authorized':False,'native_authority_issued':False,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'import_observation_supervisor.py',Path(__file__).resolve(),SOURCE)}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
