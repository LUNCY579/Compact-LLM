"""Small authored owned-runtime controls. No real process/model/data inputs."""
from collections import defaultdict,Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import ast,builtins,hashlib,io,json,math,os,random,re,sys,tempfile,threading,time,unittest,uuid

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
ENGINE=PACKAGE/'evidence/study_training_preparation_reviewed_cpu_20261006/current002/study_training.py'
AUTHORED=ENGINE.with_name('test_study_training.py')
FAKE=PACKAGE/'tests/test_study_native_chunk.py'
def module(path,name):
    m=ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m

class OwnedControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reads=[];cls.denials=[];cls.imports=[];oldopen=io.open;oldbuiltin=builtins.open;oldimport=builtins.__import__
        def reader(path,mode='r',*args,**kwargs):
            if isinstance(path,(str,bytes,Path)) and not any(c in str(mode) for c in ('w','a','x','+')):
                selected=Path(path).resolve();cls.reads.append(str(selected));allowed=selected.is_relative_to(HERE) or (selected.parent==PACKAGE and selected.suffix=='.py') or selected in (ENGINE,AUTHORED,FAKE)
                if not allowed:cls.denials.append(str(selected));raise AssertionError('real data/model/API input read denied: '+str(selected))
            return oldopen(path,mode,*args,**kwargs)
        def imports(name,*args,**kwargs):
            if name.split('.',1)[0] in {'torch','transformers','tokenizers','safetensors','psutil','ctypes','numpy'}:cls.imports.append(name);raise AssertionError('actual framework/native/API import denied: '+name)
            return oldimport(name,*args,**kwargs)
        cls.traps=[patch('io.open',side_effect=reader),patch.object(builtins,'open',side_effect=lambda p,*a,**kw:reader(p,*a,**kw) if isinstance(p,(str,bytes,Path)) else oldbuiltin(p,*a,**kw)),patch.object(builtins,'__import__',side_effect=imports)]
        for trap in cls.traps:trap.start()
        cls.T=module(HERE/'owned_study.py','authored_owned_STUDY');cls.E=module(ENGINE,'selected_original_STUDY_engine');cls.mods=cls.E._helpers()
        tree=ast.parse(AUTHORED.read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='authored_sources');ns={'deepcopy':deepcopy,'T':cls.E};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(AUTHORED),'exec'),ns)
        cls.registration,cls.pools,cls.records,cls.policy=ns['authored_sources'](cls.mods)
    @classmethod
    def tearDownClass(cls):
        for trap in reversed(cls.traps):trap.stop()
        assert cls.imports==[],cls.imports
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_owned_',dir=HERE);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve();self.events=[]
    def ref(self,name,value):
        path=self.root/name;raw=self.T.canonical(value).encode('ascii');path.write_bytes(raw);return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'value_sha256':self.T.digest(value)}
    def context(self):
        docs={'registration':self.registration,'pools':self.pools,'records':self.records,'guard_policy':self.policy};mods=self.E._fixture_modules(self.mods,docs);chunks=mods['study_native_chunk'];recipe=chunks.recipe(self.registration,'M1',chunks.NATIVE_SCOPE)
        output=self.root/'selected_owned_output';cfg=self.ref('authored_engine_config.json',{'CPU_authored':True})
        prepared={'config':{'arm':'M1','chunk':1,'resources':self.mods['native_contract'].RESOURCE_BOUNDS},'modules':mods,'registration':self.registration,'recipe':recipe,'previous':None,'latest':None,'fixture_only':True,'independent_checks':{'CPU_authored_no_qualification':True}}
        value={'engine_config':cfg,'engine_anchors':{'config_file_sha256':cfg['sha256'],'config_value_sha256':cfg['value_sha256']},'output':str(output),'output_root':str(self.root),'source_files':{},'runtime_files':{},'runtime_versions':self.mods['native_contract'].RUNTIME,'offline_environment':self.mods['native_contract'].OFFLINE_ENV}
        return self.T.Context(value,self.E,prepared,capability=self.T._CPU_capability())
    def owner_files(self,context):
        output=Path(context.value['output']);output.mkdir();self.T.write_new(output/self.T.CONTEXT,{'launch_ref':None,'value':context.value})
        owner={'worker_pid':7001,'root_worker_pid':7001,'job_name':'Local\\SYNTH_NQ_JOB_'+'a'*32,'supervisor_pid':6001,'supervisor_create_time':123.0,'config_sha256':context.value['engine_config']['sha256']}
        self.T.write_new(output/'supervisor_ownership.json',owner);self.T.write_new(output/'supervisor_heartbeat.jsonl',{'unix_seconds':time.time(),'supervisor_pid':6001});(output/'worker.stdout.log').write_bytes(b'');(output/'worker.stderr.log').write_bytes(b'');return output
    def host(self):
        class FakeOS:
            def getpid(self):return 7002
            def getppid(self):return 7001
            def __getattr__(self,name):return getattr(os,name)
        process=SimpleNamespace(is_running=lambda:True,create_time=lambda:123.0)
        return SimpleNamespace(os=FakeOS(),time=time,psutil=SimpleNamespace(Process=lambda pid:process),membership=lambda name:True,available=lambda:{'available_bytes':6*1024**3},owned_resources=lambda name:{'tree_rss_bytes':256*1024**2})
    def inert_factory(self,prepared,boundaries):
        tree=ast.parse(FAKE.read_bytes());definition=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QuadraticAdam');ns={'deepcopy':deepcopy,'Path':Path,'hashlib':hashlib,'json':json,'os':os,'math':math,'random':random,'chunk':prepared['modules']['study_native_chunk'],'schedule':prepared['modules']['study_schedule']};exec(compile(ast.Module(body=[definition],type_ignores=[]),str(FAKE),'exec'),ns)
        class Lazy:
            execution_scope=prepared['modules']['study_native_chunk'].NATIVE_SCOPE
            def __call__(inner,**kwargs):return ns['QuadraticAdam']('M1',prepared['recipe'],kwargs['call'],kwargs['guard'],scope=inner.execution_scope)
        return Lazy()
    def sup_namespace(self):
        tree=ast.parse((PACKAGE/'native_supervisor.py').read_bytes());names=['_utc','_json_safe','_json_new','_read','_new_output','_acquire_lease','_bounds_valid','_fresh_cpu','_fresh_gpu','_admission_failure','_inventory'];defs=[deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)) for name in names]
        ns={'Path':Path,'time':time,'sys':sys,'PACKAGE':PACKAGE,'os':os,'uuid':uuid,'datetime':datetime,'timezone':timezone,'json':json,'math':math,'hashlib':hashlib,'GIB':1024**3,'BOUNDS':{k:v for k,v in self.mods['native_contract'].RESOURCE_BOUNDS.items() if k not in ('CPU_sample_max_age_seconds','GPU_sample_max_age_seconds','GPU_probe_interval_seconds','GPU_probe_timeout_seconds','admission_poll_seconds','admission_seconds')},'CPU_INTERVAL':.1,'CPU_FRESHNESS':.2,'GPU_INTERVAL':.5,'GPU_FRESHNESS':2.0,'GPU_TIMEOUT':2.0,'ADMISSION_INTERVAL':2.0,'ADMISSION_DEADLINE':30.0}
        class Mutex:
            def __enter__(inner):self.events.append('lease_mutex_enter');return inner
            def __exit__(inner,*args):self.events.append('lease_mutex_exit')
        ns.update(_LeaseMutex=Mutex,_pid_identity=lambda pid:{'pid':pid,'create_time':123.0},_pid_alive=lambda identity:False,_shutdown_gpu_probe=lambda:self.events.append('gpu_shutdown'),_gpu_pending_threads=lambda:[],_LaunchFailure=type('AuthoredLaunchFailure',(RuntimeError,),{}))
        def snapshot(output):
            now=time.monotonic();return {'sample_started_monotonic':now,'sample_completed_monotonic':now,'host_available_bytes':1,'disk_free_bytes':6*1024**3,'worker_tree_rss_bytes':0,'gpu_total_bytes':8*1024**3,'gpu_free_bytes':6*1024**3,'GPU_index':0,'gpu_uuid':'CPU_AUTHORED_NOT_GPU','gpu_observation_id':'CPU_AUTHORED_1','gpu_sample_started_monotonic':now,'gpu_sample_completed_monotonic':now}
        ns['resource_snapshot']=snapshot;ns['_cpu_snapshot']=snapshot
        ns['_WindowsOwnedWorker']=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('denial must not construct worker'));ns['_Monitor']=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('denial must not construct monitor'))
        exec(compile(ast.Module(body=defs,type_ignores=[]),str(PACKAGE/'native_supervisor.py'),'exec'),ns);return ns
    def test_original_SUP_denial_lease_raw_receipt_and_no_worker_API(self):
        context=self.context();ns=self.sup_namespace();locations=self.T._external_original_helpers(ns,self.root);supervise,proof=self.T.supervisor_body(ns,context);result=supervise(context.value['engine_config']['path'],context.value['output'])
        self.assertEqual(proof,{'worker':2,'scope':1});self.assertEqual(locations,{'_new_output':1,'_acquire_lease':1});self.assertEqual(result['status'],'ADMISSION_DENIED');self.assertFalse(result['worker_attempted']);self.assertIn('lease_mutex_enter',self.events)
        output=Path(context.value['output']);self.assertEqual(json.loads((output/'SUPERVISOR_RECEIPT.json').read_bytes()),result);self.assertEqual(json.loads((Path(result['lease_path'])/'terminal.json').read_bytes()),result);self.assertEqual(result['bounds']['preworker_available_GiB'],3.3125)
    def test_same_worker_original_journal_chunk_admission_archive_cost_publications(self):
        context=self.context();output=self.owner_files(context)
        source=ast.parse((PACKAGE/'native_supervisor.py').read_bytes());heartbeat=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='_heartbeat');ns={'Path':Path,'time':time,'json':json,'os':os,'HEARTBEAT_MAX_BYTES':1024**2};exec(compile(ast.Module(body=[heartbeat],type_ignores=[]),str(PACKAGE/'native_supervisor.py'),'exec'),ns)
        stop=threading.Event()
        def heartbeat_loop():
            while not stop.wait(.1):ns['_heartbeat'](output,6001)
        beat=threading.Thread(target=heartbeat_loop,name='CPU_AUTHORED_ORIGINAL_HEARTBEAT');beat.start()
        try:report=self.T.worker_body(context,cpu=self.host(),factory=self.inert_factory)
        finally:stop.set();beat.join(2);self.assertFalse(beat.is_alive())
        protocol=report['protocol_result'];self.assertEqual(protocol['status'],'COMPLETE_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED',protocol.get('error'));resume=protocol['resume_descriptor'];self.assertEqual(resume['envelope']['binding']['completed_updates'],9);self.assertEqual(resume['envelope']['binding']['consumed_occurrences'],180)
        self.assertEqual(resume['terminal_costs']['protocol']['backward']['returned'],180);self.assertEqual(resume['terminal_costs']['protocol']['optimizer_step']['returned'],9)
        for key in ('publication','terminal_publication','promotion_publication'):
            ref=resume[key];raw=Path(ref['path']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),ref['sha256']);self.assertTrue(ref['durable'])
        self.assertTrue(Path(resume['envelope']['backend_descriptor']['path']).is_file());self.assertEqual(json.loads((output/'worker_result.json').read_bytes()),report);self.assertFalse(report['native_PASS']);self.assertEqual(report['terminal_audit']['kind'],'WORKER_STUDY_PENDING_POST_JOB_TERMINAL_AUDIT')
        observed=json.loads((output/'stage_study_worker_ownership_verified.json').read_bytes());self.assertEqual((observed['actual_worker_pid'],observed['launcher_root_pid'],observed['actual_parent_pid']),(7002,7001,7001))
    def test_foreign_existing_leaf_and_changed_config_reject_before_callbacks(self):
        context=self.context();output=self.owner_files(context);(output/'foreign.txt').write_bytes(b'UNSELECTED_AUTHORED')
        with self.assertRaisesRegex(ValueError,'foreign existing output'):self.T.build_owned_boundaries(context,cpu=self.host())
        (output/'foreign.txt').unlink();Path(context.value['engine_config']['path']).write_bytes(b'CHANGED_AUTHORED')
        with self.assertRaisesRegex(ValueError,'worker config ownership mismatch'):self.T.build_owned_boundaries(context,cpu=self.host())
    def test_original_engine_consumes_exact_selected_dict_builder_metadata(self):
        context=self.context();output=self.owner_files(context);path=HERE/'owned_study.py';raw=path.read_bytes();source={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        context.prepared['config'].update(owned_boundary_source=source,owned_boundary_name='build_owned_boundaries',output_directory=str(output),source_pins={str(path):source['sha256']});context.prepared.update(owned_launch_ref=None,owned_engine_config_path=context.value['engine_config']['path'],owned_engine_external_anchors=context.value['engine_anchors'])
        original=self.E._load_selected
        def selected(ref,pins):
            result=original(ref,pins);result._CPU_install_selected_supplier(context.value,self.E,context.prepared,self.host(),capability=result._CPU_capability());return result
        with patch.object(self.E,'_load_selected',side_effect=selected):boundaries=self.E._boundaries(context.prepared)
        self.assertIs(type(boundaries),dict);self.assertEqual(set(boundaries),self.T.CALLBACKS);self.assertTrue(all(callable(v) for v in boundaries.values()))
        with self.assertRaisesRegex(ValueError,'CPU cannot import frameworks'):boundaries['frameworks']()
        boundaries['cleanup']();boundaries['call']('CPU_original_selected_call',lambda:7);self.assertEqual(boundaries['terminal_audit']({'CPU_authored':True})['native_PASS'],False);boundaries['cleanup']._owned_state['journal'].stream.close()
    def test_partial_scientific_failure_preserves_original_error_and_costs(self):
        context=self.context();output=self.owner_files(context)
        def failed_factory(prepared,boundaries):
            lazy=self.inert_factory(prepared,boundaries)
            class Failure:
                execution_scope=lazy.execution_scope
                def __call__(inner,**kwargs):
                    backend=lazy(**kwargs)
                    def fail(row):return backend.invoke('CPU_injected_training_failure',lambda:(_ for _ in ()).throw(RuntimeError('AUTHORED_EXACT_FORWARD_ERROR')))
                    backend.forward=fail;return backend
            return Failure()
        report=self.T.worker_body(context,cpu=self.host(),factory=failed_factory);protocol=report['protocol_result'];self.assertEqual(protocol['status'],'FAILED_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED');self.assertEqual(protocol['error'],'AUTHORED_EXACT_FORWARD_ERROR');self.assertIsNone(protocol['resume_descriptor']);self.assertEqual(protocol['costs']['protocol']['forward']['failed'],1)
        self.assertEqual(json.loads((output/'worker_result.json').read_bytes()),report);self.assertFalse(report['native_PASS']);self.assertEqual(report['terminal_audit']['kind'],'WORKER_STUDY_PENDING_POST_JOB_TERMINAL_AUDIT')
    def test_staging_and_real_input_import_read_traps(self):
        with self.assertRaisesRegex(ValueError,'staged owned STUDY runtime'):self.T.run_registered({})
        with self.assertRaisesRegex(ValueError,'staged owned STUDY worker'):self.T.main()
        for path in (PACKAGE/'weights.safetensors',PACKAGE/'data/crossed_source2_v1_20261003/train.json',PACKAGE/'evidence/fresh_dev_compact002_20261003/panel.json'):
            for reader in (lambda p:builtins.open(p,'rb'),lambda p:io.open(p,'rb'),lambda p:p.open('rb'),lambda p:p.read_bytes()):
                with self.assertRaisesRegex(AssertionError,'real data/model/API input read denied'):reader(path)

if __name__=='__main__':
    before={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('owned_study.py','test_owned_study.py')};start=time.monotonic();result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(OwnedControls));after={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in before}
    value={'kind':'OWNED_STUDY_ORIGINAL_INTERFACE_FOCUSED_CPU_CONTROLS','tests':result.testsRun,'success':result.wasSuccessful(),'seconds':time.monotonic()-start,'source_before':before,'source_after':after,'source_unchanged':before==after,'read_paths_and_counts':dict(sorted(Counter(OwnedControls.reads).items())),'intentional_denials':len(OwnedControls.denials),'forbidden_import_attempts':OwnedControls.imports,'fixture_only':True,'native_PASS':False,'native_run':False,'quality_gain_established':False,'limits':['Original supervisor admission-denial lifecycle and original lease/publication helpers exercised; no actual Windows worker/Job/monitor/sampler APIs.','Original worker Journal/ownership/guard/admission/scientific nine-update body executed with authored scalar backend and fixtures. No real model/optimizer math or data.','Configured native builder metadata route/main remain source-only, needing independent review and genuine qualified packet.','Post-Job independent trained-chunk audit and genuine qualification/M0/guard adapters remain missing. No worker-side post-death PASS.']};value['sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest();path=HERE/'CPU_CONTROLS_004.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'path':str(path),'sha256':value['sha256'],'success':result.wasSuccessful(),'tests':result.testsRun,'source_unchanged':before==after}));raise SystemExit(0 if result.wasSuccessful() and before==after else 1)
