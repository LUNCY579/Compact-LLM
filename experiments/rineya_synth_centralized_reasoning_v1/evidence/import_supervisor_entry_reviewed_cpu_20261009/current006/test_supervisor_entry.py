"""Original supervisor body with inert owned host; no native imports/launches."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import ast,hashlib,importlib,json,math,sys,tempfile,unittest
from unittest import mock
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

class EntryControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='CPU_parent_dependency_');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.package=self.root/'package';self.site=self.root/'site';self.base=self.root/'python';self.package.mkdir();self.site.mkdir();(self.base/'Lib').mkdir(parents=True);(self.base/'DLLs').mkdir()
        self.names=('psutil','native_process','native_nvml');self.before_modules={n:sys.modules.get(n) for n in self.names};self.addCleanup(self.restore_modules)
        for name in self.names:sys.modules.pop(name,None)
        sources={'psutil':self.site/'psutil.py','native_process':self.package/'native_process.py','native_nvml':self.package/'native_nvml.py'}
        sources['psutil'].write_bytes(b'class NoSuchProcess(Exception):pass\nSTATUS_ZOMBIE="zombie"\nclass Process:\n def __init__(self,pid):self.pid=pid\n def create_time(self):return 7.0\n def is_running(self):return False\n def status(self):return "dead"\n')
        sources['native_process'].write_bytes(b'class Sampler:\n def is_alive(self,identity):return False\n')
        sources['native_nvml'].write_bytes(b'# inert metadata only\n')
        self.uuid=self.base/'Lib/uuid.py';self.uuid.write_bytes(b'# inert UUID source\n');self.uuid_ext=self.base/'DLLs/_uuid.pyd';self.uuid_ext.write_bytes(b'not an executable extension; never loaded')
        def d(p,role='python_source'):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'role':role}
        self.source_map={str(p):d(p) for n,p in sources.items() if n!='psutil'};runtime={str(sources['psutil']):d(sources['psutil']),str(self.uuid):d(self.uuid),str(self.uuid_ext):d(self.uuid_ext,'extension_binary')}
        row={'path':str(sources['psutil']),'descriptor':runtime[str(sources['psutil'])],'cache_path':None,'cache_descriptor':None}
        self.value={'config':{'source_files':self.source_map},'runtime':{'stdlib_layout':{'physical_base_prefix':str(self.base)},'files':runtime,'cache_policy':{'interpreter_profile':{'path':list(sys.path)},'preloaded_bootstrap_modules':{},'additional_preload_sources':{'psutil':row}}}}
        self.fd_calls=[]
        def fd(p,d,**kw):
            self.fd_calls.append(str(p));raw=Path(p).read_bytes();self.assertEqual(len(raw),d['bytes']);self.assertEqual(hashlib.sha256(raw).hexdigest(),d['sha256'])
        self.C=SimpleNamespace(PACKAGE=self.package,SITE=self.site,stage_native_gate=lambda:None,qualified_path=lambda p,**kw:Path(p).resolve(),file_descriptor=fd,read_bytes=lambda p:Path(p).read_bytes(),require=S.require)
        ns={'__builtins__':__builtins__};nodes=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in ('_psutil','_pid_alive')];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns);self.original=ns
    def restore_modules(self):
        for n,value in self.before_modules.items():
            if value is None:sys.modules.pop(n,None)
            else:sys.modules[n]=value
    def test_actual_original_psutil_and_birth_verifier_resolve_selected_inert_sources(self):
        with self.assertRaises(ModuleNotFoundError):self.original['_psutil']()
        paths=sys.path;before=list(paths);meta=sys.meta_path;mb=list(meta);finish=S._supervisor_dependency_scope(self.value,self.C)
        try:
            psutil=self.original['_psutil']();self.assertEqual(Path(psutil.__file__),self.site/'psutil.py');self.assertFalse(self.original['_pid_alive']({'pid':4,'creation_filetime_100ns':123}));process=sys.modules['native_process'];self.assertEqual(Path(process.__file__),self.package/'native_process.py');self.assertEqual(process.Sampler.is_alive.__code__.co_filename,str(self.package/'native_process.py'))
            self.assertEqual(sys.path[-2:],[str(self.package),str(self.site)]);self.assertIsInstance(sys.meta_path[0],S._SupervisorFinder)
        finally:finish()
        self.assertIs(sys.path,paths);self.assertEqual(sys.path,before);self.assertIs(sys.meta_path,meta);self.assertEqual(sys.meta_path,mb)
        self.assertIs(sys.modules['psutil'],psutil);self.assertFalse(self.original['_pid_alive']({'pid':4,'creation_filetime_100ns':123}))
    def test_finally_restores_same_containers_on_error_and_tamper(self):
        paths=sys.path;before=list(paths);meta=sys.meta_path;mb=list(meta);finish=S._supervisor_dependency_scope(self.value,self.C)
        try:raise RuntimeError('inert original lifecycle failure')
        except RuntimeError:pass
        finally:finish()
        self.assertIs(sys.path,paths);self.assertEqual(paths,before);self.assertIs(sys.meta_path,meta);self.assertEqual(meta,mb)
        finish=S._supervisor_dependency_scope(self.value,self.C);sys.path.append('unregistered')
        with self.assertRaisesRegex(ValueError,'unchanged before restore'):finish()
        self.assertIs(sys.path,paths);self.assertEqual(paths,before);self.assertIs(sys.meta_path,meta);self.assertEqual(meta,mb)
    def test_historical_parent_row_normalizes_selected_source_and_cache_descriptors(self):
        source=self.base/'Lib/historical_parent.py';source.write_bytes(b'# inert historical source\n');cache=self.base/'Lib/historical_parent.pyc';cache.write_bytes(b'inert cache metadata never executed')
        files=self.value['runtime']['files'];files[str(source)]={'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'role':'python_source'};files[str(cache)]={'bytes':cache.stat().st_size,'sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),'role':'qualified_bytecode_cache'}
        row={'path':str(source),'source_sha256':files[str(source)]['sha256'],'cache_path':str(cache),'cache_sha256':files[str(cache)]['sha256'],'cached_attribute_present':True,'cached_value_kind':'str'};self.value['runtime']['cache_policy']['preloaded_bootstrap_modules']['historical_parent']=row;before=dict(row);finder=S._SupervisorFinder(self.value,self.C)
        with mock.patch.object(S.importlib.machinery.PathFinder,'find_spec',return_value=SimpleNamespace(origin=str(source))):finder.find_spec('historical_parent')
        self.assertEqual(self.fd_calls,[str(source),str(cache)]);self.assertEqual(row,before);self.assertEqual(finder.rows['historical_parent']['descriptor'],files[str(source)]);self.assertEqual(finder.rows['historical_parent']['cache_descriptor'],files[str(cache)])
        row['source_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'pins join'):S._SupervisorFinder(self.value,self.C)
        current=json.loads((PACKAGE/'evidence/import006_no_resume_failure_20261009/dispatch/IMPORT_OBSERVATION_RUNTIME_006.json').read_bytes());sources={str(PACKAGE/(n+'.py')):{'bytes':(PACKAGE/(n+'.py')).stat().st_size,'sha256':hashlib.sha256((PACKAGE/(n+'.py')).read_bytes()).hexdigest(),'role':'python_source'} for n in ('native_process','native_nvml')};actual=S._SupervisorFinder({'runtime':current,'config':{'source_files':sources}},SimpleNamespace(PACKAGE=PACKAGE));self.assertEqual(set(actual.rows),set(current['cache_policy']['preloaded_bootstrap_modules'])|set(current['cache_policy']['additional_preload_sources'])|{'native_process','native_nvml','uuid','_uuid'})
    def test_finite_parent_uuid_and_bad_resolution_refusal(self):
        finder=S._SupervisorFinder(self.value,self.C)
        with self.assertRaisesRegex(ValueError,'unknown parent dependency'):finder.find_spec('torch')
        with mock.patch.object(S.importlib.machinery.PathFinder,'find_spec',return_value=SimpleNamespace(origin=str(self.uuid),loader=None)):
            spec=finder.find_spec('uuid');self.assertEqual(spec.loader.get_code('uuid').co_filename,str(self.uuid));self.assertIn(str(self.uuid),self.fd_calls)
        with mock.patch.object(S.importlib.machinery.PathFinder,'find_spec',return_value=SimpleNamespace(origin=str(self.root/'unselected.py'))):
            with self.assertRaisesRegex(ValueError,'matches selected source'):finder.find_spec('psutil')
    def test_original_monitor_shared_binding_only_two_AST_changes(self):
        path=self.package/'short_ram_monitor.py';raw=(PACKAGE/'short_ram_monitor.py').read_bytes();path.write_bytes(raw);self.source_map[str(path)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'};original=ModuleType('inert_original');original._Monitor=type('_Monitor',(),{})
        before=list(sys.path);prior=sys.modules.get('native_supervisor');m=S._selected_monitor(self.value,self.C,original);self.assertIs(m.shared,original);self.assertEqual(m.ShortRamMonitor.__bases__,(original._Monitor,));self.assertEqual(sys.path,before);self.assertIs(sys.modules.get('native_supervisor'),prior)
        root=compile(raw,str(path),'exec');class_code=next(c for c in root.co_consts if type(c) is type(root) and c.co_name=='ShortRamMonitor');codes={c.co_name:c for c in class_code.co_consts if type(c) is type(root)}
        for name,code in codes.items():
            actual=getattr(m.ShortRamMonitor,name).__code__;self.assertEqual(actual.co_code,code.co_code);self.assertEqual(actual.co_consts,code.co_consts);self.assertEqual(actual.co_names,code.co_names);self.assertEqual(actual.co_filename,code.co_filename)
    def test_parent_copy_optional_org_import_preserved_only_when_absent(self):
        finder=S._SupervisorFinder(self.value,self.C);copy_path=Path('C:/Users/clclc/AppData/Roaming/uv/python/cpython-3.11.15-windows-x86_64-none/Lib/copy.py');tree=ast.parse(copy_path.read_bytes());optional=next(n for n in tree.body if isinstance(n,ast.Try) and any(isinstance(x,ast.ImportFrom) and x.module=='org.python.core' for x in n.body))
        with mock.patch.object(S.importlib.machinery.PathFinder,'find_spec',return_value=None):
            with self.assertRaises(ModuleNotFoundError) as caught:finder.find_spec('org')
            self.assertEqual(caught.exception.name,'org')
        for importer in (S.importlib.machinery.BuiltinImporter,S.importlib.machinery.FrozenImporter,S.importlib.machinery.PathFinder):
            with mock.patch.object(importer,'find_spec',return_value=SimpleNamespace(origin='unselected_org')):
                with self.assertRaisesRegex(ValueError,'unselected'):finder.find_spec('org')
        trailing=SimpleNamespace(find_spec=lambda *a: self.fail('unselected trailing finder reached'));sys.meta_path.extend([trailing]);sys.meta_path.insert(0,finder)
        try:
            ns={};exec(compile(ast.Module(body=[optional],type_ignores=[]),str(copy_path),'exec'),ns);self.assertIsNone(ns['PyStringMap'])
        finally:sys.meta_path.remove(finder);sys.meta_path.remove(trailing)
        prior=sys.modules.get('org');sys.modules['org']=ModuleType('org')
        try:
            with self.assertRaisesRegex(ValueError,'fresh parent'):S._supervisor_dependency_scope(self.value,self.C)
        finally:
            if prior is None:sys.modules.pop('org',None)
            else:sys.modules['org']=prior
    def test_preloaded_pwd_refused_before_parent_or_child_transition(self):
        prior=sys.modules.get('pwd');sys.modules['pwd']=ModuleType('pwd')
        try:
            with self.assertRaisesRegex(ValueError,'fresh parent'):S._supervisor_dependency_scope(self.value,self.C)
            B=ModuleType('CPU_pwd_preloaded');B.__file__=str(HERE/'import_bootstrap.py');exec(compile(Path(B.__file__).read_bytes(),B.__file__,'exec'),B.__dict__)
            with self.assertRaisesRegex(ValueError,'preloaded unselected'):B._FinitePreludeFinder(self.value,self.C)
        finally:
            if prior is None:sys.modules.pop('pwd',None)
            else:sys.modules['pwd']=prior
    def test_parent_and_child_absent_pwd_preserve_original_optional_import(self):
        B=ModuleType('CPU_child_prelude');B.__file__=str(HERE/'import_bootstrap.py');exec(compile(Path(B.__file__).read_bytes(),B.__file__,'exec'),B.__dict__)
        finders=(S._SupervisorFinder(self.value,self.C),B._FinitePreludeFinder(self.value,self.C))
        actual_psutil=self.site/'psutil.py';installed=Path('C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Lib/site-packages/psutil/__init__.py');tree=ast.parse(installed.read_bytes());optional=next(n for n in tree.body if isinstance(n,ast.Try) and any(isinstance(x,ast.Import) and any(a.name=='pwd' for a in x.names) for x in n.body))
        for finder in finders:
            with mock.patch.object(S.importlib.machinery.PathFinder,'find_spec',return_value=None):
                with self.assertRaises(ModuleNotFoundError) as caught:finder.find_spec('pwd')
                self.assertEqual(caught.exception.name,'pwd')
            with mock.patch.object(S.importlib.machinery.PathFinder,'find_spec',return_value=SimpleNamespace(origin='unselected_pwd.py')):
                with self.assertRaisesRegex(ValueError,'unselected'):finder.find_spec('pwd')
            for importer in (S.importlib.machinery.BuiltinImporter,S.importlib.machinery.FrozenImporter):
                with mock.patch.object(importer,'find_spec',return_value=SimpleNamespace(origin='unselected_pwd')):
                    with self.assertRaisesRegex(ValueError,'unselected'):finder.find_spec('pwd')
            with self.assertRaisesRegex(ValueError,'unknown'):finder.find_spec('unselected_extra')
            trailing=SimpleNamespace(find_spec=lambda *a:self.fail('unselected trailing finder reached'));sys.meta_path.append(trailing);sys.meta_path.insert(0,finder)
            try:
                ns={};exec(compile(ast.Module(body=[optional],type_ignores=[]),str(installed),'exec'),ns);self.assertIsNone(ns['pwd'])
            finally:sys.meta_path.remove(finder);sys.meta_path.remove(trailing)

if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Controls),unittest.defaultTestLoader.loadTestsFromTestCase(EntryControls)]);result=unittest.TextTestRunner(verbosity=2).run(suite)
    v={'kind':'CPU_SOURCE_BOUND_PARENT_SUPERVISOR_ENTRY_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'original_psutil_and_pid_alive_called':True,'inert_dependency_sources_only':True,'native_framework_API_or_process_called':False,'fixture_only':True,'native_authority_issued':False,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'import_observation_supervisor.py',HERE/'import_bootstrap.py',Path(__file__).resolve(),SOURCE,PACKAGE/'short_ram_monitor.py')}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
