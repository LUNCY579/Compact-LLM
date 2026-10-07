"""Actual measurement guard and independent replay, inert provider only."""
from pathlib import Path
from copy import deepcopy
from types import ModuleType,SimpleNamespace
import hashlib,json,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent
def load(path,name):m=ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
G=load(HERE/'measurement_guard.py','CPU_measurement_guard');R=load(HERE/'measurement_replay.py','CPU_measurement_replay')
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode('ascii')).hexdigest()
def descriptor(path):raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source' if path.suffix=='.py' else 'interpreter_binary' if path.suffix=='.exe' else 'extension_binary'}
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
class Provider:
    fixture_only=True
    def __init__(self,policy):self.policy=policy;self.state={policy['initial_loaded_libraries'][0]:101};self.log=[];self.closed=False;self.fail_cleanup=False
    def begin(self,*args):return None
    def scope(self):return {'kind':'OBSERVED_MANAGED_DLL_DIRECTORY_SCOPE','managed_setup_complete':True,'directories':self.policy['search_directories'],'system_directory':self.policy['system_directory'],'default_flags':0xC00,'legacy_directory':'','foreign_directory_registration':'UNOBSERVED','evidence_basis':'explicit_injected_CPU_managed_observation','closed':False}
    def libraries(self):return {'kind':'OBSERVED_PROCESS_LIBRARY_SNAPSHOT','complete':True,'libraries':[{'path':p,'module_handle':h} for p,h in self.state.items()]}
    def events(self):return {'kind':'OBSERVED_PROCESS_LIBRARY_EVENTS','complete':True,'events':deepcopy(self.log)}
    def add(self,p,handle=None):
        h=102+len(self.log) if handle is None else handle;self.state[p]=h;self.log.append({'sequence':len(self.log)+1,'event':'load','path':p,'module_handle':h})
    def close(self):
        self.closed=True
        if self.fail_cleanup:raise OSError('inert cleanup failed')
        return {'kind':'OBSERVED_MANAGED_DLL_DIRECTORY_CLEANUP','managed_cleanup_complete':True,'remaining_managed_directories':[],'legacy_directory':'','all_managed_cookies_removed':True,'foreign_directory_registration':'UNOBSERVED','closed':True}
class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='CPU_measurement_');self.root=Path(self.tmp.name).resolve();self.paths={};self.files={}
        for name in ('base.exe','one.pyd','two.pyd','pre.dll','nested.dll','direct.dll'):
            p=self.root/name;p.write_bytes(('inert '+name).encode());self.paths[name]=str(p);self.files[str(p)]=descriptor(p)
        self.policy={'kind':'FINITE_WINDOWS_DLL_BOUNDARY_V2','search_directories':[str(self.root)],'system_directory':str(self.root),'library_files':deepcopy(self.files),'initial_loaded_libraries':[self.paths['base.exe']],'extension_transitions':{self.paths[n]:{'create':None,'exec':None} for n in ('one.pyd','two.pyd')},'framework_transition':None,'provider_source':{'path':str(G.ORIGINAL_SOURCE),'descriptor':descriptor(G.ORIGINAL_SOURCE)}}
        self.grammar={'schema_version':1,'kind':'FINITE_IMPORT_EFFECT_MEASUREMENT_GRAMMAR_V1','required_extensions':[self.paths['one.pyd']],'optional_extensions':[self.paths['two.pyd']],'maximum_boundaries':5,'maximum_depth':64,'maximum_observations':4096,'model':False};self.hash_calls=[]
        def check(p,d,**kw):self.hash_calls.append(p);self.assertEqual(descriptor(Path(p)),d)
        self.C=SimpleNamespace(parse=json.loads,canonical=canonical,digest=digest,file_descriptor=check);self.P=Provider(self.policy);self.guard=G.MeasurementGuard(self.policy,self.grammar,self.C,self.P);self.identity={'source_commit':'a'*40,'original_scope':G.SCOPE,'owned_birth':{'pid':1,'creation_filetime_100ns':2},'owned_birth_status':'VALIDATED','owner_receipt_sha256':'c'*64,'context_sha256':'b'*64,'producer_source':{'path':str(HERE/'measurement_guard.py'),'descriptor':descriptor(HERE/'measurement_guard.py')},'fixture_only':True,'native_PASS':False,'native_authority_issued':False};self.sources={str(G.ORIGINAL_SOURCE):descriptor(G.ORIGINAL_SOURCE),str(HERE/'measurement_guard.py'):descriptor(HERE/'measurement_guard.py')};self.trap=Trap();sys.meta_path.insert(0,self.trap)
    def tearDown(self):
        if not self.guard.closed:
            try:self.guard.close()
            except BaseException:pass
        sys.meta_path.remove(self.trap);self.tmp.cleanup()
    def raw(self):return (canonical(self.guard.export_raw_packet(self.identity))+'\n').encode('ascii')
    def replay(self,raw=None):
        raw=self.raw() if raw is None else raw;return R.validate_measurement(raw,policy=self.policy,grammar=self.grammar,expected_identity=self.identity,selected_file_map=self.files,selected_source_map=self.sources,expected_file_sha256=hashlib.sha256(raw).hexdigest())
    def execute(self,reverse=False):
        with self.guard.framework_import():
            self.P.add(self.paths['pre.dll'])
            for name in (('two.pyd','one.pyd') if reverse else ('one.pyd','two.pyd')):
                with self.guard.extension(self.paths[name],'create'):self.P.add(self.paths[name])
                with self.guard.extension(self.paths[name],'exec'):
                    if name=='one.pyd':self.P.add(self.paths['nested.dll'])
            self.P.add(self.paths['direct.dll'])
        self.guard.close()
    def test_actual_nested_siblings_prenested_and_own_effects(self):
        self.execute();result=self.replay();self.assertEqual(result['status'],'MEASUREMENT_RECORDED_UNQUALIFIED');self.assertFalse(result['native_qualification']);self.assertFalse(result['native_authority_issued']);framework=next(x for x in result['effects'] if x['boundary']['name']=='framework_import');self.assertEqual(framework['own_new_libraries'],sorted([self.paths['pre.dll'],self.paths['direct.dll']]));self.assertEqual(len(framework['nested_new_libraries']),3);self.assertIn(str(G.ORIGINAL_SOURCE),self.hash_calls);self.assertIn(str(HERE/'measurement_guard.py'),self.hash_calls)
    def test_unknown_unload_duplicate_handle_and_event_loss_poison(self):
        for kind in ('unknown','unload','duplicate','loss'):
            P=Provider(self.policy);guard=G.MeasurementGuard(self.policy,self.grammar,self.C,P)
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                with guard.framework_import():
                    with guard.extension(self.paths['one.pyd'],'create'):
                        P.add(str(self.root/'unknown.dll') if kind=='unknown' else self.paths['one.pyd'],101 if kind=='duplicate' else None)
                        if kind=='unload':P.log[-1]['event']='unload'
                        if kind=='loss':P.log=[]
            self.assertTrue(guard.failed)
            try:guard.close()
            except ValueError:pass
            self.assertTrue(P.closed);packet=guard.export_raw_packet(self.identity);self.assertNotEqual(packet['terminal']['status'],'complete');self.assertFalse(packet['native_PASS'])
            encoded=(canonical(packet)+'\n').encode('ascii');value=R.validate_measurement(encoded,policy=self.policy,grammar=self.grammar,expected_identity=self.identity,selected_file_map=self.files,selected_source_map=self.sources,expected_file_sha256=hashlib.sha256(encoded).hexdigest());self.assertEqual(value['status'],'MEASUREMENT_FAILURE_PRESERVED')
    def test_replay_resealed_producer_effect_event_and_parent_forgeries_refuse(self):
        self.execute(reverse=True);self.assertEqual(self.replay()['status'],'MEASUREMENT_RECORDED_UNQUALIFIED');original=json.loads(self.raw())
        for kind in ('effect','event','parent'):
            v=deepcopy(original)
            if kind=='effect':v['measured_boundary_effects'][-1]['own_new_libraries']=[]
            elif kind=='event':v['observations'][-1]['event_trace']['events'].pop()
            else:v['observations'][2]['boundary']['parent_id']=None
            v.pop('sha256');v['sha256']=digest(v);raw=(canonical(v)+'\n').encode('ascii')
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.replay(raw)
    def test_required_pairs_repeat_and_cleanup_failure_keep_failure_raw(self):
        with self.assertRaisesRegex(ValueError,'poisoned measurement'):
            with self.guard.framework_import():
                with self.guard.extension(self.paths['one.pyd'],'create'):self.P.add(self.paths['one.pyd'])
                with self.assertRaises(ValueError):
                    with self.guard.extension(self.paths['one.pyd'],'create'):pass
        with self.assertRaises(ValueError):self.guard.close()
        self.assertTrue(self.guard.closed);self.assertNotEqual(self.guard.export_raw_packet(self.identity)['terminal']['status'],'complete')
        P=Provider(self.policy);guard=G.MeasurementGuard(self.policy,self.grammar,self.C,P)
        with guard.framework_import():pass
        with self.assertRaises(ValueError):guard.close()
        packet=guard.export_raw_packet(self.identity);self.assertEqual(packet['terminal']['status'],'failed');encoded=(canonical(packet)+'\n').encode('ascii');value=R.validate_measurement(encoded,policy=self.policy,grammar=self.grammar,expected_identity=self.identity,selected_file_map=self.files,selected_source_map=self.sources,expected_file_sha256=hashlib.sha256(encoded).hexdigest());self.assertEqual(value['status'],'MEASUREMENT_FAILURE_PRESERVED')
    def test_extension_prenested_effect_is_owned_by_extension_frame(self):
        with self.guard.framework_import():
            with self.guard.extension(self.paths['one.pyd'],'create'):
                self.P.add(self.paths['one.pyd']);self.P.add(self.paths['pre.dll'])
                with self.guard.extension(self.paths['two.pyd'],'create'):self.P.add(self.paths['two.pyd'])
                with self.guard.extension(self.paths['two.pyd'],'exec'):self.P.add(self.paths['nested.dll'])
            with self.guard.extension(self.paths['one.pyd'],'exec'):pass
        self.guard.close();value=self.replay();one=next(x for x in value['effects'] if x['boundary']['name']=='extension:'+self.paths['one.pyd']+':create');self.assertEqual(one['own_new_libraries'],sorted([self.paths['one.pyd'],self.paths['pre.dll']]));self.assertEqual(one['nested_new_libraries'],sorted([self.paths['two.pyd'],self.paths['nested.dll']]))
    def test_cleanup_failure_preserves_partial_and_exact_consumer_refuses_kind(self):
        with self.guard.framework_import():
            with self.guard.extension(self.paths['one.pyd'],'create'):self.P.add(self.paths['one.pyd'])
            with self.guard.extension(self.paths['one.pyd'],'exec'):pass
        self.P.fail_cleanup=True
        with self.assertRaises(ValueError):self.guard.close()
        packet=self.guard.export_raw_packet(self.identity);self.assertEqual(packet['terminal']['status'],'failed');self.assertTrue(self.P.closed);self.assertEqual(self.replay()['status'],'MEASUREMENT_FAILURE_PRESERVED')
        with self.assertRaises(ValueError):R.E.validate_observed_packet(self.raw(),policy=self.policy,expected_identity=self.identity,selected_file_map=self.files,selected_source_map=self.sources,expected_boundaries=[],expected_file_sha256=hashlib.sha256(self.raw()).hexdigest())
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));receipt={'kind':'CPU_MEASUREMENT_GUARD_REPLAY_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_controls_completed':result.wasSuccessful(),'original_source_scope_snapshot_event_directory_cleanup_helpers_executed':True,'inert_provider_only':True,'native_API_or_framework_called':False,'native_authority_issued':False,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'measurement_guard.py',HERE/'measurement_replay.py',Path(__file__),G.ORIGINAL_SOURCE,R.SOURCE)}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
