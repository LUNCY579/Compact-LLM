"""Selected RAM function bodies with inert memory backend; no ctypes import."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import __future__,ast,hashlib,json,sys,types,unittest
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
NP=PACKAGE/'native_process.py';RAM=PACKAGE/'worker_ram_probe.py';WORKER=HERE/'import_observation_worker.py'
def require(v,m):
    if not v:raise ValueError(m)
def descriptor(path):raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}
def file_descriptor(path,d):require(descriptor(Path(path))==d,'actual selected inert source bytes')
fp=PACKAGE/'evidence/tensor_gate_dispatch_reviewed_cpu_20261007/frozen_gate_dispatch002/candidate/prefix_registered_loader.py';fpns={'types':types};nodes=[n for n in ast.parse(fp.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='code_fingerprint'];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(fp),'exec'),fpns)
class Trap:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('ctypes','torch','psutil','native_process','worker_ram_probe'):raise AssertionError('forbidden actual import '+name)
class Controls(unittest.TestCase):
    def setUp(self):
        self.values=[(11,100),(37,100),(37,100)];self.calls=0;self.error=None;self.selected=[];self.trap=Trap();sys.meta_path.insert(0,self.trap)
        C=SimpleNamespace(PACKAGE=PACKAGE,require=require,file_descriptor=file_descriptor,read_bytes=lambda p:Path(p).read_bytes(),code_fingerprint=fpns['code_fingerprint'])
        self.native=ModuleType('CPU_source_slice_native_process');self.native.__file__=str(NP);self.native.__compiled_source_descriptor__=descriptor(NP)
        tree=ast.parse(NP.read_bytes(),str(NP));selected=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='__future__' or isinstance(n,ast.ClassDef) and n.name=='Sampler' or isinstance(n,ast.FunctionDef) and n.name=='system_memory_status']
        self.native.__dict__.update(_SYSTEM_SAMPLER=None,MEMORY_STATUS_EX=lambda:SimpleNamespace(),ctypes=SimpleNamespace(sizeof=lambda v:64,byref=lambda v:v,get_last_error=lambda:123,WinError=lambda code:self.error),os=SimpleNamespace(name='CPU_INERT'))
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(NP),'exec'),self.native.__dict__)
        sampler=object.__new__(self.native.Sampler);sampler.api=SimpleNamespace(GlobalMemoryStatusEx=self.backend);self.native._SYSTEM_SAMPLER=sampler
        def selector(path,d):
            self.selected.append(str(path));file_descriptor(path,d);require(Path(path)==NP,'never execute full RAM module');return self.native
        node=next(n for n in ast.parse(WORKER.read_bytes(),str(WORKER)).body if isinstance(n,ast.FunctionDef) and n.name=='_selected_ram_available');ns={'C':C,'ModuleType':ModuleType,'ast':ast,'selected_module':selector};exec(compile(ast.Module(body=[node],type_ignores=[]),str(WORKER),'exec'),ns)
        self.factory=ns['_selected_ram_available'];self.value={'config':{'source_files':{str(NP):descriptor(NP),str(RAM):descriptor(RAM)}}}
        self.ramns={'native_process':self.native};available=next(n for n in ast.parse(RAM.read_bytes(),str(RAM)).body if isinstance(n,ast.FunctionDef) and n.name=='available_bytes');exec(compile(ast.Module(body=[available],type_ignores=[]),str(RAM),'exec'),self.ramns)
    def tearDown(self):sys.meta_path.remove(self.trap)
    def backend(self,value):
        self.calls+=1
        if self.error is not None:return False
        value.available_phys,value.total_phys=self.values.pop(0);return True
    def test_original_RAM_and_status_bodies_return_fresh_readings(self):
        before_path=tuple(sys.path);before_modules=dict(sys.modules);fresh=self.factory(self.value)
        self.assertEqual([fresh(),fresh(),fresh()],[11,37,37]);self.assertEqual(self.calls,3);self.assertEqual(self.selected,[str(NP)])
        self.assertEqual(tuple(sys.path),before_path);self.assertEqual(dict(sys.modules),before_modules);self.assertFalse(fresh.__source_binding__['numeric_RAM_cache'])
    def test_original_error_and_nonboolean_RAM_shapes(self):
        fresh=self.factory(self.value);self.error=OSError('inert GlobalMemoryStatusEx failure')
        with self.assertRaises(OSError) as raised:fresh()
        self.assertIs(raised.exception,self.error);self.assertEqual(self.calls,1)
        for bad in ({'available_bytes':True,'total_bytes':100},{'available_bytes':1,'total_bytes':False},{'available_bytes':-1,'total_bytes':100},{'available_bytes':1,'total_bytes':0},{'available_bytes':1,'total_bytes':100,'extra':1}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):self.ramns['available_bytes'](status_call=lambda:bad)
    def test_selected_identity_refusal_before_memory_backend(self):
        fresh=self.factory(self.value);original=self.native.system_memory_status;self.native.system_memory_status=lambda:{'available_bytes':99,'total_bytes':100}
        with self.assertRaises(ValueError):fresh()
        self.native.system_memory_status=original;self.native.__file__=str(RAM)
        with self.assertRaises(ValueError):fresh()
        self.native.__file__=str(NP);self.native._SYSTEM_SAMPLER=SimpleNamespace(memory_status=lambda:{'available_bytes':99,'total_bytes':100})
        with self.assertRaises(ValueError):fresh()
        self.assertEqual(self.calls,0)
    def test_original_guard_consumes_fresh_RAM_and_retains_deadline(self):
        path=PACKAGE/'import_resource_guard.py';require(descriptor(path)['sha256']=='15e98a9c1abf7d0e6700ca035eca469bcd848dede6075cb8222bdfcab0700804','unchanged original import resource guard');module=ModuleType('CPU_resource_guard');module.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
        self.values=[(2*1024**3,8*1024**3),(int(1.25*1024**3)-1,8*1024**3)];now=[10.];fresh=self.factory(self.value)
        guard=module.bind(HERE,{'job_name':'CPU_JOB','supervisor_pid':2},10.,owned_resources=lambda name:{'tree_rss_bytes':1024**3},heartbeat_reader=lambda p:{'unix_seconds':100.,'supervisor_pid':2},available=fresh,monotonic=lambda:now[0],wall=lambda:100.)
        guard('before_import')
        with self.assertRaises(ValueError):guard('after_import')
        self.assertEqual(self.calls,2);now[0]=611.
        with self.assertRaises(ValueError):guard('deadline')
        self.assertEqual(self.calls,2)
    def test_passive_copy_pair_is_separate_from_historical_maps(self):
        path=HERE/'PASSIVE_IMPORT_BOOTSTRAP_ASSUMPTIONS002.json';v=json.loads(path.read_bytes());self.assertEqual(v['sha256'],hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='sha256'},sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest())
        self.assertEqual(v['predecessor_reference']['file_sha256'],'4a47c0cf98946ce5f31ace0f358dcf5e39652b2293ea7b97785f51635cebf3ae');self.assertEqual((len(v['source_modules']),len(v['extension_modules']),len(v['additional_cache_candidates'])),(8,3,8))
        row=v['source_modules']['copy'];self.assertFalse(row['already_observed009']);self.assertFalse(row['source_execution_history_proven']);self.assertEqual(v['source_derived_baseline_count'],39);self.assertFalse(v['copy_source_or_cache_executed']);self.assertEqual(set(v['copy_dependencies_already_historical']),{'types','weakref','copyreg'})
        cp=next(c for c in v['additional_cache_candidates'].values() if c['source_path']==row['path']);self.assertTrue(cp['code_source_equivalence']);self.assertFalse(cp['cache_executed']);self.assertEqual((row['descriptor']['bytes'],cp['descriptor']['bytes']),(8681,11243))
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
receipt={'kind':'CPU_RAM_SOURCE_SLICE_AND_COPY_PASSIVE_CONTROLS','tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'all_controls_completed':result.wasSuccessful(),'actual_original_available_bytes_system_memory_status_Sampler_memory_status_executed_with_inert_backend':result.wasSuccessful(),'NP_complete_module_execution_in_CPU_fixture':False,'actual_resource_guard_source_executed':True,'source_slice_only_CPU_selector':True,'native_API_called':False,'ctypes_Torch_psutil_imported':False,'native_authority_issued':False,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (WORKER,Path(__file__),NP,RAM,PACKAGE/'import_resource_guard.py',HERE/'PASSIVE_IMPORT_BOOTSTRAP_ASSUMPTIONS002.json')}}
output=HERE/'authored_ram_controls001.json';output.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n',encoding='utf-8');raise SystemExit(0 if result.wasSuccessful() else 1)
