"""Real TorchVersion class definition, inert import and actual durable writer."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
from copy import deepcopy
from datetime import datetime,timezone
from contextlib import contextmanager
import ast,hashlib,json,os,sys,tempfile,time,unittest
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1];WORKER=HERE/'import_observation_worker.py'
class Trap:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('torch','ctypes','psutil'):raise AssertionError('forbidden native/framework import '+name)
trap=Trap();sys.meta_path.insert(0,trap)
C=ModuleType('CPU_version_contract');C.__file__=str(PACKAGE/'import_observation_contract.py');exec(compile(Path(C.__file__).read_bytes(),C.__file__,'exec'),C.__dict__)
nodes=[n for n in ast.parse(WORKER.read_bytes(),str(WORKER)).body if isinstance(n,ast.FunctionDef) and n.name in ('original_ledger','_run_import_body','_publish_measurement')]
W=ModuleType('CPU_version_writer_source');W.__file__=str(WORKER);W.__dict__.update(C=C,Path=Path,SimpleNamespace=SimpleNamespace,ast=ast,hashlib=hashlib,deepcopy=deepcopy,datetime=datetime,timezone=timezone,time=time,os=os);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(WORKER),'exec'),W.__dict__)
refs=json.loads((HERE/'VERSION_SOURCE_REFERENCES001.json').read_bytes());torch_version_path=next(Path(p) for p in refs['files'] if Path(p).name=='torch_version.py');version_path=next(Path(p) for p in refs['files'] if Path(p).name=='version.py')
for path in (torch_version_path,version_path):C.require(hashlib.sha256(path.read_bytes()).hexdigest()==refs['files'][str(path)]['sha256'],'exact installed version source bytes')
version_literal=next(ast.literal_eval(n.value) for n in ast.parse(version_path.read_bytes(),str(version_path)).body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='__version__' for x in n.targets))
hooks=[]
class VersionTrap:
    def __init__(self,*args,**kwargs):hooks.append('comparison');raise AssertionError('TorchVersion comparison hook')
namespace={'Any':object,'Iterable':object,'Version':VersionTrap,'InvalidVersion':ValueError}
tv=[n for n in ast.parse(torch_version_path.read_bytes(),str(torch_version_path)).body if isinstance(n,ast.ClassDef) and n.name=='TorchVersion' or isinstance(n,ast.For)]
exec(compile(ast.Module(body=tv,type_ignores=[]),str(torch_version_path),'exec'),namespace);TorchVersion=namespace['TorchVersion']
class Measurement:
    def __init__(self):self.closed=False;self.provider=SimpleNamespace(scope=lambda:{'fixture_only':True},libraries=lambda:[],events=lambda:[])
    @contextmanager
    def framework_import(self):yield
    def close(self):self.closed=True;return {'fixture_only':True}
    def export_raw_packet(self,identity):
        v={'kind':'CPU_INERT_VERSION_METADATA_PACKET','fixture_only':True,'native_PASS':False,'native_authority_issued':False,'closed':self.closed};return {**v,'sha256':C.digest(v)}
class Controls(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.cleanup=[];self.measurement=Measurement();hooks.clear()
    def tearDown(self):self.tmp.cleanup()
    def run_source(self,raw,expected=None):
        output=self.root/('output'+str(len(list(self.root.iterdir()))));output.mkdir();self.output=output
        api={'source_qualification':lambda:{'fixture_only':True},'ownership_verify':lambda:{'fixture_only':True},'resource_guard':lambda stage:None,'framework':lambda:SimpleNamespace(__version__=raw)}
        return W._run_import_body(api,output,lambda:self.measurement,{'fixture_only':True},owned_cleanup=lambda:self.cleanup.append('cleanup') or self.measurement.close(),expected_version=expected)
    def final(self):return json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_bytes())
    def test_actual_TorchVersion_is_plain_exact_metadata_without_comparison_hook(self):
        raw=TorchVersion(version_literal);report=self.run_source(raw,version_literal)
        self.assertEqual(type(report['torch_version']),str);self.assertEqual(report['torch_version'],version_literal);self.assertEqual(report['status'],'MEASUREMENT_RECORDED_UNQUALIFIED');self.assertEqual(report['final_phase_summary']['markers'],22);self.assertEqual(hooks,[]);self.assertTrue(self.measurement.closed)
        # Existing plain CPU fixtures remain usable without a native version pin.
        self.measurement=Measurement();self.assertEqual(self.run_source('CPU_INERT_VERSION')['torch_version'],'CPU_INERT_VERSION')
    def test_version_mismatch_is_failed_inside_journal_and_cleanup_raw_survive(self):
        with self.assertRaisesRegex(ValueError,'exact pinned import version'):self.run_source(TorchVersion('mismatched'),version_literal)
        self.assertTrue(self.measurement.closed);self.assertEqual(self.cleanup,['cleanup']);self.assertTrue((self.output/'IMPORT_LIBRARY_MEASUREMENT.json').is_file());v=self.final();self.assertEqual(v['operations']['torch_version_read'],{'attempted':1,'returned':0,'failed':1,'pending':0});self.assertNotIn('resource_after_import',v['operations']);self.assertEqual(json.loads((self.output/'worker_result.json').read_bytes())['status'],'MEASUREMENT_FAILURE_PRESERVED');self.assertEqual(hooks,[])
    def test_arbitrary_object_string_hook_refused_and_native_pin_source_bound(self):
        class Arbitrary:
            def __str__(self):hooks.append('arbitrary_str');raise AssertionError('arbitrary string hook')
        with self.assertRaisesRegex(ValueError,'Unicode import version'):self.run_source(Arbitrary(),version_literal)
        self.assertEqual(hooks,[]);self.assertTrue(self.measurement.closed);self.assertTrue((self.output/'IMPORT_LIBRARY_MEASUREMENT.json').exists());self.assertEqual(self.final()['operations']['torch_version_read']['failed'],1)
        native=next(n for n in ast.parse(WORKER.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='_run_native_owned');call=next(n for n in ast.walk(native) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_run_import_body');expected=next(k.value for k in call.keywords if k.arg=='expected_version');self.assertEqual(ast.dump(expected),ast.dump(ast.parse("value['config']['torch_version']",mode='eval').body));self.assertEqual(len(C.OPERATIONS),11);self.assertEqual(C.RESOURCES['whole_scope_seconds'],600)
        with self.assertRaises(ValueError):C.stage_native_gate()
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));sys.meta_path.remove(trap)
v={'kind':'CPU_TORCH_VERSION_UNICODE_METADATA_SUCCESSOR_CONTROLS','tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'all_controls_completed':result.wasSuccessful(),'actual_selected_TorchVersion_class_and_comparison_definitions_executed':True,'actual_original_durable_Ledger_used':True,'measurement_and_import_backend_inert':True,'native_mode_binding_source_inspected_only':True,'Torch_ctypes_psutil_imported':False,'native_API_called':False,'native_authority_issued':False,'version_source_references':refs,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (WORKER,Path(__file__),Path(C.__file__))}}
(HERE/'authored_version_controls001.json').write_text(json.dumps(v,sort_keys=True,indent=2)+'\n',encoding='utf-8');raise SystemExit(0 if result.wasSuccessful() else 1)
