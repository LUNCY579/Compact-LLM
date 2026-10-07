"""Limited current function/frame equivalence under a CPU __main__ fixture."""
from pathlib import Path
import ast,hashlib,importlib.machinery,json,sys,types,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
SELECTED=HERE/'prefix_input_owned_worker.py'

def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
class Controls(unittest.TestCase):
    def setUp(self):
        raw=SELECTED.read_bytes();tree=ast.parse(raw,str(SELECTED));self.assertIsInstance(tree.body[-1],ast.If);tree.body.pop();self.m=types.ModuleType('__main__');self.m.__file__=str(SELECTED);self.m.__spec__=None;self.m.__cached__=None;self.m.__loader__=importlib.machinery.SourceFileLoader('__main__',str(SELECTED));exec(compile(tree,str(SELECTED),'exec'),self.m.__dict__);self.C=load(HERE/'prefix_tensor_contract.py','CPU_main_contract');self.C.PACKAGE=HERE;self.m.C=self.C;self.cap=self.C._mint_CPU_native_context_fixture();self.descriptor={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'};self.value={'fixture_only':True,'config':{'source_files':{str(SELECTED):self.descriptor}}}
    def verify(self):
        with patch.dict(sys.modules,{'__main__':self.m}):return self.m._CPU_main_provenance_probe(self.value,sys,_capability=self.cap)
    def test_direct_function_frame_source_equivalence_and_prepare_record(self):
        meter=self.m._SelectedSourceMeter();self.C.__compiled_source_descriptor__={'bytes':(HERE/'prefix_tensor_contract.py').stat().st_size,'sha256':hashlib.sha256((HERE/'prefix_tensor_contract.py').read_bytes()).hexdigest()};costs={};meter.enter('startup_source_qualification',costs);meter.bind_contract(self.C)
        try:
            observation=self.verify();self.assertEqual(observation['loader_kind'],'direct_script_current_frame_source_equivalence');self.assertTrue(observation['fixture_only']);self.assertFalse(observation['source_execution_history_proven']);self.assertFalse(observation['current_module_execution_observed']);self.assertIsNone(self.m.__dict__.get('__compiled_source_descriptor__'))
            with patch.dict(sys.modules,{'__main__':self.m}):prepared=self.m._binding_compiled_source(self.value)
            self.assertEqual(prepared['compiled_source']['loader_kind'],observation['loader_kind']);self.assertGreaterEqual(costs['source_bytes'],2*self.descriptor['bytes'])
        finally:meter.leave();meter.restore()
    def test_bad_source_descriptor_and_fake_main_namespace_refused(self):
        self.value['config']['source_files'][str(SELECTED)]={**self.descriptor,'sha256':'0'*64}
        with self.assertRaises(ValueError):self.verify()
        self.value['config']['source_files'][str(SELECTED)]=self.descriptor
        with self.assertRaises(ValueError):self.m._CPU_main_provenance_probe(self.value,sys,_capability=self.cap)
        with self.assertRaises(ValueError):self.m._CPU_main_provenance_probe(self.value,sys,_capability=object())
    def test_rebound_function_code_rejected_before_prepare(self):
        self.verify();original=self.m.exact_module;self.m.exact_module=lambda *a:None
        with patch.dict(sys.modules,{'__main__':self.m}),self.assertRaises(ValueError):self.m._binding_compiled_source(self.value)
        self.m.exact_module=original
        code=original.__code__;original.__code__=(lambda *a:None).__code__
        try:
            with patch.dict(sys.modules,{'__main__':self.m}),self.assertRaises(ValueError):self.m._binding_compiled_source(self.value)
        finally:original.__code__=code
    def test_imported_module_route_retains_actual_explicit_source_descriptor(self):
        imported=load(SELECTED,'CPU_imported_worker');raw=SELECTED.read_bytes();imported.__compiled_source_descriptor__={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};imported.C.PACKAGE=HERE;value={'config':{'source_files':{str(SELECTED):self.descriptor}}};observation=imported._binding_compiled_source(value)['compiled_source'];self.assertEqual(observation['loader_kind'],'explicit_selected_source_compile');self.assertNotEqual(observation['loader_kind'],'direct_script_current_frame_source_equivalence');self.assertFalse(imported.C.stage_native_gate.__name__=='opened_native_gate')
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_CURRENT_MAIN_FUNCTION_FRAME_SOURCE_EQUIVALENCE_CONTROLS','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'module_initialization_AST_adaptation':'one final __main__ dispatch removed','source_function_and_probe_frame_bodies_changed':False,'actual_direct_script_process_executed':False,'initial_source_load_history_proven':False,'native_API_framework_called':False,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('prefix_input_owned_worker.py','prefix_tensor_contract.py','test_direct_main_provenance.py')}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
