"""Frozen current main frame/function equivalence, no direct process/history."""
from pathlib import Path
from unittest.mock import patch
import ast, hashlib, importlib.machinery, json, sys, types, unittest
HERE=Path(__file__).resolve().parent;WORK=HERE.parent/'final_direct_main001';SELECTED=WORK/'prefix_input_owned_worker.py';FILES=[p for p in WORK.iterdir() if p.is_file()]
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins()
if BEFORE[str(SELECTED)]['sha256']!='6cd0a9fd598de3fc8bacc494eb67189f7bfa9963c280100d0f396ee315cc1828':raise ValueError('frozen worker pin')
if BEFORE[str(WORK/'test_direct_main_provenance.py')]['sha256']!='80d65033d8ba578806b9f2c5e8c3529e32cc7d0bcfb9ad49f5a768e4d4d67428':raise ValueError('frozen test pin')
def load(path,name):
    raw=path.read_bytes();m=types.ModuleType(name);m.__file__=str(path);m.__compiled_source_descriptor__={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'};exec(compile(raw,str(path),'exec'),m.__dict__);return m
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
TRAP=Trap();sys.meta_path.insert(0,TRAP);DETAILS={}
class Controls(unittest.TestCase):
    def setUp(self):
        raw=SELECTED.read_bytes();tree=ast.parse(raw,str(SELECTED));dispatch=tree.body[-1];self.assertIsInstance(dispatch,ast.If);self.assertEqual(ast.unparse(dispatch.test),"__name__ == '__main__'");self.assertEqual(ast.unparse(dispatch.body[0]),'raise SystemExit(main())');tree.body.pop()
        self.m=types.ModuleType('__main__');self.m.__file__=str(SELECTED);self.m.__spec__=None;self.m.__cached__=None;self.m.__loader__=importlib.machinery.SourceFileLoader('__main__',str(SELECTED));exec(compile(tree,str(SELECTED),'exec'),self.m.__dict__);self.C=load(WORK/'prefix_tensor_contract.py','independent_direct_contract');self.C.PACKAGE=WORK;self.m.C=self.C;self.cap=self.C._mint_CPU_native_context_fixture();self.descriptor={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'};self.value={'fixture_only':True,'config':{'source_files':{str(SELECTED):self.descriptor}}}
    def verify(self):
        with patch.dict(sys.modules,{'__main__':self.m}):return self.m._CPU_main_provenance_probe(self.value,sys,_capability=self.cap)
    def test_actual_current_frame_functions_prepare_and_counted_reads(self):
        meter=self.m._SelectedSourceMeter();costs={};meter.enter('startup_source_qualification',costs);meter.bind_contract(self.C)
        try:
            observed=self.verify()
            with patch.dict(sys.modules,{'__main__':self.m}):prepared=self.m._binding_compiled_source(self.value)
            self.assertEqual(observed['loader_kind'],'direct_script_current_frame_source_equivalence');self.assertTrue(observed['current_frame_source_equivalent']);self.assertTrue(observed['current_functions_source_equivalent']);self.assertFalse(observed['current_module_execution_observed']);self.assertFalse(observed['source_execution_history_proven']);self.assertIsNone(vars(self.m).get('__compiled_source_descriptor__'));self.assertEqual(prepared['compiled_source'],observed)
            rows=meter.snapshot()['startup_source_qualification'];self.assertEqual(rows['contract.read_bytes.source']['returned'],1);self.assertEqual(rows['contract.file_descriptor.source']['returned'],2);self.assertEqual(costs['source_bytes'],3*self.descriptor['bytes']);DETAILS['actual_read_costs']=costs;DETAILS['source_boundary_rows']=rows
        finally:meter.leave();meter.restore()
    def test_bad_hash_wrong_namespace_bad_capability_refused(self):
        self.value['config']['source_files'][str(SELECTED)]={**self.descriptor,'sha256':'f'*64}
        with self.assertRaises(ValueError):self.verify()
        self.value['config']['source_files'][str(SELECTED)]=self.descriptor
        with self.assertRaises(ValueError):self.m._CPU_main_provenance_probe(self.value,sys,_capability=self.cap)
        with self.assertRaises(ValueError):self.m._CPU_main_provenance_probe(self.value,sys,_capability=object())
    def test_function_object_and_code_rebind_refused_before_preparation(self):
        self.verify();original=self.m.exact_module;self.m.exact_module=lambda *a:None
        with patch.dict(sys.modules,{'__main__':self.m}),self.assertRaises(ValueError):self.m._binding_compiled_source(self.value)
        self.m.exact_module=original;code=original.__code__;original.__code__=(lambda *a:None).__code__
        try:
            with patch.dict(sys.modules,{'__main__':self.m}),self.assertRaises(ValueError):self.m._binding_compiled_source(self.value)
        finally:original.__code__=code
    def test_explicit_import_branch_and_native_phase1_source_order(self):
        imported=load(SELECTED,'independent_imported_worker');imported.C.PACKAGE=WORK;observed=imported._binding_compiled_source({'config':{'source_files':{str(SELECTED):self.descriptor}}})['compiled_source'];self.assertEqual(observed['loader_kind'],'explicit_selected_source_compile');self.assertTrue(observed['current_module_execution_observed'])
        with self.assertRaises(ValueError):imported.C.stage_native_gate()
        tree=ast.parse(SELECTED.read_bytes());native=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_native_entry_actions');source=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name=='source');calls=[n.func.id for n in ast.walk(source) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)];self.assertIn('_registered_main_inputs',calls);self.assertIn('_verify_direct_main_source',calls);self.assertLess(calls.index('_registered_main_inputs'),calls.index('_verify_direct_main_source'));DETAILS['native_phase1_only_source_inspected']=True
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));sys.meta_path.remove(TRAP);after=pins();receipt={'kind':'INDEPENDENT_FROZEN_DIRECT_MAIN001_REVIEW','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_controls_completed':result.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'frozen_sources_unchanged':BEFORE==after,'details':DETAILS,'fixture_AST_adaptation':'exact one final __main__ dispatch removed; source functions/frame bodies unchanged','fixture_contract_PACKAGE':str(WORK),'actual_direct_script_process_executed':False,'initial_source_load_history_proven':False,'native_phase1_closure_executed':False,'native_import_trap_active':True,'native_API_or_framework_called':False,'native_authority_issued':False}
    receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    receipt['first_fixture_counter_key_error_preserved']='direct_main001_first_review_key_failure'
    receipt['sha256']=hashlib.sha256(json.dumps({k:v for k,v in receipt.items() if k!='sha256'},sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_DIRECT_MAIN001_REVIEW_SUCCESSOR.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() and BEFORE==after else 1)
