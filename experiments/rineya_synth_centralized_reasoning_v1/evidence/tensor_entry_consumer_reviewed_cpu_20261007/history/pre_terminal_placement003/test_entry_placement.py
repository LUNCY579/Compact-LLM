"""Common inert importer acquisition/preparation/handoff in actual entry phases."""
from pathlib import Path
import ast,hashlib,importlib,json,types,unittest
HERE=Path(__file__).resolve().parent

def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
B=load(HERE/'test_observed_admission.py','CPU_placement_importer_fixture');W=load(HERE/'prefix_input_owned_worker.py','CPU_placement_worker');P=load(HERE/'prefix_native_pipeline.py','CPU_placement_pipeline')
class Controls(B.Controls):
    def setUp(self):
        super().setUp();self.original=W.C;W.C=self.C;self.owner={'worker_pid':11,'supervisor_pid':22,'job_name':'CPU_fixture_only'};self.output=self.site.parent/'output';self.output.mkdir();self.events=[]
    def tearDown(self):W.C=self.original;super().tearDown()
    def state(self):return {'inputs':{'context':self.context,'owner':self.owner,'output':self.output}}
    def verify(self,*args):self.events.append('membership');return {'actual_worker_pid':11,'actual_parent_pid':22,'launcher_root_pid':11,'supervisor_pid':22,'job_name':'CPU_fixture_only','owned_job_membership_verified':True,'query_job_handle_closed_before_framework_import':True}
    def prepare(self,context,owner,output,finder):self.events.append('prepare');self.assertIsNone(self.L._check_installation_authority(finder.installed_authority,self.value,finder=finder,_CPU_capability=self.capC));self.assertNotIn('CPU_observed_pkg',self.surface.modules)
    def test_placement_actual_common_install_prepare_handoff_then_package_import(self):
        state=self.state()
        with self.active():
            result=W._prepare_acquired_entry(state,install=lambda *a:self.install(),verify=self.verify,prepare=self.prepare,_CPU_capability=self.capC);self.assertEqual(self.events,['membership','prepare']);self.assertTrue(result['fixture_only']);self.assertFalse(result['native_authority_issued']);entry=W._consume_entry_handoff(state['entry_handoff'],self.context,self.owner,self.output);self.assertIs(entry['finder'],state['acquired_finder']);self.assertEqual(entry['verified'],self.verify())
            module=importlib.import_module('CPU_observed_pkg');self.assertEqual(module.VALUE,19);self.assertIsInstance(module.__loader__,self.L.RegisteredLoader);self.L.remove(entry['finder']);self.assertEqual(self.surface.path_importer_cache,{})
            with self.assertRaises(ValueError):W._consume_entry_handoff(state['entry_handoff'],self.context,self.owner,self.output)
    def test_placement_membership_and_prepare_failure_retain_acquisition_for_cleanup(self):
        for failure_stage in ('verify','prepare'):
            with self.subTest(stage=failure_stage),self.active():
                state=self.state()
                def failed(*args):raise RuntimeError(failure_stage+' failed')
                with self.assertRaisesRegex(RuntimeError,failure_stage+' failed'):W._prepare_acquired_entry(state,install=lambda *a:self.install(),verify=failed if failure_stage=='verify' else self.verify,prepare=failed if failure_stage=='prepare' else self.prepare,_CPU_capability=self.capC)
                self.assertIn('acquired_finder',state);self.assertNotIn('entry_handoff',state);self.L.remove(state['acquired_finder']);self.assertEqual(self.surface.path_importer_cache,{})
    def test_placement_forged_rebound_handoff_and_missing_capability_refused(self):
        with self.active():
            state=self.state()
            with self.assertRaises(ValueError):W._prepare_acquired_entry(state,install=lambda *a:self.fail('closed gate'),verify=self.verify,prepare=self.prepare)
            W._prepare_acquired_entry(state,install=lambda *a:self.install(),verify=self.verify,prepare=self.prepare,_CPU_capability=self.capC)
            with self.assertRaises(ValueError):W._consume_entry_handoff(object(),self.context,self.owner,self.output)
            with self.assertRaises(ValueError):W._consume_entry_handoff(state['entry_handoff'],self.context,{**self.owner,'worker_pid':12},self.output)
            W._consume_entry_handoff(state['entry_handoff'],self.context,self.owner,self.output);self.L.remove(state['acquired_finder'])
    def test_placement_actual_native_source_calls_prepare_in_phase5_and_core_one_use(self):
        tree=ast.parse((HERE/'prefix_input_owned_worker.py').read_bytes());native=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_native_entry_actions');provider=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name=='provider');self.assertIn('_prepare_acquired_entry',ast.unparse(provider));self.assertIn('prepare=state[',ast.unparse(provider));execute=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name=='execute');self.assertIn('_entry_handoff=state[',ast.unparse(execute))
        factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_owned_execution_factory');self.assertIn('pipeline._prepare_owned_main_entry',ast.unparse(factory));self.assertIn("prepared['consumed'] = True",ast.unparse(factory));p=ast.parse((HERE/'prefix_native_pipeline.py').read_bytes());prepare=next(n for n in p.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_owned_main_entry');text=ast.unparse(prepare);self.assertIn('worker.original_guard',text);self.assertIn('host = Host()',text);self.assertIn('return run',text);self.assertIn('one source-owned prepared core consumption',text)
    def test_placement_roots_failure_always_restores_owned_meter_functions(self):
        raw=Path(self.C.__file__).read_bytes();self.C.__compiled_source_descriptor__={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'};meter=W._SelectedSourceMeter();original=self.C.read_bytes;meter.bind_contract(self.C)
        actions,execute=W._native_entry_actions(types.SimpleNamespace(),types.SimpleNamespace(),P.STARTUP_PHASES)
        with self.active():
            state={'source_meter':meter,'transition':{'original_container':self.surface.path,'path_after':['authored foreign mutation'],'before':tuple(self.surface.path)}}
            with self.assertRaisesRegex(ValueError,'transition unchanged'):actions[P.STARTUP_PHASES[5]](state,{})
        self.assertIs(self.C.read_bytes,original);self.assertNotIn('__source_meter__',self.C.__dict__)
if __name__=='__main__':
    names=[n for n in Controls.__dict__ if n.startswith('test_placement_')];result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Controls(n) for n in names));r={'kind':'CPU_PHASE5_ACQUIRED_COMMON_IMPORTER_PREPARATION_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_controls_completed':result.wasSuccessful(),'actual_common_registered_importer_executed':True,'actual_private_acquisition_helper_executed':True,'native_execution_factory_pipeline_prepare_only_source_inspected':True,'private_inert_contract_capability_only':True,'native_API_framework_calls':False,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('prefix_input_owned_worker.py','prefix_native_pipeline.py','prefix_registered_loader.py','prefix_tensor_contract.py','test_entry_placement.py')}}
    if len(__import__('sys').argv)>1:
        with Path(__import__('sys').argv[1]).open('x',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
