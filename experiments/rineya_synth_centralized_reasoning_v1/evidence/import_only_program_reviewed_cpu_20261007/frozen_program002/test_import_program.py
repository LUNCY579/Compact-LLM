"""Actual bounded import-only writer, registered source loader and effect replay."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import hashlib,importlib,importlib.machinery,json,sys,unittest
HERE=Path(__file__).resolve().parent

def load(path,name):m=ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
B=load(HERE/'test_measurement.py','CPU_measure_fixture');W=load(HERE/'import_observation_worker.py','CPU_import_worker');A=load(HERE/'import_observation_auditor.py','CPU_import_auditor');L=load(HERE/'predecessors/prefix_registered_loader.py','CPU_import_loader')
class Controls(B.Controls):
    def setUp(self):
        super().setUp();self.output=self.root/'output';self.output.mkdir();self.program_cap=W.C._CPU_capability();self.context=W.C._CPU_context({'fixture_only':True,'scope':W.C.SCOPE},self.program_cap);self.forbidden=[];self.observed={};self.program_guard=None;self.guard_samples=[]
        self.module_name='CPU_registered_framework';self.source=self.root/(self.module_name+'.py');self.source.write_bytes(b"__version__='CPU_REGISTERED_NO_FRAMEWORK'\ndef tensor(*a,**k):raise AssertionError('forbidden tensor')\ndef ones_like(*a,**k):raise AssertionError('forbidden ones_like')\nclass CUDA:\n def __getattr__(self,n):raise AssertionError('forbidden CUDA '+n)\ncuda=CUDA()\n")
        self.source_desc={'bytes':self.source.stat().st_size,'sha256':hashlib.sha256(self.source.read_bytes()).hexdigest(),'role':'python_source'}
    def tearDown(self):sys.modules.pop(self.module_name,None);super().tearDown()
    def run_program(self,fail_import=False):
        def factory():
            self.program_guard=B.G.MeasurementGuard(self.policy,self.grammar,W.C,self.P);return self.program_guard
        def framework():
            with self.program_guard.extension(self.paths['one.pyd'],'create'):self.P.add(self.paths['one.pyd'])
            with self.program_guard.extension(self.paths['one.pyd'],'exec'):self.P.add(self.paths['nested.dll'])
            spec=importlib.machinery.ModuleSpec(self.module_name,None,origin=str(self.source));actual=importlib.machinery.SourceFileLoader(self.module_name,str(self.source));registered=L.RegisteredLoader(actual,str(self.source),self.source_desc,W.C,self.observed,library_guard=self.program_guard);spec.loader=registered;module=importlib.util.module_from_spec(spec);registered.exec_module(module)
            if fail_import:raise RuntimeError('inert import failed')
            return module
        api={'source_qualification':lambda:W.C.file_descriptor(self.source,self.source_desc),'ownership_verify':lambda:{'fixture_only':True},'resource_guard':lambda stage:self.guard_samples.append(stage),'framework':framework}
        return W._CPU_run_import_fixture(self.context,api,self.output,factory,self.identity,_capability=self.program_cap)
    def test_program_registered_source_import_version_raw_replay_no_postimport_APIs(self):
        report=self.run_program();self.assertEqual(report['status'],'MEASUREMENT_RECORDED_UNQUALIFIED');self.assertEqual(report['torch_version'],'CPU_REGISTERED_NO_FRAMEWORK');self.assertEqual(list(report['operation_accounting']),list(W.C.OPERATIONS));self.assertEqual(report['final_phase_summary']['markers'],22);self.assertTrue(self.observed[self.module_name]['current_module_execution_observed']);self.assertTrue(self.P.closed);self.assertEqual(self.guard_samples,['before_import','after_import']);self.assertEqual(report['tensor_allocations'],0);self.assertFalse(report['user_CUDA_context_requested'])
        raw=(self.output/'IMPORT_LIBRARY_MEASUREMENT.json').read_bytes();result=A.replay(raw,policy=self.policy,grammar=self.grammar,identity=self.identity,files=self.files,sources=self.sources);self.assertEqual(result['status'],'MEASUREMENT_RECORDED_UNQUALIFIED');self.assertFalse(result['native_qualification']);self.assertFalse(result['native_authority_issued'])
    def test_program_failed_import_preserves_actual_measurement_and_cleanup(self):
        with self.assertRaisesRegex(RuntimeError,'inert import failed'):self.run_program(True)
        self.assertTrue(self.P.closed);self.assertTrue((self.output/'IMPORT_LIBRARY_MEASUREMENT.json').exists());result=A.replay((self.output/'IMPORT_LIBRARY_MEASUREMENT.json').read_bytes(),policy=self.policy,grammar=self.grammar,identity=self.identity,files=self.files,sources=self.sources);self.assertEqual(result['status'],'MEASUREMENT_FAILURE_PRESERVED')
    def test_program_context_grant_scope_and_forbidden_calls_stay_closed(self):
        with self.assertRaises(ValueError):W.C.stage_native_gate()
        with self.assertRaises(ValueError):W.C.native_context()
        with self.assertRaises(ValueError):W._CPU_run_import_fixture(object(),{},self.output,None,self.identity,_capability=self.program_cap)
        with self.assertRaises(ValueError):W.C._CPU_context({'fixture_only':True,'scope':'ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY'},self.program_cap)
        report=self.run_program();self.assertFalse(report['native_PASS']);self.assertEqual(report['model_forwards'],0)
        module=ModuleType('forbidden_api_probe');exec(compile(self.source.read_bytes(),str(self.source),'exec'),module.__dict__)
        with self.assertRaises(AssertionError):module.tensor()
        with self.assertRaises(AssertionError):module.cuda.init()
    def test_program_writer_same_cap_and_original_core_unchanged(self):
        ledger=W.original_ledger(self.output)
        try:
            for n in range(64):ledger.call('CPU_cap_'+str(n),lambda:None)
            with self.assertRaises(ValueError):ledger.call('129',lambda:None)
            self.assertEqual(ledger.markers,128);self.assertEqual(W.C.RESOURCES['whole_scope_seconds'],600);self.assertEqual(hashlib.sha256((W.C.B.CORE/'pipeline.py').read_bytes()).hexdigest(),W.C.B.CORE_PINS['pipeline.py']);self.assertEqual(len(W.C.OPERATIONS),11)
        finally:ledger.close()
if __name__=='__main__':
    names=[n for n in Controls.__dict__ if n.startswith('test_program_')];r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Controls(n) for n in names));v={'kind':'CPU_ACTUAL_IMPORT_ONLY_PROGRAM_REGISTERED_SOURCE_MEASUREMENT_CONTROLS','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'actual_original_Ledger_RegisteredLoader_MeasurementGuard_replay':True,'measurement_factory_called_inside_ledger':True,'unused_fixture_guard_constructed_outside_program':True,'native_binding_main_implemented':False,'native_API_framework_called':False,'native_authority_issued':False,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'import_observation_contract.py',HERE/'import_observation_worker.py',HERE/'import_observation_auditor.py',HERE/'measurement_guard.py',HERE/'measurement_replay.py']}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
