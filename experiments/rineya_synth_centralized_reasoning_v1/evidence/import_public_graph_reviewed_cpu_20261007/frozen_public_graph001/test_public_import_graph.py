"""Whole import-only public-shaped source graph; no native process or grant."""
from pathlib import Path
from types import ModuleType
import ast,hashlib,json,shutil,tempfile,unittest
HERE=Path(__file__).resolve().parent;GRAPH=HERE/'public_candidate';PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
def load(p,name):m=ModuleType(name);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
class Controls(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='CPU_import_public_graph_');self.pkg=Path(self.tmp.name)/'experiments/rineya_synth_centralized_reasoning_v1';self.pkg.mkdir(parents=True)
    def tearDown(self):self.tmp.cleanup()
    def mirror(self):
        for p in GRAPH.glob('*.py'):shutil.copy2(p,self.pkg/p.name)
        relative=Path('evidence/tensor_gate_dispatch_reviewed_cpu_20261007/frozen_gate_dispatch002/candidate');target=self.pkg/relative;target.mkdir(parents=True)
        for n in ('prefix_tensor_contract.py','prefix_input_owned_worker.py','prefix_registered_loader.py'):shutil.copy2(PACKAGE/relative/n,target/n)
        shutil.copy2(PACKAGE/'registration_git_join.py',self.pkg/'registration_git_join.py')
        # Measurement source initialization requires only this exact immutable
        # original provider definition file; no provider/API construction.
        relative=Path('evidence/raw_library_pipeline_reviewed_cpu_20261006/history/WORKER_LIBRARY_PUBLICATION_SOURCE_FROZEN_017/source');target=self.pkg/relative;target.mkdir(parents=True);shutil.copy2(PACKAGE/relative/'prefix_library_policy.py',target/'prefix_library_policy.py')
    def test_all_ten_sources_compile_and_reviewed_qualifier_helper_pins(self):
        self.assertEqual(len(list(GRAPH.glob('*.py'))),10)
        for p in GRAPH.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
        self.assertEqual(hashlib.sha256((GRAPH/'import_runtime_qualification.py').read_bytes()).hexdigest(),'9c270b3eeffab6acd69d3d1f618645b5fa7f3d57dff64819088879dce83c0607');self.assertEqual(hashlib.sha256((GRAPH/'import_bootstrap.py').read_bytes()).hexdigest(),'749b093fb05fd58f37fdc3109196e60494ac70d46fc97aaf89c9e0e366b43c5b')
    def test_actual_public_shaped_C_W_SUP_dispatch_link_default_refusal(self):
        self.mirror();C=load(self.pkg/'import_observation_contract.py','CPU_public_C');W=load(self.pkg/'import_observation_worker.py','CPU_public_W');S=load(self.pkg/'import_observation_supervisor.py','CPU_public_SUP');D=load(self.pkg/'import_observation_dispatcher.py','CPU_public_D')
        self.assertEqual(C.PACKAGE,self.pkg);self.assertEqual(C.B.HERE,self.pkg);self.assertEqual(W.C.PACKAGE,self.pkg);self.assertEqual(W.CONTRACT_DESCRIPTOR,D.CONTRACT_DESCRIPTOR);self.assertEqual(D.SUPERVISOR_DESCRIPTOR['sha256'],hashlib.sha256((self.pkg/'import_observation_supervisor.py').read_bytes()).hexdigest());self.assertEqual(len(C.OPERATIONS),11)
        with self.assertRaises(ValueError):C.stage_native_gate()
        with self.assertRaises(ValueError):D.dispatch(self.pkg/'absent.json',grant_reference=None,owner_receipt_reference=None,anchors={})
    def test_exact_durable_dependencies_main_proof_retained_and_no_TorchCUDA_endpoint(self):
        source=(GRAPH/'import_observation_worker.py').read_text();self.assertIn("report['source_qualification']=source_return",source);self.assertIn("'current_main_proof':state.get('current_main_proof')",source);self.assertIn('current_main_proof=main_proof',source)
        tree=ast.parse(source);calls=[ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)];self.assertFalse(any(name in calls for name in ('torch.tensor','torch.ones_like','torch.cuda.init','torch.cuda.mem_get_info','torch.cuda.is_initialized')))
        c=(GRAPH/'import_observation_contract.py').read_text();self.assertNotIn("HERE/'predecessors/",c);self.assertNotIn("HERE/'predecessors/",source);self.assertIn('frozen_gate_dispatch002/candidate',c)
    def test_contract_pin_drift_and_CPUToken_never_native_gate(self):
        self.mirror();C=load(self.pkg/'import_observation_contract.py','CPU_drift_C');cap=C._CPU_capability();context=C._CPU_context({'scope':C.SCOPE,'fixture_only':True},cap);self.assertTrue(C.context_value(context)['fixture_only'])
        with self.assertRaises(ValueError):C.stage_native_gate()
        p=self.pkg/'import_observation_contract.py';p.write_bytes(p.read_bytes()+b'\n# drift\n')
        with self.assertRaisesRegex(ValueError,'descriptor'):load(self.pkg/'import_observation_worker.py','CPU_bad_public_W')
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_WHOLE_IMPORT_ONLY_PUBLIC_LOCATION_GRAPH_CONTROLS','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'public_shaped_temporary_source_graph_only':True,'actual_default_grant_refusal':True,'native_process_or_psutil_imported':False,'native_API_framework_called':False,'native_authority_issued':False,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*GRAPH.glob('*.py')]}}
    if len(__import__('sys').argv)>1:
        with Path(__import__('sys').argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
