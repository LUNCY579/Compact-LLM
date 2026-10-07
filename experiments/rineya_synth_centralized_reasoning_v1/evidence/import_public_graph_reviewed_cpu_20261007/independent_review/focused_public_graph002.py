"""Frozen whole public-source graph review; temporary CPU mirror only."""
from pathlib import Path
from types import ModuleType
import ast,hashlib,importlib,json,shutil,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent;FROZEN=HERE.parent/'frozen_public_graph002';GRAPH=FROZEN/'sources';PACKAGE=next(p for p in FROZEN.parents if p.name=='rineya_synth_centralized_reasoning_v1');PLAN=FROZEN/'PUBLIC_IMPORT_SOURCE_PLAN002.json';FILES=[p for p in FROZEN.rglob('*') if p.is_file()]
plan=json.loads(PLAN.read_bytes())
for row in plan['sources'].values():
    p=Path(row['target_path'])
    if p.exists():FILES.append(p)
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(PLAN)]['sha256']=='bff53827a90a973e6558aa2505aa0b27c9028385a2f06ac14ddca35846bc36b8';assert BEFORE[str(FROZEN/'test_public_import_graph.py')]['sha256']=='327790b8f3d6eebf2acb1b5276e9c3d87abf1c4d58b03e633391c4afc9d784b0'
def load(p,name):m=ModuleType(name);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker','native_process','pynvml'):raise AssertionError('native import forbidden '+fullname)
trap=Trap();sys.meta_path.insert(0,trap);DETAILS={}
class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='independent_whole_import_graph_');self.pkg=Path(self.tmp.name)/'experiments/rineya_synth_centralized_reasoning_v1';self.pkg.mkdir(parents=True)
        shutil.copytree(FROZEN/'durable_dependencies',self.pkg,dirs_exist_ok=True)
        for p in GRAPH.glob('*.py'):shutil.copy2(p,self.pkg/p.name)
    def tearDown(self):self.tmp.cleanup()
    def test_exact_ten_source_targets_dependencies_compile_and_closed_plan(self):
        self.assertEqual(len(plan['sources']),10)
        for name,row in plan['sources'].items():
            raw=(GRAPH/name).read_bytes();self.assertEqual(row['descriptor'],{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'});self.assertEqual(Path(row['target_path']),PACKAGE/name);public=Path(row['target_path']);self.assertEqual(public.exists(),row['current_public_descriptor'] is not None)
            if public.exists():self.assertEqual(BEFORE[str(public)],{k:row['current_public_descriptor'][k] for k in ('bytes','sha256')})
            compile(raw,str(GRAPH/name),'exec')
        for path,d in plan['durable_dependencies'].items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),d['sha256']);self.assertEqual(Path(path).stat().st_size,d['bytes'])
        oldQ=PACKAGE/'artifacts/torch_import_observation_work/native_candidate/frozen_public_graph001/sources/import_runtime_qualification.py';a=ast.parse(oldQ.read_bytes());b=ast.parse((GRAPH/'import_runtime_qualification.py').read_bytes());changes=[]
        for left,right in zip(a.body,b.body):
            if ast.dump(left,include_attributes=False)!=ast.dump(right,include_attributes=False):
                self.assertIsInstance(left,ast.Assign);self.assertIsInstance(right,ast.Assign);self.assertEqual(ast.dump(left.targets,include_attributes=False) if False else [ast.unparse(x) for x in left.targets],[ast.unparse(x) for x in right.targets]);changes.append(ast.unparse(left.targets[0]))
        self.assertEqual(changes,['PACKAGE','SOURCE']);self.assertEqual(len(a.body),len(b.body));DETAILS['qualifier_only_location_assignments_changed']=changes
        self.assertFalse(plan['source_public_promotion_performed']);self.assertFalse(plan['native_launch_authorized']);self.assertIsNone(plan['owner_grant_reference']);self.assertIsNone(plan['owner_receipt_reference'])
    def test_actual_C_W_SUP_dispatcher_links_absent_grant_and_crossscope_refuse(self):
        Q=load(self.pkg/'import_runtime_qualification.py','independent_public_Q');self.assertEqual(Q.PACKAGE,self.pkg);self.assertEqual(Q.B.PACKAGE,self.pkg);self.assertEqual(Q.SOURCE,self.pkg/'evidence/tensor_gate_dispatch_reviewed_cpu_20261007/frozen_gate_dispatch002/candidate/prefix_tensor_contract.py');C=load(self.pkg/'import_observation_contract.py','independent_public_C');W=load(self.pkg/'import_observation_worker.py','independent_public_W');S=load(self.pkg/'import_observation_supervisor.py','independent_public_S');D=load(self.pkg/'import_observation_dispatcher.py','independent_public_D');self.assertEqual(C.PACKAGE,self.pkg);self.assertEqual(C.B.HERE,self.pkg);self.assertEqual(W.C.PACKAGE,self.pkg);self.assertEqual(W.CONTRACT_DESCRIPTOR,D.CONTRACT_DESCRIPTOR);self.assertEqual(D.SUPERVISOR_DESCRIPTOR['sha256'],hashlib.sha256((self.pkg/'import_observation_supervisor.py').read_bytes()).hexdigest())
        with self.assertRaises(ValueError):D.dispatch(self.pkg/'absent.json',grant_reference=None,owner_receipt_reference=None,anchors={})
        cap=C._CPU_capability()
        with self.assertRaises(ValueError):C._CPU_context({'scope':'ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY','fixture_only':True},cap)
        with self.assertRaises(ValueError):C._CPU_context({'scope':'NO_MODEL_NATIVE_LIBRARY_OBSERVER_LOAD_UNLOAD','fixture_only':True},cap)
        with self.assertRaises(ValueError):C.stage_native_gate()
    def test_actual_contract_drift_before_worker_and_CPU_context_no_gate(self):
        C=load(self.pkg/'import_observation_contract.py','independent_drift_C');cap=C._CPU_capability();context=C._CPU_context({'scope':C.SCOPE,'fixture_only':True},cap);self.assertTrue(C.context_value(context)['fixture_only'])
        with self.assertRaises(ValueError):C.stage_native_gate()
        p=self.pkg/'import_observation_contract.py';p.write_bytes(p.read_bytes()+b'\n# drift\n')
        with self.assertRaisesRegex(ValueError,'descriptor'):load(self.pkg/'import_observation_worker.py','independent_bad_W')
    def test_native_whole_source_order_retained_main_proof_and_no_tensor_calls(self):
        source=(GRAPH/'import_observation_worker.py').read_text(encoding='utf-8');self.assertIn("report['source_qualification']=source_return",source);self.assertIn("'current_main_proof':state.get('current_main_proof')",source);tree=ast.parse(source);calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)];self.assertFalse(any(x in calls for x in ('torch.tensor','torch.ones_like','torch.cuda.init','torch.cuda.mem_get_info','torch.cuda.is_initialized')))
        ct=ast.parse((GRAPH/'import_observation_contract.py').read_bytes());fn=next(n for n in ct.body if isinstance(n,ast.FunctionDef) and n.name=='validate_native_manifest');text=ast.unparse(fn);self.assertLess(text.index('qualifier.qualify_import_value'),text.index('return _mint_import_native'));self.assertNotIn("HERE/'predecessors/",source);self.assertFalse(plan['known_current_process_library_expectation']['actually_observed']);DETAILS['native_whole_path_source_only_inspected']=True
    def test_actual_final_worker_registered_source_measurement_and_independent_replay(self):
        fixture=load(PACKAGE/'evidence/torch_import_measurement_reviewed_cpu_20261007/frozen_measurement004/test_measurement.py','independent_inert_measure_fixture');base=fixture.Controls();base.setUp()
        try:
            W=load(self.pkg/'import_observation_worker.py','independent_actual_final_W');L=load(self.pkg/'evidence/tensor_gate_dispatch_reviewed_cpu_20261007/frozen_gate_dispatch002/candidate/prefix_registered_loader.py','independent_actual_L');A=load(self.pkg/'import_observation_auditor.py','independent_actual_A');output=base.root/'output';output.mkdir();name='CPU_FINAL_REGISTERED_FRAMEWORK';source=base.root/(name+'.py');source.write_bytes(b"__version__='CPU_FINAL_ONLY'\ndef tensor(*a,**k):raise AssertionError('forbidden tensor')\n");desc={'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'role':'python_source'};state={};observed={};samples=[]
            def factory():state['guard']=fixture.G.MeasurementGuard(base.policy,base.grammar,W.C,base.P);return state['guard']
            def framework():
                guard=state['guard']
                with guard.extension(base.paths['one.pyd'],'create'):base.P.add(base.paths['one.pyd'])
                with guard.extension(base.paths['one.pyd'],'exec'):base.P.add(base.paths['nested.dll'])
                loader=L.RegisteredLoader(importlib.machinery.SourceFileLoader(name,str(source)),str(source),desc,W.C,observed,library_guard=guard);spec=importlib.machinery.ModuleSpec(name,loader,origin=str(source));module=importlib.util.module_from_spec(spec);loader.exec_module(module);return module
            api={'source_qualification':lambda:{'fixture_only':True,'current_main_proof':{'authored_fixture':True}},'ownership_verify':lambda:{'fixture_only':True},'resource_guard':lambda stage:samples.append(stage),'framework':framework};cap=W.C._CPU_capability();context=W.C._CPU_context({'scope':W.C.SCOPE,'fixture_only':True},cap);report=W._CPU_run_import_fixture(context,api,output,factory,base.identity,_capability=cap);self.assertEqual(report['status'],'MEASUREMENT_RECORDED_UNQUALIFIED');self.assertEqual(report['final_phase_summary']['markers'],22);self.assertEqual(list(report['operation_accounting']),list(W.C.OPERATIONS));self.assertTrue(observed[name]['current_module_execution_observed']);self.assertTrue(base.P.closed);self.assertEqual(samples,['before_import','after_import']);self.assertEqual(report['tensor_allocations'],0)
            replay=A.replay((output/'IMPORT_LIBRARY_MEASUREMENT.json').read_bytes(),policy=base.policy,grammar=base.grammar,identity=base.identity,files=base.files,sources=base.sources);self.assertEqual(replay['status'],'MEASUREMENT_RECORDED_UNQUALIFIED');self.assertFalse(replay['native_authority_issued']);DETAILS['actual_final_worker_inert_flow_markers']=22
        finally:base.tearDown()
    def test_resource_guard_helper_interface_and_original_128_600_bounds(self):
        R=load(self.pkg/'import_resource_guard.py','independent_public_resource');self.assertTrue(callable(R.bind));W=load(self.pkg/'import_observation_worker.py','independent_bound_W');self.assertEqual(W.C.RESOURCES['whole_scope_seconds'],600);self.assertEqual(len(W.C.OPERATIONS),11);output=self.pkg/'out';output.mkdir();ledger=W.original_ledger(output)
        try:
            for n in range(64):ledger.call('CPU_'+str(n),lambda:None)
            called=[]
            with self.assertRaises(ValueError):ledger.call('129',lambda:called.append(True))
            self.assertEqual(called,[]);self.assertEqual(ledger.markers,128)
        finally:ledger.close()
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));sys.meta_path.remove(trap);after=pins();receipt={'kind':'INDEPENDENT_WHOLE_PUBLIC_IMPORT_GRAPH002_CPU_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_and_public_pins_before':BEFORE,'source_and_public_pins_after':after,'pins_unchanged':BEFORE==after,'details':DETAILS,'public_shaped_temporary_mirror_only':True,'actual_final_worker_Ledger_RegisteredLoader_measurement_replay_inert_flow':True,'ownership_resource_mainproof_callbacks_inert':True,'default_grant_refusal_actual':True,'native_whole_factory_Supervisor_psutil_or_worker_main_process_executed':False,'known39_baseline_observed':False,'public_promotion_performed':False,'full_terminal_audit_complete':False,'prior_graph001_failure_preserved':True,'native_API_or_framework_called':False,'native_authority_issued':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_PUBLIC_GRAPH002.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
