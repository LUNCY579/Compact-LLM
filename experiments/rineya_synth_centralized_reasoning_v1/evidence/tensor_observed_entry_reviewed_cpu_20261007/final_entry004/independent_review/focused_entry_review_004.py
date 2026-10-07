"""Private inert entry closures and original writer/core; native entry unexecuted."""
from pathlib import Path
import ast,hashlib,json,runpy,sys,unittest
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
FILES=('prefix_input_owned_worker.py','prefix_native_pipeline.py','prefix_tensor_contract.py','test_entry_envelope.py')
before={name:hashlib.sha256((WORK/name).read_bytes()).hexdigest() for name in FILES}
assert before=={'prefix_input_owned_worker.py':'8603e2a9ec2d35bb4bee21032e6fc5b986fe8ea9b8c601d079e10d9621bddf2e','prefix_native_pipeline.py':'a2bddf5e5195add314f55eb0677fa856424596d23604e68b22629acdd983a483','prefix_tensor_contract.py':'fd86efe022e1c2f36b5a431b37e271cedfd4dd1654ffd3d6d7e0890acea8895f','test_entry_envelope.py':'60af1e9e28e4ed5fa7390a04885e80f7bfa0795619ee698df916f80861495076'}
ns=runpy.run_path(str(WORK/'test_entry_envelope.py'),run_name='independent_entry_fixture');Base=ns['Controls'];W,P,F=ns['W'],ns['P'],ns['F']
TRACES=[]
class IndependentControls(Base):
    def test_real_shared_entry_core_durable_action_returns_costs_and_owner_seal(self):
        actions,execute=self.fixture();original=actions[self.phases[0]]
        actions[self.phases[0]]=lambda state,costs:('original_action_return',original(state,costs))
        original_execute=execute
        def check_execute(state,ledger):
            self.assertEqual(state[self.phases[0]][0],'original_action_return');return original_execute(state,ledger)
        report=W._entry_phase_body(self.ledger,self.phases,actions,check_execute);saved=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_bytes());rows=[json.loads(line) for line in (self.output/'PHASE_OBSERVATIONS.jsonl').read_bytes().splitlines()]
        self.assertTrue(saved['whole_scope_success_validated']);self.assertEqual(self.events,list(self.phases));self.assertEqual(len(rows),saved['markers']);self.assertLessEqual(len(rows),128)
        self.assertEqual(saved['startup_subcosts'][self.phases[0]]['source_bytes'],7);self.assertEqual(saved['IO']['ledger_close']['returned'],1);self.assertEqual(saved['IO']['final_summary_write']['returned'],1)
        self.assertEqual(len(report['entry_bootstrap_sources']),3);TRACES.append({'markers':len(rows),'order':saved['whole_scope_order'],'bootstrap_source_records':report['entry_bootstrap_sources']})
    def test_source_failure_skips_core_cleanup_and_preserves_original_exception(self):
        failure=RuntimeError('independent entry source failure');events=[]
        def action(name):
            def run(state,costs):
                events.append(name);costs['source_reads']=1
                if name==self.phases[0]:raise failure
                return None
            return run
        actions={name:action(name) for name in self.phases}
        with self.assertRaises(RuntimeError) as caught:W._entry_phase_body(self.ledger,self.phases,actions,lambda *args:self.fail('core executed after source failure'))
        self.assertIs(caught.exception,failure);self.assertEqual(events,[self.phases[0],self.phases[5]]);saved=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_bytes());self.assertFalse(saved['whole_scope_success_validated']);self.assertEqual(saved['operations'][self.phases[0]]['failed'],1);self.assertEqual(saved['operations'][self.phases[0]]['pending'],0)
    def test_primary_cleanup_and_final_summary_failure_keep_failed_costs(self):
        actions,execute=self.fixture();primary=RuntimeError('independent provider failure');cleanup=OSError('independent cleanup failure')
        actions[self.phases[4]]=lambda *args:(_ for _ in ()).throw(primary);actions[self.phases[5]]=lambda *args:(_ for _ in ()).throw(cleanup)
        with self.assertRaises(RuntimeError) as caught:W._entry_phase_body(self.ledger,self.phases,actions,execute)
        self.assertIs(caught.exception,primary);self.assertIs(caught.exception.__cause__,cleanup);saved=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_bytes());self.assertFalse(saved['whole_scope_success_validated']);self.assertEqual(saved['status'],'CPU_PREFIX_INPUT_CHECKS_FAILED');self.assertEqual(saved['operations'][self.phases[5]]['failed'],1)
    def test_exact_main_source_defers_C_writer_first_one_entry_route_and_closed_gate(self):
        tree=ast.parse((WORK/'prefix_input_owned_worker.py').read_bytes());assignment=next(node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='C' for t in node.targets));self.assertIsInstance(assignment.value,ast.IfExp);self.assertIsNone(assignment.value.body.value)
        main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main');calls=[node.func.id for node in ast.walk(main) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
        self.assertLess(calls.index('_entry_journal_bootstrap'),calls.index('_native_entry_actions'));self.assertEqual(calls.count('_entry_phase_body'),1)
        self.assertEqual(len([node for node in tree.body if isinstance(node,ast.If) and '__name__' in ast.unparse(node.test)]),1)
        with self.assertRaises(ValueError):W.C.stage_native_gate()
METHODS=('test_real_shared_entry_core_durable_action_returns_costs_and_owner_seal','test_source_failure_skips_core_cleanup_and_preserves_original_exception','test_primary_cleanup_and_final_summary_failure_keep_failed_costs','test_exact_main_source_defers_C_writer_first_one_entry_route_and_closed_gate')
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS));after={name:hashlib.sha256((WORK/name).read_bytes()).hexdigest() for name in FILES};assert before==after
receipt={'kind':'INDEPENDENT_CPU_LINKED_ENTRY_ENVELOPE_REVIEW_004','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_before':before,'source_after':after,'actual_private_action_writer_and_core_traces':TRACES,'original_action_returns_and_executed_inert_cost_updates_retained':True,'source_failure_core_skip_cleanup_and_primary_error_exercised':True,'native_main_source_order_only_inspected':True,'native_main_or_native_action_closures_executed':False,'early_interpreter_bootstrap_or_full_transitive_costs_qualified':False,'whole_runtime_or_provider_readiness_qualified':False,'site_disabled':sys.flags.no_site==1,'native_API_DLL_ctypes_framework_tensor_called':False,'native_PASS':False,'native_authority_issued':False}
receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest();out=HERE/'INDEPENDENT_ENTRY_REVIEW004.json'
with out.open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'file_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'body_sha256':receipt['sha256'],'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'path':str(out)},sort_keys=True));raise SystemExit(0 if result.wasSuccessful() else 1)
