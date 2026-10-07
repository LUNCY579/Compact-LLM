"""Source-owned exact entry journal composition; private inert actions only."""
from pathlib import Path
import ast,hashlib,json,tempfile,types,unittest
HERE=Path(__file__).resolve().parent

def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
W=load(HERE/'prefix_input_owned_worker.py','CPU_entry_worker');P=load(HERE/'prefix_native_pipeline.py','CPU_entry_pipeline');F=load(W.C.CORE/'fixture_support.py','CPU_entry_inert_api')
class Controls(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='CPU_entry_');self.output=Path(self.tmp.name);self.ledger,self.phases=W._entry_journal_bootstrap(self.output);self.events=[]
    def tearDown(self):
        if not self.ledger.closed:self.ledger.close()
        self.tmp.cleanup()
    def fixture(self,failed=None):
        def action(name):
            def run(state,costs):
                self.events.append(name);costs.update(source_reads=1,source_bytes=7,API_calls=0)
                if name==failed:raise RuntimeError(name+' failed')
                return {'fixture_only':True,'native_authority_issued':False}
            return run
        actions={name:action(name) for name in self.phases}
        def execute(state,ledger):
            self.assertIs(ledger,self.ledger);self.assertEqual(self.events,list(self.phases[:5]));core,namespace,proof=P._bind_shared_core_executor(W,ledger);api=F.InjectedAPI();return core(api,api.guard,self.output)
        return actions,execute
    def test_actual_writer_and_core_compose_six_phases_and_durable_live_costs(self):
        actions,execute=self.fixture();report=W._entry_phase_body(self.ledger,self.phases,actions,execute);self.assertTrue(report['whole_scope_success_validated']);self.assertEqual(self.events,list(self.phases));saved=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_text());self.assertEqual(saved['startup_subcosts'][self.phases[0]]['source_bytes'],7);self.assertEqual(saved['whole_scope_order'],list(self.phases[:5])+list(W.source_core()['C'].OPS)+[self.phases[5]]);self.assertLessEqual(saved['markers'],128);self.assertEqual(len(report['entry_bootstrap_sources']),3)
    def test_source_and_preload_failures_cleanup_and_seal_same_journal(self):
        for index in (0,2):
            with self.subTest(index=index):
                output=self.output/str(index);output.mkdir();ledger,phases=W._entry_journal_bootstrap(output);events=[]
                def invoke(name,state,costs):
                    events.append(name);costs['source_reads']=1
                    if name==phases[index]:raise RuntimeError('selected failure')
                actions={n:lambda state,costs,n=n:invoke(n,state,costs) for n in phases}
                with self.assertRaisesRegex(RuntimeError,'selected failure'):W._entry_phase_body(ledger,phases,actions,lambda *a:self.fail('core must not execute'))
                self.assertEqual(events[-1],phases[5]);saved=json.loads((output/'PHASE_ACCOUNTING_FINAL.json').read_text());self.assertFalse(saved['whole_scope_success_validated']);self.assertEqual(saved['operations'][phases[index]]['failed'],1);self.assertTrue(ledger.closed)
    def test_provider_and_cleanup_failure_preserve_primary_and_failure_status(self):
        actions,execute=self.fixture(self.phases[4]);actions[self.phases[5]]=lambda state,costs:(_ for _ in ()).throw(OSError('owned cleanup failed'))
        with self.assertRaisesRegex(RuntimeError,'startup_native_provider_binding failed') as caught:W._entry_phase_body(self.ledger,self.phases,actions,execute)
        self.assertIsInstance(caught.exception.__cause__,OSError);saved=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_text());self.assertFalse(saved['whole_scope_success_validated']);self.assertEqual(saved['status'],'CPU_PREFIX_INPUT_CHECKS_FAILED')
    def test_final_publication_failure_cannot_return_completed_report(self):
        actions,execute=self.fixture();original=self.ledger.durable
        def durable(name,value):
            if name=='PHASE_ACCOUNTING_FINAL.json':raise OSError('final publication failed')
            return original(name,value)
        self.ledger.durable=durable
        with self.assertRaisesRegex(OSError,'final publication failed'):W._entry_phase_body(self.ledger,self.phases,actions,execute)
        self.assertEqual(self.ledger.io['final_summary_write']['failed'],1);self.assertTrue(self.ledger.closed)
    def test_actual_main_defers_C_and_has_one_same_entry_and_original_cleanup_handoff(self):
        tree=ast.parse((HERE/'prefix_input_owned_worker.py').read_bytes());assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='C' for t in n.targets));self.assertIsInstance(assignment.value,ast.IfExp);self.assertIsNone(assignment.value.body.value)
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');calls=[n.func.id for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)];self.assertLess(calls.index('_entry_journal_bootstrap'),calls.index('_native_entry_actions'));self.assertEqual(calls.count('_entry_phase_body'),1)
        actions=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_native_entry_actions');text=ast.unparse(actions);self.assertIn('_registered_main_body',text);self.assertIn('_owned_execution_factory',text);self.assertIn('_defer_cleanup',text);self.assertIn('C.stage_native_gate()',text)
        with self.assertRaises(ValueError):W.C.stage_native_gate()
        for i in range(64):self.ledger.call('authored_'+str(i),lambda:None)
        with self.assertRaises(ValueError):self.ledger.call('129',lambda:None)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));r={'kind':'CPU_LINKED_ENTRY_ENVELOPE_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_original_writer_and_core_attempted':True,'all_controls_completed':result.wasSuccessful(),'private_inert_entry_closures_only':True,'native_actions_executed':False,'interpreter_initialization_metered':False,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('prefix_input_owned_worker.py','prefix_native_pipeline.py','prefix_tensor_contract.py','test_entry_envelope.py')}}
    if len(__import__('sys').argv)>1:
        with Path(__import__('sys').argv[1]).open('x',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
