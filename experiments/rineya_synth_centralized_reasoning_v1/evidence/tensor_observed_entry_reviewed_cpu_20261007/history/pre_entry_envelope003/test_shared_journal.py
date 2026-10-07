"""Same original durable core ledger instance, exact startup/core order and cap."""
from pathlib import Path
import hashlib,json,tempfile,types,unittest
HERE=Path(__file__).resolve().parent
def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
W=load(HERE/'prefix_input_owned_worker.py','CPU_worker');P=load(HERE/'prefix_native_pipeline.py','CPU_pipeline');F=load(W.C.CORE/'fixture_support.py','CPU_core_fixture')
class Controls(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='CPU_shared_ledger_');self.output=Path(self.tmp.name);self.core=W.source_core();self.ledger=P._shared_scope_ledger(self.core,self.output)
    def tearDown(self):
        if not self.ledger.closed:self.ledger.close()
        self.tmp.cleanup()
    def test_actual_core_uses_same_ledger_and_owner_cleanup_before_final_seal(self):
        for name in P.STARTUP_PHASES[:5]:self.ledger.startup(name,lambda:{'CPU_fixture':True},subcosts={'source_reads':1,'API_calls':0})
        execute,core,proof=P._bind_shared_core_executor(W,self.ledger);api=F.InjectedAPI();report=execute(api,api.guard,self.output);self.assertEqual(report['status'],'CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED');self.assertFalse(self.ledger.closed);self.assertEqual(report['final_phase_summary']['status'],'PENDING_OWNED_CLEANUP_FINAL_SEAL');self.assertEqual(proof['constructor_changes'],1)
        self.ledger.startup(P.STARTUP_PHASES[5],lambda:{'owned_cleanup':True});summary=self.ledger.finish_owned(report);self.assertTrue(summary['closed']);self.assertLessEqual(summary['markers'],128);self.assertEqual(summary['operations'][P.STARTUP_PHASES[5]]['returned'],1);self.assertTrue((self.output/'PHASE_ACCOUNTING_FINAL.json').exists());saved=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_text());self.assertTrue(saved['whole_scope_success_validated']);self.assertEqual(set(saved['startup_subcosts']),set(P.STARTUP_PHASES));self.assertEqual(saved['startup_subcosts'][P.STARTUP_PHASES[0]]['source_reads'],1)
        with self.assertRaises(ValueError):self.ledger.finish_owned(report)
    def test_early_failure_is_durable_and_same_owner_finish_reconciles(self):
        failure=RuntimeError('authored startup failure')
        with self.assertRaisesRegex(RuntimeError,'authored startup failure'):self.ledger.startup(P.STARTUP_PHASES[0],lambda:(_ for _ in ()).throw(failure))
        self.ledger.startup(P.STARTUP_PHASES[5],lambda:{'cleanup':True});report={'status':'CPU_PREFIX_INPUT_CHECKS_FAILED'};summary=self.ledger.finish_owned(report);self.assertEqual(summary['operations'][P.STARTUP_PHASES[0]]['failed'],1);self.assertEqual(summary['operations'][P.STARTUP_PHASES[0]]['pending'],0);self.assertEqual(summary['markers'],4)
    def successful_core(self):
        for name in P.STARTUP_PHASES[:5]:self.ledger.startup(name,lambda:None)
        execute,core,proof=P._bind_shared_core_executor(W,self.ledger);api=F.InjectedAPI();return execute(api,api.guard,self.output)
    def test_cleanup_failure_and_missing_costs_cannot_complete(self):
        report=self.successful_core()
        with self.assertRaisesRegex(RuntimeError,'cleanup failure'):self.ledger.startup(P.STARTUP_PHASES[5],lambda:(_ for _ in ()).throw(RuntimeError('cleanup failure')))
        summary=self.ledger.finish_owned(report);self.assertEqual(summary['status'],'CPU_PREFIX_INPUT_CHECKS_FAILED');self.assertFalse(summary['whole_scope_success_validated'])
    def test_marker_and_final_publication_failures_cannot_complete(self):
        original_marker=self.ledger.marker;self.ledger.marker=lambda *a,**k:(_ for _ in ()).throw(OSError('marker failed'))
        with self.assertRaises(OSError):self.ledger.startup(P.STARTUP_PHASES[0],lambda:None)
        self.ledger.marker=original_marker;report={'status':'CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED'};self.ledger.durable=lambda *a:(_ for _ in ()).throw(OSError('summary failed'))
        with self.assertRaisesRegex(OSError,'summary failed'):self.ledger.finish_owned(report)
        self.assertEqual(report['status'],'CPU_PREFIX_INPUT_CHECKS_FAILED');self.assertFalse(report['whole_scope_success_validated']);self.assertEqual(self.ledger.io['final_summary_write']['failed'],1)
    def test_close_failure_cannot_leave_completed_consumer_report(self):
        report=self.successful_core();self.ledger.startup(P.STARTUP_PHASES[5],lambda:None)
        original=self.ledger.close
        self.ledger.close=lambda:self.ledger.action('ledger_close',lambda:(_ for _ in ()).throw(OSError('close failed')))
        try:
            with self.assertRaisesRegex(OSError,'close failed'):self.ledger.finish_owned(report)
            self.assertEqual(report['status'],'CPU_PREFIX_INPUT_CHECKS_FAILED');self.assertFalse(report['whole_scope_success_validated']);self.assertEqual(self.ledger.io['ledger_close']['failed'],1);self.assertEqual(self.ledger.io['ledger_close']['pending'],0);self.assertFalse((self.output/'PHASE_ACCOUNTING_FINAL.json').exists())
        finally:self.ledger.close=original
    def test_startup_subcosts_reject_bool_unknown_and_out_of_order_success(self):
        with self.assertRaises(ValueError):self.ledger.startup(P.STARTUP_PHASES[0],lambda:None,{'API_calls':True})
        with self.assertRaises(ValueError):self.ledger.startup(P.STARTUP_PHASES[0],lambda:None,{'unknown':1})
        self.ledger.startup(P.STARTUP_PHASES[5],lambda:None);report={'status':'CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED'};summary=self.ledger.finish_owned(report);self.assertFalse(summary['whole_scope_success_validated']);self.assertEqual(summary['status'],'CPU_PREFIX_INPUT_CHECKS_FAILED')
    def test_same_original_128_marker_writer_refuses_129_without_widening(self):
        for i in range(64):self.ledger.call('CPU_counted_'+str(i),lambda:None)
        self.assertEqual(self.ledger.markers,128)
        with self.assertRaises(ValueError):self.ledger.call('CPU_over_limit',lambda:None)
        self.assertEqual(self.ledger.markers,128);self.assertEqual(self.core['C'].LIMITS['maximum_markers'],128)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));r={'kind':'CPU_SHARED_DURABLE_LEDGER_CORE_HANDOFF_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_original_LEDGER_and_core_source':True,'native_worker_main_composition_executed':False,'early_interpreter_bootstrap_coverage_established':False,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['prefix_tensor_contract.py','prefix_input_owned_worker.py','prefix_native_pipeline.py','test_shared_journal.py']}}
    if len(__import__('sys').argv)>1:
        with Path(__import__('sys').argv[1]).open('x',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
