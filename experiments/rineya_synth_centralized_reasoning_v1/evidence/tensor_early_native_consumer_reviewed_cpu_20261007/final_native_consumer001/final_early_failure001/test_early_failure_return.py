"""Actual phase6 early provisional return writer; inert closures, no native API."""
from pathlib import Path
import hashlib,json,tempfile,types,unittest
HERE=Path(__file__).resolve().parent

def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
W=load(HERE/'prefix_input_owned_worker.py','CPU_early_entry_worker');P=load(HERE/'prefix_native_pipeline.py','CPU_early_entry_pipeline');F=load(W.C.CORE/'fixture_support.py','CPU_early_core_api')
class Controls(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='CPU_early_entry_');self.output=Path(self.tmp.name);self.ledger,self.phases=W._entry_journal_bootstrap(self.output)
    def tearDown(self):
        if not self.ledger.closed:self.ledger.close()
        self.tmp.cleanup()
    def failed_actions(self,failed=0,metadata=None):
        def action(name):
            def call(state,costs):
                if name==self.phases[failed]:
                    if metadata is not None:state['precore_owned_membership']=metadata
                    raise ValueError('actual inert source failure')
                return {'fixture_only':True}
            return call
        return {n:action(n) for n in self.phases}
    def test_actual_source_failure_writes_provisional_native_scope_without_C_load(self):
        with self.assertRaisesRegex(ValueError,'actual inert source failure'):W._entry_phase_body(self.ledger,self.phases,self.failed_actions(),lambda *a:self.fail('core unattempted'),_native_mode=True)
        envelope=json.loads((self.output/'OWNED_WORKER_RETURN.json').read_text());self.assertEqual(set(envelope),{'kind','report','runtime_execution_modules','owned_membership_verified','IO','native_PASS','report_requires_final_phase_accounting'});self.assertEqual(envelope['kind'],'NATIVE_SHARED_ENTRY_PRECORE_FAILURE');self.assertTrue(envelope['report_requires_final_phase_accounting']);self.assertIsNone(envelope['runtime_execution_modules']);self.assertIsNone(envelope['owned_membership_verified']);report=envelope['report'];self.assertEqual(report['scope'],W.NATIVE_ENTRY_SCOPE);self.assertEqual(report['status'],'NATIVE_INPUT_CHECKS_FAILED');self.assertEqual(report['core_membership_status'],'NOT_CORE_ENROLLED');self.assertIsNone(report['precore_owned_membership']);self.assertEqual(len(report),11);self.assertNotIn('final_phase_summary',report);self.assertNotIn('operation_accounting',report)
        final=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_text());self.assertEqual(final['operations'][self.phases[5]]['returned'],1);self.assertEqual(final['IO']['owned_return_write'],{'attempted':1,'returned':1,'failed':0,'pending':0});row=final['selected_boundary_counters'][self.phases[5]]['worker.precore_return_write'];self.assertEqual(row['category'],'IO');self.assertEqual(row['returned'],1);self.assertFalse(final['whole_scope_success_validated']);self.assertEqual(final['markers'],4)
    def test_phase5_metadata_retained_separately_without_core_enrollment(self):
        metadata={'backend':'explicit_inert_metadata','worker_pid':7}
        with self.assertRaises(ValueError):W._entry_phase_body(self.ledger,self.phases,self.failed_actions(4,metadata),lambda *a:self.fail(),_native_mode=True)
        envelope=json.loads((self.output/'OWNED_WORKER_RETURN.json').read_text());self.assertEqual(envelope['report']['precore_owned_membership'],metadata);self.assertIsNone(envelope['owned_membership_verified']);self.assertEqual(envelope['report']['core_membership_status'],'NOT_CORE_ENROLLED')
    def test_early_writer_failure_leaves_honest_raw_and_failed_IO(self):
        original=self.ledger.durable
        def durable(name,value):
            if name=='OWNED_WORKER_RETURN.json':raise OSError('early writer failed')
            return original(name,value)
        self.ledger.durable=durable
        with self.assertRaisesRegex(ValueError,'actual inert source failure') as caught:W._entry_phase_body(self.ledger,self.phases,self.failed_actions(),lambda *a:self.fail(),_native_mode=True)
        self.assertIsInstance(caught.exception.__cause__,OSError);self.assertFalse((self.output/'OWNED_WORKER_RETURN.json').exists());final=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_text());self.assertEqual(final['IO']['owned_return_write']['failed'],1);self.assertEqual(final['IO']['owned_return_write']['pending'],0);self.assertEqual(final['operations'][self.phases[5]]['failed'],1);self.assertFalse(final['whole_scope_success_validated']);self.assertTrue((self.output/'PHASE_OBSERVATIONS.jsonl').exists())
    def test_cleanup_failure_still_publishes_provisional_primary_secondary(self):
        actions=self.failed_actions();actions[self.phases[5]]=lambda *a:(_ for _ in ()).throw(RuntimeError('inert cleanup failed'))
        with self.assertRaises(ValueError) as caught:W._entry_phase_body(self.ledger,self.phases,actions,lambda *a:self.fail(),_native_mode=True)
        self.assertIsInstance(caught.exception.__cause__,RuntimeError);report=json.loads((self.output/'OWNED_WORKER_RETURN.json').read_text())['report'];self.assertEqual(len(report['entry_errors']),2);self.assertNotIn('whole_scope_success_validated',report)
    def test_core_report_branch_does_not_create_an_early_return(self):
        actions={n:lambda *a:None for n in self.phases}
        def execute(state,ledger):
            run,core,proof=P._bind_shared_core_executor(W,ledger);api=F.InjectedAPI();return run(api,api.guard,self.output)
        report=W._entry_phase_body(self.ledger,self.phases,actions,execute);self.assertTrue(report['whole_scope_success_validated']);self.assertFalse((self.output/'OWNED_WORKER_RETURN.json').exists());self.assertNotIn('owned_return_write',self.ledger.io)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));receipt={'kind':'CPU_ACTUAL_EARLY_PHASE6_FAILURE_ENVELOPE_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_controls_completed':result.wasSuccessful(),'actual_original_writer_and_entry_body_executed':True,'closures':'private_inert','phase5_membership_metadata':'explicit_inert_projection','native_API_framework_called':False,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('prefix_input_owned_worker.py','prefix_native_pipeline.py','prefix_tensor_contract.py','test_early_failure_return.py')}}
    if len(__import__('sys').argv)>1:
        with Path(__import__('sys').argv[1]).open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
