"""Independent original shared writer/core closure checks; no worker-main run."""
from pathlib import Path
import hashlib,json,runpy,sys,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
FILES=('prefix_native_pipeline.py','prefix_input_owned_worker.py','prefix_tensor_contract.py','test_shared_journal.py')
before={name:hashlib.sha256((WORK/name).read_bytes()).hexdigest() for name in FILES}
assert before=={'prefix_native_pipeline.py':'4f163834a4e18eb8dc71fe34ee7acc80734c6b08fd90dd97457cd6bb7714ab49','prefix_input_owned_worker.py':'01b297a318ad9fb969655bfc486532eeb654321fcc29e5b56ad6a2f8fb220e2a','prefix_tensor_contract.py':'fd86efe022e1c2f36b5a431b37e271cedfd4dd1654ffd3d6d7e0890acea8895f','test_shared_journal.py':'693a1d4c1d9d441669178c19ac79e340c64df8794955f3c57082ad5787a1fb57'}
ns=runpy.run_path(str(WORK/'test_shared_journal.py'),run_name='independent_shared_writer_fixture');Base=ns['Controls'];P,W,F=ns['P'],ns['W'],ns['F']
TRACES=[]
class IndependentControls(Base):
    def test_same_actual_core_ledger_durable_costs_cleanup_order_and_seal(self):
        for index,name in enumerate(P.STARTUP_PHASES[:5]):self.ledger.startup(name,lambda:None,{'source_reads':index+1,'source_bytes':10*(index+1),'API_calls':0})
        execute,core,proof=P._bind_shared_core_executor(W,self.ledger);api=F.InjectedAPI();report=execute(api,api.guard,self.output)
        self.assertFalse(self.ledger.closed);self.assertEqual(report['final_phase_summary']['status'],'PENDING_OWNED_CLEANUP_FINAL_SEAL')
        self.ledger.startup(P.STARTUP_PHASES[5],lambda:None,{'IO_calls':1});summary=self.ledger.finish_owned(report)
        saved=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_bytes());rows=[json.loads(line) for line in (self.output/'PHASE_OBSERVATIONS.jsonl').read_bytes().splitlines()]
        self.assertEqual(saved,summary);self.assertEqual(len(rows),summary['markers']);self.assertLessEqual(len(rows),128)
        self.assertEqual(saved['whole_scope_order'],list(P.STARTUP_PHASES[:5])+list(core['C'].OPS)+[P.STARTUP_PHASES[5]])
        self.assertTrue(saved['whole_scope_success_validated']);self.assertTrue(all(c['pending']==c['failed']==0 for c in saved['operations'].values()))
        for index,name in enumerate(P.STARTUP_PHASES[:5]):
            returned=next(row for row in rows if row['operation']==name and row['event']=='returned')
            self.assertEqual(returned['result']['startup_subcosts'],saved['startup_subcosts'][name]);self.assertEqual(saved['startup_subcosts'][name]['source_reads'],index+1)
        self.assertEqual(saved['IO']['ledger_close'],{'attempted':1,'returned':1,'failed':0,'pending':0});self.assertEqual(saved['IO']['final_summary_write'],{'attempted':1,'returned':1,'failed':0,'pending':0})
        TRACES.append({'actual_markers':len(rows),'exact_nonguard_order':saved['whole_scope_order'],'durable_startup_costs':saved['startup_subcosts'],'source_bound':proof})
    def test_real_close_action_failure_marks_failed_and_retains_IO_cost(self):
        report=self.successful_core();self.ledger.startup(P.STARTUP_PHASES[5],lambda:None)
        failure=OSError('independent original close action failure');original=self.ledger.action
        def failing_action(name,fn):return original(name,(lambda:(_ for _ in ()).throw(failure)) if name=='ledger_close' else fn)
        try:
            with patch.object(self.ledger,'action',side_effect=failing_action):
                with self.assertRaises(OSError) as caught:self.ledger.finish_owned(report)
            self.assertIs(caught.exception,failure);self.assertEqual(report['status'],'CPU_PREFIX_INPUT_CHECKS_FAILED');self.assertFalse(report['whole_scope_success_validated'])
            self.assertEqual(self.ledger.io['ledger_close'],{'attempted':1,'returned':0,'failed':1,'pending':0});self.assertFalse((self.output/'PHASE_ACCOUNTING_FINAL.json').exists())
        finally:self.ledger.stream.close();self.ledger.closed=True
    def test_cleanup_or_publication_failure_cannot_be_complete(self):
        report=self.successful_core()
        with self.assertRaises(RuntimeError):self.ledger.startup(P.STARTUP_PHASES[5],lambda:(_ for _ in ()).throw(RuntimeError('independent cleanup failure')))
        summary=self.ledger.finish_owned(report);self.assertFalse(summary['whole_scope_success_validated']);self.assertEqual(summary['status'],'CPU_PREFIX_INPUT_CHECKS_FAILED');self.assertEqual(summary['operations'][P.STARTUP_PHASES[5]]['failed'],1)
    def test_original_marker_129_refusal_records_attempt_without_action_execution(self):
        for index in range(64):self.ledger.call('CPU_'+str(index),lambda:None)
        called=[]
        with self.assertRaises(ValueError):self.ledger.call('CPU_over_cap',lambda:called.append(True))
        self.assertEqual(called,[]);self.assertEqual(self.ledger.markers,128);self.assertEqual(self.ledger.ops['CPU_over_cap'],{'attempted':1,'returned':0,'failed':1,'pending':0});self.assertTrue(self.ledger.errors)
        summary=self.ledger.finish_owned({'status':'CPU_PREFIX_INPUT_CHECKS_COMPLETE_UNQUALIFIED'});self.assertFalse(summary['whole_scope_success_validated']);self.assertEqual(summary['status'],'CPU_PREFIX_INPUT_CHECKS_FAILED')
METHODS=('test_same_actual_core_ledger_durable_costs_cleanup_order_and_seal','test_real_close_action_failure_marks_failed_and_retains_IO_cost','test_cleanup_or_publication_failure_cannot_be_complete','test_original_marker_129_refusal_records_attempt_without_action_execution')
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS));after={name:hashlib.sha256((WORK/name).read_bytes()).hexdigest() for name in FILES};assert before==after
receipt={'kind':'INDEPENDENT_CPU_SHARED_CORE_JOURNAL_CLOSURE_REVIEW_003','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_before':before,'source_after':after,'actual_success_traces':TRACES,'original_durable_writer_and_core_execute_on_same_instance':True,'startup_subcost_values_authored_fixture_not_native_API_measurement':True,'close_failure_and_cleanup_error_not_complete':True,'marker129_refused_without_action_execution':True,'worker_main_entry_and_early_import_accounting_executed':False,'native_provider_or_whole_runtime_qualified':False,'prior_close_failure_probe_preserved_in_history':{'tool_chunk_id':'0596b3','old_pipeline_sha256':'6575bef2cb1ca4139c8e9cdd9fbfd05100588c5856e1d57453834658c161a884','old_failed_close_left_successflagTrue':True},'site_disabled':sys.flags.no_site==1,'native_API_DLL_ctypes_framework_tensor_called':False,'native_PASS':False,'native_authority_issued':False}
receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest();out=HERE/'INDEPENDENT_JOURNAL_REVIEW003.json'
with out.open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'file_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'body_sha256':receipt['sha256'],'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'path':str(out)},sort_keys=True));raise SystemExit(0 if result.wasSuccessful() else 1)
