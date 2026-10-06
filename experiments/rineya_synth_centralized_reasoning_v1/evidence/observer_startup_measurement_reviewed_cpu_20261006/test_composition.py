"""Actual original supervisor resume executes the actual observer CPU worker."""
from pathlib import Path
from copy import deepcopy
import tempfile,unittest
import test_supervisor as S
import test_worker as W
import supervisor

class SupervisedObserverComposition(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_supervised_observer_');self.addCleanup(self.temp.cleanup)
        self.c,self.context,self.value,self.transport,self.provider,self.output,self.cap=W.make_cpu_fixture(self.temp.name)
        self.value['config']['worker_path']=str(self.c.PACKAGE/supervisor.WORKER_RELATIVE)
        self.value['config']=self.c.seal(self.value['config'])
        self.context=self.c._CPU_context_value_fixture(self.value,_capability=self.cap)
        self.host=S.Host(self.temp.name);self.worker_result=None
    def run_pipeline(self):
        outer=self;original=self.host.worker
        def create(argv,output,**kwargs):
            worker=original(argv,output,**kwargs);resume=worker.resume
            def execute():
                resume()
                outer.worker_result=W.W._CPU_run_fixture(outer.context,outer.c,outer.transport,outer.provider,output,
                    _capability=outer.cap,clock=outer.host.clock)
                outer.host.returncode=0 if outer.worker_result['status']=='OBSERVER_OPERATIONS_RETURNED_PENDING_TERMINAL_AUDIT' else 2
            worker.resume=execute;return worker
        self.host.worker=create
        return supervisor._CPU_supervise_fixture(self.context,self.host,self.output,contract=self.c,
            _capability=supervisor._CPU_capability())
    def test_actual_assigned_resume_worker_and_terminal_cleanup(self):
        proof=self.run_pipeline();self.assertTrue(proof['cleanup_verified'])
        self.assertEqual(self.worker_result['status'],'OBSERVER_OPERATIONS_RETURNED_PENDING_TERMINAL_AUDIT')
        self.assertTrue(self.transport.closed);self.assertTrue((self.output/'OBSERVER_PHASES.jsonl').is_file())
        self.assertTrue((self.output/'OBSERVER_WORKER_RESULT.json').is_file())
        self.assertLess(self.host.calls.index('Job_assigned'),self.host.calls.index('resume'))
        self.assertFalse(proof['native_PASS'])
    def test_worker_failed_event_closure_still_original_terminal_cleanup(self):
        self.transport.poison={'faults':128}
        proof=self.run_pipeline();self.assertEqual(self.worker_result['status'],'OBSERVER_OPERATIONS_FAILED')
        self.assertEqual(proof['receipt']['status'],'SUPERVISED_WORKER_FAILED')
        self.assertTrue(self.transport.closed);self.assertTrue(self.host.joined);self.assertTrue(self.host.nvml_closed)
        self.assertFalse(proof['native_PASS'])

if __name__=='__main__':unittest.main()
