"""Lightweight no-framework controls; no durable measurement loops."""
from collections import Counter
import tempfile
from pathlib import Path
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch
import journal_guard_benchmark as bench


class Controls(unittest.TestCase):
    def test_constructor_receipt_consumed_without_rewrite_and_wrong_or_missing_identity_blocks_resume(self):
        with tempfile.TemporaryDirectory(dir=bench.WORK) as directory:
            path=Path(directory)/'supervisor_ownership.json';resumed=[]
            worker=SimpleNamespace(process=SimpleNamespace(pid=22),job=SimpleNamespace(name='Local\\SYNTH_NQ_JOB_'+'a'*32),resume=lambda:resumed.append(True))
            expected={'worker_pid':22,'root_worker_pid':22,'job_name':worker.job.name,'supervisor_pid':11,'supervisor_create_time':123.,'config_sha256':'b'*64}
            def dispatch():
                value=bench.constructor_ownership(directory,worker,11,123.,'b'*64);worker.resume();return value
            original=json.dumps(expected).encode();path.write_bytes(original)
            self.assertEqual(expected,dispatch());self.assertEqual(original,path.read_bytes());self.assertEqual([True],resumed)
            for key,value in (('worker_pid',23),('root_worker_pid',23),('supervisor_pid',True),('supervisor_create_time',124.),('job_name','other'),('config_sha256','c'*64)):
                damaged=dict(expected);damaged[key]=value;path.write_text(json.dumps(damaged));resumed.clear()
                with self.assertRaisesRegex(ValueError,'identity/config mismatch'):dispatch()
                self.assertFalse(resumed)
            damaged=dict(expected);del damaged['supervisor_create_time'];path.write_text(json.dumps(damaged));resumed.clear()
            with self.assertRaisesRegex(ValueError,'identity/config mismatch'):dispatch()
            self.assertFalse(resumed)
            path.unlink()
            with self.assertRaises(FileNotFoundError):dispatch()
            self.assertFalse(resumed)
    def test_exact_matched_regions_primitives_and_all_guard_layers(self):
        meter=bench.GuardMeter(lambda stage:None);journal=bench.DirectJournal(meter)
        result=bench.exercise(journal,meter)
        self.assertEqual('COMPLETE_CPU_COMPONENT_COUNTED_UPDATE_FRAME',result['status'])
        self.assertEqual(1266,sum(x['attempted'] for x in result['primitive_operations'].values()))
        self.assertEqual(7722,result['guards']['attempted']);self.assertEqual(20,journal.evidence_count)
        self.assertEqual(1120,len(result['regions']));self.assertFalse(result['decoder_failed_or_pending'])
        bench.no_framework()
    def test_actual_primitive_failure_preserves_failed_and_sticky_partial_costs(self):
        meter=bench.GuardMeter(lambda stage:None);journal=bench.DirectJournal(meter);original=bench.Layer.forward
        def fail(layer,value,**kwargs):
            if layer.index==3:raise RuntimeError('injected scalar body failure')
            return original(layer,value,**kwargs)
        with patch.object(bench.Layer,'forward',fail):result=bench.exercise(journal,meter)
        self.assertEqual('FAILED_CPU_COMPONENT_COUNTED_UPDATE_FRAME',result['status'])
        self.assertTrue(result['decoder_failed_or_pending'])
        self.assertEqual(1,result['primitive_operations']['checkpoint_layer03_forward']['failed'])
        self.assertEqual(1,result['primitive_operations']['training_forward']['failed'])
    def test_after_return_sink_failure_keeps_returned_operation_cost(self):
        meter=bench.GuardMeter(lambda stage:None);journal=bench.DirectJournal(meter);original=journal.call
        def fail(name,function):
            value=original(name,function)
            if name=='checkpoint_layer03_forward':raise OSError('injected returned sink failure')
            return value
        journal.call=fail;result=bench.exercise(journal,meter)
        item=result['primitive_operations']['checkpoint_layer03_forward']
        self.assertEqual({'attempted':1,'returned':1,'failed':0,'pending':0},item)
        self.assertEqual('FAILED_CPU_COMPONENT_COUNTED_UPDATE_FRAME',result['status'])
    def test_guard_failure_does_not_invent_primitive_execution(self):
        def fail(stage):
            if stage=='checkpoint_layer3_forward_before':raise RuntimeError('injected guard refusal')
        meter=bench.GuardMeter(fail);journal=bench.DirectJournal(meter);result=bench.exercise(journal,meter)
        self.assertNotIn('checkpoint_layer03_forward',result['primitive_operations'])
        self.assertTrue(result['decoder_failed_or_pending']);self.assertEqual(1,result['guards']['failed'])
    def test_actual_durable_journal_flush_fsync_return_failure_keeps_return_count(self):
        # Two tiny records, mocked fsync control; no measured durability loop.
        with tempfile.TemporaryDirectory(dir=bench.WORK) as directory:
            journal=bench.MeteredDurableJournal(Path(directory));calls=[]
            def fsync(fd):
                calls.append(fd)
                if len(calls)==2:raise OSError('returned record durability failure')
            try:
                with patch.object(bench.os,'fsync',fsync):
                    with self.assertRaisesRegex(OSError,'durability failure'):
                        journal.call('CPU_FIXTURE_noop',lambda:None)
                self.assertEqual({'attempted':1,'returned':1,'failed':0,'pending':0},journal.snapshot()['operations']['CPU_FIXTURE_noop'])
                self.assertEqual(2,len(calls));self.assertEqual(1,journal.emit_returned);self.assertEqual(1,journal.emit_failed)
            finally:journal.stream.close()


if __name__=='__main__':unittest.main()
