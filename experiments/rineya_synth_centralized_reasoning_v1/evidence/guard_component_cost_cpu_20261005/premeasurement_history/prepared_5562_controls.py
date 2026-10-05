"""Light CPU callback/count/failure controls; no durable measurement loop."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import guard_component_benchmark as bench


class Controls(unittest.TestCase):
    def setup_guard(self,callback_failure=None,heartbeat_age=0.):
        seen=[];owner={'job_name':'CPU_FIXTURE_JOB','supervisor_pid':31}
        def owned(name):
            seen.append(('owned',name))
            if callback_failure=='owned_resources':raise OSError('owned callback failure')
            return {'tree_rss_bytes':0}
        def available():
            seen.append(('RAM',None))
            if callback_failure=='available':raise OSError('RAM callback failure')
            return 2*1024**3
        def heartbeat(path):
            seen.append(('heartbeat',path))
            if callback_failure=='heartbeat_reader':raise OSError('heartbeat callback failure')
            return {'unix_seconds':100.-heartbeat_age,'supervisor_pid':31}
        def factory(output,owner,start,getter,**kwargs):
            options={'owned_resources':owned,'available':available,'heartbeat_reader':heartbeat};options.update(kwargs)
            return bench.make_guard(output,owner,start,getter,monotonic=lambda:10.,wall=lambda:100.,**options)
        meter,timers,origins=bench.instrument_guard(Path('CPU_FIXTURE'),owner,0.,factory=factory)
        return meter,timers,seen
    def test_512_calls_preserve_callback_order_arguments_frequency_and_returns(self):
        guard,timers,seen=self.setup_guard()
        for i in range(512):guard('fixture_%d'%i)
        self.assertEqual({'attempted':512,'returned':512,'failed':0,'pending':0},guard.counts)
        self.assertTrue(all(t.counts==guard.counts for t in timers.values()))
        self.assertEqual([('owned','CPU_FIXTURE_JOB'),('RAM',None),('heartbeat',Path('CPU_FIXTURE/supervisor_heartbeat.jsonl'))],seen[:3])
        self.assertEqual(1536,len(seen));bench.no_framework()
    def test_each_callback_failure_keeps_exact_partial_counts(self):
        for name in ('owned_resources','available','heartbeat_reader'):
            with self.subTest(name=name):
                guard,timers,seen=self.setup_guard(callback_failure=name)
                with self.assertRaises(OSError):guard('fixture')
                self.assertEqual({'attempted':1,'returned':0,'failed':1,'pending':0},guard.counts)
                self.assertEqual(1,timers[name].counts['failed']);self.assertTrue(all(t.counts['pending']==0 for t in timers.values()))
    def test_original_stale_heartbeat_check_not_weakened(self):
        guard,timers,seen=self.setup_guard(heartbeat_age=2.01)
        with self.assertRaisesRegex(ValueError,'heartbeat stale'):guard('fixture')
        self.assertEqual(1,guard.counts['failed']);self.assertTrue(all(t.counts['returned']==1 for t in timers.values()))
    def test_real_default_callback_identity_is_extracted_without_execution(self):
        guard,timers,origins=bench.instrument_guard(Path('CPU_FIXTURE'),{'job_name':'CPU_FIXTURE','supervisor_pid':31},0.)
        self.assertIs(bench.make_guard.__kwdefaults__['owned_resources'],timers['owned_resources'].function)
        self.assertIs(bench.make_guard.__kwdefaults__['heartbeat_reader'],timers['heartbeat_reader'].function)
        self.assertEqual('make_guard.<locals>.available',timers['available'].function.__qualname__)


if __name__=='__main__':unittest.main()
