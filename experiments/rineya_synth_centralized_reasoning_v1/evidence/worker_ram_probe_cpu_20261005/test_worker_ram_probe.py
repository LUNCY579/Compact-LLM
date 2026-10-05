"""Light injected RAM controls; no actual Windows API/model/GPU/durable loop."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import worker_ram_probe as candidate
from inference_qual_worker import make_guard


class Controls(unittest.TestCase):
    def test_exact_available_field_and_fresh_changing_values_no_numeric_cache(self):
        calls=[]
        def status():calls.append(True);return {'available_bytes':100-len(calls),'total_bytes':200}
        self.assertEqual([99,98,97],[candidate.available_bytes(status_call=status) for _ in range(3)])
        self.assertEqual(3,len(calls))
    def test_actual_helper_called_fresh_each_time(self):
        with patch.object(candidate.native_process,'system_memory_status',side_effect=[{'available_bytes':1,'total_bytes':2},{'available_bytes':0,'total_bytes':2}]) as api:
            self.assertEqual(1,candidate.available_bytes());self.assertEqual(0,candidate.available_bytes());self.assertEqual(2,api.call_count)
    def test_wrong_shape_field_type_boolean_range_and_total_rejected(self):
        for value in ({'available_bytes':1},{'available_bytes':1,'total_bytes':2,'cached':True},
                      {'available_bytes':True,'total_bytes':2},{'available_bytes':1.,'total_bytes':2},
                      {'available_bytes':1,'total_bytes':True},{'available_bytes':-1,'total_bytes':2},
                      {'available_bytes':3,'total_bytes':2},{'available_bytes':0,'total_bytes':0},
                      {'available_bytes':0,'total_bytes':2**64}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):candidate.available_bytes(status_call=lambda:value)
    def test_helper_error_propagates_same_object_no_fallback(self):
        error=OSError('GlobalMemoryStatusEx failure')
        def fail():raise error
        with self.assertRaises(OSError) as seen:candidate.available_bytes(status_call=fail)
        self.assertIs(error,seen.exception)
    def test_current_guard_RSS_RAM_deadline_and_heartbeat_floors_unchanged(self):
        owner={'job_name':'CPU_FIXTURE','supervisor_pid':31};seen=[]
        def make(amount,*,rss=0,elapsed=0,age=0):
            def available():seen.append(amount);return candidate.available_bytes(status_call=lambda:{'available_bytes':amount,'total_bytes':4*1024**3})
            return make_guard(Path('CPU_FIXTURE'),owner,0.,lambda:None,owned_resources=lambda name:{'tree_rss_bytes':rss},
                available=available,heartbeat_reader=lambda path:{'unix_seconds':100.-age,'supervisor_pid':31},
                monotonic=lambda:elapsed,wall=lambda:100.)
        make(int(1.25*1024**3))('fixture')
        with self.assertRaisesRegex(ValueError,'host resource'):make(int(1.25*1024**3)-1)('fixture')
        with self.assertRaisesRegex(ValueError,'host resource'):make(2*1024**3,rss=2*1024**3+1)('fixture')
        with self.assertRaisesRegex(ValueError,'deadline'):make(2*1024**3,elapsed=600.01)('fixture')
        with self.assertRaisesRegex(ValueError,'heartbeat stale'):make(2*1024**3,age=2.01)('fixture')
    def test_framework_free_import(self):
        self.assertFalse(any(name in sys.modules for name in ('torch','transformers')))


if __name__=='__main__':unittest.main()
