"""CPU fake controls for exact free semantics, identity, and fresh observations."""
from pathlib import Path
import tempfile
import unittest
from volume_disk_probe import VolumeBoundDiskProbe

WORK=Path(__file__).resolve().parent


class FakeAPI:
    def __init__(self):self.calls=0;self.identity='FIXTURE_VOLUME';self.root='C:\\'
    def volume_path(self,path):return self.root
    def volume_id(self,root):return self.identity
    def drive_type(self,root):return 3
    def space(self,path):
        self.calls+=1
        return {'caller_available_bytes':17,'caller_total_bytes':100,'total_free_bytes':1000-self.calls}


class ProbeControls(unittest.TestCase):
    def make(self,path=None,api=None):return VolumeBoundDiskProbe(path or WORK,WORK,api=api or FakeAPI())
    def test_every_sample_queries_new_values_and_preserves_total_free_policy(self):
        api=FakeAPI();probe=self.make(api=api);first,second=probe.sample(),probe.sample()
        self.assertEqual((999,998),(first['disk_free_bytes'],second['disk_free_bytes']))
        self.assertEqual(17,first['caller_available_bytes']);self.assertEqual(2,api.calls)
        self.assertGreaterEqual(first['sample_completed'],first['sample_started'])
    def test_volume_drift_rejected_before_space_query(self):
        api=FakeAPI();probe=self.make(api=api);api.identity='OTHER'
        with self.assertRaisesRegex(ValueError,'binding drift'):probe.sample()
        self.assertEqual(0,api.calls)
    def test_output_escape_rejected(self):
        with self.assertRaisesRegex(ValueError,'cannot escape'):self.make(WORK.parent)
    def test_mount_or_binding_drift_rejected(self):
        api=FakeAPI();probe=self.make(api=api);api.root='D:\\'
        with self.assertRaisesRegex(ValueError,'binding drift'):probe.verify_binding()
    def test_api_failure_has_no_cached_fallback(self):
        api=FakeAPI();probe=self.make(api=api);probe.sample()
        def failure(path):raise OSError('injected disk read failure')
        api.space=failure
        with self.assertRaisesRegex(OSError,'disk read failure'):probe.sample()


if __name__=='__main__':unittest.main()
