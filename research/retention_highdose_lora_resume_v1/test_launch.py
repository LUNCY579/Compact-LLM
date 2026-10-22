"""Pure launcher admission fixtures: no resource queries, sleeps, or subprocesses."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('highdose_launch_under_test',HERE/'launch.py')
launch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LaunchAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        (self.root/'PRELAUNCH_BINDING.json').write_text('{}',encoding='utf-8')
        self.waitroot=self.root/'wait';self.waitroot.mkdir()
        self.envelope={'native_limits':{'total_owned_work_wall_seconds':2700},'deadline_unix':10000,
                       'pretrain_host_bytes':7,'pretrain_free_disk_bytes':17}
        self.ready={'available_host_bytes':8,'free_disk_bytes':20,'lease_present':False}
        self.out=patch.object(launch,'OUT',self.root);self.out.start();self.addCleanup(self.out.stop)

    def terminal(self):
        return json.loads((self.waitroot/'PRELOAD_TERMINAL.json').read_text())

    def test_ready_resources_do_not_override_expired_wait(self):
        with patch.object(launch,'verify_frozen'),patch.object(launch,'resource_snapshot',return_value=self.ready), \
             patch.object(launch.time,'monotonic',return_value=900),patch.object(launch.time,'time',return_value=100), \
             patch.object(launch.subprocess,'Popen') as spawn:
            self.assertEqual(launch.dispatch_once(self.waitroot,'one',['unused'],self.envelope,0),2)
        spawn.assert_not_called()
        self.assertEqual(self.terminal()['reason'],'PRELOAD_WAIT_LIMIT_REACHED')
        self.assertFalse(self.terminal()['worker_launched'])

    def test_ready_resources_do_not_override_native_deadline(self):
        with patch.object(launch,'verify_frozen'),patch.object(launch,'resource_snapshot',return_value=self.ready), \
             patch.object(launch.time,'monotonic',return_value=1),patch.object(launch.time,'time',return_value=7301), \
             patch.object(launch.subprocess,'Popen') as spawn:
            self.assertEqual(launch.dispatch_once(self.waitroot,'one',['unused'],self.envelope,0),2)
        spawn.assert_not_called()
        self.assertEqual(self.terminal()['reason'],'SESSION_CANNOT_FIT_NATIVE_OWNED_WORK')
        self.assertFalse((self.root/launch.RESERVATION_NAME).exists())

    def test_final_deadline_recheck_preserves_unused_reservation(self):
        with patch.object(launch,'verify_frozen'),patch.object(launch,'resource_snapshot',return_value=self.ready), \
             patch.object(launch.time,'monotonic',return_value=1),patch.object(launch.time,'time',side_effect=[7200,7200,7301]), \
             patch.object(launch.subprocess,'Popen') as spawn:
            self.assertEqual(launch.dispatch_once(self.waitroot,'one',['unused'],self.envelope,0),2)
        spawn.assert_not_called()
        self.assertTrue((self.root/launch.RESERVATION_NAME).is_file())
        self.assertFalse(self.terminal()['controller_launched'])

    def test_raced_wait_ids_have_one_atomic_winner(self):
        def reserve(wait_id):
            try:
                launch.reserve_dispatch(wait_id,['unused'],self.envelope,self.ready)
                return 'reserved'
            except FileExistsError:
                return 'refused'
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes=list(pool.map(reserve,('first','second')))
        self.assertCountEqual(outcomes,['reserved','refused'])
        claim=json.loads((self.root/launch.RESERVATION_NAME).read_text())
        self.assertIn(claim['wait_id'],('first','second'))
        self.assertEqual(claim['owner_pid'],launch.os.getpid())
        self.assertEqual(claim['native_owned_seconds_reserved'],2700)

    def test_spawn_failure_records_no_worker_and_cannot_redispatch(self):
        with patch.object(launch,'verify_frozen'),patch.object(launch,'resource_snapshot',return_value=self.ready), \
             patch.object(launch.time,'monotonic',return_value=1),patch.object(launch.time,'time',return_value=100), \
             patch.object(launch.subprocess,'Popen',side_effect=OSError('fixture launch failure')) as spawn:
            self.assertEqual(launch.dispatch_once(self.waitroot,'one',['unused'],self.envelope,0),2)
        self.assertEqual(spawn.call_count,1)
        self.assertFalse(self.terminal()['worker_launched'])
        self.assertFalse(self.terminal()['controller_launched'])
        with self.assertRaises(FileExistsError):
            launch.reserve_dispatch('two',['unused'],self.envelope,self.ready)

    def test_mocked_controller_start_does_not_claim_model_worker_started(self):
        child=SimpleNamespace(pid=123,wait=Mock(return_value=0))
        with patch.object(launch,'verify_frozen'),patch.object(launch,'resource_snapshot',return_value=self.ready), \
             patch.object(launch.time,'monotonic',return_value=1),patch.object(launch.time,'time',return_value=100), \
             patch.object(launch.subprocess,'Popen',return_value=child) as spawn:
            self.assertEqual(launch.dispatch_once(self.waitroot,'one',['unused'],self.envelope,0),0)
        spawn.assert_called_once();child.wait.assert_called_once()
        self.assertTrue(self.terminal()['controller_launched'])
        self.assertIsNone(self.terminal()['worker_launched'])
        self.assertTrue((self.root/launch.RESERVATION_NAME).is_file())


if __name__=='__main__':
    unittest.main()
