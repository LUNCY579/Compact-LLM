"""CPU-only regression controls; no process launch or Windows API calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import diagnose_windows_job as probe
from experiments.post_training_mimo_centralized_v1.scripts import windows_job_cpu as windows
from experiments.post_training_mimo_centralized_v1.scripts.windows_job_cpu import Kernel, membership_result, validate_owned_identity, cleanup_owned


class FakeProcess:
    pid = 99
    returncode = None
    def poll(self):
        return self.returncode
    def kill(self):
        self.returncode = 2
    def wait(self, timeout):
        return self.returncode


class FakeKernel:
    def __init__(self, items):
        self.items, self.terminated, self.closed = items, [], []
        self.logging_errors = []
    def identity(self, handle, pid):
        return self.items[pid]['identity']
    def call(self, name, handle, *args, **kwargs):
        if name == 'TerminateProcess':
            self.terminated.append(handle)
            return 1, 0
        if name == 'WaitForSingleObject':
            return (0 if handle in self.terminated else 258), 0
        raise AssertionError(name)
    def close(self, handle):
        self.closed.append(handle)


def items():
    return {pid: {'handle': pid, 'identity': {'pid': pid, 'created_filetime': 123 + pid, 'image_path': 'python.exe'},
                  'lineage': [{'pid': pid}, {'pid': 99}, {'pid': 9}]} for pid in (100, 200)}


class CPUJobControls(unittest.TestCase):
    def test_api_failure_never_reads_truthy_output_as_membership(self):
        self.assertEqual(membership_result(0, 6, -1), {'api_return': 0, 'last_error': 6, 'api_success': False, 'member': None})

    def test_successful_nonmembership_is_distinct_from_query_failure(self):
        self.assertEqual(membership_result(1, 0, 0)['member'], False)
        self.assertTrue(membership_result(1, 0, 0)['api_success'])
        self.assertTrue(membership_result(1, 0, 1)['member'])

    def test_pid_creation_and_image_replacement_are_rejected(self):
        expected = items()[100]
        for key, value in [('pid', 101), ('created_filetime', 9999), ('image_path', 'foreign.exe')]:
            observed = dict(expected['identity'])
            observed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                validate_owned_identity(expected['identity'], observed, expected['lineage'], 9, 99)

    def test_unrelated_or_incomplete_lineage_is_rejected(self):
        entry = items()[100]
        for lineage in ([{'pid': 100}, {'pid': 9}], [{'pid': 200}, {'pid': 99}, {'pid': 9}],
                        [{'pid': 100}, {'pid': 99}, {'pid': 8}]):
            with self.subTest(lineage=lineage), self.assertRaisesRegex(RuntimeError, 'lineage mismatch'):
                validate_owned_identity(entry['identity'], entry['identity'], lineage, 9, 99)

    def test_ready_is_unpublished_until_complete_data_is_fsynced(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'READY.json'
            real_rename, real_fsync, witnessed = probe.os.rename, probe.os.fsync, []
            def fsync(fd):
                real_fsync(fd)
                witnessed.append('fsync')
            def publish(pending, destination):
                self.assertEqual(witnessed, ['fsync'])
                self.assertFalse(path.exists())
                self.assertEqual(json.loads(Path(pending).read_bytes()), {'token': 'complete'})
                return real_rename(pending, destination)
            with patch.object(probe.os, 'fsync', side_effect=fsync), patch.object(probe.os, 'rename', side_effect=publish):
                probe.write(path, {'token': 'complete'})
            self.assertEqual(probe.read(path), {'token': 'complete'})

    def test_existing_evidence_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'GO.json'
            original = b'{"token":"original"}\n'
            path.write_bytes(original)
            with self.assertRaises(FileExistsError):
                probe.write(path, {'token': 'replacement'})
            self.assertEqual(path.read_bytes(), original)

    def test_already_published_file_cannot_bypass_phase_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'READY.json'
            probe.write(path, {'ready': True})
            with patch.object(probe.time, 'monotonic', return_value=4), self.assertRaisesRegex(RuntimeError, 'phase deadline'):
                probe.wait_file(path, 3)

    def test_cleanup_continues_after_job_close_failure(self):
        class BadJob:
            def close(self):
                raise RuntimeError('injected Job close failure')
        owned = items()
        kernel, process = FakeKernel(owned), FakeProcess()
        result = cleanup_owned(kernel, BadJob(), owned, process, 9)
        self.assertEqual(kernel.terminated, [100, 200])
        self.assertEqual(kernel.closed, [100, 200])
        self.assertEqual(process.returncode, 2)
        self.assertEqual(result['status'], 'BLOCKED_OWNED_CLEANUP')

    def test_changed_identity_is_preserved_while_other_owned_cleanup_completes(self):
        owned = items()
        observed = copy.deepcopy(owned)
        observed[100]['identity']['created_filetime'] += 1
        kernel, process = FakeKernel(observed), FakeProcess()
        result = cleanup_owned(kernel, None, owned, process, 9)
        self.assertEqual(kernel.terminated, [200])
        self.assertEqual(kernel.closed, [100, 200])
        self.assertEqual(process.returncode, 2)
        self.assertEqual(result['status'], 'BLOCKED_OWNED_CLEANUP')

    def test_cached_lineage_supports_cleanup_after_parent_exit(self):
        owned = items()
        kernel, process = FakeKernel(owned), FakeProcess()
        process.returncode = 0
        result = cleanup_owned(kernel, None, owned, process, 9)
        self.assertEqual(result['status'], 'OWNED_CPU_PROCESSES_EXITED')
        self.assertEqual(kernel.terminated, [100, 200])
        # FakeKernel has no lineage query: cleanup uses previously verified lineage.

    def test_logging_failure_is_retained_without_interrupting_native_cleanup_call(self):
        kernel = object.__new__(Kernel)
        kernel.logging_errors = []
        def failing_observer(event):
            raise OSError('injected log failure')
        kernel.observer = failing_observer
        kernel.record({'api': 'CloseHandle', 'return': 1})
        self.assertIn('injected log failure', kernel.logging_errors[0])

    def test_imports_and_frozen_scope_allow_no_framework_or_native_capture(self):
        self.assertEqual(probe.forbidden_modules(), [])
        cfg = probe.read(probe.CONFIG)
        self.assertEqual(cfg['limits']['worker_roots'], 2)
        self.assertEqual(cfg['limits']['suite_seconds'], 30)
        self.assertEqual(cfg['limits']['external_ceiling_seconds'], 45)
        self.assertEqual(cfg['model_operations'], 0)
        self.assertEqual(cfg['gpu_operations'], 0)
        self.assertFalse(cfg['native_capture_authorized'])

    def test_disk_drop_after_initial_admission_blocks_the_next_guard(self):
        kernel = SimpleNamespace(available=lambda: 5)
        with patch.object(windows.shutil, 'disk_usage', side_effect=[SimpleNamespace(free=5 * 2**30), SimpleNamespace(free=3 * 2**30)]):
            windows.require_reserves(kernel, 'owned-directory')
            with self.assertRaisesRegex(RuntimeError, 'disk reserve'):
                windows.require_reserves(kernel, 'owned-directory')

    def test_worker_lost_api_log_cannot_publish_success(self):
        class WorkerKernel:
            logging_errors = ['injected worker evidence loss']
            def current(self): return 1
            def identity(self, handle, pid): return {'pid': pid, 'created_filetime': 123, 'image_path': 'python.exe'}
            def lineage(self, pid, parent): return [{'pid': pid}, {'pid': parent}]
            def open(self, pid, rights): return 10
            def call(self, name, *args):
                self_outer.assertEqual(name, 'DuplicateHandle')
                args[3]._obj.value = 2
                return 1, 0
            def membership(self, handle, job, label):
                return membership_result(0, 6, -1) if 'negative' in label else membership_result(1, 0, 0)
            def close(self, handle): pass
        self_outer = self
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            directory = output / 'IMMEDIATE_ASSIGN'
            directory.mkdir()
            args = SimpleNamespace(case_path=directory, supervisor_pid=123, token='a' * 32)
            go = {'token': args.token, 'sources': {'fixed': 'digest'}, 'supervisor_pid': 123,
                  'supervisor_identity': {'pid': 123, 'created_filetime': 123, 'image_path': 'python.exe'}, 'job_handle': 99}
            with patch.object(probe, 'OUTPUT', output), patch.object(probe, 'Kernel', return_value=WorkerKernel()), \
                 patch.object(probe, 'sources', return_value=go['sources']), patch.object(probe, 'wait_file', return_value=go):
                with self.assertRaisesRegex(RuntimeError, 'evidence logging failed'):
                    probe.worker(args)
            self.assertFalse((directory / 'WORKER_RESULT.json').exists())


if __name__ == '__main__':
    unittest.main()
