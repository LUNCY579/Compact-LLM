"""CPU fixtures for fail-closed launch ordering and held-handle cleanup."""
import json
import io
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import suspended_job_launch as launch
from experiments.post_training_mimo_centralized_v1.scripts import qualify_suspended_job_cpu as qual
from experiments.post_training_mimo_centralized_v1.scripts.windows_job_cpu import cleanup_owned, membership_result
from experiments.post_training_mimo_centralized_v1.tests.test_windows_job_cpu import FakeKernel, FakeProcess, items


class LaunchKernel:
    def __init__(self):
        self.logging_errors, self.events, self.closed = [], [], []
        self.threads, self.owner, self.thread_created, self.previous = [7], 99, 124, 1
        self.member, self.identity_calls = membership_result(1, 0, 1), 0
        self.changed_identity = False
    def record(self, event):
        self.events.append(event)
    def identity(self, handle, pid):
        self.identity_calls += 1
        return {'pid': pid, 'created_filetime': 123 + int(self.changed_identity and self.identity_calls > 1),
                'image_path': str(Path('python.exe').resolve())}
    def lineage(self, pid, parent):
        return [{'pid': pid}, {'pid': parent}]
    def membership(self, handle, job, label):
        self.events.append(label)
        return self.member
    def thread_ids(self, pid):
        return self.threads
    def call(self, name, *args, **kwargs):
        self.events.append(name)
        if name == 'OpenThread': return 70, 0
        if name == 'GetProcessIdOfThread': return self.owner, 0
        if name == 'GetThreadTimes':
            args[1]._obj.dwLowDateTime = self.thread_created
            return 1, 0
        if name == 'ResumeThread': return self.previous, 0
        raise AssertionError(name)
    def close(self, handle):
        self.closed.append(handle)


class FakeJob:
    handle = 800
    def __init__(self, kernel, process):
        kernel.events.append('ASSIGN_JOB')


class SuspendedLaunchControls(unittest.TestCase):
    def invoke(self, kernel, **kwargs):
        def create(command, **options):
            self.assertEqual(options['creationflags'], 0x204)
            return FakeProcess()
        with patch.object(launch, 'cleanup_owned', return_value={'status': 'fixture_cleanup'}) as cleanup:
            try:
                value = launch.launch_suspended(kernel, ['python.exe'], Path.cwd(), None, None,
                                               5, popen_factory=create, job_factory=FakeJob, **kwargs)
            except Exception:
                self.assertEqual(cleanup.call_count, 1)
                raise
        return value

    def test_specific_membership_is_verified_twice_before_exactly_one_resume(self):
        kernel = LaunchKernel()
        with patch.object(launch.time, 'monotonic', return_value=0):
            process, job, proof = self.invoke(kernel)
        events = kernel.events
        self.assertLess(events.index('ASSIGN_JOB'), events.index('suspended_root_assigned_before_resume'))
        self.assertLess(events.index('suspended_root_immediately_before_resume'), events.index('ResumeThread'))
        self.assertEqual(events.count('ResumeThread'), 1)
        self.assertEqual(proof['previous_suspend_count'], 1)
        self.assertTrue(proof['specific_membership_before_resume']['member'])
        self.assertEqual(kernel.closed, [70])

    def test_failed_api_or_successful_nonmembership_never_resumes(self):
        for witness in (membership_result(0, 5, -1), membership_result(1, 0, 0)):
            kernel = LaunchKernel()
            kernel.member = witness
            with patch.object(launch.time, 'monotonic', return_value=0), self.assertRaisesRegex(RuntimeError, 'membership'):
                self.invoke(kernel)
            self.assertNotIn('ResumeThread', kernel.events)

    def test_zero_or_multiple_owned_threads_never_resume(self):
        for threads in ([], [7, 8]):
            kernel = LaunchKernel()
            kernel.threads = threads
            with patch.object(launch.time, 'monotonic', return_value=0), self.assertRaisesRegex(RuntimeError, 'exactly one'):
                self.invoke(kernel)
            self.assertNotIn('ResumeThread', kernel.events)

    def test_wrong_thread_owner_or_stale_thread_creation_never_resumes(self):
        for field, value in [('owner', 100), ('thread_created', 122)]:
            kernel = LaunchKernel()
            setattr(kernel, field, value)
            with patch.object(launch.time, 'monotonic', return_value=0), self.assertRaises(RuntimeError):
                self.invoke(kernel)
            self.assertNotIn('ResumeThread', kernel.events)
            self.assertEqual(kernel.closed, [70])

    def test_replaced_root_creation_identity_never_resumes(self):
        kernel = LaunchKernel()
        kernel.changed_identity = True
        with patch.object(launch.time, 'monotonic', return_value=0), self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
            self.invoke(kernel)
        self.assertNotIn('ResumeThread', kernel.events)

    def test_expired_deadline_immediately_before_resume_blocks(self):
        kernel = LaunchKernel()
        with patch.object(launch.time, 'monotonic', side_effect=[0, 6]), self.assertRaisesRegex(RuntimeError, 'resume membership/deadline'):
            self.invoke(kernel)
        self.assertNotIn('ResumeThread', kernel.events)

    def test_bad_previous_suspend_count_is_blocked_without_an_extra_resume(self):
        for previous in (0, 2, 0xffffffff):
            kernel = LaunchKernel()
            kernel.previous = previous
            with patch.object(launch.time, 'monotonic', return_value=0), self.assertRaisesRegex(RuntimeError, 'suspend count'):
                self.invoke(kernel)
            self.assertEqual(kernel.events.count('ResumeThread'), 1)
            self.assertEqual(kernel.closed, [70])

    def test_postresume_close_logging_failure_blocks_returning_a_usable_launch(self):
        kernel = LaunchKernel()
        def close(handle):
            kernel.closed.append(handle)
            kernel.logging_errors.append('injected close evidence loss')
        kernel.close = close
        with patch.object(launch.time, 'monotonic', return_value=0), self.assertRaisesRegex(RuntimeError, 'evidence lost'):
            self.invoke(kernel)
        self.assertEqual(kernel.events.count('ResumeThread'), 1)
        self.assertEqual(kernel.closed, [70])

    def test_job_assignment_failure_never_resumes_and_cleans_the_owned_root(self):
        kernel = LaunchKernel()
        with patch.object(launch.time, 'monotonic', return_value=0), \
             patch.object(launch, 'cleanup_owned', return_value={'status': 'fixture_cleanup'}) as cleanup, \
             self.assertRaisesRegex(RuntimeError, 'injected assignment'):
            launch.launch_suspended(kernel, ['python.exe'], Path.cwd(), None, None, 5,
                popen_factory=lambda *args, **kwargs: FakeProcess(),
                job_factory=lambda *args: (_ for _ in ()).throw(RuntimeError('injected assignment')))
        self.assertNotIn('ResumeThread', kernel.events)
        self.assertEqual(cleanup.call_args.args[3].pid, 99)

    def test_signaled_held_process_needs_no_exiting_image_query_or_manual_kill(self):
        owned = items()
        class ExitedKernel(FakeKernel):
            def identity(self, handle, pid):
                raise AssertionError('already exited process image must not be queried')
            def call(self, name, handle, *args, **kwargs):
                return (0, 0) if name == 'WaitForSingleObject' else super().call(name, handle, *args, **kwargs)
        kernel, process = ExitedKernel(owned), FakeProcess()
        process.returncode = 0
        result = cleanup_owned(kernel, None, owned, process, 9, manual_fallback=False)
        self.assertEqual(result['status'], 'OWNED_CPU_PROCESSES_EXITED')
        self.assertEqual(result['owned_process_fallback_pids'], [])

    def test_image_query_failure_is_an_exit_race_only_after_same_held_handle_signals(self):
        owned = items()
        class ExitingKernel(FakeKernel):
            def __init__(self, entries):
                super().__init__(entries)
                self.waits = {}
            def identity(self, handle, pid):
                raise RuntimeError('image query 31 while exiting')
            def call(self, name, handle, *args, **kwargs):
                if name == 'WaitForSingleObject' and handle != 99:
                    self.waits[handle] = self.waits.get(handle, 0) + 1
                    return (258 if self.waits[handle] == 1 else 0), 0
                return super().call(name, handle, *args, **kwargs)
        kernel, process = ExitingKernel(owned), FakeProcess()
        process.returncode = 0
        result = cleanup_owned(kernel, None, owned, process, 9, manual_fallback=False)
        self.assertEqual(result['status'], 'OWNED_CPU_PROCESSES_EXITED')
        self.assertEqual(kernel.terminated, [])
        self.assertTrue(all('exit_during_identity_query' in row for row in result['processes'][:-1]))

    def test_unverified_live_image_failure_remains_blocked_without_kill(self):
        owned = items()
        class BadImageKernel(FakeKernel):
            def identity(self, handle, pid):
                raise RuntimeError('image query 31')
        kernel, process = BadImageKernel(owned), FakeProcess()
        process.returncode = 0
        result = cleanup_owned(kernel, None, owned, process, 9, manual_fallback=False)
        self.assertEqual(result['status'], 'BLOCKED_OWNED_CLEANUP')
        self.assertEqual(kernel.terminated, [])
        self.assertEqual(kernel.closed, [100, 200])

    def test_physical_cleanup_controls_cannot_pass_after_manual_fallback(self):
        owned, process = items(), FakeProcess()
        kernel = FakeKernel(owned)
        process.returncode = 0
        result = cleanup_owned(kernel, None, owned, process, 9, manual_fallback=False)
        self.assertEqual(result['status'], 'BLOCKED_OWNED_CLEANUP')
        self.assertEqual(kernel.terminated, [])

    def test_go_is_not_published_after_identity_api_evidence_loss(self):
        class LostKernel:
            def __init__(self): self.logging_errors = []
            def current(self): return 1
            def identity(self, handle, pid):
                self.logging_errors.append('injected identity log loss')
                return {'pid': pid}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaisesRegex(RuntimeError, 'GO evidence'):
                qual.publish_go(directory, 'GO.json', 'a' * 32, {'pid': 99}, {}, LostKernel(), 1, 'source')
            self.assertFalse((directory / 'GO.json').exists())

    def test_unobserved_resources_are_json_null_and_never_fabricated(self):
        with tempfile.TemporaryDirectory() as temp:
            tree = qual.Tree(LaunchKernel(), Path(temp), 5)
            data = json.loads(json.dumps(tree.resources(), allow_nan=False))
            self.assertIsNone(data['minimum_observed_available_ram_gib'])
            self.assertIsNone(data['minimum_observed_disk_free_gib'])
            self.assertEqual(tree.observed_created_pids(), [])

    def test_registration_keeps_resource_limits_and_framework_grants_frozen(self):
        cfg = qual.config()
        limits = cfg['limits']
        self.assertEqual(limits['maximum_simultaneous_job_processes'], 4)
        self.assertEqual([limits[k] for k in ('suite_seconds', 'external_ceiling_seconds', 'tree_rss_gib',
                                             'host_reserve_gib', 'preworker_host_gib', 'disk_free_gib')],
                         [30, 45, .5, 1.25, 3.3125, 4])
        self.assertEqual(qual.zero_scope()['forbidden_modules'], [])
        self.assertEqual(cfg['model_operations'], 0)
        self.assertEqual(cfg['gpu_operations'], 0)
        self.assertFalse(cfg['native_capture_authorized'])

    def test_active_termination_proof_waits_on_every_owned_handle_with_job_still_open(self):
        owned = items()
        class SignaledKernel(FakeKernel):
            def call(self, name, handle, *args, **kwargs):
                self_outer.assertEqual(name, 'WaitForSingleObject')
                return 0, 0
            def exit_code(self, handle, label):
                return 2
        self_outer = self
        tree = SimpleNamespace(job=SimpleNamespace(handle=800), process=FakeProcess(),
                               root_identity={'pid': 99}, children=owned, kernel=SignaledKernel(owned),
                               guard=lambda: None, deadline=5)
        with patch.object(qual.time, 'monotonic', return_value=0):
            witnesses = qual.confirm_job_termination_before_close(tree)
        self.assertEqual([row['identity']['pid'] for row in witnesses], [99, 100, 200])
        self.assertTrue(all(row['wait_return'] == 0 and row['inner_job_handle_still_open'] == 800 for row in witnesses))
        self.assertEqual(tree.job.handle, 800)

    def test_active_termination_timeout_blocks_before_any_job_close(self):
        tree = SimpleNamespace(job=SimpleNamespace(handle=800), process=FakeProcess(),
                               root_identity={'pid': 99}, children=items(), kernel=FakeKernel(items()),
                               guard=lambda: None, deadline=5)
        with patch.object(qual.time, 'monotonic', return_value=0), \
             self.assertRaisesRegex(RuntimeError, 'termination unproved before close'):
            qual.confirm_job_termination_before_close(tree)
        self.assertEqual(tree.job.handle, 800)

    def test_active_termination_wrong_exit_code_blocks_with_inner_job_still_open(self):
        kernel = FakeKernel(items())
        kernel.call = lambda *args, **kwargs: (0, 0)
        kernel.exit_code = lambda *args: 1
        tree = SimpleNamespace(job=SimpleNamespace(handle=800), process=FakeProcess(),
                               root_identity={'pid': 99}, children=items(), kernel=kernel,
                               guard=lambda: None, deadline=5)
        with patch.object(qual.time, 'monotonic', return_value=0), \
             self.assertRaisesRegex(RuntimeError, 'exit code two'):
            qual.confirm_job_termination_before_close(tree)
        self.assertEqual(tree.job.handle, 800)

    def test_startup_exception_consumes_one_cpu_attempt_with_blocking_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'owned'
            with patch.object(qual, 'OUTPUT', output), \
                 patch.object(qual, 'outer', side_effect=RuntimeError('injected admission failure')), \
                 patch.object(qual.sys, 'stdout', io.StringIO()):
                self.assertEqual(qual.guarded_outer(), 2)
            receipt = qual.read(output.with_name('owned_outer') / 'OUTER_EXCEPTION.json')
            self.assertEqual(receipt['status'], 'BLOCKED_CPU_OUTER_EXCEPTION')
            self.assertFalse(receipt['native_capture_authorized'])
            self.assertEqual(receipt['model_operations'], 0)
            self.assertIn('not assumed zero', receipt['process_creation_accounting'])

    def test_startup_failure_cannot_overwrite_a_previous_blocking_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'owned'
            directory = output.with_name('owned_outer')
            directory.mkdir()
            receipt = directory / 'OUTER_EXCEPTION.json'
            qual.write(receipt, {'status': 'ORIGINAL_CPU_BLOCK'})
            original = receipt.read_bytes()
            with patch.object(qual, 'OUTPUT', output), \
                 patch.object(qual, 'outer', side_effect=RuntimeError('later startup blocked')), \
                 patch.object(qual.sys, 'stdout', io.StringIO()):
                self.assertEqual(qual.guarded_outer(), 2)
            self.assertEqual(receipt.read_bytes(), original)

    def test_redirected_actual_interpreter_or_changed_image_bytes_block(self):
        with tempfile.TemporaryDirectory() as temp:
            pinned = Path(temp) / 'pinned.exe'
            pinned.write_bytes(b'frozen image')
            cfg = {'cpu_interpreter': str(pinned),
                   'cpu_interpreter_sha256': hashlib.sha256(pinned.read_bytes()).hexdigest()}
            qual.require_pinned_actual_image({'image_path': str(pinned)}, cfg)
            with self.assertRaisesRegex(RuntimeError, 'redirected'):
                qual.require_pinned_actual_image({'image_path': str(Path(temp) / 'redirected.exe')}, cfg)
            pinned.write_bytes(b'changed image')
            with self.assertRaisesRegex(RuntimeError, 'bytes changed'):
                qual.require_pinned_actual_image({'image_path': str(pinned)}, cfg)

    def test_job_lifetime_ceiling_blocks_even_with_only_one_sampled_pid(self):
        kernel = LaunchKernel()
        kernel.available = lambda: 5
        kernel.job_accounting = lambda handle: {'total_processes': 10, 'active_processes': 1}
        with tempfile.TemporaryDirectory() as temp:
            tree = qual.Tree(kernel, Path(temp), 5)
            tree.process, tree.job = FakeProcess(), SimpleNamespace(handle=800)
            tree.rss_if_active = lambda handle: 0
            with patch.object(qual.time, 'monotonic', return_value=0), \
                 self.assertRaisesRegex(RuntimeError, 'lifetime/active'):
                tree.guard()
            self.assertEqual(tree.observed_created_pids(), [99])
            self.assertEqual(tree.peak_job_lifetime_processes, 10)

    def test_final_job_accounting_failure_still_attempts_owned_cleanup(self):
        kernel = LaunchKernel()
        kernel.job_accounting = lambda handle: {'total_processes': 10, 'active_processes': 0}
        with tempfile.TemporaryDirectory() as temp:
            tree = qual.Tree(kernel, Path(temp), 5)
            tree.job = SimpleNamespace(handle=800)
            expected = {'status': 'OWNED_CPU_PROCESSES_EXITED', 'errors': []}
            with patch.object(qual, 'cleanup_owned', return_value=expected) as cleanup:
                result = tree.cleanup()
            cleanup.assert_called_once()
            self.assertEqual(result['status'], 'BLOCKED_OWNED_CLEANUP')
            self.assertEqual(result['final_job_accounting_before_close']['total_processes'], 10)


if __name__ == '__main__':
    unittest.main()
