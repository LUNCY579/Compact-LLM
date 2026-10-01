"""Stdlib fixtures; no Windows Kernel creation, subprocesses, models or GPU work."""
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, Mock

from experiments.post_training_mimo_centralized_v1.scripts import qualify_supervisor_death_cpu as qual

FROZEN = {'fixture': 'frozen'}
SOURCE = 'a' * 40
TOKEN = 'b' * 32


class Clock:
    def __init__(self): self.now = 0
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds


def item(pid, handle):
    return {'handle': handle, 'identity': {'pid': pid, 'created_filetime': pid * 100,
            'image_path': 'fixture-python.exe'},
            'lineage': [{'pid': pid}, {'pid': 100}, {'pid': os.getpid()}]}


class LifetimeKernel:
    """Independent reference-count fixture for the documented last-close rule."""
    def __init__(self, extra_reference=0):
        self.calls, self.logging_errors = [], []
        self.active, self.refs = {101: True, 201: True}, 2 + extra_reference
        self.identities = {101: item(100, 101)['identity'], 201: item(200, 201)['identity']}
        self.codes, self.remote_return, self.remote_error = {}, 1, 0
        self.wait_failure, self.change_identity = False, False
    def call(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if name == 'WaitForSingleObject':
            if self.wait_failure: return 0xffffffff, 6
            return (qual.WAIT_TIMEOUT if self.active[args[0]] else 0), 0
        if name == 'TerminateProcess':
            self.active[args[0]] = False
            self.codes[args[0]] = args[1]
            self.refs -= 1
        elif name == 'DuplicateHandle':
            assert args[0] == 201 and args[1].value == 900
            assert args[2:4] == (None, None) and args[4:] == (0, False, 1)
            self.refs -= 1  # CLOSE_SOURCE may consume even when API returns zero.
        else:
            raise AssertionError(name)
        if self.refs == 0:
            self.active[201], self.codes[201] = False, 1
        if name == 'DuplicateHandle': return self.remote_return, self.remote_error
        return 1, 0
    def identity(self, handle, pid):
        identity = dict(self.identities[handle])
        if self.change_identity: identity['created_filetime'] += 1
        return identity
    def exit_code(self, handle, label): return self.codes[handle]


class FixtureTree:
    def __init__(self, clock, kernel=None):
        self.clock, self.kernel, self.cfg = clock, kernel or LifetimeKernel(), qual.config()
        self.deadline, self.frozen_sources, self.remote_duplicate_close_attempts = 30, FROZEN, set()
        self.watchdog = SimpleNamespace(intervened=False, errors=[])
        self.process = SimpleNamespace(pid=100, _handle=101)
    def guard(self):
        qual.require(self.clock.now < self.deadline and not self.watchdog.intervened
                     and not self.kernel.logging_errors, 'fixture deadline/evidence/safety')


class ControlFixtures(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.patches = [patch.object(qual.time, 'monotonic', self.clock.monotonic),
                        patch.object(qual.time, 'sleep', self.clock.sleep),
                        patch.object(qual.prior, 'require_pinned_actual_image'),
                        patch.object(qual, 'sources', return_value=FROZEN)]
        for value in self.patches: value.start()
        self.addCleanup(lambda: [value.stop() for value in reversed(self.patches)])
        self.supervisor, self.retainer = item(100, 101), item(200, 201)
        self.armed = {'identity': self.retainer['identity'], 'token': TOKEN, 'sources': FROZEN,
                      'membership': {'api_success': True, 'member': True}, 'duplicate_value': 900}
    def invoke(self, tree, index):
        return qual.experiment_sequence(tree, self.supervisor, self.retainer, self.armed,
                                        TOKEN, FROZEN, tree.cfg['cases'][index])
    def test_retained_case_proves_supervisor_exit_then_survival_then_child_exit(self):
        tree = FixtureTree(self.clock)
        result = self.invoke(tree, 0)
        self.assertEqual(result['supervisor_exit']['exit_code'], 37)
        self.assertGreaterEqual(result['retainer_alive_after_supervisor_death']['elapsed_seconds'], 1)
        self.assertGreaterEqual(result['retainer_alive_after_supervisor_death']['samples'], 2)
        self.assertEqual(result['retainer_exit']['wait_return'], 0)
        names = [name for name, _, _ in tree.kernel.calls]
        self.assertLess(names.index('TerminateProcess'), names.index('DuplicateHandle'))
        self.assertEqual(names.count('DuplicateHandle'), 1)
    def test_closed_first_case_keeps_both_alive_before_supervisor_death(self):
        tree = FixtureTree(self.clock)
        result = self.invoke(tree, 1)
        self.assertGreaterEqual(result['both_alive_after_duplicate_close']['elapsed_seconds'], 1)
        self.assertEqual(len(result['both_alive_after_duplicate_close']['held_identities']), 2)
        names = [name for name, _, _ in tree.kernel.calls]
        self.assertLess(names.index('DuplicateHandle'), names.index('TerminateProcess'))
        self.assertFalse(result['remote_duplicate_close']['observer_inner_job_reference_acquired'])
    def test_remote_close_has_null_target_and_output_and_never_duplicates_into_observer(self):
        tree = FixtureTree(self.clock)
        result = qual.close_remote_duplicate(tree, self.retainer, self.armed, TOKEN, FROZEN)
        self.assertIsNone(result['target_process'])
        self.assertIsNone(result['output_handle'])
        self.assertEqual(result['options'], 1)
        self.assertFalse(result['observer_inner_job_reference_acquired'])
    def test_remote_close_failure_is_not_rescued_by_child_exit(self):
        kernel = LifetimeKernel()
        tree = FixtureTree(self.clock, kernel)
        qual.terminate_supervisor(tree, self.supervisor)
        kernel.remote_return, kernel.remote_error = 0, 5
        with self.assertRaisesRegex(RuntimeError, 'not credited'):
            qual.close_remote_duplicate(tree, self.retainer, self.armed, TOKEN, FROZEN)
        self.assertFalse(kernel.active[201])
        self.assertEqual(len(tree.remote_duplicate_close_attempts), 1)
    def test_remote_close_cannot_retry_after_failure_or_success(self):
        for returned in (0, 1):
            tree = FixtureTree(self.clock)
            tree.kernel.remote_return = returned
            try: qual.close_remote_duplicate(tree, self.retainer, self.armed, TOKEN, FROZEN)
            except RuntimeError: pass
            with self.assertRaisesRegex(RuntimeError, 'single-attempt'):
                qual.close_remote_duplicate(tree, self.retainer, self.armed, TOKEN, FROZEN)
            self.assertEqual(sum(name == 'DuplicateHandle' for name, _, _ in tree.kernel.calls), 1)
    def test_changed_identity_blocks_remote_close_and_supervisor_termination(self):
        for operation in ('close', 'terminate'):
            tree = FixtureTree(self.clock)
            tree.kernel.change_identity = True
            with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                if operation == 'close': qual.close_remote_duplicate(tree, self.retainer, self.armed, TOKEN, FROZEN)
                else: qual.terminate_supervisor(tree, self.supervisor)
            self.assertFalse(any(name in ('DuplicateHandle', 'TerminateProcess') for name, _, _ in tree.kernel.calls))
    def test_wrong_token_source_membership_or_handle_blocks_remote_close(self):
        changes = [('token', 'x'), ('sources', {}), ('duplicate_value', True),
                   ('membership', {'api_success': False, 'member': None}),
                   ('membership', {'api_success': True, 'member': False})]
        for key, value in changes:
            tree = FixtureTree(self.clock)
            armed = dict(self.armed, **{key: value})
            with self.assertRaises(RuntimeError):
                qual.close_remote_duplicate(tree, self.retainer, armed, TOKEN, FROZEN)
            self.assertFalse(any(name == 'DuplicateHandle' for name, _, _ in tree.kernel.calls))
    def test_extra_job_reference_prevents_exit_and_fails_control(self):
        tree = FixtureTree(self.clock, LifetimeKernel(extra_reference=1))
        with self.assertRaisesRegex(RuntimeError, 'exit phase expired'):
            self.invoke(tree, 0)
        self.assertTrue(tree.kernel.active[201])
    def test_premature_child_exit_cannot_supply_alive_interval(self):
        tree = FixtureTree(self.clock)
        tree.kernel.refs = 1  # Simulate loss of the supervisor reference too early.
        with self.assertRaisesRegex(RuntimeError, 'alive interval'):
            self.invoke(tree, 1)
    def test_failed_wait_query_cannot_supply_exit_or_alive_witness(self):
        for operation in ('exit', 'alive'):
            tree = FixtureTree(self.clock)
            tree.kernel.wait_failure = True
            with self.assertRaises(RuntimeError):
                if operation == 'exit': qual.exit_witness(tree, self.retainer)
                else: qual.observe_alive(tree, [self.retainer])
    def test_insufficient_alive_budget_blocks_before_destructive_operation(self):
        tree = FixtureTree(self.clock)
        tree.deadline = .5
        with self.assertRaisesRegex(RuntimeError, 'insufficient'):
            qual.observe_alive(tree, [self.retainer])
        self.assertEqual(tree.kernel.calls, [])
    def test_watchdog_intervention_cannot_supply_experimental_exit(self):
        tree = FixtureTree(self.clock)
        tree.watchdog.intervened = True
        with self.assertRaisesRegex(RuntimeError, 'safety'):
            self.invoke(tree, 0)
        self.assertEqual(tree.kernel.calls, [])
    def test_success_aggregation_rejects_cleanup_fallback_logging_and_safety(self):
        tree = FixtureTree(self.clock)
        cleanup = {'status': 'OWNED_CPU_PROCESSES_EXITED', 'owned_process_fallback_pids': [],
                   'uncredited_outer_safety_cleanup_used': False, 'logging_errors': []}
        witnesses = {'retainer_exit': {}, 'supervisor_exit': {}}
        self.assertTrue(qual.case_passed(None, tree, cleanup, witnesses))
        for key, value in [('status', 'BLOCKED_OWNED_CLEANUP'), ('owned_process_fallback_pids', [200]),
                           ('uncredited_outer_safety_cleanup_used', True), ('logging_errors', ['lost'])]:
            self.assertFalse(qual.case_passed(None, tree, dict(cleanup, **{key: value}), witnesses))
        tree.watchdog.intervened = True
        self.assertFalse(qual.case_passed(None, tree, cleanup, witnesses))


class DuplicateFixtures(unittest.TestCase):
    def kernel(self, member=True, api=True, close_error=False):
        identity = item(100, 101)['identity']
        value = SimpleNamespace(logging_errors=[], closed=[], identity=lambda handle, pid: identity,
                                open=lambda pid, rights: 101, current=lambda: 201)
        value.membership = lambda *args: {'api_success': api, 'member': member if api else None}
        def call(name, *args):
            self.assertEqual(name, 'DuplicateHandle')
            self.assertFalse(args[5])
            self.assertEqual(args[6], 2)
            args[3]._obj.value = 900
            return 1, 0
        def close(handle):
            numeric = handle.value if isinstance(handle, w.HANDLE) else handle
            value.closed.append(numeric)
            if close_error and numeric == 101: raise RuntimeError('fixture parent close denied')
        value.call, value.close = call, close
        return value, {'supervisor_pid': 100, 'supervisor_identity': identity, 'job_handle': w.HANDLE(800)}
    def test_specific_duplicate_is_noninheritable_and_preserves_typed_handle_fix(self):
        kernel, go = self.kernel()
        duplicate, member = qual.duplicate_inner_job(kernel, go)
        self.assertEqual(duplicate.value, 900)
        self.assertTrue(member['member'])
        self.assertEqual(kernel.closed, [101])
    def test_failed_or_wrong_membership_closes_duplicate(self):
        for member, api in ((False, True), (True, False)):
            kernel, go = self.kernel(member, api)
            with self.assertRaisesRegex(RuntimeError, 'membership'):
                qual.duplicate_inner_job(kernel, go)
            self.assertIn(900, kernel.closed)
            self.assertIn(101, kernel.closed)
    def test_parent_close_failure_also_closes_duplicate(self):
        kernel, go = self.kernel(close_error=True)
        with self.assertRaisesRegex(RuntimeError, 'parent close denied'):
            qual.duplicate_inner_job(kernel, go)
        self.assertEqual(kernel.closed, [101, 900])


class AuthorizationAndAccounting(unittest.TestCase):
    def approved(self, cfg):
        return {'parent_approved': True, 'physical_cpu_run_authorized': True,
                'experiment_id': cfg['experiment_id'], 'config_sha256': qual.CONFIG_SHA256,
                'source_sha': SOURCE, 'sources': FROZEN, 'limits': cfg['limits'],
                'independent_review': {'status': 'PASS_CPU_FIXTURES_ONLY', 'source_sha': SOURCE},
                'native_capture_authorized': False, 'capture_authorized': False,
                'capability_authorized': False, 'E1_authorized': False}
    def test_config_remains_cpu_only_with_original_strict_limits(self):
        cfg = qual.config()
        self.assertEqual(hashlib.sha256(qual.CONFIG.read_bytes()).hexdigest(), qual.CONFIG_SHA256)
        self.assertEqual([cfg['limits'][name] for name in ('suite_seconds', 'external_ceiling_seconds',
                         'tree_rss_gib', 'maximum_simultaneous_job_processes', 'maximum_total_os_processes_created')],
                         [30, 45, .5, 4, 9])
        self.assertEqual(cfg['limits']['planned_total_os_processes_created'], 4)
        for key in ('physical_run_authorized', 'native_capture_authorized', 'capture_authorized',
                    'capability_authorized', 'E1_authorized', 'retry_authorized'):
            self.assertIs(cfg[key], False)
    def test_authorization_binds_parent_review_source_budget_and_cpu_scope(self):
        cfg = qual.config()
        approved = self.approved(cfg)
        self.assertIs(qual.validate_authorization(approved, cfg, SOURCE, FROZEN), approved)
        for key, value in [('parent_approved', False), ('physical_cpu_run_authorized', 1),
                           ('source_sha', 'c' * 40), ('sources', {}), ('limits', {}),
                           ('independent_review', {}), ('native_capture_authorized', True),
                           ('capability_authorized', True)]:
            with self.assertRaises(RuntimeError):
                qual.validate_authorization(dict(approved, **{key: value}), cfg, SOURCE, FROZEN)
    def test_missing_authorization_refuses_before_kernel_popen_or_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, auth = Path(temporary) / 'output', Path(temporary) / 'missing-auth.json'
            with patch.object(qual, 'OUTPUT', output), patch.object(qual, 'AUTHORIZATION', auth), \
                 patch.object(qual, 'CONSUMED', Path(temporary) / 'receipt.json'), \
                 patch.object(qual.metadata, 'git', return_value=SOURCE), \
                 patch.object(qual.sys, 'argv', ['fixture']), patch.object(qual, 'Kernel') as kernel, \
                 patch.object(qual.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(RuntimeError, 'no parent-reviewed'):
                    qual.main()
                kernel.assert_not_called()
                popen.assert_not_called()
                self.assertFalse(output.exists())
    def test_tracked_consumed_receipt_refuses_even_without_ignored_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / 'consumed.json'
            receipt.write_text('{}')
            with patch.object(qual, 'CONSUMED', receipt), patch.object(qual, 'OUTPUT', Path(temporary) / 'absent'), \
                 patch.object(qual.sys, 'argv', ['fixture']), patch.object(qual, 'Kernel') as kernel:
                with self.assertRaisesRegex(RuntimeError, 'consumed'):
                    qual.main()
                kernel.assert_not_called()
    def fixture_receipt(self, directory, name, **changes):
        target = directory / name
        target.mkdir(exist_ok=True)
        value = dict(status='PASS_CPU_LAST_HANDLE_CONTROL_ONLY', case=name, source_sha=SOURCE, sources=FROZEN,
                     job_lifetime_processes_created=2, observed_process_creation_lower_bound=2)
        value.update(changes)
        (target / 'CASE_RESULT.json').write_text(json.dumps(value))
    def test_complete_receipts_report_native_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture_receipt(directory, 'A')
            self.fixture_receipt(directory, 'B')
            with patch.object(qual, 'OUTPUT', directory):
                result = qual.case_accounting(['A', 'B'], FROZEN, SOURCE)
            self.assertEqual(result['experimental_os_processes_created'], 4)
            self.assertEqual(result['confirmed_observed_process_lower_bound'], 4)
    def test_failed_receipt_is_unknown_with_confirmed_nonzero_lower_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture_receipt(directory, 'A', status='BLOCKED_CPU_SUPERVISOR_DEATH_CASE')
            with patch.object(qual, 'OUTPUT', directory):
                result = qual.case_accounting(['A'], FROZEN, SOURCE)
            self.assertIsNone(result['experimental_os_processes_created'])
            self.assertEqual(result['confirmed_observed_process_lower_bound'], 2)
    def test_missing_final_receipt_preserves_api_log_lower_bound_and_partial_log_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            case = directory / 'A'
            case.mkdir()
            (case / 'OBSERVER_API.jsonl').write_text(json.dumps({'event': 'owned_supervisor_root_created', 'pid': 100})
                + '\n' + json.dumps({'event': 'owned_job_lifetime_accounting', 'total_processes': 2}) + '\n{partial')
            with patch.object(qual, 'OUTPUT', directory):
                result = qual.case_accounting(['A'], FROZEN, SOURCE)
            self.assertIsNone(result['experimental_os_processes_created'])
            self.assertEqual(result['confirmed_observed_process_lower_bound'], 2)
            self.assertIn('log:', result['case_receipt_errors']['A'])
    def test_native_budget_violation_remains_unknown_but_keeps_observed_ten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture_receipt(directory, 'A', job_lifetime_processes_created=10,
                                 observed_process_creation_lower_bound=10)
            with patch.object(qual, 'OUTPUT', directory):
                result = qual.case_accounting(['A'], FROZEN, SOURCE)
            self.assertIsNone(result['experimental_os_processes_created'])
            self.assertEqual(result['confirmed_observed_process_lower_bound'], 10)
    def test_receipt_count_cannot_undercount_valid_log_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture_receipt(directory, 'A')
            (directory / 'A' / 'OBSERVER_API.jsonl').write_text(json.dumps(
                {'event': 'owned_job_lifetime_accounting', 'total_processes': 3}))
            with patch.object(qual, 'OUTPUT', directory):
                result = qual.case_accounting(['A'], FROZEN, SOURCE)
            self.assertIsNone(result['experimental_os_processes_created'])
            self.assertEqual(result['confirmed_observed_process_lower_bound'], 3)
    def test_all_48_original_and_tracked_evidence_bytes_remain_preserved(self):
        self.assertEqual(list(qual.preserved_evidence().values()), [13, 9, 26])


class ResourceAndSafetyFixtures(unittest.TestCase):
    def tree(self, directory):
        cfg = qual.config()
        kernel = SimpleNamespace(logging_errors=[], available=lambda: 10,
            call=lambda *args, **kwargs: (qual.WAIT_TIMEOUT, 0), rss=lambda handle: 1024,
            job_accounting=lambda handle: {'total_processes': 2, 'active_processes': 2},
            job_pids=lambda handle: [100, 200])
        watchdog = SimpleNamespace(intervened=False, errors=[], detach=lambda job: None)
        tree = qual.SafetyTree(kernel, directory, 30, cfg, watchdog)
        tree.process, tree.job = SimpleNamespace(pid=100, _handle=101), SimpleNamespace(handle=800)
        tree.root_identity, tree.root_lineage = item(100, 101)['identity'], item(100, 101)['lineage']
        tree.children = {200: item(200, 201)}
        return tree
    def test_live_guard_keeps_ram_rss_active_and_lifetime_limits(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(qual.time, 'monotonic', return_value=0), \
             patch.object(qual, 'OUTPUT', Path(temporary)), patch.object(qual.shutil, 'disk_usage', return_value=SimpleNamespace(free=8 * 2**30)):
            tree = self.tree(Path(temporary))
            tree.guard()
            for kind in ('ram', 'rss', 'active', 'lifetime'):
                tree = self.tree(Path(temporary))
                if kind == 'ram': tree.kernel.available = lambda: 1.249
                if kind == 'rss': tree.kernel.rss = lambda handle: .3 * 2**30
                if kind == 'active': tree.kernel.job_accounting = lambda handle: {'total_processes': 2, 'active_processes': 5}
                if kind == 'lifetime': tree.kernel.job_accounting = lambda handle: {'total_processes': 10, 'active_processes': 2}
                with self.assertRaises(RuntimeError): tree.guard()
    def test_guard_subtracts_previous_case_lifetime_from_total_ceiling(self):
        with tempfile.TemporaryDirectory() as temporary:
            tree = self.tree(Path(temporary))
            new = qual.SafetyTree(tree.kernel, Path(temporary), 30, tree.cfg, tree.watchdog, prior_total=2)
            self.assertEqual(new.maximum_total_processes, 7)
    def test_unheld_active_process_makes_outer_cleanup_uncredited(self):
        with tempfile.TemporaryDirectory() as temporary:
            tree = self.tree(Path(temporary))
            tree.kernel.call = lambda *args, **kwargs: (0, 0)
            cleanup = {'status': 'OWNED_CPU_PROCESSES_EXITED', 'final_job_accounting_before_close': {'active_processes': 1}}
            with patch.object(qual.prior.Tree, 'cleanup', return_value=cleanup):
                self.assertTrue(tree.cleanup()['uncredited_outer_safety_cleanup_used'])
    def test_watchdog_uses_only_selected_outer_job_and_keeps_persistent_failure(self):
        watchdog = qual.DeadlineSafety(100)
        job = SimpleNamespace(kernel=SimpleNamespace(record=Mock()), terminate=Mock())
        with patch.object(qual.time, 'monotonic', return_value=0): watchdog.install(job)
        watchdog.fire()
        job.terminate.assert_called_once_with()
        self.assertTrue(watchdog.intervened)
        watchdog.detach(job)
        self.assertTrue(watchdog.intervened)
        with self.assertRaisesRegex(RuntimeError, 'intervention'):
            watchdog.install(job)


if __name__ == '__main__':
    unittest.main()
