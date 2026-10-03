"""Metadata-only recovery fixtures; never run against real completed outputs."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.binding_integration_v1 import cost_report_recovered as c
from research.binding_integration_v1.test_cost_report import probe_fixture, read, write


def fixture(root):
    env = probe_fixture(root)
    old_path = root / 'posttrain/TERMINAL.json'
    canonical = read(old_path)
    old = {'status': 'FAILED_OR_BLOCKED', 'attempts': canonical['attempts'],
           'failure': {'type': 'ValueError', 'message': 'six-probe attempt receipt inventory differs'},
           'global_reservation_accounting': {'attempts': 6}}
    write(old_path, old)
    dispatch = root / 'posttrain/DISPATCH_RESERVATION.json'
    write(dispatch, {'owner': {'owner_pid': 11}, 'legs': list(c.LEGS), 'automatic_retries': 0})
    code = root / 'code'; code.mkdir()
    report = code / 'report.py'; report.write_text('# frozen fixture reporter\n', encoding='utf-8')
    helper = code / 'accounting_recovery.py'; helper.write_text('# reviewed fixture helper\n', encoding='utf-8')
    formal = root / 'ACCOUNTING_CORRECTION_FREEZE.json'; write(formal, {'scope': 'fixture-only correction'})
    observation = root / 'CONTROLLER_EXIT_OBSERVATION.json'; write(observation, {'observed_returncode': 1, 'source': 'parent observed exec'})
    write(root / 'PRELAUNCH_BINDING.json', {'code_files_sha256': {'report.py': c.sha(report)}})
    receipt = {'status': c.EXPECTED_RECOVERY_STATUS, 'original_wrapper_returncode': 1,
        'elapsed_time_basis': c.ELAPSED_BASIS, 'session_elapsed_seconds': 120.,
        'final_binding_sha256': canonical['final_binding_sha256'], 'cost_accounting': canonical['cost_accounting']}
    for stem, path in [('prior_terminal', old_path), ('dispatch', dispatch), ('correction_freeze', formal),
                       ('original_report', report), ('correction_module', helper), ('controller_exit_observation', observation)]:
        receipt[stem + '_path'] = str(path); receipt[stem + '_sha256'] = c.sha(path)
    receipt['proof_files_sha256'] = {str(old_path): c.sha(old_path), str(dispatch): c.sha(dispatch)}
    output = root / 'posttrain/ACCOUNTING_COMPLETE.json'; write(output, receipt)
    canonical['session_elapsed_seconds'] = 120.
    canonical['accounting_correction'] = {'receipt': str(output), 'original_terminal_status': 'FAILED_OR_BLOCKED'}
    api = SimpleNamespace(verify_accounting_complete=lambda selected: read(selected / 'posttrain/ACCOUNTING_COMPLETE.json'),
                          completed_terminal_view=lambda value: copy.deepcopy(canonical))
    return env, code, api, old_path, canonical


class RecoveredCostTests(unittest.TestCase):
    def test_original_probe_ast_changes_only_one_metadata_exclusion(self):
        c.assert_original_checks_preserved()

    def test_verified_correction_passes_same_six_probe_cost_checks_without_rewriting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); env, here, api, old_path, _ = fixture(root)
            original_bytes = old_path.read_bytes()
            with patch.object(c, 'HERE', here):
                context = c.recovery_context(root, api)
                costs = c.recovered_probe_costs(root, env, read, context=context)
            self.assertEqual(costs['totals']['attempts'], 6)
            self.assertEqual(costs['totals']['rows_reserved'], 2454)
            self.assertEqual(costs['totals']['generation_tokens_reserved'], 138768)
            self.assertEqual(costs['original_posttrain_wrapper_returncode'], 1)
            self.assertEqual(costs['original_posttrain_terminal_status'], 'FAILED_OR_BLOCKED')
            self.assertEqual(costs['session_elapsed_at_accounting_correction_seconds'], 120.)
            self.assertNotIn('session_elapsed_at_probe_completion_seconds', costs)
            self.assertEqual(old_path.read_bytes(), original_bytes)
            self.assertEqual(read(old_path)['status'], 'FAILED_OR_BLOCKED')

    def test_other_failure_cannot_be_called_bookkeeping_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); _, here, api, old_path, _ = fixture(root)
            old = read(old_path); old['failure']['message'] = 'a model worker failed'; write(old_path, old)
            path = root / 'posttrain/ACCOUNTING_COMPLETE.json'; receipt = read(path)
            receipt['prior_terminal_sha256'] = c.sha(old_path); receipt['proof_files_sha256'][str(old_path)] = c.sha(old_path); write(path, receipt)
            with patch.object(c, 'HERE', here), self.assertRaisesRegex(ValueError, 'exact post-six-success'):
                c.recovery_context(root, api)

    def test_missing_wrapper_observation_cannot_silently_become_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); _, here, api, _, _ = fixture(root)
            path = root / 'posttrain/ACCOUNTING_COMPLETE.json'; receipt = read(path); receipt['original_wrapper_returncode'] = 0; write(path, receipt)
            with patch.object(c, 'HERE', here), self.assertRaisesRegex(ValueError, 'wrapper failure'):
                c.recovery_context(root, api)

    def test_changed_dispatch_proof_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); _, here, api, _, _ = fixture(root)
            write(root / 'posttrain/DISPATCH_RESERVATION.json', {'changed': True})
            with patch.object(c, 'HERE', here), self.assertRaisesRegex(ValueError, 'provenance bytes'):
                c.recovery_context(root, api)

    def test_model_retry_map_is_not_an_accounting_correction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); _, here, api, _, _ = fixture(root)
            write(root / 'ATTEMPT_MAP.json', {'replacements': {}})
            with patch.object(c, 'HERE', here), self.assertRaisesRegex(ValueError, 'model retry'):
                c.recovery_context(root, api)

    def test_virtual_costs_cannot_omit_or_change_one_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); _, here, api, _, canonical = fixture(root)
            canonical['attempts'].pop()
            with patch.object(c, 'HERE', here), self.assertRaisesRegex(ValueError, 'virtual accounting view'):
                c.recovery_context(root, api)

    def test_extra_posttrain_metadata_not_globally_whitelisted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); env, here, api, _, _ = fixture(root)
            write(root / 'posttrain/OTHER.json', {'not': 'authorized'})
            with patch.object(c, 'HERE', here):
                context = c.recovery_context(root, api)
                with self.assertRaisesRegex(ValueError, 'unexpected six-leg'):
                    c.recovered_probe_costs(root, env, read, context=context)

    def test_in_memory_reader_override_is_always_restored(self):
        saved = c.original.probe_costs
        with self.assertRaises(RuntimeError):
            with c.cost_reader_adapter(None):
                self.assertIsNot(c.original.probe_costs, saved)
                raise RuntimeError('fixture interruption')
        self.assertIs(c.original.probe_costs, saved)

    def test_failed_recovery_verification_cannot_produce_complete_cost_report(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(c, 'load_recovery_api', side_effect=ValueError('unverified correction')):
                result = c.build_report(Path(folder))
            self.assertEqual(result['status'], 'BLOCKED_COST_ACCOUNTING')
            self.assertEqual(result['blockers'], ['unverified correction'])
            self.assertFalse(result['model_execution'])
            self.assertNotIn('bookkeeping_recovery', result)


if __name__ == '__main__':
    unittest.main()
