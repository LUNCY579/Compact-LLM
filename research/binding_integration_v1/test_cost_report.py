"""Cost receipt fixtures only; never inspect a live run or genuine weights."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from research.binding_integration_v1 import cost_report as c


def write(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in value) if lines else json.dumps(value, sort_keys=True) + '\n'
    path.write_text(text, encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def envelope():
    return {'native_limits': {'total_owned_work_wall_seconds': 2700,
        'model_worker_wall_seconds_and_conservative_gpu_seconds': 2500,
        'processed_training_tokens_all_attempts': 600000, 'evaluation_and_qualification_model_calls': 128,
        'generated_tokens_all_candidates': 0, 'on_policy_rl_rollout_tokens': 0,
        'new_disk_bytes': 256 * 2**20, 'adaptive_model_selection_rounds': 1,
        'model_worker_launches': 1, 'retries': 0, 'cloud_or_teacher_spend': 0},
        'expected_native_charged_tokens': 590245, 'expected_native_qualification_calls': 76,
        'probe_workers_all_attempts': 7, 'probe_worker_seconds_all_attempts': 6300,
        'probe_rows_reserved_all_attempts': 3500, 'probe_generation_tokens_reserved_all_attempts': 200000,
        'probe_retries_total': 1, 'total_owned_work_wall_seconds': 12600,
        'cpu_export_seconds': 600, 'compression_seconds_each': 600}


def launch(directory, supervisor, timeout):
    record = {'out': str(directory), 'timeout_seconds': timeout, 'supervisor_pid': 11}
    write(directory / 'launch.json', record)
    write(directory / 'spawn.json', {'launch_sha256': c.digest(record), 'supervisor_pid': 11, 'child_pid': supervisor['child_pid']})
    write(directory / 'worker_claim.json', {'launch_sha256': c.digest(record), 'supervisor_pid': 11, 'pid': supervisor['child_pid']})


def native_fixture(root):
    env = envelope(); grant = {'experiment': 'BINDING-INTEGRATION-001', 'limits': env['native_limits']}
    write(root / 'GRANT.json', grant); write(root / 'journal/grant.json', grant)
    run = root / 'journal/runs/train-001'
    supervisor = {'status': 'EXITED', 'returncode': 0, 'wall_seconds': 100.25, 'child_pid': 22}
    write(run / 'supervisor.json', supervisor); launch(run, supervisor, 2400)
    plan = {'native_processed_cumulative': [380921 * i // 193 for i in range(1, 194)],
            'applied_native_learning_rates': [0.0001] * 193}
    write(root / 'PLAN.json', plan)
    resource = {'available_host_bytes': 5 * 2**30, 'rss_bytes': 100, 'gpu0': None}
    steps = [{'updates': i, 'processed_tokens': 577417 * i // 193, 'supervised_tokens': 39502 * i // 193,
              'native_clock_tokens': plan['native_processed_cumulative'][i-1], 'lr': .0001, 'resources': resource} for i in range(1, 194)]
    write(run / 'steps.jsonl', steps, lines=True)
    binding = {'grant_sha256': c.digest(grant), 'plan_sha256': c.digest(plan), 'runtime': {'fixture': True}}
    write(run / 'binding.json', {'binding': binding, 'resources_before_load': resource})
    write(run / 'native_base_qualification.json', {'status': 'PASS', 'optimizer_steps': 0, 'gradient_probes': [{'probe': 'native_chat', 'tokens': 31}]})
    for name, tokens, calls in [('adapter_qualification.json', 2203, 26), ('merge_qualification.json', 4390, 6)]:
        write(run / name, {'status': 'PASS', 'budget_charges': {'processed_training_tokens_all_attempts': tokens, 'evaluation_and_qualification_model_calls': calls}})
    zero = {'record': 'BINDING_INTEGRATION_ZERO_ADDEND_QUALIFICATION', 'status': 'PASS', 'complete': True,
            'scope': 'EXACT_TWO_UPDATE_PREFIX_ONLY', 'optimizer_step_count': 2, 'native_row_ids': list(range(28)),
            'binding_forward_calls': 0, 'generation_calls': 0, 'qualification_optimizer_discarded': True,
            'trainable_state_restored_exactly': True, 'rng_restored_exactly': True,
            'restored_trainable_sha256': 'a' * 64, 'rng_before_sha256': 'b' * 64, 'rng_after_restore_sha256': 'b' * 64,
            'budget_charges': {'processed_training_tokens_all_attempts': 4032, 'evaluation_and_qualification_model_calls': 28}}
    write(run / 'zero_addend_qualification.json', zero)
    write(run / 'result.json', {'record': 'BINDING_INTEGRATION_TRAINING_RESULT', 'status': 'COMPLETE', 'complete': True,
        'progress': {'updates': 193, 'next_group': 193, 'processed_tokens': 577417, 'supervised_tokens': 39502},
        'native_clock_tokens': 380921, 'binding': binding, 'steps_sha256': c.sha(run / 'steps.jsonl'),
        'zero_addend_qualification_path': str(run / 'zero_addend_qualification.json'),
        'zero_addend_qualification_sha256': c.sha(run / 'zero_addend_qualification.json'),
        'active_optimizer_started_fresh': True, 'active_initial_trainable_sha256': 'a' * 64, 'active_initial_rng_sha256': 'b' * 64})
    event_bodies = [
        {'kind': 'start', 'run_id': 'train-001', 'phase': 'train', 'charges': {'model_worker_launches': 1,
            'adaptive_model_selection_rounds': 1, 'new_disk_bytes': 4 * 2**20,
            c.WALL_KEYS[0]: 2420, c.WALL_KEYS[1]: 2540}},
        {'kind': 'charge', 'charges': {'processed_training_tokens_all_attempts': 590245,
            'evaluation_and_qualification_model_calls': 76, 'new_disk_bytes': 2 * (7255040 * 12 + 2**20)}},
        {'kind': 'settle', 'run_id': 'train-001', 'charges': {}, 'receipt_sha256': c.sha(run / 'supervisor.json'),
            'actual_wall': {c.WALL_KEYS[0]: 101, c.WALL_KEYS[1]: 102}},
    ]
    previous = None; paths = []
    for index, body in enumerate(event_bodies):
        body.update(index=index, previous=previous); previous = c.digest(body)
        path = root / 'journal/events' / f'{index:06d}.json'; write(path, {**body, 'sha256': previous}); paths.append(path)
    state = c.replay_events(paths, read, env['native_limits'])[0]
    write(run / 'run_summary.json', {'experiment': 'BINDING-INTEGRATION-001', 'run_id': 'train-001',
        'returncode': 0, 'worker_failed': False, 'supervisor': supervisor, 'budget_after': state})
    return env, run


def probe_fixture(root, *, complete=True, failure=False, unobserved=False):
    env = envelope(); write(root / 'SESSION_ENVELOPE.json', env)
    records, bindings, cost_pins, details = [], {}, {}, []
    choices = c.LEGS.items() if complete else [('diagnostic', c.LEGS['diagnostic'])]
    for suffix, (n, maximum) in sorted(choices):
        key = 'integrated-' + suffix; out = root / 'eval' / key; out.mkdir(parents=True)
        reserve = {'suffix': suffix, 'attempt': 1, 'out': str(out), 'reserved_rows': n,
            'reserved_generation_tokens': n * maximum, 'reserved_worker_seconds': 900,
            'session_envelope_sha256': c.sha(root / 'SESSION_ENVELOPE.json')}
        reservation = root / 'posttrain/reservations' / (key + '.json'); write(reservation, reserve)
        if unobserved:
            continue
        auxiliary = suffix in ('binding', 'diagnostic'); supervisor_path = out / ('supervisor.json' if auxiliary else 'worker/supervisor.json')
        code = 1 if failure else 0
        supervisor = {'status': 'EXITED', 'returncode': code, 'wall_seconds': 1.5, 'child_pid': 22}
        write(supervisor_path, supervisor)
        record = {'leg': key, 'out': str(out), 'returncode': code, 'wall_seconds': 2.,
            'worker_seconds_charged': 1.5, 'reserved_rows': n, 'reserved_generation_tokens': n * maximum}
        receipt = root / 'posttrain' / (key + '.json'); write(receipt, record); records.append(record)
        cost_pins[receipt.relative_to(root).as_posix()] = c.sha(receipt); cost_pins[supervisor_path.relative_to(root).as_posix()] = c.sha(supervisor_path)
        details.append({k:v for k,v in record.items() if k != 'wall_seconds'})
        if failure:
            continue
        launch(out if auxiliary else out / 'worker', supervisor, 900)
        terminal_path = out / ('worker_terminal.json' if auxiliary else 'worker/probe_terminal.json')
        write(terminal_path, {'status': 'COMPLETE' if auxiliary else 'COMPLETED', 'complete': True, 'failure': None, 'system_exit_code': 0})
        bound = {'status': 'COMPLETE', 'complete': True, 'controller_returncode' if auxiliary else 'returncode': 0,
                 'supervisor_sha256': c.sha(supervisor_path), 'worker_terminal_sha256' if auxiliary else 'terminal_sha256': c.sha(terminal_path)}
        if auxiliary:
            local = root / 'eval' / ('AUX_RESERVATION_' + key + '.json')
            write(local, {'parent_reservation_sha256': c.sha(reservation), 'rows': n, 'reserved_generated_tokens': n * maximum})
            bound.update(worker_returncode=0, reservation_sha256=c.sha(local))
        binding_path = out / ('AUX_BINDING.json' if auxiliary else 'PROBE_WEIGHT_BINDING.json'); write(binding_path, bound); bindings[key] = c.sha(binding_path)
    if complete:
        accounting = {'probe_worker_seconds': 9., 'probe_rows_reserved': 2454, 'probe_generation_tokens_reserved': 138768,
            'attempt_count': 6, 'successful_attempts': 6, 'failed_attempts': 0, 'individual_attempts': details, 'cost_source_sha256': cost_pins}
        write(root / 'posttrain/TERMINAL.json', {'status': 'COMPLETE_SIX_PROBES', 'attempts': records,
            'final_binding_sha256': bindings, 'cost_accounting': accounting, 'session_elapsed_seconds': 100.,
            'probe_worker_seconds': 9., 'probe_rows_reserved': 2454, 'probe_generation_tokens_reserved': 138768})
    return env


class CostReportTests(unittest.TestCase):
    def test_native_193_plus_discarded_two_update_qualification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); env, _ = native_fixture(root)
            value = c.native_costs(root, env, read, jsonl)
            self.assertEqual((value['active_optimizer_updates'], value['qualification_optimizer_updates']), (193, 2))
            self.assertEqual(value['all_optimizer_data_tokens'], 581449)
            self.assertEqual(sum(value['qualification_token_charges'].values()), 12828)
            self.assertEqual(sum(value['qualification_call_charges'].values()), 76)
            self.assertEqual(value['journal_state']['used_or_reserved']['processed_training_tokens_all_attempts'], 590245)

    def test_warm_start_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); env, run = native_fixture(root)
            value = read(run / 'result.json'); value['active_optimizer_started_fresh'] = False; write(run / 'result.json', value)
            with self.assertRaisesRegex(ValueError, 'fresh active optimizer'):
                c.native_costs(root, env, read, jsonl)

    def test_unsettled_native_launch_retains_full_wall_reservations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); env, _ = native_fixture(root)
            paths = sorted((root / 'journal/events').glob('*.json'))[:2]
            value, _, _ = c.replay_events(paths, read, env['native_limits'])
            self.assertEqual(value['active_reservations'], ['train-001'])
            self.assertEqual(value['used_or_reserved'][c.WALL_KEYS[0]], 2420)
            self.assertEqual(value['used_or_reserved']['processed_training_tokens_all_attempts'], 590245)

    def test_journal_tampering_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); env, _ = native_fixture(root)
            path = root / 'journal/events/000001.json'; value = read(path); value['charges']['processed_training_tokens_all_attempts'] -= 1; write(path, value)
            with self.assertRaisesRegex(ValueError, 'seal'):
                c.replay_events(sorted((root / 'journal/events').glob('*.json')), read, env['native_limits'])

    def test_six_successful_probe_costs_match_exact_terminal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); env = probe_fixture(root)
            value = c.probe_costs(root, env, read)
            self.assertEqual(value['totals']['attempts'], 6)
            self.assertEqual(value['totals']['rows_reserved'], 2454)
            self.assertEqual(value['totals']['generation_tokens_reserved'], 138768)
            self.assertIsNone(value['actual_generated_tokens'])

    def test_terminal_cannot_omit_binding_hash_or_costs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); env = probe_fixture(root)
            path = root / 'posttrain/TERMINAL.json'; value = read(path); value['cost_accounting']['probe_rows_reserved'] -= 1; write(path, value)
            with self.assertRaisesRegex(ValueError, 'cost_accounting'):
                c.probe_costs(root, env, read)

    def test_unobserved_probe_keeps_full900_and_token_reserve(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); env = probe_fixture(root, complete=False, unobserved=True); partial = {}
            with self.assertRaisesRegex(ValueError, 'parent result missing'):
                c.probe_costs(root, env, read, partial)
            total = partial['observed_or_reserved_totals']
            self.assertEqual((total['worker_seconds_charged'], total['rows_reserved'], total['generation_tokens_reserved']), (900., 896, 21504))

    def test_failed_probe_cost_is_not_zeroed_by_completion_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); env = probe_fixture(root, complete=False, failure=True); partial = {}
            with self.assertRaises(ValueError):
                c.probe_costs(root, env, read, partial)
            self.assertEqual(partial['observed_or_reserved_totals']['failed_attempts'], 1)
            self.assertEqual(partial['observed_or_reserved_totals']['worker_seconds_charged'], 1.5)
            self.assertEqual(partial['observed_or_reserved_totals']['generation_tokens_reserved'], 21504)

    def test_unobserved_export_keeps_full600_reservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); (root / 'posttrain/build-integrated').mkdir(parents=True); partial = {}
            with self.assertRaises(OSError):
                c.export_costs(root, envelope(), read, partial)
            self.assertEqual(partial['export_worker_seconds_charged'], 600)
            self.assertTrue(partial['missing_terminal_is_not_zero_cost'])

    def test_settled_cpu_export_and_compression_use_metadata_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); stage = root / 'posttrain/build-integrated'
            supervisor = {'status': 'EXITED', 'returncode': 0, 'wall_seconds': 4., 'child_pid': 22}
            write(stage / 'supervisor.json', supervisor); launch(stage, supervisor, 600)
            write(stage / 'CPU_TERMINAL.json', {'status': 'COMPLETE', 'returncode': 0, 'failure': None,
                'model_execution': False, 'wall_seconds': 3.})
            write(stage / 'COMPRESSION.json', {'returncode': 0, 'model_execution': False,
                'before_sha256': 'a' * 64, 'after_sha256': 'a' * 64, 'logical_bytes': 4,
                'wall_seconds': 2., 'free_disk_bytes': 100000})
            arm = root / 'arms/integrated'; arm.mkdir(parents=True)
            # A four-byte synthetic placeholder proves only a stat is required.
            # c.sha explicitly refuses this filename, so no real weight is used.
            (arm / 'model.safetensors').write_bytes(b'test')
            write(arm / 'ARM.json', {'arm_weight_sha256': 'a' * 64, 'available_host_bytes_before': 200000})
            write(arm / 'MATERIALIZATION.json', {'out_weight_sha256': 'a' * 64})
            value = c.export_costs(root, envelope(), read)
            self.assertEqual((value['export_worker_wall_seconds'], value['export_supervisor_wall_seconds'], value['compression_wall_seconds']), (3., 4., 2.))
            self.assertFalse(value['weight_contents_read'])

    def test_existing_file_compression_is_not_new_artifact_spend(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            values = [('STORAGE_QUALIFICATION.json', 'PASS_LOSSLESS_CHECKPOINT_COMPRESSION', 'checkpoint_contents_changed'),
                      ('SIDECAR_STORAGE_QUALIFICATION.json', 'PASS_LOSSLESS_OWNED_SIDECAR_COMPRESSION', 'contents_changed')]
            for name, status, key in values:
                data = {'schema': 1, 'status': status, 'model_calls': 0, key: False, 'evidence_deleted': False,
                        'file_sha256_before_after': {'C:/old/file': 'a' * 64}, 'files': 1, 'logical_bytes': 1000,
                        'wall_seconds': 2., 'volume_free_bytes_before': 10000, 'volume_free_bytes_after': 10100,
                        'runs': [{'returncode': 0}] * 4, 'returncode': 0}
                write(root / name, data)
            value = c.storage_preparation_costs(root, envelope(), read)
            self.assertEqual(value['logical_existing_bytes_processed'], 2000)
            self.assertEqual(value['observed_volume_free_delta_bytes'], 200)
            self.assertEqual(value['new_artifact_logical_bytes_charge'], 0)
            self.assertTrue(all(v['attributable_file_allocation_savings_bytes'] is None for v in value['stages']))

    def test_cost_has_no_weight_or_generation_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('model.safetensors', 'GENERATIONS.json'):
                path = root / name; path.write_bytes(b'fixture')
                with self.assertRaises(ValueError):
                    c.sha(path)


if __name__ == '__main__':
    unittest.main()
