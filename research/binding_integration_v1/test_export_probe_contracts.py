"""Model-free integration export/probe contract tests; never read model weights."""
import copy
import json
import math
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from research.binding_integration_v1 import materialize as m, run_probe as p

RECORD = p.BASE / 'rineya-dataset-retention-records/binding-integration001-20260920'


def zero_fixture():
    shapes = p.qualified_trainable_shapes()
    files = {f'tensor-{i:04d}.safetensors': 'a' * 64 for i in range(360)}
    checks = []
    for (name, shape), filename in zip(sorted(shapes.items()), files):
        checks.append({'name': name, 'shape': shape, 'reference_file': filename, 'reference_file_sha256': files[filename],
                       **{key: {'equal': True, 'actual_sha256': 'b' * 64, 'expected_sha256': 'b' * 64}
                          for key in ('parameter', 'exp_avg', 'exp_avg_sq')}})
    plan = {'zero_addend_reference': {'checkpoint_path': str(p.NATIVE_CONTROL / 'journal/runs/train-001/checkpoint-00002'),
                                     'manifest_sha256': 'c' * 64, 'tensor_files_sha256': files}}
    record = {'schema': 1, 'record': 'BINDING_INTEGRATION_ZERO_ADDEND_QUALIFICATION', 'experiment': p.EXPERIMENT,
              'status': 'PASS', 'complete': True, 'scope': 'EXACT_TWO_UPDATE_PREFIX_ONLY', 'optimizer_updates': 2,
              'optimizer_step_count': 2, 'binding_forward_calls': 0, 'checked_trainable_tensors': 360,
              'checked_component_tensors': 1080, 'reference_manifest_sha256': 'c' * 64,
              'reference_tensor_files_sha256': files, 'checks': checks,
              'budget_charges': {'processed_training_tokens_all_attempts': 4032, 'evaluation_and_qualification_model_calls': 28},
              'failure': None, 'full_193_update_trajectory_equality_measured': False, 'generation_calls': 0,
              **{key: 'd' * 64 for key in ('initial_trainable_sha256', 'restored_trainable_sha256',
                   'rng_before_sha256', 'rng_after_restore_sha256', 'reference_rng_sha256', 'actual_prefix_rng_sha256',
                   'frozen_base_sha256_before', 'frozen_base_sha256_after')},
              **{key: True for key in ('all_trainable_parameters_bitwise_equal', 'all_first_moments_bitwise_equal',
                   'all_second_moments_bitwise_equal', 'trainable_state_restored_exactly', 'rng_restored_exactly',
                   'reference_rng_bitwise_equal', 'frozen_base_unchanged', 'qualification_optimizer_discarded',
                   'initial_model_training_mode_restored')}}
    return record, plan


class ExportProbeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared = RECORD / 'prepared'
        cls.rows = [json.loads(line) for line in (prepared / 'microfit-train.jsonl').read_text().splitlines()]
        cls.membership = json.loads((prepared / 'SOURCE_MEMBERSHIP.json').read_text())
        cls.groups = p.read(prepared / 'INTEGRATION_GROUPS.json')
        cls.plan = {'experiment': p.EXPERIMENT, 'exposures_per_view': 1, 'presentations': 3465,
                    'unique_worlds': 2761, 'processed_tokens': 577417, 'supervised_tokens': 39502,
                    'row_order_per_exposure': [row['id'] for row in cls.rows],
                    'update_groups': cls.groups['combined_update_groups'],
                    **{key: cls.groups[key] for key in ('native_update_groups', 'binding_update_groups',
                        'native_denominators', 'native_processed_cumulative', 'applied_native_learning_rates')}}

    def test_registered_accumulation_and_native_clock(self):
        result = p.validate_integration_groups(self.plan, self.rows, self.membership, self.groups)
        self.assertEqual(result['updates'], 193)
        self.assertEqual(result['native_clock_tokens'], 380921)
        self.assertEqual(result['native_supervised_tokens'], 37090)
        self.assertEqual(result['binding_supervised_tokens'], 2412)
        self.assertEqual(result['maximum_combined_update_tokens'], 3114)
        self.assertLess(max(len(row['ids']) for row in self.rows), 2048)

    def test_combined_denominator_cannot_replace_native_denominator(self):
        plan, groups = copy.deepcopy(self.plan), copy.deepcopy(self.groups)
        plan['native_denominators'][0] += groups['binding_denominators'][0]
        groups['native_denominators'][0] = plan['native_denominators'][0]
        with self.assertRaisesRegex(ValueError, 'native denominator'):
            p.validate_integration_groups(plan, self.rows, self.membership, groups)

    def test_binding_cannot_change_native_lr_clock(self):
        plan, groups = copy.deepcopy(self.plan), copy.deepcopy(self.groups)
        plan['native_processed_cumulative'][0] += groups['binding_processed_per_update'][0]
        groups['native_processed_cumulative'][0] = plan['native_processed_cumulative'][0]
        with self.assertRaisesRegex(ValueError, 'native LR clock'):
            p.validate_integration_groups(plan, self.rows, self.membership, groups)

    def test_duplicate_or_reordered_binding_view_is_rejected(self):
        plan, groups = copy.deepcopy(self.plan), copy.deepcopy(self.groups)
        plan['binding_update_groups'][0][0], plan['binding_update_groups'][1][0] = (
            plan['binding_update_groups'][1][0], plan['binding_update_groups'][0][0])
        groups['binding_update_groups'] = plan['binding_update_groups']
        with self.assertRaises(ValueError):
            p.validate_integration_groups(plan, self.rows, self.membership, groups)

    def test_unchanged_lora_geometry_without_tensor_allocations(self):
        header = {}
        for layer in range(28):
            for suffix, (nin, nout) in m.PRODUCTION_SHAPES.items():
                header[f'model.layers.{layer}.{suffix}.weight'] = {'shape': [nout, nin]}
            for suffix, width in m.NORM_WIDTHS.items():
                header[f'model.layers.{layer}.{suffix}'] = {'shape': [width]}
        header['model.embed_tokens.weight'] = {'shape': [151936, 1024]}
        header['model.norm.weight'] = {'shape': [1024]}
        spec = m.expected_lora_spec(header)
        self.assertEqual(spec['trainable_parameters'], 7255040)
        self.assertEqual(spec['base_parameters'], 596049920)
        self.assertEqual(len(m.DENSE_NAMES), 220)
        self.assertEqual(set(spec['trainable_names']), set(p.qualified_trainable_shapes()))
        self.assertEqual(sum(math.prod(shape) for shape in p.qualified_trainable_shapes().values()), 7255040)

    def test_training_data_snapshot_recounts_real_admitted_bytes_without_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); plan_path = root / 'PLAN.json'
            p.write(plan_path, self.plan)
            prepared = RECORD / 'prepared'; contract = p.read(prepared / 'DATA_CONTRACT.json')
            bank = root / 'views.jsonl'
            bank.write_text(json.dumps({'id': 'fixture-eval', 'parent_id': 'fixture-eval-parent'}) + '\n', encoding='utf-8')
            assets = {'model.safetensors': 'a' * 64}
            gate = {'plan_path': str(plan_path), 'plan_file_sha256': p.sha(plan_path), 'prepared': str(prepared),
                    'training_rows': 3465, 'training_parent_count': 2761, 'data_build_sha256': p.sha(prepared / 'DATA_BUILD.json'),
                    'binding_data_admission_sha256': contract['admission_scope']['binding_admission_sha256'],
                    'source_assets_sha256': assets,
                    'banks': {'fixture': {'path': str(bank), 'sha256': p.sha(bank), 'parents': 1, 'training_parent_overlap': 0}}}
            bound = {'plan_sha256': p.native_digest(self.plan), 'data_contract_sha256': p.sha(prepared / 'DATA_CONTRACT.json'),
                     'training_files_sha256': contract['files'], 'experiment': p.EXPERIMENT,
                     'training_inventory': {'views': 3465, 'parent_worlds': 2761, 'exposures_per_view': 1},
                     'source': {'assets': assets, 'sha256': p.native_digest(assets)}}
            receipt = {'checkpoint_binding': bound, 'checkpoint_binding_sha256': p.native_digest(bound),
                       'checkpoint_progress': p.FINAL_PROGRESS, 'native_clock_tokens': 380921}
            with patch.object(p, 'zero_addend_snapshot', return_value={'native_clock_tokens': 380921}):
                result = p.training_snapshot(gate, (receipt, receipt))
            self.assertEqual(result['updates'], 193)
            self.assertEqual(result['native_clock_tokens'], 380921)
            self.assertEqual(len(result['data_files_sha256']), 4)

    def test_zero_addend_requires_complete_bitwise_optimizer_and_restore_evidence(self):
        record, plan = zero_fixture()
        p.validate_zero_addend_body(record, plan)
        for field, bad in (('scope', 'FULL_193_TRAJECTORY'), ('optimizer_updates', 193), ('binding_forward_calls', 1),
                           ('trainable_state_restored_exactly', False), ('rng_restored_exactly', False),
                           ('all_first_moments_bitwise_equal', False), ('reference_rng_bitwise_equal', False)):
            changed = copy.deepcopy(record); changed[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.validate_zero_addend_body(changed, plan)
        changed = copy.deepcopy(record)
        changed['checks'][3]['exp_avg_sq']['expected_sha256'] = 'd' * 64
        with self.assertRaises(ValueError):
            p.validate_zero_addend_body(changed, plan)
        changed = copy.deepcopy(record); changed['checks'].pop()
        with self.assertRaises(ValueError):
            p.validate_zero_addend_body(changed, plan)

    def test_old_final_progress_cannot_export(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint-00193'; checkpoint.mkdir()
            manifest = {'schema': 1, 'record': 'BINDING_INTEGRATION_TRAINING_CHECKPOINT', 'experiment': p.EXPERIMENT,
                        'status': 'COMPLETE', 'binding': {}, 'progress': {**m.FINAL_PROGRESS, 'updates': 296, 'next_group': 296},
                        'rng': {}, 'entries': [], 'adapter_spec': {}, 'zero_addend_qualification_sha256': 'a' * 64}
            m.write_new(checkpoint / 'manifest.json', m.seal(manifest))
            with self.assertRaisesRegex(ValueError, 'exact final progress'):
                m.inspect_checkpoint(Path(directory) / 'unused-source', checkpoint, 'lora')

    def test_arbitrary_arm_output_rejected_before_any_tensor_work(self):
        with patch.object(m, 'lease', side_effect=AssertionError('lease must not be entered')):
            with self.assertRaisesRegex(ValueError, 'registered integrated arm'):
                m.materialize_candidate(Path('/unused-source'), Path('/records/journal/runs/train-001/checkpoint-00193'), Path('/other-arm'))

    def test_original_readonly_relocation_qualifies(self):
        path = RECORD / 'SOURCE_RELOCATION.json'
        gate = {'source_relocation_path': str(path), 'source_relocation_sha256': p.sha(path),
                'implementation_digest': p.implementation_digest()}
        result = p.source_relocation_snapshot(gate)
        self.assertEqual(len(result['qualified_additional_input_aliases']), 3)
        self.assertFalse(result['weight_or_result_aliases_permitted'])

    def test_relocation_cannot_alias_weight_files(self):
        path = RECORD / 'SOURCE_RELOCATION.json'
        gate = {'source_relocation_path': str(path), 'source_relocation_sha256': p.sha(path),
                'implementation_digest': p.implementation_digest()}
        changed = p.read(path)
        changed['files'].append({'original_path': str(p.ORIGINAL_PILOT / 'model.safetensors'),
                                'archive_path': str(p.PILOT / 'model.safetensors'), 'role': 'additional_input', 'sha256': 'a' * 64})
        original_read, original_sha = p.read, p.sha
        def read(value):
            return changed if value.resolve() == path.resolve() else original_read(value)
        def sha(value):
            self.assertNotEqual(value.name, 'model.safetensors', 'test attempted to read model weights')
            return original_sha(value)
        with patch.object(p, 'read', side_effect=read), patch.object(p, 'sha', side_effect=sha):
            with self.assertRaisesRegex(ValueError, 'unqualified file'):
                p.source_relocation_snapshot(gate)

    def test_modules_and_contracts_are_model_free(self):
        code = ('import sys; from research.binding_integration_v1 import materialize,run_probe; '
                'assert not any(name in sys.modules for name in ("torch","transformers","safetensors"))')
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(m.FINAL_PROGRESS, p.FINAL_PROGRESS)


if __name__ == '__main__':
    unittest.main()
