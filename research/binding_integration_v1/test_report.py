"""Model-free statistical and fail-closed reporting tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.binding_integration_v1 import report as r


def fixture_binding():
    inputs, gold, bundles = [], {}, {role: {'predictions': []} for role in ('source', 'lora', 'mixed', 'integrated')}
    for world in range(32):
        parent = f'world-{world:02d}'
        for category in r.bind_score.CATEGORIES:
            reference = 'UNKNOWN' if category in r.bind_score.UNKNOWN_CATEGORIES else 'A Person' if category == 'person_answerable' else 'steward'
            for view in range(3):
                key = f'{parent}/{category}/{view}'
                inputs.append({'id': key, 'parent_id': parent, 'prompt': [{'role': 'system', 'content': 'fixture'}, {'role': 'user', 'content': reference}]})
                gold[key] = {'id': key, 'parent_id': parent, 'category': category, 'reference': reference}
                for role in bundles:
                    wrong = category in r.bind_score.UNKNOWN_CATEGORIES and role in ('source', 'lora')
                    bundles[role]['predictions'].append({'id': key, 'parent_id': parent,
                        'prediction': 'A Person' if wrong else reference, 'terminated': True})
    def decode(value):
        return {'world': {'people': ['A Person', 'B Person', 'C Person', 'D Person', 'E Person', 'F Person']}, 'answer': value}
    endpoints, maps, roster = {}, {}, {}
    for role, bundle in bundles.items():
        endpoints[role], maps[role], roster[role] = r.bind_score.score_rows(inputs, gold, bundle['predictions'], decode)
    baseline = {'endpoints': {role: endpoints[role] for role in ('source', 'lora')},
                'world_maps': {role: maps[role] for role in ('source', 'lora')},
                'roster_person_world_any_maps': {role: roster[role] for role in ('source', 'lora')}, 'normalization': 'fixture'}
    mixed = {'binding': {'endpoints': {'mixed': endpoints['mixed']}, 'world_maps': {'mixed': maps['mixed']}}}
    return inputs, gold, bundles, baseline, mixed, decode


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.criteria = r.read(r.ROOT / 'CRITERIA.json')

    def test_registered_thresholds_and_report_only_diagnostic(self):
        r.validate_criteria(self.criteria)
        altered = copy.deepcopy(self.criteria)
        altered['primary_engineering']['fixed47_retained_at_least'] = 39
        with self.assertRaises(ValueError):
            r.validate_criteria(altered)

    def test_external_code_snapshot_covers_inherited_readers_without_weights(self):
        pins = r.external_code_snapshot()
        for module in (r.prior, r.paired, r.mixed_report.bc, r.control_report.run_probe, r.diagnostic_score):
            self.assertEqual(pins[str(Path(module.__file__).resolve())], r.sha(module.__file__))
        self.assertTrue(all(Path(path).suffix == '.py' for path in pins))

    def test_admission_binds_frozen_statistical_helper_code(self):
        path = r.ROOT / 'DATA_USE_ADMISSION.json'
        gate = {'data_use_admission_path': str(path), 'data_use_admission_sha256': r.sha(path)}
        _, pins = r.verify_admission(r.ROOT.resolve(), gate)
        helper = Path(r.paired.__file__).resolve()
        self.assertIn(str(helper), pins)
        actual = r.sha
        with patch.object(r, 'sha', side_effect=lambda value: '0' * 64 if Path(value).resolve() == helper else actual(value)):
            with self.assertRaisesRegex(ValueError, 'reference/scoring helper'):
                r.verify_admission(r.ROOT.resolve(), gate)
        altered = copy.deepcopy(self.criteria); altered['binding_efficacy']['comparator'] = 'mixed'
        with self.assertRaises(ValueError):
            r.validate_criteria(altered)

    def test_acquisition_has_separate_holm_families_and_twenty_floor(self):
        maps = {role: {bank: {str(i): (i < 19 if role == 'integrated' else False) for i in range(384)}
                      for bank in ('dose', 'select')} for role in ('mixed', 'lora', 'integrated')}
        contrasts = r.acquisition_contrasts(maps, self.criteria)
        for family in contrasts.values():
            for value in family.values():
                self.assertEqual(value['holm_family_size'], 2)
                self.assertEqual((value['net'], value['nd']), (19, 19))
                self.assertLess(value['holm_adjusted_p'], .05)
                self.assertFalse(value['structural_claim_eligible'])
                self.assertFalse(value['cross_training_draw_floor_cleared'])
        for bank in ('dose', 'select'):
            maps['integrated'][bank]['19'] = True
        contrasts = r.acquisition_contrasts(maps, self.criteria)
        self.assertTrue(contrasts['lora']['dose']['structural_claim_eligible'])

    def test_qualified_acquisition_recounts_without_reading_weights(self):
        loaded, frozen = {}, {}
        for role, root in (('mixed', r.MIXTURE), ('lora', r.CONTROL)):
            frozen[role] = r.read(root / 'RESULTS.json'); loaded[role] = {}
            for bank in ('dose', 'select'):
                directory = Path(frozen[role]['custody'][role][bank]['directory'])
                raw = r.read(directory / 'result.json'); binding = r.read(directory / 'PROBE_WEIGHT_BINDING.json')
                loaded[role][bank] = raw, binding, {row['id'] for row in raw['predictions']}
        maps = r.acquisition_maps(loaded)
        for role in maps:
            for bank in maps[role]:
                self.assertEqual(sum(maps[role][bank].values()), frozen[role]['endpoints'][role][bank]['correct'])

    def test_binding_primary_is_32_worlds_vs_lora_not_96_rows(self):
        result = r.score_binding(*fixture_binding())
        value = result['contrasts_integrated_minus']['lora']['canonical_complete_exact']['person_missing']
        self.assertEqual((value['n'], value['net'], value['nd']), (32, 32, 32))
        self.assertEqual(value['exact_p'], 2 ** -31)
        self.assertEqual(value['holm_family_size'], 2)
        self.assertNotIn('holm_adjusted_p', result['contrasts_integrated_minus']['mixed']['canonical_complete_exact']['person_missing'])
        self.assertNotIn('cross_training_draw_abs_net_floor', value)

    def candidate(self, retained=40):
        return {'dose': {'correct': 300}, 'select': {'correct': 300},
                'retention': {'policies': {'permissive': {'references_retained': retained, 'panel_correct': 76}}},
                'long_context': {'correct': 170, 'literal_unknown': 30}}

    def test_false_unknown_band_cannot_be_rescued_by_unknown_gains(self):
        binding = r.score_binding(*fixture_binding())
        binding['endpoints']['integrated']['totals']['false_unknown_on_positive'] = 11
        acq = {role: {bank: {'net': 0} for bank in ('dose', 'select')} for role in ('mixed', 'lora')}
        value = r.prediction_verdicts(self.candidate(), binding, acq, self.criteria)
        self.assertEqual(value['primary_engineering_conjunction'], 'FAILED_PREDICTION')
        self.assertEqual(value['binding_efficacy'], 'NOT_ESTABLISHED')

    def test_retention_broad_band_does_not_replace_primary_minimum(self):
        binding = r.score_binding(*fixture_binding())
        acq = {role: {bank: {'net': 0} for bank in ('dose', 'select')} for role in ('mixed', 'lora')}
        value = r.prediction_verdicts(self.candidate(retained=39), binding, acq, self.criteria)
        self.assertEqual(value['bands']['references_retained']['status'], 'PASS')
        self.assertEqual(value['primary_engineering_conjunction'], 'FAILED_PREDICTION')
        self.assertEqual(value['binding_efficacy'], 'ESTABLISHED_EXPLORATORY_DEV_GAIN')
        self.assertFalse(value['controlled_diagnostic_used_for_selection_or_confirmation'])
        self.assertFalse(value['acquisition_cost_engineering']['lora']['dose']['establishes_preservation'])

    def test_sixth_leg_must_be_present(self):
        with self.assertRaises(ValueError):
            r.require_six_legs({leg: {} for leg in r.prior.LEGS}, {'binding': {}})
        r.require_six_legs({leg: {} for leg in r.prior.LEGS}, {'binding': {}, 'diagnostic': {}})

    def test_controlled_cohort_stays_four_and_has_no_primary_or_selected_pvalues(self):
        b = r.diagnostic_score.binding.build_rows(31); a = r.diagnostic_score.arithmetic.build_rows(31)
        inputs = [{**row, 'prompt_ids': [1]} for row in b['policy_inputs'] + a['policy_inputs']]
        targets = b['gold'] + [{**row, 'section': 'arithmetic'} for row in a['gold']]
        gold = {row['id']: row for row in targets}
        anchors = [row['id'] for row in targets if row['section'] == 'arithmetic' and
                   row['numeric_variant'] == 0 and row['response_type'] == 'unit_explicit']
        typed = r.diagnostic_score.load_typed_scorer()
        bundles, endpoints, maps = {}, {}, {}
        for role in ('source', 'lora', 'mixed', 'integrated'):
            pred = [{'id': row['id'], 'parent_id': row['parent_id'], 'prediction': row['reference'], 'terminated': True} for row in targets]
            for row in pred:
                if role != 'integrated' and row['id'] in anchors[4:]:
                    row['prediction'] = 'incorrect'
            bundles[role] = {'predictions': pred}
            endpoints[role], maps[role] = r.diagnostic_score.score_rows(inputs, gold, pred, typed)
        cohort = sorted(parent for parent, value in maps['source']['arithmetic']['strict_anchor'].items() if value)
        frozen = {'endpoints': {role: endpoints[role] for role in ('source', 'lora', 'mixed')},
                  'parent_maps': {role: maps[role] for role in ('source', 'lora', 'mixed')},
                  'source_conditioned_retention_descriptive': {'parent_ids': cohort, 'n': 4}}
        with patch.object(r.diagnostic_score, 'analyze', side_effect=AssertionError('old primary analysis must not run')):
            result = r.score_controlled(inputs, gold, bundles, frozen)
        selected = result['source_conditioned_retention_descriptive']
        self.assertEqual(selected['n'], 4)
        self.assertEqual(selected['status'], 'INSUFFICIENT_REFERENCE_COHORT')
        self.assertFalse(selected['inferential_tests_performed'])
        self.assertEqual(selected['models']['integrated']['retained'], 4)
        self.assertNotIn('primary', result)
        self.assertNotIn('exact_p', selected['models']['integrated'])
        predictions = r.controlled_predictions(result, self.criteria)
        self.assertEqual(predictions['bands']['all16_binding_parents']['status'], 'FAILED_PREDICTION')
        self.assertFalse(predictions['primary_endpoint_substitution'])

    def test_auxiliary_token_text_and_eos_validation(self):
        inputs = [{'id': 'x', 'parent_id': 'p', 'prompt_ids': [5, 6]}]
        generation = {'rows': 1, 'eos_ids': [1], 'numerics': r.mixed_report.bc.baseline.NUMERICS, 'gold_file_read': False,
            'predictions': [{'id': 'x', 'parent_id': 'p', 'prediction': 'x', 'terminated': True,
                'input_tokens': 2, 'generated_tokens': 2, 'forward_calls': 2, 'processed_input_tokens': 3,
                'generated_ids_including_eos': [0, 1]}],
            'actual_budget_charges': {'rows': 1, 'generated_tokens_including_eos': 2, 'forward_calls': 2, 'processed_input_tokens': 3}}
        class Decoder:
            def decode(self, ids):
                return 'x'
        r.verify_generation_tokens(generation, inputs, Decoder(), [1])
        changed = copy.deepcopy(generation); changed['predictions'][0]['prediction'] = 'forged'
        with self.assertRaises(ValueError):
            r.verify_generation_tokens(changed, inputs, Decoder(), [1])
        changed = copy.deepcopy(generation); changed['predictions'][0]['generated_ids_including_eos'] = [1, 0]
        with self.assertRaises(ValueError):
            r.verify_generation_tokens(changed, inputs, Decoder(), [1])

    def test_global_six_probe_costs_and_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); (root / 'posttrain').mkdir()
            records, bindings = [], {}
            for leg, (n, maximum) in r.RESERVATIONS.items():
                key = 'integrated-' + leg; out = root / 'eval' / key
                supervisor = out / ('supervisor.json' if leg in ('binding', 'diagnostic') else 'worker/supervisor.json')
                supervisor.parent.mkdir(parents=True)
                r.run_probe.write(supervisor, {'status': 'EXITED', 'returncode': 0, 'wall_seconds': 1.25})
                record = {'leg': key, 'out': str(out), 'returncode': 0, 'worker_seconds_charged': 1.25,
                          'reserved_rows': n, 'reserved_generation_tokens': n * maximum}
                r.run_probe.write(root / 'posttrain' / (key + '.json'), record); records.append(record)
                name = 'AUX_BINDING.json' if leg in ('binding', 'diagnostic') else 'PROBE_WEIGHT_BINDING.json'
                r.run_probe.write(out / name, {'status': 'COMPLETE', 'complete': True}); bindings[key] = r.sha(out / name)
            envelope = {'probe_worker_seconds_all_attempts': 6300, 'probe_rows_reserved_all_attempts': 3500,
                        'probe_generation_tokens_reserved_all_attempts': 200000, 'probe_workers_all_attempts': 7, 'probe_retries_total': 1}
            costs = r.probe_cost_accounting(root, envelope)
            self.assertEqual(costs['probe_rows_reserved'], 2454)
            self.assertEqual(costs['probe_generation_tokens_reserved'], 138768)
            self.assertEqual(costs['probe_worker_seconds'], 7.5)
            r.run_probe.write(root / 'posttrain/TERMINAL.json', {'status': 'COMPLETE_SIX_PROBES', 'attempts': records,
                'final_binding_sha256': bindings, 'cost_accounting': costs})
            self.assertEqual(r.completion_receipt(root, envelope)['cost_accounting'], costs)
            r.run_probe.write(root / 'ATTEMPT_MAP.json', {'replacements': {}})
            with self.assertRaises(ValueError):
                r.completion_receipt(root, envelope)

    def test_fail_closed_output_contains_no_partial_endpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            result = r.run(Path(folder))
            self.assertEqual(result['status'], 'BLOCKED_ON_COMPARABILITY')
            self.assertFalse(result['scientific_endpoints_emitted'])
            for key in ('endpoints', 'binding', 'controlled_diagnostic', 'predictions'):
                self.assertNotIn(key, result)
            self.assertEqual({path.name for path in Path(folder).iterdir()}, {'RESULTS.json', 'RESULTS.md'})
            with self.assertRaises(ValueError):
                r.run(Path(folder))

    def test_complete_markdown_renders_all_primary_and_report_only_tables(self):
        binding = r.score_binding(*fixture_binding())
        frozen = {role: r.read(root / 'RESULTS.json') for role, root in (('mixed', r.MIXTURE), ('lora', r.CONTROL))}
        endpoints = {role: frozen[role]['endpoints'][role] for role in frozen}
        endpoints['integrated'] = copy.deepcopy(endpoints['mixed'])
        maps = {role: {bank: {str(i): (i < 20 if role == 'integrated' else False) for i in range(384)}
                      for bank in ('dose', 'select')} for role in ('mixed', 'lora', 'integrated')}
        acquisition = r.acquisition_contrasts(maps, self.criteria)
        controlled = {'endpoints': {role: {'parent_counts': {'binding': {'canonical_complete_exact/all16': 0},
                          'arithmetic': {'strict_anchor': 4}}, 'totals': {'binding': {'false_exact_unknown': 52}}}
                          for role in ('source', 'lora', 'mixed', 'integrated')},
                      'source_conditioned_retention_descriptive': {'models': {role: {'retained': 0}
                          for role in ('source', 'lora', 'mixed', 'integrated')}}}
        controlled['prospective_predictions'] = r.controlled_predictions(controlled, self.criteria)
        result = {'status': 'COMPLETE_COMPARABLE', 'endpoints': endpoints, 'binding': binding,
                  'acquisition_contrasts_integrated_minus': acquisition, 'controlled_diagnostic': controlled,
                  'predictions': r.prediction_verdicts(endpoints['integrated'], binding, acquisition, self.criteria),
                  'limitations': ['Fixture limitation.']}
        text = r.markdown(result)
        self.assertIn('FAILED_PREDICTION', text)
        self.assertIn('N=4 / INSUFFICIENT_REFERENCE_COHORT', text)
        self.assertIn('Controlled report-only prediction', text)
        self.assertIn('Exact threshold at nd', text)


if __name__ == '__main__':
    unittest.main()
