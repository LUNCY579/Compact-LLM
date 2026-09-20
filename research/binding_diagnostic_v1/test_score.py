"""Model-free regression tests for scoring units, eligibility and fail-closed output."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.binding_diagnostic_v1 import arithmetic, binding, score


CRITERIA = {'primary': {'adjusted_exact_p_at_most': .05,
    'binding': {'positive_net_at_least': 8},
    'arithmetic': {'minimum_reference_n': 24, 'minimum_abs_net': 7, 'minimum_abs_net_fraction': .20,
                   'insufficient_label': 'INSUFFICIENT_REFERENCE_COHORT'}},
    'prediction_bands': {'binding_all16_source': [0, 4], 'binding_all16_lora': [0, 4],
        'binding_all16_mixed': [8, 24], 'source_strict_anchor_count': [20, 48],
        'source_conditioned_strict_retention_fraction_lora': [.50, .85],
        'source_conditioned_strict_retention_fraction_mixed': [.40, .80]}}


class ScoresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b, a = binding.build_rows(31), arithmetic.build_rows(31)
        cls.inputs = [{**row, 'prompt_ids': [1, 2]} for row in b['policy_inputs'] + a['policy_inputs']]
        cls.gold_rows = b['gold'] + [{**row, 'section': 'arithmetic'} for row in a['gold']]
        cls.gold = score.validate_data(cls.inputs, cls.gold_rows, {'binding': b['metadata'], 'arithmetic': a['metadata']})
        cls.typed = score.load_typed_scorer()
        cls.anchors = [row for row in cls.gold_rows if row['section'] == 'arithmetic' and
                       row['numeric_variant'] == 0 and row['response_type'] == 'unit_explicit']

    def predictions(self):
        return [{'id': row['id'], 'parent_id': row['parent_id'], 'prediction': row['reference'], 'terminated': True}
                for row in self.gold_rows]

    def analyze_fixture(self, n=30, lora_retained=15, mixed_retained=22):
        endpoints, maps = {}, {}
        for role in score.ROLES:
            pred = self.predictions(); by_id = {row['id']: row for row in pred}
            for index, anchor in enumerate(self.anchors):
                keep = index < n if role == 'source' else index < (lora_retained if role == 'lora' else mixed_retained)
                if not keep:
                    by_id[anchor['id']]['prediction'] = 'incorrect'
            if role != 'mixed':
                seen = set()
                for target in self.gold_rows:
                    if target['section'] == 'binding' and target['parent_id'] not in seen:
                        by_id[target['id']]['prediction'] = 'incorrect'
                        seen.add(target['parent_id'])
            endpoints[role], maps[role] = score.score_rows(self.inputs, self.gold, pred, self.typed)
        return score.analyze(endpoints, maps, self.gold, CRITERIA)

    def test_worlds_not_512_views_are_primary_units(self):
        result = self.analyze_fixture()
        primary = result['primary']['binding']
        self.assertEqual((primary['n'], primary['net'], primary['nd']), (32, 32, 32))
        self.assertEqual(primary['exact_p'], 2 ** -31)
        self.assertEqual(len(result['parent_maps']['mixed']['binding']['canonical_complete_exact/all16']), 32)
        self.assertEqual(result['endpoints']['mixed']['totals']['binding']['canonical_complete_exact'], 512)

    def test_source_anchor_eligibility_not_candidate_or_all_four(self):
        result = self.analyze_fixture()
        selected = result['source_conditioned_retention_descriptive']
        self.assertEqual(selected['n'], 30)
        self.assertTrue(selected['source_only_eligibility'])
        self.assertFalse(selected['candidate_conditioned_eligibility'])
        self.assertEqual(selected['anchor_ids'], [r['id'] for r in self.anchors[:30]])
        primary = result['primary']['arithmetic']
        self.assertEqual((primary['n'], primary['net'], primary['nd']), (30, 7, 7))
        self.assertTrue(primary['large_relative_diagnostic_difference'])
        self.assertEqual(primary['direction'], 'mixed_higher')
        for record in selected['models'].values():
            self.assertNotIn('exact_p', record)
            self.assertNotIn('holm_adjusted_p', record)
            self.assertFalse(record['inferential_test_performed'])
        changed = self.analyze_fixture(lora_retained=0, mixed_retained=96)
        self.assertEqual(selected['anchor_ids'], changed['source_conditioned_retention_descriptive']['anchor_ids'])

    def test_empty_cohort_retains_holm_placeholder_not_observed_test(self):
        result = self.analyze_fixture(n=0)
        item = result['primary']['arithmetic']
        self.assertIsNone(item['exact_p'])
        self.assertFalse(item['observed_test'])
        self.assertEqual(item['multiplicity_p_fraction'], '1')
        self.assertEqual(item['holm_adjusted_p'], 1)
        self.assertEqual(item['reference_cohort_status'], 'INSUFFICIENT_REFERENCE_COHORT')
        self.assertEqual(result['primary']['binding']['holm_adjusted_p'], 2 ** -30)
        self.assertEqual(result['predictions']['source_conditioned_strict_retention_fraction_mixed']['verdict'],
                         'UNAVAILABLE_EMPTY_REFERENCE_COHORT')

    def test_small_nonempty_cohort_keeps_observed_p_without_claim(self):
        item = self.analyze_fixture(n=20, lora_retained=0, mixed_retained=20)['primary']['arithmetic']
        self.assertTrue(item['observed_test'])
        self.assertEqual(item['exact_p'], 2 ** -19)
        self.assertFalse(item['large_relative_diagnostic_difference'])
        self.assertEqual(item['reference_cohort_status'], 'INSUFFICIENT_REFERENCE_COHORT')

    def test_paired_thresholds_and_sign(self):
        values = {str(i): True for i in range(6)}
        item = score.exact_pair(values, {key: False for key in values})
        self.assertEqual((item['net'], item['nd'], item['exact_p']), (-6, 6, .03125))
        self.assertEqual(item['minimum_significant_abs_net_at_observed_nd'], 6)
        self.assertIsNone(item['minimum_initial_Holm2_abs_net_at_observed_nd'])
        for n in (0, 1, 2, 3, 4, 5):
            self.assertEqual(score.probability(n, 0), min(1, 2 ** (1 - n)))

    def test_numeric_only_units_are_not_required(self):
        pred = self.predictions()
        endpoint, _ = score.score_rows(self.inputs, self.gold, pred, self.typed)
        rows = [row for row in endpoint['rows'].values() if row['section'] == 'arithmetic' and
                row['response_type'] == 'numeric_value_only']
        self.assertEqual(len(rows), 192)
        self.assertTrue(all(row['strict_correct'] for row in rows))

    def test_nonanchor_source_failure_does_not_change_eligibility(self):
        result = self.analyze_fixture()
        maps = copy.deepcopy(result['parent_maps'])
        # Eligibility uses strict_anchor alone, never four-view complete success.
        maps['source']['arithmetic']['strict_all4'] = {key: False for key in maps['source']['arithmetic']['strict_all4']}
        altered = score.analyze(result['endpoints'], maps, self.gold, CRITERIA)
        self.assertEqual(altered['source_conditioned_retention_descriptive']['anchor_ids'],
                         result['source_conditioned_retention_descriptive']['anchor_ids'])

    def test_eos_and_case_sensitivity(self):
        pred = self.predictions()
        target = next(row for row in self.gold_rows if row['section'] == 'binding' and not row['answerable'])
        row = next(row for row in pred if row['id'] == target['id'])
        row['prediction'] = '  unknown\n'
        endpoint, maps = score.score_rows(self.inputs, self.gold, pred, self.typed)
        self.assertTrue(endpoint['rows'][target['id']]['canonical_complete_exact'])
        self.assertFalse(endpoint['rows'][target['id']]['strict_complete_exact'])
        row['terminated'] = False
        endpoint, maps = score.score_rows(self.inputs, self.gold, pred, self.typed)
        self.assertFalse(endpoint['rows'][target['id']]['canonical_complete_exact'])
        self.assertFalse(maps['binding']['canonical_complete_exact/all16'][target['parent_id']])

    def test_no_partial_scientific_output_after_gate_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with patch.object(score, 'build_report', side_effect=ValueError('custody failure')):
                result = score.run(root)
            self.assertEqual(result['status'], 'BLOCKED_ON_COMPARABILITY')
            self.assertNotIn('endpoints', result)
            self.assertNotIn('primary', result)
            self.assertEqual({path.name for path in root.iterdir()}, {'RESULTS.json', 'RESULTS.md'})
            self.assertFalse(json.loads((root / 'RESULTS.json').read_text())['scientific_endpoints_emitted'])
            with self.assertRaises(ValueError):
                score.run(root)

    def test_bytelevel_decoder_skip_special_and_utf8(self):
        token = {'decoder': {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': False, 'use_regex': False},
                 'model': {'vocab': {'hello': 0, 'Ġworld': 1, 'Ã©': 2, 'Ċ': 3}},
                 'added_tokens': [{'id': 4, 'content': '<eos>', 'special': True}]}
        decoder = score.ByteLevelDecoder(token, {'clean_up_tokenization_spaces': False})
        self.assertEqual(decoder.decode([0, 1, 3, 2, 4]), 'hello world\né')
        with self.assertRaises(ValueError):
            decoder.decode([20])

    def test_token_charge_and_decoded_text_tampering_are_rejected(self):
        tokenizer = {'decoder': {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': False, 'use_regex': False},
                     'model': {'vocab': {'x': 0}}, 'added_tokens': [{'id': 1, 'content': '<eos>', 'special': True}]}
        decoder = score.ByteLevelDecoder(tokenizer, {})
        inputs = [{'id': str(index), 'parent_id': 'parent' + str(index), 'prompt_ids': [10, 11, 12]} for index in range(896)]
        rows = [{'id': r['id'], 'parent_id': r['parent_id'], 'prediction': 'x', 'terminated': True,
                 'input_tokens': 3, 'generated_tokens': 2, 'generated_ids_including_eos': [0, 1],
                 'forward_calls': 2, 'processed_input_tokens': 4} for r in inputs]
        generated = {'predictions': rows, 'eos_ids': [1], 'extra_warmup_or_qualification_calls': 0,
                     'actual_budget_charges': {'rows': 896, 'generated_tokens_including_eos': 1792,
                                              'forward_calls': 1792, 'processed_input_tokens': 3584}}
        reservation = {'reserved_generated_tokens': 21504, 'forward_calls': 21504, 'processed_input_tokens': 25088}
        self.assertEqual(score.verify_predictions(generated, inputs, reservation, decoder, [1])['rows'], 896)
        for field, bad in (('prediction', 'forged'), ('input_tokens', 4), ('terminated', False),
                           ('generated_tokens', 3), ('forward_calls', 1), ('processed_input_tokens', 5),
                           ('parent_id', 'other'), ('generated_ids_including_eos', [1, 0])):
            changed = copy.deepcopy(generated)
            changed['predictions'][0][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                score.verify_predictions(changed, inputs, reservation, decoder, [1])

    def test_original_assets_bytelevel_decoding_without_tokenizer_import(self):
        source = Path('C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze')
        assets = score.c.read(source / 'tokenizer.json')
        config = score.c.read(source / 'tokenizer_config.json')
        decoder = score.ByteLevelDecoder(assets, config)
        reverse = {byte: char for char, byte in decoder.bytes.items()}
        text = 'UNKNOWN\n17 dollars; 12:05 PM; café'
        ids = [assets['model']['vocab'][reverse[value]] for value in text.encode('utf-8')]
        ids.append(151645)
        self.assertEqual(decoder.decode(ids), text)
        self.assertEqual(score.expected_eos(source, assets, config), [151643, 151645])
        score.no_model_imports()


if __name__ == '__main__':
    unittest.main()
