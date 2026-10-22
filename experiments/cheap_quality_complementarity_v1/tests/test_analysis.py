"""Synthetic, bounded algorithm checks. Does not execute the historical study."""
import tempfile
import unittest
from pathlib import Path

from experiments.cheap_quality_complementarity_v1.analyze import (
    PACKAGE, ROOT, analyze, comparison_kind, exact_ids, failure_kind, load_json,
    load_scorer, pair_stats, subset_oracles)


class ComplementarityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = load_scorer(ROOT / 'experiments/architecture_benchmark_001/retention.py')
        (PACKAGE / 'runs').mkdir(exist_ok=True)

    def test_pair_partition_and_oracle_ceiling(self):
        pair = pair_stats({'a', 'b'}, {'b', 'c'}, ['a', 'b', 'c', 'd'])
        self.assertEqual([pair[key] for key in ('both_correct', 'left_only_correct',
                                               'right_only_correct', 'neither_correct')], [1, 1, 1, 1])
        self.assertEqual(pair['oracle_union'], 3)
        self.assertEqual(pair['oracle_headroom_over_best_single'], 1)
        with self.assertRaises(ValueError):
            pair_stats({'outside'}, set(), ['a'])

    def test_all_subsets_including_singletons_and_joint_failure(self):
        result = subset_oracles({'A': {'a'}, 'B': {'b'}, 'C': {'a'}}, ['a', 'b', 'c'])
        self.assertEqual(len(result), 7)
        all_models = next(row for row in result if len(row['models']) == 3)
        self.assertEqual(all_models['oracle_union'], 2)
        self.assertEqual(all_models['joint_failure_ids'], ['c'])
        identical_pair = next(row for row in result if row['models'] == ['A', 'C'])
        self.assertEqual(identical_pair['oracle_headroom_over_best_single'], 0)

    def test_reject_missing_extra_ids_and_truthy_termination(self):
        exact_ids({'a': {'text': '5', 'terminated': True}}, ['a'], 'synthetic')
        for value in ({}, {'a': {'text': '5', 'terminated': 1}},
                      {'a': {'text': '5', 'terminated': True}, 'b': {'text': '5', 'terminated': True}}):
            with self.assertRaises(ValueError):
                exact_ids(value, ['a'], 'synthetic')

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory(dir=PACKAGE / 'runs') as temp:
            path = Path(temp) / 'duplicate.json'
            path.write_text('{"predictions":{"a":{},"a":{}}}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                load_json(path)

    def test_format_type_value_and_eos_categories_are_distinct(self):
        target = {'kind': 'number', 'value': '5', 'bare_allowed': True}
        cases = [('not 5', True, 'format_or_type_rejection'),
                 ('5 hours', True, 'format_or_type_rejection'),
                 ('6', True, 'parsed_value_mismatch'),
                 ('5', False, 'nontermination'), ('5', True, 'correct')]
        for text, terminated, expected in cases:
            self.assertEqual(failure_kind(self.scorer.score_prediction(text, terminated, target)), expected)

    def test_typed_equal_value_does_not_manufacture_sampling_gain(self):
        target = {'kind': 'number', 'value': '5', 'bare_allowed': True}
        kind = comparison_kind({'text': '5.', 'terminated': True},
                               {'text': 'The answer is 5.00.', 'terminated': True},
                               self.scorer, target)
        self.assertEqual(kind, 'same_typed_value_different_text')

    def test_analyze_ignores_supplied_correctness_and_keeps_losses(self):
        probe = [{'id': 'a', 'group_id': 'family_a', 'reply_class': 'synthetic'},
                 {'id': 'b', 'group_id': 'family_b', 'reply_class': 'synthetic'}]
        audit = {key: {'id': key, 'proposed_typed_target': {'type': 'number', 'value': {'rational': '5'}},
                       'independent_recomputation': {'template': 'synthetic'}} for key in ('a', 'b')}
        outputs = {
            'supa_initial': {'a': {'text': '5', 'terminated': True},
                             'b': {'text': '6', 'terminated': True}},
            'candidate': {'a': {'text': 'not 5', 'terminated': True, 'correct': True},
                          'b': {'text': '5', 'terminated': True, 'correct': False}}}
        result, records, _ = analyze(probe, audit, outputs, self.scorer)
        for policy in ('strict', 'permissive'):
            self.assertEqual(result[policy]['models']['candidate']['correct'], 1)
            retention = result[policy]['models']['candidate']['source_retention']
            self.assertEqual((retention['retained'], retention['lost'], retention['gained']), (0, 1, 1))
            self.assertEqual(result[policy]['correct_ids']['candidate'], ['b'])
        self.assertEqual(len(records), 8)


if __name__ == '__main__':
    unittest.main()
