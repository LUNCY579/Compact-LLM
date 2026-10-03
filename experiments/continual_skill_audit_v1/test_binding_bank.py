from collections import defaultdict
import datetime
import json
import random
import re
import unittest

from experiments.continual_skill_audit_v1.binding_bank import (
    PEOPLE, arithmetic_key, binding_answer, binding_world, build,
)
from experiments.continual_skill_audit_v1.binding_shortcuts import predictors


class BindingBankTests(unittest.TestCase):
    def test_relational_oracle_does_not_infer_missing_roles_by_elimination(self):
        data = {'bindings': [('porter', 'Nora'), ('clerk', 'Theo')],
                'actions': [('guide', 'key', 'gate')],
                'query': {'task': 'two_hop', 'object': 'key', 'place': 'gate'}}
        self.assertEqual(binding_answer(data), 'UNKNOWN')
        data['bindings'].append(('guide', 'Nora'))
        self.assertEqual(binding_answer(data), 'Nora')
        data['actions'].append(('porter', 'key', 'gate'))
        self.assertEqual(binding_answer(data), 'Nora')
        data['bindings'].append(('guide', 'Theo'))
        self.assertEqual(binding_answer(data), 'UNKNOWN')

    def test_swap_and_missing_preserve_surface_counts_and_change_semantics(self):
        for salience in ['low', 'equal', 'high']:
            rows = binding_world(random.Random(4), 'anchor', salience)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({tuple(r['mention_counts']) for r in rows}), 1)
            available = [r for r in rows if r['answerable'] and r['task'] == 'two_hop']
            self.assertNotEqual(available[0]['reference'], available[1]['reference'])
            self.assertTrue(all(r['reference'] == 'UNKNOWN' for r in rows if not r['answerable']))
            for row in rows:
                if row['answerable']:
                    counts = dict(zip(row['people'], row['mention_counts']))
                    self.assertGreaterEqual(sum(v == counts[row['reference']] for v in counts.values()), 2)

    def test_rendered_relations_reconstruct_oracle(self):
        for split in ['anchor', 'dev']:
            rows = binding_world(random.Random(8), split, 'equal')
            for row in rows:
                if split == 'anchor':
                    bindings = [(role, person) for person, role in re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+) served as the ([a-z ]+)\.', row['context'])]
                    actions = re.findall(r'The ([a-z ]+) delivered the ([a-z ]+) to the ([a-z ]+)\.', row['context'])
                else:
                    bindings = re.findall(r'The ([a-z ]+) duties were assigned to ([A-Z][a-z]+ [A-Z][a-z]+)\.', row['context'])
                    actions = [(role, obj, place) for obj, place, role in re.findall(r'The ([a-z ]+) was taken to the ([a-z ]+) by the ([a-z ]+)\.', row['context'])]
                self.assertEqual(len(bindings), 4)
                self.assertEqual(len(actions), 4)
                recovered = {'bindings': bindings, 'actions': actions, 'query': row['oracle_payload']['query']}
                self.assertEqual(binding_answer(recovered), row['reference'])
                self.assertNotIn(row['world_id'], json.dumps(row['prompt']))
                self.assertNotIn('not named again', row['context'])

    def test_grouped_disjointness_determinism_and_clock_oracle(self):
        args = dict(seed=33, binding_worlds_per_condition=2, clock_worlds=4, arithmetic_worlds=3)
        rows = build(**args)
        self.assertEqual(rows, build(**args))
        self.assertEqual(len(rows), len({r['semantic_key'] for r in rows}))
        worlds = defaultdict(list)
        for row in rows:
            worlds[row['world_id']].append(row)
            if row['family'] == 'time':
                p = row['oracle_payload']
                value = datetime.datetime(2000, 1, 1, p['hour'], p['minute']) + datetime.timedelta(minutes=p['delay'])
                self.assertEqual(row['reference'], value.strftime('%H:%M'))
                self.assertEqual(p['carry'], p['minute'] + p['delay'] >= 60)
        for siblings in worlds.values():
            self.assertEqual(len({r['split'] for r in siblings}), 1)
        self.assertFalse(set(PEOPLE['anchor']) & set(PEOPLE['dev']))
        excluded = {r['semantic_key'] for r in rows[:8]}
        self.assertFalse(excluded & {r['semantic_key'] for r in build(**args, excluded=excluded)})

    def test_addition_semantics_are_commutative_for_exclusion(self):
        self.assertEqual(arithmetic_key(20, '+', 4), arithmetic_key(4, '+', 20))
        self.assertNotEqual(arithmetic_key(20, '-', 4), arithmetic_key(4, '-', 20))

    def test_question_and_unique_frequency_predictors_cannot_extract_target(self):
        for salience in ['low', 'equal', 'high']:
            for row in binding_world(random.Random(5), 'dev', salience):
                outputs = predictors(row)
                self.assertEqual(outputs['question_name_else_unknown'], 'UNKNOWN')
                self.assertEqual(outputs['unique_least_frequent_else_unknown'], 'UNKNOWN')
                self.assertEqual(outputs['unique_most_frequent_else_unknown'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
