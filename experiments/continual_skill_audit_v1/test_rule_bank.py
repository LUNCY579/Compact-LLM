from collections import defaultdict
import json
import random
import unittest

from experiments.continual_skill_audit_v1.rule_bank import build, joint_answer, make_world, transition, typed_shape
from experiments.continual_skill_audit_v1.bank_shortcuts import features


class RuleBankTests(unittest.TestCase):
    def test_prerequisite_matches_independent_set_semantics(self):
        rng = random.Random(3)
        for before in range(256):
            event = {k: rng.randrange(256) for k in ['pre_on', 'pre_off', 'add', 'delete']}
            bits = lambda n: {i for i in range(8) if (n >> i) & 1}
            original = bits(before)
            applicable = bits(event['pre_on']) <= original and not bits(event['pre_off']) & original
            expected = (original - bits(event['delete'])) | bits(event['add']) if applicable else original
            self.assertEqual(transition('prerequisite_plan', before, event), sum(1 << i for i in expected))

    def test_preconditions_and_order_are_decisive(self):
        for seed in range(30):
            before, direct, equal, unequal = make_world(random.Random(seed), 'prerequisite_plan')
            first = direct[0][1]
            full = transition('prerequisite_plan', before, first)
            wrong_order = ((before | first['add']) & ~first['delete']) & 255
            self.assertNotEqual(full, wrong_order)
            for state, event, label in direct[-2:]:
                self.assertEqual(transition('prerequisite_plan', state, event), state)
                self.assertNotEqual(((state & ~event['delete']) | event['add']) & 255, state)
            self.assertEqual(typed_shape(equal), typed_shape(unequal))

    def test_ledger_and_register_oracles(self):
        before = {'A': 5, 'B': -2}
        event = dict(kind='transfer', status='posted', account='A', other='B', amount=3)
        self.assertEqual(transition('ledger_replay', before, event), {'A': 2, 'B': 1})
        self.assertEqual(transition('ledger_replay', before, {**event, 'status': 'void'}), before)
        registers = {'X': 96, 'Y': 2}
        instruction = dict(kind='conditional', register='X', other='Y', value=5, threshold=2)
        self.assertEqual(transition('register_machine', registers, instruction), {'X': 91, 'Y': 2})
        self.assertEqual(transition('register_machine', registers, {**instruction, 'threshold': 1}), {'X': 4, 'Y': 2})

    def test_banks_are_disjoint_grouped_and_oracle_consistent(self):
        rows = build(seed=123, train_worlds=3, dev_worlds=2)
        self.assertEqual(len({r['semantic_key'] for r in rows}), len(rows))
        groups = defaultdict(list)
        for row in rows:
            groups[row['world_id']].append(row)
            payload = json.loads(row['prompt'][1]['content'])
            if row['task'] == 'joint':
                self.assertEqual(row['reference'], joint_answer(row['family'], payload['before'], payload['events']))
            self.assertNotIn(row['world_id'], row['prompt'][1]['content'])
        self.assertEqual(len(groups), 15)
        for siblings in groups.values():
            self.assertEqual(len({r['split'] for r in siblings}), 1)
            joint = [r for r in siblings if r['task'] == 'joint']
            self.assertEqual(sum(r['reference'] == 'UNKNOWN' for r in joint), 1)
            payloads = [json.loads(r['prompt'][1]['content']) for r in joint]
            self.assertEqual(typed_shape(payloads[0]), typed_shape(payloads[1]))
        self.assertEqual(rows, build(seed=123, train_worlds=3, dev_worlds=2))

    def test_v2_pairs_block_detected_parity_and_guard_shortcuts(self):
        groups = defaultdict(list)
        for row in build(seed=20260919, train_worlds=8, dev_worlds=4):
            if row['task'] == 'joint':
                groups[row['world_id']].append(row)
        for pair in groups.values():
            left, right = map(features, pair)
            for key in ['same_value_parity', 'same_amount_parity', 'add_touches_pre_on', 'add_touches_pre_off']:
                if key in left:
                    self.assertEqual(left[key], right[key])


if __name__ == '__main__':
    unittest.main()
