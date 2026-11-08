"""Stdlib binding fixtures only; neither label/model efficacy nor native qualification."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.skill_acquisition_native_v1 import ce_plan as cp


class FixtureTests(unittest.TestCase):
    @staticmethod
    def records(count=16, suffix='a'):
        return [{'id': 'fixture-%03d' % index, 'target_sha256': cp.digest([suffix, index])}
                for index in range(count)]

    def test_deterministic_common_order_distinct_label_bindings_and_export(self):
        first = cp.build_synthetic(self.records(), 'first')
        again = cp.build_synthetic(list(reversed(self.records())), 'first')
        second = cp.build_synthetic(self.records(suffix='b'), 'second')
        self.assertEqual(first, again)
        self.assertEqual(first['shared_exposure_sha256'], second['shared_exposure_sha256'])
        self.assertNotEqual(first['label_binding_sha256'], second['label_binding_sha256'])
        self.assertEqual(len(cp.validate_plan(first)), 2)
        exported = cp.checkpoint_export(first)
        self.assertEqual(exported['ordered_plan_sha256'], cp.digest(exported['ordered_plan']))
        self.assertEqual(exported['ce_plan_sha256'], first['plan_sha256'])
        self.assertEqual(exported['checkpoint_mode'], 'CE8_SYNTHETIC_CPU_1')

    def test_fixture_partial_duplicate_or_malformed_identity_rejected(self):
        for change in ('partial', 'duplicate', 'malformed'):
            records = self.records()
            if change == 'partial': records.pop()
            if change == 'duplicate': records[-1]['id'] = records[0]['id']
            if change == 'malformed': records[0]['target_sha256'] = 'bad'
            with self.subTest(change=change), self.assertRaises(ValueError):
                cp.build_synthetic(records)

    def test_future_references_do_not_supply_seed_budget_qualification_or_grant(self):
        references = {'recipe': {'sha256': 'a' * 64, 'approved': True, 'training_authorized': True},
                      'anchors': {'common_reference': 'b' * 64, 'own_start': 'b' * 64}}
        plan = cp.build_synthetic(self.records(), binding_references=references)
        self.assertEqual(plan['unbound_reference_fields'], ['runtime', 'source'])
        self.assertIsNone(plan['binding_references']['runtime'])
        for key in ('approved', 'training_authorized', 'native_qualified', 'execution_ready',
                    'teacher_seed_is_training_rng_seed'):
            self.assertIs(plan[key], False)
        self.assertNotIn('training_rng_seed', plan)
        self.assertNotIn('budgets', plan)
        forged = copy.deepcopy(plan)
        forged['training_authorized'] = True
        forged['plan_sha256'] = cp.digest({k: v for k, v in forged.items() if k != 'plan_sha256'})
        with self.assertRaisesRegex(ValueError, 'never grants'):
            cp.checkpoint_export(forged)

    def test_recomputed_digest_cannot_hide_changed_order_or_target_binding(self):
        for change in ('order', 'target_binding'):
            plan = cp.build_synthetic(self.records())
            if change == 'order':
                plan['shared_exposure']['boundaries'][0]['row_ids'].reverse()
                plan['ordered_plan'] = plan['shared_exposure']['boundaries']
                plan['shared_exposure_sha256'] = cp.digest(plan['shared_exposure'])
            else:
                plan['label_binding']['boundaries'][0]['row_target_pairs'][0][0] = 'unplanned'
                plan['label_binding']['boundary_labels_sha256'] = cp.digest(plan['label_binding']['boundaries'])
                plan['label_binding_sha256'] = cp.digest(plan['label_binding'])
            plan['plan_sha256'] = cp.digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
            with self.subTest(change=change), self.assertRaises(ValueError):
                cp.validate_plan(plan)


class ProposalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='ce8-plan-binding-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / 'data'
        (self.data / 'manifests').mkdir(parents=True)
        self.out = self.root / 'prepared'
        self.out.mkdir()
        self.ids = cp._order(['row-%04d' % index for index in range(3000)])
        atomic = {'name': 'atomic_R', 'rows': 3000, 'ids': list(reversed(self.ids))}
        raw = cp.canonical(atomic) + b'\n'
        (self.data / 'manifests/atomic_R.json').write_bytes(raw)
        self.preparation = {'schema': 'R-ORACLE-SUPERVISION-PREPARATION-1', 'approved': False,
            'training_authorized': False, 'source_sha256': 'b' * 64, 'rows': 3000, 'parent_ids': 3000,
            'order_rule': cp.ORDER, 'permutation_rule': [cp.PERMUTE, 1301, 'row_id', 0, 'index'],
            'atomic_R_sha256': cp.digest(atomic), 'census': None,
            'data': {'dataset_root': str(self.data), 'manifest_sha256': 'c' * 64, 'rows': 3000,
                'selected_ids_sha256': cp.digest(sorted(self.ids)),
                'files': {'manifests/atomic_R.json': {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}},
            'targets': {}}
        self.records = {}
        for arm in cp.ARMS:
            values = []
            for index, row_id in enumerate(self.ids):
                answer = [1, 2, 3]
                if arm == 'permuted':
                    answer = sorted(answer, key=lambda item: (cp.digest([cp.PERMUTE, 1301, row_id, 0, item]), item))
                target = cp.canonical({'answer': answer}).decode('ascii')
                prompt = [{'role': 'system', 'content': 'synthetic receipt parser fixture'},
                          {'role': 'user', 'content': row_id}]
                values.append({'id': row_id, 'parent_id': 'parent-' + row_id, 'line': index + 1,
                    'logical_exposure': 0, 'prompt': prompt,
                    'messages': prompt + [{'role': 'assistant', 'content': target}],
                    'target': target, 'target_sha256': hashlib.sha256(target.encode()).hexdigest()})
            self.records[arm] = values
            self.write_arm(arm)
        self.pin()

    def write_arm(self, arm):
        path = self.out / (arm + '.messages.jsonl')
        with path.open('wb') as stream:
            for row in self.records[arm]: stream.write(cp.canonical(row) + b'\n')
        self.preparation['targets'][arm] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size, 'rows': len(self.records[arm]),
            'ordered_target_digest': cp.digest([[r['id'], r['target_sha256']] for r in self.records[arm]])}

    def pin(self):
        self.path = self.out / 'PREPARATION.json'
        raw = cp.canonical(self.preparation) + b'\n'
        self.path.write_bytes(raw)
        self.expected = hashlib.sha256(raw).hexdigest()

    def build(self, arm='canonical'):
        return cp.build_proposal(self.path, self.expected, arm)

    def test_exact_proposal_scope_shared_order_and_distinct_policy_labels(self):
        canonical = self.build()
        permuted = self.build('permuted')
        self.assertEqual(len(canonical['ordered_plan']), 375)
        self.assertTrue(all(len(b['row_ids']) == 8 and b['logical_exposure'] == 0 for b in canonical['ordered_plan']))
        self.assertEqual(canonical['shared_exposure_sha256'], permuted['shared_exposure_sha256'])
        self.assertNotEqual(canonical['label_binding_sha256'], permuted['label_binding_sha256'])
        self.assertEqual(canonical['preparation']['producer_source_sha256'], 'b' * 64)
        self.assertNotEqual(canonical['builder_source_sha256'], 'b' * 64)
        self.assertEqual(canonical['unbound_reference_fields'], list(cp.REFERENCE_FIELDS))
        with self.assertRaisesRegex(ValueError, 'not a checkpoint execution scope'):
            cp.checkpoint_export(canonical)

    def test_raw_preparation_and_target_file_pins_cannot_be_replaced(self):
        with self.assertRaisesRegex(ValueError, 'preparation raw hash'):
            cp.build_proposal(self.path, 'a' * 64, 'canonical')
        path = Path(self.preparation['targets']['canonical']['path'])
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b'parent-', b'Parent-', 1))
        with self.assertRaisesRegex(ValueError, 'file bytes/hash'):
            self.build()

    def test_partial_duplicate_order_ordinal_and_exposure_drift_even_with_new_file_pins(self):
        original = copy.deepcopy(self.records['canonical'])
        for change in ('partial', 'duplicate', 'order', 'ordinal', 'exposure'):
            values = copy.deepcopy(original)
            if change == 'partial': values.pop()
            if change == 'duplicate': values[-1]['id'] = values[0]['id']
            if change == 'order': values[0], values[1] = values[1], values[0]
            if change == 'ordinal': values[0]['line'] = 2
            if change == 'exposure': values[0]['logical_exposure'] = 1
            self.records['canonical'] = values
            self.write_arm('canonical'); self.pin()
            with self.subTest(change=change), self.assertRaises(ValueError): self.build()

    def test_changed_target_messages_and_permutation_policy_refused(self):
        original = copy.deepcopy(self.records['permuted'])
        for change in ('target_hash', 'messages', 'policy'):
            values = copy.deepcopy(original)
            if change == 'target_hash': values[0]['target'] = '{"answer":[99]}'
            if change == 'messages': values[0]['messages'][-1]['content'] = 'different'
            if change == 'policy':
                answer = json.loads(values[0]['target'])['answer']
                target = cp.canonical({'answer': answer[1:] + answer[:1]}).decode()
                values[0]['target'] = target
                values[0]['target_sha256'] = hashlib.sha256(target.encode()).hexdigest()
                values[0]['messages'][-1]['content'] = target
            self.records['permuted'] = values
            self.write_arm('permuted'); self.pin()
            with self.subTest(change=change), self.assertRaises(ValueError): self.build('permuted')

    def test_wrong_arm_malformed_metadata_selected_membership_and_grant_flags(self):
        with self.assertRaisesRegex(ValueError, 'unknown proposal arm'): self.build('native')
        for change in ('grant', 'selected', 'malformed'):
            original = copy.deepcopy(self.preparation)
            if change == 'grant': self.preparation['training_authorized'] = True
            if change == 'selected': self.preparation['data']['selected_ids_sha256'] = 'a' * 64
            if change == 'malformed': self.preparation['data']['files'] = []
            self.pin()
            with self.subTest(change=change), self.assertRaises(ValueError): self.build()
            self.preparation = original


if __name__ == '__main__':
    unittest.main()
