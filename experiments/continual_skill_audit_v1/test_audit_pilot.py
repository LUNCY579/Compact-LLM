import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.continual_skill_audit_v1.audit_pilot import audit_run, whole_value


def write(path, data, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(('\n'.join(json.dumps(x) for x in data) if jsonl else json.dumps(data)) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PilotAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        old = [{'id': f'o{i}', 'world_id': 'world', 'family': 'binding', 'prompt': [{'role': 'user', 'content': str(i)}],
                'reference': 'Nora' if i % 2 == 0 else 'UNKNOWN', 'answerable': i % 2 == 0} for i in range(4)]
        new = [{'id': f'{family}{i}', 'parent_id': f'{family}{i}', 'family': family,
                'prompt': [{'role': 'user', 'content': f'{family} {i}'}], 'reference': '{"A":1}'}
               for family in ['ledger_replay', 'prerequisite_plan', 'register_machine'] for i in range(128)]
        write(self.root / 'dev.json', old)
        write(self.root / 'anchor.json', [])
        acquisition = self.root / 'selection.jsonl'
        write(acquisition, new, True)
        write(self.root / 'PLAN.json', {'acquisition': str(acquisition), 'bindings': {str(acquisition): sha(acquisition)},
                                      'local_data': {name: sha(self.root / name) for name in ['dev.json', 'anchor.json']}})
        for stage in ['baseline', 'eval-control']:
            for endpoint, rows in [('old_dev', old), ('acquisition', new)]:
                preds = [{'id': r['id'], 'world_id': r.get('world_id', r.get('parent_id')), 'family': r['family'],
                          'reference': r['reference'], 'prediction': r['reference'], 'correct': True, 'terminated': True} for r in rows]
                if stage == 'eval-control' and endpoint == 'old_dev':
                    preds[0].update(prediction='Wrong', correct=False)
                write(self.root / stage / f'{endpoint}.jsonl', preds, True)
            self.complete(stage)
        self.complete('train-control')

    def complete(self, stage):
        directory = self.root / stage
        directory.mkdir(exist_ok=True)
        artifacts = {p.name: sha(p) for p in directory.glob('*.jsonl')}
        write(directory / 'COMPLETE.json', {'stage': stage, 'artifacts': artifacts})
        write(directory / 'supervisor.json', {'status': 'COMPLETE', 'child_returncode': 0})

    def test_rescore_and_cluster_from_complete_receipts(self):
        result = audit_run(self.root)
        paired = result['contrasts']['base_to_control']['old_dev']
        self.assertEqual(paired['lost'], 1)
        self.assertEqual(paired['groups'], 1)
        self.assertEqual(result['arms']['control']['binding_joint_worlds']['correct'], 0)
        self.assertFalse(result['quality_improvement_established'])

    def test_signed_duplicate_rows_do_not_silently_collapse(self):
        path = self.root / 'eval-control/old_dev.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        write(path, rows + rows[:1], True)
        self.complete('eval-control')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit_run(self.root)

    def test_signed_false_success_is_rejected(self):
        path = self.root / 'eval-control/old_dev.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['correct'] = True
        write(path, rows, True)
        self.complete('eval-control')
        with self.assertRaisesRegex(ValueError, 'correctness'):
            audit_run(self.root)

    def test_changed_bank_is_rejected(self):
        write(self.root / 'dev.json', [])
        with self.assertRaisesRegex(ValueError, 'data changed'):
            audit_run(self.root)

    def test_failed_supervision_is_rejected(self):
        write(self.root / 'eval-control/supervisor.json', {'status': 'FAILED', 'child_returncode': 1})
        with self.assertRaisesRegex(ValueError, 'unsuccessful'):
            audit_run(self.root)

    def test_empty_constraint_mask_is_uninformative_not_a_perfect_score(self):
        coverage = self.root / 'coverage.json'
        write(coverage, {'input_sha256': {'selection': sha(self.root / 'selection.jsonl')},
                         'families': {'prerequisite_plan': {'ablations': {
                             'ignore_preconditions': {'decisive_ids': [], 'decisive': 0}}}}})
        result = audit_run(self.root, coverage)['rule_ablation_decisive_subsets_exploratory']
        row = result['prerequisite_plan/ignore_preconditions']
        self.assertFalse(row['informative'])
        self.assertEqual(row['contrasts'], {})

    def test_typed_sensitivity_does_not_scavenge_answers(self):
        self.assertEqual(whole_value('9:05.', 'time'), whole_value('09:05', 'time'))
        for text in ['9:85', '24:00', '09:05 AM', 'It is 09:05', '09:05\nwrong', '9:5']:
            self.assertIsNone(whole_value(text, 'time'))
        self.assertEqual(whole_value('-3.', 'arithmetic'), -3)
        for text in ['-3 or 4', 'not -3', '3.0', '3..']:
            self.assertIsNone(whole_value(text, 'arithmetic'))


if __name__ == '__main__':
    unittest.main()
