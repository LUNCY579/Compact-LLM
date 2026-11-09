import collections
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('cc0_audit', Path(__file__).with_name('cc0_audit.py'))
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditSafetyTests(unittest.TestCase):
    def test_exact_shard_population(self):
        self.assertEqual(len(AUDIT.SHARDS['train']), 22)
        self.assertEqual(len(AUDIT.SHARDS['validation']), 5)
        self.assertFalse(any('/test/' in p or 'sealed' in p or p.endswith('.zip') for p in AUDIT.ALLOWED))

    def test_protected_paths_rejected_before_open(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(Path, 'open', side_effect=AssertionError('must not read')):
            for forbidden in ('data/sft/test/shard-000-of-005.jsonl', 'sealed_eval/prompts/prompts.jsonl',
                              'reports/main_lineage/test.jsonl', 'raw_sources/expanded_anchor_cards.json',
                              '../data/sft/train/shard-000-of-022.jsonl', 'main.zip'):
                with self.assertRaises(ValueError):
                    AUDIT.approved_path(Path(temp), forbidden)

    def test_imbalanced_accuracy_not_balanced_accuracy(self):
        counts = collections.Counter()
        for _ in range(11):
            AUDIT.add_confusion(counts, False, False)
        AUDIT.add_confusion(counts, False, True)
        report = AUDIT.confusion_report(counts)
        self.assertEqual(report['balanced_accuracy'], .5)
        self.assertEqual(report['accuracy'], 11 / 12)

    def test_multiturn_cue_uses_final_user_only(self):
        messages = [{'role':'system','content':'rules'}, {'role':'user','content':AUDIT.CUE},
                    {'role':'assistant','content':'UNKNOWN'}, {'role':'user','content':'What is the quantity?'},
                    {'role':'assistant','content':'7'}]
        self.assertNotIn(AUDIT.CUE, AUDIT.last_user(messages))

    def test_selection_ties_are_id_stable(self):
        a, b = [], []
        for key in [(2,'b'), (1,'c'), (1,'a'), (3,'d')]:
            AUDIT.keep_smallest(a, key, key, 2)
        for key in reversed([(2,'b'), (1,'c'), (1,'a'), (3,'d')]):
            AUDIT.keep_smallest(b, key, key, 2)
        self.assertEqual(a, b)
        self.assertEqual([item[0] for item in a], [(1,'a'), (1,'c')])

    def test_create_only_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'receipt.json'
            AUDIT.dump_new(path, {'original':True})
            with self.assertRaises(FileExistsError):
                AUDIT.dump_new(path, {'overwritten':True})
            self.assertIn('original', path.read_text())

    def test_first_registry_does_not_return_later_member(self):
        self.assertEqual(AUDIT.registry_first('R01 :: exact_name=Alice ; card=X\nR02 :: exact_name=Bob ; card=Y', 'train'), 'Alice')
        self.assertEqual(AUDIT.registry_first('Audit member 01 is exactly “Alice” (X).', 'validation'), 'Alice')
        self.assertIsNone(AUDIT.registry_first('R02 :: exact_name=Bob ; card=Y', 'train'))


if __name__ == '__main__':
    unittest.main()
