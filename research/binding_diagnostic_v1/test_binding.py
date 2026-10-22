import copy
import unittest
from . import binding


class BindingDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = binding.build_rows()

    def test_balanced_crossed_counterfactuals(self):
        bundle = self.bundle
        self.assertEqual(sum(row['answerable'] for row in bundle['gold']), 256)
        self.assertEqual(len(bundle['metadata']), 32)
        for domain in binding.DOMAINS:
            self.assertEqual(sum(row['domain'] == domain for row in bundle['gold']), 256)
        for state in ('missing', 'vacant'):
            self.assertEqual(sum(row['state'] == state for row in bundle['gold']), 128)

    def test_missing_assignment_not_filled_from_roster(self):
        index = next(i for i, row in enumerate(self.bundle['gold']) if row['state'] == 'missing')
        decoded = binding.parse_policy_input(self.bundle['policy_inputs'][index]['prompt'])
        self.assertEqual(decoded['reference'], 'UNKNOWN')
        self.assertGreater(len(set(decoded['roster']) - set(decoded['assignments'].values())), 1)

    def test_changed_decisive_fact_changes_answer(self):
        prompt = copy.deepcopy(self.bundle['policy_inputs'][0]['prompt'])
        before = binding.parse_policy_input(prompt)
        replacement = before['roster'][-1]
        prompt[1]['content'] = prompt[1]['content'].replace(
            f"The {before['queried_role']} is {before['reference']}.",
            f"The {before['queried_role']} is {replacement}.")
        self.assertEqual(binding.parse_policy_input(prompt)['reference'], replacement)

    def test_gold_corruption_rejected(self):
        bad = copy.deepcopy(self.bundle)
        bad['gold'][0]['reference'] = 'UNKNOWN'
        with self.assertRaises(ValueError):
            binding.validate_rows(bad)

    def test_conflicting_fact_rejected(self):
        prompt = copy.deepcopy(self.bundle['policy_inputs'][0]['prompt'])
        parsed = binding.parse_policy_input(prompt)
        prompt[1]['content'] += '\nThe ' + parsed['queried_role'] + ' has no occupant.'
        with self.assertRaises(ValueError):
            binding.parse_policy_input(prompt)

    def test_no_policy_gold_fields(self):
        self.assertTrue(all(set(row) == {'id', 'parent_id', 'prompt'} for row in self.bundle['policy_inputs']))
        self.assertEqual(binding.build_rows(), self.bundle)


if __name__ == '__main__':
    unittest.main()
