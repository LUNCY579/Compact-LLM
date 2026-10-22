"""Finite enumeration audits the factorial closed form independently."""
import unittest
from experiments.skill_acquisition_method_lab_v1 import permutation_complexity as pc


class PermutationComplexityTests(unittest.TestCase):
    def test_closed_form_matches_explicit_tries_1_through_6(self):
        for n in range(1, 7):
            with self.subTest(n=n):
                self.assertTrue(pc.enumerate_small(n)['closed_form_equal'])

    def test_actual_train_histogram_aggregate(self):
        panel = pc.train_panel({6: 1000, 8: 1000, 10: 1000})
        self.assertEqual(panel['prompt_rows'], 3000)
        self.assertEqual(panel['valid_compact_orderings_total'], 3669840000)
        self.assertEqual(panel['item_prefix_nodes_total_including_roots'], 9975659000)
        self.assertEqual(panel['item_prefix_edges_total'], 9975656000)
        self.assertEqual(panel['single_teacher_item_edges_total'], 24000)

    def test_invalid_sizes_and_unbounded_enumeration_rejected(self):
        for value in (True, 0, -1, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pc.counts(value)
        with self.assertRaises(ValueError):
            pc.enumerate_small(8)
        with self.assertRaises(ValueError):
            pc.train_panel({6: False})


if __name__ == '__main__':
    unittest.main()
