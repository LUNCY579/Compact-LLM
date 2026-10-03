import math
import unittest

from experiments.continual_skill_audit_v1.metrics import (
    Outcome, compare, composition, exact_sign_flip_p, loss_upper_bound, sequence,
)


def bank(values, groups=None):
    return [Outcome(str(i), (groups or [str(j) for j in range(len(values))])[i],
                    f"prompt-and-target-{i}", value) for i, value in enumerate(values)]


class AuditTests(unittest.TestCase):
    def test_equal_totals_do_not_hide_forgetting(self):
        result = compare(bank([True, False]), bank([False, True]))
        self.assertEqual((result['net'], result['lost'], result['gained'], result['churn']), (0, 1, 1, 2))
        self.assertFalse(result['observed_zero_loss'])

    def test_zero_loss_does_not_prove_population_retention(self):
        result = compare(bank([True] * 47), bank([True] * 47))
        self.assertTrue(result['observed_zero_loss'])
        self.assertAlmostEqual(result['row_loss_upper95_iid_only'], 1 - .05 ** (1 / 47))
        self.assertGreater(result['row_loss_upper95_iid_only'], .06)
        self.assertFalse(result['population_no_forgetting_established'])

    def test_no_source_success_is_uninformative(self):
        result = compare(bank([False]), bank([True]))
        self.assertIsNone(result['row_loss_upper95_iid_only'])
        self.assertIsNone(result['group_any_loss_upper95_iid_only'])

    def test_grouped_variants_are_not_independent(self):
        result = compare(bank([True] * 8, ['a'] * 4 + ['b'] * 4),
                         bank([False] * 8, ['a'] * 4 + ['b'] * 4))
        self.assertEqual(result['damaged_groups'], 2)
        self.assertEqual(result['group_sign_flip_p_descriptive'], .5)
        self.assertEqual(result['row_mcnemar_p_descriptive'], 2 / 256)

    def test_refuses_silent_inner_join_duplicate_and_drift(self):
        good = bank([True, False])
        variants = [good[:1], good + good[:1],
                    [Outcome('0', 'wrong-parent', good[0].fingerprint, True), good[1]],
                    [Outcome('0', '0', 'different-prompt', True), good[1]],
                    [Outcome('0', '0', good[0].fingerprint, 1), good[1]]]
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compare(good, bad)

    def test_sequential_loss_of_acquired_skill_is_visible(self):
        result = sequence([bank([True, False]), bank([True, True]), bank([True, False])])[-1]
        self.assertEqual(result['vs_base']['lost'], 0)
        self.assertEqual(result['vs_previous']['lost'], 1)
        self.assertEqual(result['vs_ever_correct']['lost'], 1)

    def test_exact_statistics_known_values(self):
        self.assertEqual(exact_sign_flip_p([]), 1)
        self.assertEqual(exact_sign_flip_p([1, -1]), 1)
        self.assertEqual(exact_sign_flip_p([1] * 6), .03125)
        self.assertAlmostEqual(loss_upper_bound(1, 2), math.sqrt(.95))
        self.assertEqual(loss_upper_bound(2, 2), 1)
        self.assertIsNone(loss_upper_bound(0, 0))
        for bad in [(-1, 2), (3, 2), (0.5, 2)]:
            with self.assertRaises(ValueError):
                loss_upper_bound(*bad)

    def test_composition_reports_asymmetry_and_unreachable_component(self):
        result = composition({'transition': bank([True, True]), 'boundary': bank([False, False])},
                             {'transition': bank([True, False]), 'boundary': bank([False, False])})
        self.assertEqual(result['component_successes_lost'], 1)
        self.assertFalse(result['all_components_have_measured_competence'])
        self.assertFalse(result['joint_task_generalization_established'])
        with self.assertRaises(ValueError):
            composition({'transition': bank([True])}, {})


if __name__ == '__main__':
    unittest.main()
