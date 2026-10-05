"""Stdlib statistical-definition controls; no new model-quality evidence."""
from __future__ import annotations

import math
import unittest

from experiments.rineya_quality_queue_v1.pattern_statistics import (
    clopper_pearson, critical_count, is_floor_clear, upper_tail,
)


class PatternStatisticsTests(unittest.TestCase):
    def test_d0_seed17_point_estimate_exceeds_floor_but_two_sided_rule_fails(self):
        self.assertGreater(41 / 128, 31 / 128)
        pvalue = upper_tail(41, 128, 31 / 128)
        self.assertAlmostEqual(pvalue, .02759750548516958, places=14)
        self.assertLess(pvalue, .05)
        self.assertGreater(pvalue, .025)
        lower, upper = clopper_pearson(41, 128)
        self.assertAlmostEqual(lower, .2406, delta=.00005)
        self.assertAlmostEqual(upper, .4085, delta=.00005)
        self.assertFalse(is_floor_clear(41, 128, 31))
        self.assertTrue(is_floor_clear(42, 128, 31))

    def test_one_sided_and_two_sided_critical_counts_are_distinct(self):
        for floor_count, one_sided, two_sided in ((31, 40, 42), (7, 12, 13), (9, 15, 16)):
            with self.subTest(floor_count=floor_count):
                floor = floor_count / 128
                self.assertEqual(critical_count(128, floor, .05), one_sided)
                self.assertEqual(critical_count(128, floor, .025), two_sided)
                self.assertFalse(is_floor_clear(two_sided - 1, 128, floor_count))
                self.assertTrue(is_floor_clear(two_sided, 128, floor_count))

    def test_remaining_d0_ledger_vectors_do_not_clear_two_sided_floor(self):
        for k, expected_lower in ((36, .2054), (39, .2265)):
            with self.subTest(k=k):
                self.assertGreater(k / 128, 31 / 128)
                self.assertFalse(is_floor_clear(k, 128, 31))
                self.assertAlmostEqual(clopper_pearson(k, 128)[0], expected_lower, delta=.00005)

    def test_binomial_tail_has_known_fair_coin_values(self):
        self.assertEqual(upper_tail(2, 3, .5), .5)
        self.assertEqual(upper_tail(3, 3, .5), .125)
        self.assertEqual(upper_tail(0, 3, .5), 1)
        self.assertEqual(upper_tail(1, 3, .5), .875)

    def test_tail_endpoints_and_unattainable_critical_count(self):
        self.assertEqual(upper_tail(0, 3, 0), 1)
        self.assertEqual(upper_tail(1, 3, 0), 0)
        self.assertEqual(upper_tail(3, 3, 1), 1)
        self.assertEqual(critical_count(3, 0), 1)
        self.assertEqual(critical_count(3, 1), 4)

    def test_rejection_boundary_is_strict(self):
        self.assertEqual(upper_tail(1, 1, .5), .5)
        self.assertEqual(critical_count(1, .5, .5), 2)
        self.assertEqual(critical_count(1, .5, math.nextafter(.5, 1)), 1)

    def test_exact_interval_endpoints_and_three_of_three_uncertainty(self):
        lo, hi = clopper_pearson(0, 3)
        self.assertEqual(lo, 0)
        self.assertAlmostEqual(hi, 1 - .025 ** (1 / 3), places=14)
        lo, hi = clopper_pearson(3, 3)
        self.assertAlmostEqual(lo, .025 ** (1 / 3), places=14)
        self.assertEqual(hi, 1)
        self.assertLess(lo, .3)  # Three passes cannot establish a reliable pass rate.

    def test_interval_symmetry_and_confidence_width(self):
        a = clopper_pearson(4, 10)
        b = clopper_pearson(6, 10)
        self.assertAlmostEqual(a[0], 1 - b[1], places=14)
        self.assertAlmostEqual(a[1], 1 - b[0], places=14)
        narrower = clopper_pearson(4, 10, .9)
        self.assertLess(a[0], narrower[0])
        self.assertGreater(a[1], narrower[1])

    def test_scope_boundary_and_tail_monotonicity(self):
        self.assertEqual(upper_tail(512, 512, .5), 2 ** -512)
        self.assertGreater(upper_tail(3, 10, .6), upper_tail(3, 10, .4))
        self.assertGreater(upper_tail(3, 10, .5), upper_tail(4, 10, .5))
        for function, args in ((upper_tail, (1, 513, .5)), (clopper_pearson, (1, 513)),
                               (critical_count, (513, .5)), (is_floor_clear, (1, 513, 0))):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function(*args)

    def test_invalid_counts_and_booleans_are_not_statistical_evidence(self):
        for k, n in ((True, 3), (1, True), (1.0, 3), (1, 3.0), (-1, 3), (4, 3),
                     (0, 0), (1, -1)):
            for function in (upper_tail, clopper_pearson):
                with self.subTest(k=k, n=n, function=function.__name__):
                    with self.assertRaises(ValueError):
                        function(k, n, .5)
        for floor_count in (True, -1, 4, 1.0):
            with self.subTest(floor_count=floor_count):
                with self.assertRaises(ValueError):
                    is_floor_clear(1, 3, floor_count)
        for n in (True, 0, -1, 3.0):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    critical_count(n, .5)

    def test_invalid_probabilities_confidences_and_alpha_rejected(self):
        for value in (True, False, float("nan"), float("inf"), -float("inf"), -.1, 1.1, 10 ** 400, "0.5"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    upper_tail(1, 3, value)
                with self.assertRaises(ValueError):
                    critical_count(3, value)
        for value in (True, False, float("nan"), float("inf"), 0, 1, -.1, 1.1, 10 ** 400, "0.95"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    clopper_pearson(1, 3, value)
                with self.assertRaises(ValueError):
                    is_floor_clear(1, 3, 0, value)
                with self.assertRaises(ValueError):
                    critical_count(3, .5, value)


if __name__ == "__main__":
    unittest.main()
