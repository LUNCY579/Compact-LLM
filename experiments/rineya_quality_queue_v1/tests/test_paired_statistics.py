"""Exact small-parent enumeration and recorded-count controls, no native evidence."""

from __future__ import annotations

import itertools
import math
import unittest
from fractions import Fraction

from experiments.rineya_quality_queue_v1.paired_statistics import (
    _empirical_pmf,
    exact_mcnemar,
    paired_bootstrap_interval,
    stratified_paired_bootstrap_interval,
)


def enumeration_interval(benefits: int, harms: int, n: int, confidence: str) -> tuple[float, float]:
    """Independent reference: enumerate each of the 3**n categorical sequences."""
    probabilities = {-1: Fraction(harms, n), 0: Fraction(n - benefits - harms, n), 1: Fraction(benefits, n)}
    mass: dict[int, Fraction] = {}
    for outcomes in itertools.product((-1, 0, 1), repeat=n):
        probability = math.prod(probabilities[outcome] for outcome in outcomes)
        mass[sum(outcomes)] = mass.get(sum(outcomes), Fraction()) + probability
    assert sum(mass.values()) == 1
    tail = (1 - Fraction(confidence)) / 2

    def quantile(q: Fraction) -> float:
        cumulative = Fraction()
        for total, probability in sorted(mass.items()):
            cumulative += probability
            if probability and cumulative >= q:
                return total / n
        raise AssertionError("enumeration support incomplete")

    return quantile(tail), quantile(1 - tail)


def stratified_enumeration_interval(strata: list[tuple[int, int, int]], confidence: str) -> tuple[float, float]:
    """Independent exact reference: enumerate heterogeneous categorical trials."""
    trials = [
        {-1: Fraction(harms, n), 0: Fraction(n - benefits - harms, n), 1: Fraction(benefits, n)}
        for benefits, harms, n in strata for _ in range(n)
    ]
    mass: dict[int, Fraction] = {}
    for outcomes in itertools.product((-1, 0, 1), repeat=len(trials)):
        probability = math.prod(trial[outcome] for trial, outcome in zip(trials, outcomes))
        mass[sum(outcomes)] = mass.get(sum(outcomes), Fraction()) + probability
    assert sum(mass.values()) == 1
    tail = (1 - Fraction(confidence)) / 2

    def quantile(q: Fraction) -> float:
        cumulative = Fraction()
        for total, probability in sorted(mass.items()):
            cumulative += probability
            if probability and cumulative >= q:
                return total / len(trials)
        raise AssertionError("enumeration support incomplete")

    return quantile(tail), quantile(1 - tail)


class PairedStatisticsTests(unittest.TestCase):
    def test_zero_events_is_empirical_zero_not_evidence_of_no_future_harm(self):
        result = paired_bootstrap_interval(0, 0, 47)
        self.assertEqual((result["net_rate"], result["lower"], result["upper"]), (0, 0, 0))
        self.assertEqual(result["coverage"], "EXPLORATORY_NOT_CONFIRMATION")
        self.assertEqual(result["method"], "exact_empirical_paired_percentile")
        self.assertEqual(exact_mcnemar(0, 0), 1)

    def test_constant_benefits_and_harms_are_degenerate_empirical_distributions(self):
        for benefits, harms, expected in ((47, 0, 1), (0, 47, -1)):
            with self.subTest(benefits=benefits):
                result = paired_bootstrap_interval(benefits, harms, 47)
                self.assertEqual((result["net_rate"], result["lower"], result["upper"]), (expected,) * 3)

    def test_two_benefits_no_harms_do_not_establish_a_positive_lower_bound(self):
        result = paired_bootstrap_interval(2, 0, 47)
        self.assertEqual(result["net_rate"], 2 / 47)
        self.assertEqual(result["lower"], 0)
        self.assertGreater(result["upper"], 0)
        self.assertEqual(exact_mcnemar(2, 0), .5)

    def test_full_distribution_matches_independent_rational_enumeration(self):
        for benefits, harms, n in ((1, 1, 3), (2, 1, 4), (3, 0, 4), (1, 2, 5), (0, 0, 3), (4, 0, 4)):
            for confidence in ("0.90", "0.50", "0.875"):
                with self.subTest(benefits=benefits, harms=harms, n=n, confidence=confidence):
                    expected = enumeration_interval(benefits, harms, n, confidence)
                    result = paired_bootstrap_interval(benefits, harms, n, float(confidence))
                    self.assertEqual((result["lower"], result["upper"]), expected)

    def test_recorded_acquisition_and_family_contrasts_keep_uncertainty(self):
        for benefits, harms, n in ((5, 7, 384), (3, 2, 128), (1, 2, 128), (1, 3, 128)):
            with self.subTest(benefits=benefits, harms=harms):
                result = paired_bootstrap_interval(benefits, harms, n)
                self.assertEqual(result["net_rate"], (benefits - harms) / n)
                self.assertLessEqual(result["lower"], 0)
                self.assertGreaterEqual(result["upper"], 0)
                self.assertLess(result["lower"], result["upper"])
                self.assertEqual(result, paired_bootstrap_interval(benefits, harms, n))

    def test_probability_mass_is_finite_normalized_and_bounded(self):
        for benefits, harms, n in ((5, 7, 384), (1, 1, 512), (256, 256, 512), (512, 0, 512), (0, 0, 512)):
            with self.subTest(benefits=benefits, harms=harms, n=n):
                mass = _empirical_pmf(benefits, harms, n)
                self.assertEqual(len(mass), 2 * n + 1)
                self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1 for value in mass))
                self.assertAlmostEqual(math.fsum(mass), 1, places=14)
                expected = math.fsum((index - n) / n * probability for index, probability in enumerate(mass))
                self.assertAlmostEqual(expected, (benefits - harms) / n, places=14)

    def test_mcnemar_matches_small_discordant_sequence_enumeration(self):
        for benefits in range(6):
            for harms in range(6 - benefits):
                with self.subTest(benefits=benefits, harms=harms):
                    n = benefits + harms
                    if n:
                        high = max(benefits, harms)
                        count = sum(sum(outcomes) >= high for outcomes in itertools.product((0, 1), repeat=n))
                        expected = min(1.0, 2 * count / 2 ** n)
                    else:
                        expected = 1.0
                    self.assertEqual(exact_mcnemar(benefits, harms), expected)
                    self.assertEqual(exact_mcnemar(benefits, harms), exact_mcnemar(harms, benefits))
        self.assertEqual(exact_mcnemar(1, 1), 1)
        self.assertEqual(exact_mcnemar(256, 256), 1)
        self.assertEqual(exact_mcnemar(512, 0), 2 ** -511)

    def test_invalid_counts_and_denominators_are_rejected(self):
        for benefits, harms, n in ((True, 0, 1), (0, False, 1), (0, 0, True), (1.0, 0, 3), (0, 1.0, 3),
                                  (0, 0, 3.0), (-1, 0, 3), (0, -1, 3), (2, 2, 3), (0, 0, 0),
                                  (0, 0, 513), (float("nan"), 0, 3), (0, float("inf"), 3), (0, 0, None)):
            with self.subTest(benefits=benefits, harms=harms, n=n):
                with self.assertRaises(ValueError):
                    paired_bootstrap_interval(benefits, harms, n)
        for benefits, harms in ((True, 0), (0, False), (1.0, 0), (-1, 0), (0, -1),
                                (float("nan"), 0), (0, float("inf")), (256, 257), (513, 0)):
            with self.subTest(benefits=benefits, harms=harms):
                with self.assertRaises(ValueError):
                    exact_mcnemar(benefits, harms)
        with self.assertRaises(TypeError):
            paired_bootstrap_interval(1, 0)
        with self.assertRaises(TypeError):
            exact_mcnemar(1)

    def test_invalid_confidence_is_rejected(self):
        for confidence in (True, False, None, "0.90", 0, 1, -.1, 1.1, float("nan"), float("inf"), 10 ** 400):
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    paired_bootstrap_interval(1, 0, 3, confidence)

    def test_fixed_strata_match_independent_heterogeneous_enumeration(self):
        for strata in ([(1, 1, 2), (1, 0, 2)], [(2, 0, 2), (0, 2, 2)], [(0, 0, 2), (0, 1, 2)]):
            for confidence in ("0.90", "0.50", "0.875"):
                with self.subTest(strata=strata, confidence=confidence):
                    expected = stratified_enumeration_interval(strata, confidence)
                    result = stratified_paired_bootstrap_interval(strata, float(confidence))
                    self.assertEqual((result["lower"], result["upper"]), expected)
                    self.assertEqual(result["net_rate"], sum(benefits - harms for benefits, harms, _ in strata) / 4)

    def test_fixed_families_cannot_be_substituted_with_pooled_mixture(self):
        fixed = stratified_paired_bootstrap_interval([(2, 0, 2), (0, 2, 2)])
        pooled = paired_bootstrap_interval(2, 2, 4)
        self.assertEqual((fixed["net_rate"], fixed["lower"], fixed["upper"]), (0, 0, 0))
        self.assertLess(pooled["lower"], fixed["lower"])
        self.assertGreater(pooled["upper"], fixed["upper"])
        self.assertEqual(fixed["method"], "exact_empirical_stratified_paired_percentile")
        self.assertEqual(fixed["coverage"], "EXPLORATORY_NOT_CONFIRMATION")
        self.assertEqual(fixed["strata_count"], 2)
        self.assertEqual(fixed["fixed_allocation"], [2, 2])

    def test_recorded_fixed_family_contrast_is_permutation_invariant(self):
        strata = [(3, 2, 128), (1, 2, 128), (1, 3, 128)]
        result = stratified_paired_bootstrap_interval(strata)
        self.assertEqual(result["net_rate"], (5 - 7) / 384)
        self.assertEqual(result["fixed_allocation"], [128, 128, 128])
        self.assertEqual(result["strata_count"], 3)
        self.assertLessEqual(result["lower"], 0)
        self.assertGreaterEqual(result["upper"], 0)
        for permuted in itertools.permutations(strata):
            self.assertEqual(result, stratified_paired_bootstrap_interval(list(permuted)))

    def test_single_fixed_stratum_agrees_with_original_pooled_function(self):
        pooled = paired_bootstrap_interval(2, 1, 47)
        fixed = stratified_paired_bootstrap_interval([(2, 1, 47)])
        for key in ("net_rate", "lower", "upper", "confidence", "coverage"):
            self.assertEqual(pooled[key], fixed[key])

    def test_invalid_fixed_strata_are_rejected(self):
        for strata in (None, [], (), [(1, 0, 2), (1, 0, 3)], [(0, 0, 256)] * 3,
                       [(True, 0, 2)], [(0, -1, 2)], [(0, 0, 0)], [(1, 1, 1)],
                       [(1, 0, 2, 3)], [[1, 0, 2]], [None], [(float("nan"), 0, 2)]):
            with self.subTest(strata=strata):
                with self.assertRaises(ValueError):
                    stratified_paired_bootstrap_interval(strata)
        for confidence in (True, False, None, "0.90", 0, 1, float("nan"), float("inf"), 10 ** 400):
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    stratified_paired_bootstrap_interval([(1, 0, 2)], confidence)


if __name__ == "__main__":
    unittest.main()
