"""Exact rational references for prospective mechanics; no model or bank I/O."""
from fractions import Fraction
import math
import unittest

from .operating_characteristics import (CONDITIONAL_CEILING, LOSS_PROBABILITIES,
    SOURCE_CORRECT_DENOMINATORS, audit, dependence_safe_all_pass_bounds,
    maximum_allowed_failures, minimum_n_with_zero_losses)
from .statistics import ENDPOINT_ALPHA, clopper_pearson_upper

EXACT_ALPHA = Fraction(1, 960)  # .05 / 48 without floating-point inversion


def exact_cdf(k, n, probability):
    probability = Fraction(str(probability))
    a, b = probability.numerator, probability.denominator
    numerator = sum(math.comb(n, i) * a ** i * (b - a) ** (n - i) for i in range(k + 1))
    return Fraction(numerator, b ** n)


def exact_allowed(n, ceiling):
    if n == 0:
        return None
    probability = Fraction(str(ceiling))
    a, b = probability.numerator, probability.denominator
    denominator, numerator, maximum = b ** n, 0, -1
    for k in range(n + 1):
        numerator += math.comb(n, k) * a ** k * (b - a) ** (n - k)
        # U_CP(k,n)<=ceiling iff CDF_ceiling(k)<=alpha.
        if numerator * EXACT_ALPHA.denominator <= denominator * EXACT_ALPHA.numerator:
            maximum = k
        else:
            break
    return maximum


class OperatingCharacteristicsTests(unittest.TestCase):
    def test_zero_loss_denominator_minima_independent_exact_arithmetic(self):
        for ceiling, expected in ((.05, 134), (.10, 66)):
            self.assertEqual(minimum_n_with_zero_losses(ceiling), expected)
            probability = Fraction(str(1 - ceiling))
            self.assertLessEqual(probability ** expected, EXACT_ALPHA)
            self.assertGreater(probability ** (expected - 1), EXACT_ALPHA)
            self.assertLessEqual(clopper_pearson_upper(0, expected), ceiling)
            self.assertGreater(clopper_pearson_upper(0, expected - 1), ceiling)

    def test_population_allowance_matches_rational_binomial_reference(self):
        maximum = maximum_allowed_failures(512, .05)
        self.assertEqual(maximum, exact_allowed(512, .05))
        self.assertLessEqual(exact_cdf(maximum, 512, .05), EXACT_ALPHA)
        self.assertGreater(exact_cdf(maximum + 1, 512, .05), EXACT_ALPHA)
        self.assertLessEqual(clopper_pearson_upper(maximum, 512), .05)
        self.assertGreater(clopper_pearson_upper(maximum + 1, 512), .05)

    def test_all_conditional_denominators_match_independent_count_boundaries(self):
        for n in SOURCE_CORRECT_DENOMINATORS:
            self.assertEqual(maximum_allowed_failures(n, .10), exact_allowed(n, .10))
        self.assertIsNone(maximum_allowed_failures(0, .10))
        self.assertEqual(maximum_allowed_failures(32, .10), -1)
        self.assertEqual(maximum_allowed_failures(64, .10), -1)
        self.assertEqual(maximum_allowed_failures(66, .10), 0)

    def test_marginal_curves_match_exact_rational_cdf(self):
        report = audit()
        population = report["semantic_and_strict_population_margins"]
        k = population["maximum_allowed_failure_count"]
        for entry in population["marginal_curve"]:
            probability = entry["declared_true_loss_probability_per_this_denominator"]
            self.assertAlmostEqual(entry["exact_binomial_marginal_acceptance_probability"],
                                   float(exact_cdf(k, 512, probability)), places=11)
        conditional = next(item for item in report["conditional_semantic_margins_by_declared_source_correct_n"] if item["n"] == 66)
        p01 = next(item for item in conditional["marginal_curve"]
                   if item["declared_true_loss_probability_per_this_denominator"] == .01)
        self.assertAlmostEqual(p01["exact_binomial_marginal_acceptance_probability"],
                               float(Fraction(99, 100) ** 66), places=12)

    def test_zero_probability_and_monotonic_loss_grid_do_not_fake_power(self):
        report = audit()
        all_curves = [report["semantic_and_strict_population_margins"],
                      *report["conditional_semantic_margins_by_declared_source_correct_n"]]
        for margin in all_curves:
            entries = margin["marginal_curve"]
            self.assertEqual(tuple(item["declared_true_loss_probability_per_this_denominator"] for item in entries), LOSS_PROBABILITIES)
            probabilities = [item["exact_binomial_marginal_acceptance_probability"] for item in entries]
            self.assertTrue(all(left >= right for left, right in zip(probabilities, probabilities[1:])))
            expected_at_zero = 1 if margin["maximum_allowed_failure_count"] is not None and margin["maximum_allowed_failure_count"] >= 0 else 0
            self.assertEqual(probabilities[0], expected_at_zero)

    def test_same_zero_losses_can_pass_population_and_fail_conditional(self):
        report = audit()
        example = next(item for item in report["same_damage_count_examples"]
                       if item["source_correct_n"] == 64 and item["same_actual_damage_count"] == 0)
        self.assertTrue(example["population_bound_passes"])
        self.assertFalse(example["conditional_bound_passes"])
        self.assertTrue(example["population_pass_conditional_fail"])
        example = next(item for item in report["same_damage_count_examples"]
                       if item["source_correct_n"] == 66 and item["same_actual_damage_count"] == 1)
        self.assertTrue(example["population_bound_passes"])
        self.assertFalse(example["conditional_bound_passes"])
        missing = next(item for item in report["same_damage_count_examples"] if item["source_correct_n"] == 0)
        self.assertEqual(missing["conditional_status"], "INCONCLUSIVE")
        self.assertIsNone(missing["conditional_upper"])

    def test_dependence_safe_bounds_are_not_independent_products(self):
        result = dependence_safe_all_pass_bounds([.99] * 14)
        self.assertAlmostEqual(result["all_pass_lower_union_bound"], .86)
        self.assertAlmostEqual(result["all_pass_upper_minimum_marginal"], .99)
        self.assertFalse(result["independence_assumed"])
        self.assertIsNone(result["joint_power_estimate"])
        result = dependence_safe_all_pass_bounds([.9] * 14)
        self.assertEqual(result["all_pass_lower_union_bound"], 0)
        self.assertEqual(result["all_pass_upper_minimum_marginal"], .9)
        result = dependence_safe_all_pass_bounds([1] * 13 + [.5])
        self.assertEqual(result["all_pass_lower_union_bound"], .5)
        self.assertEqual(result["all_pass_upper_minimum_marginal"], .5)
        with self.assertRaises(ValueError):
            dependence_safe_all_pass_bounds([.99] * 48)

    def test_no_observed_outcomes_final_access_or_threshold_changes(self):
        report = audit()
        self.assertEqual(report["status"], "PROSPECTIVE_DESIGN_MECHANICS_NO_MODEL_EVIDENCE")
        self.assertEqual(report["observed_model_outcomes_used"], 0)
        self.assertEqual(report["monte_carlo_draws"], 0)
        self.assertIsNone(report["joint_power_estimate"])
        self.assertFalse(report["quality_claim"])
        self.assertFalse(report["final_payload_accessed"])
        self.assertFalse(report["frozen_thresholds_changed"])
        self.assertFalse(report["sample_sizes_changed"])
        self.assertEqual(report["rules"]["bonferroni_endpoints"], 48)
        self.assertEqual(report["rules"]["endpoint_alpha"], ENDPOINT_ALPHA)
        self.assertIn("approximately half", report["three_percentage_point_floor_note"])
        self.assertIn("not Rineya failures", report["design_interpretation"])

    def test_invalid_margin_and_probability_arguments_are_rejected(self):
        for args in ((True, .1), (-1, .1), (512, 0), (512, 1)):
            with self.assertRaises(ValueError):
                maximum_allowed_failures(*args)
        with self.assertRaises(ValueError):
            dependence_safe_all_pass_bounds([float("nan")] * 14)


if __name__ == "__main__":
    unittest.main()
