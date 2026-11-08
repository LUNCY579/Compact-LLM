"""Independent tiny resampling references and strict metadata controls only."""

from __future__ import annotations

import copy
import itertools
import math
import random
import unittest
from fractions import Fraction
from unittest.mock import patch

from experiments.rineya_quality_queue_v1.cluster_statistics import (
    _inverse_empirical_cdf,
    stratified_cluster_bootstrap_interval,
)


def exact_cluster_mass(families: list[list[tuple[int, int]]]) -> dict[Fraction, Fraction]:
    """Enumerate every ordered whole-cluster selection with rational probabilities."""
    supports = []
    for clusters in families:
        mass = {}
        for indices in itertools.product(range(len(clusters)), repeat=len(clusters)):
            rate = Fraction(sum(clusters[index][0] for index in indices),
                            sum(clusters[index][1] for index in indices))
            mass[rate] = mass.get(rate, Fraction()) + Fraction(1, len(clusters)**len(clusters))
        supports.append(mass)
    result = {}
    for values in itertools.product(*(mass.items() for mass in supports)):
        macro = sum((rate for rate, _ in values), Fraction()) / len(families)
        probability = math.prod(probability for _, probability in values)
        result[macro] = result.get(macro, Fraction()) + probability
    assert sum(result.values()) == 1
    return result


def exact_percentiles(mass: dict[Fraction, Fraction], confidence: str) -> tuple[float, float]:
    tail = (1 - Fraction(confidence)) / 2

    def inverse(q):
        cumulative = Fraction()
        for value, probability in sorted(mass.items()):
            cumulative += probability
            if cumulative >= q:
                return float(value)
        raise AssertionError("exhaustive support incomplete")

    return inverse(tail), inverse(1 - tail)


def metadata(cases: list[tuple[str, str, str, int]]) -> tuple[list[dict], list[dict]]:
    """Toy externally declared identities and paired outcomes, not native evidence."""
    enrollment, outcomes = [], []
    for case_id, family, cluster_id, delta in cases:
        enrollment.append({"case_id": case_id, "family": family, "cluster_id": cluster_id})
        outcomes.append({"case_id": case_id, "complete": True,
                         "parent_correct": delta != 1, "candidate_correct": delta != -1})
    return enrollment, outcomes


def uneven_metadata():
    return metadata([("a1", "alpha", "alpha-A", 1), ("a2", "alpha", "alpha-A", 0),
                     ("a3", "alpha", "alpha-B", -1), ("b1", "beta", "beta-A", 1)])


class ClusterStatisticsTests(unittest.TestCase):
    def test_tiny_whole_cluster_support_and_quantiles_match_independent_enumeration(self):
        enrollment, outcomes = uneven_metadata()
        mass = exact_cluster_mass([[(1, 2), (-1, 1)], [(1, 1)]])
        self.assertEqual(mass, {Fraction(0): Fraction(1, 4), Fraction(1, 2): Fraction(1, 2),
                                Fraction(3, 4): Fraction(1, 4)})
        result = stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=90210, draws=1000)
        # Endpoint mass is 25%, safely away from the 5%/95% Monte Carlo cutoffs.
        self.assertEqual((result["lower"], result["upper"]), exact_percentiles(mass, "0.90"))
        self.assertIn(result["lower"], {float(value) for value in mass})
        self.assertIn(result["upper"], {float(value) for value in mass})
        self.assertEqual(result["net_rate"], .5)
        # A positive metadata contrast supplies none of these external proofs.
        self.assertIs(result["native_execution_authorized"], False)
        self.assertIs(result["quality_improvement_established"], False)
        self.assertIs(result["parent_independence_verified"], False)
        self.assertEqual(result["simulation"], "MONTE_CARLO")
        self.assertEqual(result["coverage"], "EXPLORATORY_NOT_CONFIRMATION")

    def test_unequal_cluster_sizes_keep_observed_case_weighted_point_and_counts(self):
        enrollment, outcomes = uneven_metadata()
        result = stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=4, draws=1000)
        alpha = result["families"][0]
        self.assertEqual((alpha["n"], alpha["cluster_count"], alpha["paired_delta"], alpha["net_rate"]),
                         (3, 2, 0, 0))
        self.assertNotEqual(alpha["net_rate"], (.5 - 1) / 2)
        self.assertEqual([(cluster["paired_delta"], cluster["n"]) for cluster in alpha["clusters"]],
                         [(1, 2), (-1, 1)])
        self.assertEqual((alpha["parent_correct"], alpha["candidate_correct"], alpha["repairs"], alpha["losses"]),
                         (2, 2, 1, 1))
        self.assertEqual((result["n"], result["cluster_count"], result["parent_correct"],
                          result["candidate_correct"], result["repairs"], result["losses"]), (4, 3, 2, 3, 2, 1))

    def test_repeated_siblings_have_wider_uncertainty_than_independent_row_shortcut(self):
        cases = [(f"plus{i}", "one", "plus-parent", 1) for i in range(10)]
        cases += [(f"minus{i}", "one", "minus-parent", -1) for i in range(10)]
        enrollment, outcomes = metadata(cases)
        result = stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=17, draws=1000)
        cluster_mass = exact_cluster_mass([[(10, 10), (-10, 10)]])
        row_mass = {Fraction(2 * plus - 20, 20): Fraction(math.comb(20, plus), 2**20)
                    for plus in range(21)}
        row_lower, row_upper = exact_percentiles(row_mass, "0.90")
        self.assertEqual((result["lower"], result["upper"]), exact_percentiles(cluster_mass, "0.90"))
        self.assertLess(result["lower"], row_lower)
        self.assertGreater(result["upper"], row_upper)
        self.assertEqual(result["net_rate"], 0)

    def test_fixed_equal_family_macro_does_not_pool_unequal_family_sizes(self):
        cases = [("one", "alpha", "alpha-parent", 1)]
        cases += [(f"many{i}", "beta", "beta-parent", -1) for i in range(9)]
        enrollment, outcomes = metadata(cases)
        result = stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=99, draws=1000)
        self.assertEqual((result["net_rate"], result["lower"], result["upper"]), (0, 0, 0))
        self.assertNotEqual(result["net_rate"], result["paired_delta"] / result["n"])
        self.assertEqual([family["n"] for family in result["families"]], [1, 9])
        self.assertEqual([family["net_rate"] for family in result["families"]], [1, -1])

    def test_input_order_seed_reproducibility_global_rng_and_inputs_are_preserved(self):
        enrollment, outcomes = uneven_metadata()
        originals = copy.deepcopy((enrollment, outcomes))
        global_state = random.getstate()
        expected = stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=123, draws=1000)
        self.assertEqual(global_state, random.getstate())
        self.assertEqual((enrollment, outcomes), originals)
        for order in itertools.permutations(range(len(enrollment))):
            actual = stratified_cluster_bootstrap_interval([enrollment[index] for index in order],
                                                           list(reversed(outcomes)), seed=123, draws=1000)
            self.assertEqual(actual, expected)
        try:
            random.seed(11)
            first_state = random.getstate()
            first = stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=123, draws=1000)
            self.assertEqual(first_state, random.getstate())
            random.seed(22)
            second_state = random.getstate()
            second = stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=123, draws=1000)
            self.assertEqual(second_state, random.getstate())
            self.assertEqual(first, second)
            self.assertEqual(second, expected)
        finally:
            random.setstate(global_state)
        self.assertEqual(expected, stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=123, draws=1000))
        changed_seed = stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=124, draws=1000)
        self.assertEqual(changed_seed["seed"], 124)
        self.assertEqual(changed_seed["net_rate"], expected["net_rate"])

    def test_zero_events_are_empirical_zero_not_a_future_harm_proof(self):
        enrollment, outcomes = metadata([("a", "one", "parent-A", 0), ("b", "one", "parent-B", 0)])
        outcomes[1].update(parent_correct=False, candidate_correct=False)
        result = stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=0, draws=1000)
        self.assertEqual((result["net_rate"], result["lower"], result["upper"]), (0, 0, 0))
        self.assertEqual((result["parent_correct"], result["candidate_correct"], result["repairs"], result["losses"]),
                         (1, 1, 0, 0))
        self.assertIn("Zero observed differences do not exclude unobserved future harms.", result["limitations"])
        self.assertEqual(result["cluster_identity"], "EXTERNALLY_DECLARED_PARENT_IDS_NOT_VERIFIED_INDEPENDENCE")
        self.assertNotIn("acceptance_rule", result)
        self.assertNotIn("practical_margin", result)

    def test_inverse_empirical_cdf_uses_supported_values_without_interpolation(self):
        values = [0.0, 0.0, 1.0, 2.0]
        self.assertEqual(_inverse_empirical_cdf(values, .5), 0)
        self.assertEqual(_inverse_empirical_cdf(values, math.nextafter(.5, 1.0)), 1)
        self.assertEqual(_inverse_empirical_cdf(values, .75), 1)
        self.assertEqual(_inverse_empirical_cdf(values, math.nextafter(.75, 1.0)), 2)

    def test_confidence_tail_ranks_preserve_exact_supplied_float_at_cdf_boundaries(self):
        enrollment, outcomes = metadata([("a", "one", "parent-A", 0),
                                         ("b", "one", "parent-B", 1)])
        for confidence, zero_samples in ((2**-55, 500), (.90, 950)):
            with self.subTest(confidence=confidence):
                class ControlledEmpiricalDistribution:
                    def __init__(self, seed):
                        self.calls = 0
                        self.cutoff = 2 * zero_samples

                    def randrange(self, n):
                        assert n == 2
                        index = int(self.calls >= self.cutoff)
                        self.calls += 1
                        return index

                mass = {Fraction(0): Fraction(zero_samples, 1000),
                        Fraction(1): Fraction(1000 - zero_samples, 1000)}
                expected = exact_percentiles(mass, str(Fraction.from_float(confidence)))
                with patch("experiments.rineya_quality_queue_v1.cluster_statistics.random.Random",
                           ControlledEmpiricalDistribution):
                    result = stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=0,
                                                                   draws=1000, confidence=confidence)
                self.assertEqual(expected, (0, 1))
                self.assertEqual((result["lower"], result["upper"]), expected)
                self.assertEqual(result["confidence_probability_convention"], "EXACT_SUPPLIED_BINARY_FLOAT_RATIO")
                self.assertEqual(result["confidence_ratio"], list(confidence.as_integer_ratio()))

    def test_partial_missing_duplicate_and_unmatched_outcomes_are_rejected(self):
        enrollment, outcomes = uneven_metadata()
        variants = [outcomes[:-1], outcomes + [copy.deepcopy(outcomes[0])]]
        partial = copy.deepcopy(outcomes); partial[0]["complete"] = False; variants.append(partial)
        extra = copy.deepcopy(outcomes); extra[0]["case_id"] = "unmatched"; variants.append(extra)
        duplicate = copy.deepcopy(outcomes); duplicate[1]["case_id"] = duplicate[0]["case_id"]; variants.append(duplicate)
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(ValueError):
                    stratified_cluster_bootstrap_interval(enrollment, variant, seed=1, draws=1000)
        with self.assertRaises(ValueError):
            stratified_cluster_bootstrap_interval(enrollment + [copy.deepcopy(enrollment[0])], outcomes,
                                                  seed=1, draws=1000)

    def test_shared_cluster_across_families_is_rejected(self):
        enrollment, outcomes = uneven_metadata()
        enrollment[-1]["cluster_id"] = enrollment[0]["cluster_id"]
        with self.assertRaises(ValueError):
            stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=1, draws=1000)

    def test_exact_plain_records_keys_identifiers_and_boolean_flags_are_required(self):
        class ListSubclass(list): pass
        class DictSubclass(dict): pass
        class StringSubclass(str): pass

        enrollment, outcomes = uneven_metadata()
        bad_enrollments = [tuple(enrollment), ListSubclass(enrollment), [], [None],
                           [DictSubclass(enrollment[0])]]
        for key, value in (("case_id", ""), ("family", "x" * 129), ("cluster_id", 1),
                           ("case_id", StringSubclass("a1"))):
            rows = copy.deepcopy(enrollment); rows[0][key] = value; bad_enrollments.append(rows)
        missing = copy.deepcopy(enrollment); del missing[0]["family"]; bad_enrollments.append(missing)
        extraneous = copy.deepcopy(enrollment); extraneous[0]["extra"] = 0; bad_enrollments.append(extraneous)
        nonstring = copy.deepcopy(enrollment); nonstring[0][0] = "extra"; bad_enrollments.append(nonstring)
        for rows in bad_enrollments:
            with self.subTest(enrollment=rows):
                with self.assertRaises(ValueError):
                    stratified_cluster_bootstrap_interval(rows, outcomes, seed=1, draws=1000)
        bad_outcomes = [tuple(outcomes), ListSubclass(outcomes), [], [None], [DictSubclass(outcomes[0])]]
        for field, value in (("complete", 1), ("complete", "true"), ("parent_correct", 1),
                             ("candidate_correct", 0), ("parent_correct", None)):
            rows = copy.deepcopy(outcomes); rows[0][field] = value; bad_outcomes.append(rows)
        missing = copy.deepcopy(outcomes); del missing[0]["complete"]; bad_outcomes.append(missing)
        extraneous = copy.deepcopy(outcomes); extraneous[0]["extra"] = False; bad_outcomes.append(extraneous)
        for rows in bad_outcomes:
            with self.subTest(outcomes=rows):
                with self.assertRaises(ValueError):
                    stratified_cluster_bootstrap_interval(enrollment, rows, seed=1, draws=1000)

    def test_row_family_identifier_and_simulation_bounds(self):
        enrollment, outcomes = metadata([(f"case{i}", "one", "parent", 0) for i in range(384)])
        self.assertEqual(stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=2**63 - 1,
                                                              draws=1000)["n"], 384)
        extra_enrollment = enrollment + [{"case_id": "extra", "family": "one", "cluster_id": "parent"}]
        extra_outcomes = outcomes + [{"case_id": "extra", "complete": True,
                                      "parent_correct": True, "candidate_correct": True}]
        with self.assertRaises(ValueError):
            stratified_cluster_bootstrap_interval(extra_enrollment, extra_outcomes, seed=1, draws=1000)
        for family_count, valid in ((16, True), (17, False)):
            rows, answers = metadata([(f"case{i}", f"family{i:02d}", f"parent{i}", 0)
                                      for i in range(family_count)])
            if valid:
                self.assertEqual(stratified_cluster_bootstrap_interval(rows, answers, seed=1, draws=1000)
                                 ["family_count"], 16)
            else:
                with self.assertRaises(ValueError):
                    stratified_cluster_bootstrap_interval(rows, answers, seed=1, draws=1000)
        rows, answers = metadata([("i" * 128, "f" * 128, "p" * 128, 0)])
        self.assertEqual(stratified_cluster_bootstrap_interval(rows, answers, seed=0, draws=50000)["n"], 1)
        for seed in (True, False, -1, 2**63, 1.0, None, "1"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=seed, draws=1000)
        for draws in (True, 999, 50001, 1000.0, None, "1000"):
            with self.subTest(draws=draws), self.assertRaises(ValueError):
                stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=1, draws=draws)
        for confidence in (True, False, None, "0.90", 0, 1, -.1, 1.1, float("nan"), float("inf"), 10**400):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                stratified_cluster_bootstrap_interval(enrollment, outcomes, seed=1, draws=1000, confidence=confidence)


if __name__ == "__main__":
    unittest.main()
