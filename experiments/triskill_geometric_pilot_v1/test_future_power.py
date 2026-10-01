"""Independent small-N checks of the prospective stdlib power calculator."""
from fractions import Fraction
import itertools
import math
import unittest

from future_power import (
    COMPARATORS, PER_COMPARISON_ALPHA, SAMPLE_SIZES, SCENARIOS,
    binomial_pmf, critical_gain_count, exact_two_sided_p,
    paired_gain_power, rejects_beneficial, simultaneous_power_lower_bound,
)


def brute_power(n, gain, loss, alpha):
    """Enumerate ternary world outcomes, using rational null tails separately."""
    total = Fraction(0)
    concordance = 1 - gain - loss
    probabilities = {"g": gain, "l": loss, "c": concordance}
    for outcomes in itertools.product("glc", repeat=n):
        g, l = outcomes.count("g"), outcomes.count("l")
        d = g + l
        if g <= l:
            continue
        null_tail = Fraction(sum(math.comb(d, k) for k in range(g, d + 1)), 2 ** d)
        if 2 * null_tail <= alpha:
            probability = Fraction(1)
            for outcome in outcomes:
                probability *= probabilities[outcome]
            total += probability
    return float(total)


class FuturePowerTests(unittest.TestCase):
    def test_binomial_matches_direct_combinatorial_masses(self):
        for n in (0, 1, 7, 20):
            for p in (0.0, 0.13, 0.5, 0.87, 1.0):
                expected = [math.comb(n, k) * p ** k * (1-p) ** (n-k) for k in range(n+1)]
                actual = binomial_pmf(n, p)
                self.assertAlmostEqual(math.fsum(actual), 1.0, places=14)
                for observed, target in zip(actual, expected):
                    self.assertAlmostEqual(observed, target, places=14)

    def test_stable_large_and_extreme_probabilities(self):
        for p in (1e-12, 0.01, 0.5, 0.99, 1-1e-12):
            pmf = binomial_pmf(512, p)
            self.assertEqual(len(pmf), 513)
            self.assertTrue(all(math.isfinite(x) and 0 <= x <= 1 for x in pmf))
            self.assertAlmostEqual(math.fsum(pmf), 1, places=14)
            self.assertAlmostEqual(math.fsum(k*x for k, x in enumerate(pmf)), 512*p, places=10)

    def test_small_n_matches_independent_exact_ternary_enumeration(self):
        for n in (0, 3, 6, 7):
            for gain, loss in ((Fraction(2, 10), Fraction(1, 10)),
                               (Fraction(1, 10), Fraction(2, 10)),
                               (Fraction(1, 10), Fraction(1, 10))):
                for alpha in (Fraction(1, 2), Fraction(1, 10), Fraction(1, 160)):
                    with self.subTest(n=n, gain=gain, loss=loss, alpha=alpha):
                        self.assertAlmostEqual(paired_gain_power(n, float(gain), float(loss), float(alpha)),
                                               brute_power(n, gain, loss, alpha), places=13)

    def test_critical_region_matches_exact_rational_null(self):
        alpha = Fraction(1, 160)
        for d in range(65):
            expected = None
            for g in range(d // 2 + 1, d + 1):
                tail = Fraction(sum(math.comb(d, k) for k in range(g, d+1)), 2**d)
                if 2 * tail <= alpha:
                    expected = g
                    break
            self.assertEqual(critical_gain_count(d), expected)

    def test_three_wins_cannot_confirm_and_p_is_symmetric(self):
        self.assertEqual(exact_two_sided_p(3, 0), 0.25)
        self.assertEqual(exact_two_sided_p(0, 3), 0.25)
        self.assertFalse(rejects_beneficial(3, 0))
        self.assertTrue(rejects_beneficial(9, 0))
        self.assertFalse(rejects_beneficial(0, 9))
        self.assertEqual(exact_two_sided_p(0, 0), 1.0)
        self.assertFalse(rejects_beneficial(0, 0))
        self.assertIsNone(critical_gain_count(0))

    def test_no_worlds_or_no_discordance_cannot_reject(self):
        self.assertEqual(paired_gain_power(0, .15, .05), 0.0)
        self.assertEqual(paired_gain_power(512, 0.0, 0.0), 0.0)
        self.assertEqual(paired_gain_power(512, 0.0, 0.2), 0.0)

    def test_null_and_wrong_direction_do_not_have_beneficial_power(self):
        null_power = paired_gain_power(512, .10, .10)
        wrong_power = paired_gain_power(512, .05, .15)
        self.assertGreater(null_power, 0)
        self.assertLessEqual(null_power, PER_COMPARISON_ALPHA / 2)
        self.assertLess(wrong_power, null_power)
        self.assertLess(wrong_power, 1e-8)

    def test_prospective_powers_have_sensible_bounds_and_order(self):
        for _, gain, loss in SCENARIOS:
            powers = [paired_gain_power(n, gain, loss) for n in SAMPLE_SIZES]
            self.assertTrue(all(0 <= p <= 1 for p in powers))
            self.assertEqual(sorted(powers), powers)
        self.assertGreater(paired_gain_power(512, .15, .05), .90)
        self.assertGreater(paired_gain_power(512, .15, .05), paired_gain_power(512, .12, .04))
        self.assertGreater(paired_gain_power(512, .12, .04), paired_gain_power(512, .10, .05))

    def test_simultaneous_bound_does_not_assume_independent_tests(self):
        self.assertEqual(len(COMPARATORS), 8)
        self.assertAlmostEqual(PER_COMPARISON_ALPHA, .00625)
        self.assertEqual(simultaneous_power_lower_bound((1.0,) * 8), 1.0)
        self.assertEqual(simultaneous_power_lower_bound((.90,) * 8), .20000000000000018)
        self.assertEqual(simultaneous_power_lower_bound((.50,) * 8), 0.0)
        self.assertAlmostEqual(simultaneous_power_lower_bound((.99,) * 8), .92)
        self.assertEqual(simultaneous_power_lower_bound((.75,)), .75)

    def test_invalid_inputs_fail_closed(self):
        for n in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                paired_gain_power(n, .15, .05)
        for gain, loss in ((-.1, .1), (.8, .3), (float("nan"), .1), (.1, float("inf"))):
            with self.assertRaises(ValueError):
                paired_gain_power(10, gain, loss)
        for alpha in (0, 1, -1, float("nan")):
            with self.assertRaises(ValueError):
                paired_gain_power(10, .15, .05, alpha)
        with self.assertRaises(ValueError):
            simultaneous_power_lower_bound(())


if __name__ == "__main__":
    unittest.main()
