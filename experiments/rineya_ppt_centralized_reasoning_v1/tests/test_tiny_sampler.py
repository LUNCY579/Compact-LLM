"""Exact, CPU-only sampler qualification against an independent seven-state oracle.

The oracle uses only the literal table below. It never calls production scoring,
proposal, acceptance, or transition helpers to calculate an expected result.
This is mathematical fixture evidence, not native Rineya scoring or reasoning
evidence. The deliberately broken kernel is expected to fail qualification.
"""
from __future__ import annotations

from fractions import Fraction as F
from itertools import product
from math import lcm
from pathlib import Path
import random
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import tiny_sampler as sampler  # noqa: E402


STATES = (
    ("A", "A"), ("A", "B"), ("A", "EOS"),
    ("B", "A"), ("B", "B"), ("B", "EOS"),
    ("EOS", "PAD"),
)
TOKENS = ("A", "B", "EOS")
TABLE = {
    (): {"A": F(1, 2), "B": F(1, 3), "EOS": F(1, 6)},
    ("A",): {"A": F(1, 5), "B": F(1, 2), "EOS": F(3, 10)},
    ("B",): {"A": F(1, 2), "B": F(1, 8), "EOS": F(3, 8)},
}
ZERO = F(0)
ONE = F(1)


def base(record):
    """Independent complete-record probability, including EOS, excluding PAD."""
    mass = ONE
    prefix = ()
    for token in record:
        if token == "PAD":
            break
        mass *= TABLE[prefix][token]
        if token == "EOS":
            break
        prefix += (token,)
    return mass


def pi(power):
    weights = {s: base(s) ** power for s in STATES}
    total = sum(weights.values(), ZERO)
    return {s: weights[s] / total for s in STATES}


def token_law(prefix, power):
    weights = {t: TABLE[prefix][t] ** power for t in TOKENS}
    total = sum(weights.values(), ZERO)
    return {t: weights[t] / total for t in TOKENS}


def suffix_probability(record, power, restart):
    if restart == 1 and record[0] == "EOS":
        return ONE
    probability = ONE
    prefix = record[:restart]
    for token in record[restart:]:
        if token == "PAD":
            break
        probability *= token_law(prefix, power)[token]
        if token == "EOS":
            break
        prefix += (token,)
    return probability


def compatible(old, new, restart):
    return old[:restart] == new[:restart]


def acceptance(old, new, power, restart):
    if not compatible(old, new, restart):
        return ZERO
    return min(ONE, (base(new) / base(old)) ** power
               * suffix_probability(old, power, restart)
               / suffix_probability(new, power, restart))


def zero_kernel(states):
    return {old: {new: ZERO for new in states} for old in states}


def local_oracle(power, restart=None, broken=False):
    result = zero_kernel(STATES)
    for old in STATES:
        restarts = ((restart,) if restart is not None else
                    range(old.index("EOS") + 1) if broken and "EOS" in old
                    else range(2))
        restarts = tuple(restarts)
        for r in restarts:
            for new in STATES:
                if not compatible(old, new, r):
                    continue
                proposal = suffix_probability(new, power, r) / len(restarts)
                accepted = proposal * acceptance(old, new, power, r)
                result[old][new] += accepted
                result[old][old] += proposal - accepted
    return result


def swap_probability(left, right, left_power, right_power):
    return min(ONE, (base(right) / base(left)) ** (left_power - right_power))


def joint_oracle(powers, exchange_enabled):
    states = tuple(product(STATES, repeat=2))
    result = zero_kernel(states)
    local_left, local_right = (local_oracle(p) for p in powers)
    for old in states:
        for updated in states:
            mass = local_left[old[0]][updated[0]] * local_right[old[1]][updated[1]]
            if not mass:
                continue
            alpha = swap_probability(*updated, *powers) if exchange_enabled else ZERO
            result[old][updated] += mass * (ONE - alpha)
            result[old][updated[::-1]] += mass * alpha
    return result


class ScriptedRandom:
    """Forced integer tickets, with independently expected stop arguments."""

    def __init__(self, draws):
        self.draws = list(draws)
        self.calls = 0

    def randrange(self, stop):
        if self.calls >= len(self.draws):
            raise AssertionError("unexpected extra RNG draw")
        expected_stop, result = self.draws[self.calls]
        self.calls += 1
        if stop != expected_stop:
            raise AssertionError(("unexpected draw denominator", stop, expected_stop))
        if not 0 <= result < stop:
            raise AssertionError(("invalid scripted ticket", result, stop))
        return result


class TinySamplerExactTests(unittest.TestCase):
    def setUp(self):
        self.model = sampler.TinyModel()

    def assert_kernel_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for old, row in actual.items():
            self.assertLessEqual(set(row), set(expected[old]))
            for new, mass in row.items():
                self.assertIsInstance(mass, F)
                self.assertGreaterEqual(mass, ZERO)
            for new, expected_mass in expected[old].items():
                self.assertEqual(row.get(new, ZERO), expected_mass,
                                 (old, new, row.get(new, ZERO), expected_mass))

    def assert_stochastic(self, kernel):
        for old, row in kernel.items():
            self.assertEqual(sum(row.values(), ZERO), ONE, old)
            self.assertTrue(all(isinstance(x, F) and ZERO <= x <= ONE
                                for x in row.values()), old)

    def assert_stationary(self, kernel, target):
        for new in target:
            incoming = sum((target[old] * kernel[old].get(new, ZERO)
                            for old in target), ZERO)
            self.assertEqual(incoming, target[new], (new, incoming, target[new]))

    def assert_detailed_balance(self, kernel, target):
        for old, new in product(target, repeat=2):
            self.assertEqual(target[old] * kernel[old].get(new, ZERO),
                             target[new] * kernel[new].get(old, ZERO), (old, new))

    def test_canonical_complete_records_and_base_probabilities(self):
        self.assertEqual(set(self.model.states), set(STATES))
        self.assertEqual(len(self.model.states), 7)
        expected = (F(1, 10), F(1, 4), F(3, 20),
                    F(1, 6), F(1, 24), F(1, 8), F(1, 6))
        self.assertEqual(tuple(base(s) for s in STATES), expected)
        for state, mass in zip(STATES, expected):
            self.model.validate(state)
            self.assertIsInstance(self.model.probability(state), F)
            self.assertEqual(self.model.probability(state), mass)
        self.assertEqual(sum(expected, ZERO), ONE)
        invalid = ((), ("A",), ("A", "A", "A"), ("PAD", "PAD"),
                   ("A", "PAD"), ("EOS", "A"), ("EOS", "EOS"),
                   ("A", "invalid"))
        for state in invalid:
            with self.subTest(state=state):
                with self.assertRaises((ValueError, TypeError)):
                    self.model.validate(state)
                with self.assertRaises((ValueError, TypeError)):
                    self.model.probability(state)

    def test_power_targets_are_exact_sequence_targets(self):
        for power in (1, 2, 3):
            with self.subTest(power=power):
                self.assertEqual(self.model.target(power), pi(power))
                self.assertEqual(sum(self.model.target(power).values(), ZERO), ONE)
        # Tokenwise powered proposals have prefix-dependent normalizers and
        # therefore do not already sample the complete sequence-level target.
        for power in (2, 3):
            proposals = {s: suffix_probability(s, power, 0) for s in STATES}
            self.assertNotEqual(proposals, pi(power))

    def test_all_suffix_scores_and_uniform_restart_support(self):
        for state in STATES:
            self.assertEqual(self.model.restart_distribution(state), (F(1, 2), F(1, 2)))
        for state, power, restart in product(STATES, (1, 2, 3), (0, 1)):
            with self.subTest(state=state, power=power, restart=restart):
                actual = self.model.proposal_probability(state, power, restart)
                self.assertIsInstance(actual, F)
                self.assertEqual(actual, suffix_probability(state, power, restart))

    def test_invalid_power_and_restart_fail_closed(self):
        state = STATES[0]
        for power in (0, -1, F(3, 2), 2.0, True, None):
            with self.subTest(power=power):
                for operation in (lambda: sampler.TinySampler(self.model, power),
                                  lambda: self.model.target(power),
                                  lambda: self.model.proposal_probability(state, power, 0),
                                  lambda: sampler.make_replica(self.model, power, state),
                                  lambda: sampler.swap_acceptance(self.model, state, state, power, 1)):
                    with self.assertRaises((ValueError, TypeError)):
                        operation()
        chain = sampler.TinySampler(self.model, 2)
        for restart in (-1, 2, 0.5, F(1), True):
            with self.subTest(restart=restart):
                for operation in (lambda: self.model.proposal_probability(state, 2, restart),
                                  lambda: chain.local_acceptance(state, state, restart),
                                  lambda: chain.local_kernel(restart),
                                  lambda: chain.step(state, ScriptedRandom([]), restart)):
                    with self.assertRaises((ValueError, TypeError)):
                        operation()

    def test_all_local_acceptance_ratios(self):
        for power in (1, 2, 3):
            chain = sampler.TinySampler(self.model, power)
            for old, new, restart in product(STATES, STATES, (0, 1)):
                self.assertEqual(chain.local_acceptance(old, new, restart),
                                 acceptance(old, new, power, restart),
                                 (power, old, new, restart))

    def test_every_local_transition_normalization_balance_and_stationarity(self):
        for power, restart in product((1, 2, 3), (None, 0, 1)):
            with self.subTest(power=power, restart=restart):
                actual = sampler.TinySampler(self.model, power).local_kernel(restart)
                self.assert_kernel_equal(actual, local_oracle(power, restart))
                self.assert_stochastic(actual)
                self.assert_detailed_balance(actual, pi(power))
                self.assert_stationary(actual, pi(power))

    def test_eos_can_lengthen_and_shorten_only_when_prefix_is_restarted(self):
        ended = ("EOS", "PAD")
        for power in (1, 2, 3):
            chain = sampler.TinySampler(self.model, power)
            post_eos = chain.local_kernel(1)[ended]
            self.assertEqual(post_eos.get(ended), ONE)
            self.assertTrue(all(mass == ZERO for state, mass in post_eos.items()
                                if state != ended))
            full = chain.local_kernel(0)
            for state in STATES[:-1]:
                self.assertGreater(full[ended].get(state, ZERO), ZERO)
                self.assertGreater(full[state].get(ended, ZERO), ZERO)
            after_a = chain.local_kernel(1)
            self.assertGreater(after_a[("A", "EOS")].get(("A", "A"), ZERO), ZERO)
            self.assertGreater(after_a[("A", "A")].get(("A", "EOS"), ZERO), ZERO)
            self.assertEqual(after_a[("A", "EOS")].get(("B", "A"), ZERO), ZERO)

    def test_rejected_work_contributes_to_self_transition(self):
        old = ("EOS", "PAD")
        power = 2
        rejected_mass = sum((suffix_probability(new, power, 0)
                             * (ONE - acceptance(old, new, power, 0))
                             for new in STATES), ZERO)
        self.assertGreater(rejected_mass, ZERO)
        kernel = sampler.TinySampler(self.model, power).local_kernel(0)
        self.assertEqual(kernel[old][old], suffix_probability(old, power, 0) + rejected_mass)

    def test_sampled_step_all_proposal_boundaries_and_acceptance_branches(self):
        # Force both ends of every integer proposal interval, then the exact
        # accept/reject boundary. This exercises execution independently of
        # the production transition-matrix calculation, without Monte Carlo.
        rejections = 0
        for old, power, restart in product(STATES, (1, 2, 3), (0, 1)):
            chain = sampler.TinySampler(self.model, power)
            proposals = {new: suffix_probability(new, power, restart)
                         for new in STATES if compatible(old, new, restart)}
            denominator = lcm(*(p.denominator for p in proposals.values()))
            lower = 0
            for new, mass in proposals.items():
                upper = lower + mass.numerator * (denominator // mass.denominator)
                alpha = acceptance(old, new, power, restart)
                decisions = [(alpha.numerator - 1, True)]
                if alpha < ONE:
                    decisions.append((alpha.numerator, False))
                for proposal_ticket in set((lower, upper - 1)):
                    for decision_ticket, accepted in decisions:
                        rng = ScriptedRandom(((denominator, proposal_ticket),
                                              (alpha.denominator, decision_ticket)))
                        event = chain.step(old, rng, restart)
                        self.assertEqual(rng.calls, 2)
                        self.assertEqual(event.before, old)
                        self.assertEqual(event.proposal, new)
                        self.assertEqual(event.after, new if accepted else old)
                        self.assertEqual(event.restart, restart)
                        self.assertEqual(event.power, power)
                        self.assertEqual(event.forward, mass)
                        self.assertEqual(event.reverse, suffix_probability(old, power, restart))
                        self.assertEqual(event.acceptance, alpha)
                        self.assertIs(event.accepted, accepted)
                        self.assertEqual(event.token_draws,
                                         sum(token != "PAD" for token in new[restart:]))
                        rejections += not accepted
                lower = upper
            self.assertEqual(lower, denominator)
        self.assertGreater(rejections, 0)

    def test_random_restart_post_eos_and_seeded_replay(self):
        ended = ("EOS", "PAD")
        chain = sampler.TinySampler(self.model, 3)
        rng = ScriptedRandom(((2, 1), (1, 0), (1, 0)))
        event = chain.step(ended, rng)
        self.assertEqual(rng.calls, 3)
        self.assertEqual(event.restart, 1)
        self.assertEqual(event.proposal, ended)
        self.assertEqual(event.after, ended)
        self.assertEqual(event.forward, ONE)
        self.assertEqual(event.reverse, ONE)
        self.assertEqual(event.acceptance, ONE)
        self.assertTrue(event.accepted)
        self.assertEqual(event.token_draws, 0)

        def trace(seed):
            rng = random.Random(seed)
            record = ended
            result = []
            for _ in range(24):
                event = chain.step(record, rng)
                result.append(event)
                record = event.after
            return result

        self.assertEqual(trace(173), trace(173))
        self.assertNotEqual(trace(173), trace(174))

    def test_swap_orientation_and_detailed_balance(self):
        joint_states = tuple(product(STATES, repeat=2))
        for powers in ((1, 2), (2, 3), (3, 1)):
            target = {s: pi(powers[0])[s[0]] * pi(powers[1])[s[1]]
                      for s in joint_states}
            swaps = zero_kernel(joint_states)
            for left, right in joint_states:
                actual = sampler.swap_acceptance(self.model, left, right, *powers)
                expected = swap_probability(left, right, *powers)
                self.assertEqual(actual, expected, (left, right, powers))
                swaps[(left, right)][(right, left)] += actual
                swaps[(left, right)][(left, right)] += ONE - actual
            self.assert_stochastic(swaps)
            self.assert_detailed_balance(swaps, target)
            self.assert_stationary(swaps, target)
        self.assertEqual(sampler.swap_acceptance(self.model, ("A", "B"),
                                                ("B", "B"), 1, 3), ONE)
        self.assertEqual(sampler.swap_acceptance(self.model, ("B", "B"),
                                                ("A", "B"), 1, 3), F(1, 36))

    def test_exchange_execution_recomputes_receiver_power_scores(self):
        # q_power differs at the two receivers even though the record moves.
        left_record, right_record = ("B", "B"), ("A", "B")
        left = sampler.make_replica(self.model, 1, left_record)
        right = sampler.make_replica(self.model, 3, right_record)
        for replica in (left, right):
            self.assertEqual(replica.base_probability, base(replica.record))
            self.assertEqual(replica.proposal_probability,
                             suffix_probability(replica.record, replica.power, 0))
        alpha = swap_probability(left_record, right_record, 1, 3)
        self.assertEqual(alpha, F(1, 36))
        for ticket, accepted in ((0, True), (1, False), (35, False)):
            event = sampler.exchange(self.model, left, right, ScriptedRandom(((36, ticket),)))
            self.assertEqual(event.before, (left, right))
            self.assertEqual(event.acceptance, alpha)
            self.assertIs(event.accepted, accepted)
            expected_records = (right_record, left_record) if accepted else (left_record, right_record)
            for replica, record, power in zip(event.after, expected_records, (1, 3)):
                self.assertEqual(replica.record, record)
                self.assertEqual(replica.power, power)
                self.assertEqual(replica.base_probability, base(record))
                self.assertEqual(replica.proposal_probability, suffix_probability(record, power, 0))
            if accepted:
                self.assertNotEqual(event.after[0].proposal_probability, right.proposal_probability)
                self.assertNotEqual(event.after[1].proposal_probability, left.proposal_probability)
            else:
                self.assertEqual(event.after, event.before)
        certain = sampler.exchange(self.model,
                                   sampler.make_replica(self.model, 1, right_record),
                                   sampler.make_replica(self.model, 3, left_record),
                                   ScriptedRandom(((1, 0),)))
        self.assertEqual(certain.acceptance, ONE)
        self.assertTrue(certain.accepted)

    def test_stale_replica_scores_refuse_exchange_before_any_draw(self):
        left = sampler.make_replica(self.model, 1, ("B", "B"))
        right = sampler.make_replica(self.model, 3, ("A", "B"))
        invalid_left = (
            sampler.Replica(left.power, left.record, left.base_probability + F(1, 10),
                            left.proposal_probability),
            sampler.Replica(left.power, left.record, left.base_probability,
                            right.proposal_probability),
        )
        for invalid in invalid_left:
            for pair in ((invalid, right), (right, invalid)):
                with self.assertRaises(ValueError):
                    sampler.exchange(self.model, *pair, ScriptedRandom([]))

    def test_all_joint_transitions_and_stationarity_after_composition(self):
        for powers, enabled in product(((1, 2), (2, 3), (3, 1)), (False, True)):
            with self.subTest(powers=powers, exchange_enabled=enabled):
                actual = sampler.joint_kernel(self.model, powers, exchange_enabled=enabled)
                self.assert_kernel_equal(actual, joint_oracle(powers, enabled))
                self.assert_stochastic(actual)
                target = {s: pi(powers[0])[s[0]] * pi(powers[1])[s[1]]
                          for s in actual}
                self.assert_stationary(actual, target)
                # Composed invariant kernels need not obey detailed balance.
                # Qualification checks stationarity rather than imposing it.

    def test_finite_initialized_chain_is_not_automatically_stationary(self):
        initial = ("EOS", "PAD")
        kernel = sampler.TinySampler(self.model, 2).local_kernel()
        first_step = {s: kernel[initial].get(s, ZERO) for s in STATES}
        self.assertNotEqual(first_step, pi(2))
        self.assertGreater(sum((abs(first_step[s] - pi(2)[s]) for s in STATES), ZERO), ZERO)

    def test_deliberately_broken_truncation_is_rejected_by_exact_qualification(self):
        for power in (1, 2, 3):
            with self.subTest(power=power):
                broken = sampler.broken_truncation_kernel(self.model, power)
                self.assert_kernel_equal(broken, local_oracle(power, broken=True))
                self.assert_stochastic(broken)
                with self.assertRaises(AssertionError):
                    self.assert_stationary(broken, pi(power))
                with self.assertRaises(AssertionError):
                    self.assert_detailed_balance(broken, pi(power))


if __name__ == "__main__":
    unittest.main()
