"""Synthetic grammar/semantic controls, never a Rineya capability evaluation."""
import copy
import itertools
import unittest
from unittest.mock import patch

from experiments.rineya_quality_queue_v1.r_lookup_diagnostic import analyze_atomic_r_fixture


def fixture(links, requested, n=6):
    return {"initial": [0] * n, "lookup": {"links": links, "requested": requested}}


class RLookupDiagnosticTests(unittest.TestCase):
    def test_hand_checked_semantic_cases(self):
        cases = [
            ({"a": 0, "b": 1}, ["a", "b"], [0, 1]),
            ({"a": 0, "b": 0}, ["a", "b"], [0]),
            ({"a": 0}, ["a", "a"], [0]),
            ({"a": "c", "b": "c", "c": 0, "d": 1}, ["a", "b", "d"], [0, 1]),
            ({"a": "a"}, ["a"], "UNKNOWN"),
            ({"a": "b", "b": "a"}, ["a"], "UNKNOWN"),
            ({"a": "b", "b": "c", "c": "b"}, ["a"], "UNKNOWN"),
            ({"a": 0}, ["missing"], "UNKNOWN"),
            ({"a": "missing"}, ["a"], "UNKNOWN"),
            ({"a": 0, "b": "missing"}, ["a", "b"], "UNKNOWN"),
            ({"a": 0, "x": "y", "y": "x"}, ["a"], [0]),
            ({"a": 0, "x": "missing"}, ["a"], [0]),
        ]
        for links, requested, expected in cases:
            with self.subTest(links=links, requested=requested):
                result = analyze_atomic_r_fixture(fixture(links, requested))
                self.assertEqual(result["fixture_reference"], {"answer": expected})
                self.assertTrue(result["routes_agree"])
                self.assertFalse(result["model_quality_evidence"])

    def test_valid_duplicate_requests_distinct_from_shared_endpoints(self):
        repeated = analyze_atomic_r_fixture(fixture({"a": 0}, ["a", "a"]))
        shared = analyze_atomic_r_fixture(fixture({"a": 0, "b": 0}, ["a", "b"]))
        self.assertIn("REPEATED_REQUEST_STRINGS", repeated["coverage_labels"])
        self.assertNotIn("DISTINCT_ALIASES_SHARED_ENDPOINT", repeated["coverage_labels"])
        self.assertIn("DISTINCT_ALIASES_SHARED_ENDPOINT", shared["coverage_labels"])
        self.assertNotIn("REPEATED_REQUEST_STRINGS", shared["coverage_labels"])
        self.assertNotIn("FULLY_RESOLVED_DISTINCT_ENDPOINTS", shared["coverage_labels"])

    def test_unknown_never_returns_partial_success_as_answer(self):
        result = analyze_atomic_r_fixture(fixture({"a": 0, "b": "missing"}, ["a", "b"]))
        self.assertEqual(result["fixture_reference"], {"answer": "UNKNOWN"})
        self.assertEqual(result["resolved_endpoint_count"], 1)
        self.assertFalse(result["partial_endpoints_are_an_answer"])

    def test_requested_cycle_and_dangling_causes_are_kept_separate(self):
        result = analyze_atomic_r_fixture(fixture({"a": "a", "b": "missing"}, ["a", "b"]))
        self.assertEqual(result["requested_causes_from_traversal"], ["CYCLE", "DANGLING"])

    def test_irrelevant_components_cannot_force_unknown(self):
        result = analyze_atomic_r_fixture(fixture({"a": 0, "x": "y", "y": "x", "z": "missing"}, ["a"]))
        self.assertEqual(result["fixture_reference"], {"answer": [0]})
        self.assertEqual(result["irrelevant_causes_from_traversal"], ["CYCLE", "DANGLING"])

    def test_unrequested_feeder_does_not_double_label_requested_failure(self):
        for links, cause in (({"a": "a", "x": "a"}, "CYCLE"),
                             ({"a": "missing", "x": "a"}, "DANGLING"),
                             ({"a": "missing", "x": "missing"}, "DANGLING")):
            result = analyze_atomic_r_fixture(fixture(links, ["a"]))
            self.assertEqual(result["requested_causes_from_traversal"], [cause])
            self.assertEqual(result["irrelevant_causes_from_traversal"], [])

    def test_separate_failure_component_remains_irrelevant(self):
        result = analyze_atomic_r_fixture(fixture({"a": "a", "x": "y", "y": "x", "z": "other"}, ["a"]))
        self.assertEqual(result["requested_causes_from_traversal"], ["CYCLE"])
        self.assertEqual(result["irrelevant_causes_from_traversal"], ["CYCLE", "DANGLING"])

    def test_maximum_chain_no_recursion(self):
        links = {"a" + str(i): "a" + str(i + 1) for i in range(79)}
        links["a79"] = 5
        result = analyze_atomic_r_fixture(fixture(links, ["a0"]))
        self.assertEqual(result["fixture_reference"], {"answer": [5]})
        self.assertEqual(result["logical_work"]["request_traversal_edges"], 80)
        self.assertLessEqual(result["logical_work"]["closure_rounds"], 80)

    def test_maximum_request_domain_and_index_zero(self):
        links = {"a" + str(i): i for i in range(15)}
        result = analyze_atomic_r_fixture(fixture(links, list(links), n=15))
        self.assertEqual(result["fixture_reference"], {"answer": list(range(15))})

    def test_input_is_not_mutated_and_source_values_do_not_affect_R(self):
        original = fixture({"a": "c", "b": "c", "c": 0, "d": 1}, ["a", "b", "d"])
        snapshot = copy.deepcopy(original)
        baseline = analyze_atomic_r_fixture(original)
        self.assertEqual(original, snapshot)
        changed = copy.deepcopy(original)
        changed["initial"] = [-100, -3, 0, 17, 99, 100]
        self.assertEqual(analyze_atomic_r_fixture(changed), baseline)

    def test_order_renaming_repetition_and_irrelevant_extension_invariance(self):
        base = fixture({"a": "c", "b": "c", "c": 0, "d": 1}, ["a", "b", "d"])
        expected = analyze_atomic_r_fixture(base)["fixture_reference"]
        variants = [
            fixture(dict(reversed(list(base["lookup"]["links"].items()))), ["d", "b", "a"]),
            fixture({"p": "r", "q": "r", "r": 0, "s": 1}, ["p", "q", "s"]),
            fixture(base["lookup"]["links"], ["a", "b", "d", "a"]),
            fixture(dict(base["lookup"]["links"], x="x", y="missing"), ["a", "b", "d"]),
        ]
        for item in variants:
            self.assertEqual(analyze_atomic_r_fixture(item)["fixture_reference"], expected)

    def test_bounded_exhaustive_two_alias_graph_agreement(self):
        # All25 labeled graphs over this explicit5-target alphabet, and12
        # request sequences. This is300 tiny synthetic cases, not bank parents.
        count = 0
        for targets in itertools.product((0, 1, "a", "b", "missing"), repeat=2):
            for length in (1, 2):
                for requests in itertools.product(("a", "b", "missing"), repeat=length):
                    result = analyze_atomic_r_fixture(fixture(dict(zip(("a", "b"), targets)), list(requests)))
                    self.assertTrue(result["routes_agree"])
                    count += 1
        self.assertEqual(count, 300)

    def test_malformed_and_out_of_scope_are_refusals_not_unknown(self):
        invalid = [
            {}, fixture({}, ["a"]), fixture({"a": 0}, []),
            fixture({"a": 0}, ["a"] * 7), fixture({"a": 0}, ["a"], n=5),
            fixture({"a": 0}, ["a"], n=16), fixture({"A": 0}, ["A"]),
            fixture({"a" * 21: 0}, ["a" * 21]), fixture({"a": True}, ["a"]),
            fixture({"a": -1}, ["a"]), fixture({"a": 6}, ["a"]),
            fixture({"a": None}, ["a"]), fixture({"a": 0}, [1]),
            fixture({"a" + str(i): 0 for i in range(81)}, ["a0"]),
            {"initial": [0] * 6, "lookup": {"links": {"a": 0}, "requested": ["a"], "extra": 1}},
            dict(fixture({"a": 0}, ["a"]), updates={}),
            {"initial": [False] * 6, "lookup": {"links": {"a": 0}, "requested": ["a"]}},
            {"extra" + str(i): 0 for i in range(1000)},
            {"initial": [0] * 6, "lookup": {"extra" + str(i): 0 for i in range(1000)}},
        ]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                analyze_atomic_r_fixture(item)

    def test_operational_failure_or_route_disagreement_is_not_unknown(self):
        value = fixture({"a": 0}, ["a"])
        module = "experiments.rineya_quality_queue_v1.r_lookup_diagnostic"
        with patch(module + "._closure_endpoints", side_effect=RuntimeError("stopped")):
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                analyze_atomic_r_fixture(value)
        with patch(module + "._closure_endpoints", return_value=([1], 1)):
            with self.assertRaisesRegex(RuntimeError, "routes disagree"):
                analyze_atomic_r_fixture(value)


if __name__ == "__main__":
    unittest.main()
