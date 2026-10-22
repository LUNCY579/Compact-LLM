"""Evidence guards and non-vacuous preservation decisions, using CPU fixtures."""

from __future__ import annotations

import copy
import unittest

from experiments.rineya_quality_queue_v1.contracts import (
    REQUIRED_GATES,
    preservation_decision,
    validate_metric,
)


class MetricContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifications = {
            "historical-result": "historical",
            "native-receipt": "native",
            "dev-result": "exploratory",
            "confirmation-result": "confirmed",
            "toy-test": "fixture",
            "prepared-plan": "preparation",
        }
        self.metric = {
            "name": "source_correct_retained",
            "status": "MEASURED",
            "numerator": 41,
            "denominator": 47,
            "unit": "count",
            "evidence_id": "historical-result",
            "text_anchor": "41 of 47 source successes retained",
        }

    def assert_invalid(self, **changes) -> None:
        metric = {**self.metric, **changes}
        with self.assertRaises(ValueError):
            validate_metric(metric, self.classifications)

    def test_each_admitted_evidence_scope_requires_an_actual_assertion(self) -> None:
        for evidence_id in ("historical-result", "native-receipt", "dev-result", "confirmation-result"):
            with self.subTest(evidence_id=evidence_id):
                validate_metric({**self.metric, "evidence_id": evidence_id}, self.classifications)
        metric = {key: value for key, value in self.metric.items() if key != "text_anchor"}
        metric["json_pointer"] = "/retention/retained"
        validate_metric(metric, self.classifications)
        del metric["json_pointer"]
        with self.assertRaises(ValueError):
            validate_metric(metric, self.classifications)

    def test_fixture_and_preparation_gains_are_rejected(self) -> None:
        for evidence_id in ("toy-test", "prepared-plan"):
            with self.subTest(evidence_id=evidence_id):
                self.assert_invalid(evidence_id=evidence_id)

    def test_unknown_measurement_is_explicitly_null_never_zero(self) -> None:
        metric = {"name": "semantic_abstention", "status": "NOT_MEASURED", "numerator": None, "denominator": None}
        validate_metric(metric, self.classifications)
        for numerator, denominator in ((0, None), (None, 0), (0, 0), (0, 64), (None, 64)):
            with self.subTest(numerator=numerator, denominator=denominator):
                with self.assertRaises(ValueError):
                    validate_metric({**metric, "numerator": numerator, "denominator": denominator}, self.classifications)
        for key in ("numerator", "denominator"):
            missing = dict(metric)
            del missing[key]
            with self.assertRaises(ValueError):
                validate_metric(missing, self.classifications)

    def test_zero_known_hits_differ_from_unmeasured_hits(self) -> None:
        validate_metric({**self.metric, "numerator": 0, "denominator": 64}, self.classifications)
        self.assert_invalid(denominator=0)

    def test_nonfinite_boolean_negative_and_missing_numbers_are_rejected(self) -> None:
        for field in ("numerator", "denominator"):
            for value in (float("nan"), float("inf"), float("-inf"), True, False, -1, None, "47"):
                with self.subTest(field=field, value=value):
                    self.assert_invalid(**{field: value})
            missing = dict(self.metric)
            del missing[field]
            with self.assertRaises(ValueError):
                validate_metric(missing, self.classifications)

    def test_count_cannot_exceed_its_denominator(self) -> None:
        self.assert_invalid(numerator=48)
        for unit in ("rate", "tokens", "seconds"):
            validate_metric({**self.metric, "unit": unit, "numerator": 48}, self.classifications)

    def test_missing_or_unknown_evidence_and_malformed_identity_are_rejected(self) -> None:
        for field, value in (("evidence_id", "unknown"), ("evidence_id", " "), ("name", ""), ("name", 1), ("status", "PASS"), ("unit", "percent"), ("unit", [])):
            with self.subTest(field=field, value=value):
                self.assert_invalid(**{field: value})
        for value in (None, [], 1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_metric(value, self.classifications)
        with self.assertRaises(ValueError):
            validate_metric(self.metric, {"historical-result": []})

    def test_denominator_source_is_independently_admissible_and_locatable(self) -> None:
        validate_metric(
            {**self.metric, "denominator_evidence_id": "native-receipt", "denominator_json_pointer": "/source_correct/count"},
            self.classifications,
        )
        validate_metric({**self.metric, "denominator_text_anchor": "47 source-correct rows"}, self.classifications)
        for changes in ({"denominator_evidence_id": "toy-test"}, {"denominator_evidence_id": "unknown"}, {"denominator_json_pointer": ""}, {"denominator_text_anchor": None}):
            with self.subTest(changes=changes):
                self.assert_invalid(**changes)

    def test_unrelated_optional_properties_are_retained_without_mutation(self) -> None:
        metric = {**self.metric, "limitations": {"only_arithmetic": True}, "source": "pinned historical source"}
        before = copy.deepcopy(metric)
        validate_metric(metric, self.classifications)
        self.assertEqual(metric, before)


class PreservationDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gates = [{"id": gate_id, "status": "PASS"} for gate_id in REQUIRED_GATES]

    def assert_permissions_closed(self, result) -> None:
        for flag in ("native_execution_authorized", "release_permitted", "promotion_permitted"):
            self.assertIs(result[flag], False)

    def test_all_required_passes_only_enable_separate_review(self) -> None:
        before = copy.deepcopy(self.gates)
        result = preservation_decision(self.gates)
        self.assertEqual(result["status"], "ELIGIBLE_FOR_SEPARATE_REVIEW")
        self.assertEqual(result["blocking_gates"], [])
        self.assertEqual(result["missing_gates"], [])
        self.assert_permissions_closed(result)
        self.assertEqual(self.gates, before)

    def test_empty_gates_cannot_pass_vacuously(self) -> None:
        result = preservation_decision([])
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["missing_gates"], list(REQUIRED_GATES))
        self.assert_permissions_closed(result)

    def test_missing_each_required_gate_holds_the_decision(self) -> None:
        for missing in REQUIRED_GATES:
            with self.subTest(missing=missing):
                result = preservation_decision([gate for gate in self.gates if gate["id"] != missing])
                self.assertEqual(result["status"], "HOLD")
                self.assertEqual(result["missing_gates"], [missing])

    def test_acquisition_cannot_offset_retention_or_other_failures(self) -> None:
        for gate_id in REQUIRED_GATES:
            for status in ("FAIL", "BLOCKED", "NOT_MEASURED"):
                with self.subTest(gate_id=gate_id, status=status):
                    gates = [{**gate, "status": status if gate["id"] == gate_id else "PASS"} for gate in self.gates]
                    result = preservation_decision(gates)
                    self.assertEqual(result["status"], "HOLD")
                    self.assertEqual(result["blocking_gates"], [gate_id])
                    self.assert_permissions_closed(result)

    def test_every_registered_extra_gate_is_mandatory(self) -> None:
        for status in ("FAIL", "BLOCKED", "NOT_MEASURED"):
            with self.subTest(status=status):
                result = preservation_decision(self.gates + [{"id": "long_context", "status": status}])
                self.assertEqual(result["status"], "HOLD")
                self.assertEqual(result["blocking_gates"], ["long_context"])
        result = preservation_decision(self.gates + [{"id": "long_context", "status": "PASS"}])
        self.assertEqual(result["status"], "ELIGIBLE_FOR_SEPARATE_REVIEW")
        self.assert_permissions_closed(result)

    def test_duplicate_gate_cannot_pool_or_replace_a_failure(self) -> None:
        for status in ("PASS", "FAIL"):
            with self.subTest(status=status):
                with self.assertRaises(ValueError):
                    preservation_decision(self.gates + [{"id": "retention", "status": status}])

    def test_invalid_gate_structures_and_statuses_are_rejected(self) -> None:
        for invalid in ({"id": "retention", "status": "UNKNOWN"}, {"id": "retention", "status": []}, {"id": "", "status": "PASS"}, {"id": 3, "status": "PASS"}, {"id": "retention"}, None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    preservation_decision([invalid])
        for invalid in (None, {}, tuple(self.gates)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    preservation_decision(invalid)


if __name__ == "__main__":
    unittest.main()
