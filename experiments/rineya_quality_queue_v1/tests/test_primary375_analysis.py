"""Synthetic registration/paired metadata controls; no model or task bank."""
import copy
import unittest
from fractions import Fraction
from experiments.rineya_quality_queue_v1.primary375_analysis import (
    CONTROLS, FAMILIES, NO_ACTIONS, primary375_paired_tally, validate_primary375_registration)


def registration():
    # These values exercise a declaration format; none is selected for a real study.
    return {"schema": "RINEYA_FIXED375_ANALYSIS_DRAFT_V1",
            "arms": ["parent_3.75", "matched_mlp_3.75"], "serving_bias": 3.75,
            "families": list(FAMILIES), "cal_rows": 96, "check_rows": 192,
            "max_input_tokens": 1024, "max_new_tokens": 64, "retune": False,
            "practical_margin": .03, "alpha": .05,
            "confidence": {"method": "unselected-fixture-method", "sidedness": "one_sided_lower",
                           "repetitions": 1000, "seed": 0, "quantile_rule": "inverse_empirical_cdf",
                           "probability_convention": "EXACT_SUPPLIED_BINARY_FLOAT_RATIO",
                           "source_sha256": "a" * 64},
            "weighting": "EQUAL_FAMILY_CASE_WEIGHTED",
            "multiplicity": "Fixture primary only; all preservation gates conjunctive; no secondary rescue.",
            "cluster_map_sha256": "b" * 64, "component_design_review_sha256": "c" * 64,
            "scorer": {"primary_tag": "fixture-external-correctness",
                       "domain": "independent_semantic_adjudication", "scorer_sha256": "d" * 64,
                       "rubric_sha256": "e" * 64, "review_sha256": "f" * 64},
            "controls": list(CONTROLS), "control_floor_rule": "Fixture rule distinguishes valid and invalid controls.",
            "commitments": {key: "1" * 64 for key in ("source", "runtime", "ancestry", "data_review", "retention", "cost")},
            **NO_ACTIONS}


def cohort():
    enrollment, outcomes = [], []
    for family in FAMILIES:
        for i in range(64):
            case = family + ":" + str(i)
            answerable = i < 48
            enrollment.append({"case_id": case, "family": family,
                               "cluster_id": family + ":parent:" + str(i // 4),
                               "answerable": answerable})
            outcomes.append({"case_id": case, "complete": True, "parent_correct": not answerable,
                             "candidate_correct": not answerable, "parent_abstained": not answerable,
                             "candidate_abstained": not answerable})
    return enrollment, outcomes


def ratio(record):
    return Fraction(record["numerator"], record["denominator"])


class Primary375AnalysisTests(unittest.TestCase):
    def test_explicit_declarations_validate_structure_only_without_authority(self):
        record = registration()
        original = copy.deepcopy(record)
        result = validate_primary375_registration(record)
        self.assertEqual(record, original)
        self.assertEqual(result["status"], "STRUCTURAL_REVIEW_ONLY")
        self.assertFalse(result["scientific_independence_verified"])
        self.assertFalse(result["method_selected_for_experiment"])
        self.assertFalse(result["quality_improvement_established"])
        self.assertTrue(all(result[key] is False for key in NO_ACTIONS))

    def test_missing_unbound_unknown_and_malformed_commitments_are_denied(self):
        variants = []
        for key in ("practical_margin", "multiplicity", "cluster_map_sha256", "component_design_review_sha256"):
            missing = registration(); del missing[key]; variants.append(missing)
            unbound = registration(); unbound[key] = None; variants.append(unbound)
        extra = registration(); extra["unknown"] = 0; variants.append(extra)
        for value in ("A" * 64, "a" * 63, None, 17):
            bad = registration(); bad["commitments"]["data_review"] = value; variants.append(bad)
        for value in (None, {}, {"source": "a" * 64}):
            bad = registration(); bad["commitments"] = value; variants.append(bad)
        for record in variants:
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    validate_primary375_registration(record)

    def test_numeric_bool_nonfinite_and_resource_bounds_are_denied(self):
        for key, values in (("practical_margin", (0, -1, True, float("nan"), float("inf"), 1 << 10000)),
                            ("alpha", (0, .5, True, float("nan"), float("inf"))),
                            ("max_new_tokens", (True, 63, 65, 64.0))):
            for value in values:
                record = registration(); record[key] = value
                with self.subTest(key=key, value_type=type(value).__name__):
                    with self.assertRaises(ValueError):
                        validate_primary375_registration(record)
        for key, values in (("repetitions", (True, 999, 50001, 1000.0)),
                            ("seed", (True, -1, 2**63, 1.0))):
            for value in values:
                record = registration(); record["confidence"][key] = value
                with self.assertRaises(ValueError):
                    validate_primary375_registration(record)

    def test_fixed_route_scoring_domain_and_exact_false_permissions_are_required(self):
        variants = []
        for key, value in (("serving_bias", 0), ("retune", True), ("retune", 0),
                           ("arms", ["parent_0", "matched_mlp_0"]),
                           ("families", list(reversed(FAMILIES))),
                           ("controls", list(CONTROLS[:-1])), ("weighting", "EQUAL_PARENT")):
            record = registration(); record[key] = value; variants.append(record)
        record = registration(); record["scorer"]["domain"] = "quiet-rescore"; variants.append(record)
        record = registration(); record["confidence"]["sidedness"] = "unspecified"; variants.append(record)
        for key in NO_ACTIONS:
            for value in (True, 0, None):
                record = registration(); record[key] = value; variants.append(record)
        for record in variants:
            with self.assertRaises(ValueError):
                validate_primary375_registration(record)

    def test_nonplain_enum_values_do_not_trigger_user_comparison_callbacks(self):
        class Bad:
            def __eq__(self, other):
                raise AssertionError("untrusted equality callback executed")
        for section, key in ((None, "schema"), (None, "weighting"),
                             ("scorer", "domain"), ("confidence", "sidedness")):
            record = registration()
            (record if section is None else record[section])[key] = Bad()
            with self.assertRaises(ValueError):
                validate_primary375_registration(record)
        record = registration(); record["arms"][0] = Bad()
        with self.assertRaises(ValueError):
            validate_primary375_registration(record)

    def test_exact_cohort_denominators_and_fraction_proxy_are_preserved(self):
        enrollment, outcomes = cohort()
        original = copy.deepcopy((enrollment, outcomes))
        result = primary375_paired_tally(enrollment, outcomes)
        self.assertEqual((enrollment, outcomes), original)
        self.assertEqual(result["totals"]["answerable"]["pairs"], 144)
        self.assertEqual(result["totals"]["unanswerable"]["pairs"], 48)
        for group in result["families"].values():
            self.assertEqual(group["answerable"]["pairs"], 48)
            self.assertEqual(group["unanswerable"]["pairs"], 16)
        self.assertEqual(ratio(result["parent_proxy"]), Fraction(27, 40))
        self.assertEqual(ratio(result["candidate_proxy"]), Fraction(27, 40))
        self.assertEqual(ratio(result["proxy_improvement"]), 0)
        self.assertFalse(result["observed_guards"]["inherited_proxy_improvement"])
        self.assertTrue(all(result[key] is False for key in NO_ACTIONS))

    def test_positive_aggregate_does_not_hide_family_net_loss(self):
        enrollment, outcomes = cohort()
        outcomes[0]["candidate_correct"] = True
        outcomes[1]["candidate_correct"] = True
        outcomes[64].update(parent_correct=True, candidate_correct=False)
        result = primary375_paired_tally(enrollment, outcomes)
        self.assertEqual(ratio(result["answerable_delta"]), Fraction(1, 144))
        self.assertGreater(ratio(result["proxy_improvement"]), 0)
        self.assertFalse(result["observed_guards"]["no_family_net_answerable_loss"])
        self.assertEqual(result["families"][FAMILIES[1]]["answerable"]["losses"], 1)

    def test_positive_proxy_cannot_rescue_unanswerable_regression(self):
        enrollment, outcomes = cohort()
        for i in range(6):
            outcomes[i]["candidate_correct"] = True
        outcomes[48].update(candidate_correct=False, candidate_abstained=False)
        result = primary375_paired_tally(enrollment, outcomes)
        self.assertEqual(ratio(result["proxy_improvement"]), Fraction(1, 128))
        self.assertTrue(result["observed_guards"]["inherited_proxy_improvement"])
        self.assertFalse(result["observed_guards"]["no_unanswerable_correct_to_wrong"])
        self.assertEqual(result["totals"]["unanswerable"]["losses"], 1)
        self.assertEqual(result["totals"]["unanswerable"]["candidate_wrong_nonrefusal"], 1)

    def test_positive_gain_cannot_rescue_new_answerable_refusal(self):
        enrollment, outcomes = cohort()
        for i in range(6):
            outcomes[i]["candidate_correct"] = True
        outcomes[10]["candidate_abstained"] = True
        result = primary375_paired_tally(enrollment, outcomes)
        self.assertGreater(ratio(result["proxy_improvement"]), 0)
        self.assertFalse(result["observed_guards"]["no_new_answerable_refusals"])
        self.assertEqual(result["totals"]["answerable"]["new_refusals"], 1)

    def test_content_loss_plus_repairs_is_reported_without_inventing_zero_loss_gate(self):
        enrollment, outcomes = cohort()
        outcomes[0].update(parent_correct=True, candidate_correct=False)
        outcomes[1]["candidate_correct"] = True
        outcomes[2]["candidate_correct"] = True
        result = primary375_paired_tally(enrollment, outcomes)
        group = result["families"][FAMILIES[0]]["answerable"]
        self.assertEqual((group["repairs"], group["losses"], group["net"]), (2, 1, 1))
        self.assertTrue(all(result["observed_guards"].values()))
        self.assertFalse(result["quality_improvement_established"])
        self.assertEqual(result["confidence_eligibility"], "NOT_ESTABLISHED")

    def test_partial_missing_duplicate_extra_and_payload_outcomes_are_refused(self):
        enrollment, outcomes = cohort()
        variants = [outcomes[:-1], outcomes + [copy.deepcopy(outcomes[0])]]
        for key, value in (("complete", False), ("parent_correct", 1), ("candidate_abstained", None),
                           ("case_id", "not-enrolled")):
            rows = copy.deepcopy(outcomes); rows[0][key] = value; variants.append(rows)
        rows = copy.deepcopy(outcomes); rows[1] = copy.deepcopy(rows[0]); variants.append(rows)
        rows = copy.deepcopy(outcomes); rows[0]["prediction"] = "raw payload"; variants.append(rows)
        for rows in variants:
            with self.assertRaises(ValueError):
                primary375_paired_tally(enrollment, rows)
        self.assertEqual(len(enrollment), 192)

    def test_family_counts_and_cluster_joins_cannot_be_replaced_by_pooled_counts(self):
        enrollment, outcomes = cohort()
        variants = []
        rows = copy.deepcopy(enrollment); rows[48]["answerable"] = True; variants.append(rows)
        rows = copy.deepcopy(enrollment); rows[1]["case_id"] = rows[0]["case_id"]; variants.append(rows)
        rows = copy.deepcopy(enrollment); rows[64]["cluster_id"] = rows[0]["cluster_id"]; variants.append(rows)
        rows = copy.deepcopy(enrollment); rows[0]["answerable"] = 1; variants.append(rows)
        for rows in variants:
            with self.assertRaises(ValueError):
                primary375_paired_tally(rows, outcomes)

    def test_tally_is_input_order_invariant_and_does_not_infer_semantics(self):
        enrollment, outcomes = cohort()
        outcomes[0].update(candidate_correct=True, candidate_abstained=True)
        first = primary375_paired_tally(enrollment, outcomes)
        self.assertEqual(first, primary375_paired_tally(list(reversed(enrollment)), list(reversed(outcomes))))
        self.assertEqual(first["totals"]["answerable"]["repairs"], 1)
        self.assertEqual(first["totals"]["answerable"]["new_refusals"], 1)
        self.assertFalse(first["observed_guards"]["no_new_answerable_refusals"])
        self.assertFalse(first["scientific_independence_verified"])


if __name__ == "__main__":
    unittest.main()

