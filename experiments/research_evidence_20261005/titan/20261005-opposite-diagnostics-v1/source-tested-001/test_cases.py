"""Tiny input-only fixtures; never Rineya capability or model-outcome evidence."""
from __future__ import annotations

from collections import Counter
import copy
import unittest

from . import cases


def row(index, family="ledger", style="canonical", sign="positive", branch=None,
        answerable=True, expected=None, scenario="supported", pair=None):
    value = 3 if sign == "positive" else -3
    payload = {"expected_kind": "object"}
    if family == "ledger":
        payload.update(before={"A": 2, "B": 1}, event={"kind": "add", "amount": value, "account": "A", "other": "B"})
        expected = {"A": 2 + value, "B": 1}
    elif family == "register":
        payload.update(before={"X": 3, "Y": 10}, event={"kind": "conditional", "value": value,
            "register": "X", "other": "Y", "threshold": 9 if branch else 10})
        expected = {"X": (3 + (value if branch else -value)) % 97, "Y": 10}
    else:
        payload = {"expected_kind": "integer", "facts": [], "question": {"subject": "fixture", "field": "quantity"}}
        expected = (value if expected is None else expected) if answerable else None
    return {"row_id": f"fixture-{index}", "world_id": "world-" + str(pair if pair is not None else index),
            "pair_id": None if pair is None else "pair-" + str(pair), "family": family,
            "variant_id": style, "split": "train", "operative_key": "operative-" + str(index),
            "payload": payload, "strata": {"operand_sign": sign, "scenario": scenario},
            "answerable": answerable, "expected": expected, "user": "fixture prompt"}


class CaseAuditTests(unittest.TestCase):
    def test_hard_ledger_style_sign_confound_and_explicit_zero_cells(self):
        rows = [row(i, style=cases.STYLES[i % 4], sign="positive" if i % 2 == 0 else "negative", pair=i // 2) for i in range(32)]
        result = cases.summarize_rows(rows, {"prompt_coverage": "diverse"})
        observed = {(c["style"], c["operand_sign"]): c["count"] for c in result["transition_joint_cases"]}
        self.assertEqual(observed, {("canonical", "positive"): 8, ("native_a", "negative"): 8,
                                   ("native_b", "positive"): 8, ("natural", "negative"): 8})
        absent = {(z["style"], z["missing"]) for z in result["explicit_zero_cells"] if z["dimension"] == "operand_sign"}
        self.assertIn(("canonical", "negative"), absent)
        self.assertIn(("natural", "positive"), absent)
        self.assertEqual(result["hard_pair_structure"][0]["distinct_operative_keys"], 2)
        self.assertEqual(result["hard_pair_structure"][0]["distinct_world_ids"], 1)
        self.assertEqual(result["native_effects"], "UNMEASURED")

    def test_conditional_margin_and_joint_cells_reveal_boundary_only_cases(self):
        rows = [row(i, family="register", style=cases.STYLES[i % 4], sign="positive" if i % 4 < 2 else "negative", branch=i % 2 == 0) for i in range(32)]
        result = cases.summarize_rows(rows, {"prompt_coverage": "diverse"})
        self.assertEqual({c["branch_margin"] for c in result["conditional_margins"]}, {0, 1})
        observed = {(c["style"], c["operand_sign"], c["branch"]) for c in result["transition_joint_cases"]}
        self.assertEqual(observed, {("canonical", "positive", "true"), ("native_a", "positive", "false"),
                                   ("native_b", "negative", "true"), ("natural", "negative", "false")})
        self.assertTrue(any(z["dimension"] == "conditional_sign_branch" for z in result["explicit_zero_cells"]))

    def test_latent_negative_metadata_does_not_count_as_negative_supported_gold(self):
        rows = [row(0, family="answerability", expected=7, sign="positive"),
                row(1, family="answerability", answerable=False, sign="negative", scenario="relevant_missing")]
        result = cases.summarize_rows(rows)
        supported = [cell for cell in result["supported_numeric_gold"] if cell["answerable"]]
        self.assertEqual({c["supported_numeric_gold_sign"] for c in supported}, {"positive"})
        self.assertTrue(any(z["dimension"] == "supported_numeric_gold_sign" and z["missing"] == "negative" for z in result["explicit_zero_cells"]))

    def test_replay_is_separate_and_original_canonical_label_is_not_prompt_identity(self):
        fresh, replay = row(0), row(1)
        replay.update(split="replay", is_replay=True, system="original historical wording")
        result = cases.summarize_rows([fresh, replay])
        self.assertEqual({c["scope"] for c in result["by_family_style"]}, {"fresh_repair", "historical_replay"})
        self.assertIn("original system prompts", result["style_caveat"])

    def test_seed_membership_is_same_while_order_differs(self):
        left = [row(i) for i in range(4)]
        right = list(reversed(left))
        result = cases.compare_membership(left, right)
        self.assertTrue(result["semantic_multiset_equal"])
        self.assertTrue(result["row_identity_multiset_equal"])
        self.assertFalse(result["row_order_equal"])
        self.assertEqual(result["identical_semantic_occurrences"], 4)

    def test_prompt_contract_change_preserves_semantic_membership(self):
        left = [row(i) for i in range(4)]
        right = copy.deepcopy(left)
        for i, item in enumerate(right):
            item.update(variant_id=cases.STYLES[i], system="changed prompt", contract={"kind": "text"}, row_id="style-" + item["row_id"])
        result = cases.compare_membership(left, right)
        self.assertTrue(result["semantic_multiset_equal"])
        self.assertFalse(result["row_identity_multiset_equal"])

    def test_semantic_gold_match_does_not_hide_changed_abstention_target_strings(self):
        left = [row(0, family="answerability", answerable=False)]
        right = copy.deepcopy(left)
        left[0]["contract"] = {"abstention": "UNKNOWN"}
        right[0].update(variant_id="natural", contract={"abstention": "I don't know based on the available information."})
        result = cases.compare_membership(left, right)
        self.assertTrue(result["semantic_multiset_equal"])
        self.assertTrue(result["supervised_targets_available"])
        self.assertFalse(result["supervised_target_multiset_equal"])

    def test_changed_gold_or_world_cannot_masquerade_as_held_factor_match(self):
        left, right = [row(0)], [row(0)]
        right[0]["expected"]["A"] += 1
        self.assertFalse(cases.compare_membership(left, right)["semantic_multiset_equal"])
        right = [row(1)]
        self.assertEqual(cases.compare_membership(left, right)["world_intersection"], 0)

    def test_prospective_assignment_balances_cases_without_mutating_inputs(self):
        rows = [row(i, style=cases.STYLES[i % 4], sign="positive" if i % 2 == 0 else "negative", pair=i // 2) for i in range(32)]
        before = copy.deepcopy(rows)
        result = cases.prospective_balanced_style_assignment(rows)
        self.assertEqual(rows, before)
        self.assertEqual(Counter(result["assignments"].values()), dict.fromkeys(cases.STYLES, 8))
        assigned = copy.deepcopy(rows)
        for item in assigned:
            item["variant_id"] = result["assignments"][item["row_id"]]
        audit = cases.summarize_rows(assigned, {"prompt_coverage": "diverse"})
        self.assertFalse(any(z["dimension"] == "operand_sign" for z in audit["explicit_zero_cells"]))
        self.assertIn("PROSPECTIVE_UNADOPTED", result["status"])
        self.assertEqual(cases.compare_membership(rows, assigned)["identical_semantic_occurrences"], 32)

    def test_odd_case_cells_use_floor_ceil_while_style_total_is_exact(self):
        rows = [row(i, sign="positive" if i < 6 else "negative") for i in range(20)]
        result = cases.prospective_balanced_style_assignment(rows)
        self.assertEqual(Counter(result["assignments"].values()), dict.fromkeys(cases.STYLES, 5))
        positive = Counter(result["assignments"][r["row_id"]] for r in rows if r["payload"]["event"]["amount"] > 0)
        self.assertLessEqual(max(positive.values()) - min(positive.values()), 1)

    def test_final_payload_and_boolean_scalar_gold_are_rejected(self):
        forbidden = row(0)
        forbidden["split"] = "final"
        with self.assertRaisesRegex(ValueError, "TRAINING_DEVELOPMENT_ONLY"):
            cases.summarize_rows([forbidden])
        bad = row(0, family="answerability", expected=True)
        with self.assertRaisesRegex(ValueError, "TYPED_INTEGER_GOLD"):
            cases.summarize_rows([bad])

    def test_prepared_metadata_requires_all_sixteen_pinned_trial_descriptors(self):
        fixed = {"registry_sha256": "fixture-registry", "trials": [{"trial_id": "T" + str(i)} for i in range(16)]}
        frozen = {"registry": fixed, "validation_sha256": "fixture-validation"}
        validation = {"offline_validation_passed": True, "native_execution_authorized": False,
                      "registry_sha256": "fixture-registry", "datasets": {"T" + str(i): {} for i in range(16)}}
        self.assertEqual(len(cases._validate_prepared_metadata(validation, frozen, fixed, "fixture-validation")), 16)
        missing = copy.deepcopy(validation)
        del missing["datasets"]["T0"]
        with self.assertRaisesRegex(ValueError, "PREPARED_PACKET_BINDING"):
            cases._validate_prepared_metadata(missing, frozen, fixed, "fixture-validation")
        with self.assertRaisesRegex(ValueError, "PREPARED_PACKET_BINDING"):
            cases._validate_prepared_metadata(validation, frozen, fixed, "changed-validation")

    def test_prospective_repair_spec_is_unadopted_and_addresses_separate_input_defects(self):
        spec = cases.prospective_repair_decisions()
        self.assertFalse(spec["current_recipes_and_endpoints_changed"])
        self.assertEqual(len(spec["alternatives"]), 4)
        self.assertTrue(any("supported scalar" in option["problem"] for option in spec["alternatives"]))
        self.assertTrue(any("parent worlds" in option["problem"] for option in spec["alternatives"]))
        self.assertEqual(spec["native_effects"], "UNMEASURED")

    def test_observed_pattern_claims_are_data_checks_and_never_native_gain_claims(self):
        rows = [row(i, style=cases.STYLES[i % 4], sign="positive" if i % 2 == 0 else "negative") for i in range(16)]
        rows += [row(i + 16, family="register", style=cases.STYLES[i % 4], branch=i % 2 == 0,
                     sign="positive" if i % 4 < 2 else "negative") for i in range(16)]
        result = cases.observed_input_patterns(rows, {"prompt_coverage": "diverse", "difficulty": "counterfactual"})
        self.assertTrue(result["hard_diverse_ledger_style_sign_mapping_observed"])
        self.assertTrue(result["diverse_conditional_branch_style_mapping_observed"])
        self.assertTrue(result["fresh_conditional_margins_exactly_zero_one"])
        self.assertFalse(result["model_quality_gain"])


if __name__ == "__main__":
    unittest.main()
