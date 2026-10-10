"""Synthetic DEV-shaped fixtures only; no model or experiment bank access."""
import copy
import unittest

from experiments.titan_quality_repair_v1 import registry
from experiments.titan_quality_repair_v1.schema import ABSTENTION, FAMILIES, REPAIR_FAMILIES, SEEDS, STYLES, digest
from experiments.titan_quality_repair_v1.scoring import aggregate_rows
from .gain_attribution import analyze_gain_attribution, prospective_spec


def summary(*, wrong=(), variant_wrong=(), abstain=(), empty=()):
    rows, outputs = [], {}
    for family in FAMILIES:
        for index in range(64):
            world = f"fixture:{family}:{index:02d}"
            supported = family not in {"answerability", "grounded_qa"} or index % 2 == 0
            fixture = {"schema_version": 1, "row_id": world + ":canonical", "world_id": world,
                "pair_id": None, "split": "dev", "family": family, "variant_id": "canonical",
                "system": "CPU fixture only", "user": "Return the fixture value", "answerable": supported,
                "expected": 7 if supported else None, "contract": {"kind": "integer", "abstention": "UNKNOWN"},
                "payload": {}, "operative_key": world, "support_expansion": family in {"ledger", "register"} and index % 2 == 0,
                "strata": {"fixture_only": True}}
            rows.append(fixture)
            answer = "7" if supported else "UNKNOWN"
            if (family, index) in wrong:
                answer = "8" if supported else "7"
            if (family, index) in abstain:
                answer = "UNKNOWN"
            if (family, index) in empty:
                answer = ""
            outputs[fixture["row_id"]] = answer
    primary = {(item["family"], int(item["world_id"].split(":")[-1])): item for item in rows}
    for offset, family in enumerate(REPAIR_FAMILIES):
        for index in range(16):
            style = STYLES[1 + (index + offset) % 3]
            fixture = copy.deepcopy(primary[family, index])
            fixture.update(row_id=fixture["world_id"] + ":" + style, variant_id=style)
            if style == "natural":
                fixture["contract"] = {"kind": "text", "abstention": ABSTENTION}
            rows.append(fixture)
            answer = "7" if fixture["answerable"] else fixture["contract"]["abstention"]
            if (family, index, style) in variant_wrong:
                answer = "8" if fixture["answerable"] else "7"
            outputs[fixture["row_id"]] = answer
    return aggregate_rows(rows, outputs)


def trials(builder):
    return {recipe["recipe_id"]: {seed: builder(recipe, seed) for seed in SEEDS}
            for recipe in registry.build_registry()["recipes"]}


class GainAttributionTests(unittest.TestCase):
    def test_empty_is_unmeasured_with_no_gain_values(self):
        report = analyze_gain_attribution()
        self.assertEqual(report["status"], "UNMEASURED")
        self.assertEqual(report["analyzed_trials"], 0)
        self.assertEqual(report["recipes"], {})
        self.assertFalse(report["quality_gain_verified"])
        self.assertFalse(report["selection_authority"])
        self.assertEqual(prospective_spec()["extra_model_calls"], 0)

    def test_expansion_gain_and_nonexpanded_loss_are_not_pooled_away(self):
        baseline = summary(wrong=(("ledger", 0), ("ledger", 2)))
        candidate = summary(wrong=(("ledger", 1),))
        report = analyze_gain_attribution(baseline, trials(lambda recipe, seed: copy.deepcopy(candidate)))
        result = report["recipes"]["R01"]["seeds"]["17"]
        ledger = result["families"]["ledger"]
        expanded = ledger["by_support_expansion"]["expanded"]
        nonexpanded = ledger["by_support_expansion"]["nonexpanded_operand_tag"]
        self.assertEqual(expanded["semantic"]["gained"], 2)
        self.assertEqual(nonexpanded["semantic"]["lost"], 1)
        self.assertAlmostEqual(result["repair_macro_semantic_delta"], 1 / 256)
        self.assertAlmostEqual(result["repair_macro_contributions_by_support_tag"]["expanded"], 2 / 256)
        self.assertAlmostEqual(result["repair_macro_contributions_by_support_tag"]["nonexpanded_operand_tag"], -1 / 256)
        self.assertIn("ledger balance support is unknown", report["spec"]["expansion_label_boundary"])

    def test_supported_correctness_abstention_risks_and_blank_decompose(self):
        baseline = summary(wrong=(("answerability", 0), ("answerability", 1)))
        candidate = summary(wrong=(("grounded_qa", 1),), abstain=(("grounded_qa", 0),), empty=(("ledger", 3),))
        result = analyze_gain_attribution(baseline, trials(lambda recipe, seed: copy.deepcopy(candidate)))["recipes"]["R01"]["seeds"]["17"]
        answerability = result["families"]["answerability"]["all"]
        self.assertEqual(answerability["supported_answer_semantic"]["gained"], 1)
        self.assertEqual(answerability["correct_abstention_on_unanswerable"]["gained"], 1)
        self.assertEqual(answerability["unsupported_answer"]["removed"], 1)
        grounded = result["families"]["grounded_qa"]["all"]
        self.assertEqual(grounded["unnecessary_abstention"]["introduced"], 1)
        self.assertEqual(grounded["unsupported_answer"]["introduced"], 1)
        self.assertEqual(result["families"]["ledger"]["all"]["no_answer"]["introduced"], 1)

    def test_same_aggregate_sign_can_have_disjoint_corrected_worlds(self):
        baseline = summary(wrong=(("ledger", 0), ("ledger", 1)))
        data = trials(lambda recipe, seed: summary(wrong=(("ledger", 1 if seed == 17 else 0),)))
        report = analyze_gain_attribution(baseline, data)
        overlap = report["recipes"]["R01"]["same_world_cross_seed_overlap"]["ledger"]
        self.assertEqual(overlap["corrected"]["seed17_n"], 1)
        self.assertEqual(overlap["corrected"]["seed29_n"], 1)
        self.assertEqual(overlap["corrected"]["common_n"], 0)
        self.assertEqual(overlap["corrected"]["jaccard"], 0)
        self.assertIsNone(overlap["lost"]["jaccard"])
        self.assertFalse(report["native_evidence"])

    def test_variant_effect_uses_same_parents_not_all_canonical_worlds(self):
        baseline = summary(wrong=(("ledger", 0),), variant_wrong=(("ledger", 0, "native_a"),))
        candidate = summary(variant_wrong=(("ledger", 0, "native_a"),))
        report = analyze_gain_attribution(baseline, trials(lambda recipe, seed: copy.deepcopy(candidate)))
        style = report["recipes"]["R01"]["seeds"]["17"]["attached_style_comparisons"]["ledger"]["native_a"]
        self.assertEqual(style["attached_row_n"], 6)
        self.assertEqual(style["matched_parent_world_n"], 6)
        self.assertEqual(style["variant_independent_n"], 0)
        self.assertEqual(style["variant_delta"]["semantic"]["gained"], 0)
        self.assertEqual(style["same_parent_canonical_delta"]["semantic"]["gained"], 1)
        self.assertAlmostEqual(style["style_minus_parent_semantic_improvement"], -1 / 6)

    def test_full_fixed_recipe_seed_grid_and_final_refusal(self):
        baseline = summary()
        data = trials(lambda recipe, seed: copy.deepcopy(baseline))
        incomplete = copy.deepcopy(data); del incomplete["R08"]
        with self.assertRaisesRegex(ValueError, "EXACT_RECIPE_GRID"):
            analyze_gain_attribution(baseline, incomplete)
        incomplete = copy.deepcopy(data); del incomplete["R01"][29]
        with self.assertRaisesRegex(ValueError, "EXACT_TWO_SEEDS"):
            analyze_gain_attribution(baseline, incomplete)
        final = copy.deepcopy(data); final["R01"][17]["split"] = "final"
        with self.assertRaisesRegex(ValueError, "DEVELOPMENT_ONLY"):
            analyze_gain_attribution(baseline, final)
        with self.assertRaisesRegex(ValueError, "MISSING_BASELINE"):
            analyze_gain_attribution(None, data)

    def test_exact_variant_projection_duplicate_ids_and_world_pairing(self):
        baseline = summary()
        data = trials(lambda recipe, seed: copy.deepcopy(baseline))
        wrong = copy.deepcopy(data); wrong["R01"][17]["variants"]["items"][0]["variant_id"] = "natural"
        with self.assertRaisesRegex(ValueError, "REGISTERED_VARIANT_PROJECTION"):
            analyze_gain_attribution(baseline, wrong)
        wrong = copy.deepcopy(data); wrong["R01"][17]["primary"]["ledger"][0]["world_id"] = "other-world"
        # Its attached variant can no longer address a present canonical parent.
        with self.assertRaises(ValueError):
            analyze_gain_attribution(baseline, wrong)
        wrong = copy.deepcopy(data)
        wrong["R01"][17]["primary"]["ledger"][1]["row_id"] = wrong["R01"][17]["primary"]["ledger"][0]["row_id"]
        with self.assertRaisesRegex(ValueError, "DUPLICATE_OR_EMPTY_ID"):
            analyze_gain_attribution(baseline, wrong)

    def test_untrusted_impossible_flags_are_rejected(self):
        baseline = summary()
        data = trials(lambda recipe, seed: copy.deepcopy(baseline))
        data["R01"][17]["primary"]["ledger"][0]["no_answer"] = True
        with self.assertRaisesRegex(ValueError, "BLANK_SCORE_INCONSISTENT"):
            analyze_gain_attribution(baseline, data)

    def test_empty_strata_are_null_and_no_seven_family_macro_is_added(self):
        baseline = summary()
        report = analyze_gain_attribution(baseline, trials(lambda recipe, seed: copy.deepcopy(baseline)))
        result = report["recipes"]["R01"]["seeds"]["17"]
        expanded = result["families"]["answerability"]["by_support_expansion"]["expanded"]
        self.assertEqual(expanded["semantic"]["n"], 0)
        self.assertIsNone(expanded["semantic"]["net_rate_change"])
        self.assertIsNone(expanded["baseline"]["semantic_accuracy"])
        self.assertNotIn("overall_macro", result)
        self.assertEqual(report["spec"]["variant_independent_n"], 0)

    def test_registry_and_inputs_stay_unchanged_and_no_seed_is_selected(self):
        baseline = summary()
        data = trials(lambda recipe, seed: copy.deepcopy(baseline))
        frozen = registry.build_registry()
        before = digest([baseline, data, frozen])
        report = analyze_gain_attribution(baseline, data, frozen)
        self.assertEqual(digest([baseline, data, frozen]), before)
        self.assertEqual(set(report["recipes"]["R01"]["seeds"]), {"17", "29"})
        self.assertFalse(report["seed_cherry_picking"])
        self.assertFalse(report["final_recipe_fallback_allowed"])
        self.assertFalse(report["new_gates"])


if __name__ == "__main__":
    unittest.main()
