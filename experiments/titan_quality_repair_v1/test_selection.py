"""Synthetic scoring/selection fixtures only, not a native authorization ledger."""
import copy
import dataclasses
import math
import unittest
from unittest.mock import patch

from .schema import EXPERIMENT, FAMILIES, REPAIR_FAMILIES, SEEDS
from .scoring import aggregate_rows
from .selection import (SOURCE_SHA256, SelectionReceipt, development_gates, final_acceptance,
                        select_recipe, validate_final_gate)
from .statistics import (ENDPOINTS, ENDPOINT_ALPHA, ENDPOINT_REGISTRY, binomial_cdf,
                         clopper_pearson_upper, damage_bounds, macro_gain_interval, paired_worlds)
from .test_scoring import row

RECIPES = tuple(f"recipe{i}" for i in range(8))


def summary(*, split="dev", n=64, wrong=(), abstain=(), hallucinate=()):
    rows, outputs = [], {}
    for family in FAMILIES:
        for i in range(n):
            supported = family not in {"answerability", "grounded_qa"} or i % 2 == 0
            fixture = row(7, row_id=f"{family}:{i}", world_id=f"{family}:{i}", family=family,
                          split=split, answerable=supported)
            rows.append(fixture)
            response = "7" if supported else "UNKNOWN"
            if (family, i) in wrong:
                response = "8" if supported else "7"
            if (family, i) in abstain:
                response = "UNKNOWN"
            if (family, i) in hallucinate:
                response = "7"
            outputs[fixture["row_id"]] = response
    for i in range(n):
        fixture = row(7, row_id=f"ledger:{i}:natural", world_id=f"ledger:{i}",
                      variant="natural", family="ledger", split=split)
        rows.append(fixture)
        outputs[fixture["row_id"]] = "7"
    return aggregate_rows(rows, outputs)


def trials(candidate):
    return {recipe: {seed: copy.deepcopy(candidate) for seed in SEEDS} for recipe in RECIPES}


def fake_proof(receipt):
    """Clearly synthetic proof fields used solely to exercise pure predicates."""
    return {
        "experiment": EXPERIMENT, "selected_recipe_id": receipt.selected_recipe_id,
        "selection_sha256": receipt.sha256,
        "exports": {str(seed): {"artifact_sha256": str(index) * 64, "architecture": "Qwen3ForCausalLM",
                                "dtype": "float32", "custom_abstention": False, "dev_passed": True,
                                "dev_receipt_sha256": "d" * 64, "recipe_id": receipt.selected_recipe_id,
                                "seed": seed} for index, seed in enumerate(SEEDS, start=1)},
        "qualification": {"passed": True, "source_sha256": SOURCE_SHA256, "receipt_sha256": "a" * 64},
        "operator_permit": {"experiment": EXPERIMENT, "source_sha256": SOURCE_SHA256,
                            "stage": "final_evaluation", "hardware_slot": "FIXTURE_NO_NATIVE_SLOT",
                            "permit_sha256": "b" * 64},
        "ledger": {"final_unopened": True, "spent": False},
    }


class DevelopmentSelectionTests(unittest.TestCase):
    def setUp(self):
        self.baseline = summary()

    def test_recipe_id_deterministic_tie_and_deliverable_seed(self):
        receipt = select_recipe(self.baseline, trials(self.baseline), RECIPES)
        self.assertEqual(receipt.selected_recipe_id, "recipe0")
        self.assertEqual(receipt.as_dict()["decision"]["deliverable_seed"], 17)
        self.assertFalse(receipt.as_dict()["decision"]["fallback_permitted"])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            receipt.sha256 = "x"
        with self.assertRaises(ValueError):
            receipt.require_recipe("recipe1")

    def test_no_eligible_recipe_has_terminal_receipt(self):
        failed = summary(wrong=(("arithmetic", 1), ("arithmetic", 2)))
        receipt = select_recipe(self.baseline, trials(failed), RECIPES)
        self.assertIsNone(receipt.selected_recipe_id)
        self.assertEqual(receipt.as_dict()["decision"]["status"], "STOP_NO_ELIGIBLE_RECIPE")
        with self.assertRaises(ValueError):
            receipt.require_recipe("recipe0")

    def test_both_seeds_must_pass(self):
        all_trials = trials(self.baseline)
        all_trials["recipe0"][29] = summary(wrong=(("arithmetic", 1), ("arithmetic", 2)))
        receipt = select_recipe(self.baseline, all_trials, RECIPES)
        self.assertEqual(receipt.selected_recipe_id, "recipe1")

    def test_macro_gain_selects_recipe_using_both_seeds(self):
        baseline = summary(wrong=tuple((family, 0) for family in REPAIR_FAMILIES))
        all_trials = trials(baseline)
        all_trials["recipe7"] = {seed: summary() for seed in SEEDS}
        receipt = select_recipe(baseline, all_trials, RECIPES)
        self.assertEqual(receipt.selected_recipe_id, "recipe7")
        self.assertGreater(receipt.as_dict()["decision"]["audits"]["recipe7"]["seeds"]["17"]["repair_macro_gain"], 0)

    def test_blanket_abstention_and_zero_risk_increase_fail(self):
        candidate = summary(abstain=(("answerability", 0),))
        result = development_gates(self.baseline, candidate)
        self.assertFalse(result["passed"])
        self.assertIn("repair:unnecessary_increase", result["failures"])
        candidate = summary(hallucinate=(("answerability", 1),))
        result = development_gates(self.baseline, candidate)
        self.assertFalse(result["passed"])
        self.assertIn("repair:unsupported_increase", result["failures"])

    def test_point_margin_does_not_allow_missing_family_or_seed(self):
        incomplete = trials(self.baseline)
        del incomplete["recipe0"][29]
        with self.assertRaises(ValueError):
            select_recipe(self.baseline, incomplete, RECIPES)
        incomplete = trials(self.baseline)
        del incomplete["recipe0"]
        with self.assertRaises(ValueError):
            select_recipe(self.baseline, incomplete, RECIPES)
        incomplete = copy.deepcopy(self.baseline)
        incomplete["primary"]["ledger"].pop()
        with self.assertRaises(ValueError):
            development_gates(self.baseline, incomplete)

    def test_same_count_different_world_is_rejected(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["primary"]["ledger"][0]["world_id"] = "other-world"
        with self.assertRaises(ValueError):
            development_gates(self.baseline, candidate)

    def test_per_family_retention_strict_damage_visible(self):
        candidate = copy.deepcopy(self.baseline)
        for item in candidate["primary"]["instruction"][:2]:
            item["strict_correct"] = False
        result = development_gates(self.baseline, candidate)
        self.assertIn("instruction:retention_strict_decline", result["failures"])


class FrozenStatisticsTests(unittest.TestCase):
    def test_exact_binomial_reference_and_zero_denominator(self):
        self.assertAlmostEqual(binomial_cdf(1, 2, .5), .75, places=14)
        self.assertAlmostEqual(clopper_pearson_upper(0, 10, .05), 1 - .05 ** .1, places=14)
        self.assertAlmostEqual(clopper_pearson_upper(1, 2, .05), math.sqrt(.95), places=13)
        self.assertIsNone(clopper_pearson_upper(0, 0))
        self.assertEqual(clopper_pearson_upper(10, 10), 1)
        upper = clopper_pearson_upper(3, 512)
        self.assertAlmostEqual(binomial_cdf(3, 512, upper), ENDPOINT_ALPHA, places=11)

    def test_bonferroni_endpoint_registry_fixed(self):
        self.assertEqual(ENDPOINTS, 48)
        self.assertEqual(len(set(ENDPOINT_REGISTRY)), 48)
        self.assertAlmostEqual(ENDPOINT_ALPHA, .05 / 48)
        self.assertEqual(sum("supported_population_loss" in name for name in ENDPOINT_REGISTRY), 4)

    def test_damage_uses_baseline_correct_denominator(self):
        baseline = summary(wrong=(("ledger", 0),))
        candidate = summary(wrong=(("ledger", 0), ("ledger", 1)))
        bounds = damage_bounds(paired_worlds(baseline, candidate, "ledger"))
        self.assertEqual(bounds["semantic_population_damage"]["n"], 64)
        self.assertEqual(bounds["conditional_semantic_damage"]["n"], 63)
        self.assertEqual(bounds["semantic_population_damage"]["failures"], 1)
        self.assertGreater(bounds["conditional_semantic_damage"]["upper"], bounds["semantic_population_damage"]["upper"])

    def test_bootstrap_fixed_configuration_and_variants_do_not_enter(self):
        baseline = summary()
        tiny = copy.deepcopy(baseline)
        for family in FAMILIES:
            tiny["primary"][family] = tiny["primary"][family][:2]
        interval = macro_gain_interval(tiny, tiny)
        self.assertEqual(interval["point"], 0)
        self.assertEqual(interval["lower"], 0)
        self.assertEqual(interval["variant_independent_n"], 0)
        self.assertEqual(interval["family_world_n"], {family: 2 for family in REPAIR_FAMILIES})
        with self.assertRaises(ValueError):
            macro_gain_interval(tiny, tiny, draws=200)


class FinalGateTests(unittest.TestCase):
    def setUp(self):
        baseline = summary()
        self.receipt = select_recipe(baseline, trials(baseline), RECIPES)
        self.proof = fake_proof(self.receipt)

    def test_fields_only_predicate_is_not_native_authorization(self):
        result = validate_final_gate(self.receipt, self.proof)
        self.assertTrue(result["ready_for_verified_atomic_access_wrapper"])
        self.assertFalse(result["native_execution_authorized_by_this_function"])

    def test_export_dev_gate_source_slot_and_spent_final_fail_closed(self):
        mutations = (
            lambda proof: proof["exports"]["17"].update(dev_passed=False),
            lambda proof: proof["exports"]["17"].update(dtype="bfloat16"),
            lambda proof: proof["exports"]["29"].update(seed=17),
            lambda proof: proof["qualification"].update(source_sha256="0" * 64),
            lambda proof: proof["operator_permit"].update(experiment="OTHER"),
            lambda proof: proof["operator_permit"].update(hardware_slot=""),
            lambda proof: proof["ledger"].update(spent=True),
            lambda proof: proof.update(selected_recipe_id="recipe1"),
        )
        for mutation in mutations:
            proof = copy.deepcopy(self.proof)
            mutation(proof)
            with self.assertRaises(ValueError):
                validate_final_gate(self.receipt, proof)

    def test_final_cannot_accept_dev_instrument_or_wrong_seed(self):
        baseline = summary()
        with self.assertRaises(ValueError):
            final_acceptance(baseline, {17: baseline, 29: baseline}, receipt=self.receipt, access_evidence=self.proof)
        with self.assertRaises(ValueError):
            final_acceptance(baseline, {17: baseline}, receipt=self.receipt, access_evidence=self.proof)

    def test_receipt_hash_is_immutable_and_source_bound(self):
        with self.assertRaises(ValueError):
            SelectionReceipt(self.receipt.decision_json, "0" * 64)

    def test_final_fixed_seeds_success_and_replication_failure(self):
        # Complete synthetic final-shaped worlds; bootstrap is independently
        # tested above, and a controlled interval isolates acceptance routing.
        errors = tuple((family, i) for family in REPAIR_FAMILIES
                       for i in range(64) if family not in {"answerability", "grounded_qa"} or i % 2 == 0)
        baseline = summary(split="final", n=512, wrong=errors)
        candidate = summary(split="final", n=512)
        good_interval = {"point": .09375, "lower": .04, "draws": 20000, "seed": 20261005}
        with patch("experiments.titan_quality_repair_v1.selection.macro_gain_interval", return_value=good_interval):
            result = final_acceptance(baseline, {17: candidate, 29: candidate},
                                      receipt=self.receipt, access_evidence=self.proof)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["deliverable_seed"], 17)
        self.assertFalse(result["native_capability_claim"])
        bad_interval = {"point": .09375, "lower": 0, "draws": 20000, "seed": 20261005}
        with patch("experiments.titan_quality_repair_v1.selection.macro_gain_interval",
                   side_effect=[good_interval, bad_interval]):
            result = final_acceptance(baseline, {17: candidate, 29: candidate},
                                      receipt=self.receipt, access_evidence=self.proof)
        self.assertFalse(result["accepted"])
        self.assertTrue(result["seeds"]["17"]["passed"])
        self.assertFalse(result["seeds"]["29"]["passed"])
        self.assertFalse(result["fallback_permitted"])


if __name__ == "__main__":
    unittest.main()
