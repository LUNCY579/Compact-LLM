"""Synthetic full-DEV arithmetic diagnostics; never load a model or any bank."""
import copy
import unittest

from . import registry
from .opposite_analysis import analyze_opposite_directions
from .schema import RETENTION_FAMILIES, SEEDS, digest
from .test_selection import summary


def observations(builder):
    return {recipe["recipe_id"]: {seed: builder(recipe, seed) for seed in SEEDS}
            for recipe in registry.build_registry()["recipes"]}


class OppositeAnalysisTests(unittest.TestCase):
    def test_no_observations_cannot_fabricate_gain(self):
        for results in (None, {}):
            report = analyze_opposite_directions(trial_summaries=results)
            self.assertEqual(report["status"], "UNMEASURED")
            self.assertEqual(report["analyzed_trial_count"], 0)
            self.assertEqual(report["matched_cell_contrasts"], [])
            self.assertEqual(report["main_effects"], {})
            self.assertIsNone(report["triple_interaction"])
            self.assertEqual(report["hypothesis_candidates"], [])
            self.assertFalse(report["model_gain_verified"])

    def test_no_factor_difference_is_not_a_positive_pattern(self):
        baseline = summary()
        results = observations(lambda recipe, seed: copy.deepcopy(baseline))
        report = analyze_opposite_directions(baseline, results)
        self.assertEqual(report["analyzed_trial_count"], 16)
        self.assertEqual(len(report["matched_cell_contrasts"]), 12)
        self.assertEqual(len(report["pairwise_interactions"]), 6)
        self.assertEqual(report["sign_reversals"], [])
        self.assertEqual(report["hypothesis_candidates"], [])
        self.assertTrue(all(value == 0 for effect in report["main_effects"].values()
                            for values in effect["by_seed"].values() for value in values.values()))

    def test_replay_can_help_retention_while_hurting_fresh_repair(self):
        baseline = summary()

        def builder(recipe, seed):
            if recipe["replay_fraction"]:
                return summary(wrong=tuple(("ledger", i) for i in range(4)))
            return summary(wrong=tuple((family, i) for family in RETENTION_FAMILIES for i in range(8)))

        report = analyze_opposite_directions(baseline, observations(builder))
        effect = report["main_effects"]["replay_fraction"]
        for seed in SEEDS:
            values = effect["by_seed"][str(seed)]
            self.assertAlmostEqual(values["repair_macro_semantic_accuracy"], -4 / 64 / 4)
            self.assertAlmostEqual(values["retention_population_semantic_damage"], -8 / 64)
            self.assertLess(values["supported_answer_accuracy"], 0)
            self.assertTrue(effect["tradeoffs_by_seed"][str(seed)]["has_opposite_quality_directions"])
        self.assertIn("replace 400 fresh", report["replay_estimand"])
        self.assertFalse(report["changes_selection_rules"])
        self.assertFalse(report["changes_final_endpoint_registry"])

    def test_prompt_effect_reverses_across_replay_and_interaction_is_exact(self):
        baseline = summary()

        def builder(recipe, seed):
            errors = 4 if ((recipe["prompt_coverage"] == "canonical") != bool(recipe["replay_fraction"])) else 0
            return summary(wrong=tuple(("ledger", i) for i in range(errors)))

        report = analyze_opposite_directions(baseline, observations(builder))
        reversals = [item for item in report["sign_reversals"]
                     if item["factor"] == "prompt_coverage" and item["metric"] == "repair_macro_semantic_accuracy"]
        self.assertEqual({item["seed"] for item in reversals}, set(SEEDS))
        self.assertTrue(all(item["positive_held_cells"] and item["negative_held_cells"] for item in reversals))
        interactions = [item for item in report["pairwise_interactions"]
                        if item["factors"] == ["prompt_coverage", "replay_fraction"]]
        for item in interactions:
            for seed in SEEDS:
                self.assertAlmostEqual(item["by_seed"][str(seed)]["repair_macro_semantic_accuracy"], -8 / 64 / 4)
        self.assertAlmostEqual(report["triple_interaction"]["by_seed"]["17"]["repair_macro_semantic_accuracy"], 0)

    def test_seed_opposition_is_exposed_without_mean_masking(self):
        baseline = summary()

        def builder(recipe, seed):
            diversity = recipe["prompt_coverage"] == "diverse"
            errors = 4 if (diversity if seed == 29 else not diversity) else 0
            return summary(wrong=tuple(("ledger", i) for i in range(errors)))

        report = analyze_opposite_directions(baseline, observations(builder))
        self.assertEqual(report["main_effects"]["prompt_coverage"]["seed_agreement"]["repair_macro_semantic_accuracy"],
                         "opposed_seed_directions")
        self.assertFalse(any(item["factor"] == "prompt_coverage" and
                             item["metric"] == "repair_macro_semantic_accuracy" for item in report["hypothesis_candidates"]))

    def test_consistent_two_seed_difference_is_only_a_dev_hypothesis(self):
        baseline = summary()
        results = observations(lambda recipe, seed: summary(
            wrong=tuple(("ledger", i) for i in range(4)) if recipe["prompt_coverage"] == "canonical" else ()))
        report = analyze_opposite_directions(baseline, results)
        candidates = [item for item in report["hypothesis_candidates"]
                      if item["factor"] == "prompt_coverage" and item["metric"] == "repair_macro_semantic_accuracy"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["status"], "CONSISTENT_DEV_HYPOTHESIS_PENDING_PROSPECTIVE_REPLICATION")
        self.assertFalse(candidates[0]["model_gain_verified"])
        self.assertIn("cannot confirm all factorial contrasts", report["replication_scope"])
        self.assertFalse(report["native_execution_authorized"])

    def test_missing_recipe_seed_and_baseline_fail_closed(self):
        baseline = summary()
        results = observations(lambda recipe, seed: copy.deepcopy(baseline))
        with self.assertRaisesRegex(ValueError, "MISSING_BASELINE"):
            analyze_opposite_directions(None, results)
        incomplete = copy.deepcopy(results)
        del incomplete["R08"]
        with self.assertRaisesRegex(ValueError, "INCOMPLETE_EIGHT_CELLS"):
            analyze_opposite_directions(baseline, incomplete)
        incomplete = copy.deepcopy(results)
        del incomplete["R01"][29]
        with self.assertRaisesRegex(ValueError, "FIXED_SEED_COVERAGE"):
            analyze_opposite_directions(baseline, incomplete)

    def test_final_results_and_final_baseline_are_refused(self):
        baseline = summary()
        results = observations(lambda recipe, seed: copy.deepcopy(baseline))
        results["R01"][17]["split"] = "final"
        with self.assertRaisesRegex(ValueError, "FINAL_OR_NONDEVELOPMENT"):
            analyze_opposite_directions(baseline, results)
        baseline["split"] = "final"
        with self.assertRaisesRegex(ValueError, "DEVELOPMENT_BASELINE"):
            analyze_opposite_directions(baseline, {})

    def test_unequal_world_identity_is_refused_even_at_equal_counts(self):
        baseline = summary()
        results = observations(lambda recipe, seed: copy.deepcopy(baseline))
        results["R01"][17]["primary"]["ledger"][0]["world_id"] = "new-world"
        with self.assertRaisesRegex(ValueError, "UNPAIRED_WORLD_MEMBERSHIP"):
            analyze_opposite_directions(baseline, results)

    def test_frozen_registry_and_inputs_are_preserved(self):
        baseline = summary()
        results = observations(lambda recipe, seed: copy.deepcopy(baseline))
        frozen = registry.build_registry()
        before = digest([baseline, results, frozen])
        analyze_opposite_directions(baseline, results, frozen)
        self.assertEqual(digest([baseline, results, frozen]), before)
        drifted = copy.deepcopy(frozen)
        drifted["recipes"][0]["prompt_coverage"] = "new-style"
        with self.assertRaisesRegex(ValueError, "REGISTRY_DRIFT"):
            analyze_opposite_directions(baseline, results, drifted)


if __name__ == "__main__":
    unittest.main()
