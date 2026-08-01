from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase_2c_targeted_recovery.build_phase_2c_dataset import (
    generated_rows,
    interleave_rows,
    prompt_target_key,
)
from scripts.phase_2c_targeted_recovery.phase2c_common import (
    assess_training_candidate,
    candidate_specs,
    combine_importance,
    complementary_layers,
    select_profiles,
)
from scripts.phase_2c_targeted_recovery.resolve_phase_2c_profile import resolve_profile


class Phase2CTargetedRecoveryTests(unittest.TestCase):
    def test_generated_reasoning_splits_do_not_overlap(self) -> None:
        train = generated_rows("train", train=True)
        dev = generated_rows("dev", train=False)
        self.assertGreaterEqual(len(train), 100)
        self.assertGreaterEqual(len(dev), 50)
        self.assertFalse({prompt_target_key(row) for row in train} & {prompt_target_key(row) for row in dev})
        self.assertTrue(all(row["messages"][-1]["role"] == "assistant" for row in train + dev))

    def test_limited_dev_prefix_contains_original_and_phase2c_cases(self) -> None:
        base = [{"id": f"base_{index}"} for index in range(100)]
        added = [{"id": f"p2c_{index}"} for index in range(20)]
        mixed = interleave_rows(base, added, base_per_added=2)
        prefix_ids = [row["id"] for row in mixed[:30]]
        self.assertTrue(any(value.startswith("base_") for value in prefix_ids))
        self.assertTrue(any(value.startswith("p2c_") for value in prefix_ids))
        self.assertEqual(len(mixed), 120)

    def test_importance_combination_ranks_damage_signals(self) -> None:
        rows = [
            {
                "current_layer_id": 0,
                "original_layer_id": 0,
                "gradient_taylor_score": 0.1,
                "ablation_nll_increase": 0.1,
                "structural_sensitivity": 0.0,
            },
            {
                "current_layer_id": 1,
                "original_layer_id": 3,
                "gradient_taylor_score": 0.9,
                "ablation_nll_increase": 1.2,
                "structural_sensitivity": 0.8,
            },
        ]
        ranked = combine_importance(rows)
        self.assertEqual(ranked[0]["current_layer_id"], 1)
        self.assertEqual(ranked[0]["importance_rank"], 1)
        self.assertGreater(ranked[0]["combined_importance_score"], ranked[1]["combined_importance_score"])

    def test_candidate_matrix_contains_required_controls(self) -> None:
        ranking = [
            {"current_layer_id": layer, "combined_importance_score": 1.0 - layer / 20}
            for layer in range(12)
        ]
        specs = candidate_specs(ranking, seed=11)
        names = {row["name"] for row in specs}
        self.assertTrue(
            {
                "all_remaining", "top1", "top2", "top4", "top8",
                "complementary4", "random4", "low4", "top4_attention", "top4_mlp",
            }.issubset(names)
        )
        self.assertEqual(len(complementary_layers(list(range(12)), 12, 4)), 4)

    @staticmethod
    def candidate(name: str, trainable: int, quality: float, nll: float = 1.0) -> dict:
        summary = {
            "quality_evaluation_version": "2.0",
            "mean_quality_score": quality,
            "quality_pass_rate": 1.0,
            "intelligibility_pass_rate": 1.0,
            "hard_failure_case_count": 0,
            "worst_decile_mean_quality_score": quality,
            "p10_quality_score": quality,
            "mean_reference_alignment_score": quality,
            "major_case_count": 0,
            "category_quality_scores": {"logic": quality},
            "mean_heuristic_score": quality,
            "semantic_pass_rate": quality,
            "mean_reference_token_f1": quality,
            "complete_utterance_rate": 1.0,
            "critical_failure_count": 0,
            "category_scores": {"logic": quality},
            "category_semantic_pass_rates": {"logic": quality},
            "category_reference_token_f1": {"logic": quality},
            "category_complete_utterance_rates": {"logic": 1.0},
        }
        return {
            "name": name,
            "generation_summary": summary,
            "nll_summary": {"mean_target_nll": nll},
            "output_deviation": {
                "new_critical_failure_count": 0,
                "new_major_issue_count": 0,
            },
            "training_manifest": {
                "trainable_parameter_count": trainable,
                "peak_allocated_vram_gb": 1.0,
                "train_metrics": {"train_runtime": 10.0},
            },
        }

    def test_candidate_gate_rewards_quality_with_trainable_reduction(self) -> None:
        frozen = self.candidate("reduced_frozen", 0, 0.90, 1.2)
        all_remaining = self.candidate("all_remaining", 1000, 1.0, 1.0)
        selective = self.candidate("top4", 200, 0.995, 1.03)
        assessment = assess_training_candidate(
            candidate=selective,
            all_remaining=all_remaining,
            reduced_frozen=frozen,
            max_quality_index_loss=0.02,
            max_frozen_quality_loss=0.01,
            max_heuristic_loss=0.02,
            max_semantic_pass_loss=0.02,
            max_reference_f1_loss=0.03,
            max_category_loss_allowed=0.05,
            max_frozen_category_loss=0.03,
            max_nll_increase=0.10,
            min_complete_utterance_rate=0.99,
            minimum_trainable_reduction=0.25,
            max_new_major_issues=0,
        )
        self.assertTrue(assessment["eligible"])
        self.assertAlmostEqual(assessment["trainable_parameter_reduction_vs_all_remaining"], 0.8)
        self.assertGreater(assessment["fraction_of_all_remaining_gain_recovered"], 0.9)

    def test_category_semantic_collapse_fails_even_when_heuristic_is_high(self) -> None:
        frozen = self.candidate("reduced_frozen", 0, 1.0, 1.0)
        all_remaining = self.candidate("all_remaining", 1000, 1.0, 1.0)
        selective = self.candidate("top4", 200, 1.0, 1.0)
        selective["generation_summary"]["category_semantic_pass_rates"]["logic"] = 0.0
        assessment = assess_training_candidate(
            candidate=selective,
            all_remaining=all_remaining,
            reduced_frozen=frozen,
            max_quality_index_loss=0.50,
            max_frozen_quality_loss=0.50,
            max_heuristic_loss=0.02,
            max_semantic_pass_loss=0.50,
            max_reference_f1_loss=0.50,
            max_category_loss_allowed=0.05,
            max_frozen_category_loss=0.05,
            max_nll_increase=0.10,
            min_complete_utterance_rate=0.99,
            minimum_trainable_reduction=0.25,
            max_new_major_issues=0,
        )
        self.assertFalse(assessment["eligible"])
        self.assertGreater(assessment["worst_category_loss_vs_all_remaining"], 0.05)

    def test_new_major_issue_fails_strict_gate(self) -> None:
        frozen = self.candidate("reduced_frozen", 0, 0.90, 1.2)
        all_remaining = self.candidate("all_remaining", 1000, 1.0, 1.0)
        selective = self.candidate("top4", 200, 0.995, 1.03)
        selective["output_deviation"]["new_major_issue_count"] = 1
        assessment = assess_training_candidate(
            candidate=selective, all_remaining=all_remaining, reduced_frozen=frozen,
            max_quality_index_loss=0.02, max_frozen_quality_loss=0.01,
            max_heuristic_loss=0.02, max_semantic_pass_loss=0.02,
            max_reference_f1_loss=0.03, max_category_loss_allowed=0.05,
            max_frozen_category_loss=0.03, max_nll_increase=0.10,
            min_complete_utterance_rate=0.99, minimum_trainable_reduction=0.25,
            max_new_major_issues=0,
        )
        self.assertFalse(assessment["eligible"])
        self.assertTrue(any("new major issues" in reason for reason in assessment["rejection_reasons"]))

    def test_profile_selection_can_choose_one_efficient_candidate_for_two_profiles(self) -> None:
        all_remaining = self.candidate("all_remaining", 1000, 1.0)
        all_remaining["assessment"] = {"eligible": True, "quality_index": 1.0}
        top4 = self.candidate("top4", 200, 0.995)
        top4["assessment"] = {"eligible": True, "quality_index": 0.995}
        top1 = self.candidate("top1", 80, 0.95)
        top1["assessment"] = {"eligible": False, "quality_index": 0.95}
        profiles = select_profiles([all_remaining, top4, top1], balanced_tolerance=0.01)
        self.assertIs(profiles["maximum_quality"], all_remaining)
        self.assertIs(profiles["balanced_training"], top4)
        self.assertIs(profiles["minimum_training_memory"], top4)

    def test_phase2c_profile_requires_pass_and_explicit_name(self) -> None:
        catalog = {
            "profiles": {
                "balanced_training": {
                    "training_model_id": "reduced",
                    "training_adapter": "adapter",
                    "evaluation_summary": "summary.json",
                    "assessment": {"eligible": True},
                    "seed_stability_pass": True,
                }
            }
        }
        with self.assertRaises(RuntimeError):
            resolve_profile(catalog, "balanced_training", "- Human decision: `PENDING`\n")
        gate = "- Human decision: `PASS`\n- Approved profiles: `balanced_training`\n"
        row = resolve_profile(catalog, "balanced_training", gate)
        self.assertEqual(row["training_adapter"], "adapter")


if __name__ == "__main__":
    unittest.main()
