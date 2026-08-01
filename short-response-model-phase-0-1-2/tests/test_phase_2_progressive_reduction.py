from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from scripts.common.depth_reduction import choose_layers, reduce_model_depth
from scripts.phase_2_layer_selection.build_structural_probe import select_balanced
from scripts.phase_2_layer_selection.progressive_reduction import (
    assess_candidate,
    select_profiles,
)
from scripts.phase_2_layer_selection.resolve_phase_2_profile import resolve_profile


class DummyDecoder(nn.Module):
    def __init__(self, layers: int = 4, *, with_layer_types: bool = False) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(3, 3) for _ in range(layers)])
        self.config = SimpleNamespace(num_hidden_layers=layers)
        if with_layer_types:
            # Mirrors strict Qwen3 configuration metadata: one attention type
            # must remain aligned with every physical decoder block.
            self.config.layer_types = [
                "full_attention" if index % 2 == 0 else "sliding_attention"
                for index in range(layers)
            ]


class Phase2ProgressiveReductionTests(unittest.TestCase):
    def test_common_depth_reduction_preserves_original_lineage(self) -> None:
        model = DummyDecoder(4)
        result = reduce_model_depth(
            model,
            retained_current_layer_ids=[0, 2, 3],
            current_original_layer_ids=[0, 4, 7, 9],
        )
        self.assertEqual(result.retained_original_layer_ids, [0, 7, 9])
        self.assertEqual(result.dropped_original_layer_ids, [4])
        self.assertEqual(model.config.num_hidden_layers, 3)
        self.assertEqual(len(model.model.layers), 3)
        self.assertGreater(result.parameter_reduction_fraction, 0.0)

    def test_depth_reduction_updates_qwen3_layer_types(self) -> None:
        model = DummyDecoder(4, with_layer_types=True)
        reduce_model_depth(
            model,
            retained_current_layer_ids=[0, 2, 3],
        )
        self.assertEqual(model.config.num_hidden_layers, 3)
        self.assertEqual(
            model.config.layer_types,
            ["full_attention", "full_attention", "sliding_attention"],
        )
        self.assertEqual(len(model.config.layer_types), len(model.model.layers))

    def test_choose_layers_shared_with_phase4b(self) -> None:
        self.assertEqual(choose_layers(8, "uniform-drop", 2), [0, 1, 3, 4, 6, 7])
        self.assertEqual(choose_layers(8, "explicit", 0, "0,3,7"), [0, 3, 7])

    def test_probe_selection_is_balanced_and_deterministic(self) -> None:
        rows = [
            {"id": f"a{i}", "category": "a"} for i in range(5)
        ] + [
            {"id": f"b{i}", "category": "b"} for i in range(5)
        ]
        first = select_balanced(rows, 6, 11)
        second = select_balanced(rows, 6, 11)
        self.assertEqual(first, second)
        self.assertEqual(sum(row["category"] == "a" for row in first), 3)
        self.assertEqual(sum(row["category"] == "b" for row in first), 3)

    @staticmethod
    def quality_summary(score: float) -> dict:
        dimensions = {
            "comprehensibility": score,
            "task_adequacy": score,
            "logic_factual_consistency": score,
            "safety_boundary": score,
            "context_consistency": score,
            "response_completeness": score,
            "brevity_appropriateness": score,
            "conversation_naturalness": score,
            "content_anchor_preservation": score,
        }
        return {
            "quality_evaluation_version": "3.0",
            "case_count": 100,
            "mean_quality_score": score,
            "worst_decile_mean_quality_score": score,
            "mean_quality_dimensions": dimensions,
            "category_quality_scores": {"facts": score},
            "mean_issue_cost": 0.0,
            "severe_issue_case_rate": 0.0,
        }

    def test_memory_reduction_must_exceed_combined_quality_loss(self) -> None:
        baseline_generation = self.quality_summary(1.0)
        candidate_generation = self.quality_summary(0.97)
        assessment = assess_candidate(
            baseline_generation=baseline_generation,
            candidate_generation=candidate_generation,
            baseline_nll={"mean_target_nll": 1.0},
            candidate_nll={"mean_target_nll": 1.02},
            baseline_footprint={"parameter_tensor_bytes_loaded": 1000},
            candidate_footprint={"parameter_tensor_bytes_loaded": 900},
            output_deviation={},
        )
        self.assertTrue(assessment["eligible_for_progression"])
        self.assertAlmostEqual(assessment["weight_memory_reduction_fraction"], 0.10)
        self.assertGreater(assessment["opportunity_margin"], 0.0)

        rejected = assess_candidate(
            baseline_generation=baseline_generation,
            candidate_generation=self.quality_summary(0.70),
            baseline_nll={"mean_target_nll": 1.0},
            candidate_nll={"mean_target_nll": 1.20},
            baseline_footprint={"parameter_tensor_bytes_loaded": 1000},
            candidate_footprint={"parameter_tensor_bytes_loaded": 900},
            output_deviation={},
        )
        self.assertFalse(rejected["eligible_for_progression"])
        self.assertLessEqual(rejected["opportunity_margin"], 0.0)

    def test_profile_selection_uses_margin_and_resource_gain(self) -> None:
        baseline = {
            "profile": "full_quality",
            "remaining_layer_count": 28,
            "assessment": {"eligible_for_progression": True},
        }
        stage1 = {
            "remaining_layer_count": 27,
            "assessment": {
                "eligible_for_progression": True,
                "weight_memory_reduction_fraction": 0.04,
                "opportunity_margin": 0.03,
            },
        }
        stage2 = {
            "remaining_layer_count": 26,
            "assessment": {
                "eligible_for_progression": True,
                "weight_memory_reduction_fraction": 0.08,
                "opportunity_margin": 0.02,
            },
        }
        profiles = select_profiles(baseline, [stage1, stage2])
        self.assertIs(profiles["balanced"], stage1)
        self.assertIs(profiles["min_memory"], stage2)
        self.assertIs(profiles["full_quality"], baseline)

    def test_phase2b_wrapper_exposes_only_relative_policy_controls(self) -> None:
        root = Path(__file__).resolve().parents[1]
        wrapper = (root / "scripts" / "phase_2_layer_selection" / "run_phase_2_structural_reduction.ps1").read_text(encoding="utf-8")
        self.assertIn('[string]$QualityProfile = "balanced"', wrapper)
        self.assertIn('"--quality-profile", $QualityProfile', wrapper)
        self.assertIn("weight-memory gain > combined proportional quality loss", wrapper)
        for forbidden in (
            "MinQualityPassRate", "MinCompleteUtteranceRate", "MaxScoreLoss",
            "MaxCategoryLoss", "MaxNllIncrease", "MaxNewMajorIssues",
        ):
            self.assertNotIn(forbidden, wrapper)

    def test_reduced_profile_requires_explicit_approval(self) -> None:
        catalog = {
            "profiles": {
                "full_quality": {"training_model_id": "base"},
                "balanced": {"training_model_id": "reduced"},
            }
        }
        self.assertEqual(resolve_profile(catalog, "full_quality", "")["training_model_id"], "base")
        with self.assertRaises(RuntimeError):
            resolve_profile(catalog, "balanced", "- Human decision: `PENDING`\n")
        gate = "- Human decision: `PASS`\n- Approved profiles: `full_quality,balanced`\n"
        self.assertEqual(resolve_profile(catalog, "balanced", gate)["training_model_id"], "reduced")


if __name__ == "__main__":
    unittest.main()
