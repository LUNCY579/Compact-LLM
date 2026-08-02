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
    build_run_configuration,
    build_variant_provenance,
    combine_progression_assessments,
    prepare_variant_evidence,
    profile_validation_dir,
    restore_resume_state,
    select_profiles,
    validate_resume_catalog,
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
            "quality_evaluation_version": "4.3",
            "case_count": 100,
            "mean_quality_score": score,
            "worst_decile_mean_quality_score": score,
            "mean_quality_dimensions": dimensions,
            "category_quality_scores": {"facts": score},
            "mean_issue_cost": 0.0,
            "severe_issue_case_rate": 0.0,
        }


    @staticmethod
    def phase2b_args(**overrides):
        values = {
            "model_id": "Qwen/Qwen3-0.6B",
            "sft_adapter": Path("artifacts/dense_sft/adapter"),
            "train_file": Path("data/train/train.jsonl"),
            "dev_file": Path("data/dev/dev.jsonl"),
            "probe_file": Path("data/phase_2_structural_reduction/probe_150_thorough_v2.jsonl"),
            "phase_benchmark": Path("data/benchmark/shortchatbench_pilot.jsonl"),
            "probe_limit": 150,
            "probe_seed": 20260728,
            "quality_profile": "balanced",
            "recovery_top_k": 6,
            "recovery_steps": 50,
            "recovery_seed": 11,
            "max_new_tokens": 32,
            "qlora": True,
            "gradient_checkpointing": True,
            "skip_recovery": False,
            "max_drops": 27,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_progression_requires_positive_incremental_margin(self) -> None:
        cumulative = {
            "eligible_for_progression": True,
            "rejection_reasons": [],
            "weight_memory_reduction_fraction": 0.30,
            "quality_opportunity_loss_fraction": 0.10,
            "opportunity_margin": 0.20,
        }
        incremental = {
            "eligible_for_progression": False,
            "rejection_reasons": ["incremental quality cost exceeds gain"],
            "weight_memory_reduction_fraction": 0.04,
            "quality_opportunity_loss_fraction": 0.08,
            "opportunity_margin": -0.04,
        }
        combined = combine_progression_assessments(cumulative, incremental)
        self.assertFalse(combined["eligible_for_progression"])
        self.assertTrue(combined["cumulative_eligible_for_progression"])
        self.assertFalse(combined["incremental_eligible_for_progression"])
        self.assertAlmostEqual(combined["incremental_opportunity_margin"], -0.04)
        self.assertTrue(any(reason.startswith("incremental:") for reason in combined["rejection_reasons"]))

    def test_resume_restores_last_frontier_without_replaying_prior_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage_model = root / "frontier" / "stage_10" / "model"
            stage_model.mkdir(parents=True)
            registry = root / "candidate_registry.json"
            registry.write_text('{"candidates": [{"stage": 1, "variant_id": "old"}]}', encoding="utf-8")
            catalog = {
                "frontier": [
                    {
                        "stage": 10,
                        "inference_model": str(stage_model),
                        "dropped_original_layer_ids": [16, 26, 7, 21, 17, 13, 3, 15, 6, 11],
                    }
                ]
            }
            current, dropped, candidates, frontier, start_stage = restore_resume_state(
                catalog,
                reference_model=root / "reference",
                candidate_registry_path=registry,
            )
            self.assertEqual(current, stage_model)
            self.assertEqual(start_stage, 11)
            self.assertEqual(len(dropped), 10)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(len(frontier), 1)

    def test_resume_validation_allows_only_maxdrops_extension(self) -> None:
        first = self.phase2b_args(max_drops=10)
        catalog = {
            "source_model_id": first.model_id,
            "source_sft_adapter": str(first.sft_adapter),
            "probe_file": str(first.probe_file),
            "phase_benchmark": str(first.phase_benchmark),
            "probe_case_count": first.probe_limit,
            "opportunity_policy": {
                "quality_profile": first.quality_profile,
                "quality_evaluation_version": "4.3",
            },
            "run_configuration": build_run_configuration(first),
        }
        extended = self.phase2b_args(max_drops=27)
        result = validate_resume_catalog(catalog, extended)
        self.assertEqual(result["status"], "fully_verified")

        changed = self.phase2b_args(max_drops=27, recovery_seed=99)
        with self.assertRaises(RuntimeError):
            validate_resume_catalog(catalog, changed)

    def test_evidence_provenance_invalidates_changed_parent_and_records_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir) / "evidence"
            evidence.mkdir()
            (evidence / "stale.txt").write_text("old branch", encoding="utf-8")
            args = self.phase2b_args()
            expected = build_variant_provenance(
                stage=11,
                variant_id="stage_11_drop_original_12_recovered",
                variant_kind="deletion + recovery SFT",
                parent_model="frontier/stage_10/model",
                dropped_original_layer_id=12,
                retained_original_layer_ids=[0, 1, 2],
                args=args,
            )
            self.assertEqual(expected["recovery_steps"], 50)
            self.assertEqual(expected["recovery_seed"], 11)
            self.assertTrue(expected["qlora"])
            removed = prepare_variant_evidence(evidence, expected)
            self.assertTrue(removed)
            self.assertFalse((evidence / "stale.txt").exists())
            self.assertTrue((evidence / "evidence_provenance.json").exists())
            self.assertFalse(prepare_variant_evidence(evidence, expected))

    def test_profile_validation_cache_is_model_keyed(self) -> None:
        root = Path("artifacts/test")
        first = profile_validation_dir(root, "balanced", "frontier/stage_10/model")
        second = profile_validation_dir(root, "balanced", "frontier/stage_11/model")
        self.assertNotEqual(first, second)

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
        self.assertIn("cumulative AND incremental weight-memory gain > quality loss", wrapper)
        runner = (root / "scripts" / "phase_2_layer_selection" / "progressive_reduction.py").read_text(encoding="utf-8")
        self.assertIn("focused recovery found no eligible deletion", runner)
        self.assertIn("continuing from accepted stage", runner)
        self.assertIn("incremental parent-relative", runner)
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
