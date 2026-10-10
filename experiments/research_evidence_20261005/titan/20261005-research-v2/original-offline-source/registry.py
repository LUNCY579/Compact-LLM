"""Prospective immutable Titan quality recipe registry; no model/framework import.

Every recipe, seed and training default is fixed before outcomes. Registry data
are declarations, not native qualification or permission to spend the budget.
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path

from . import schema

SOURCE_WEIGHTS_SHA256 = "a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567"
SOURCE_CORE = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Transition-1.0.0-devfreeze")
SOURCE_WEIGHT_BYTES = 2_384_235_008
PROMPT_FACTORS = ("canonical", "diverse")
DIFFICULTY_FACTORS = ("regular", "counterfactual")
REPLAY_FACTORS = (0, 0.2)
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TRAINING = {
    "method": "LoRA", "base_parameters_frozen": True, "precision": "float32",
    "rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": list(LORA_TARGETS),
    "optimizer": "AdamW", "learning_rate": 5e-5, "learning_rate_schedule": "constant",
    "warmup_steps": 0, "weight_decay": 0, "microbatch": 1, "gradient_accumulation": 8,
    "adamw_betas": [0.9, 0.999], "adamw_eps": 1e-8, "gradient_clip_norm": 1.0,
    "gradient_checkpointing": True, "preserve_rng_state": True,
    "training_use_cache": False, "inference_use_cache": True, "tf32": False, "amp": False,
    "attention_implementation": "eager", "deterministic_algorithms": True,
    "loss_reduction": "mean_over_positive_assistant_and_final_EOS_tokens",
    "data_order": "fixed_registered_seed_shuffle_single_2000_example_epoch",
    "qualification_search": False, "dtype_fallback": False,
    "optimizer_steps": 250, "examples_per_trial": 2000, "epochs": 1,
    "max_sequence_length": 2048, "max_prompt_tokens": 1984, "max_target_tokens_including_eos": 64,
    "seed_scope": ["data_order", "adapter_initialization", "dropout"],
    "target_mass_policy": "REPORT_EXAMPLE_FRACTIONS_AND_MEASURED_LOSS_TOKEN_MASSES",
}


def build_registry() -> dict:
    recipes, trials = [], []
    for number, (prompt, difficulty, replay) in enumerate(
            itertools.product(PROMPT_FACTORS, DIFFICULTY_FACTORS, REPLAY_FACTORS), 1):
        recipe_id = f"R{number:02d}"
        replay_n = schema.REPLAY_EXAMPLES if replay else 0
        recipe = {"recipe_id": recipe_id, "prompt_coverage": prompt, "difficulty": difficulty,
                  "replay_fraction": replay, "total_examples": schema.TRAIN_EXAMPLES,
                  "repair_examples": schema.TRAIN_EXAMPLES - replay_n, "replay_examples": replay_n,
                  "prompt_styles": ["canonical"] if prompt == "canonical" else list(schema.STYLES),
                  "training": copy.deepcopy(TRAINING),
                  "replay_requirement": "GENUINE_PRIOR_TRAINING_GOLD_ONLY_NO_SYNTHETIC_FALLBACK" if replay else "NONE",
                  "replay_tradeoff": "400 prior examples replace 400 repair examples" if replay else "2000 repair examples"}
        recipes.append(recipe)
        for seed in schema.SEEDS:
            trials.append({"trial_id": f"TQR-{recipe_id}-S{seed}", "recipe_id": recipe_id,
                           "seed": seed, "source_weights_sha256": SOURCE_WEIGHTS_SHA256,
                           "declared_attempt_wall_seconds": 1800, "retry_policy": "ONE_ATTEMPT_NO_RETRY",
                           "status": "PROSPECTIVE_NOT_RUN"})
    value = {"experiment": schema.EXPERIMENT, "schema_version": schema.SCHEMA_VERSION,
             "status": "UNFROZEN_OFFLINE_PACKET", "native_execution_authorized": False,
             "source": {"name": "TitanTransitionFP32", "core_dir": str(SOURCE_CORE),
                        "weight_sha256": SOURCE_WEIGHTS_SHA256, "declared_weight_bytes": SOURCE_WEIGHT_BYTES,
                        "architecture": "Qwen3ForCausalLM", "model_type": "qwen3", "precision": "float32",
                        "source_preservation_policy": "READ_ONLY_NO_MODIFICATION",
                        "fresh_weight_rehash": "NOT_PERFORMED_OFFLINE",
                        "activation_offsets": False, "activation_routers": False},
             "factors": {"prompt": list(PROMPT_FACTORS), "difficulty": list(DIFFICULTY_FACTORS),
                         "replay_fraction": list(REPLAY_FACTORS)},
             "seeds": list(schema.SEEDS), "recipes": recipes, "trials": trials,
             "selection": {"data": "DEVELOPMENT_ONLY", "factor_search_after_outcomes": False,
                           "max_selected_exports": 2, "final_data_used_for_selection": False},
             "evaluation_membership": {
                 "development": {"canonical_worlds_per_family": 64, "canonical_families": list(schema.FAMILIES),
                                 "canonical_prompts": 448, "attached_variants_per_repair_family": 16,
                                 "attached_variant_prompts": 64, "prompts_per_arm": 512},
                 "final": {"canonical_worlds_per_family": 512, "canonical_families": list(schema.FAMILIES),
                           "canonical_prompts": 3584, "attached_variants_per_repair_family": 128,
                           "attached_variant_prompts": 512, "prompts_per_arm": 4096},
                 "variant_membership": "generator._variant_parents fixed stratum quotas, first eligible constructor-order rows",
                 "extra_style_assignment": "one of native_a/native_b/natural by fixed family offset and selected-row index modulo 3",
                 "retention_has_no_attached_variants": True,
                 "variant_unit": "REPEATED_OBSERVATION_IN_SOURCE_WORLD_NOT_NEW_INDEPENDENT_WORLD",
                 "all_four_styles_scope": "Diverse training distributes one style per example; evaluation adds one style on fixed subsets",
                 "selection_uses_model_outcomes": False},
             "claims": {"quality_evidence": False, "native_qualification": False, "promotion": False},
             "replay_is_fixed_total_budget_tradeoff": True}
    value["registry_sha256"] = schema.digest(value)
    return value


def validate_registry(registry: dict) -> dict:
    """Reject a supplied variant; changing choices requires a new approved study."""
    if registry != build_registry():
        raise ValueError("STOP_REGISTRY_DRIFT")
    return registry


def make_plan(registry: dict | None = None) -> dict:
    registry = build_registry() if registry is None else validate_registry(registry)
    calls = {"baseline_development": 512, "trial_development": 8192, "final_evaluation": 12288,
             "selected_export_development": 1024, "qualification_and_parity": 256}
    plan = {"experiment": schema.EXPERIMENT, "registry_sha256": registry["registry_sha256"],
            "evaluation_membership": copy.deepcopy(registry["evaluation_membership"]),
            "budget_seconds": {"qualification": 3600, "training": 28800, "evaluation": 54000},
            "total_budget_seconds": 86400, "training_trials": 16, "trial_wall_seconds": 1800,
            "optimizer_steps_per_trial": 250, "training_microbatches_per_trial": 2000,
            "training_microbatches_total": 32000, "training_optimizer_updates_total": 4000,
            "training_teacher_forced_forwards_lower_bound": 32000,
            "training_backward_requests": 32000, "max_teacher_forced_token_positions": 65536000,
            "checkpoint_recompute_forwards": "UNMEASURED_MUST_ACCOUNT_NATIVE_IF_USED",
            "call_budget": calls, "base_evaluation_prompts": 20992,
            "evaluation_prompts_with_export_checks": 22016, "total_generation_calls": 22272,
            "generation_call_budget_excludes_training_forwards_and_backward": True,
            "max_output_tokens": schema.MAX_OUTPUT_TOKENS,
            "max_decode_token_slots": 22272 * schema.MAX_OUTPUT_TOKENS,
            "failed_attempt_cost_policy": "CHARGE_DECLARED_CAP_BEFORE_ATTEMPT_NO_REFUND; RECORD_OVERRUNS",
            "exhaustion_policy": "CREATE_ONLY_SKIPPED_BUDGET_EXHAUSTED; NO_RESET_NO_ROOT_ROTATION",
            "evaluation_contingency": "Unneeded allocated final/export calls remain unused; no extra candidate is substituted",
            "feasibility": {"status": "BLOCKED_GENUINE_MEASURED_THROUGHPUT_REQUIRED",
                            "training_load_train_and_export_cost": "UNMEASURED",
                            "evaluation_generation_and_arm_load_cost": "UNMEASURED",
                            "requires": "Source/runtime-bound measured receipt under a named exclusive slot; no fabricated estimate"},
            "native_execution_authorized": False}
    if (sum(calls.values()) != plan["total_generation_calls"]
            or sum(plan["budget_seconds"].values()) != plan["total_budget_seconds"]
            or plan["training_trials"] * plan["trial_wall_seconds"] != plan["budget_seconds"]["training"]):
        raise AssertionError("STOP_PLAN_ARITHMETIC")
    plan["plan_sha256"] = schema.digest(plan)
    return plan
