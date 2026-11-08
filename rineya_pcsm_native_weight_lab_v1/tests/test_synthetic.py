from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rineya_pcsm.synthetic import (
    MATCHED_SPARSE_ARMS,
    REQUIRED_SECTION16_ARMS,
    SYNTHETIC_DISCLAIMER,
    SYNTHETIC_LABEL,
    SyntheticConfig,
    generate_problem,
    make_arm_plan,
    run_arm,
    run_experiment,
    summarize_experiment,
)


def _config() -> SyntheticConfig:
    return SyntheticConfig(
        label=SYNTHETIC_LABEL,
        seeds=(11, 17, 23, 29, 31),
        matrix_shape=(12, 16),
        latent_rank=3,
        noise_standard_deviation=0.18,
        budget_fractions=(0.10, 0.50),
        steps=8,
        learning_rate=0.04,
        arms=REQUIRED_SECTION16_ARMS,
    )


def test_problem_is_structured_deterministic_and_meta_loss_decreases() -> None:
    config = _config()
    first = generate_problem(config, 11)
    second = generate_problem(config, 11)
    assert first.target.shape == (12, 16)
    assert first.peers.shape == (config.peer_count, 12, 16)
    assert first.step_gradient_noise.shape == (1, 12, 16)
    assert np.array_equal(first.target, second.target)
    assert np.array_equal(first.peers, second.peers)
    assert np.array_equal(first.meta_feature_weights, second.meta_feature_weights)
    assert first.meta_feature_weights.dtype == np.float32
    assert sum(first.meta_feature_weights) == pytest.approx(1.0)
    assert len(first.meta_training_loss_curve) == config.steps + 1
    assert first.meta_training_loss_curve[-1] < first.meta_training_loss_curve[0]


def test_all_section16_plans_have_explicit_exact_support_and_fairness_labels() -> None:
    config = _config()
    problem = generate_problem(config, 11)
    expected_k = 20
    plans = {
        arm: make_arm_plan(problem, config, arm, 0.10)
        for arm in REQUIRED_SECTION16_ARMS
    }
    assert np.count_nonzero(plans["no_update"].support) == 0
    assert np.count_nonzero(plans["dense_noisy_update"].support) == 192
    assert np.count_nonzero(plans["oracle_support_update"].support) == expected_k
    assert all(
        np.count_nonzero(plans[arm].support) == expected_k
        for arm in MATCHED_SPARSE_ARMS
    )
    assert all(plans[arm].fairness_matched for arm in MATCHED_SPARSE_ARMS)
    assert plans["dense_noisy_update"].comparison_class == ("unmatched_dense_reference")
    assert plans["oracle_support_update"].comparison_class == (
        "unmatched_privileged_oracle_reference"
    )
    assert not np.array_equal(
        plans["real_pcsm_peers"].support,
        plans["coordinate_shuffled_peers"].support,
    )
    assert sum(plans["real_pcsm_peers"].peer_attention) == pytest.approx(1.0)


@pytest.mark.parametrize("arm", REQUIRED_SECTION16_ARMS)
def test_each_required_arm_obeys_one_update_and_exact_support(arm: str) -> None:
    config = _config()
    problem = generate_problem(config, 17)
    record = run_arm(problem, config, arm, 0.10)
    assert record.arm == arm
    assert record.section16_required_arm
    assert record.support_cardinality_exact
    assert record.support_k == record.support_union_k == record.declared_support_k
    assert record.exact_frozen_coordinates
    assert record.changed_frozen_coordinate_count == 0
    assert record.changed_parameter_count <= record.support_k
    if arm == "no_update":
        assert record.update_count == 0
        assert record.per_step_update_l2 == ()
        assert record.final_loss == record.initial_loss
    else:
        assert record.update_count == 1
        assert len(record.per_step_update_l2) == 1
    if arm in MATCHED_SPARSE_ARMS:
        assert record.fairness_matched
        assert record.per_step_update_l2[0] == pytest.approx(
            record.target_per_step_update_l2, abs=1e-12
        )


def test_full_experiment_has_a_through_h_decisive_checks_and_nonclaim() -> None:
    config = _config()
    records = run_experiment(config)
    assert len(records) == 5 * 2 * 8
    assert {record.arm for record in records} == set(REQUIRED_SECTION16_ARMS)
    assert all(record.exact_frozen_coordinates for record in records)
    summary = summarize_experiment(
        records, config, bootstrap_resamples=500, statistics_seed=99
    )
    assert summary["evidence_label"] == SYNTHETIC_LABEL
    assert summary["disclaimer"] == SYNTHETIC_DISCLAIMER
    assert summary["real_model_improvement_claim"] is False
    assert summary["advancement_decision"] == (
        "NOT_APPLICABLE_SYNTHETIC_FEASIBILITY_ONLY"
    )
    assert summary["required_section16_arm_audit"]["passed"] is True  # type: ignore[index]
    assert summary["one_update_protocol_audit"]["passed"] is True  # type: ignore[index]
    assert summary["exact_support_audit"]["passed"] is True  # type: ignore[index]
    assert summary["matched_sparse_update_budget_audit"]["passed"] is True  # type: ignore[index]
    assert summary["meta_training_audit"]["all_seeds_decreased"] is True  # type: ignore[index]
    decisive = summary["decisive_section16_checks"]
    assert decisive["all_passed"] is True  # type: ignore[index]
    checks = decisive["checks"]  # type: ignore[index]
    assert checks["loss_real_peers_lt_random"]["passed"] is True
    assert checks["loss_real_peers_lt_gradient_topk"]["passed"] is True
    assert checks["loss_real_peers_lt_coordinate_shuffled_peers"]["passed"] is True
    assert checks["recall_real_peers_gt_random"]["passed"] is True


def test_legacy_yaml_names_resolve_to_required_arms_without_hiding_ablations() -> None:
    config = SyntheticConfig.from_mapping(
        {
            "label": SYNTHETIC_LABEL,
            "seeds": [1, 2, 3, 4, 5],
            "matrix_shape": [8, 10],
            "latent_rank": 2,
            "noise_standard_deviation": 0.18,
            "budget_fractions": [0.1],
            "steps": 4,
            "learning_rate": 0.04,
            "arms": [
                "dense_adamw",
                "random_mask",
                "magnitude_mask",
                "gradient_mask",
                "shuffled_peer_pcsm",
                "no_peer_pcsm",
                "pcsm_no_stabilizer",
                "real_peer_pcsm",
            ],
        }
    )
    assert config.resolved_arms[:8] == REQUIRED_SECTION16_ARMS
    assert config.resolved_arms[8:] == (
        "magnitude_support",
        "real_pcsm_peers_no_stabilizer",
    )


def test_configuration_requires_five_distinct_seeds_and_known_arms() -> None:
    config = _config()
    with pytest.raises(ValueError, match="five distinct"):
        replace(config, seeds=(1, 2, 3, 4)).validate()
    with pytest.raises(ValueError, match="unknown control arm"):
        replace(config, arms=("oracle",)).validate()
