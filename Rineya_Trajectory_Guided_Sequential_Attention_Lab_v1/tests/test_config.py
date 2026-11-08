from __future__ import annotations

import pytest

from tgsa import TGSAConfig


def test_config_validates_head_partition() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TGSAConfig(embed_dim=63, num_heads=4)


def test_ablation_presets_preserve_mechanism_boundaries() -> None:
    cascade = TGSAConfig.for_ablation(embed_dim=64, num_heads=4, arm="cascade")
    full = TGSAConfig.for_ablation(embed_dim=64, num_heads=4, arm="full")

    assert cascade.trajectory_mode == "none"
    assert not cascade.use_control_injection
    assert not cascade.use_predictor
    assert not cascade.condition_q
    assert not cascade.use_importance_bias

    assert full.trajectory_mode == "learned"
    assert full.use_control_injection
    assert full.use_predictor
    assert full.condition_q and full.condition_k and full.condition_v
    assert full.use_importance_bias


def test_history_control_and_learned_delta_are_parameterization_matched() -> None:
    from tgsa import TrajectoryGuidedSequentialAttention, count_parameters

    history = TGSAConfig.for_ablation(
        embed_dim=64,
        num_heads=4,
        arm="history_control",
        trajectory_rank=16,
    )
    delta = TGSAConfig.for_ablation(
        embed_dim=64,
        num_heads=4,
        arm="learned_delta",
        trajectory_rank=16,
    )

    assert history.control_source == "history_mean"
    assert delta.control_source == "velocity"
    assert count_parameters(TrajectoryGuidedSequentialAttention(history)) == count_parameters(
        TrajectoryGuidedSequentialAttention(delta)
    )
