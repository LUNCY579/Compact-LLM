from __future__ import annotations

import pytest
import torch

from tgsa import (
    ParallelCausalSelfAttention,
    TGSAConfig,
    TrajectoryGuidedSequentialAttention,
)


def make_config(**updates: object) -> TGSAConfig:
    base = TGSAConfig(
        embed_dim=32,
        num_heads=4,
        trajectory_rank=8,
        dropout=0.0,
        telemetry_level="summary",
        detach_telemetry=False,
    )
    return base.with_updates(**updates)


def test_forward_shape_and_summary_telemetry() -> None:
    torch.manual_seed(7)
    model = TrajectoryGuidedSequentialAttention(make_config())
    inputs = torch.randn(2, 6, 32)

    output, telemetry = model(inputs, telemetry_level="summary")

    assert output.shape == inputs.shape
    for field_name in (
        "velocity_rms",
        "control_signal_rms",
        "acceleration_rms",
        "velocity_cosine",
        "carrier_update_rms",
        "state_gate_mean",
        "control_gate_mean",
        "predictor_gate_mean",
        "correction_gate_mean",
        "importance_mean",
        "importance_min",
        "importance_max",
        "attention_entropy",
    ):
        assert getattr(telemetry, field_name).shape == (2, 4)
    assert torch.allclose(telemetry.velocity_rms[:, 0], torch.zeros(2), atol=1.0e-7)
    assert torch.all((telemetry.importance_min >= 0) & (telemetry.importance_max <= 1))
    assert telemetry.carrier_states is None


def test_full_telemetry_shapes() -> None:
    torch.manual_seed(8)
    model = TrajectoryGuidedSequentialAttention(make_config())
    inputs = torch.randn(2, 5, 32)

    _, telemetry = model(inputs, telemetry_level="full")

    assert telemetry.carrier_states is not None
    assert telemetry.velocities is not None
    assert telemetry.control_signals is not None
    assert telemetry.importance_scores is not None
    assert telemetry.attention_maps is not None
    assert telemetry.carrier_states.shape == (2, 5, 5, 8)
    assert telemetry.velocities.shape == (2, 4, 5, 8)
    assert telemetry.control_signals.shape == (2, 4, 5, 8)
    assert telemetry.importance_scores.shape == (2, 4, 5, 1)
    assert telemetry.attention_maps.shape == (2, 4, 5, 5)


def test_backward_reaches_learned_trajectory_path() -> None:
    torch.manual_seed(11)
    model = TrajectoryGuidedSequentialAttention(make_config())
    inputs = torch.randn(2, 7, 32, requires_grad=True)

    output = model(inputs, telemetry_level="none")
    loss = output.float().square().mean()
    loss.backward()

    trajectory_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "trajectory_encoders" in name and parameter.requires_grad
    ]
    assert trajectory_gradients
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.abs().sum() > 0
        for gradient in trajectory_gradients
    )
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize(
    "arm",
    [
        "cascade",
        "history_control",
        "raw_delta",
        "learned_delta",
        "qkv_conditioned",
        "predictor_corrector",
        "full",
    ],
)
def test_every_preregistered_ablation_executes(arm: str) -> None:
    torch.manual_seed(13)
    config = TGSAConfig.for_ablation(
        embed_dim=32,
        num_heads=4,
        arm=arm,  # type: ignore[arg-type]
        trajectory_rank=8,
        dropout=0.0,
        telemetry_level="summary",
    )
    model = TrajectoryGuidedSequentialAttention(config)
    output, telemetry = model(torch.randn(2, 5, 32), telemetry_level="summary")
    assert output.shape == (2, 5, 32)
    assert torch.isfinite(output).all()
    assert torch.isfinite(telemetry.carrier_update_rms).all()


def test_causal_prefix_is_invariant_to_future_changes() -> None:
    torch.manual_seed(17)
    model = TrajectoryGuidedSequentialAttention(make_config()).eval()
    original = torch.randn(2, 8, 32)
    changed = original.clone()
    changed[:, 5:, :] = torch.randn_like(changed[:, 5:, :]) * 20

    original_output = model(original, telemetry_level="none")
    changed_output = model(changed, telemetry_level="none")

    torch.testing.assert_close(original_output[:, :5], changed_output[:, :5], atol=1.0e-6, rtol=1.0e-5)


def test_noncausal_mode_can_respond_to_future_changes() -> None:
    torch.manual_seed(19)
    model = TrajectoryGuidedSequentialAttention(make_config()).eval()
    original = torch.randn(1, 7, 32)
    changed = original.clone()
    changed[:, 4:, :] += 10

    original_output = model(original, is_causal=False, telemetry_level="none")
    changed_output = model(changed, is_causal=False, telemetry_level="none")

    assert not torch.allclose(original_output[:, :4], changed_output[:, :4])


def test_key_padding_mask_and_additive_bias_are_supported() -> None:
    torch.manual_seed(23)
    model = TrajectoryGuidedSequentialAttention(make_config())
    inputs = torch.randn(2, 6, 32)
    key_padding_mask = torch.tensor(
        [[False, False, False, False, True, True], [False, False, False, False, False, True]]
    )
    bias = torch.zeros(6, 6)
    bias[:, 0] = 0.25

    output = model(
        inputs,
        attention_bias=bias,
        key_padding_mask=key_padding_mask,
        telemetry_level="none",
    )

    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()


def test_eval_is_deterministic() -> None:
    torch.manual_seed(29)
    model = TrajectoryGuidedSequentialAttention(make_config(dropout=0.2)).eval()
    inputs = torch.randn(2, 6, 32)
    first = model(inputs, telemetry_level="none")
    second = model(inputs, telemetry_level="none")
    torch.testing.assert_close(first, second)


def test_parallel_control_is_causal_and_differentiable() -> None:
    torch.manual_seed(31)
    model = ParallelCausalSelfAttention(32, 4, dropout=0.0).eval()
    original = torch.randn(2, 8, 32, requires_grad=True)
    changed = original.detach().clone()
    changed[:, 5:, :] += 20

    original_output = model(original)
    changed_output = model(changed)
    torch.testing.assert_close(original_output[:, :5], changed_output[:, :5], atol=1.0e-6, rtol=1.0e-5)

    original_output.square().mean().backward()
    assert original.grad is not None and torch.isfinite(original.grad).all()


def test_invalid_mask_contracts_raise() -> None:
    model = TrajectoryGuidedSequentialAttention(make_config())
    inputs = torch.randn(2, 4, 32)
    with pytest.raises(TypeError, match="floating point"):
        model(inputs, attention_bias=torch.ones(4, 4, dtype=torch.bool), telemetry_level="none")
    with pytest.raises(TypeError, match="bool"):
        model(inputs, key_padding_mask=torch.zeros(2, 4), telemetry_level="none")
