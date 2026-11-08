"""Long-context-safe telemetry contracts for TGSA."""

from __future__ import annotations

import pytest
import torch

from tgsa import TGSAConfig, TrajectoryGuidedSequentialAttention


def _operator(*, detach_telemetry: bool) -> TrajectoryGuidedSequentialAttention:
    config = TGSAConfig.for_ablation(
        embed_dim=32,
        num_heads=4,
        arm="learned_delta",
        dropout=0.0,
        telemetry_level="control_only",
        detach_telemetry=detach_telemetry,
    )
    return TrajectoryGuidedSequentialAttention(config)


def test_control_only_retains_exact_control_without_quadratic_tensors() -> None:
    module = _operator(detach_telemetry=False)
    hidden = torch.randn(2, 11, 32, requires_grad=True)

    control_output, control = module(
        hidden,
        telemetry_level="control_only",
    )
    full_output, full = module(
        hidden,
        telemetry_level="full",
    )

    assert torch.equal(control_output, full_output)
    assert control.control_signals is not None
    assert full.control_signals is not None
    assert torch.equal(control.control_signals, full.control_signals)
    assert control.control_signals.shape == (2, 4, 11, 8)
    assert control.control_signals.requires_grad

    assert control.carrier_states is None
    assert control.velocities is None
    assert control.importance_scores is None
    assert control.attention_maps is None
    assert full.attention_maps is not None
    assert full.attention_maps.shape == (2, 4, 11, 11)

    # The retained control tensor is O(B*H*T*D_h); a full attention map is
    # O(B*H*T^2). This assertion protects the intended memory contract.
    assert control.control_signals.numel() == 2 * 4 * 11 * 8
    assert full.attention_maps.numel() == 2 * 4 * 11 * 11


def test_control_only_honors_detach_telemetry() -> None:
    module = _operator(detach_telemetry=True)
    hidden = torch.randn(2, 7, 32, requires_grad=True)
    output, telemetry = module(
        hidden,
        telemetry_level="control_only",
    )

    assert output.requires_grad
    assert telemetry.control_signals is not None
    assert not telemetry.control_signals.requires_grad
    assert not telemetry.velocity_rms.requires_grad


def test_control_only_preserves_summary_metrics() -> None:
    module = _operator(detach_telemetry=True)
    hidden = torch.randn(3, 5, 32)
    _output, telemetry = module(
        hidden,
        telemetry_level="control_only",
    )

    for name in (
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
        assert getattr(telemetry, name).shape == (3, 4)


def test_telemetry_level_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="unsupported telemetry_level"):
        TGSAConfig(
            embed_dim=32,
            num_heads=4,
            telemetry_level="attention_only",  # type: ignore[arg-type]
        )
