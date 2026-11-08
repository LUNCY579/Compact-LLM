from __future__ import annotations

import math

import pytest
import torch

from tgsa import (
    TGSAConfig,
    TrajectoryGuidedSequentialAttention,
    count_parameters,
    paired_bootstrap_mean_difference,
    telemetry_summary,
)


def test_paired_bootstrap_detects_uniform_improvement() -> None:
    baseline = [0.0, 1.0, 2.0, 3.0]
    candidate = [1.0, 2.0, 3.0, 4.0]
    result = paired_bootstrap_mean_difference(candidate, baseline, samples=500, seed=3)
    assert result.mean_delta == pytest.approx(1.0)
    assert result.confidence_low == pytest.approx(1.0)
    assert result.confidence_high == pytest.approx(1.0)
    assert result.probability_positive == pytest.approx(1.0)


def test_paired_bootstrap_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        paired_bootstrap_mean_difference([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_mean_difference([math.nan], [0.0])


def test_telemetry_summary_and_parameter_count() -> None:
    torch.manual_seed(37)
    config = TGSAConfig(embed_dim=32, num_heads=4, trajectory_rank=8, dropout=0.0)
    model = TrajectoryGuidedSequentialAttention(config)
    _, telemetry = model(torch.randn(2, 6, 32), telemetry_level="summary")
    summary = telemetry_summary(telemetry)

    assert count_parameters(model) > 0
    assert set(summary) == {
        "velocity_rms_mean",
        "control_signal_rms_mean",
        "carrier_update_rms_mean",
        "velocity_convergence_fraction",
        "velocity_oscillation_fraction",
        "importance_mean",
        "state_gate_mean",
        "control_gate_mean",
        "predictor_gate_mean",
        "correction_gate_mean",
        "attention_entropy_mean",
    }
    assert 0.0 <= summary["importance_mean"] <= 1.0
