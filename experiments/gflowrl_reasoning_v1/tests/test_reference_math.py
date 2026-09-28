from __future__ import annotations

import pytest
import torch

from experiments.gflowrl_reasoning_v1.objective import GFlowRLConfig, RolloutBatch, compute_gflowrl_loss
from experiments.gflowrl_reasoning_v1.reference_math import reference_gflowrl_loss


def test_torch_objective_matches_independent_scalar_reference_with_masking():
    current = [[-0.4, -0.2, -0.1, -0.3], [-2.0, -1.0, -3.0, -0.5]]
    old = [[-0.5, -0.4, -0.2, -0.25], [-2.2, -1.1, -2.8, -0.7]]
    reference = [[-0.6, -0.1, -0.3, -0.5], [-1.8, -1.5, -2.9, -0.9]]
    rewards = [[0.0, 1.0, 0.5, -0.5], [1.0, 0.0, 1.0, 0.0]]
    lengths = [[2.0, 4.0, 1.0, 3.0], [5.0, 2.0, 6.0, 3.0]]
    valid = [[True, True, False, True], [True, False, True, True]]
    config = GFlowRLConfig(beta=0.7, epsilon_low=0.2, epsilon_high=0.28, importance_clip=2.0)

    current_tensor = torch.tensor(current, dtype=torch.float64, requires_grad=True)
    batch = RolloutBatch(
        current_tensor,
        torch.tensor(old, dtype=torch.float64),
        torch.tensor(reference, dtype=torch.float64),
        torch.tensor(rewards, dtype=torch.float64),
        torch.tensor(lengths, dtype=torch.float64),
        torch.tensor(valid, dtype=torch.bool),
    )
    got = compute_gflowrl_loss(batch, config)
    expected = reference_gflowrl_loss(
        current, old, reference, rewards, lengths, valid,
        beta=0.7, epsilon_low=0.2, epsilon_high=0.28, importance_clip=2.0,
    )

    assert got.loss.item() == pytest.approx(expected.loss, abs=1e-12)
    assert got.diagnostics["partition_log_z"].tolist() == pytest.approx(expected.partition_log_z)
    assert got.diagnostics["valid_counts"].tolist() == list(expected.valid_counts)


def test_scalar_reference_all_invalid_group_is_zero_cost():
    result = reference_gflowrl_loss(
        [[0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0]],
        [[1.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0, 1.0]],
        [[False, False, False, False]],
    )
    assert result.loss == 0.0
    assert result.valid_counts == (0,)
