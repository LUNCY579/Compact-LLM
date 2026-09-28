from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.fwanda_delta_retention_v1.scripts.fwanda_delta import (
    ActivationAccumulator,
    FisherAccumulator,
    activation_l2_norms,
    allocate_row_budgets,
    apply_optional_prune,
    empirical_row_fisher,
    mask_sha256,
    prune_layer,
    prune_layerwise,
    reconstruct,
)


def test_uniform_fisher_is_uniform_up_to_deterministic_integer_rounding():
    budgets = allocate_row_budgets([1.0, 1.0, 1.0, 1.0], 10, 4)
    assert budgets.tolist() == [3, 3, 2, 2]


def test_larger_fisher_gets_no_smaller_row_budget():
    budgets = allocate_row_budgets([1.0, 2.0, 4.0, 8.0], 8, 4)
    assert budgets.tolist() == sorted(budgets.tolist())


def test_exact_global_budget_preservation():
    budgets = allocate_row_budgets([0.0, 0.0, 100.0], 7, 5)
    assert int(budgets.sum()) == 7
    assert int(budgets.min()) >= 0
    assert int(budgets.max()) <= 5


def test_largest_remainder_rounding_is_deterministic():
    first = allocate_row_budgets([1.0, 1.0, 1.0], 4, 3)
    second = allocate_row_budgets([1.0, 1.0, 1.0], 4, 3)
    assert first.tolist() == second.tolist() == [2, 1, 1]


def test_masks_are_deterministic_and_tie_by_low_column_index():
    delta = torch.ones(3, 4, dtype=torch.float32)
    first = prune_layer(delta, arm="F1", keep_fraction=0.5, activation_norms=torch.ones(4))
    second = prune_layer(delta, arm="F1", keep_fraction=0.5, activation_norms=torch.ones(4))
    assert torch.equal(first.mask, second.mask)
    assert first.mask[0].tolist() == [True, True, False, False]
    assert mask_sha256(first.mask) == mask_sha256(second.mask)


def test_fisher_row_scalar_multiplication_is_a_ranking_noop():
    row = torch.tensor([0.3, 2.0, 1.0, 2.0])
    base_order = torch.argsort(row, descending=True, stable=True)
    scaled_order = torch.argsort(row * 17.0, descending=True, stable=True)
    assert base_order.tolist() == scaled_order.tolist()


def test_activation_score_is_hand_computable():
    activations = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    assert torch.allclose(
        activation_l2_norms(activations), torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)
    )


def test_empirical_fisher_is_hand_computable():
    gradients = torch.tensor([[1.0, -2.0], [3.0, 0.0]])
    assert torch.allclose(
        empirical_row_fisher(gradients), torch.tensor([5.0, 2.0], dtype=torch.float64)
    )


def test_delta_reconstruction_and_mask_application_are_exact():
    base = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    delta = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[True, False], [False, True]])
    result = reconstruct(base, delta, mask)
    assert torch.equal(result, torch.tensor([[11.0, 20.0], [30.0, 44.0]]))
    assert torch.equal(base, torch.tensor([[10.0, 20.0], [30.0, 40.0]]))
    assert torch.equal(delta, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_disabled_pruning_does_not_mutate_or_alias_baseline():
    base = torch.ones(2, 2)
    delta = torch.full((2, 2), 3.0)
    mask = torch.zeros(2, 2, dtype=torch.bool)
    disabled = apply_optional_prune(base, delta, mask, enabled=False)
    disabled[0, 0] = 99.0
    assert base[0, 0].item() == 1.0
    assert torch.equal(disabled, torch.tensor([[99.0, 1.0], [1.0, 1.0]]))


def test_layerwise_streaming_agrees_with_tiny_full_matrix_reference():
    delta = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    norms = torch.tensor([1.0, 2.0, 1.0])
    fisher = torch.tensor([1.0, 9.0])
    reference = prune_layer(
        delta,
        arm="F2",
        keep_fraction=0.5,
        activation_norms=norms,
        row_fisher=fisher,
        layer_name="tiny",
        min_keep=0,
    )
    streamed = list(
        prune_layerwise(
            [{"name": "tiny", "delta": delta, "activation_norms": norms, "row_fisher": fisher}],
            arm="F2",
            keep_fraction=0.5,
        )
    )[0]
    assert torch.equal(reference.mask, streamed.mask)
    assert torch.equal(reference.sparse_delta, streamed.sparse_delta)
    assert reference.record() == streamed.record()


def test_dtype_behavior_is_explicit():
    delta = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
    result = prune_layer(delta, arm="F1", keep_fraction=0.5, activation_norms=torch.ones(2))
    assert result.sparse_delta.dtype == torch.float16
    base = torch.zeros_like(delta, dtype=torch.bfloat16)
    reconstructed = reconstruct(base, delta, result.mask)
    assert reconstructed.dtype == torch.bfloat16


def test_nan_and_inf_statistics_are_rejected():
    with pytest.raises(ValueError, match="NaN or Inf"):
        activation_l2_norms(torch.tensor([[float("nan")]]))
    with pytest.raises(ValueError, match="NaN or Inf"):
        empirical_row_fisher(torch.tensor([[float("inf")]]))


def test_streaming_accumulators_match_full_statistics():
    x1 = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    x2 = torch.tensor([[0.0, 5.0]])
    activations = ActivationAccumulator(2)
    activations.update(x1)
    activations.update(x2)
    assert torch.allclose(activations.finalize(), activation_l2_norms(torch.cat([x1, x2])))

    g1 = torch.tensor([[1.0, 2.0]])
    g2 = torch.tensor([[3.0, 0.0]])
    fisher = FisherAccumulator(2)
    fisher.update(g1)
    fisher.update(g2)
    assert torch.allclose(fisher.finalize(), empirical_row_fisher(torch.cat([g1, g2])))


def test_no_sealed_evaluation_access_in_qualification_surface():
    config_path = Path(__file__).parents[1] / "configs" / "qualification.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["sealed_evaluation_access"] is False
    assert config["efficacy_run"]["performed"] is False
    source = (Path(__file__).parents[1] / "scripts" / "fwanda_delta.py").read_text(
        encoding="utf-8"
    )
    assert "run_sealed_evaluation" not in source


def test_all_three_arms_match_the_same_layer_budget():
    delta = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    kwargs = {"keep_fraction": 0.5, "activation_norms": torch.ones(4)}
    f0 = prune_layer(delta, arm="F0", **kwargs)
    f1 = prune_layer(delta, arm="F1", **kwargs)
    f2 = prune_layer(delta, arm="F2", row_fisher=torch.tensor([1.0, 4.0, 9.0]), **kwargs)
    assert [int(item.mask.sum()) for item in (f0, f1, f2)] == [6, 6, 6]
