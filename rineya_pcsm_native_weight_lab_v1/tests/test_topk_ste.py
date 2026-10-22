from __future__ import annotations

import pytest
import torch

from rineya_pcsm.topk import (
    budget_to_k,
    flat_support_indices,
    freeze_support,
    stable_topk_mask,
    support_union_stats,
    topk_ste,
)


def test_stable_topk_has_exact_k_and_breaks_ties_by_flat_index() -> None:
    scores = torch.tensor([[2.0, 2.0], [2.0, 1.0]])
    mask = stable_topk_mask(scores, 2)
    assert torch.equal(mask, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    assert mask.sum().item() == 2
    frozen = freeze_support(scores, 2)
    assert torch.equal(frozen.flat_indices, torch.tensor([0, 1]))
    assert frozen.k == 2


def test_topk_ste_forward_is_bitwise_binary_and_backward_is_smooth() -> None:
    scores = torch.tensor([0.2, 0.8, 0.4], requires_grad=True)
    mask = topk_ste(scores, 1)
    assert torch.equal(mask, torch.tensor([0.0, 1.0, 0.0]))
    weights = torch.tensor([3.0, 5.0, 7.0])
    (mask * weights).sum().backward()
    assert scores.grad is not None
    assert torch.all(scores.grad > 0)
    assert not torch.equal(scores.grad, weights)


def test_union_stats_report_per_target_k_and_true_union() -> None:
    masks = {
        "a": torch.tensor([1, 1, 0, 0], dtype=torch.bool),
        "b": torch.tensor([0, 1, 1, 0], dtype=torch.bool),
    }
    stats = support_union_stats(masks)
    assert stats["per_target_k"] == {"a": 2, "b": 2}
    assert stats["sum_k"] == 4
    assert stats["union_k"] == 3
    assert stats["intersection_k"] == 1


def test_budget_and_binary_validation_fail_closed() -> None:
    assert budget_to_k(101, 0.01) == 1
    assert budget_to_k(150, 0.01) == 2
    with pytest.raises(ValueError, match="binary"):
        flat_support_indices(torch.tensor([0.0, 0.5, 1.0]))
