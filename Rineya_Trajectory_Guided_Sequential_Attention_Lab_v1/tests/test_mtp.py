from __future__ import annotations

import torch
import torch.nn.functional as F

from tgsa.mtp import (
    MTPConfig,
    MultiTokenPredictionHeads,
    build_offset_targets,
    joint_causal_mtp_loss,
)


def test_future_target_alignment_and_masking() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    target = torch.tensor([[False, False, True, True, True, True]])
    labels = build_offset_targets(
        input_ids,
        2,
        attention_mask=attention,
        target_mask=target,
    )
    assert labels.tolist() == [[3, 4, 5, 6, -100, -100]]

    duplicate = build_offset_targets(
        input_ids,
        3,
        target_mode="duplicate_next",
        attention_mask=attention,
        target_mask=target,
    )
    # Same number of source positions as offset 3, but every label is next-token.
    assert duplicate.tolist() == [[-100, 3, 4, -100, -100, -100]]


def test_shuffled_targets_are_fixed_point_free_and_receiver_masked() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    target = torch.tensor([[False, True, True, True], [False, True, True, True]])
    donor = torch.tensor([1, 0])
    labels = build_offset_targets(
        input_ids,
        2,
        target_mode="shuffled_batch",
        target_mask=target,
        donor_index=donor,
    )
    assert labels.tolist() == [[7, 8, -100, -100], [3, 4, -100, -100]]


def test_zero_init_residual_heads_begin_in_parent_vocabulary_geometry() -> None:
    torch.manual_seed(7)
    config = MTPConfig(hidden_size=8, vocab_size=13, max_offset=4, rank=4)
    heads = MultiTokenPredictionHeads(config)
    hidden = torch.randn(2, 5, 8)
    weight = torch.randn(13, 8)
    expected = F.linear(hidden, weight)
    outputs = heads(hidden, output_weight=weight)
    assert set(outputs) == {2, 3, 4}
    for logits in outputs.values():
        assert torch.equal(logits, expected)


def test_joint_loss_reaches_auxiliary_and_backbone_parameters() -> None:
    torch.manual_seed(9)
    config = MTPConfig(
        hidden_size=8,
        vocab_size=17,
        max_offset=3,
        rank=4,
        loss_weight=0.4,
    )
    heads = MultiTokenPredictionHeads(config)
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    weight = torch.randn(17, 8, requires_grad=True)
    main = F.linear(hidden, weight)
    mtp = heads(hidden, output_weight=weight)
    input_ids = torch.randint(0, 17, (2, 7))
    target = torch.ones_like(input_ids, dtype=torch.bool)
    loss = joint_causal_mtp_loss(main, mtp, input_ids, config, target_mask=target)
    loss.total.backward()
    assert heads.heads["2"].up.weight.grad is not None
    assert float(heads.heads["2"].up.weight.grad.abs().sum()) > 0.0
    assert hidden.grad is not None and float(hidden.grad.abs().sum()) > 0.0
    assert weight.grad is not None and float(weight.grad.abs().sum()) > 0.0


def test_detached_mtp_does_not_add_backbone_gradient() -> None:
    torch.manual_seed(11)
    config = MTPConfig(
        hidden_size=6,
        vocab_size=11,
        max_offset=3,
        rank=3,
        detach_backbone=True,
    )
    heads = MultiTokenPredictionHeads(config)
    hidden = torch.randn(1, 6, 6, requires_grad=True)
    weight = torch.randn(11, 6, requires_grad=True)
    outputs = heads(hidden, output_weight=weight.detach())
    input_ids = torch.randint(0, 11, (1, 6))
    # Auxiliary-only backward: detached hidden and detached output weight.
    loss = sum(value.float().square().mean() for value in outputs.values())
    loss.backward()
    assert hidden.grad is None
    assert heads.heads["2"].up.weight.grad is not None
