from __future__ import annotations

import pytest
import torch

from rineya_pcsm.attention import PeerAttention


def test_attention_is_similarity_plus_persistable_residual_and_alphas() -> None:
    attention = PeerAttention(4, residual_hidden=3)
    result = attention(
        "layers.1.weight",
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        {
            "layers.2.weight": torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            "layers.0.weight": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        },
    )
    assert result.peer_names == ("layers.0.weight", "layers.2.weight")
    assert torch.equal(result.residuals, torch.zeros(2))
    assert torch.allclose(result.logits, result.similarities)
    assert result.alphas[0] > result.alphas[1]
    assert result.alphas.sum().item() == pytest.approx(1.0)
    assert result.record()["alphas"] == result.alphas.tolist()
    assert result.record()["self_alpha"] == 0.0
    assert torch.equal(result.depth_distances, torch.zeros(2))


def test_attention_rejects_self_advice() -> None:
    attention = PeerAttention(2)
    with pytest.raises(ValueError, match="self advice"):
        attention("target", torch.ones(2), {"target": torch.ones(2)})


def test_attention_residual_path_receives_meta_gradient() -> None:
    attention = PeerAttention(3, residual_hidden=4)
    result = attention(
        "target",
        torch.tensor([1.0, 0.0, 0.0]),
        {"a": torch.ones(3), "b": -torch.ones(3)},
    )
    result.alphas[0].backward()
    final = attention.residual_network[-1]
    assert isinstance(final, torch.nn.Linear)
    assert final.weight.grad is not None
    assert torch.count_nonzero(final.weight.grad) > 0


def test_attention_depth_distance_is_transparent_and_persisted() -> None:
    attention = PeerAttention(2, temperature=2.0, depth_penalty=3.0)
    result = attention(
        "target",
        torch.tensor([1.0, 0.0]),
        {
            "far": torch.tensor([1.0, 0.0]),
            "near": torch.tensor([1.0, 0.0]),
        },
        target_depth=0.5,
        peer_depths={"far": 1.0, "near": 0.6},
    )
    assert result.peer_names == ("far", "near")
    assert torch.allclose(result.depth_distances, torch.tensor([0.5, 0.1]))
    assert torch.allclose(
        result.logits,
        result.similarities / 2.0 - 3.0 * result.depth_distances,
    )
    assert result.alphas[1] > result.alphas[0]
    assert result.record()["depth_distances"] == pytest.approx([0.5, 0.1])


def test_attention_depth_metadata_must_exactly_match_peers() -> None:
    attention = PeerAttention(2, depth_penalty=1.0)
    with pytest.raises(ValueError, match="exactly match"):
        attention(
            "target",
            torch.ones(2),
            {"peer": torch.ones(2)},
            target_depth=0.5,
            peer_depths={"wrong": 0.4},
        )
