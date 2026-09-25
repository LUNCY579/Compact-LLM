from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from experiments.factorized_memory_attention_v1.fma.calibration import calibrate_frozen_backbone
from experiments.factorized_memory_attention_v1.fma.config import FMAConfig
from experiments.factorized_memory_attention_v1.fma.memory import (
    FactorizedMemoryBank,
    HeadwiseRMSNorm,
    calibration_loss,
    freeze_backbone_for_calibration,
    module_digest,
    teacher_residual,
)
from experiments.factorized_memory_attention_v1.fma.toy import tiny_shape


@pytest.fixture
def bank() -> FactorizedMemoryBank:
    torch.manual_seed(11)
    return FactorizedMemoryBank(tiny_shape(), FMAConfig(rank=4, layers=(0,), gate=0.0))


def test_g0_is_exact_and_g1_structurally_bypasses_standard_value(bank: FactorizedMemoryBank) -> None:
    hidden = torch.randn(2, 3, 24)
    keys = torch.randn(2, 3, 24)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    standard = torch.randn_like(keys)
    calls = {"count": 0}

    def standard_fn(value: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        return standard

    got = bank.value(hidden_states=hidden, token_ids=ids, layer_idx=0, key_pre_rope=keys, standard_value_fn=standard_fn)
    assert torch.equal(got, standard)
    assert calls["count"] == 1

    bank.set_all_gates(1.0)

    def forbidden(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("W_V path was called at g=1")

    hard = bank.value(hidden_states=hidden, token_ids=ids, layer_idx=0, key_pre_rope=keys, standard_value_fn=forbidden)
    assert torch.equal(hard, bank.memory_path(ids, 0, keys))


def test_token_lookup_and_config_derived_shapes(bank: FactorizedMemoryBank) -> None:
    assert tuple(bank.token_factors.weight.shape) == (41, 4)
    assert tuple(bank.projections[0].weight.shape) == (24, 4)
    assert tuple(bank.norms[0].weight.shape) == (8,)
    ids = torch.tensor([[7, 2, 7]], dtype=torch.long)
    raw = bank.raw_memory(ids, 0)
    assert torch.equal(raw[:, 0], raw[:, 2])
    assert not torch.equal(raw[:, 0], raw[:, 1])


def test_headwise_rmsnorm_matches_manual_per_head_calculation() -> None:
    norm = HeadwiseRMSNorm(num_heads=2, head_dim=3, eps=1e-6, affine=True).double()
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    value = torch.tensor([[[1.0, 2.0, 2.0, 3.0, 4.0, 0.0]]], dtype=torch.float64)
    got = norm(value).reshape(1, 1, 2, 3)
    expected = value.reshape(1, 1, 2, 3)
    expected = expected / torch.sqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
    expected = expected * norm.weight
    assert torch.allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_teacher_residual_is_hand_verifiable() -> None:
    standard = torch.tensor([[[3.0, 4.0], [5.0, 9.0]]])
    key = torch.tensor([[[1.0, 1.5], [2.0, 3.0]]])
    assert torch.equal(teacher_residual(standard, key), torch.tensor([[[2.0, 2.5], [3.0, 6.0]]]))


def test_freeze_contract_and_calibration_option(bank: FactorizedMemoryBank) -> None:
    backbone = nn.Linear(24, 24)
    before = module_digest(backbone)
    counts = freeze_backbone_for_calibration(backbone, bank)
    assert counts["backbone_trainable"] == 0
    assert counts["memory_trainable"] == bank.added_parameter_count()
    assert all(not p.requires_grad for p in backbone.parameters())
    assert all(p.requires_grad for p in bank.parameters())
    assert module_digest(backbone) == before

    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    key = torch.randn(1, 3, 24)
    standard = torch.randn(1, 3, 24)
    result = calibrate_frozen_backbone(
        backbone,
        bank,
        [(ids, 0, standard, key)],
        data_ids=("synthetic-teacher-001",),
        steps=2,
    )
    assert result.steps == 2
    assert result.data_ids == ("synthetic-teacher-001",)
    assert result.backbone_trainable == 0
    assert torch.isfinite(torch.tensor(result.final_loss))
    assert module_digest(backbone) == before


def test_calibration_loss_uses_detached_teacher(bank: FactorizedMemoryBank) -> None:
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    key = torch.randn(1, 2, 24, requires_grad=True)
    standard = torch.randn(1, 2, 24, requires_grad=True)
    loss = calibration_loss(bank, ids, 0, standard, key)
    loss.backward()
    assert key.grad is None
    assert standard.grad is None
    assert any(parameter.grad is not None for parameter in bank.parameters())


def test_serialization_reload_and_baseline_nonmutation(bank: FactorizedMemoryBank) -> None:
    baseline = nn.Linear(24, 24)
    baseline_before = module_digest(baseline)
    state = copy.deepcopy(bank.state_dict())
    reloaded = FactorizedMemoryBank(tiny_shape(), FMAConfig(rank=4, layers=(0,), gate=0.0))
    reloaded.load_state_dict(state)
    for name, tensor in bank.state_dict().items():
        assert torch.equal(tensor, reloaded.state_dict()[name]), name
    assert module_digest(baseline) == baseline_before


def test_dtype_transfer_and_finite_values(bank: FactorizedMemoryBank) -> None:
    bank = bank.to(dtype=torch.float64)
    ids = torch.tensor([[0, 40]], dtype=torch.long)
    values = bank.memory(ids, 0)
    assert values.dtype == torch.float64
    assert torch.isfinite(values).all()


def test_invalid_token_ids_are_rejected(bank: FactorizedMemoryBank) -> None:
    with pytest.raises(IndexError):
        bank.memory(torch.tensor([[41]]), 0)
    with pytest.raises(TypeError):
        bank.memory(torch.tensor([[1.0]]), 0)
