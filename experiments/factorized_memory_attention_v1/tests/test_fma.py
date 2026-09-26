import io

import pytest
import torch
from torch import nn

from experiments.factorized_memory_attention_v1.fma.config import FMAConfig, ModelShape
from experiments.factorized_memory_attention_v1.fma.control import ParameterMatchedValueAdapter
from experiments.factorized_memory_attention_v1.fma.memory import (
    FactorizedMemoryBank,
    HeadwiseRMSNorm,
    calibration_loss,
    freeze_backbone_for_calibration,
    module_digest,
    teacher_residual,
)
from experiments.factorized_memory_attention_v1.fma.toy import tiny_shape


def bank(gate=0.0):
    torch.manual_seed(7)
    return FactorizedMemoryBank(tiny_shape(), FMAConfig(rank=4, layers=(0,), gate=gate))


def test_g0_is_numerically_identical_and_does_not_touch_memory_path(monkeypatch):
    memory = bank(0)
    hidden = torch.randn(2, 3, 24)
    ids = torch.randint(0, 41, (2, 3))
    key = torch.randn(2, 3, 24)
    value_proj = nn.Linear(24, 24, bias=False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("memory path used at g=0")

    monkeypatch.setattr(memory, "memory_path", forbidden)
    got = memory.value(
        hidden_states=hidden,
        token_ids=ids,
        layer_idx=0,
        key_pre_rope=key,
        standard_value_fn=value_proj,
    )
    assert torch.equal(got, value_proj(hidden))


def test_g1_structurally_bypasses_standard_value_projection():
    memory = bank(1)
    hidden = torch.randn(1, 2, 24)
    ids = torch.tensor([[3, 5]])
    key = torch.randn(1, 2, 24)

    def forbidden(_hidden):
        raise AssertionError("W_V was called at g=1")

    got = memory.value(
        hidden_states=hidden,
        token_ids=ids,
        layer_idx=0,
        key_pre_rope=key,
        standard_value_fn=forbidden,
    )
    torch.testing.assert_close(got, key + memory.memory(ids, 0))


def test_token_lookup_indexes_exact_vocabulary_rows():
    memory = bank(1)
    with torch.no_grad():
        memory.token_factors.weight.zero_()
        memory.token_factors.weight[2].fill_(1)
        memory.token_factors.weight[9].fill_(3)
        memory.projections[0].weight.fill_(1)
    raw = memory.raw_memory(torch.tensor([[2, 9]]), 0)
    assert torch.all(raw[:, 0] == 4)
    assert torch.all(raw[:, 1] == 12)


def test_headwise_rmsnorm_is_independent_per_kv_head():
    norm = HeadwiseRMSNorm(2, 2, 1e-12, affine=False)
    value = torch.tensor([[[3.0, 4.0, 6.0, 8.0]]])
    out = norm(value).reshape(1, 1, 2, 2)
    rms = torch.sqrt(out.square().mean(-1))
    torch.testing.assert_close(rms, torch.ones_like(rms))


def test_teacher_residual_is_hand_verifiable():
    value = torch.tensor([[[4.0, 7.0]]])
    key = torch.tensor([[[1.0, 2.0]]])
    assert torch.equal(teacher_residual(value, key), torch.tensor([[[3.0, 5.0]]]))


def test_factorized_raw_lookup_matches_explicit_dense_equivalent():
    memory = bank(1)
    ids = torch.tensor([[1, 7, 3]])
    dense = memory.materialize_dense_layer(0, max_elements=10_000)
    torch.testing.assert_close(memory.raw_memory(ids, 0), dense[ids])


def test_real_scale_dense_materialization_is_guarded():
    shape = ModelShape(151936, 1024, 28, 16, 8, 128, 1e-6)
    memory = FactorizedMemoryBank(shape, FMAConfig(rank=2, layers=(0,), gate=1))
    with pytest.raises(MemoryError):
        memory.materialize_dense_layer(0)


def test_frozen_backbone_calibration_does_not_mutate_weights():
    backbone = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    memory = bank(0)
    before = module_digest(backbone)
    summary = freeze_backbone_for_calibration(backbone, memory)
    after = module_digest(backbone)
    assert summary["backbone_trainable"] == 0
    assert summary["memory_trainable"] == memory.added_parameter_count()
    assert before == after


def test_calibration_loss_has_no_backbone_gradient_when_frozen():
    backbone = nn.Linear(24, 24, bias=False)
    memory = bank(0)
    freeze_backbone_for_calibration(backbone, memory)
    ids = torch.tensor([[1, 2]])
    hidden = torch.randn(1, 2, 24)
    standard = backbone(hidden)
    key = torch.randn_like(standard)
    calibration_loss(memory, ids, 0, standard, key).backward()
    assert backbone.weight.grad is None
    assert any(parameter.grad is not None for parameter in memory.parameters())


def test_serialization_reload_round_trip():
    memory = bank(0.37)
    ids = torch.tensor([[1, 4, 8]])
    expected = memory.memory(ids, 0)
    payload = io.BytesIO()
    torch.save(memory.state_dict(), payload)
    payload.seek(0)
    restored = bank(0)
    restored.load_state_dict(torch.load(payload, weights_only=True))
    torch.testing.assert_close(expected, restored.memory(ids, 0))
    assert float(restored.gates[0]) == pytest.approx(0.37)


def test_dtype_device_transfer_and_finiteness():
    memory = bank(1).to(dtype=torch.float64)
    ids = torch.tensor([[1, 2]])
    key = torch.randn(1, 2, 24, dtype=torch.float64)
    hidden = torch.randn_like(key)
    out = memory.value(
        hidden_states=hidden,
        token_ids=ids,
        layer_idx=0,
        key_pre_rope=key,
        standard_value_fn=lambda x: x,
    )
    assert out.dtype == torch.float64
    assert torch.isfinite(out).all()


def test_parameter_matched_control_has_nonzero_capacity():
    shape = tiny_shape()
    control = ParameterMatchedValueAdapter(shape, FMAConfig(rank=4, layers=(0,)))
    assert control.added_parameter_count() > 0
