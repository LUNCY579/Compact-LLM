from pathlib import Path
import io
import json

import pytest
import torch
from torch import nn

from experiments.factorized_memory_attention_v1.fma.budget import parameter_budget
from experiments.factorized_memory_attention_v1.fma.config import FMAConfig, ModelShape, shape_from_dict
from experiments.factorized_memory_attention_v1.fma.control import ParameterMatchedValueAdapter
from experiments.factorized_memory_attention_v1.fma.calibration import native_qwen3_teacher
from experiments.factorized_memory_attention_v1.fma.memory import (
    FactorizedMemoryBank,
    HeadwiseRMSNorm,
    calibration_loss,
    freeze_backbone_for_calibration,
    module_digest,
    teacher_residual,
)
from experiments.factorized_memory_attention_v1.fma.toy import ToyLM, tiny_shape


def bank(g=0.0, r=4):
    return FactorizedMemoryBank(tiny_shape(), FMAConfig(rank=r, layers=(0,), gate=g))


def supa_shape():
    return ModelShape(151936, 1024, 28, 16, 8, 128, 1e-6, "qwen3")


def test_01_g0_exact_standard_path():
    torch.manual_seed(1)
    b = bank(0)
    h = torch.randn(2, 3, 24)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    k = torch.randn(2, 3, 24)
    v = nn.Linear(24, 24, bias=False)
    actual = b.value(hidden_states=h, token_ids=ids, layer_idx=0, key_pre_rope=k, standard_value_fn=v)
    assert torch.equal(actual, v(h))


def test_02_g1_structurally_bypasses_wv():
    b = bank(1)
    h = torch.randn(1, 2, 24)
    ids = torch.tensor([[1, 2]])
    k = torch.randn(1, 2, 24)

    def forbidden(_):
        raise AssertionError("W_V executed at g=1")

    assert b.value(hidden_states=h, token_ids=ids, layer_idx=0, key_pre_rope=k, standard_value_fn=forbidden).shape == k.shape


def test_03_token_ids_are_exact_lookup_indices():
    b = bank(1, 2)
    with torch.no_grad():
        b.token_factors.weight.zero_()
        b.token_factors.weight[3] = torch.tensor([1.0, 2.0])
        b.token_factors.weight[7] = torch.tensor([4.0, 5.0])
        b.projections[0].weight.zero_()
        b.projections[0].weight[0, 0] = 1
        b.projections[0].weight[1, 1] = 1
    x = b.raw_memory(torch.tensor([[3, 7]]), 0)
    assert x[0, 0, :2].tolist() == [1.0, 2.0]
    assert x[0, 1, :2].tolist() == [4.0, 5.0]


def test_04_shapes_derive_from_actual_config():
    p = Path(__file__).resolve().parents[3] / "Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze/config.json"
    raw = json.loads(p.read_text())
    s = shape_from_dict(raw)
    assert s.value_width == raw["num_key_value_heads"] * raw["head_dim"]
    assert s.vocab_size == raw["vocab_size"]
    assert s.num_hidden_layers == raw["num_hidden_layers"]


def test_05_06_exact_budget_and_rank_scaling():
    s = supa_shape()
    b = parameter_budget(s, FMAConfig(rank=32), baseline_params=596049920)
    b16 = parameter_budget(s, FMAConfig(rank=16))
    assert (b.shared_token_table_params, b.layer_projection_params, b.normalization_params, b.added_fma_params) == (4861952, 917504, 3584, 5783040)
    assert b.removed_standard_value_params == 29360128
    assert b.hard_replacement_model_total_params == 572472832
    assert b.added_fma_params - b16.added_fma_params == s.vocab_size * 16 + s.num_hidden_layers * 16 * s.value_width


def test_07_headwise_rmsnorm_semantics():
    n = HeadwiseRMSNorm(2, 2, 1e-6, affine=False)
    y = n(torch.tensor([[[3.0, 4.0, 0.0, 10.0]]])).reshape(1, 1, 2, 2)
    assert torch.allclose(torch.sqrt(y.square().mean(-1)), torch.ones(1, 1, 2), atol=1e-5)


def test_08_native_source_preserves_backend_mask_and_raw_key_ordering():
    src = (Path(__file__).parents[1] / "fma/qwen3.py").read_text()
    raw = src.index("key_raw_heads = self.k_proj")
    val = src.index("key_pre_rope = key_raw_heads")
    norm = src.index("key_for_scores = self.k_norm(key_raw_heads)")
    rope = src.index("apply_rotary_pos_emb(query, key_for_scores")
    assert raw < val < norm < rope
    assert "return super().forward(" in src
    assert "ALL_ATTENTION_FUNCTIONS.get_interface" in src
    assert "attention_mask," in src


def test_08b_calibration_teacher_uses_raw_key_before_qk_norm():
    class ForbiddenNorm(nn.Module):
        def forward(self, _value):
            raise AssertionError("k_norm must not be used by the MA teacher target")

    class TinyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.v_proj = nn.Linear(4, 4, bias=False)
            self.k_proj = nn.Linear(4, 4, bias=False)
            self.k_norm = ForbiddenNorm()

    torch.manual_seed(3)
    attn = TinyAttention()
    hidden = torch.randn(2, 3, 4)
    standard, key_raw = native_qwen3_teacher(attn, hidden)
    torch.testing.assert_close(standard, attn.v_proj(hidden))
    torch.testing.assert_close(key_raw, attn.k_proj(hidden))


def make_lm(g):
    torch.manual_seed(42)
    return ToyLM(tiny_shape(), FMAConfig(rank=4, layers=(0,), gate=g)).eval()


def test_09_no_future_leakage():
    for g in (0.0, 1.0):
        m = make_lm(g)
        a, _ = m(torch.tensor([[1, 2, 3, 4]]))
        b, _ = m(torch.tensor([[1, 2, 9, 10]]))
        assert torch.allclose(a[:, :2], b[:, :2], atol=1e-6, rtol=1e-6)


def test_10_cache_semantics():
    for g in (0.0, 1.0):
        m = make_lm(g)
        ids = torch.tensor([[1, 2, 3, 4]])
        full, _ = m(ids)
        past = None
        parts = []
        for i in range(ids.shape[1]):
            x, past = m(ids[:, i:i+1], past=past, use_cache=True)
            parts.append(x)
        assert torch.allclose(full, torch.cat(parts, 1), atol=2e-6, rtol=2e-6)


def test_11_serialization_round_trip():
    torch.manual_seed(2)
    a = bank(0.37)
    ids = torch.tensor([[1, 2]])
    buf = io.BytesIO()
    torch.save(a.state_dict(), buf)
    buf.seek(0)
    b = bank(0)
    b.load_state_dict(torch.load(buf, weights_only=True))
    assert torch.equal(a.memory(ids, 0), b.memory(ids, 0))
    assert torch.equal(a.gates, b.gates)


def test_12_construction_does_not_mutate_baseline():
    base = nn.Linear(24, 24, bias=False)
    digest = module_digest(base)
    _ = bank()
    assert module_digest(base) == digest


def test_13_frozen_backbone_contract():
    base = nn.Sequential(nn.Linear(24, 32), nn.Linear(32, 24))
    b = bank()
    stats = freeze_backbone_for_calibration(base, b)
    assert stats["backbone_trainable"] == 0
    assert stats["memory_trainable"] == b.added_parameter_count()
    assert all(not p.requires_grad for p in base.parameters())


def test_14_teacher_residual_and_calibration_loss():
    expected = torch.tensor([[[3.0, 4.0]]])
    assert torch.equal(teacher_residual(torch.tensor([[[5.0, 7.0]]]), torch.tensor([[[2.0, 3.0]]])), expected)
    b = bank()
    loss = calibration_loss(b, torch.tensor([[1, 2]]), 0, torch.randn(1, 2, 24), torch.randn(1, 2, 24))
    loss.backward()
    assert torch.isfinite(loss)


def test_15_dense_toy_equivalence_and_dense_guard():
    b = bank(1, 3)
    ids = torch.tensor([[0, 4, 8, 4]])
    dense = b.materialize_dense_layer(0, max_elements=100000)
    assert torch.allclose(b.raw_memory(ids, 0), dense[ids], atol=1e-7, rtol=0)
    with pytest.raises(MemoryError):
        b.materialize_dense_layer(0, max_elements=100)


def test_16_17_dtype_device_and_finite():
    out = bank().to(dtype=torch.bfloat16).memory(torch.tensor([[1, 2]]), 0)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_18_micro_generation():
    for g in (0.0, 1.0):
        out = make_lm(g).generate(torch.tensor([[1, 2, 3]]), 4)
        assert out.shape == (1, 7)
        assert int(out.min()) >= 0
        assert int(out.max()) < tiny_shape().vocab_size


def test_19_resource_and_matched_control_math():
    b = parameter_budget(supa_shape(), FMAConfig(rank=32))
    c = ParameterMatchedValueAdapter(supa_shape(), FMAConfig(rank=32))
    assert b.full_ma_params == 4356312576
    assert b.full_ma_bf16_bytes == 8712625152
    assert b.full_ma_bf16_bytes / (1024**3) > 8
    assert c.rank == 101
    assert c.added_parameter_count() == 5791744
    assert abs(b.control_budget_delta_fraction) < 0.002


def test_20_no_sealed_access_declared_in_manifest():
    manifest = json.loads((Path(__file__).parents[1] / "MANIFEST.json").read_text())
    assert manifest["sealed_data_access"] is False


def test_native_qwen3_exact_host_is_explicitly_blocked_here():
    try:
        import transformers  # noqa: F401
    except Exception:
        pytest.skip("Transformers absent; exact-host native qualification BLOCKED")
    pytest.skip("Native checkpoint/RTX 4070 not materialized in this agent runtime")
