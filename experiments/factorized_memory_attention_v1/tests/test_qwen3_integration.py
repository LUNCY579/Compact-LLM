import pytest
import torch

transformers = pytest.importorskip("transformers", reason="native Qwen3 exact-host dependency absent")
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.factorized_memory_attention_v1.fma.config import FMAConfig, ModelShape
from experiments.factorized_memory_attention_v1.fma.memory import FactorizedMemoryBank, module_digest
from experiments.factorized_memory_attention_v1.fma.qwen3 import convert_qwen3_model


def tiny_config():
    cfg = Qwen3Config(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
    )
    cfg._attn_implementation = "eager"
    return cfg


def tiny_shape():
    return ModelShape(67, 32, 2, 4, 2, 8, 1e-6, "qwen3")


def build_pair(gate):
    torch.manual_seed(17)
    baseline = Qwen3ForCausalLM(tiny_config()).eval()
    candidate = Qwen3ForCausalLM(tiny_config()).eval()
    candidate.load_state_dict(baseline.state_dict())
    bank = FactorizedMemoryBank(tiny_shape(), FMAConfig(rank=4, layers=(0, 1), gate=gate))
    record = convert_qwen3_model(candidate, bank, layers=(0, 1))
    return baseline, candidate, bank, record


def test_g0_matches_untouched_native_qwen3_logits():
    baseline, candidate, _, _ = build_pair(0)
    ids = torch.tensor([[4, 5, 6, 7]])
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        expected = baseline(ids, attention_mask=mask, use_cache=False).logits
        got = candidate(ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-6)


def test_g1_never_calls_native_value_projection(monkeypatch):
    _, candidate, bank, _ = build_pair(1)
    for layer in candidate.model.layers:
        def forbidden(*_args, **_kwargs):
            raise AssertionError("native v_proj executed at g=1")
        monkeypatch.setattr(layer.self_attn.v_proj, "forward", forbidden)
    ids = torch.tensor([[4, 5, 6]])
    with torch.inference_mode():
        out = candidate(ids, use_cache=False).logits
    assert torch.isfinite(out).all()
    assert all(float(gate) == 1.0 for gate in bank.gates)


def test_native_conversion_preserves_every_native_attention_tensor():
    baseline, _, _, record = build_pair(0)
    for row in record["records"]:
        layer = row["layer_index"]
        native_keys = sorted(baseline.model.layers[layer].self_attn.state_dict())
        assert row["reused_native_tensors"] == native_keys
        assert {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"} <= set(native_keys)


def test_native_source_model_is_not_mutated_by_candidate_construction():
    torch.manual_seed(17)
    baseline = Qwen3ForCausalLM(tiny_config()).eval()
    before = module_digest(baseline)
    candidate = Qwen3ForCausalLM(tiny_config()).eval()
    candidate.load_state_dict(baseline.state_dict())
    convert_qwen3_model(
        candidate,
        FactorizedMemoryBank(tiny_shape(), FMAConfig(rank=4, layers=(0, 1), gate=0)),
        layers=(0, 1),
    )
    assert module_digest(baseline) == before


def test_cached_decode_matches_full_reforward_for_hard_memory():
    _, candidate, _, _ = build_pair(1)
    ids = torch.tensor([[4, 5, 6]])
    with torch.inference_mode():
        first = candidate(ids, use_cache=True)
        token = first.logits[:, -1].argmax(-1, keepdim=True)
        cached = candidate(token, past_key_values=first.past_key_values, use_cache=True)
        full = candidate(torch.cat((ids, token), dim=1), use_cache=False)
    torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], atol=2e-6, rtol=2e-6)


def test_padded_attention_mask_behavior_remains_causal_at_g0():
    baseline, candidate, _, _ = build_pair(0)
    ids = torch.tensor([[0, 0, 4, 5, 6]])
    mask = (ids != 0).long()
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    with torch.inference_mode():
        expected = baseline(ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
        got = candidate(ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
    torch.testing.assert_close(got[mask.bool()], expected[mask.bool()], atol=2e-6, rtol=2e-6)
