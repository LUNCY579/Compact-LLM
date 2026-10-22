"""Actual tiny random Qwen3 architecture checks; still NOT Rineya efficacy."""
import importlib.util
import pytest
import torch
from experiments.rineya_triadic_working_memory_v1.core import MemoryConfig
from experiments.rineya_triadic_working_memory_v1.sidecar import MemorySidecar
from experiments.rineya_triadic_working_memory_v1.native import FrozenQwenSidecar, greedy_decode


@pytest.fixture
def native():
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers==4.57.1 is unavailable; native Qwen3 test NOT executed")
    import transformers
    if transformers.__version__ != "4.57.1":
        pytest.skip("only transformers==4.57.1 is qualified by this contract")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(8)
    c = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                   num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                   head_dim=8, attention_dropout=0.0, max_position_embeddings=128,
                   pad_token_id=0, bos_token_id=1, eos_token_id=2)
    c._attn_implementation = "eager"
    base = Qwen3ForCausalLM(c).eval()
    memory = MemorySidecar(MemoryConfig(d_model=32, heads=2, key_dim=4, value_dim=4))
    wrapper = FrozenQwenSidecar(base, memory, 1).eval()
    yield wrapper, torch.tensor([[3, 7, 5, 9, 12, 8]])
    wrapper.close()


def test_native_qwen3_zero_gate_identity(native):
    model, ids = native
    with torch.no_grad():
        assert torch.equal(model(ids).logits, model(ids, enabled=False).logits)


def test_native_qwen3_active_prefill_decode_equivalence(native):
    model, ids = native
    with torch.no_grad():
        model.sidecar.gate.fill_(0.2)
        full = model(ids).logits
        cache, parts = None, []
        for i in range(ids.shape[1]):
            out = model(ids[:, i:i+1], use_cache=True, cache=cache)
            cache = out.cache
            parts.append(out.logits)
        torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-5, rtol=1e-5)


def test_native_qwen3_greedy_equality(native):
    model, ids = native
    model.sidecar.gate.data.fill_(0.2)
    assert greedy_decode(model, ids, 8, cached=True) == greedy_decode(model, ids, 8, cached=False)


def test_native_qwen3_frozen_backward(native):
    model, ids = native
    model.train()
    model(ids).logits.square().mean().backward()
    model.assert_frozen()
    assert model.sidecar.gate.grad is not None and model.sidecar.gate.grad.abs() > 0
