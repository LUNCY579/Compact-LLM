from __future__ import annotations

import pytest
import torch

from experiments.factorized_memory_attention_v1.fma.config import FMAConfig, ModelShape
from experiments.factorized_memory_attention_v1.fma.memory import FactorizedMemoryBank
from experiments.factorized_memory_attention_v1.fma.qwen3 import TRANSFORMERS_AVAILABLE, convert_qwen3_model


def test_native_qwen3_path_is_explicitly_optional() -> None:
    if not TRANSFORMERS_AVAILABLE:
        pytest.skip("exact-host Transformers Qwen3 stack is not installed")
    assert TRANSFORMERS_AVAILABLE


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="exact-host Transformers Qwen3 stack is not installed")
def test_native_qwen3_gate_zero_preserves_logits() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=41,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=3,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        tie_word_embeddings=True,
        _attn_implementation="eager",
    )
    torch.manual_seed(23)
    baseline = Qwen3ForCausalLM(config).eval()
    candidate = Qwen3ForCausalLM(config).eval()
    candidate.load_state_dict(baseline.state_dict())
    shape = ModelShape(41, 24, 2, 3, 3, 8, 1e-6, "qwen3")
    bank = FactorizedMemoryBank(shape, FMAConfig(rank=4, layers=(0,), gate=0.0))
    convert_qwen3_model(candidate, bank, layers=(0,))
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = baseline(input_ids=ids, use_cache=False).logits
        actual = candidate(input_ids=ids, use_cache=False).logits
    assert torch.allclose(expected, actual, atol=2e-5, rtol=2e-5)

    bank.set_all_gates(1.0)
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("native v_proj was called at g=1")
    handle = candidate.model.layers[0].self_attn.v_proj.register_forward_hook(lambda *args: fail_if_called())
    try:
        with torch.no_grad():
            hard = candidate(input_ids=ids, use_cache=False).logits
        assert torch.isfinite(hard).all()
    finally:
        handle.remove()
