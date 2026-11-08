"""Native random-Qwen checks, separate from source-backed official P1.

An absent Transformers runtime is an explicit skip, never a passing native gate.
"""
from pathlib import Path
import json
import torch
import pytest
from experiments.architecture_benchmark_001.model import DenseReference
from experiments.cross_scale_ced_v1.config import tiny_config
from experiments.cross_scale_ced_v1.source import native_from_reference
from experiments.cross_scale_ced_v1.checks import native_parity


@pytest.mark.parametrize("role", ["encoder", "decoder"])
def test_native_random_qwen(role, tmp_path):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.1":
        pytest.fail("native checks require Transformers 4.57.1, not a silent substitute")
    torch.set_num_threads(1); torch.manual_seed(17)
    c = getattr(tiny_config(), role)
    config = dict(model_type="qwen3", architectures=["Qwen3ForCausalLM"],
                  vocab_size=c.vocab_size, hidden_size=c.hidden_size, intermediate_size=c.intermediate_size,
                  num_hidden_layers=c.encoder_layers, num_attention_heads=c.num_attention_heads,
                  num_key_value_heads=c.num_key_value_heads, head_dim=c.head_dim,
                  rms_norm_eps=c.rms_norm_eps, rope_theta=c.rope_theta,
                  max_position_embeddings=c.max_position_embeddings, tie_word_embeddings=True,
                  attention_dropout=0.0, attention_bias=False, hidden_act="silu", use_sliding_window=False)
    (tmp_path/"config.json").write_text(json.dumps(config))
    reference = DenseReference(c).eval()
    native = native_from_reference(reference, tmp_path)
    assert native_parity(reference, native)["passed"]
