"""Tests for the real-transformers Qwen3 TGSA adapter.

The defect these lock down: real Qwen3 applies per-head RMSNorm to Q and K
(``q_norm``, ``k_norm``) before RoPE. A hand-written retrofit that omits them
silently drops two pretrained tensors per layer, and an exactness test against
that same hand-written reference stays green.

Subclassing ``Qwen3Attention`` removes the whole class of error: every native
tensor is created and shaped by transformers itself. ``load_native_`` then
refuses any native tensor the retrofit does not have, so a future transformers
version that adds a pretrained sublayer fails loudly instead of silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from transformers import AutoConfig  # noqa: E402
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import (  # noqa: E402
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3RotaryEmbedding,
)

from rineya_tgsa import RineyaTGSAConfig  # noqa: E402
from rineya_tgsa.qwen3_adapter import Qwen3TGSAAttention, convert_qwen3_model  # noqa: E402

REAL_CONFIG = Path(__file__).resolve().parent / "fixtures" / "rineya_supa_a1_config.json"


def _configs(hidden=256, heads=8, kv=4, head_dim=32, layers=4):
    hf = Qwen3Config(
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=head_dim,
        num_hidden_layers=layers,
        intermediate_size=hidden * 2,
        vocab_size=256,
        rope_theta=1_000_000.0,
    )
    hf._attn_implementation = "eager"
    tgsa = RineyaTGSAConfig(
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=head_dim,
        rope_theta=1_000_000.0,
        trajectory_rank=8,
    )
    return hf, tgsa


def _inputs(hf, dtype, batch=2, seq=12):
    x = torch.randn(batch, seq, hf.hidden_size, dtype=dtype)
    rotary = Qwen3RotaryEmbedding(hf).to(dtype)
    positions = torch.arange(seq)[None].expand(batch, -1)
    cos, sin = rotary(x, positions)
    mask = torch.full((seq, seq), torch.finfo(dtype).min, dtype=dtype).triu(1)
    mask = mask[None, None].expand(batch, 1, seq, seq).contiguous()
    return x, (cos, sin), mask


def test_adapter_inherits_q_norm_and_k_norm() -> None:
    """The tensors a hand-rolled reimplementation would have dropped."""

    hf, tgsa_config = _configs()
    module = Qwen3TGSAAttention(hf, 0, tgsa_config)
    names = dict(module.named_parameters())
    assert "q_norm.weight" in names, "q_norm missing: a pretrained tensor would be dropped"
    assert "k_norm.weight" in names, "k_norm missing: a pretrained tensor would be dropped"


def test_load_native_copies_every_pretrained_tensor() -> None:
    hf, tgsa_config = _configs()
    torch.manual_seed(3)
    native = Qwen3Attention(hf, 0)
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.normal_(0.0, 0.05)
    module = Qwen3TGSAAttention(hf, 0, tgsa_config)
    reused = module.load_native_(native)
    assert set(reused) == set(native.state_dict())
    assert "q_norm.weight" in reused and "k_norm.weight" in reused
    for key, tensor in native.state_dict().items():
        assert torch.equal(module.state_dict()[key], tensor), key


def test_load_native_refuses_an_unknown_pretrained_tensor() -> None:
    """A future transformers version adding a sublayer must fail loudly."""

    hf, tgsa_config = _configs()
    native = Qwen3Attention(hf, 0)
    native.register_parameter("some_new_pretrained_thing", nn.Parameter(torch.zeros(4)))
    module = Qwen3TGSAAttention(hf, 0, tgsa_config)
    with pytest.raises(RuntimeError, match="silently drop pretrained weights"):
        module.load_native_(native)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_adapter_is_bitwise_identical_to_native_eager(dtype) -> None:
    """The exactness guarantee, against real transformers' own eager path."""

    hf, tgsa_config = _configs()
    torch.manual_seed(5)
    native = Qwen3Attention(hf, 0).eval().to(dtype)
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.normal_(0.0, 0.05)
    module = Qwen3TGSAAttention(hf, 0, tgsa_config).eval().to(dtype)
    module.load_native_(native)

    hidden, position_embeddings, mask = _inputs(hf, dtype)
    with torch.no_grad():
        want, _ = native(hidden, position_embeddings, mask)
        got, _ = module(hidden, position_embeddings, mask)
    assert torch.equal(got, want), f"{dtype}: max|diff| = {(got - want).abs().max():.3e}"


def test_adapter_conditioning_is_neutral_and_head_zero_is_inert() -> None:
    hf, tgsa_config = _configs()
    module = Qwen3TGSAAttention(hf, 0, tgsa_config)
    assert module.conditioning_is_neutral()
    assert module.effective_conditioned_heads == hf.num_attention_heads - 1


def test_adapter_nonzero_conditioning_changes_the_output() -> None:
    """Guards against the exactness test passing for a trivial reason."""

    hf, tgsa_config = _configs()
    torch.manual_seed(6)
    native = Qwen3Attention(hf, 0).eval().double()
    module = Qwen3TGSAAttention(hf, 0, tgsa_config).eval().double()
    module.load_native_(native)
    hidden, position_embeddings, mask = _inputs(hf, torch.float64)
    with torch.no_grad():
        base, _ = module(hidden, position_embeddings, mask)
        nn.init.normal_(module.conditioning["q"][3].weight, std=0.5)
        changed, _ = module(hidden, position_embeddings, mask)
    assert not torch.equal(base, changed)
    assert not module.conditioning_is_neutral()


def test_adapter_trajectory_gradients_are_live_except_head_zero() -> None:
    hf, tgsa_config = _configs()
    torch.manual_seed(7)
    native = Qwen3Attention(hf, 0)
    module = Qwen3TGSAAttention(hf, 0, tgsa_config)
    module.load_native_(native)
    hidden, position_embeddings, mask = _inputs(hf, torch.float32)
    output, _ = module(hidden, position_embeddings, mask)
    output.square().mean().backward()

    live = module.conditioning["q"][3].weight
    assert live.grad is not None and torch.count_nonzero(live.grad) > 0

    bootstrap = module.conditioning["q"][0].weight
    assert bootstrap.grad is None or torch.count_nonzero(bootstrap.grad) == 0, (
        "head 0 is the bootstrap head and must have no trajectory gradient"
    )


def test_convert_qwen3_model_replaces_only_requested_layers() -> None:
    hf, tgsa_config = _configs(layers=4)
    torch.manual_seed(8)
    model = Qwen3ForCausalLM(hf).eval()
    record = convert_qwen3_model(model, tgsa_config, layers=[1, 3])
    assert record["converted_layers"] == [1, 3]
    assert record["total_layers"] == 4
    for index in range(4):
        module = model.model.layers[index].self_attn
        expected = Qwen3TGSAAttention if index in (1, 3) else Qwen3Attention
        assert isinstance(module, expected), index
    for entry in record["layers"]:
        assert entry["conditioning_is_neutral"] is True
        assert "q_norm.weight" in entry["reused_tensors"]


def test_convert_qwen3_model_preserves_model_output_exactly() -> None:
    """End-to-end: a converted model must produce the native logits."""

    hf, tgsa_config = _configs(layers=3)
    torch.manual_seed(9)
    model = Qwen3ForCausalLM(hf).eval().double()
    ids = torch.randint(0, hf.vocab_size, (2, 10))
    with torch.no_grad():
        before = model(input_ids=ids).logits
    convert_qwen3_model(model, tgsa_config, layers=[0, 2])
    with torch.no_grad():
        after = model(input_ids=ids).logits
    assert torch.equal(before, after), f"max|diff| = {(before - after).abs().max():.3e}"


def test_convert_refuses_an_out_of_range_layer() -> None:
    hf, tgsa_config = _configs(layers=3)
    model = Qwen3ForCausalLM(hf).eval()
    with pytest.raises(ValueError, match="outside the model"):
        convert_qwen3_model(model, tgsa_config, layers=[7])


def test_real_rineya_config_builds_an_adapter() -> None:
    """Against the actual published Rineya-Supa-A1 architecture."""

    source = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    hf = AutoConfig.for_model(
        "qwen3",
        **{
            key: source[key]
            for key in (
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "num_hidden_layers",
                "intermediate_size",
                "vocab_size",
                "max_position_embeddings",
            )
            if key in source
        },
    )
    hf._attn_implementation = "eager"
    tgsa_config = RineyaTGSAConfig.from_source_config(source, trajectory_rank=32)
    module = Qwen3TGSAAttention(hf, 0, tgsa_config)
    assert module.head_dim == 128
    assert module.num_heads == 16 and module.num_kv_heads == 8
    assert module.effective_conditioned_heads == 15
