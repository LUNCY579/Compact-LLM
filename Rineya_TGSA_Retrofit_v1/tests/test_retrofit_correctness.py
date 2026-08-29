"""Model-baking safety checks for the pretrained TGSA retrofit.

Every check the bake gate requires, plus a regression test for each defect found
during development. These are correctness tests, not benchmarks: a retrofit that
runs fast but silently breaks the KV cache, the GQA grouping, or the pretrained
function is worthless.

The central one is `test_conversion_is_bitwise_function_preserving`. If that
fails, the conversion is no longer exact and the conversion-damage report must be
regenerated before anything is baked.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from rineya_tgsa import (
    NativeReferenceAttention,
    RineyaTGSAAttention,
    RineyaTGSAConfig,
    apply_rotary,
    rotate_half,
)
from rineya_tgsa.convert import (
    convert_attention_module,
    convert_checkpoint,
    derive_layer_seed,
    tensor_sha256,
    verify_conversion_manifest,
)

HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 128, 8, 2, 16


def _config(**overrides) -> RineyaTGSAConfig:
    base = dict(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        trajectory_rank=8,
    )
    base.update(overrides)
    return RineyaTGSAConfig(**base)


def _paired(dtype=torch.float64, **overrides) -> tuple[RineyaTGSAAttention, NativeReferenceAttention]:
    """A retrofit and an independent native reference sharing pretrained weights."""

    torch.manual_seed(7)
    config = _config(**overrides)
    tgsa = RineyaTGSAAttention(config, layer_index=0).eval().to(dtype)
    native = NativeReferenceAttention(config).eval().to(dtype)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(tgsa, name).load_state_dict(getattr(native, name).state_dict())
    return tgsa, native


def _causal(length: int, dtype=torch.float64) -> torch.Tensor:
    return torch.full((length, length), float("-inf"), dtype=dtype).triu(1)[None, None]


# --------------------------------------------------------------------------
# The exactness guarantee
# --------------------------------------------------------------------------


def test_conversion_is_bitwise_function_preserving() -> None:
    """The whole retrofit thesis. Do not weaken this test."""

    tgsa, native = _paired()
    x = torch.randn(3, 12, HIDDEN, dtype=torch.float64)
    mask = _causal(12)
    with torch.no_grad():
        got, _, _, _ = tgsa(x, attention_mask=mask)
        want, _ = native(x, attention_mask=mask)
    assert torch.equal(got, want), f"max|diff| = {(got - want).abs().max():.3e}"


def test_exactness_holds_for_every_conditioning_target() -> None:
    for targets in (
        dict(condition_q=True, condition_k=False, condition_v=False),
        dict(condition_q=False, condition_k=True, condition_v=False),
        dict(condition_q=False, condition_k=False, condition_v=True),
        dict(condition_q=True, condition_k=True, condition_v=True),
    ):
        tgsa, native = _paired(**targets)
        x = torch.randn(2, 9, HIDDEN, dtype=torch.float64)
        mask = _causal(9)
        with torch.no_grad():
            got, _, _, _ = tgsa(x, attention_mask=mask)
            want, _ = native(x, attention_mask=mask)
        assert torch.equal(got, want), f"{targets} broke exactness"


def test_head_zero_conditioning_is_structurally_inert() -> None:
    """Head 0 has no previous carrier, so its velocity - and therefore its
    control signal - is identically zero. Its conditioning weights can never
    affect the output and can never receive gradient. This is inherited from the
    Phase-1 operator and is arguably correct (there is no trajectory before the
    first head), but it means only ``num_attention_heads - 1`` heads are
    trajectory-conditioned. It is asserted here so the property is explicit
    rather than an accident nobody notices.
    """

    tgsa, native = _paired()
    with torch.no_grad():
        nn.init.normal_(tgsa.conditioning["q"][0].weight, std=0.5)
    x = torch.randn(2, 9, HIDDEN, dtype=torch.float64)
    mask = _causal(9)
    with torch.no_grad():
        got, _, _, _ = tgsa(x, attention_mask=mask)
        want, _ = native(x, attention_mask=mask)
    assert torch.equal(got, want), "head 0 conditioning unexpectedly reached the output"
    assert tgsa.effective_conditioned_heads == HEADS - 1


def test_a_nonzero_conditioning_weight_does_change_the_output() -> None:
    """Guards against the exactness test passing for a trivial reason."""

    tgsa, native = _paired()
    with torch.no_grad():
        nn.init.normal_(tgsa.conditioning["q"][3].weight, std=0.5)
    x = torch.randn(2, 9, HIDDEN, dtype=torch.float64)
    mask = _causal(9)
    with torch.no_grad():
        got, _, _, _ = tgsa(x, attention_mask=mask)
        want, _ = native(x, attention_mask=mask)
    assert not torch.equal(got, want)
    assert not tgsa.conditioning_is_neutral()


# --------------------------------------------------------------------------
# RoPE
# --------------------------------------------------------------------------


def test_rope_is_a_rotation() -> None:
    """RoPE must preserve per-pair norms exactly."""

    torch.manual_seed(1)
    q = torch.randn(2, 4, 6, HEAD_DIM, dtype=torch.float64)
    k = torch.randn(2, 4, 6, HEAD_DIM, dtype=torch.float64)
    from rineya_tgsa.attention import RotaryEmbedding

    rotary = RotaryEmbedding(HEAD_DIM).double()
    pos = torch.arange(6)[None].expand(2, -1)
    cos, sin = rotary(pos, dtype=torch.float64)
    qr, kr = apply_rotary(q, k, cos, sin)
    half = HEAD_DIM // 2
    for original, rotated in ((q, qr), (k, kr)):
        before = original[..., :half] ** 2 + original[..., half:] ** 2
        after = rotated[..., :half] ** 2 + rotated[..., half:] ** 2
        torch.testing.assert_close(before, after, atol=1e-10, rtol=1e-10)


def test_rope_makes_attention_relative() -> None:
    """Shifting both query and key positions equally must not change logits."""

    from rineya_tgsa.attention import RotaryEmbedding

    torch.manual_seed(2)
    rotary = RotaryEmbedding(HEAD_DIM).double()
    q = torch.randn(1, 1, 4, HEAD_DIM, dtype=torch.float64)
    k = torch.randn(1, 1, 4, HEAD_DIM, dtype=torch.float64)

    def logits(offset: int) -> torch.Tensor:
        pos = (torch.arange(4) + offset)[None]
        cos, sin = rotary(pos, dtype=torch.float64)
        qr, kr = apply_rotary(q, k, cos, sin)
        return qr @ kr.transpose(-2, -1)

    torch.testing.assert_close(logits(0), logits(37), atol=1e-9, rtol=1e-9)


def test_rotate_half_is_involutive_up_to_sign() -> None:
    x = torch.randn(3, 8, dtype=torch.float64)
    torch.testing.assert_close(rotate_half(rotate_half(x)), -x, atol=1e-12, rtol=1e-12)


# --------------------------------------------------------------------------
# GQA
# --------------------------------------------------------------------------


def test_gqa_head_group_mapping_matches_repeat_interleave() -> None:
    """Query head j must read kv group j // group_size, exactly as native does."""

    config = _config()
    assert config.group_size == HEADS // KV_HEADS
    expected = torch.arange(HEADS) // config.group_size
    native_order = torch.arange(KV_HEADS).repeat_interleave(config.group_size)
    assert torch.equal(expected, native_order)


def test_kv_cache_holds_groups_not_query_heads() -> None:
    """GQA's memory advantage must survive the retrofit."""

    tgsa, _ = _paired()
    x = torch.randn(2, 5, HIDDEN, dtype=torch.float64)
    _, _, present, _ = tgsa(x, attention_mask=_causal(5), use_cache=True)
    assert present[0].shape == (2, KV_HEADS, 5, HEAD_DIM)
    assert present[1].shape == (2, KV_HEADS, 5, HEAD_DIM)


def test_single_kv_head_and_full_multihead_both_work() -> None:
    for kv in (1, HEADS):
        tgsa, native = _paired(num_key_value_heads=kv)
        x = torch.randn(2, 7, HIDDEN, dtype=torch.float64)
        mask = _causal(7)
        with torch.no_grad():
            got, _, _, _ = tgsa(x, attention_mask=mask)
            want, _ = native(x, attention_mask=mask)
        assert torch.equal(got, want), f"kv_heads={kv}"


# --------------------------------------------------------------------------
# Cache, prefill, decode
# --------------------------------------------------------------------------


def test_cached_and_uncached_generation_agree() -> None:
    """Incremental decoding must reproduce a full forward pass."""

    tgsa, _ = _paired()
    torch.manual_seed(3)
    x = torch.randn(2, 10, HIDDEN, dtype=torch.float64)

    with torch.no_grad():
        full, _, _, _ = tgsa(x, attention_mask=_causal(10))

    with torch.no_grad():
        prefill, _, cache, _ = tgsa(x[:, :6], attention_mask=_causal(6), use_cache=True)
        outputs = [prefill]
        for step in range(6, 10):
            token = x[:, step : step + 1]
            position = torch.full((2, 1), step, dtype=torch.long)
            out, _, cache, _ = tgsa(
                token, position_ids=position, past_key_value=cache, use_cache=True
            )
            outputs.append(out)
        incremental = torch.cat(outputs, dim=1)

    torch.testing.assert_close(full, incremental, atol=1e-10, rtol=1e-10)


def test_prefill_and_decode_agree_with_the_native_reference() -> None:
    tgsa, native = _paired()
    torch.manual_seed(4)
    x = torch.randn(1, 8, HIDDEN, dtype=torch.float64)
    with torch.no_grad():
        _, _, tgsa_cache, _ = tgsa(x[:, :5], attention_mask=_causal(5), use_cache=True)
        _, native_cache = native(x[:, :5], attention_mask=_causal(5), use_cache=True)
        assert torch.equal(tgsa_cache[0], native_cache[0])
        assert torch.equal(tgsa_cache[1], native_cache[1])
        position = torch.tensor([[5]])
        step_tgsa, _, _, _ = tgsa(
            x[:, 5:6], position_ids=position, past_key_value=tgsa_cache, use_cache=True
        )
        step_native, _ = native(
            x[:, 5:6], position_ids=position, past_key_value=native_cache, use_cache=True
        )
    assert torch.equal(step_tgsa, step_native)


def test_long_context_forward_runs_and_stays_finite() -> None:
    tgsa, _ = _paired()
    x = torch.randn(1, 512, HIDDEN, dtype=torch.float64)
    with torch.no_grad():
        out, _, _, _ = tgsa(x, attention_mask=_causal(512))
    assert out.shape == (1, 512, HIDDEN)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# Masking invariants
# --------------------------------------------------------------------------


def test_causal_prefix_invariance() -> None:
    """Changing future tokens must not change earlier outputs."""

    tgsa, _ = _paired()
    torch.manual_seed(5)
    x = torch.randn(2, 12, HIDDEN, dtype=torch.float64)
    changed = x.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:])
    mask = _causal(12)
    with torch.no_grad():
        a, _, _, _ = tgsa(x, attention_mask=mask)
        b, _, _, _ = tgsa(changed, attention_mask=mask)
    torch.testing.assert_close(a[:, :7], b[:, :7], atol=1e-11, rtol=1e-11)


def test_padding_invariance() -> None:
    """A masked-out padded suffix must not change unpadded outputs."""

    tgsa, _ = _paired()
    torch.manual_seed(6)
    valid = 8
    x = torch.randn(1, valid, HIDDEN, dtype=torch.float64)
    padded = torch.cat((x, torch.randn(1, 4, HIDDEN, dtype=torch.float64)), dim=1)

    mask = _causal(valid)
    padded_mask = _causal(valid + 4).clone()
    padded_mask[..., valid:] = float("-inf")  # padded keys are unattendable

    with torch.no_grad():
        a, _, _, _ = tgsa(x, attention_mask=mask)
        b, _, _, _ = tgsa(padded, attention_mask=padded_mask)
    torch.testing.assert_close(a, b[:, :valid], atol=1e-11, rtol=1e-11)


def test_batch_size_consistency() -> None:
    """Per-example outputs must not depend on batch composition."""

    tgsa, _ = _paired()
    torch.manual_seed(8)
    x = torch.randn(4, 9, HIDDEN, dtype=torch.float64)
    mask = _causal(9)
    with torch.no_grad():
        batched, _, _, _ = tgsa(x, attention_mask=mask)
        singles = torch.cat([tgsa(x[i : i + 1], attention_mask=mask)[0] for i in range(4)])
    torch.testing.assert_close(batched, singles, atol=1e-11, rtol=1e-11)


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------


def test_trajectory_gradients_are_connected_at_step_zero() -> None:
    """Exactness must not come at the price of a dead gradient.

    Zero-initializing both sides of an adapter would make the output exact AND
    the gradient zero, which looks correct but never learns.
    """

    torch.manual_seed(9)
    tgsa, _ = _paired(dtype=torch.float32)
    x = torch.randn(2, 8, HIDDEN, requires_grad=True)
    out, _, _, _ = tgsa(x, attention_mask=_causal(8, dtype=torch.float32))
    out.square().mean().backward()

    conditioning = tgsa.conditioning["q"][3].weight
    assert conditioning.grad is not None
    assert torch.count_nonzero(conditioning.grad) > 0, (
        "conditioning weight has no gradient: the trajectory path cannot learn"
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in tgsa.parameters()
    )


def test_carrier_is_non_degenerate_at_initialization() -> None:
    """The control signal must actually carry information at step 0."""

    tgsa, _ = _paired(dtype=torch.float32)
    x = torch.randn(2, 10, HIDDEN)
    with torch.no_grad():
        _, _, _, telemetry = tgsa(x, attention_mask=_causal(10, dtype=torch.float32), telemetry="summary")
    assert telemetry is not None
    assert telemetry.velocity_rms[1] > 1e-6, "carrier does not evolve across heads"
    assert telemetry.control_rms[1] > 1e-6, "control signal is degenerate"
    # Head 0 has no previous carrier, so its velocity is zero by construction.
    assert telemetry.velocity_rms[0] == pytest.approx(0.0, abs=1e-12)
    # Conditioning contributes nothing at step 0, which is the exactness property.
    assert max(telemetry.conditioning_rms) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------


def test_checkpoint_save_reload_identity(tmp_path: Path) -> None:
    tgsa, _ = _paired(dtype=torch.float32)
    torch.manual_seed(11)
    with torch.no_grad():
        for target in tgsa.conditioning:
            for layer in tgsa.conditioning[target]:
                nn.init.normal_(layer.weight, std=0.1)
    path = tmp_path / "retrofit.pt"
    torch.save(tgsa.state_dict(), path)

    restored = RineyaTGSAAttention(_config(), layer_index=0).eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    x = torch.randn(2, 7, HIDDEN)
    mask = _causal(7, dtype=torch.float32)
    with torch.no_grad():
        a, _, _, _ = tgsa(x, attention_mask=mask)
        b, _, _, _ = restored(x, attention_mask=mask)
    assert torch.equal(a, b)


def test_mixed_precision_stays_finite() -> None:
    tgsa, _ = _paired(dtype=torch.float32)
    x = torch.randn(2, 16, HIDDEN)
    mask = _causal(16, dtype=torch.float32)
    with torch.no_grad():
        for dtype in (torch.float32, torch.bfloat16):
            module = RineyaTGSAAttention(_config(), layer_index=0).eval().to(dtype)
            module.load_state_dict({k: v.to(dtype) for k, v in tgsa.state_dict().items()})
            out, _, _, _ = module(x.to(dtype), attention_mask=mask.to(dtype))
            assert torch.isfinite(out.float()).all(), f"{dtype} produced non-finite output"


# --------------------------------------------------------------------------
# Conversion and manifest
# --------------------------------------------------------------------------


def _source_checkpoint(layers: int = 4) -> tuple[dict[str, torch.Tensor], dict]:
    torch.manual_seed(13)
    config = {
        "hidden_size": HIDDEN,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_hidden_layers": layers,
        "rope_theta": 1000000.0,
    }
    state: dict[str, torch.Tensor] = {}
    for index in range(layers):
        prefix = f"model.layers.{index}.self_attn"
        state[f"{prefix}.q_proj.weight"] = torch.randn(HEADS * HEAD_DIM, HIDDEN)
        state[f"{prefix}.k_proj.weight"] = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN)
        state[f"{prefix}.v_proj.weight"] = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN)
        state[f"{prefix}.o_proj.weight"] = torch.randn(HIDDEN, HEADS * HEAD_DIM)
    return state, config


def test_conversion_reuses_native_tensors_verbatim() -> None:
    state, config = _source_checkpoint()
    modules, manifest = convert_checkpoint(
        source_state=state,
        source_config=config,
        convert_layers=[1, 3],
        trajectory_overrides={"trajectory_rank": 8},
    )
    assert sorted(modules) == [1, 3]
    for index, module in modules.items():
        prefix = f"model.layers.{index}.self_attn"
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert torch.equal(getattr(module, name).weight, state[f"{prefix}.{name}.weight"])
    assert manifest["converted_layers"] == [1, 3]
    assert manifest["retrofit_config"]["rope_theta"] == 1000000.0


def test_conversion_is_reproducible_byte_for_byte() -> None:
    state, config = _source_checkpoint()
    first, manifest_a = convert_checkpoint(
        source_state=state, source_config=config, convert_layers=[0, 2], master_seed=999
    )
    second, manifest_b = convert_checkpoint(
        source_state=state, source_config=config, convert_layers=[0, 2], master_seed=999
    )
    assert manifest_a["converted_checkpoint_sha256"] == manifest_b["converted_checkpoint_sha256"]
    for index in first:
        for key, tensor in first[index].state_dict().items():
            assert torch.equal(tensor, second[index].state_dict()[key]), key


def test_different_master_seed_changes_only_new_tensors() -> None:
    state, config = _source_checkpoint()
    a, _ = convert_checkpoint(source_state=state, source_config=config, convert_layers=[0], master_seed=1)
    b, _ = convert_checkpoint(source_state=state, source_config=config, convert_layers=[0], master_seed=2)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert torch.equal(getattr(a[0], name).weight, getattr(b[0], name).weight)
    assert not torch.equal(a[0].carrier_init.weight, b[0].carrier_init.weight)


def test_manifest_verification_detects_a_mutated_source_tensor() -> None:
    state, config = _source_checkpoint()
    modules, manifest = convert_checkpoint(
        source_state=state, source_config=config, convert_layers=[0, 1]
    )
    assert verify_conversion_manifest(manifest, source_state=state, modules=modules)["passed"]

    tampered = dict(state)
    tampered["model.layers.0.self_attn.q_proj.weight"] = (
        tampered["model.layers.0.self_attn.q_proj.weight"] + 1e-3
    )
    report = verify_conversion_manifest(manifest, source_state=tampered, modules=modules)
    assert report["passed"] is False
    assert any("q_proj" in error for error in report["errors"])


def test_conversion_refuses_a_shape_mismatch() -> None:
    state, config = _source_checkpoint()
    broken = dict(state)
    broken["model.layers.0.self_attn.q_proj.weight"] = torch.randn(HEADS * HEAD_DIM, HIDDEN + 8)
    with pytest.raises(ValueError, match="does not match"):
        convert_checkpoint(source_state=broken, source_config=config, convert_layers=[0])


def test_conversion_refuses_to_silently_drop_a_pretrained_bias() -> None:
    state, config = _source_checkpoint()
    state["model.layers.0.self_attn.q_proj.bias"] = torch.randn(HEADS * HEAD_DIM)
    with pytest.raises(ValueError, match="silently drop"):
        convert_checkpoint(source_state=state, source_config=config, convert_layers=[0])


def test_conversion_refuses_an_out_of_range_layer() -> None:
    state, config = _source_checkpoint(layers=4)
    with pytest.raises(ValueError, match="outside the source model"):
        convert_checkpoint(source_state=state, source_config=config, convert_layers=[9])


def test_layer_seeds_are_distinct_and_deterministic() -> None:
    seeds = [derive_layer_seed(5, index) for index in range(8)]
    assert len(set(seeds)) == 8
    assert seeds == [derive_layer_seed(5, index) for index in range(8)]


def test_config_rejects_incompatible_gqa() -> None:
    with pytest.raises(ValueError, match="divisible"):
        RineyaTGSAConfig(hidden_size=128, num_attention_heads=8, num_key_value_heads=3, head_dim=16)


def test_config_rejects_a_disconnected_trajectory_pathway() -> None:
    with pytest.raises(ValueError, match="at least one conditioning target"):
        _config(condition_q=False, condition_k=False, condition_v=False)


# --------------------------------------------------------------------------
# The real Rineya target architecture
# --------------------------------------------------------------------------

REAL_CONFIG = Path(__file__).resolve().parent / "fixtures" / "rineya_supa_a1_config.json"


def _real_source() -> dict:
    return json.loads(REAL_CONFIG.read_text(encoding="utf-8"))


def test_real_rineya_config_maps_to_the_right_shapes() -> None:
    """Pin the mapping against the actual Rineya-Supa-A1 config.

    Qwen3 declares `head_dim` independently of `hidden_size`: this model has
    hidden 1024 with 16 heads of 128, so the common
    `hidden_size // num_attention_heads` shortcut would give 64 and every
    projection shape would silently be wrong.
    """

    config = RineyaTGSAConfig.from_source_config(_real_source(), trajectory_rank=32)
    assert (config.hidden_size, config.num_attention_heads) == (1024, 16)
    assert (config.num_key_value_heads, config.head_dim) == (8, 128)
    assert config.group_size == 2
    assert config.num_attention_heads * config.head_dim == 2048 != config.hidden_size


def test_real_rineya_rope_theta_is_read_from_the_nested_field() -> None:
    """Regression: transformers 5.x nests rope_theta under `rope_parameters`.

    Defaulting to 10000 when the model uses 1000000 would corrupt every
    positional frequency, and an exactness test would stay green because the
    reference would be built from the same wrong value.
    """

    config = RineyaTGSAConfig.from_source_config(_real_source())
    assert config.rope_theta == 1_000_000.0


def test_config_refuses_to_guess_a_missing_rope_convention() -> None:
    source = _real_source()
    source.pop("rope_parameters", None)
    source.pop("rope_theta", None)
    with pytest.raises(KeyError, match="rope_theta"):
        RineyaTGSAConfig.from_source_config(source)


def test_config_refuses_an_unsupported_rope_type() -> None:
    source = _real_source()
    source["rope_parameters"] = {"rope_theta": 1e6, "rope_type": "yarn"}
    with pytest.raises(ValueError, match="unsupported rope_type"):
        RineyaTGSAConfig.from_source_config(source)


def test_exactness_at_the_real_rineya_architecture() -> None:
    """The property that matters, at the dimensions of the actual target."""

    torch.manual_seed(21)
    config = RineyaTGSAConfig.from_source_config(_real_source(), trajectory_rank=32)
    tgsa = RineyaTGSAAttention(config, layer_index=0).eval().double()
    native = NativeReferenceAttention(config).eval().double()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(tgsa, name).load_state_dict(getattr(native, name).state_dict())

    x = torch.randn(1, 6, config.hidden_size, dtype=torch.float64)
    mask = torch.full((6, 6), float("-inf"), dtype=torch.float64).triu(1)[None, None]
    with torch.no_grad():
        got, _, _, _ = tgsa(x, attention_mask=mask)
        want, _ = native(x, attention_mask=mask)
    assert torch.equal(got, want)


def test_conversion_at_the_real_rineya_layer_count() -> None:
    """Shape and mapping logic against the real 28-layer, GQA-8 target."""

    source = _real_source()
    config = RineyaTGSAConfig.from_source_config(source, trajectory_rank=32)
    torch.manual_seed(22)
    state: dict[str, torch.Tensor] = {}
    for index in range(int(source["num_hidden_layers"])):
        prefix = f"model.layers.{index}.self_attn"
        q = config.num_attention_heads * config.head_dim
        kv = config.num_key_value_heads * config.head_dim
        # bfloat16 mirrors the published dtype.
        state[f"{prefix}.q_proj.weight"] = torch.randn(q, config.hidden_size).bfloat16()
        state[f"{prefix}.k_proj.weight"] = torch.randn(kv, config.hidden_size).bfloat16()
        state[f"{prefix}.v_proj.weight"] = torch.randn(kv, config.hidden_size).bfloat16()
        state[f"{prefix}.o_proj.weight"] = torch.randn(config.hidden_size, q).bfloat16()

    modules, manifest = convert_checkpoint(
        source_state=state,
        source_config=source,
        convert_layers=list(range(0, 28, 7)),
        trajectory_overrides={"trajectory_rank": 32},
    )
    assert manifest["converted_layers"] == [0, 7, 14, 21]
    assert manifest["total_source_layers"] == 28
    assert manifest["retrofit_config"]["rope_theta"] == 1_000_000.0
    for index, module in modules.items():
        prefix = f"model.layers.{index}.self_attn"
        assert torch.equal(
            module.q_proj.weight.bfloat16(), state[f"{prefix}.q_proj.weight"]
        )
        assert module.conditioning_is_neutral()
    assert verify_conversion_manifest(manifest, source_state=state, modules=modules)["passed"]
