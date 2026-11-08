"""Tiny actual HF decoders only; no checkpoints, efficacy or GPU execution."""
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from cross_scale.generation_stage.v1.src.operators import (
    GATE_GRID, buffer_hashes, make_graft_wrapper, make_intact_wrapper,
)
from shared.integrity import state_hash
from shared.legacy import legacy_graft


def objects(gate=.1, arm="XG", enabled=True):
    torch.manual_seed(37)
    small = Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=20,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=43,
        max_position_embeddings=64)
    small._attn_implementation = "eager"
    model = Qwen3ForCausalLM(small).eval().requires_grad_(False)
    big = Qwen3Config(hidden_size=24, intermediate_size=48, num_hidden_layers=36,
        num_attention_heads=2, num_key_value_heads=1, head_dim=12, vocab_size=43,
        max_position_embeddings=64)
    big._attn_implementation = "eager"
    donor = Qwen3DecoderLayer(big, 12).eval().requires_grad_(False)
    state = {"up": torch.randn(16, 24) * .05, "up_bias": torch.randn(24) * .05,
             "down": torch.randn(24, 16) * .05, "down_bias": torch.ones(16) * 100}
    wrapper = make_graft_wrapper(model, donor, big, state, arm=arm,
        insertion_layer=8, donor_index=12, gate=gate, emergency_bounds=(.05, 10.), enabled=enabled)
    return model, wrapper, state


@pytest.mark.parametrize("gate", GATE_GRID)
def test_each_dose_preserves_actual_mask_rope_and_canonical_state(gate):
    model, wrapper, _ = objects(gate)
    model_before, donor_before = state_hash(model), state_hash(wrapper.expert)
    layer_forward = wrapper.expert.layer.forward
    seen = []
    def capture(hidden, **kwargs):
        seen.append(kwargs)
        return layer_forward(hidden, **kwargs)
    wrapper.expert.layer.forward = capture
    with torch.inference_mode(), wrapper:
        a = model(torch.tensor([[2, 4, 6, 8]]), use_cache=False).logits
        b = model(torch.tensor([[2, 4, 9, 11]]), use_cache=False).logits
    assert torch.equal(a[:, :2], b[:, :2])
    assert len(seen) == 2
    assert all(torch.equal(v["position_ids"], torch.arange(4).unsqueeze(0)) for v in seen)
    assert all(v["attention_mask"].shape == (1, 1, 4, 4) for v in seen)
    assert all(v["position_embeddings"][0].shape[-1] == 12 for v in seen)
    assert all(v["use_cache"] is False and v["past_key_values"] is None for v in seen)
    assert state_hash(model) == model_before
    assert state_hash(wrapper.expert) == donor_before
    assert wrapper.integrity_receipt["buffers_before"] == wrapper.integrity_receipt["buffers_after"]
    assert not model._forward_pre_hooks and not wrapper.active


@pytest.mark.parametrize("disabled", [False, True])
def test_zero_or_disabled_is_exact_and_reports_zero_active_work(disabled):
    model, wrapper, _ = objects(.1 if disabled else 0., enabled=not disabled)
    wrapper.expert.layer.forward = lambda *a, **k: pytest.fail("bypassed donor was invoked")
    ids = torch.tensor([[2, 3, 4]])
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits
        with wrapper:
            actual = model(ids, use_cache=False).logits
        restored = model(ids, use_cache=False).logits
    assert torch.equal(expected, actual) and torch.equal(expected, restored)
    assert wrapper.added_parameter_count > 0
    assert wrapper.added_active_parameters == wrapper.extra_matmul_flops_per_token == 0


@pytest.mark.parametrize("arm", ["XG", "XS", "XR"])
def test_exact_input_bridge_is_cloned_and_down_bias_cancels(arm):
    _, wrapper, state = objects(arm=arm)
    expert = wrapper.expert
    assert all(torch.equal(getattr(expert.bridges, key), value) for key, value in state.items())
    assert all(getattr(expert.bridges, key).data_ptr() != value.data_ptr() for key, value in state.items())
    expert.layer.forward = lambda z, **kwargs: z + 2
    h = torch.randn(1, 4, 16)
    positions, mask = legacy_graft().positions_mask(h)
    with torch.inference_mode():
        output = expert(h, attention_mask=mask, position_ids=positions)
        expected = h + .1 * (torch.full((1, 4, 24), 2.) @ state["down"])
    assert torch.allclose(output, expected, atol=1e-6)
    assert wrapper.extra_matmul_flops_per_token == legacy_graft().extra_flops(expert)
    assert wrapper.added_active_parameters == wrapper.added_parameter_count - 16


def test_exception_restores_hooks_and_retains_expert_integrity():
    model, wrapper, _ = objects()
    original = model.model.layers[8].forward
    def fail(*a, **k):
        raise RuntimeError("injected donor failure")
    wrapper.expert.layer.forward = fail
    with pytest.raises(RuntimeError, match="injected"), torch.inference_mode(), wrapper:
        model(torch.tensor([[2, 3]]), use_cache=False)
    assert model.model.layers[8].forward == original
    assert not model._forward_pre_hooks and not wrapper.active
    assert wrapper.integrity_receipt["state_sha256_before"] == wrapper.integrity_receipt["state_sha256_after"]


@pytest.mark.parametrize("mutation", ["weights", "rotary", "gate", "gradient", "identity", "enabled", "bounds", "donor_layer"])
def test_mutations_fail_after_restoring_canonical_hook(mutation):
    model, wrapper, _ = objects()
    original = model.model.layers[8].forward
    with pytest.raises(RuntimeError), torch.inference_mode(), wrapper:
        if mutation == "weights":
            next(wrapper.expert.layer.parameters()).add_(1)
        elif mutation == "rotary":
            wrapper.expert.rope.inv_freq.add_(1)
        elif mutation == "gate":
            wrapper.expert.gate = .2
        elif mutation == "gradient":
            next(wrapper.expert.layer.parameters()).grad = torch.zeros_like(next(wrapper.expert.layer.parameters()))
        elif mutation == "identity":
            wrapper.expert.identity = True
        elif mutation == "enabled":
            wrapper.enabled = False
        elif mutation == "bounds":
            wrapper.bounds = (.001, 1000.)
        else:
            wrapper.expert.layer.self_attn.layer_idx = 16
    assert model.model.layers[8].forward == original
    assert not model._forward_pre_hooks and not wrapper.active


def test_enter_rejects_identity_mode_and_reentrant_integrity_baseline_reset():
    model, wrapper, _ = objects()
    wrapper.expert.identity = True
    with pytest.raises(RuntimeError, match="actual donor"):
        with wrapper:
            pass
    assert not model._forward_pre_hooks
    wrapper.expert.identity = False
    with pytest.raises(RuntimeError, match="frozen donor"), torch.inference_mode(), wrapper:
        next(wrapper.expert.layer.parameters()).add_(1)
        with pytest.raises(RuntimeError, match="already active"):
            wrapper.__enter__()
    assert not model._forward_pre_hooks and not wrapper.active


@pytest.mark.parametrize("gate", [.11, .2])
def test_configured_gate_cannot_change_before_entry_or_between_contexts(gate):
    model, wrapper, _ = objects()
    with torch.inference_mode(), wrapper:
        model(torch.tensor([[2, 3]]), use_cache=False)
    wrapper.expert.gate = gate
    with pytest.raises(RuntimeError, match="configured graft treatment"):
        with wrapper:
            pass
    assert not model._forward_pre_hooks and not wrapper.active


@pytest.mark.parametrize("kwargs", [{"gate": .11}, {"gate": True}, {"gate": float("nan")},
    {"insertion_layer": 3}, {"donor_index": 16}, {"arm": "XM"}, {"emergency_bounds": (1., .5)}])
def test_unregistered_or_mismatched_inputs_rejected(kwargs):
    model, wrapper, state = objects()
    config = dict(arm="XG", insertion_layer=8, donor_index=12, gate=.1, emergency_bounds=(.05, 10.))
    config.update(kwargs)
    with pytest.raises(ValueError):
        make_graft_wrapper(model, wrapper.expert.layer, wrapper.expert.layer.self_attn.config,
                           state, **config)


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "requires_grad", "extra_key"])
def test_malformed_bridge_rejected(kind):
    model, wrapper, state = objects()
    if kind == "shape": state["up"] = state["up"][:1]
    elif kind == "dtype": state["up"] = state["up"].double()
    elif kind == "nan": state["up"][0, 0] = float("nan")
    elif kind == "requires_grad": state["up"].requires_grad_(True)
    else: state["unexpected"] = torch.ones(1)
    with pytest.raises(ValueError):
        make_graft_wrapper(model, wrapper.expert.layer, wrapper.expert.layer.self_attn.config,
            state, arm="XG", insertion_layer=8, donor_index=12, gate=.1, emergency_bounds=(.05, 10.))


def test_intact_baseline_has_no_added_work():
    model, _, _ = objects()
    wrapper = make_intact_wrapper(model)
    with torch.inference_mode(), wrapper:
        assert model(torch.tensor([[2, 3]]), use_cache=False).logits.isfinite().all()
    assert wrapper.arm == "X0"
    assert wrapper.extra_matmul_flops_per_token == wrapper.added_parameter_count == 0
