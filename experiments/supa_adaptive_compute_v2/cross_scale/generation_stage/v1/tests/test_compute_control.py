"""Small actual HF decoder fixtures; no checkpoint, efficacy or GPU execution."""
from copy import deepcopy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from cross_scale.generation_stage.v1.src.compute_control import (
    CONTROL, PARAMETERS, MATMUL_FLOPS_PER_TOKEN, GATE_GRID,
    CrossScaleComputeControl, FrozenSwiGLU, _FrozenComputeWrapper,
)
from shared.integrity import state_hash


def objects(gate=.1, alpha=.7, enabled=True):
    torch.manual_seed(33)
    config = Qwen3Config(hidden_size=8, intermediate_size=16, num_hidden_layers=20,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=31,
        rms_norm_eps=1e-6, max_position_embeddings=64)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    transform = FrozenSwiGLU(8, 18, seed=2702, std=.02, rms_epsilon=1e-6)
    wrapper = _FrozenComputeWrapper(model, transform, alpha=alpha, gate=gate,
        expected_state_sha256=state_hash(transform), layer=19, enabled=enabled,
        emergency_bounds=(.25, 4.))
    return model, wrapper


def test_production_geometry_and_exact_donor_bridge_matrix_work():
    assert CONTROL == {"kind": "independent_frozen_swiglu", "hidden_size": 1024,
        "intermediate_size": 34560, "seed": 2702, "weight_std": .02,
        "unit_rms_epsilon": 1e-6, "insertion_layer": 19}
    assert 3 * CONTROL["hidden_size"] * CONTROL["intermediate_size"] == PARAMETERS == 106168320
    assert 2 * PARAMETERS == MATMUL_FLOPS_PER_TOKEN == 212336640


def test_independent_cpu_generator_is_reproducible_and_does_not_consume_global_rng():
    torch.manual_seed(22)
    rng_before = torch.random.get_rng_state().clone()
    a = FrozenSwiGLU(8, 18, seed=2702, std=.02, rms_epsilon=1e-6)
    b = FrozenSwiGLU(8, 18, seed=2702, std=.02, rms_epsilon=1e-6)
    c = FrozenSwiGLU(8, 18, seed=2703, std=.02, rms_epsilon=1e-6)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert state_hash(a) == state_hash(b) != state_hash(c)


@pytest.mark.parametrize("gate", GATE_GRID)
def test_post_layer_residual_uses_fitted_alpha_and_gate_once(gate):
    model, wrapper = objects(gate=gate)
    ids = torch.tensor([[2, 3, 4]])
    normal_states, actual_states, transformer_inputs = [], [], []
    original_forward = wrapper.transform.forward
    def control(hidden):
        transformer_inputs.append(hidden.clone())
        return original_forward(hidden)
    wrapper.transform.forward = control
    handle = model.model.layers[19].register_forward_hook(lambda m, a, out: normal_states.append(out.clone()))
    with torch.inference_mode():
        model(ids, use_cache=False)
    handle.remove()
    before = state_hash(model)
    handle = model.model.layers[19].register_forward_hook(lambda m, a, out: actual_states.append(out.clone()))
    with torch.inference_mode(), wrapper:
        model(ids, use_cache=False)
    handle.remove()
    with torch.inference_mode():
        expected = normal_states[0] + gate * .7 * original_forward(normal_states[0])
    assert torch.equal(transformer_inputs[0], normal_states[0])
    assert torch.equal(actual_states[0], expected)
    assert wrapper.calls == wrapper.control_calls == 1
    assert wrapper.extra_layer_applications == 0
    assert wrapper.added_parameter_count == wrapper.added_active_parameters == 3 * 8 * 18
    assert wrapper.extra_matmul_flops_per_token == 6 * 8 * 18
    assert state_hash(model) == before
    assert not any("transform" in name for name, _ in model.named_modules())
    assert wrapper.integrity_receipt["state_sha256_before"] == wrapper.integrity_receipt["state_sha256_after"]


@pytest.mark.parametrize("disabled", [False, True])
def test_zero_disabled_and_restore_are_bit_identical_with_no_added_work(disabled):
    model, wrapper = objects(gate=.1 if disabled else 0., enabled=not disabled)
    wrapper.transform.forward = lambda hidden: pytest.fail("bypassed XM was invoked")
    ids = torch.tensor([[2, 3, 4]])
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
        with wrapper:
            treatment = model(ids, use_cache=False).logits
        restored = model(ids, use_cache=False).logits
    assert torch.equal(baseline, treatment) and torch.equal(baseline, restored)
    assert wrapper.control_calls == 0
    assert wrapper.added_active_parameters == wrapper.extra_matmul_flops_per_token == 0
    assert wrapper.added_parameter_count > 0 and not wrapper.active
    assert not model._forward_pre_hooks and "_armr_owner" not in model.model.layers[19].__dict__


@pytest.mark.parametrize("field,value", [("alpha", .8), ("gate", .2), ("enabled", False),
    ("layer", 18), ("bounds", (.1, 5.)), ("applications", 2), ("skip", True)])
def test_treatment_mutation_rejected_before_forward_and_restoration(field, value):
    model, wrapper = objects()
    original = model.model.layers[19].forward
    with pytest.raises(BaseExceptionGroup), torch.inference_mode(), wrapper:
        setattr(wrapper, field, value)
        model(torch.tensor([[2, 3]]), use_cache=False)
    assert model.model.layers[19].forward == original
    assert not wrapper.active and not model._forward_pre_hooks


@pytest.mark.parametrize("kind", ["weights", "epsilon", "gradient", "training"])
def test_control_mutation_fails_after_restoration(kind):
    model, wrapper = objects()
    original = model.model.layers[19].forward
    with pytest.raises(RuntimeError), torch.inference_mode(), wrapper:
        if kind == "weights":
            next(wrapper.transform.parameters()).add_(1)
        elif kind == "epsilon":
            wrapper.transform.rms_epsilon = 1e-5
        elif kind == "gradient":
            p = next(wrapper.transform.parameters())
            p.grad = torch.zeros_like(p)
        else:
            wrapper.transform.train()
    assert model.model.layers[19].forward == original
    assert not wrapper.active and not model._forward_pre_hooks


def test_exception_restores_hooks_and_keeps_integrity_receipt():
    model, wrapper = objects()
    def fail(hidden):
        raise RuntimeError("injected XM failure")
    wrapper.transform.forward = fail
    with pytest.raises(RuntimeError, match="injected"), torch.inference_mode(), wrapper:
        model(torch.tensor([[2, 3]]), use_cache=False)
    assert not model._forward_pre_hooks and not wrapper.active
    assert wrapper.integrity_receipt["state_sha256_before"] == wrapper.integrity_receipt["state_sha256_after"]


@pytest.mark.parametrize("alpha", [True, 0., -1., float("inf"), float("nan")])
def test_invalid_alpha_fails_before_construction(alpha):
    model, wrapper = objects()
    with pytest.raises(ValueError):
        _FrozenComputeWrapper(model, wrapper.transform, alpha=alpha, gate=.1,
            expected_state_sha256=state_hash(wrapper.transform), layer=19, enabled=True,
            emergency_bounds=(.25, 4.))


@pytest.mark.parametrize("gate", [True, -.1, .11, 1., float("nan")])
def test_unregistered_gate_rejected(gate):
    model, wrapper = objects()
    with pytest.raises(ValueError):
        _FrozenComputeWrapper(model, wrapper.transform, alpha=.7, gate=gate,
            expected_state_sha256=state_hash(wrapper.transform), layer=19, enabled=True,
            emergency_bounds=(.25, 4.))


def test_wrong_control_artifact_identity_rejected():
    model, wrapper = objects()
    with pytest.raises(RuntimeError, match="calibrated"):
        _FrozenComputeWrapper(model, wrapper.transform, alpha=.7, gate=.1,
            expected_state_sha256="0" * 64, layer=19, enabled=True, emergency_bounds=(.25, 4.))


def test_production_rejects_toy_geometry_without_allocating_large_control(monkeypatch):
    model, _ = objects()
    import cross_scale.generation_stage.v1.src.compute_control as module
    monkeypatch.setattr(module, "new_transform", lambda: pytest.fail("allocation before geometry validation"))
    with pytest.raises(ValueError, match="Supa"):
        CrossScaleComputeControl(model, alpha=.7, gate=.1, expected_state_sha256="0" * 64)
