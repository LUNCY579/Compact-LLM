"""Tiny actual-HF mechanics fixtures; no production model/capability claims."""
from copy import deepcopy

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from shared.integrity import state_hash
from native_armr.src.learned_control import (
    FrozenParallelFFN, NativeFFNControl, SOURCE_LAYERS, SafetyError,
)
from native_armr.src.operators import layer_matmul_flops


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(214)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=18, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8, pad_token_id=0,
                        eos_token_id=2, layer_types=["full_attention"] * 18)
    config._attn_implementation = "eager"
    value = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    # Different nonunit learned gammas expose accidentally shared/unit norms.
    with torch.no_grad():
        for index in SOURCE_LAYERS:
            value.model.layers[index].post_attention_layernorm.weight.copy_(
                torch.linspace(.7, 1.3, 16) + (index - 12) * .2)
    return value


def transform_for(model):
    return FrozenParallelFFN.from_layers({index: model.model.layers[index] for index in SOURCE_LAYERS})


def test_intact_source_copies_provenance_and_parallel_not_sequential(model):
    before, source_keys = state_hash(model), set(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    transform = transform_for(model)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert transform.state_sha256 == state_hash(transform)
    assert transform.source_hashes == transform.copy_hashes
    assert not any("attn" in name for name, _ in transform.named_parameters()
                   if "post_attention_layernorm" not in name)
    hidden = torch.linspace(-1, 2, 3 * 16).reshape(1, 3, 16)
    with torch.inference_mode():
        native = [model.model.layers[i].mlp(model.model.layers[i].post_attention_layernorm(hidden))
                  for i in SOURCE_LAYERS]
        expected = (native[0] + native[1]) + native[2]
        assert torch.equal(transform(hidden), expected)
        sequential = hidden
        for i in SOURCE_LAYERS:
            layer = model.model.layers[i]
            sequential = sequential + layer.mlp(layer.post_attention_layernorm(sequential))
        assert not torch.equal(expected, sequential - hidden)
    for index in SOURCE_LAYERS:
        branch, source = transform.branches[str(index)], model.model.layers[index]
        receipt = transform.branch_provenance[str(index)]
        assert receipt["source_layer"] == index
        assert receipt["norm_epsilon"] == source.post_attention_layernorm.variance_epsilon
        assert receipt["activation_class"] == "SiLUActivation"
        for name in ("mlp", "post_attention_layernorm"):
            left, right = getattr(source, name), getattr(branch, name)
            assert type(left) is type(right)
            assert state_hash(left) == state_hash(right)
            for a, b in zip(left.parameters(), right.parameters(), strict=True):
                assert a is not b and a.data_ptr() != b.data_ptr()
                assert torch.equal(a, b) and not b.requires_grad and b.grad is None
    assert transform.assert_unchanged() == transform.state_sha256
    assert state_hash(model) == before and set(model.state_dict()) == source_keys


def test_branch_provenance_accessors_cannot_change_receipt(model):
    transform = transform_for(model)
    source = transform.source_hashes
    source["12"]["mlp"] = "changed"
    receipt = transform.branch_provenance
    receipt["12"]["source_layer"] = 0
    assert transform.source_hashes == transform.copy_hashes
    assert transform.branch_provenance["12"]["source_layer"] == 12


def test_fp32_left_associative_branch_sum_is_observable(model):
    # Actual HF branches with cancellation expose reassociation deterministically;
    # these synthetic fixture weights are established before control construction.
    with torch.no_grad():
        for index, magnitude in zip(SOURCE_LAYERS, (1e20, -1e20, 1.), strict=True):
            layer = model.model.layers[index]
            layer.post_attention_layernorm.weight.fill_(1.)
            for projection in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                projection.weight.zero_()
            layer.mlp.gate_proj.weight[0, 0] = 1.
            layer.mlp.up_proj.weight[0, 0] = 1.
            layer.mlp.down_proj.weight[0, 0] = magnitude
    transform = transform_for(model)
    hidden = torch.ones(1, 2, 16)
    with torch.inference_mode():
        first, second, third = (transform.branches[str(i)](hidden) for i in SOURCE_LAYERS)
        expected = (first + second) + third
        reassociated = first + (second + third)
        assert not torch.equal(expected, reassociated)
        assert torch.equal(transform(hidden), expected)


def test_source_mutation_does_not_change_wrapper_copy(model):
    transform = transform_for(model)
    before = transform.state_sha256
    with torch.no_grad():
        model.model.layers[14].mlp.up_proj.weight.add_(1)
    assert state_hash(transform) == before
    assert state_hash(model.model.layers[14].mlp) != transform.source_hashes["14"]["mlp"]


def test_wrapper_order_positions_stem_state_and_restore(model):
    calls = []
    for index, layer in enumerate(model.model.layers):
        original = layer.forward
        def spy(*args, _index=index, _original=original, **kwargs):
            calls.append((_index, kwargs["position_ids"], kwargs["position_embeddings"],
                          kwargs["attention_mask"], kwargs.get("past_key_values")))
            return _original(*args, **kwargs)
        layer.forward = spy
    ids, before = torch.tensor([[1, 4, 6]]), state_hash(model)
    source_keys = set(model.state_dict())
    wrapper = NativeFFNControl(model, alpha=.3)
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
        calls.clear()
        with wrapper:
            output = model(ids, use_cache=False).logits
        assert [row[0] for row in calls] == list(range(18))
        assert all(row[1] is calls[0][1] and row[2] is calls[0][2] and
                   row[3] is calls[0][3] and row[4] is None for row in calls)
        assert torch.equal(baseline, model(ids, use_cache=False).logits)
    assert not torch.equal(baseline, output)
    assert wrapper.calls == wrapper.control_calls == 1
    assert wrapper.extra_layer_applications == 0
    assert len(wrapper.records) == 2
    assert set(model.state_dict()) == source_keys and state_hash(model) == before
    assert "_armr_owner" not in model.model.layers[13].__dict__
    assert not model._forward_pre_hooks


@pytest.mark.parametrize("kwargs,expected_calls", [({"alpha": 0.}, 1),
                                                  ({"alpha": .3, "enabled": False}, 0)])
def test_zero_and_disabled_exact_identity_including_inference_construction(model, kwargs, expected_calls):
    ids = torch.tensor([[1, 4, 6]])
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
        with NativeFFNControl(model, **kwargs) as wrapper:
            assert torch.equal(baseline, model(ids, use_cache=False).logits)
        assert torch.equal(baseline, model(ids, use_cache=False).logits)
    assert wrapper.control_calls == expected_calls
    assert bool(wrapper.extra_matmul_flops_per_token) == bool(expected_calls)
    assert bool(wrapper.added_active_parameters) == bool(expected_calls)
    assert not model._forward_pre_hooks


@pytest.mark.parametrize("key,value", [("alpha", .2), ("enabled", False), ("bounds", (.1, 5.)),
                                      ("layer", 12), ("applications", 2), ("skip", True),
                                      ("transform", None), ("model", None)])
def test_active_treatment_mutation_rejected(model, key, value):
    wrapper = NativeFFNControl(model, alpha=.3)
    with wrapper:
        with pytest.raises(SafetyError, match="cannot change"):
            setattr(wrapper, key, value)
    assert not model._forward_pre_hooks


def test_alpha_change_before_activation_rejected(model):
    wrapper = NativeFFNControl(model, alpha=.3)
    wrapper.alpha = .2
    with pytest.raises(SafetyError, match="construction"):
        with wrapper:
            pass
    assert not model._forward_pre_hooks


@pytest.mark.parametrize("failure", ["training", "trainable", "gradient", "epsilon", "activation",
                                     "weight", "branch_order", "nonfinite_input"])
def test_transform_contract_guards(model, failure):
    transform = transform_for(model)
    branch = transform.branches["12"]
    hidden = torch.ones(1, 3, 16)
    if failure == "training":
        transform.train()
    elif failure == "trainable":
        branch.mlp.up_proj.weight.requires_grad_(True)
    elif failure == "gradient":
        branch.mlp.up_proj.weight.grad = torch.zeros_like(branch.mlp.up_proj.weight)
    elif failure == "epsilon":
        branch.post_attention_layernorm.variance_epsilon *= 2
    elif failure == "activation":
        branch.mlp.act_fn = nn.ReLU()
    elif failure == "weight":
        with torch.no_grad():
            branch.mlp.up_proj.weight.add_(1)
    elif failure == "branch_order":
        transform.branches = nn.ModuleDict(reversed(list(transform.branches.items())))
    else:
        hidden[0, 0, 0] = float("nan")
    with torch.inference_mode(), pytest.raises((SafetyError, FloatingPointError)):
        transform(hidden)


def test_data_mutation_detected_on_exit_after_canonical_restoration(model):
    ids = torch.tensor([[1, 4]])
    wrapper = NativeFFNControl(model, alpha=.3)
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
        with pytest.raises(SafetyError, match="state changed"):
            with wrapper:
                wrapper.transform.branches["12"].mlp.up_proj.weight.data.add_(1)
        assert torch.equal(baseline, model(ids, use_cache=False).logits)
    assert not model._forward_pre_hooks
    assert "_armr_owner" not in model.model.layers[13].__dict__


def test_rms_guard_failure_restores(model):
    ids = torch.tensor([[1, 4]])
    wrapper = NativeFFNControl(model, alpha=.3, emergency_bounds=(2., 3.))
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
        with pytest.raises(SafetyError, match="hidden RMS"):
            with wrapper:
                model(ids, use_cache=False)
        assert torch.equal(baseline, model(ids, use_cache=False).logits)
    assert not model._forward_pre_hooks


@pytest.mark.parametrize("kwargs", [{}, {"use_cache": True},
                                    {"use_cache": False, "past_key_values": object()}])
def test_cache_rejected_before_forward(model, kwargs):
    with torch.inference_mode(), NativeFFNControl(model, alpha=.3) as wrapper:
        with pytest.raises(SafetyError, match="cache"):
            model(torch.tensor([[1, 4]]), **kwargs)
    assert wrapper.calls == wrapper.control_calls == 0


@pytest.mark.parametrize("kwargs", [{"use_cache": True}, {"cache": object()},
                                    {"past_key_values": object()}, {"past_key_value": object()}])
def test_direct_decoder_cache_rejected(model, kwargs):
    with torch.inference_mode(), NativeFFNControl(model, alpha=.3) as wrapper:
        with pytest.raises(SafetyError, match="cache"):
            wrapper._forward(torch.ones(1, 2, 16), **kwargs)
    assert wrapper.calls == wrapper.control_calls == 0


def test_gradient_context_rejected(model):
    with NativeFFNControl(model, alpha=.3) as wrapper:
        with pytest.raises(SafetyError, match="inference/no_grad"):
            model(torch.tensor([[1, 4]]), use_cache=False)
    assert wrapper.control_calls == 0


@pytest.mark.parametrize("alpha", [-.1, float("nan"), float("inf"), True])
def test_invalid_alpha_rejected(model, alpha):
    with pytest.raises(ValueError, match="alpha"):
        NativeFFNControl(model, alpha=alpha)


def test_wrong_source_index_and_truncated_matrix_rejected(model):
    layers = {i: model.model.layers[i] for i in SOURCE_LAYERS}
    layers[14] = model.model.layers[15]
    with pytest.raises(SafetyError, match="identified"):
        FrozenParallelFFN.from_layers(layers)
    model.model.layers[14].mlp.gate_proj = nn.Linear(16, 12, bias=False).eval().requires_grad_(False)
    with pytest.raises(SafetyError, match="complete"):
        transform_for(model)


def test_exact_inventory_and_approximate_canonical_cost(model):
    wrapper = NativeFFNControl(model, alpha=.3)
    inventory = wrapper.parameter_inventory
    assert inventory == {"matrix_parameters": 9 * 16 * 24, "norm_parameters": 3 * 16,
                         "resident_copied_parameters": 9 * 16 * 24 + 3 * 16,
                         "active_copied_parameters": 9 * 16 * 24 + 3 * 16,
                         "resident_bytes": 4 * (9 * 16 * 24 + 3 * 16),
                         "newly_learned_parameters": 0,
                         "extra_matmul_flops_per_token": 18 * 16 * 24}
    assert wrapper.added_parameter_count == sum(p.numel() for p in wrapper.transform.parameters())
    assert wrapper.added_active_parameters == wrapper.added_parameter_count
    assert wrapper.extra_matmul_flops_per_token == 2 * inventory["matrix_parameters"]
    class Canonical:
        hidden_size, intermediate_size = 1024, 3072
        num_attention_heads, num_key_value_heads, head_dim = 16, 8, 128
    matrix = 9 * Canonical.hidden_size * Canonical.intermediate_size
    assert matrix == 28311552
    assert matrix + 3 * Canonical.hidden_size == 28314624
    assert 2 * matrix == 56623104
    assert 2 * matrix / (2 * layer_matmul_flops(Canonical())) == .9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_hash_provenance_repeat_and_restore(model):
    gpu_model = deepcopy(model).to("cuda:0")
    cpu = NativeFFNControl(model, alpha=.3)
    gpu = NativeFFNControl(gpu_model, alpha=.3)
    assert cpu.transform.state_sha256 == gpu.transform.state_sha256
    assert cpu.transform.branch_provenance == gpu.transform.branch_provenance
    transferred = transform_for(model).to("cuda:0")
    assert transferred.assert_unchanged() == cpu.transform.state_sha256
    ids = torch.tensor([[1, 4, 6]])
    before = state_hash(gpu_model)
    prior = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            baseline = gpu_model(ids.cuda(), use_cache=False).logits
            with cpu:
                reference = model(ids, use_cache=False).logits
            with gpu:
                first = gpu_model(ids.cuda(), use_cache=False).logits
                second = gpu_model(ids.cuda(), use_cache=False).logits
            assert torch.equal(first, second)
            torch.testing.assert_close(first.cpu(), reference, atol=2e-5, rtol=1e-4)
            assert torch.equal(baseline, gpu_model(ids.cuda(), use_cache=False).logits)
        assert state_hash(gpu_model) == before == state_hash(model)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prior
