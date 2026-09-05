import hashlib
import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from native_armr.src.operators import (FrozenSwiGLU, NativeComputeControl,
                                      layer_matmul_flops, matched_control_width,
                                      operator_factory)
from native_armr.src.protocol import DEFAULT_CONFIG, load_protocol


def weights_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@pytest.fixture
def tiny_model():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=28, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8, pad_token_id=0,
                        eos_token_id=2, layer_types=["full_attention"] * 28)
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval().requires_grad_(False)


@pytest.fixture
def protocol(tiny_model):
    config = load_protocol(DEFAULT_CONFIG)
    # Only direct operator tests scale AM geometry to the tiny synthetic model;
    # the public protocol loader rejects this changed registered geometry.
    config["control"]["intermediate_size"] = matched_control_width(tiny_model.config)
    return config


@pytest.mark.parametrize("arm", ["A0", "A1", "A3", "A16", "AM"])
def test_native_order_positions_stem_and_restore(tiny_model, protocol, arm):
    model, seen = tiny_model, []
    for index, block in enumerate(model.model.layers):
        original = block.forward
        def spy(*args, _index=index, _original=original, **kwargs):
            seen.append((_index, kwargs["position_ids"], kwargs["position_embeddings"],
                         kwargs["attention_mask"], kwargs.get("past_key_values")))
            return _original(*args, **kwargs)
        block.forward = spy
    ids = torch.tensor([[1, 4, 6]])
    before = weights_hash(model)
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
        seen.clear()
        with operator_factory(model, arm, protocol) as wrapper:
            output = model(ids, use_cache=False).logits
        expected = list(range(28))
        if arm in {"A1", "A16"}:
            layer = 12 if arm == "A1" else 16
            expected.insert(layer + 1, layer)
        elif arm == "A3":
            expected[14:14] = [12, 13]
        assert [row[0] for row in seen] == expected
        assert [row[0] for row in seen[:3]] == [0, 1, 2]
        assert all(row[1] is seen[0][1] and row[2] is seen[0][2]
                   and row[3] is seen[0][3] and row[4] is None for row in seen)
        assert torch.equal(baseline, model(ids, use_cache=False).logits)
    assert weights_hash(model) == before
    assert all(not value.requires_grad and value.grad is None for value in model.parameters())
    assert torch.isfinite(output).all()
    if arm == "A0":
        assert torch.equal(baseline, output)
    if arm == "AM":
        assert wrapper.control_calls == 1
        assert wrapper.extra_matmul_flops_per_token == 2 * layer_matmul_flops(model.config)
        assert not torch.equal(baseline, output)


def test_control_disabled_and_alpha_zero_identity(tiny_model):
    model, ids = tiny_model, torch.tensor([[1, 4, 6]])
    with torch.inference_mode():
        base = model(ids, use_cache=False).logits
        for kwargs in ({"enabled": False}, {"alpha": 0.}):
            with NativeComputeControl(model, **kwargs):
                assert torch.equal(base, model(ids, use_cache=False).logits)
        assert not model._forward_pre_hooks


@pytest.mark.parametrize("arm", ["A1", "A3", "A16", "AM"])
def test_disabled_candidate_identity(tiny_model, protocol, arm):
    ids = torch.tensor([[1, 4, 6]])
    with torch.inference_mode():
        baseline = tiny_model(ids, use_cache=False).logits
        wrapper = operator_factory(tiny_model, arm, protocol)
        wrapper.enabled = False
        with wrapper:
            assert torch.equal(baseline, tiny_model(ids, use_cache=False).logits)
        assert wrapper.calls == 0
        assert not tiny_model._forward_pre_hooks


def test_random_control_reproducibility_and_rng_isolation():
    before = torch.random.get_rng_state().clone()
    one = FrozenSwiGLU(16, 80, seed=7301, std=.02, rms_epsilon=1e-6)
    two = FrozenSwiGLU(16, 80, seed=7301, std=.02, rms_epsilon=1e-6)
    other = FrozenSwiGLU(16, 80, seed=7302, std=.02, rms_epsilon=1e-6)
    assert torch.equal(before, torch.random.get_rng_state())
    assert weights_hash(one) == weights_hash(two)
    assert weights_hash(one) != weights_hash(other)


def test_control_initialization_ignores_default_dtype_and_preserves_reference_draws():
    reference = FrozenSwiGLU(16, 80, seed=7301, std=.02, rms_epsilon=1e-6)
    prior = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        changed_default = FrozenSwiGLU(16, 80, seed=7301, std=.02, rms_epsilon=1e-6)
    finally:
        torch.set_default_dtype(prior)
    assert weights_hash(reference) == weights_hash(changed_default)
    generator = torch.Generator(device="cpu").manual_seed(7301)
    for shape, parameter in zip(((80, 16), (80, 16), (16, 80)), reference.parameters()):
        expected = torch.randn(*shape, generator=generator, device="cpu", dtype=torch.float32) * .02
        assert torch.equal(parameter, expected)
        assert parameter.device.type == "cpu" and parameter.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("arm", ["A0", "A1", "A3", "A16", "AM"])
def test_cuda_native_operator_preserves_reference_weights_and_restores(tiny_model, protocol, arm):
    # This synthetic mechanics test is not real-model runtime qualification.
    import copy
    cpu_model = tiny_model
    gpu_model = copy.deepcopy(cpu_model).to("cuda:0")
    seen = []
    for index, block in enumerate(gpu_model.model.layers):
        original = block.forward
        def spy(*args, _index=index, _original=original, **kwargs):
            seen.append((_index, kwargs["position_ids"], kwargs["position_embeddings"],
                         kwargs["attention_mask"], kwargs.get("past_key_values")))
            return _original(*args, **kwargs)
        block.forward = spy
    ids = torch.tensor([[1, 4, 6]])
    before = weights_hash(cpu_model)
    prior_matmul, prior_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cpu_wrapper = operator_factory(cpu_model, arm, protocol)
        gpu_wrapper = operator_factory(gpu_model, arm, protocol)
        if arm == "AM":
            assert weights_hash(cpu_wrapper.transform) == weights_hash(gpu_wrapper.transform)
            assert all(p.device.type == "cuda" and p.dtype == torch.float32
                       and not p.requires_grad and p.grad is None
                       for p in gpu_wrapper.transform.parameters())
        with torch.inference_mode():
            baseline = gpu_model(ids.to("cuda:0"), use_cache=False).logits
            was_enabled = gpu_wrapper.enabled
            gpu_wrapper.enabled = False
            with gpu_wrapper:
                assert torch.equal(baseline, gpu_model(ids.to("cuda:0"), use_cache=False).logits)
            gpu_wrapper.enabled = was_enabled
            seen.clear()
            with cpu_wrapper:
                cpu_output = cpu_model(ids, use_cache=False).logits
            with gpu_wrapper:
                gpu_output = gpu_model(ids.to("cuda:0"), use_cache=False).logits
                repeated = gpu_model(ids.to("cuda:0"), use_cache=False).logits
            expected = list(range(28))
            if arm in {"A1", "A16"}:
                layer = 12 if arm == "A1" else 16
                expected.insert(layer + 1, layer)
            elif arm == "A3":
                expected[14:14] = [12, 13]
            assert [row[0] for row in seen] == expected * 2
            for offset in (0, len(expected)):
                call = seen[offset:offset + len(expected)]
                assert [row[0] for row in call[:3]] == [0, 1, 2]
                assert all(row[1] is call[0][1] and row[2] is call[0][2]
                           and row[3] is call[0][3] and row[4] is None for row in call)
            torch.testing.assert_close(gpu_output.cpu(), cpu_output, atol=2e-5, rtol=1e-4)
            assert torch.equal(gpu_output, repeated)
            assert torch.equal(baseline, gpu_model(ids.to("cuda:0"), use_cache=False).logits)
        assert weights_hash(gpu_model) == before == weights_hash(cpu_model)
        assert all(not p.requires_grad and p.grad is None for p in gpu_model.parameters())
        assert not gpu_model._forward_pre_hooks
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul
        torch.backends.cudnn.allow_tf32 = prior_cudnn


@pytest.mark.parametrize("failure", ["gradient", "nonfinite", "rms"])
def test_compute_control_fail_closed_and_restore(tiny_model, failure):
    model, ids = tiny_model, torch.tensor([[1, 4]])
    wrapper = NativeComputeControl(model)
    if failure == "gradient":
        wrapper.transform.up.requires_grad_(True)
    elif failure == "nonfinite":
        wrapper.transform.down.data.fill_(float("nan"))
    else:
        wrapper.bounds = (2., 3.)
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
        with pytest.raises((RuntimeError, FloatingPointError)):
            with wrapper:
                model(ids, use_cache=False)
        assert torch.equal(baseline, model(ids, use_cache=False).logits)
    assert not model._forward_pre_hooks
    assert "_armr_owner" not in model.model.layers[13].__dict__


@pytest.mark.parametrize("cache", [True, None])
def test_compute_control_requires_explicit_cache_disabled(tiny_model, cache):
    with torch.inference_mode(), NativeComputeControl(tiny_model) as wrapper:
        with pytest.raises(RuntimeError, match="use_cache=False"):
            tiny_model(torch.tensor([[1, 4]]), **({} if cache is None else {"use_cache": cache}))
    assert wrapper.control_calls == 0


def test_exact_canonical_work_and_parameter_budget():
    class Canonical:
        hidden_size = 1024
        intermediate_size = 3072
        num_attention_heads = 16
        num_key_value_heads = 8
        head_dim = 128
    config = Canonical()
    assert matched_control_width(config) == 10240
    assert layer_matmul_flops(config) == 31457280
    assert 3 * config.hidden_size * matched_control_width(config) == 31457280
    assert 6 * config.hidden_size * matched_control_width(config) == 62914560


@pytest.mark.parametrize("arm", ["A2", "A14", "router", "XG"])
def test_unregistered_native_arm_rejected(tiny_model, protocol, arm):
    with pytest.raises(ValueError, match="unregistered native arm"):
        operator_factory(tiny_model, arm, protocol)


@pytest.mark.parametrize("mutation", ["layer", "seed", "gate", "width", "hierarchy"])
def test_protocol_rejects_unregistered_scientific_changes(tmp_path, mutation):
    config = json.loads(DEFAULT_CONFIG.read_text())
    if mutation == "layer":
        config["arms"]["A16"]["layer"] = 17
    elif mutation == "seed":
        config["control"]["seed"] += 1
    elif mutation == "gate":
        config["control"]["alpha"] = .1
    elif mutation == "width":
        config["control"]["intermediate_size"] -= 1
    else:
        config["analysis"]["primary_comparisons_hierarchical"].reverse()
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_protocol(path)


def test_preservation_strata_exist_and_diagnostics_separate():
    from shared.datasets import FAMILIES
    config = load_protocol(DEFAULT_CONFIG)
    assert set(config["analysis"]["preservation_families"]) <= set(FAMILIES)
    assert config["data"]["target_primary_examples"] == 1024
    assert config["data"]["diagnostic_variants_separate"]
