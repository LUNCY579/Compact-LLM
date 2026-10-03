import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from armr.hooks import SafetyError
from graft import (Bridges, Graft, ResidualExpert, capture_native, fit_ridge,
                   positions_mask, scenario_split)


def config(width=16, head_dim=8):
    c = Qwen3Config(hidden_size=width, intermediate_size=32, num_hidden_layers=5,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=head_dim,
                    vocab_size=43, max_position_embeddings=64)
    c._attn_implementation = "eager"
    return c


def objects(gate=.1, identity=False):
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(config()).eval().requires_grad_(False)
    c = config(24, 12)
    donor = Qwen3DecoderLayer(c, 3).eval().requires_grad_(False)
    bridge = Bridges(torch.randn(16, 24)*.1, torch.randn(24)*.1,
                     torch.randn(24, 16)*.1, torch.ones(16)*11)
    expert = ResidualExpert(donor, c, bridge, gate, identity)
    return model, expert


@pytest.mark.parametrize("gate,identity", [(0, False), (.1, True)])
def test_exact_identity_and_restore(gate, identity):
    model, expert = objects(gate, identity)
    ids = torch.tensor([[2, 4, 6]])
    original = model.model.layers[3].forward
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
        with Graft(model, expert, layer=3):
            assert torch.equal(baseline, model(ids, use_cache=False).logits)
        assert model.model.layers[3].forward == original
        assert torch.equal(baseline, model(ids, use_cache=False).logits)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


def test_zero_gate_returns_original_tensor_without_call():
    _, expert = objects(0)
    expert.layer.forward = lambda *a, **k: pytest.fail("zero gate invoked donor")
    h = torch.randn(1, 3, 16)
    assert expert(h, attention_mask=None, position_ids=None) is h


def test_delta_subtracts_input_and_excludes_affine_bias():
    _, expert = objects(.1)
    h = torch.randn(1, 3, 16)
    pos, mask = positions_mask(h)
    expert.layer.forward = lambda z, **kwargs: z + 2
    expected = h + .1 * (torch.full((1, 3, 24), 2.) @ expert.bridges.down)
    with torch.inference_mode():
        assert torch.allclose(expert(h, attention_mask=mask, position_ids=pos), expected, atol=1e-6)


def test_future_tokens_cannot_change_prefix_with_different_donor_rope_geometry():
    model, expert = objects()
    a, b = torch.tensor([[2, 4, 6, 8]]), torch.tensor([[2, 4, 9, 11]])
    with torch.inference_mode(), Graft(model, expert, layer=3):
        first = model(a, use_cache=False).logits[:, :2]
        second = model(b, use_cache=False).logits[:, :2]
        assert torch.equal(first, second)


@pytest.mark.parametrize("kwargs", [{"use_cache": True}, {"use_cache": False, "past_key_values": object()}, {}])
def test_cache_rejected_before_model_or_donor(kwargs):
    model, expert = objects()
    with torch.inference_mode(), Graft(model, expert, layer=3):
        with pytest.raises(SafetyError):
            model(torch.tensor([[2, 3]]), **kwargs)


def test_exception_restores_wrapper():
    model, expert = objects()
    original = model.model.layers[3].forward
    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")
    expert.layer.forward = fail
    with pytest.raises(RuntimeError, match="injected"), torch.inference_mode():
        with Graft(model, expert, layer=3):
            model(torch.tensor([[2, 3]]), use_cache=False)
    assert model.model.layers[3].forward == original
    assert not model._forward_pre_hooks


def test_native_streaming_matches_monolithic_raw_layers(tmp_path):
    model, _ = objects()
    model.save_pretrained(tmp_path)
    ids = [torch.tensor([[2, 4, 6, 8]]), torch.tensor([[1, 5]])]
    expected = {1: [], 3: []}
    handles = [model.model.layers[i].register_forward_hook(
        lambda mod, args, out, index=i: expected[index].append(out[0].clone())) for i in expected]
    with torch.inference_mode():
        for prompt in ids:
            model(prompt, use_cache=False)
    for h in handles:
        h.remove()
    actual = capture_native(tmp_path, ids, (1, 3))
    for i in expected:
        for a, b in zip(actual[i], expected[i]):
            assert torch.equal(a, b)


def test_ridge_generalizes_known_affine_map_and_keeps_inputs():
    torch.manual_seed(8)
    x = torch.randn(100, 6)
    w, bias = torch.randn(6, 9), torch.randn(9)
    y, original = x @ w + bias, x.clone()
    fitted, b = fit_ridge(x[:80], y[:80], 1e-7)
    assert torch.allclose(x[80:] @ fitted+b, y[80:], atol=1e-5)
    assert torch.equal(x, original)


def test_scenario_split_groups_paraphrases():
    rows = [{"scenario_id": str(i//3)} for i in range(30)]
    train, held = scenario_split(rows, 17, 3)
    assert len(held) == 3 and sum(train) == 21
    assert all(len(set(train[i:i+3])) == 1 for i in range(0, 30, 3))


@pytest.mark.parametrize("layer", [0, 1, 2, 5, -1])
def test_stem_and_invalid_sites_rejected(layer):
    model, expert = objects()
    with pytest.raises(ValueError):
        Graft(model, expert, layer=layer)


def test_nonfinite_donor_stops():
    _, expert = objects()
    expert.layer.forward = lambda z, **kwargs: z * float("nan")
    h = torch.randn(1, 3, 16)
    pos, mask = positions_mask(h)
    with pytest.raises(FloatingPointError):
        expert(h, attention_mask=mask, position_ids=pos)
