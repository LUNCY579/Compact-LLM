"""Frozen random CPU fixture and independent analytic sensitivity checks only."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.sensitivity import (
    DiagnosticBlock, diagnose, fixed_rollout, moment_energies,
)
from experiments.rineya_taps_adaptive_recurrence_v1.integration import BlockRecurrence
from experiments.rineya_taps_adaptive_recurrence_v1.scheduler import TAPSConfig


class Layer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.98 + index * .001))
        self.bias = nn.Parameter(torch.tensor([.04, -.03]))
        self.nonlinear = index >= 14

    def forward(self, hidden_states, **kwargs):
        value = hidden_states * self.scale + self.bias
        return torch.tanh(value) if self.nonlinear else value


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(11, 2)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(22).reshape(11, 2).float() / 20)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(Layer(i) for i in range(16))
        self.head = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1.3, -.4], [-.2, 1.1], [.5, .7]]))
        self.register_buffer("abstain_offset", torch.tensor(3.))
        self.config = SimpleNamespace(layer_types=["full_attention"] * 16)
        self.forward_calls, self.seen_inputs, self.fail_suffix, self.bad_position = 0, [], False, False
        self.eval().requires_grad_(False)

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=1, **kwargs):
        self.forward_calls += 1
        self.seen_inputs.append(input_ids.detach().clone())
        hidden = self.embedding(input_ids)
        length = input_ids.shape[1]
        position = torch.arange(length, device=input_ids.device).unsqueeze(0) + int(self.bad_position)
        causal = torch.zeros((1, 1, length, length), device=input_ids.device)
        embeddings = (torch.ones_like(hidden), torch.zeros_like(hidden))
        layer_kwargs = {"position_ids": position, "attention_mask": causal,
                        "position_embeddings": embeddings, "cache_position": position[0]}
        for index, layer in enumerate(self.model.layers):
            if index == 14 and self.fail_suffix:
                raise RuntimeError("forced fixture suffix failure")
            hidden = layer(hidden, **layer_kwargs)
        logits = self.head(hidden).clone()
        logits[..., 2] = logits[..., 2] + self.abstain_offset
        return SimpleNamespace(logits=logits[:, -logits_to_keep:], past_key_values=None)


@pytest.fixture
def model():
    return FixtureModel()


@pytest.fixture
def ids():
    return torch.tensor([[3, 5, 7]], dtype=torch.long)


def test_unit_and_noextra_values_match_ordinary_and_legacy(model, ids):
    with torch.no_grad():
        ordinary = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        with BlockRecurrence(model, config=TAPSConfig(scheduler_mode="unit")):
            legacy = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    with DiagnosticBlock(model, extra_loops=0) as block:
        noextra = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        assert block.d2 is None
    with DiagnosticBlock(model) as block:
        unit = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        assert torch.equal(block.leaf.detach(), block.proposal)
        assert block.leaf.is_leaf and block.leaf.requires_grad and block.leaf.grad_fn is None
    assert torch.equal(ordinary, noextra) and torch.equal(legacy, unit)


def test_actual_nonlinear_suffix_custom_offset_gradient_matches_recomputed_fd(model, ids):
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = diagnose(model, ids, 0)
    assert result["full_model_calls"] == 3 and result["state_gradient_evaluations"] == 1
    assert model.forward_calls == 3 and all(torch.equal(x, ids) for x in model.seen_inputs)
    assert result["absolute_error"] < 3e-4
    assert result["nominal"]["state_gradient_norm"] > 0
    assert result["nominal"]["anchor_counts"] == [1, 1, 1]
    assert result["plus"]["state_gradient_evaluations"] == result["minus"]["state_gradient_evaluations"] == 0
    assert not result["trajectory_reuse"] and not result["gold_enters_forward_input"]
    assert not result["generation_pool_access"]
    assert all(p.requires_grad is False and p.grad is None for p in model.parameters())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert all("_sensitivity_owner" not in layer.__dict__ for layer in model.model.layers)


def test_independent_analytic_nonlinear_readout_and_abstention_offset():
    h = torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)
    delta = torch.tensor([.2, -.1], dtype=torch.float64)
    def loss(state):
        z = torch.tanh(.7 * state + .2)
        logits = torch.stack((z[0] + z[1]**2, -z[0]**2 + .5*z[1], z[0]*z[1] + 3.))
        return -torch.log_softmax(logits, 0)[0]
    actual, = torch.autograd.grad(loss(h), h)
    z = torch.tanh(.7 * h.detach() + .2)
    logits = torch.stack((z[0] + z[1]**2, -z[0]**2 + .5*z[1], z[0]*z[1] + 3.))
    coefficients = logits.softmax(0) - torch.tensor([1., 0., 0.], dtype=torch.float64)
    jacobian = torch.stack((torch.stack((torch.tensor(1., dtype=torch.float64), 2*z[1])),
                            torch.stack((-2*z[0], torch.tensor(.5, dtype=torch.float64))),
                            torch.stack((z[1], z[0]))))
    expected = .7 * (1-z.square()) * (jacobian.T @ coefficients)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    difference = (loss(h.detach() + .01*delta)-loss(h.detach()-.01*delta))/.02
    assert float(difference) == pytest.approx(float(expected @ delta), abs=1e-6)


def test_weighted_variance_identity_against_centered_double_precision():
    a, b = torch.tensor([[1., -2., 3.]]), torch.tensor([[-.2, 1.4, 2.]])
    p, s, _ = moment_energies(a, b)
    weights = torch.tensor([.8/1.8, 1/1.8], dtype=torch.float64)
    observations = torch.stack((a.double(), b.double()))
    mean = (weights[:, None, None] * observations).sum(0)
    variance = (weights[:, None] * (observations-mean).square().mean(2)).sum(0)
    assert float(p) == pytest.approx(float(mean.square().mean()), abs=5e-7)
    assert float(s) == pytest.approx(float(variance), abs=5e-7)


def test_stable_wrong_updates_have_positive_balance_but_harm_loss():
    first, second = torch.tensor([[1.]]), torch.tensor([[.8]])
    _, _, balance = moment_energies(first, second)
    h = torch.tensor([[1.8]], requires_grad=True)
    gradient, = torch.autograd.grad(.5*h.square().sum(), h)
    assert .8 < 1  # fixed map F(h)=.8h+1 is locally contracting
    assert float(balance) > 0 and float((gradient*second).sum()) > 0
    assert .5*(1 + 1.1*.8)**2 > .5*1.8**2  # over-relaxation moves farther from target0


def test_forced_suffix_exception_restores_all_methods_and_owners(model, ids):
    originals = [layer.forward for layer in model.model.layers]
    hooks = (len(model._forward_pre_hooks), len(model._forward_hooks))
    model.fail_suffix = True
    with pytest.raises(RuntimeError, match="forced fixture suffix"):
        fixed_rollout(model, ids, 0)
    assert all(layer.forward == original for layer, original in zip(model.model.layers, originals))
    assert hooks == (len(model._forward_pre_hooks), len(model._forward_hooks))
    assert all("_sensitivity_owner" not in layer.__dict__ for layer in model.model.layers)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("kwargs", [
    {"use_cache": True}, {"use_cache": False, "past_key_values": object()},
])
def test_cache_is_rejected_before_fixture_forward(model, ids, kwargs):
    with pytest.raises(RuntimeError, match="cache"), DiagnosticBlock(model):
        model(input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs)
    assert model.forward_calls == 0


@pytest.mark.parametrize("shape", [(2, 3), (1, 257)])
def test_unsupported_shape_rejects_before_forward(model, shape):
    tokens = torch.ones(shape, dtype=torch.long)
    with pytest.raises(RuntimeError, match="prefix<=256"), DiagnosticBlock(model):
        model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
    assert model.forward_calls == 0


def test_padding_position_and_training_guards(model, ids):
    with pytest.raises(RuntimeError, match="unpadded"), DiagnosticBlock(model):
        model(input_ids=ids, attention_mask=torch.tensor([[1, 0, 1]]), use_cache=False)
    model.bad_position = True
    with pytest.raises(RuntimeError, match="positions"):
        fixed_rollout(model, ids, 0)
    model.bad_position = False
    model.train()
    with pytest.raises(RuntimeError, match="frozen eval"):
        fixed_rollout(model, ids, 0)


def test_inference_mode_and_trainable_parameter_refusal(model, ids):
    with torch.inference_mode(), pytest.raises(RuntimeError, match="requires grad mode"), DiagnosticBlock(model):
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    next(model.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="parameters must remain frozen"):
        fixed_rollout(model, ids, 0)


def test_missing_target_no_forward_noextra_no_fake_second_observation(model, ids):
    assert diagnose(model, ids, None)["full_model_calls"] == 0 and model.forward_calls == 0
    result = fixed_rollout(model, ids, 0, extra_loops=0)
    assert result["sensitivity"] == 0
    assert result["raw_update_norm"] is result["progress"] is result["fluctuation"] is None


def test_native_resource_contract_refuses_without_touching_cuda(model):
    with pytest.raises(RuntimeError, match="explicit resource guard"):
        DiagnosticBlock(model, device="cuda:0")
    with pytest.raises(RuntimeError, match="explicit CUDA synchronization"):
        fixed_rollout(model, torch.ones((1, 2), dtype=torch.long), 0, device="cuda:0", guard=lambda: None)


def test_resource_failure_restores_wrapper_before_next_attempt(model, ids):
    checks = []
    def refuse():
        checks.append(1)
        if len(checks) == 2:
            raise MemoryError("fixture resource ceiling")
    originals = [layer.forward for layer in model.model.layers]
    with pytest.raises(MemoryError, match="fixture resource ceiling"):
        fixed_rollout(model, ids, 0, guard=refuse)
    assert all(layer.forward == original for layer, original in zip(model.model.layers, originals))
    assert all(p.grad is None for p in model.parameters())
