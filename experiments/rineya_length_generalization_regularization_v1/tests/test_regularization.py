import copy
from dataclasses import replace
import pytest
import torch
from torch.utils.checkpoint import checkpoint

from experiments.rineya_length_generalization_regularization_v1.config import Plan
from experiments.rineya_length_generalization_regularization_v1.fixtures import TinyLM
from experiments.rineya_length_generalization_regularization_v1.regularization import (
    vpad, norm_targets, regularize, activation_probe)


@pytest.mark.parametrize("mode", ["none", "pre_linear", "vpad", "pre_norm_control"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exact_disabled_eval_and_state_dict(mode, dtype):
    model = TinyLM().to(dtype)
    plan = Plan(mode=mode, probability=0 if mode == "none" else 0.3, expected_layers=2)
    ids = torch.tensor([[10, 20, 30, 40]])
    names = list(model.state_dict())
    parameters = [id(p) for p in model.parameters()]
    model.eval()
    before = model(ids).logits
    with regularize(model, plan) as patched:
        assert len(patched) == 4
        assert torch.equal(before, model(ids).logits)
        assert "forward" not in model.model.layers[0].self_attn.q_norm.__dict__
        assert "forward" not in model.model.layers[0].self_attn.k_norm.__dict__
        assert "forward" not in model.model.norm.__dict__
        assert list(model.state_dict()) == names
        assert [id(p) for p in model.parameters()] == parameters
    model.train()
    before = model(ids).logits
    with regularize(model, replace(plan, probability=0)):
        assert torch.equal(before, model(ids).logits)
    assert all("forward" not in m.__dict__ for _, m, _ in norm_targets(model, 2))


@pytest.mark.parametrize("value", [0.0, 1.0, -2.0, 1e-10])
@pytest.mark.parametrize("detach_alpha", [True, False])
@pytest.mark.parametrize("detach_beta", [True, False])
def test_degenerate_inputs_finite(value, detach_alpha, detach_beta):
    x = torch.full((2, 3, 16), value, requires_grad=True)
    result = vpad(x, 0.3, 1e-6, detach_alpha=detach_alpha, detach_beta=detach_beta)
    result.square().sum().backward()
    assert torch.isfinite(result).all() and torch.isfinite(x.grad).all()


def test_moment_match_and_randomness():
    torch.manual_seed(47)
    x = torch.tensor([-2.0, 1.0, 3.0, 5.0]).expand(100000, -1)
    standard = x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    result = vpad(x, 0.3, 1e-6)
    assert abs(result.mean().item() - standard.mean().item()) < 0.01
    assert abs(result.var(unbiased=False).item() - standard.var(unbiased=False).item()) < 0.02
    assert not torch.equal(result, standard)


@pytest.mark.parametrize("probability", [-1.0, 1.0, float("nan")])
def test_invalid_probability(probability):
    with pytest.raises(ValueError):
        vpad(torch.ones(2, 4), probability, 1e-6)


def test_restore_after_exception_and_reject_nested():
    model = TinyLM()
    plan = Plan(expected_layers=2, mode="vpad", probability=0.1)
    with pytest.raises(RuntimeError):
        with regularize(model, plan):
            with pytest.raises(ValueError):
                with regularize(model, plan):
                    pass
            raise RuntimeError("fixture")
    assert len(norm_targets(model, 2)) == 4
    with pytest.raises(ValueError):
        norm_targets(model, 3)


@pytest.mark.parametrize("mode", ["pre_linear", "vpad", "pre_norm_control"])
def test_checkpoint_rng_equivalence(mode):
    plan = Plan(expected_layers=2, mode=mode, probability=0.3)
    direct, replay = TinyLM(), TinyLM()
    replay.load_state_dict(direct.state_dict())
    ids = torch.tensor([[8, 10, 13, 22]])
    def compute(model, ckpt):
        with regularize(model, plan):
            torch.manual_seed(33)
            loss = checkpoint(lambda: model(ids).logits.square().mean(), use_reentrant=False,
                              preserve_rng_state=True) if ckpt else model(ids).logits.square().mean()
            loss.backward()
        return loss.detach(), [p.grad for p in model.parameters()]
    loss1, grad1 = compute(direct, False)
    loss2, grad2 = compute(replay, True)
    assert torch.equal(loss1, loss2)
    assert all(torch.equal(a, b) for a, b in zip(grad1, grad2))


def test_bounded_probe_does_not_leave_hooks():
    model = TinyLM()
    plan = Plan(expected_layers=2, mode="vpad", probability=0.1)
    with activation_probe(model, plan, maximum_calls=2) as records:
        with regularize(model, plan):
            model(torch.tensor([[10, 20, 30]]))
    assert len(records) == 2 and all(r["finite"] for r in records)
    assert all(not m._forward_hooks for _, m, _ in norm_targets(model, 2))
