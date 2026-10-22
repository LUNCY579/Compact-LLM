from dataclasses import replace
import math
import pytest
import torch
from experiments.rineya_layered_post_training_v1.data import fixtures, score
from experiments.rineya_layered_post_training_v1.runtime import dpo_loss, critic_objective, prefix_features
from experiments.rineya_layered_post_training_v1.soup import state_hash
from experiments.progressive_rl_curriculum_v1.objectives import advantages, policy_loss


def test_dpo_uses_sequence_sum_and_detaches_reference():
    c = torch.tensor([-1., -2.], requires_grad=True)
    r = torch.tensor([-3.], requires_grad=True)
    rc = torch.tensor([-2., -2.], requires_grad=True)
    rr = torch.tensor([-2.], requires_grad=True)
    loss = dpo_loss(c, r, rc, rr, .1)
    torch.testing.assert_close(loss, -torch.nn.functional.logsigmoid(torch.tensor(.2)))
    loss.backward()
    assert c.grad is not None and r.grad is not None and rc.grad is None and rr.grad is None


@pytest.mark.parametrize("reward", [0., 1.])
def test_raw_critic_not_probability(reward):
    values = torch.tensor([-2., 3.], requires_grad=True)
    loss = critic_objective(values, reward)
    torch.testing.assert_close(loss, .5 * ((values - reward)**2).mean())
    loss.backward()
    assert torch.isfinite(values.grad).all()


def test_prefix_alignment():
    h = torch.arange(18.).reshape(1, 6, 3)
    assert prefix_features(h, 3, 2).tolist() == h[0, 2:4].tolist()
    changed = h.clone()
    changed[:, 4:] = -100
    torch.testing.assert_close(prefix_features(h, 3, 2), prefix_features(changed, 3, 2))


def test_grpo_zero_signal_and_mask():
    assert advantages([0., 0., 0.]) == [0., 0., 0.]
    assert advantages([1., 1.]) == [0., 0.]
    x = torch.tensor([[-1., -2.]], requires_grad=True)
    old = x.detach().clone()
    mask = torch.tensor([[True, False]])
    loss = policy_loss(x, old, old, torch.tensor([1.]), mask)
    loss.backward()
    assert x.grad[0, 1] == 0


def test_sampler_gold_blind_and_reproducible(rt):
    first = rt.collect("sum 0 0", 4, 17)
    second = rt.collect("sum 0 0", 4, 17)
    assert first == second
    assert all(ep.terminated and len(ep.response) == 2 for ep in first)
    assert "target" not in __import__("inspect").signature(rt.collect).parameters
    with pytest.raises(ValueError):
        rt.prompt_ids("unrecognized test prompt")


@pytest.mark.parametrize("method", ["sft", "dpo", "rl", "critic"])
def test_real_autograd_updates(rt, method):
    row = fixtures()[0]
    before_actor = state_hash(rt.state())
    before_critic = state_hash(rt.critic.state_dict())
    if method == "sft":
        loss = rt.sft(row, fixtures()[25])
    elif method == "dpo":
        loss = rt.dpo(row, fixtures()[25])
    else:
        eps = rt.collect(row.prompt, 4, 11)
        # Explicit synthetic labels test objective mechanics, not verifier performance.
        rewards = [0., 1., 0., 1.]
        loss = rt.rl(eps, rewards, fixtures()[25], use_critic=True) if method == "rl" else rt.critic_fit(eps, rewards)
    assert math.isfinite(loss)
    if method == "critic":
        assert before_actor == state_hash(rt.state())
        assert before_critic != state_hash(rt.critic.state_dict())
    else:
        assert before_actor != state_hash(rt.state())


def test_stale_and_truncated_episodes_rejected(rt):
    eps = rt.collect("sum 0 0", 2, 0)
    rt.restore(rt.state())
    with pytest.raises(ValueError, match="stale"):
        rt.rl(eps, [0., 1.])
    eps = rt.collect("sum 0 0", 2, 0)
    with pytest.raises(ValueError, match="truncated"):
        rt.rl([replace(ep, terminated=False) for ep in eps], [0., 1.])


def test_sampler_trainer_mutation_rejected(rt):
    eps = rt.collect("sum 0 0", 2, 0)
    forged = [replace(ep, old_logps=tuple(x - 1 for x in ep.old_logps)) for ep in eps]
    with pytest.raises(ValueError, match="likelihood"):
        rt.rl(forged, [0., 1.])


def test_soup_resets_populated_adam(rt):
    rt.sft(fixtures()[0])
    assert rt.optimizer.state
    rt.restore(rt.state())
    assert not rt.optimizer.state


def test_reference_is_frozen(rt):
    before = state_hash(dict(rt.reference.named_parameters()))
    rt.dpo(fixtures()[0])
    assert before == state_hash(dict(rt.reference.named_parameters()))
    assert all(p.grad is None and not p.requires_grad for p in rt.reference.parameters())


def test_causal_features_cannot_see_future_response(rt):
    p = rt.prompt_ids("sum 0 0")
    _, a = rt.logps(p, (5, 1), features=True)
    _, b = rt.logps(p, (9, 1), features=True)
    torch.testing.assert_close(a[0], b[0])
    assert not torch.allclose(a[1], b[1])
