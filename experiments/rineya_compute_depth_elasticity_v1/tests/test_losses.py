import pytest
import torch
import torch.nn.functional as F
from experiments.rineya_compute_depth_elasticity_v1.losses import ARMS, objective


def data():
    gen = torch.Generator().manual_seed(17)
    logits = torch.randn(4, 2, 3, 5, generator=gen, requires_grad=True)
    labels = torch.tensor([[0, 1, -100], [2, 3, 4]])
    return logits, labels, labels != -100


@pytest.mark.parametrize("arm", ARMS)
def test_all_five_arms_are_differentiable(arm):
    logits, labels, mask = data()
    kwargs = {"sampled_depth": 2} if arm == "sampled" else {}
    out = objective(logits, labels, mask, arm=arm, **kwargs)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, 0, 2].abs().sum() == 0


def test_exact_full_depth_ce():
    logits, labels, mask = data()
    out = objective(logits, labels, mask, arm="full")
    assert torch.allclose(out["loss"], F.cross_entropy(logits[-1][mask], labels[mask]))


def test_detached_deepest_teacher_receives_no_kl_gradient():
    logits, labels, mask = data()
    objective(logits, labels, mask, arm="deepest")["depth_kl"].backward()
    assert logits.grad[-1].abs().sum() == 0
    assert logits.grad[:-1].abs().sum() > 0


def test_adjacent_last_exit_teacher_detached_but_middle_is_student():
    logits, labels, mask = data()
    objective(logits, labels, mask, arm="adjacent")["depth_kl"].backward()
    assert logits.grad[-1].abs().sum() == 0
    assert logits.grad[1].abs().sum() > 0


def test_identical_depths_zero_kl_even_when_wrong():
    logits, labels, mask = data()
    logits = logits[:1].expand(4, -1, -1, -1)
    out = objective(logits, labels, mask)
    assert out["depth_kl"].item() == pytest.approx(0, abs=1e-7)
    assert out["task_loss"].item() > 0


def test_kl_direction_temperature_and_normalization():
    logits, labels, mask = data()
    out = objective(logits, labels, mask, temperature=2)
    z = logits[:, mask].float()
    teachers = F.softmax(z[1:].detach() / 2, -1)
    manual = (teachers * (F.log_softmax(z[1:].detach() / 2, -1)
                          - F.log_softmax(z[:-1] / 2, -1))).sum(-1).mean() * 4
    assert torch.allclose(out["depth_kl"], manual, atol=1e-6)


@pytest.mark.parametrize("kwargs", [dict(arm="unknown"), dict(coefficient=-1),
                                   dict(temperature=0), dict(temperature=float("nan")),
                                   dict(arm="sampled"), dict(arm="sampled", sampled_depth=True),
                                   dict(arm="sampled", sampled_depth=5), dict(sampled_depth=1)])
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        objective(*data(), **kwargs)


def test_invalid_tensors_rejected():
    z, labels, mask = data()
    for args in ((z[0], labels, mask), (z, labels, torch.zeros_like(mask)),
                 (z, labels.float(), mask), (z, labels, mask.float()),
                 (z.detach().clone().fill_(float("nan")), labels, mask)):
        with pytest.raises(ValueError):
            objective(*args)


def test_distillation_needs_two_depths():
    z, y, m = data()
    with pytest.raises(ValueError):
        objective(z[:1], y, m)
    assert torch.isfinite(objective(z[:1], y, m, arm="full")["loss"])
