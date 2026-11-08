import pytest
import torch
from torch.autograd import gradcheck

from experiments.rineya_distillation_dynamics_v1.contracts import ContractError
from experiments.rineya_distillation_dynamics_v1.objectives import dense_kl, projected_ce, projected_kl


def tensors(dtype=torch.float64):
    g = torch.Generator().manual_seed(41)
    sh = torch.randn(5, 3, generator=g, dtype=dtype, requires_grad=True)
    sw = torch.randn(11, 3, generator=g, dtype=dtype)
    th = torch.randn(5, 7, generator=g, dtype=dtype, requires_grad=True)
    tw = torch.randn(11, 7, generator=g, dtype=dtype)
    return sh, sw, th, tw


@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("chunk", [1, 2, 5, 20])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_dense_value_and_gradient(direction, chunk, dtype):
    sh, sw, th, tw = tensors(dtype)
    loss = projected_kl(sh, sw, th, tw, direction, chunk)
    ref = dense_kl(sh @ sw.T, th @ tw.T, direction)
    a = torch.autograd.grad(loss, sh, retain_graph=True)[0]
    b = torch.autograd.grad(ref, sh)[0]
    assert torch.allclose(loss, ref, atol=2e-6, rtol=2e-6)
    assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)
    loss.backward()
    assert th.grad is None and tw.grad is None and sw.grad is None


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_finite_difference(direction):
    sh, sw, th, tw = tensors()
    assert gradcheck(lambda x: projected_kl(x, sw, th, tw, direction, 2), (sh,), eps=1e-6)


@pytest.mark.parametrize("chunk", [1, 2, 8])
def test_ce_reference(chunk):
    sh, sw, _, _ = tensors()
    labels = torch.tensor([1, 2, 3, 4, 5])
    loss = projected_ce(sh, sw, labels, chunk)
    ref = torch.nn.functional.cross_entropy(sh @ sw.T, labels)
    assert torch.allclose(loss, ref)
    assert torch.allclose(torch.autograd.grad(loss, sh)[0], torch.autograd.grad(ref, sh)[0])


def test_reverse_is_not_forward_gradient():
    sh, sw, th, tw = tensors()
    f = torch.autograd.grad(projected_kl(sh, sw, th, tw, "forward"), sh)[0]
    r = torch.autograd.grad(projected_kl(sh, sw, th, tw, "reverse"), sh)[0]
    assert not torch.allclose(f, r)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_zero_kl_at_identical_distributions(direction):
    sh, sw, _, _ = tensors()
    result = projected_kl(sh, sw, sh.detach(), sw, direction)
    assert abs(float(result.detach())) < 1e-12
    assert torch.autograd.grad(result, sh)[0].abs().max() < 1e-12


@pytest.mark.parametrize("kind", ["empty", "vocab", "head_grad", "zero_chunk", "direction", "nan", "teacher_length"])
def test_fail_closed(kind):
    sh, sw, th, tw = tensors()
    direction, chunk = "forward", 2
    if kind == "empty": sh, th = sh[:0], th[:0]
    if kind == "vocab": tw = tw[:-1]
    if kind == "head_grad": sw.requires_grad_(True)
    if kind == "zero_chunk": chunk = 0
    if kind == "direction": direction = "sampled-reverse"
    if kind == "nan": th = torch.full_like(th, float("nan"))
    if kind == "teacher_length": th = th[:-1]
    with pytest.raises(ContractError):
        projected_kl(sh, sw, th, tw, direction, chunk)


def test_ce_rejects_negative_label():
    sh, sw, _, _ = tensors()
    with pytest.raises(ContractError):
        projected_ce(sh, sw, torch.tensor([-100, 2, 3, 4, 5]))


def test_no_response_length_vocabulary_history_saved():
    sh, sw, th, tw = tensors()
    result = projected_kl(sh, sw, th, tw)
    saved = result.grad_fn.saved_tensors
    assert [tuple(t.shape) for t in saved] == [tuple(x.shape) for x in (sh, sw, th, tw)]
