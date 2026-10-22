import copy
import pytest
import torch
from experiments.rineya_self_distillation_axes_v1.objectives import (
    ema_update, reference_nll, response_logits, sequence_mean, token_kl,
)
from experiments.rineya_self_distillation_axes_v1.teacher import TeacherBank, tensor_digest


@pytest.mark.parametrize('direction', ['forward', 'reverse'])
def test_exact_kl_and_analytic_gradient(direction):
    p = torch.tensor([[.1, .2, .7]], dtype=torch.float64)
    q = torch.tensor([[.4, .3, .3]], dtype=torch.float64)
    z, target = p.log().requires_grad_(), q.log().requires_grad_()
    loss = token_kl(z, target, direction).sum()
    expected = (q * (q.log() - p.log())).sum() if direction == 'forward' else (p * (p.log() - q.log())).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    grad = p - q if direction == 'forward' else p * (p.log() - q.log() - expected)
    torch.testing.assert_close(z.grad, grad)
    assert target.grad is None


@pytest.mark.parametrize('direction', ['forward', 'reverse'])
def test_double_precision_gradcheck(direction):
    s = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    q = torch.randn_like(s)
    assert torch.autograd.gradcheck(lambda x: token_kl(x, q, direction), (s,))


def test_sequence_reduction_is_per_example_not_pooled():
    values = torch.tensor([[2., 0.], [4., 8.]])
    mask = torch.tensor([[True, False], [True, True]])
    assert sequence_mean(values, mask) == 4.
    with pytest.raises(ValueError):
        sequence_mean(values, torch.zeros_like(mask))


@pytest.mark.parametrize('length, response', [(1, 1), (2, 3), (9, 4)])
def test_causal_alignment(length, response):
    logits = torch.arange((length + response) * 5).view(1, length + response, 5)
    result = response_logits(logits, length, response)
    assert torch.equal(result[0, 0], logits[0, length - 1])
    assert torch.equal(result[0, -1], logits[0, length + response - 2])
    with pytest.raises(ValueError):
        response_logits(logits, length, response + 1)


@pytest.mark.parametrize('rate', [0., .02, .05, .1, .25, 1.])
def test_ema_orientation_endpoints_and_no_alias(rate):
    teacher = {'w': torch.tensor([2., -1.])}
    student = {'w': torch.tensor([4., 3.], requires_grad=True)}
    ema_update(teacher, student, rate)
    torch.testing.assert_close(teacher['w'], torch.tensor([2., -1.]) * (1 - rate) + torch.tensor([4., 3.]) * rate)
    assert not teacher['w'].requires_grad and teacher['w'].data_ptr() != student['w'].data_ptr()


@pytest.mark.parametrize('rate', [-.1, 1.1, float('nan'), True])
def test_bad_ema_rate(rate):
    with pytest.raises(ValueError):
        ema_update({'w': torch.ones(2)}, {'w': torch.ones(2)}, rate)


def test_ema_validation_is_transactional():
    teacher = {'a': torch.ones(2), 'b': torch.ones(2)}
    with pytest.raises(ValueError):
        ema_update(teacher, {'a': torch.zeros(2), 'b': torch.ones(3)}, .5)
    assert torch.equal(teacher['a'], torch.ones(2))


@pytest.mark.parametrize('kind', ['shape', 'direction', 'nan', 'integer'])
def test_bad_logits(kind):
    p, q, direction = torch.ones(2, 3), torch.ones(2, 3), 'forward'
    if kind == 'shape': q = q[:, :2]
    if kind == 'direction': direction = 'jsd'
    if kind == 'nan': p[0, 0] = float('nan')
    if kind == 'integer': p = p.long()
    with pytest.raises(ValueError): token_kl(p, q, direction)


def test_bank_restores_on_exception_and_prevents_live_graph_swap():
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    bank = TeacherBank(model)
    original = tensor_digest(model.named_parameters())
    with torch.no_grad(): model.weight.add_(1.)
    changed = tensor_digest(model.named_parameters())
    with pytest.raises(RuntimeError):
        with bank.use('initial'):
            assert tensor_digest(model.named_parameters()) == original
            raise RuntimeError('interrupted')
    assert tensor_digest(model.named_parameters()) == changed
    with bank.student_graph():
        with pytest.raises(RuntimeError):
            with bank.use(): pass
        with pytest.raises(RuntimeError): bank.update(.1)
    bank.reset()
    assert tensor_digest(model.named_parameters()) == original


def test_banked_teacher_matches_independent_teacher():
    model = torch.nn.Linear(3, 2)
    independent = copy.deepcopy(model)
    bank = TeacherBank(model)
    with torch.no_grad():
        for p in model.parameters(): p.add_(.3)
    bank.update(.25)
    with torch.no_grad():
        for t, s in zip(independent.parameters(), model.parameters()): t.mul_(.75).add_(s, alpha=.25)
    x = torch.randn(4, 3)
    with bank.use(): output = model(x)
    torch.testing.assert_close(output, independent(x))
    assert not output.requires_grad


def test_sft_matches_cross_entropy():
    z = torch.randn(1, 4, 8, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4]])
    expected = torch.nn.functional.cross_entropy(z.view(-1, 8), labels.view(-1))
    torch.testing.assert_close(reference_nll(z, labels), expected)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_analytic_kl_double_backward_and_exact_stationary_point(direction):
    torch.manual_seed(8)
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(x)
    assert torch.autograd.gradgradcheck(lambda z: token_kl(z, teacher, direction), (x,))
    loss = token_kl(x, x.detach().clone(), direction).sum()
    derivative, = torch.autograd.grad(loss, x)
    assert torch.equal(derivative, torch.zeros_like(derivative))
