import pytest
import torch
from experiments.rineya_triadic_working_memory_v1.core import (
    MemoryConfig, delta_step, flattened_delta_step, read_state, normalize, shuffle_second_keys)


def inputs(e=4, batch=2, heads=2):
    torch.manual_seed(42)
    b, h, k, v = batch, heads, 3, 2
    state = torch.randn(b, h, e, k, v, dtype=torch.float64)
    key = normalize(torch.randn(b, h, k, dtype=torch.float64))
    second = normalize(torch.randn(b, h, e, dtype=torch.float64))
    value = torch.randn(b, h, v, dtype=torch.float64)
    alpha = torch.rand(b, h, e, dtype=torch.float64)
    beta = torch.rand(b, h, dtype=torch.float64)
    return state, key, second, value, alpha, beta


@pytest.mark.parametrize("e", [1, 2, 4, 8])
@pytest.mark.parametrize("batch", [1, 2])
def test_tensor_flattened_equivalence(e, batch):
    args = inputs(e, batch)
    torch.testing.assert_close(delta_step(*args), flattened_delta_step(*args), atol=1e-12, rtol=1e-12)


def test_all_input_gradients_match_independent_matrix_formula():
    a = tuple(x.clone().requires_grad_() for x in inputs())
    b = tuple(x.clone().requires_grad_() for x in inputs())
    delta_step(*a).square().sum().backward()
    flattened_delta_step(*b).square().sum().backward()
    for x, y in zip(a, b):
        torch.testing.assert_close(x.grad, y.grad, atol=1e-11, rtol=1e-11)


def test_double_precision_gradcheck():
    args = tuple(x.requires_grad_() for x in inputs(2, 1, 1))
    assert torch.autograd.gradcheck(delta_step, args, eps=1e-6, atol=1e-5)


def test_e1_matches_ordinary_delta_rule():
    state, k, k2, v, alpha, beta = inputs(1)
    k2.fill_(1)
    decayed = state[:, :, 0] * alpha[..., None]
    residual = v - torch.einsum("bhkv,bhk->bhv", decayed, k)
    expected = decayed + beta[..., None, None] * k[..., None] * residual[..., None, :]
    torch.testing.assert_close(delta_step(state, k, k2, v, alpha, beta)[:, :, 0], expected)


def test_joint_residual_is_not_independent_slice_residuals():
    state, k, k2, v, alpha, beta = inputs(4, 1, 1)
    alpha.fill_(1)
    beta.fill_(1)
    updated = delta_step(state, k, k2, v, alpha, beta)
    torch.testing.assert_close(read_state(updated, k, k2), v, atol=1e-12, rtol=1e-12)
    local_prediction = torch.einsum("bhedv,bhd->bhev", state, k)
    wrong_residual = v.unsqueeze(-2) - local_prediction
    wrong = state + torch.einsum("bhe,bhd,bhev->bhedv", k2, k, wrong_residual)
    assert not torch.allclose(wrong, updated)


def test_targeted_update_preserves_orthogonal_second_address():
    state = torch.zeros(1, 1, 2, 2, 1, dtype=torch.float64)
    k = torch.tensor([[[1., 0.]]], dtype=torch.float64)
    keys = torch.eye(2, dtype=torch.float64).reshape(2, 1, 1, 2)
    alpha, beta = torch.ones(1, 1, 2, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)
    def write(state, address, value):
        return delta_step(state, k, address, torch.tensor([[[value]]], dtype=torch.float64), alpha, beta)
    state = write(write(state, keys[0], 3.), keys[1], 7.)
    state = write(state, keys[0], 11.)
    assert read_state(state, k, keys[0]).item() == 11
    assert read_state(state, k, keys[1]).item() == 7


def test_decay_and_no_inplace_mutation():
    args = inputs()
    original = args[0].clone()
    beta = torch.zeros_like(args[-1])
    out = delta_step(*args[:-1], beta)
    torch.testing.assert_close(out, original * args[-2][..., None, None])
    assert torch.equal(args[0], original)


@pytest.mark.parametrize("field", ["heads", "slices", "conv_width", "max_state_bytes"])
def test_invalid_config(field):
    with pytest.raises(ValueError):
        MemoryConfig(**{field: 0})


def test_invalid_shapes_and_dtypes_rejected():
    args = list(inputs())
    args[-1] = args[-1][..., None]
    with pytest.raises(ValueError):
        delta_step(*args)
    args = list(inputs())
    args[2] = args[2].float()
    with pytest.raises(ValueError):
        delta_step(*args)


def test_long_repeated_updates_remain_finite():
    state, k, k2, v, alpha, beta = inputs(4)
    alpha.fill_(0.995)
    beta.fill_(0.8)
    for _ in range(300):
        state = delta_step(state, k, k2, v, alpha, beta)
    assert torch.isfinite(state).all()
    assert state.norm() < 100


def test_offline_shuffle_stays_inside_documents_and_batches():
    x = torch.arange(16.).reshape(2, 8, 1, 1)
    reset = torch.zeros(2, 8, dtype=torch.bool)
    reset[:, [0, 4]] = True
    y = shuffle_second_keys(x, reset, seed=42)
    for b in range(2):
        for start in (0, 4):
            assert sorted(x[b, start:start+4].flatten().tolist()) == sorted(y[b, start:start+4].flatten().tolist())
    assert not torch.equal(x, y)
