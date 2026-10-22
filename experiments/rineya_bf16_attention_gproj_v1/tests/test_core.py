import dataclasses

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.rineya_bf16_attention_gproj_v1.core import (
    Case, effective_mask, emulate, error_metrics, project_contracts, reference, witness,
)
from experiments.rineya_bf16_attention_gproj_v1.backends import dense_attention, native_vjp, precision_guard
from experiments.rineya_bf16_attention_gproj_v1.audit import audit_case, numerical_gate, synthetic_report


def random_case(heads=4, kv_heads=2, n=5, m=5, width=8, dtype=torch.float64, causal=False):
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(1, heads, n, width, generator=generator).to(dtype)
    k = torch.randn(1, kv_heads, m, width, generator=generator).to(dtype)
    v = torch.randn(1, kv_heads, m, 6, generator=generator).to(dtype)
    u = torch.randn(1, heads, n, 6, generator=generator).to(dtype)
    return Case(q, k, v, u, causal=causal)


@pytest.mark.parametrize("width", [2, 64, 128])
@pytest.mark.parametrize("offset", [0.0, 64.0, 1024.0, 65536.0])
def test_exact_cast_witness(width, offset):
    case = witness(width, offset)
    ref, raw, fixed = reference(case), emulate(case, "cast_only"), emulate(case)
    assert ref["dq"][0, 0, 0, 0] == 0
    assert raw["dq"][0, 0, 0, 0] == -offset * 3 / 8192
    assert fixed["dq"][0, 0, 0, 0] == 0
    assert torch.equal(raw["output"], fixed["output"])
    assert torch.equal(raw["dv"], fixed["dv"])


@pytest.mark.parametrize("heads,kv_heads", [(1, 1), (4, 2), (4, 1)])
@pytest.mark.parametrize("causal,n,m", [(False, 4, 4), (True, 4, 4), (True, 3, 5)])
def test_reference_matches_independent_autograd(heads, kv_heads, causal, n, m):
    case = random_case(heads, kv_heads, n, m, causal=causal)
    ref = reference(case)
    q, k, v = (x.clone().requires_grad_(True) for x in (case.q, case.k, case.v))
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                            scale=case.alpha, enable_gqa=heads != kv_heads)
    grads = torch.autograd.grad(out, (q, k, v), case.u)
    torch.testing.assert_close(out, ref["output"], rtol=1e-12, atol=1e-12)
    for key, actual in zip(("dq", "dk", "dv"), grads):
        torch.testing.assert_close(actual, ref[key], rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("kind", ["boolean", "additive", "all_masked", "causal_padding"])
def test_masks(kind):
    case = random_case(dtype=torch.float64)
    mask = torch.ones(5, 5, dtype=torch.bool)
    mask[:, -2:] = False
    if kind == "all_masked":
        mask[1] = False
    if kind == "causal_padding":
        case.causal = True
    case.mask = torch.zeros(5, 5).double().masked_fill(~mask, -torch.inf) if kind == "additive" else mask
    ref = reference(case)
    q, k, v = (x.clone().requires_grad_() for x in (case.q, case.k, case.v))
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=effective_mask(case, torch.float64), enable_gqa=True)
    actual = torch.autograd.grad(out, (q, k, v), case.u)
    for name, grad in zip(("dq", "dk", "dv"), actual):
        torch.testing.assert_close(grad, ref[name], rtol=1e-11, atol=1e-12)
    if kind == "all_masked":
        assert not ref["output"][..., 1, :].any()
        case = dataclasses.replace(case, **{k: getattr(case, k).bfloat16() for k in ("q", "k", "v", "u")})
        fixed = emulate(case)
        assert not fixed["dq"][..., 1, :].any()
        assert torch.isfinite(fixed["lambda"]).all()


def test_actual_probability_mass_required_and_both_contractions():
    r = torch.tensor([[[[0.2001953125, 0.2001953125, 0.59765625]]]], dtype=torch.float64)
    t = torch.tensor([[[[1.0, -0.5, -0.499]]]], dtype=torch.float64)
    q = torch.tensor([[[[2.0, 3.0]]]], dtype=torch.float64)
    k = torch.tensor([[[[65536.0, 1.0], [65536.0, 2.0], [65536.0, 4.0]]]], dtype=torch.float64)
    result = project_contracts(t, r, q, k, 0.125)
    projected = t - t.sum(-1, keepdim=True) / r.sum(-1, keepdim=True) * r
    torch.testing.assert_close(result["dq"], .125 * projected @ k, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(result["dk"], .125 * projected.transpose(-1, -2) @ q)
    assert r.sum() != 1
    assert (t - t.sum(-1, keepdim=True) * r).sum().abs() > 1e-7
    assert result["projected_mass"].abs().max() < 1e-15


def test_gproj_q_does_not_correct_key_gradient():
    case = random_case(dtype=torch.bfloat16)
    raw, q_only, full = (emulate(case, mode) for mode in ("cast_only", "gproj_q", "gproj"))
    assert torch.equal(raw["dk"], q_only["dk"])
    assert torch.equal(q_only["dq"], full["dq"])
    assert not torch.equal(raw["dk"], full["dk"])
    assert torch.equal(raw["dv"], full["dv"])


def test_saved_output_channel_is_separate():
    case = random_case(dtype=torch.bfloat16)
    correct = emulate(case)
    corrupt = dataclasses.replace(case, saved_output=correct["output"] + 0.5)
    assert not torch.equal(emulate(corrupt, "cast_only")["dq"], emulate(case, "cast_only")["dq"])
    torch.testing.assert_close(emulate(corrupt, center="exact")["dq"], emulate(case, center="exact")["dq"])


def test_zero_reference_metric_never_reports_false_relative_accuracy():
    metric = error_metrics(torch.ones(2), torch.zeros(2))
    assert metric["relative_l2"] is None and metric["cosine"] is None
    assert metric["absolute_l2"] > 0 and metric["near_zero_reference"]
    assert not error_metrics(torch.tensor([float("nan")]), torch.zeros(1))["finite"]


@pytest.mark.parametrize("invalid", ["empty", "heads", "mask", "mask_nan", "q_nan", "scale", "u_shape", "dtype"])
def test_invalid_cases_fail_closed(invalid):
    c = random_case()
    if invalid == "empty": c.q = c.q[..., :0, :]
    if invalid == "heads": c.k = c.k.repeat(1, 3, 1, 1)
    if invalid == "mask": c.mask = torch.ones(3, 7, dtype=torch.bool)
    if invalid == "mask_nan": c.mask = torch.full((5, 5), float("nan"))
    if invalid == "q_nan": c.q[0, 0, 0, 0] = float("nan")
    if invalid == "scale": c.scale = -1
    if invalid == "u_shape": c.u = c.u[..., :2]
    if invalid == "dtype": c.q = c.q.float()
    with pytest.raises(ValueError): c.validate()


def test_dense_budget_precedes_quadratic_allocation():
    with pytest.raises(ValueError, match="budget"):
        random_case().validate(max_score_elements=1)


@pytest.mark.parametrize("mode", ["cast_only", "gproj_q", "gproj"])
def test_custom_backward_matches_explicit_algebra(mode):
    case = random_case(dtype=torch.bfloat16, causal=True)
    q, k, v = (x.clone().requires_grad_() for x in (case.q, case.k, case.v))
    out = dense_attention(q, k, v, causal=True, mode=mode)
    actual = torch.autograd.grad(out, (q, k, v), case.u)
    expected = emulate(case, mode)
    for name, grad in zip(("dq", "dk", "dv"), actual):
        torch.testing.assert_close(grad, expected[name].to(grad.dtype), rtol=0, atol=0)


def test_dense_forward_is_identical_across_all_modes():
    case = random_case(dtype=torch.bfloat16)
    outputs = [dense_attention(case.q, case.k, case.v, mode=mode) for mode in ("cast_only", "gproj_q", "gproj")]
    assert all(torch.equal(outputs[0], out) for out in outputs)


def test_native_math_and_backend_unavailability_are_distinct():
    case = witness()
    assert native_vjp(case)["status"] == "MEASURED"
    assert native_vjp(case, "flash")["status"] == "UNAVAILABLE"
    with pytest.raises(ValueError): native_vjp(case, "auto")


def test_runtime_flags_are_restored_on_error():
    before = torch.get_float32_matmul_precision()
    with pytest.raises(RuntimeError), precision_guard():
        raise RuntimeError("deliberate")
    assert torch.get_float32_matmul_precision() == before


def test_report_separates_synthetic_from_native_qualification():
    report = synthetic_report()
    assert not report["native_qualified"]
    assert len(report["witness_sweep"]) == 12
    assert not any(report["claims"].values())
    assert not numerical_gate([], .05, 1e-6)
    audit = audit_case(witness())
    assert audit["variants"]["dense_gproj"]["status"] == "DENSE_EMULATION"


@pytest.mark.parametrize("alpha", [.03125, .125, 1.0])
def test_scale_applied_once_in_both_gradient_contractions(alpha):
    case = random_case(dtype=torch.bfloat16)
    case.scale = alpha
    result = emulate(case, round_key_operand=False)
    t, r = result["t"], result["r"]
    z = t - t.sum(-1, keepdim=True) / r.sum(-1, keepdim=True) * r
    k = case.k.float().repeat_interleave(2, 1)
    torch.testing.assert_close(result["dq"], alpha * z @ k, rtol=3e-5, atol=2e-6)
    expected_k = alpha * z.transpose(-1, -2) @ case.q.float()
    expected_k = expected_k.reshape(1, 2, 2, 5, 8).sum(2)
    torch.testing.assert_close(result["dk"], expected_k, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("sharpness", [1.0, 8.0, 32.0, 128.0])
def test_sharp_attention_remains_finite_and_reports_zero_cases(sharpness):
    case = random_case(dtype=torch.bfloat16)
    case.q *= sharpness
    for mode in ("cast_only", "gproj"):
        result = emulate(case, mode)
        assert all(torch.isfinite(result[k]).all() for k in ("dq", "dk", "dv", "lambda"))
    ref = reference(case)
    assert torch.isfinite(ref["dq"]).all()


def test_fixed_operand_translation_invariance_and_zero_mass():
    case = random_case()
    result = emulate(case, accumulation_dtype=torch.float64, round_key_operand=False)
    t, r = result["t"], result["r"]
    k = case.k.repeat_interleave(2, 1)
    base = project_contracts(t, r, case.q, k, case.alpha)
    shift = torch.zeros_like(k)
    shift[..., 0] = 65536.0
    translated = project_contracts(t, r, case.q, k + shift, case.alpha)
    torch.testing.assert_close(base["dq"], translated["dq"], rtol=1e-9, atol=1e-10)
    zero = project_contracts(torch.zeros_like(t), torch.zeros_like(r), case.q, k, case.alpha)
    assert not zero["dq"].any() and not zero["dk"].any()
    assert torch.isfinite(zero["lambda"]).all()
