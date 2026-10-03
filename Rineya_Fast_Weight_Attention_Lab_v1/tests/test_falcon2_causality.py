"""Unit tests for prefix-aligned FALCON-2 causality and the 12 P0 invariants."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rineya_fwa.falcon2 import FWAConfig, FastWeightBlock, falcon2_scan
from rineya_fwa.wrapper import FWAAttentionWrapper, token_valid_mask


def _inputs(batch=2, length=8, heads=2, rank=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, length, heads, rank, generator=g)
    k = torch.randn(batch, length, heads, rank, generator=g)
    v = torch.randn(batch, length, heads, rank, generator=g)
    valid = torch.ones(batch, length, dtype=torch.bool)
    beta = torch.full((heads, rank), 0.8)
    lam = torch.full((heads, rank), 0.1)
    return q, k, v, valid, beta, lam


def test_causal_state_update_future_tokens_do_not_change_past():
    q, k, v, valid, beta, lam = _inputs()
    mem, _ = falcon2_scan(q, k, v, valid, beta, lam, 1e-6, recurrent=True)
    k2 = k.clone()
    v2 = v.clone()
    k2[:, 5:] += 3.0
    v2[:, 6:] -= 2.0
    mem2, _ = falcon2_scan(q, k2, v2, valid, beta, lam, 1e-6, recurrent=True)
    torch.testing.assert_close(mem[:, :5], mem2[:, :5], rtol=1e-5, atol=1e-5)


def test_no_future_token_leakage_in_gradients():
    q, k, v, valid, beta, lam = _inputs(batch=1, length=6)
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    mem, _ = falcon2_scan(q, k, v, valid, beta, lam, 1e-6, recurrent=True)
    mem[:, 2].sum().backward()
    assert torch.count_nonzero(q.grad[:, 3:]) == 0
    assert torch.count_nonzero(k.grad[:, 3:]) == 0
    assert torch.count_nonzero(v.grad[:, 3:]) == 0


def test_deterministic_recurrence():
    args = _inputs(seed=3)
    a, sa = falcon2_scan(*args, 1e-6, recurrent=True)
    b, sb = falcon2_scan(*args, 1e-6, recurrent=True)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(sa, sb)


def test_state_reset_between_sequences():
    q, k, v, valid, beta, lam = _inputs(batch=1)
    _, s1 = falcon2_scan(q, k, v, valid, beta, lam, 1e-6, recurrent=True)
    _, s2 = falcon2_scan(q, k, v, valid, beta, lam, 1e-6, recurrent=True)
    torch.testing.assert_close(s1, s2)


def test_batch_isolation():
    q, k, v, valid, beta, lam = _inputs(batch=2, length=7)
    mem, state = falcon2_scan(q, k, v, valid, beta, lam, 1e-6, recurrent=True)
    mem0, state0 = falcon2_scan(q[:1], k[:1], v[:1], valid[:1], beta, lam, 1e-6, recurrent=True)
    torch.testing.assert_close(mem[:1], mem0)
    torch.testing.assert_close(state[:1], state0)
    assert not torch.allclose(mem[0], mem[1])


def test_sequence_boundary_sentinel_x0_is_zero_write():
    q, k, v, valid, beta, lam = _inputs(batch=1, length=1)
    mem, state = falcon2_scan(q, k, v, valid, beta, lam, 1e-6, recurrent=True)
    # t=1 uses x=0 so S stays 0 and m = 0
    torch.testing.assert_close(state, torch.zeros_like(state))
    torch.testing.assert_close(mem, torch.zeros_like(mem))


def test_padding_mask_does_not_update_state():
    q, k, v, valid, beta, lam = _inputs(batch=1, length=6)
    valid_pad = valid.clone()
    valid_pad[:, 4:] = False
    _, state_pad = falcon2_scan(q, k, v, valid_pad, beta, lam, 1e-6, recurrent=True)
    _, state_short = falcon2_scan(
        q[:, :4], k[:, :4], v[:, :4], valid[:, :4], beta, lam, 1e-6, recurrent=True
    )
    torch.testing.assert_close(state_pad, state_short)


def test_dtype_state_is_fp32():
    q, k, v, valid, beta, lam = _inputs()
    q = q.half()
    k = k.half()
    v = v.half()
    mem, state = falcon2_scan(q, k, v, valid, beta, lam, 1e-6, recurrent=True)
    assert state.dtype == torch.float32
    assert mem.dtype == torch.float32


def test_save_load_block():
    cfg = FWAConfig(hidden_size=16, n_kv_heads=2, rank=4)
    a = FastWeightBlock(cfg)
    b = FastWeightBlock(cfg)
    hidden = torch.randn(1, 5, 16)
    valid = torch.ones(1, 5, dtype=torch.bool)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        ya = a(hidden, valid)
        yb = b(hidden, valid)
    torch.testing.assert_close(ya, yb)


def test_long_sequence_numerical_stability():
    q, k, v, valid, beta, lam = _inputs(batch=1, length=256, rank=4)
    mem, state = falcon2_scan(q, k, v, valid, beta, lam, 1e-6, recurrent=True)
    assert torch.isfinite(mem).all()
    assert torch.isfinite(state).all()


def test_state_size_accounting():
    cfg = FWAConfig(hidden_size=32, n_kv_heads=8, rank=16, recurrent=True)
    block = FastWeightBlock(cfg)
    assert block.state_bytes(batch=1) == 8 * 16 * 16 * 4
    cfg_s = FWAConfig(hidden_size=32, n_kv_heads=8, rank=16, recurrent=False)
    static = FastWeightBlock(cfg_s)
    assert static.state_bytes() == 0


def test_alpha_zero_reproduces_inner_attention():
    inner = nn.Linear(8, 8, bias=False)
    cfg = FWAConfig(hidden_size=8, n_kv_heads=2, rank=4, init_alpha=0.0)
    block = FastWeightBlock(cfg)
    wrapper = FWAAttentionWrapper(inner, block)
    wrapper.eval()
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        baseline = inner(x)
        wrapped = wrapper(x)
    torch.testing.assert_close(wrapped, baseline)


def test_token_valid_mask_2d():
    hidden = torch.zeros(2, 4, 3)
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 0]])
    got = token_valid_mask(mask, hidden)
    assert got.tolist() == [[False, False, True, True], [True, True, True, False]]


def test_batch_isolation_and_batch_gt_one():
    """Regression: ``outputs[:, t] = mem * gate.squeeze(-1)``.

    The squeeze made the validity gate (B,1) against a (B,H,R) memory, so
    broadcasting aligned the gate's batch axis with the memory's HEAD axis:

    * B != H  -> RuntimeError;
    * B == H  -> silently applies row j's validity to head j.

    ``test_batch_isolation`` above uses batch=2, heads=2 and all-valid tokens, so
    B == H and the gate was all ones: it passed while the bug was present. Every
    P1/P2 run used batch 1, so nothing ever exercised it. This test uses B=3,
    H=4 so the shapes cannot coincide.
    """
    import torch

    from rineya_fwa.falcon2 import FastWeightBlock, FWAConfig

    torch.manual_seed(0)
    cfg = FWAConfig(hidden_size=32, n_kv_heads=4, rank=8, recurrent=True)
    block = FastWeightBlock(cfg)
    block.P.weight.data.normal_(0, 0.05)  # P is zero-init; make the output observable

    h = torch.randn(3, 12, cfg.hidden_size)
    valid = torch.ones(3, 12, dtype=torch.bool)

    batched = block(h, valid)
    assert batched.shape == (3, 12, cfg.hidden_size)

    for i in range(3):
        alone = block(h[i : i + 1], valid[i : i + 1])
        assert torch.allclose(batched[i : i + 1], alone, atol=1e-5), (
            f"row {i} differs when batched; rows are leaking into each other"
        )


def test_padding_is_masked_out():
    """Invalid tokens must neither write to the state nor emit memory."""
    import torch

    from rineya_fwa.falcon2 import FastWeightBlock, FWAConfig

    torch.manual_seed(1)
    cfg = FWAConfig(hidden_size=32, n_kv_heads=4, rank=8, recurrent=True)
    block = FastWeightBlock(cfg)
    block.P.weight.data.normal_(0, 0.05)

    h = torch.randn(2, 10, cfg.hidden_size)
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[:, 6:] = False  # last four are padding

    out = block(h, valid)
    assert torch.count_nonzero(out[:, 6:]) == 0, "padded positions must emit zero memory"

    # perturbing only the padded region must not change any real position
    h2 = h.clone()
    h2[:, 6:] = torch.randn_like(h2[:, 6:])
    out2 = block(h2, valid)
    assert torch.allclose(out[:, :6], out2[:, :6], atol=1e-6)


def test_padding_gate_is_applied_per_row_not_per_head():
    """The silent half of the squeeze bug: B == H with real padding.

    With B == H the broken broadcast does not raise, it applies row j's validity
    to head j. Constructing B == H == 4 and masking exactly one row makes that
    visible: only the masked row may go to zero, and only in its padded span.
    """
    import torch

    from rineya_fwa.falcon2 import falcon2_scan

    g = torch.Generator().manual_seed(3)
    B = H = 4
    T, R = 6, 5
    q = torch.randn(B, T, H, R, generator=g)
    k = torch.randn(B, T, H, R, generator=g)
    v = torch.randn(B, T, H, R, generator=g)
    beta = torch.full((H, R), 0.8)
    lam = torch.full((H, R), 0.1)

    valid = torch.ones(B, T, dtype=torch.bool)
    valid[1, 3:] = False  # only row 1 is padded, from t=3

    mem, _ = falcon2_scan(q, k, v, valid, beta, lam, 1e-6, recurrent=True)

    assert torch.count_nonzero(mem[1, 3:]) == 0, "padded span of row 1 must be zero"
    for b in range(B):
        if b == 1:
            assert torch.count_nonzero(mem[b, :3]) > 0
        else:
            assert torch.count_nonzero(mem[b]) > 0, (
                f"row {b} is fully valid but was zeroed; the gate is being applied per-head"
            )
