"""Arm D must differ from arm C by the minimum mechanism (brief section 9).

The P1/P2 gate replaced beta/lambda with absolute ranges, which (a) orphaned
beta_raw/lambda_raw from the graph -- in the saved P2 arm-D adapters they are
bit-identical to their init -- and (b) gave arm D a different retention horizon
from arm C, confounding the preregistered D-C contrast.
"""

from __future__ import annotations

import torch

from rineya_fwa.falcon2 import FastWeightBlock, FWAConfig, beta_from_raw, lambda_from_raw

BASE = dict(hidden_size=64, n_kv_heads=4, rank=8, beta_raw_init=-1.1, lambda_raw_init=-5.9)


def _pair() -> tuple[FastWeightBlock, FastWeightBlock]:
    torch.manual_seed(0)
    c = FastWeightBlock(FWAConfig(**BASE, recurrent=True, significance_gate=False))
    torch.manual_seed(0)
    d = FastWeightBlock(FWAConfig(**BASE, recurrent=True, significance_gate=True))
    # share every tensor the two arms have in common; only sig_gate is extra
    d.load_state_dict(c.state_dict(), strict=False)
    return c, d


def test_arm_d_reduces_to_arm_c_at_init():
    c, d = _pair()
    h = torch.randn(2, 16, BASE["hidden_size"])
    bc, lc, sc = c.rates(h)
    bd, ld, sd = d.rates(h)
    assert sc is None and sd is not None
    # zero-initialised gate => s = 0.5 exactly => u = 0 => no modulation
    assert torch.allclose(sd, torch.full_like(sd, 0.5))
    assert torch.allclose(bd, bc.expand_as(bd), atol=1e-6)
    assert torch.allclose(ld, lc.expand_as(ld), atol=1e-6)

    valid = torch.ones(2, 16, dtype=torch.bool)
    assert torch.allclose(c(h, valid), d(h, valid), atol=1e-5)


def test_significance_moves_beta_and_lambda_in_opposite_directions():
    _, d = _pair()
    h = torch.randn(1, 4, BASE["hidden_size"])
    beta0 = beta_from_raw(d.beta_raw)
    lam0 = lambda_from_raw(d.lambda_raw)
    span = d.sig_span

    with torch.no_grad():
        d.sig_gate.bias.fill_(20.0)  # s -> 1
    b_hi, l_hi, s_hi = d.rates(h)
    with torch.no_grad():
        d.sig_gate.bias.fill_(-20.0)  # s -> 0
    b_lo, l_lo, s_lo = d.rates(h)

    assert float(s_hi.mean()) > 0.99 and float(s_lo.mean()) < 0.01
    # high significance: stronger write, slower forgetting
    assert float(b_hi.mean()) > float(beta0.mean()) > float(b_lo.mean())
    assert float(l_hi.mean()) < float(lam0.mean()) < float(l_lo.mean())
    assert float(b_hi.mean()) == __import__("pytest").approx(
        float(beta0.mean()) * (1 + span), rel=1e-3
    )


def test_beta_and_lambda_still_receive_gradient_in_arm_d():
    _, d = _pair()
    h = torch.randn(1, 8, BASE["hidden_size"], requires_grad=False)
    valid = torch.ones(1, 8, dtype=torch.bool)
    with torch.no_grad():
        d.sig_gate.weight.normal_(0, 0.02)  # break the exact s=0.5 symmetry
    d.P.weight.data.normal_(0, 0.02)
    d(h, valid).pow(2).sum().backward()
    assert d.beta_raw.grad is not None and float(d.beta_raw.grad.abs().sum()) > 0
    assert d.lambda_raw.grad is not None and float(d.lambda_raw.grad.abs().sum()) > 0
    assert d.sig_gate.weight.grad is not None


def test_long_horizon_init_has_the_intended_retention():
    """The P2b init must actually give a horizon that spans a multi-K context."""
    cfg = FWAConfig(**BASE, recurrent=True, significance_gate=False)
    block = FastWeightBlock(cfg)
    beta = float(beta_from_raw(block.beta_raw).mean())
    lam = float(lambda_from_raw(block.lambda_raw).mean())
    x_sq = cfg.rank  # RMSNorm'd key with unit weights
    eta = beta / (x_sq + lam + cfg.eps)
    decay = 1.0 - lam * eta
    import math

    half_life = math.log(0.5) / math.log(decay)
    assert 0.45 < beta < 0.55, "beta must be well away from the 1.0 deadbeat point"
    assert half_life > 4000, f"half-life {half_life:.0f} tokens cannot span a 4K context"
    # retention along the written direction must not be a full replacement
    assert 0.4 < 1.0 - eta * x_sq < 0.6


def test_p1_p2_init_could_not_span_its_own_context():
    """Regression guard: the init the withdrawn P2 result was measured under."""
    import math

    cfg = FWAConfig(hidden_size=64, n_kv_heads=4, rank=16, beta_raw_init=0.0, lambda_raw_init=-2.0)
    block = FastWeightBlock(cfg)
    beta = float(beta_from_raw(block.beta_raw).mean())
    lam = float(lambda_from_raw(block.lambda_raw).mean())
    eta = beta / (cfg.rank + lam + cfg.eps)
    half_life = math.log(0.5) / math.log(1.0 - lam * eta)
    assert abs(half_life - 87.5) < 1.0, f"expected the measured 87.5-token half-life, got {half_life}"
    # beta = 1.0 is the deadbeat point: retention along the written direction,
    # decay - eta*||x||^2, collapses to zero, i.e. the write fully replaces it.
    decay = 1.0 - lam * eta
    assert abs(decay - eta * cfg.rank) < 1e-4, "expected the beta=1 deadbeat collapse"
