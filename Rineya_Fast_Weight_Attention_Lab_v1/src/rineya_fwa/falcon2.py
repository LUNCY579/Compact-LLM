"""Prefix-aligned FALCON-2 recurrence (arXiv:2608.27763).

Paper (section A in PAPER_NOTES.md):
    (x_t, y_t) = (φ(k_{t-1}), v_t)
    r_t = v_t - S_{t-1}^T x_t
    η_{j,t} = β_{j,t} / (‖x_t‖² + λ_t + ε)
    S_t = S_{t-1} (I − λ_t diag(η_t)) + x_t (η_t ⊙ r_t)^T
    m_t = S_t^T φ(q_t)

Arm B uses the same algebra with no carry: S is zeroed every token and
x_t = φ(k_t) (same-step). That is an engineering control, not a FALCON variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def beta_from_raw(raw: torch.Tensor) -> torch.Tensor:
    """Map unconstrained parameters into (0, 2), the Falcon-2 descent interval."""
    return 2.0 * torch.sigmoid(raw)


def lambda_from_raw(raw: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(raw)


def falcon2_scan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid: torch.Tensor,
    beta: torch.Tensor,
    lam: torch.Tensor,
    eps: float,
    *,
    recurrent: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the sequential FALCON-2 scan.

    Args:
        query, key, value: (B, T, H, R) float32
        valid: (B, T) bool, True = real token
        beta, lam: (H, R) or (B, T, H, R)
        recurrent: if False, reset S every token and use same-step x_t = k_t

    Returns:
        memory: (B, T, H, R) float32
        final_state: (B, H, R, R) float32  (zeros when not recurrent)
    """
    if query.ndim != 4:
        raise ValueError(f"expected (B,T,H,R), got {tuple(query.shape)}")
    batch, length, n_heads, rank = query.shape
    device = query.device
    dtype = torch.float32
    q = query.float()
    k = key.float()
    v = value.float()
    valid_f = valid.to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1)

    state = torch.zeros(batch, n_heads, rank, rank, device=device, dtype=dtype)
    k_prev = torch.zeros(batch, n_heads, rank, device=device, dtype=dtype)
    outputs = q.new_zeros(batch, length, n_heads, rank)

    def _at(param: torch.Tensor, t: int) -> torch.Tensor:
        if param.ndim == 2:
            return param.to(device=device, dtype=dtype)
        if param.ndim == 4:
            return param[:, t].to(device=device, dtype=dtype)
        raise ValueError(f"beta/lambda rank must be 2 or 4, got {param.ndim}")

    for t in range(length):
        qt = q[:, t]
        kt = k[:, t]
        vt = v[:, t]
        gate = valid_f[:, t]  # (B,1,1)
        beta_t = _at(beta, t)
        lam_t = _at(lam, t)
        if recurrent:
            xt = k_prev
        else:
            xt = kt
            state = torch.zeros_like(state)

        pred = torch.einsum("bhrd,bhr->bhd", state, xt)
        residual = vt - pred
        denom = (xt * xt).sum(dim=-1, keepdim=True) + lam_t + eps
        eta = (beta_t / denom.clamp_min(eps)).clamp(0.0, 1.5)
        residual = residual.clamp(-50.0, 50.0)
        decay = (1.0 - lam_t * eta).clamp(0.0, 1.0)
        write = torch.einsum("bhr,bhd->bhrd", xt, eta * residual)
        new_state = (state * decay.unsqueeze(-2) + write).clamp(-1.0e4, 1.0e4)
        state = torch.where(gate.unsqueeze(-1).bool(), new_state, state)
        mem = torch.einsum("bhrd,bhr->bhd", state, qt)
        # gate is (B,1,1) and mem is (B,H,R); squeezing the gate to (B,1) makes
        # broadcasting align H against B and raises for any batch > 1. Every P1/P2
        # run used batch 1, so this never fired.
        outputs[:, t] = mem * gate
        if recurrent:
            k_prev = torch.where(gate.bool(), kt, k_prev)

    if not recurrent:
        state = torch.zeros_like(state)
    return outputs, state


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms * self.weight.float()).to(dtype=x.dtype)


@dataclass(frozen=True)
class FWAConfig:
    hidden_size: int = 1024
    n_kv_heads: int = 8
    rank: int = 16
    eps: float = 1e-6
    recurrent: bool = True
    significance_gate: bool = False
    init_alpha: float = 0.0
    # Plasticity / forgetting initialisation.
    #
    # These set the memory's retention horizon before any training. The P1/P2
    # runs used the defaults below, which give lambda = softplus(-2) = 0.1269
    # and beta = 2*sigmoid(0) = 1.0. With an RMSNorm'd rank-16 key, ||x||^2 = 16,
    # so eta = 0.0622, the ridge decay is 1 - lambda*eta = 0.99211 per token
    # (half-life 87.5 tokens) and beta = 1.0 is the NLMS deadbeat point, which
    # fully replaces the stored value along the current key direction. Measured,
    # not assumed -- see reports/MECHANISM_DIAGNOSTIC.md.
    #
    # A horizon of ~87 tokens cannot carry a fact across a 1K-4K context, so P2b
    # makes these configurable and matches them across arms B/C/D.
    beta_raw_init: float = 0.0
    lambda_raw_init: float = -2.0
    # Arm D only: how far significance may push beta/lambda around their base
    # values, as a fraction. 0.9 => beta in [0.1*beta0, 1.9*beta0].
    significance_span: float = 0.9


class FastWeightBlock(nn.Module):
    """Projections + FALCON-2 scan + merge residual (without adding to h).

    ``forward`` returns the projected memory in hidden space, *before* α.
    """

    def __init__(self, cfg: FWAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        inner = cfg.n_kv_heads * cfg.rank
        self.W_q = nn.Linear(cfg.hidden_size, inner, bias=False)
        self.W_k = nn.Linear(cfg.hidden_size, inner, bias=False)
        self.W_v = nn.Linear(cfg.hidden_size, inner, bias=False)
        self.P = nn.Linear(inner, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.rank)
        self.k_norm = RMSNorm(cfg.rank)
        self.v_norm = RMSNorm(cfg.rank)
        self.beta_raw = nn.Parameter(
            torch.full((cfg.n_kv_heads, cfg.rank), float(cfg.beta_raw_init))
        )
        self.lambda_raw = nn.Parameter(
            torch.full((cfg.n_kv_heads, cfg.rank), float(cfg.lambda_raw_init))
        )
        self.alpha = nn.Parameter(torch.tensor(float(cfg.init_alpha)))
        if cfg.significance_gate:
            self.sig_gate = nn.Linear(cfg.hidden_size, 1)
            nn.init.zeros_(self.sig_gate.weight)
            nn.init.zeros_(self.sig_gate.bias)
            self.sig_span = cfg.significance_span
        else:
            self.sig_gate = None
        nn.init.zeros_(self.P.weight)

    def _features(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = hidden.shape
        heads, rank = self.cfg.n_kv_heads, self.cfg.rank
        q = self.q_norm(self.W_q(hidden).view(batch, length, heads, rank))
        k = self.k_norm(self.W_k(hidden).view(batch, length, heads, rank))
        v = self.v_norm(self.W_v(hidden).view(batch, length, heads, rank))
        return q, k, v

    def rates(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Plasticity beta and forgetting lambda, optionally significance-modulated.

        Arm D must differ from arm C by the *minimum* mechanism (brief section 9),
        so the gate is a symmetric multiplicative deviation around arm C's own
        trained rates rather than an independent range:

            u_t  = 2*s_t - 1          in (-1, 1)
            beta_t   = beta0   * (1 + span * u_t)
            lambda_t = lambda0 * (1 - span * u_t)

        High significance -> stronger write, slower forgetting. At s_t = 0.5 this
        is exactly arm C, and ``sig_gate`` is zero-initialised so s_t = 0.5
        identically at step 0: arm D *starts* as arm C and can only depart from
        it by learning. beta0/lambda0 stay in the graph, so beta_raw/lambda_raw
        keep receiving gradient in arm D (in P1/P2 they were orphaned and stayed
        bit-identical to their init).

        Earlier versions used hardcoded ranges (beta 0.05-1.9, lambda 0.01-0.5),
        which would have made arm D differ from arm C in the retention horizon as
        well as the gate, confounding the D-C contrast.
        """
        beta0 = beta_from_raw(self.beta_raw)
        lam0 = lambda_from_raw(self.lambda_raw)
        scores = None
        if self.sig_gate is None:
            return beta0, lam0, scores
        scores = torch.sigmoid(self.sig_gate(hidden.float())).squeeze(-1)  # (B, T)
        u = (2.0 * scores - 1.0).unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1)
        span = float(self.sig_span)
        beta = (beta0 * (1.0 + span * u)).clamp(1e-4, 1.9999)
        lam = (lam0 * (1.0 - span * u)).clamp_min(0.0)
        return beta, lam, scores

    def forward(
        self,
        hidden: torch.Tensor,
        valid: torch.Tensor,
        *,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self._features(hidden.float())
        beta, lam, _ = self.rates(hidden.float())
        memory, state = falcon2_scan(
            q, k, v, valid, beta, lam, self.cfg.eps, recurrent=self.cfg.recurrent
        )
        batch, length, heads, rank = memory.shape
        projected = self.P(memory.reshape(batch, length, heads * rank).float()).to(dtype=hidden.dtype)
        if return_state:
            return projected, state
        return projected

    def state_bytes(self, batch: int = 1) -> int:
        if not self.cfg.recurrent:
            return 0
        return batch * self.cfg.n_kv_heads * self.cfg.rank * self.cfg.rank * 4
