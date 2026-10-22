"""Algorithm-7 TAPS moments, with explicit Rineya safety/refresh adaptations."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import random
from typing import Sequence
import torch


@dataclass(frozen=True)
class TAPSConfig:
    enabled: bool = True
    beta: float = .8
    gamma: float = 1.5
    rho: float = .275
    alpha_min: float = .8
    alpha_max: float = 1.2
    alpha_init: float = 1.
    eps: float = 1e-12
    refresh_interval: int = 1
    warmup_loops: int = 1
    normalization_mode: str = "mean"
    scheduler_mode: str = "taps"
    fixed_alpha: float = 1.
    seed: int = 17
    random_pool: tuple[float, ...] = ()

    def __post_init__(self):
        for name in ("beta", "gamma", "rho", "alpha_min", "alpha_max", "alpha_init", "eps", "fixed_alpha"):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x):
                raise ValueError(f"{name} must be finite numeric")
        if type(self.enabled) is not bool or type(self.seed) is not int:
            raise ValueError("enabled/seed types")
        if not 0 <= self.beta < 1 or self.gamma <= 0 or self.rho < 0 or self.eps <= 0:
            raise ValueError("invalid beta/gamma/rho/eps")
        if not 0 < self.alpha_min <= 1 <= self.alpha_max or self.alpha_init != 1:
            raise ValueError("positive bounds must include unit step; alpha_init must be 1")
        if not self.alpha_min <= self.fixed_alpha <= self.alpha_max:
            raise ValueError("fixed alpha outside safe bounds")
        if type(self.refresh_interval) is not int or self.refresh_interval < 1:
            raise ValueError("refresh_interval must be positive integer")
        if type(self.warmup_loops) is not int or self.warmup_loops < 0:
            raise ValueError("warmup_loops must be nonnegative integer")
        if self.normalization_mode not in ("mean", "masked_mean"):
            raise ValueError("unknown normalization_mode")
        if self.scheduler_mode not in ("taps", "unit", "fixed", "random"):
            raise ValueError("unknown scheduler_mode")
        if not isinstance(self.random_pool, tuple) or any(
            isinstance(a, bool) or not isinstance(a, (int, float)) or
            not math.isfinite(a) or not self.alpha_min <= a <= self.alpha_max
            for a in self.random_pool
        ):
            raise ValueError("random_pool must be a finite bounded tuple")
        if self.scheduler_mode == "random" and not self.random_pool:
            raise ValueError("random mode needs a DEV-frozen empirical pool")


@dataclass
class TAPSState:
    step: int = 0
    mean: torch.Tensor | None = None
    energy: torch.Tensor | None = None
    previous: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    sequence_ids: tuple[str, ...] = ()


def _finite(*values):
    if any(not bool(torch.isfinite(x).all()) for x in values):
        raise FloatingPointError("nonfinite recurrent state/statistics")


def _cosine(a, b):
    if b is None:
        return torch.full((a.shape[0],), float("nan"), device=a.device)
    a, b = a.flatten(1), b.flatten(1)
    den = a.norm(dim=1) * b.norm(dim=1)
    return torch.where(den > 0, (a*b).sum(1)/den.clamp_min(1e-30), float("nan")).clamp(-1, 1)


class TAPSScheduler:
    """One controller per independent trajectory. Changed identities require reset.

    Moments observe every raw update. q>1 holds alpha, not the moment statistics.
    A diagnostic event contains detached per-example scalars, never hidden arrays.
    """
    def __init__(self, config: TAPSConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.state = TAPSState()
        self._rng: list[random.Random] = []

    @torch.inference_mode()
    def observe(self, delta: torch.Tensor, sequence_ids: Sequence[str], *, mask=None,
                force_unit: bool = False, alpha_override: float | None = None):
        c, s = self.config, self.state
        ids = tuple(sequence_ids)
        if delta.ndim < 2 or delta.shape[0] == 0 or delta.numel() == 0 or not delta.is_floating_point():
            raise ValueError("delta must be a nonempty floating [batch,...] tensor")
        if len(ids) != delta.shape[0] or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("unique nonempty sequence IDs required")
        d = delta.detach().float()
        _finite(d)
        if c.normalization_mode == "masked_mean":
            if mask is None or mask.dtype != torch.bool or tuple(mask.shape) != tuple(d.shape[:-1]):
                raise ValueError("masked_mean needs a Boolean [batch,...except channel] mask")
            if mask.device != d.device:
                raise ValueError("mask device mismatch")
            weights = mask.unsqueeze(-1).expand_as(d)
        else:
            if mask is not None:
                raise ValueError("mask supplied to mean mode")
            weights = torch.ones_like(d, dtype=torch.bool)
        count = weights.flatten(1).sum(1)
        if bool((count == 0).any()):
            raise ValueError("all-masked sequence")
        d = torch.where(weights, d, 0.)
        if s.step == 0:
            s.sequence_ids = ids
            s.mean = torch.zeros_like(d)
            s.energy = torch.zeros(d.shape[0], device=d.device)
            s.alpha = torch.ones_like(s.energy)
            s.mask = weights.clone()
            self._rng = [random.Random(int.from_bytes(hashlib.sha256(f"{c.seed}:{i}".encode()).digest()[:8], "big")) for i in ids]
        elif ids != s.sequence_ids or d.shape != s.mean.shape or d.device != s.mean.device or not torch.equal(weights, s.mask):
            raise ValueError("trajectory identity/shape/device/mask changed: reset required")
        mean_square = lambda x: x.square().flatten(1).sum(1)/count
        old_mean = s.mean / (1-c.beta**s.step) if s.step else None
        previous_cosine = _cosine(d, s.previous)
        previous_ema_cosine = _cosine(d, old_mean)
        s.mean = c.beta*s.mean + (1-c.beta)*d
        s.energy = c.beta*s.energy + (1-c.beta)*mean_square(d)
        k = s.step
        s.step += 1
        corrected = s.mean/(1-c.beta**s.step)
        progress = mean_square(corrected)
        fluctuation = (s.energy/(1-c.beta**s.step)-progress).clamp_min(0)
        balance = (progress-c.gamma*fluctuation)/(progress+c.gamma*fluctuation+c.eps)
        ratio = progress/(fluctuation+c.eps)
        _finite(s.mean, s.energy, progress, fluctuation, balance, ratio)
        refreshed = False
        if not c.enabled or c.scheduler_mode == "unit" or force_unit:
            alpha = torch.ones_like(progress)
        elif alpha_override is not None:
            if not math.isfinite(alpha_override) or not c.alpha_min <= alpha_override <= c.alpha_max:
                raise ValueError("override outside bounds")
            alpha = torch.full_like(progress, alpha_override)
        elif c.scheduler_mode == "fixed":
            alpha = torch.full_like(progress, c.fixed_alpha)
        elif c.scheduler_mode == "random":
            alpha = torch.tensor([r.choice(c.random_pool) for r in self._rng], device=d.device)
        elif k < c.warmup_loops:
            alpha = torch.ones_like(progress)
        elif (k-c.warmup_loops) % c.refresh_interval == 0:
            alpha = (1+c.rho*balance).clamp(c.alpha_min, c.alpha_max)
            refreshed = True
        else:
            alpha = s.alpha
        s.alpha = alpha
        s.previous = d.clone()
        return alpha, {
            "iteration": k, "delta_norm": d.flatten(1).norm(dim=1),
            "delta_cosine": previous_cosine, "previous_ema_cosine": previous_ema_cosine,
            "ema_cosine": _cosine(d, corrected), "progress": progress,
            "fluctuation": fluctuation, "progress_fluctuation_ratio": ratio,
            "balance": balance, "alpha": alpha, "refreshed": refreshed,
        }


def scalar_events(event: dict) -> list[dict]:
    n = event["alpha"].numel()
    rows = [{} for _ in range(n)]
    for key, value in event.items():
        values = value.detach().cpu().tolist() if torch.is_tensor(value) else [value]*n
        for row, x in zip(rows, values):
            row[key] = None if isinstance(x, float) and not math.isfinite(x) else x
    return rows


@torch.inference_mode()
def apply_update(old: torch.Tensor, proposal: torch.Tensor, alpha: torch.Tensor):
    if old.shape != proposal.shape or old.dtype != proposal.dtype or old.device != proposal.device:
        raise ValueError("proposal contract mismatch")
    if alpha.shape != (old.shape[0],) or alpha.device != old.device:
        raise ValueError("per-sequence alpha contract mismatch")
    _finite(old, proposal, alpha)
    if bool((alpha == 1).all()):
        return proposal  # exact baseline, avoiding h+(proposal-h) cancellation
    a = alpha.reshape((-1,) + (1,)*(old.ndim-1))
    result = (old.float()+a*(proposal.float()-old.float())).to(old.dtype)
    result = torch.where(a == 1, proposal, result)
    _finite(result)
    return result
