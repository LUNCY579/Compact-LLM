"""Independent equation-based implementation of arXiv:2607.10034v1.

Algorithm 1 and Appendix B.2.5; see PAPER_NOTES.md for deliberate distinctions
between mean-loss ridge, paper-branch ridge, and the upstream score objective.
All construction is detached, bounded, and CPU float64. Deployed weights are
buffers: no optimizer or checkpoint mutation is introduced.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BuildConfig:
    width: int = 64
    seed: int = 17
    variant: str = "whitened"
    ridge: float = 1e-6
    ridge_convention: str = "mean_loss"
    projection_objective: str = "vector"
    projection_ridge: float = 1e-6
    cg_tolerance: float = 1e-8
    cg_max_iterations: int = 4096
    max_elements: int = 8_000_000

    def __post_init__(self) -> None:
        for name in ("width", "cg_max_iterations", "max_elements"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if self.variant not in {"unwhitened", "diagonal", "whitened", "data_dependent"}:
            raise ValueError("unknown construction variant")
        if self.ridge_convention not in {"mean_loss", "paper_branch"}:
            raise ValueError("unknown ridge convention")
        if self.projection_objective not in {"vector", "upstream_score"}:
            raise ValueError("unknown projection objective")
        for name in ("ridge", "projection_ridge", "cg_tolerance"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.cg_tolerance == 0:
            raise ValueError("cg_tolerance must be positive")


def matrix(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point numbers")
    return value.detach().to(device="cpu", dtype=torch.float64).clone()


def budget(limit: int, *sizes: int) -> None:
    """Preallocation bound on major tensors, not a hard process RSS limit."""
    if any(size > limit for size in sizes):
        raise MemoryError(f"requested tensor exceeds {limit} elements")


def ridge_readout(phi: Tensor, targets: Tensor, ridge: float,
                  convention: str = "mean_loss", route: str = "auto",
                  max_elements: int = 8_000_000) -> tuple[Tensor, dict]:
    phi, targets = matrix(phi, "features"), matrix(targets, "targets")
    n, m = phi.shape
    if len(targets) != n or not math.isfinite(ridge) or ridge < 0:
        raise ValueError("invalid targets or ridge")
    if convention not in {"mean_loss", "paper_branch"}:
        raise ValueError("unknown ridge convention")
    if route not in {"auto", "primal", "dual"}:
        raise ValueError("unknown solve route")
    route = ("primal" if m <= n else "dual") if route == "auto" else route
    size = m if route == "primal" else n
    budget(max_elements, size * size, m * targets.shape[1], phi.numel())
    # mean_loss means ||Phi W-Y||^2/n + lambda ||W||^2 on BOTH routes.
    regularizer = n * ridge if convention == "mean_loss" or route == "primal" else ridge
    if ridge == 0:
        # Minimum-norm solution, NOT a claim of invertible-covariance equivalence.
        u, s, vh = torch.linalg.svd(phi, full_matrices=False)
        tol = max(phi.shape) * torch.finfo(phi.dtype).eps * s.max()
        inv = torch.where(s > tol, s.reciprocal(), torch.zeros_like(s))
        weights = ((vh.T * inv) @ (u.T @ targets)).T
        rank = int((s > tol).sum())
        return weights, {"route": "svd_minimum_norm", "rank": rank,
                         "regularizer": 0.0, "singular_allowed": True}
    gram = phi.T @ phi if route == "primal" else phi @ phi.T
    gram.diagonal().add_(regularizer)
    rhs = phi.T @ targets if route == "primal" else targets
    solution = torch.linalg.solve(gram, rhs)
    residual = float(torch.linalg.norm(gram @ solution - rhs) /
                     torch.linalg.norm(rhs).clamp_min(1e-30))
    weights = solution.T if route == "primal" else solution.T @ phi
    if not torch.isfinite(weights).all() or residual > 1e-5:
        raise RuntimeError("ridge solve failed numerical qualification")
    return weights, {"route": route, "regularizer": regularizer,
                     "relative_solve_residual": residual, "convention": convention}



def construct_readout(phi: Tensor, targets: Tensor, config: BuildConfig) -> tuple[Tensor, dict]:
    """Preserve the named readout algorithm for both initial construction and edits."""
    if config.variant == "unwhitened":
        return targets.T @ phi / len(phi), {"route": "raw_hebbian_mean"}
    if config.variant == "diagonal":
        denom = phi.square().mean(0) + config.ridge
        if (denom == 0).any():
            raise RuntimeError("singular diagonal whitening")
        return (targets.T @ phi / len(phi)) / denom, {"route": "diagonal_mean_covariance", "ridge": config.ridge}
    return ridge_readout(phi, targets, config.ridge, config.ridge_convention,
                         max_elements=config.max_elements)


def conjugate_gradient(operator: Callable[[Tensor], Tensor], rhs: Tensor,
                       diagonal: Tensor, tolerance: float, max_iterations: int
                       ) -> tuple[Tensor, dict]:
    """Preconditioned CG; nonconvergence is an error, never a successful fit."""
    solution = torch.zeros_like(rhs)
    norm = float(torch.linalg.norm(rhs))
    if norm == 0:
        return solution, {"iterations": 0, "relative_residual": 0.0, "converged": True}
    residual = rhs.clone()
    inverse = diagonal.clamp_min(torch.finfo(rhs.dtype).eps).reciprocal()
    scaled = residual * inverse
    direction = scaled.clone()
    old_dot = (residual * scaled).sum()
    for iteration in range(1, max_iterations + 1):
        projected = operator(direction)
        denom = (direction * projected).sum()
        if not torch.isfinite(denom) or denom <= 0:
            raise RuntimeError("projection solve lost positive definiteness")
        step = old_dot / denom
        solution += step * direction
        residual -= step * projected
        if float(torch.linalg.norm(residual)) <= tolerance * norm:
            actual = float(torch.linalg.norm(rhs - operator(solution))) / norm
            if math.isfinite(actual) and actual <= max(tolerance * 10, 1e-10):
                return solution, {"iterations": iteration,
                                  "relative_residual": actual, "converged": True}
            residual = rhs - operator(solution)
        scaled = inverse * residual
        new_dot = (residual * scaled).sum()
        direction = scaled + (new_dot / old_dot) * direction
        old_dot = new_dot
    raise RuntimeError(f"projection solve did not converge in {max_iterations} iterations")


def fit_projection(keys: Tensor, targets: Tensor, partner: Tensor,
                   readout: Tensor, config: BuildConfig) -> tuple[Tensor, dict]:
    """Solve one linear factor subproblem without a Kronecker design matrix.

    vector: ||(K W^T * K P^T) B0^T - Y||_F^2 + rho ||W||_F^2.
    upstream_score: ||(K W^T * K P^T) B0^T Y^T - I||_F^2 + rho ||W||_F^2.
    The latter is the released code's objective, not paper Eqs. 17-18.
    """
    p = keys @ partner.T
    rhs = ((targets @ readout) * p).T @ keys
    budget(config.max_elements, targets.shape[1] ** 2, p.numel(), rhs.numel())
    value_gram = targets.T @ targets if config.projection_objective == "upstream_score" else None
    weighted_b = readout if value_gram is None else value_gram @ readout
    diagonal = (readout * weighted_b).sum(0)[:, None] * (p.square().T @ keys.square())
    diagonal = diagonal + config.projection_ridge

    def operator(candidate: Tensor) -> Tensor:
        decoded = ((keys @ candidate.T) * p) @ readout.T
        if value_gram is not None:
            decoded = decoded @ value_gram
        return ((decoded @ readout) * p).T @ keys + config.projection_ridge * candidate

    return conjugate_gradient(operator, rhs, diagonal,
                              config.cg_tolerance, config.cg_max_iterations)


class FrozenBilinear(nn.Module):
    """Native gated layout B[(Ax)*(Gx)], with identity rather than SiLU gate."""
    def __init__(self, a: Tensor, g: Tensor, b: Tensor) -> None:
        super().__init__()
        a, g, b = matrix(a, "A"), matrix(g, "G"), matrix(b, "B")
        if a.shape != g.shape or b.shape[1] != a.shape[0]:
            raise ValueError("incompatible bilinear weights")
        self.register_buffer("a", a)
        self.register_buffer("g", g)
        self.register_buffer("b", b)

    def features(self, x: Tensor) -> Tensor:
        return (x @ self.a.T) * (x @ self.g.T)

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x) @ self.b.T

    def storage(self) -> dict:
        return {"stored_scalars": sum(v.numel() for v in self.buffers()),
                "stored_bytes": sum(v.numel() * v.element_size() for v in self.buffers()),
                "optimizer_parameters": 0,
                "key_table_required_for_inference": False}


@torch.no_grad()
def build(keys: Tensor, targets: Tensor, config: BuildConfig = BuildConfig()
          ) -> tuple[FrozenBilinear, dict]:
    keys, targets = matrix(keys, "keys"), matrix(targets, "targets")
    n, d = keys.shape
    if len(targets) != n:
        raise ValueError("one target vector is required per key")
    if (keys.norm(dim=1) == 0).any():
        raise ValueError("zero keys cannot be stored by a bias-free bilinear MLP")
    m = config.width
    budget(config.max_elements, n*m, m*d, m*targets.shape[1], min(n, m)**2)
    rng = torch.Generator().manual_seed(config.seed)
    # Standard Gaussians / m**0.25 in each factor produce Phi/sqrt(m).
    # Data-dependent initialization follows the released unnormalized convention.
    scale = 1.0 if config.variant == "data_dependent" else m**-0.25
    a = torch.randn(m, d, generator=rng, dtype=torch.float64) * scale
    g = torch.randn(m, d, generator=rng, dtype=torch.float64) * scale
    receipt = {"config": asdict(config), "n": n, "input_dim": d,
               "output_dim": targets.shape[1], "native_qualified": False,
               "construction_optimizer_steps": 0}
    if config.variant == "data_dependent":
        b0 = targets.T @ ((keys @ a.T) * (keys @ g.T)) / n
        g, first = fit_projection(keys, targets, a, b0, config)
        a, second = fit_projection(keys, targets, g, b0, config)
        receipt["factor_solves"] = [first, second]
    phi = (keys @ a.T) * (keys @ g.T)
    b, solve = construct_readout(phi, targets, config)
    receipt["readout_solve"] = solve
    model = FrozenBilinear(a, g, b)
    receipt.update(model.storage())
    receipt["fit_mse"] = float((model(keys) - targets).square().mean())
    return model, receipt


@torch.no_grad()
def swap_readout(model: FrozenBilinear, keys: Tensor, targets: Tensor,
                 config: BuildConfig = BuildConfig()) -> tuple[FrozenBilinear, dict]:
    """Rebuild B on ALL retained/new facts; preserve A/G and the old module.

    This is a full-bank edit, not a claim of constant-time or local editing.
    Data-dependent features are kept fixed to isolate the readout-swap effect.
    """
    keys, targets = matrix(keys, "keys"), matrix(targets, "targets")
    if keys.shape[1] != model.a.shape[1] or len(keys) != len(targets):
        raise ValueError("invalid swap tensors")
    a, g = matrix(model.a, "A"), matrix(model.g, "G")
    phi = (keys @ a.T) * (keys @ g.T)
    b, receipt = construct_readout(phi, targets, config)
    return FrozenBilinear(a, g, b), receipt


def effective_kernel(phi: Tensor, config: BuildConfig) -> Tensor:
    """Small diagnostic only: H such that predictions = H @ targets."""
    n, m = phi.shape
    budget(config.max_elements, n*n, n*m)
    if config.variant == "unwhitened":
        return phi @ phi.T / n
    if config.variant == "diagonal":
        return (phi / (phi.square().mean(0) + config.ridge)) @ phi.T / n
    b, _ = ridge_readout(phi, torch.eye(n, dtype=phi.dtype), config.ridge,
                          config.ridge_convention, max_elements=config.max_elements)
    return phi @ b.T
