"""FP64 affine ridge with bounded streaming sufficient statistics, not n-by-n Gram matrices."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .contracts import Capture, capture_digest, disjoint, finite, paired, positive_int, require


@dataclass(frozen=True)
class AffineFit:
    weight: torch.Tensor  # PyTorch [out, in]; y = x @ weight.T + bias
    bias: torch.Tensor
    count: int
    ridge_absolute: float
    objective: str

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=self.weight.dtype) @ self.weight.T + self.bias

    def delta(self, change: torch.Tensor) -> torch.Tensor:
        """Push a residual displacement through the linear part ONLY."""
        return change.to(dtype=self.weight.dtype) @ self.weight.T


class RidgeMoments:
    """Merge-centered moments using the Chan correction; intercept is unpenalized.

    Objective: sum_i ||x_i W + b - y_i||^2 + lambda ||W||_F^2.
    All rows carry equal weight. Freeze the per-example token sampling policy first.
    Conservative workspace estimate covers solver copies, not total process RSS.
    """
    def __init__(self, input_dim: int, output_dim: int, *, max_workspace_bytes: int = 536870912):
        positive_int(input_dim, "input_dim")
        positive_int(output_dim, "output_dim")
        positive_int(max_workspace_bytes, "max_workspace_bytes")
        self.workspace_bound = 8 * (6 * input_dim**2 + 4 * input_dim * output_dim +
                                    4 * input_dim + 4 * output_dim)
        require(self.workspace_bound <= max_workspace_bytes, "ridge workspace exceeds budget")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.count = 0
        self.mx = torch.zeros(input_dim, dtype=torch.float64)
        self.my = torch.zeros(output_dim, dtype=torch.float64)
        self.xx = torch.zeros(input_dim, input_dim, dtype=torch.float64)
        self.xy = torch.zeros(input_dim, output_dim, dtype=torch.float64)

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        require(x.device.type == y.device.type == "cpu", "ridge fitting is CPU-only")
        require(x.ndim == y.ndim == 2 and x.shape[0] == y.shape[0] and x.shape[0] > 0,
                "bad sample matrices")
        require(x.shape[1] == self.input_dim and y.shape[1] == self.output_dim, "feature width mismatch")
        require(x.is_floating_point() and y.is_floating_point(), "expected floating tensors")
        require(not x.requires_grad and not y.requires_grad, "detach activations before fitting")
        require(torch.isfinite(x).all().item() and torch.isfinite(y).all().item(), "nonfinite fit input")
        x, y = x.double(), y.double()
        n = x.shape[0]
        bx, by = x.mean(0), y.mean(0)
        xc, yc = x - bx, y - by
        total = self.count + n
        dx, dy = bx - self.mx, by - self.my
        correction = self.count * n / total
        self.xx += xc.T @ xc + correction * torch.outer(dx, dx)
        self.xy += xc.T @ yc + correction * torch.outer(dx, dy)
        self.mx += dx * (n / total)
        self.my += dy * (n / total)
        self.count = total

    @torch.no_grad()
    def solve(self, *, ridge_relative: float = 1e-3, ridge_floor: float = 1e-8) -> AffineFit:
        finite(ridge_relative, "ridge_relative")
        finite(ridge_floor, "ridge_floor")
        require(ridge_relative > 0 and ridge_floor > 0, "regularization must be positive")
        require(self.count >= 2, "at least two fitting rows required")
        penalty = max(ridge_floor, ridge_relative * float(torch.trace(self.xx)) / self.input_dim)
        gram = (self.xx + self.xx.T) * 0.5
        gram.diagonal().add_(penalty)
        chol, info = torch.linalg.cholesky_ex(gram)
        require(int(info) == 0, "ridge Cholesky failed; do not silently switch solvers")
        w = torch.cholesky_solve(self.xy, chol)
        b = self.my - self.mx @ w
        require(torch.isfinite(w).all().item() and torch.isfinite(b).all().item(), "nonfinite fit")
        return AffineFit(w.T.contiguous(), b, self.count, penalty, "centered_affine_ridge_sum_v1")


def metrics(actual: torch.Tensor, predicted: torch.Tensor) -> dict:
    require(actual.shape == predicted.shape and actual.ndim == 2 and actual.numel() > 0,
            "metric shapes differ or empty")
    a, p = actual.double(), predicted.double()
    require(torch.isfinite(a).all().item() and torch.isfinite(p).all().item(), "nonfinite metric input")
    sse = float((a - p).square().sum())
    sst = float((a - a.mean(0)).square().sum())
    energy = float(a.square().sum())
    nonzero = (a.norm(dim=1) > 0) & (p.norm(dim=1) > 0)
    cos = torch.nn.functional.cosine_similarity(a[nonzero], p[nonzero], dim=1)
    return {"mse": sse / a.numel(), "r2": None if sst == 0 else 1 - sse / sst,
            "relative_rmse": None if energy == 0 else (sse / energy)**0.5,
            "mean_cosine": None if not nonzero.any() else float(cos.mean()),
            "undefined_cosine_rows": int((~nonzero).sum())}


def fit_bridge(source: Capture, target: Capture, check_source: Capture, check_target: Capture,
               *, chunk_rows: int = 256, ridge_relative: float = 1e-3,
               max_workspace_bytes: int = 536870912) -> tuple[AffineFit, dict]:
    paired(source, target)
    paired(check_source, check_target)
    disjoint((source, target, check_source, check_target))
    require(source.role == "FIT" and check_source.role == "MAP_CHECK", "FIT/MAP_CHECK roles required")
    for train, check in ((source, check_source), (target, check_target)):
        require(train.identity == check.identity and train.site == check.site, "fit/check identity or site drift")
    positive_int(chunk_rows, "chunk_rows")
    state = RidgeMoments(source.site.width, target.site.width, max_workspace_bytes=max_workspace_bytes)
    for start in range(0, len(source.rows), chunk_rows):
        state.update(source.values[start:start + chunk_rows], target.values[start:start + chunk_rows])
    fit = state.solve(ridge_relative=ridge_relative)
    return fit, {"fit": metrics(target.values, fit(source.values)),
                 "map_check": metrics(check_target.values, fit(check_source.values)),
                 "fit_count": fit.count, "ridge_absolute": fit.ridge_absolute,
                 "workspace_bound_bytes": state.workspace_bound,
                 "data_digests": {"source_fit": capture_digest(source), "target_fit": capture_digest(target),
                                  "source_map_check": capture_digest(check_source), "target_map_check": capture_digest(check_target)},
                 "functional_transfer_established": False}
