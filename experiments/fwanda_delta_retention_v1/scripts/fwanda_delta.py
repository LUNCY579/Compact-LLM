"""Deterministic, layerwise Fisher-aware pruning of fine-tuning deltas.

The implementation deliberately separates three signals:

* F0: the historical Rineya per-tensor magnitude top-k mask;
* F1: WANDA-style ``abs(delta) * ||X_j||_2`` with a uniform row budget;
* F2: the same WANDA score with the F-WANDA empirical-Fisher row budget.

Fisher is never multiplied into entries within a row.  It only changes the
number of entries allocated to each output row.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Iterator, Mapping, Sequence

import torch


EPSILON = 1.0e-8


def _finite_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return tensor


def _vector(value: torch.Tensor | Sequence[float], *, name: str) -> torch.Tensor:
    tensor = _finite_tensor(torch.as_tensor(value), name=name).detach()
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    return tensor.to(dtype=torch.float64, device="cpu")


def _validate_budget(total_keep: int, rows: int, row_width: int, min_keep: int) -> None:
    if not isinstance(total_keep, int):
        raise TypeError("total_keep must be an integer")
    if total_keep < rows * min_keep or total_keep > rows * row_width:
        raise ValueError(
            f"total_keep={total_keep} is incompatible with "
            f"rows={rows}, row_width={row_width}, min_keep={min_keep}"
        )


def allocate_row_budgets(
    row_importance: torch.Tensor | Sequence[float],
    total_keep: int,
    row_width: int,
    *,
    min_keep: int = 0,
    fisher_floor: float = EPSILON,
) -> torch.Tensor:
    """Allocate exactly ``total_keep`` entries across output rows.

    This is the paper's water-filling plus largest-remainder allocator.  The
    proportional quantity is ``sqrt(row_importance)``.  Bounds are applied
    before integer rounding; fractional ties are resolved by ascending row
    index.  A zero Fisher vector is therefore well-defined and uniform.
    """

    importance = _vector(row_importance, name="row_importance")
    rows = int(importance.numel())
    if row_width <= 0 or min_keep < 0 or min_keep > row_width:
        raise ValueError("invalid row width or minimum row budget")
    _validate_budget(total_keep, rows, row_width, min_keep)

    weights = torch.sqrt(torch.clamp(importance, min=fisher_floor))
    fixed: dict[int, int] = {}
    active = list(range(rows))
    remaining = total_keep

    while active:
        denominator = sum(float(weights[index]) for index in active)
        if denominator <= 0.0 or not math.isfinite(denominator):
            denominator = float(len(active))
            active_weights = {index: 1.0 for index in active}
        else:
            active_weights = {index: float(weights[index]) for index in active}
        proposed = {
            index: remaining * active_weights[index] / denominator for index in active
        }
        violations = [
            index
            for index in active
            if proposed[index] < min_keep or proposed[index] > row_width
        ]
        if not violations:
            break
        for index in sorted(violations):
            value = min_keep if proposed[index] < min_keep else row_width
            fixed[index] = value
            remaining -= value
            active.remove(index)
        if remaining < 0:
            raise RuntimeError("water-filling produced a negative residual budget")

    if not active:
        result = torch.tensor([fixed[index] for index in range(rows)], dtype=torch.long)
        if int(result.sum()) != total_keep:
            raise RuntimeError("bounded allocation failed to preserve total budget")
        return result

    denominator = sum(float(weights[index]) for index in active)
    if denominator <= 0.0 or not math.isfinite(denominator):
        denominator = float(len(active))
        active_weights = {index: 1.0 for index in active}
    else:
        active_weights = {index: float(weights[index]) for index in active}
    raw = {
        index: remaining * active_weights[index] / denominator for index in active
    }
    floors = {
        index: max(min_keep, min(row_width, math.floor(raw[index])))
        for index in active
    }
    remainder = remaining - sum(floors.values())
    if remainder < 0 or remainder > len(active):
        raise RuntimeError("largest-remainder rounding produced an invalid remainder")

    order = sorted(
        active,
        key=lambda index: (-(raw[index] - math.floor(raw[index])), index),
    )
    for index in order:
        if remainder == 0:
            break
        if floors[index] < row_width:
            floors[index] += 1
            remainder -= 1
    if remainder:
        raise RuntimeError("largest-remainder rounding could not spend the budget")

    fixed.update(floors)
    result = torch.tensor([fixed[index] for index in range(rows)], dtype=torch.long)
    if int(result.sum()) != total_keep:
        raise RuntimeError("allocation did not preserve the exact total budget")
    if int(result.min()) < min_keep or int(result.max()) > row_width:
        raise RuntimeError("allocation violated row bounds")
    return result


def uniform_row_budgets(
    rows: int, row_width: int, total_keep: int, *, min_keep: int = 0
) -> torch.Tensor:
    """Return a deterministic uniform allocation for the F1 control."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    return allocate_row_budgets(
        torch.ones(rows, dtype=torch.float64),
        total_keep,
        row_width,
        min_keep=min_keep,
    )


def activation_l2_norms(activations: torch.Tensor) -> torch.Tensor:
    """Compute ``||X_j||_2`` over every leading activation dimension."""

    x = _finite_tensor(activations, name="activations")
    if x.ndim < 2:
        raise ValueError("activations must have at least two dimensions")
    flat = x.detach().to(dtype=torch.float64).reshape(-1, x.shape[-1])
    norms = torch.sqrt(torch.sum(flat * flat, dim=0))
    return _finite_tensor(norms, name="activation norms")


def empirical_row_fisher(output_gradients: torch.Tensor) -> torch.Tensor:
    """Compute the empirical Fisher ``mean_t[(dL/dy_ti)^2]`` per output row."""

    gradients = _finite_tensor(output_gradients, name="output_gradients")
    if gradients.ndim < 2:
        raise ValueError("output_gradients must have at least two dimensions")
    flat = gradients.detach().to(dtype=torch.float64).reshape(-1, gradients.shape[-1])
    fisher = torch.mean(flat * flat, dim=0)
    return _finite_tensor(fisher, name="row Fisher")


def _stable_row_mask(scores: torch.Tensor, row_budgets: torch.Tensor) -> torch.Tensor:
    scores = _finite_tensor(scores, name="saliency scores")
    if scores.ndim != 2:
        raise ValueError("scores must be a two-dimensional matrix")
    rows, row_width = scores.shape
    budgets = torch.as_tensor(row_budgets, dtype=torch.long, device="cpu")
    if budgets.shape != (rows,):
        raise ValueError("row_budgets shape does not match score rows")
    if int(budgets.min()) < 0 or int(budgets.max()) > row_width:
        raise ValueError("row budget outside row bounds")

    mask = torch.zeros((rows, row_width), dtype=torch.bool, device=scores.device)
    for row in range(rows):
        keep = int(budgets[row])
        if keep == 0:
            continue
        # stable=True is the explicit flat-column tie rule: equal scores keep
        # lower column indices first.
        order = torch.argsort(scores[row], descending=True, stable=True)
        mask[row, order[:keep]] = True
    return mask


def _target_keep(delta: torch.Tensor, keep_fraction: float) -> int:
    if not 0.0 <= keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in [0, 1]")
    return int(math.floor(float(delta.numel()) * keep_fraction))


def mask_sha256(mask: torch.Tensor) -> str:
    """Hash shape, dtype, and row-major mask bytes."""

    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")
    payload = (
        str(tuple(mask.shape)).encode("ascii")
        + b"|bool|"
        + mask.detach().to(dtype=torch.uint8).contiguous().cpu().numpy().tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _summary(scores: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int | bool]:
    values = scores.detach().to(dtype=torch.float64).reshape(-1)
    kept = values[mask.reshape(-1)]
    if values.numel() == 0 or kept.numel() == 0:
        minimum = maximum = mean = kept_mean = 0.0
    else:
        minimum = float(values.min())
        maximum = float(values.max())
        mean = float(values.mean())
        kept_mean = float(kept.mean())
    return {
        "count": int(values.numel()),
        "kept_count": int(kept.numel()),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "kept_mean": kept_mean,
        "all_finite": bool(torch.isfinite(values).all()),
    }


@dataclass(frozen=True)
class LayerPruneResult:
    arm: str
    layer_name: str
    sparse_delta: torch.Tensor
    mask: torch.Tensor
    row_budgets: torch.Tensor
    row_fisher: torch.Tensor | None
    scores: torch.Tensor
    target_keep: int

    @property
    def density(self) -> float:
        return float(self.mask.to(dtype=torch.float64).mean())

    def record(self) -> dict[str, object]:
        fisher = None
        if self.row_fisher is not None:
            fisher = [float(value) for value in self.row_fisher.tolist()]
        return {
            "layer_name": self.layer_name,
            "arm": self.arm,
            "shape": list(self.mask.shape),
            "dtype": str(self.sparse_delta.dtype).replace("torch.", ""),
            "target_keep": self.target_keep,
            "kept": int(self.mask.sum()),
            "density": self.density,
            "row_budgets": [int(value) for value in self.row_budgets.tolist()],
            "row_fisher": fisher,
            "saliency": _summary(self.scores, self.mask),
            "mask_sha256": mask_sha256(self.mask),
        }


def prune_layer(
    delta: torch.Tensor,
    *,
    arm: str,
    keep_fraction: float,
    activation_norms: torch.Tensor | None = None,
    row_fisher: torch.Tensor | None = None,
    layer_name: str = "unnamed",
    min_keep: int = 1,
) -> LayerPruneResult:
    """Prune one two-dimensional delta tensor without mutating its input."""

    delta = _finite_tensor(delta, name="delta")
    if delta.ndim != 2:
        raise ValueError("delta pruning expects a two-dimensional linear-layer delta")
    rows, row_width = delta.shape
    target_keep = _target_keep(delta, keep_fraction)
    if arm == "F0":
        scores = delta.detach().to(dtype=torch.float32).abs()
        # Historical Rineya method: one flat top-k per trained tensor.
        flat_mask = _stable_row_mask(
            scores.reshape(1, -1), torch.tensor([target_keep], dtype=torch.long)
        )
        mask = flat_mask.reshape_as(delta)
        fisher = None
        row_budgets = mask.sum(dim=1).to(dtype=torch.long, device="cpu")
    elif arm in {"F1", "F2"}:
        if activation_norms is None:
            raise ValueError(f"{arm} requires activation_norms")
        norms = _vector(activation_norms, name="activation_norms")
        if norms.numel() != row_width:
            raise ValueError("activation_norms width does not match delta")
        scores = delta.detach().to(dtype=torch.float32).abs() * norms.to(
            dtype=torch.float32, device=delta.device
        ).reshape(1, -1)
        if arm == "F1":
            fisher = None
            row_budgets = uniform_row_budgets(
                rows, row_width, target_keep, min_keep=min_keep
            )
        else:
            if row_fisher is None:
                raise ValueError("F2 requires row_fisher")
            fisher = _vector(row_fisher, name="row_fisher")
            if fisher.numel() != rows:
                raise ValueError("row_fisher length does not match delta rows")
            row_budgets = allocate_row_budgets(
                fisher, target_keep, row_width, min_keep=min_keep
            )
        mask = _stable_row_mask(scores, row_budgets)
    else:
        raise ValueError(f"unknown arm: {arm}")

    sparse_delta = torch.where(mask, delta, torch.zeros((), dtype=delta.dtype, device=delta.device))
    if int(mask.sum()) != target_keep:
        raise RuntimeError("mask does not preserve the requested layer budget")
    if not bool(torch.isfinite(sparse_delta).all()):
        raise RuntimeError("masked delta contains NaN or Inf")
    return LayerPruneResult(
        arm=arm,
        layer_name=layer_name,
        sparse_delta=sparse_delta,
        mask=mask,
        row_budgets=row_budgets,
        row_fisher=fisher,
        scores=scores,
        target_keep=target_keep,
    )


def prune_layerwise(
    layers: Iterable[Mapping[str, object]],
    *,
    arm: str,
    keep_fraction: float,
    min_keep: int = 1,
) -> Iterator[LayerPruneResult]:
    """Process layers one at a time so masks and deltas need not be global."""

    for layer in layers:
        name = str(layer.get("name", "unnamed"))
        yield prune_layer(
            torch.as_tensor(layer["delta"]),
            arm=arm,
            keep_fraction=keep_fraction,
            activation_norms=(
                None
                if layer.get("activation_norms") is None
                else torch.as_tensor(layer["activation_norms"])
            ),
            row_fisher=(
                None
                if layer.get("row_fisher") is None
                else torch.as_tensor(layer["row_fisher"])
            ),
            layer_name=name,
            min_keep=min_keep,
        )


def apply_mask(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a boolean mask while preserving the delta dtype and shape."""

    if delta.shape != mask.shape:
        raise ValueError("delta and mask shapes differ")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")
    return torch.where(mask, delta, torch.zeros((), dtype=delta.dtype, device=delta.device))


def reconstruct(base: torch.Tensor, delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reconstruct ``base + masked_delta`` without mutating either source."""

    if base.shape != delta.shape or delta.shape != mask.shape:
        raise ValueError("base, delta, and mask shapes differ")
    sparse = apply_mask(delta, mask).to(dtype=base.dtype, device=base.device)
    return base + sparse


def apply_optional_prune(
    base: torch.Tensor,
    delta: torch.Tensor,
    mask: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    """Return an independent baseline copy when pruning is disabled."""

    if not enabled:
        return base.clone()
    return reconstruct(base, delta, mask)


class ActivationAccumulator:
    """Bounded-memory accumulator for activation column norms."""

    def __init__(self, width: int) -> None:
        if width <= 0:
            raise ValueError("width must be positive")
        self.sum_sq = torch.zeros(width, dtype=torch.float64)
        self.count = 0

    def update(self, activations: torch.Tensor) -> None:
        norms = _finite_tensor(activations, name="activations")
        if norms.ndim < 2 or norms.shape[-1] != self.sum_sq.numel():
            raise ValueError("activation batch width mismatch")
        flat = norms.detach().to(dtype=torch.float64).reshape(-1, norms.shape[-1])
        self.sum_sq += torch.sum(flat * flat, dim=0)
        self.count += int(flat.shape[0])

    def finalize(self) -> torch.Tensor:
        if self.count == 0:
            raise ValueError("cannot finalize an empty activation accumulator")
        return torch.sqrt(self.sum_sq)


class FisherAccumulator:
    """Bounded-memory accumulator for rowwise empirical Fisher statistics."""

    def __init__(self, rows: int) -> None:
        if rows <= 0:
            raise ValueError("rows must be positive")
        self.sum_sq = torch.zeros(rows, dtype=torch.float64)
        self.count = 0

    def update(self, output_gradients: torch.Tensor) -> None:
        gradients = _finite_tensor(output_gradients, name="output_gradients")
        if gradients.ndim < 2 or gradients.shape[-1] != self.sum_sq.numel():
            raise ValueError("gradient batch width mismatch")
        flat = gradients.detach().to(dtype=torch.float64).reshape(-1, gradients.shape[-1])
        self.sum_sq += torch.sum(flat * flat, dim=0)
        self.count += int(flat.shape[0])

    def finalize(self) -> torch.Tensor:
        if self.count == 0:
            raise ValueError("cannot finalize an empty Fisher accumulator")
        return self.sum_sq / self.count
