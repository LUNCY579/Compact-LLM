"""Strict decoding margins, geometric interference, and paired retention."""
from __future__ import annotations

import math
import torch
from torch import Tensor
from .memory import matrix, budget


def unit(x: Tensor) -> Tensor:
    norms = x.norm(dim=-1, keepdim=True)
    if (norms == 0).any() or not torch.isfinite(x).all():
        raise ValueError("normalization requires finite nonzero vectors")
    return x / norms


def recall(outputs: Tensor, candidates: Tensor, labels: Tensor) -> dict:
    outputs, candidates = matrix(outputs, "outputs"), matrix(candidates, "candidates")
    labels = labels.detach().cpu()
    if outputs.shape[1] != candidates.shape[1] or len(candidates) < 2:
        raise ValueError("invalid decoding candidates")
    if labels.dtype != torch.int64 or labels.shape != (len(outputs),):
        raise ValueError("labels must be one int64 index per query")
    if (labels < 0).any() or (labels >= len(candidates)).any():
        raise ValueError("candidate index out of bounds")
    scores = outputs @ candidates.T
    true = scores.gather(1, labels[:, None]).squeeze(1)
    competing = scores.clone()
    competing.scatter_(1, labels[:, None], -torch.inf)
    margins = true - competing.max(1).values
    # Raw argmax favors low indices on ties; the strict metric does not.
    normalized = margins / outputs.norm(dim=1).clamp_min(1e-30)
    return {"accuracy": float((scores.argmax(1) == labels).double().mean()),
            "strict_accuracy": float((margins > 0).double().mean()),
            "minimum_margin": float(margins.min()),
            "margin_p05": float(torch.quantile(margins, 0.05)),
            "median_margin": float(margins.median()),
            "minimum_output_normalized_margin": float(normalized.min()),
            "per_row_margin": margins.tolist(),
            "predictions": scores.argmax(1).tolist()}


def geometry(kernel: Tensor, candidates: Tensor, labels: Tensor,
             max_elements: int = 8_000_000) -> dict:
    """Generalized Eq. 6; handles many-to-one maps. O(F^2 V) diagnostic.

    Uses the effective (including readout scaling) kernel, not an unnormalized
    random-feature Gram matrix. No claim of fitting the asymptotic theorem.
    """
    h, v = matrix(kernel, "kernel"), matrix(candidates, "candidates")
    n = len(h)
    if h.shape != (n, n) or labels.dtype != torch.int64 or labels.shape != (n,):
        raise ValueError("invalid kernel or labels")
    if (labels < 0).any() or (labels >= len(v)).any() or len(v) < 2:
        raise ValueError("invalid labels or candidates")
    budget(max_elements, n*n, n*len(v), n*n*len(v))
    labels = labels.cpu()
    targets = v[labels]
    ek = ev = coupling = max_cross = 0.0
    min_signal = min_margin = math.inf
    for i in range(n):
        competitors = torch.arange(len(v)) != labels[i]
        differences = v[labels[i]] - v[competitors]
        overlaps = differences @ targets.T
        signal = overlaps[:, i] * h[i, i]
        pattern = h[i].clone()
        pattern[i] = 0
        overlaps[:, i] = 0
        cross = overlaps @ pattern
        key_energy = pattern.square().sum()
        value_energy = overlaps.square().sum(1)
        denom = (key_energy * value_energy).sqrt()
        kappas = torch.where(denom > 0, cross.abs()/denom.clamp_min(1e-300), 0)
        ek = max(ek, float(key_energy))
        ev = max(ev, float(value_energy.max()))
        coupling = max(coupling, float(kappas.max()))
        max_cross = max(max_cross, float(cross.abs().max()))
        min_signal = min(min_signal, float(signal.min()))
        min_margin = min(min_margin, float((signal + cross).min()))
    singular = torch.linalg.svdvals(h)
    rank = int(torch.linalg.matrix_rank(h))
    energy = singular.square()
    effective_rank = float(energy.sum().square()/energy.square().sum().clamp_min(1e-300))
    bound = min_signal - math.sqrt(ek * ev) * coupling
    return {"key_offdiagonal_energy": ek, "value_difference_energy": ev,
            "coupling": coupling, "minimum_signal": min_signal,
            "maximum_absolute_crosstalk": max_cross,
            "decomposed_minimum_margin": min_margin,
            "cauchy_lower_bound": bound, "kernel_rank": rank,
            "kernel_energy_effective_rank": effective_rank}


def noisy_queries(keys: Tensor, epsilon: float, seed: int) -> Tensor:
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    keys = matrix(keys, "keys")
    noise = torch.randn(keys.shape, generator=torch.Generator().manual_seed(seed), dtype=keys.dtype)
    return keys + epsilon * keys.norm(dim=1, keepdim=True) * unit(noise)


def paired(base: list[bool], candidate: list[bool]) -> dict:
    if not base or len(base) != len(candidate):
        raise ValueError("paired outcomes must be nonempty and aligned")
    if any(type(v) is not bool for v in base + candidate):
        raise ValueError("outcomes must be booleans")
    repairs = sum(not b and c for b, c in zip(base, candidate))
    damages = sum(b and not c for b, c in zip(base, candidate))
    return {"n": len(base), "repairs": repairs, "damages": damages,
            "accuracy_delta": (repairs-damages)/len(base),
            "base_correct": sum(base), "candidate_correct": sum(candidate)}


def paired_bootstrap(base: list[bool], candidate: list[bool], seed: int = 71,
                     repetitions: int = 2000) -> dict:
    result = paired(base, candidate)
    if type(repetitions) is not int or repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    delta = torch.tensor(candidate, dtype=torch.float64) - torch.tensor(base, dtype=torch.float64)
    rng = torch.Generator().manual_seed(seed)
    # Stream one bootstrap sample at a time: no repetitions x corpus allocation.
    means = torch.stack([delta[torch.randint(len(delta), (len(delta),), generator=rng)].mean()
                         for _ in range(repetitions)])
    result["bootstrap_95_percent_interval"] = torch.quantile(means, torch.tensor([.025, .975], dtype=means.dtype)).tolist()
    result["bootstrap_unit"] = "paired_fact_id; aggregate templates first"
    return result
