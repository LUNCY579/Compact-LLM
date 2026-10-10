from __future__ import annotations

from typing import Any

import torch

from .checkpoint import CheckpointError


def ridge_linear_map(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float = 1e-3,
) -> torch.Tensor:
    """Map source features to target features: target ≈ source @ W."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("linear maps require sample-by-feature matrices")
    if source.shape[0] != target.shape[0]:
        raise ValueError("sample counts differ")
    x = source.float()
    y = target.float()
    gram = x.transpose(0, 1) @ x
    identity = torch.eye(gram.shape[0], dtype=torch.float32)
    solved = torch.linalg.solve(gram + ridge * identity, x.transpose(0, 1) @ y)
    return solved


def project_weight(
    donor: torch.Tensor,
    input_map: torch.Tensor,
    output_map: torch.Tensor,
) -> torch.Tensor:
    """Project donor [out_d, in_d] into target [out_t, in_t] via feature maps."""
    if donor.ndim != 2:
        raise CheckpointError("direction transfer requires a matrix")
    if input_map.shape[0] != donor.shape[1]:
        raise CheckpointError("input map does not match donor in_features")
    if output_map.shape[0] != donor.shape[0]:
        raise CheckpointError("output map does not match donor out_features")
    return output_map.float().transpose(0, 1) @ donor.float() @ input_map.float()


def spectral_direction_transfer(
    target: torch.Tensor,
    donor: torch.Tensor,
    *,
    rank: int = 8,
    strength: float = 0.1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Transfer donor left/right singular vectors after width truncation.

    Exploratory: truncate donor factors to the target dimensions, replace the
    target's top-k factors, and mix with strength. This is not confirmatory
    training-free evidence.
    """
    if target.ndim != 2 or donor.ndim != 2:
        raise CheckpointError("spectral direction transfer requires matrices")
    if not 0.0 <= strength <= 1.0:
        raise CheckpointError("strength must be in [0, 1]")
    target_value = target.float()
    donor_value = donor.float()
    out_t, in_t = target_value.shape
    out_d, in_d = donor_value.shape
    if out_d < out_t or in_d < in_t:
        raise CheckpointError("donor is smaller than target; refuse expansion")
    u, s, vh = torch.linalg.svd(donor_value, full_matrices=False)
    k = min(rank, out_t, in_t, s.numel())
    donor_approx = (u[:out_t, :k] * s[:k]) @ vh[:k, :in_t]
    mixed = (1.0 - strength) * target_value + strength * donor_approx
    stats = {
        "rank": k,
        "strength": strength,
        "donor_kept_energy": float(s[:k].pow(2).sum().item() / max(1e-12, s.pow(2).sum().item())),
        "relative_frobenius": float(
            torch.linalg.norm(mixed - target_value).item()
            / max(1e-12, torch.linalg.norm(target_value).item())
        ),
    }
    return mixed, stats
