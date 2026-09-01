"""Closed-form geometry of the P4 target-basis spectral transport operator.

`spectral_projection.spectral_profile_transfer` writes

    projected = residual_scale * W + U diag(desired - residual_scale * s) V^T

where `U`, `s`, `V` are the *target's* own top-r factors and `desired` blends the
target's spectrum toward the donor's normalised one. The pre-rescale Frobenius
norm already equals the target's, so the operator's final norm restoration is
the identity and the induced delta collapses to

    Delta = (residual_scale - 1) * W_perp + sum_i (desired_i - s_i) u_i v_i^T.

Because `{u_1 v_1^T, ..., u_r v_r^T, W_perp/||W_perp||}` is orthonormal, and
because both donors are projected against the same target with the same per-key
seed, the 4B and 8B deltas are expressed in the *same* orthonormal basis. Their
inner products are therefore computable exactly from the recorded scalars, with
no tensor arithmetic.

Shared by v3 E1 and E1c so that both use one implementation of the geometry.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

Coordinates = tuple[list[float], float]


def delta_coordinates_from_profile(
    stat: dict[str, Any],
    profile: list[float] | None,
    strength: float,
) -> Coordinates:
    """Delta coordinates in the shared orthonormal target basis.

    `profile` is the donor's normalised top-r spectrum, the operator's entire
    donor-derived payload. Passing `None` replays the operator's own recorded
    `desired` values instead of recomputing them.
    """
    target_s = stat["target_top_singular_values"]
    frobenius = stat["target_fro"]
    total_energy = frobenius * frobenius
    top_energy = sum(v * v for v in target_s)

    if profile is None:
        desired = list(stat["desired_target_basis_singular_values"])
    else:
        desired = [
            (1.0 - strength) * t + strength * p * frobenius
            for t, p in zip(target_s, profile)
        ]
        desired_energy = sum(d * d for d in desired)
        if desired_energy >= total_energy:  # the operator's own guard rail
            scale = math.sqrt(total_energy * 0.95 / desired_energy)
            desired = [d * scale for d in desired]

    desired_energy = sum(d * d for d in desired)
    residual_scale = math.sqrt(
        max(0.0, total_energy - desired_energy) / max(total_energy - top_energy, 1e-30)
    )
    perpendicular = math.sqrt(max(0.0, total_energy - top_energy))
    return (
        [d - t for d, t in zip(desired, target_s)],
        (residual_scale - 1.0) * perpendicular,
    )


def global_cosine(
    left: dict[str, Coordinates],
    right: dict[str, Coordinates],
    keys: Iterable[str],
) -> float:
    """Cosine between two deltas over all coordinates of all listed tensors."""
    dot = x2 = y2 = 0.0
    for key in keys:
        a, a_perp = left[key]
        b, b_perp = right[key]
        dot += sum(i * j for i, j in zip(a, b)) + a_perp * b_perp
        x2 += sum(i * i for i in a) + a_perp * a_perp
        y2 += sum(j * j for j in b) + b_perp * b_perp
    return dot / max(math.sqrt(x2 * y2), 1e-30)


def donor_profile(stat: dict[str, Any]) -> list[float]:
    """The operator's complete donor-derived payload for one tensor."""
    return [v / stat["donor_fro"] for v in stat["donor_top_singular_values"]]
