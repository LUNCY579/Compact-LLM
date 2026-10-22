"""Control-arm definitions and exact-support audit helpers.

These utilities deliberately have no dependency on the PCSM geometry or optimizer
modules.  The synthetic qualification harness can therefore establish fixed-mask
and matched-budget invariants before those production components are integrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

ArmName = Literal[
    "dense_adamw",
    "random_mask",
    "magnitude_mask",
    "gradient_mask",
    "shuffled_peer_pcsm",
    "no_peer_pcsm",
    "pcsm_no_stabilizer",
    "real_peer_pcsm",
]


@dataclass(frozen=True, slots=True)
class ArmDefinition:
    """Transparent declaration of the information and mechanism used by an arm."""

    name: ArmName
    support_rule: str
    uses_aligned_peers: bool
    uses_peer_stabilizer: bool
    full_support_reference: bool = False


_ARM_DEFINITIONS: Final[dict[str, ArmDefinition]] = {
    "dense_adamw": ArmDefinition(
        name="dense_adamw",
        support_rule="all_parameters",
        uses_aligned_peers=False,
        uses_peer_stabilizer=False,
        full_support_reference=True,
    ),
    "random_mask": ArmDefinition(
        name="random_mask",
        support_rule="seeded_uniform_without_replacement",
        uses_aligned_peers=False,
        uses_peer_stabilizer=False,
    ),
    "magnitude_mask": ArmDefinition(
        name="magnitude_mask",
        support_rule="topk_initial_weight_magnitude",
        uses_aligned_peers=False,
        uses_peer_stabilizer=False,
    ),
    "gradient_mask": ArmDefinition(
        name="gradient_mask",
        support_rule="topk_calibration_gradient_magnitude",
        uses_aligned_peers=False,
        uses_peer_stabilizer=False,
    ),
    "shuffled_peer_pcsm": ArmDefinition(
        name="shuffled_peer_pcsm",
        support_rule="topk_pcsm_with_coordinate_shuffled_peers",
        uses_aligned_peers=False,
        uses_peer_stabilizer=True,
    ),
    "no_peer_pcsm": ArmDefinition(
        name="no_peer_pcsm",
        support_rule="topk_target_only_structural_descriptor",
        uses_aligned_peers=False,
        uses_peer_stabilizer=False,
    ),
    "pcsm_no_stabilizer": ArmDefinition(
        name="pcsm_no_stabilizer",
        support_rule="topk_aligned_peer_pcsm",
        uses_aligned_peers=True,
        uses_peer_stabilizer=False,
    ),
    "real_peer_pcsm": ArmDefinition(
        name="real_peer_pcsm",
        support_rule="topk_aligned_peer_pcsm",
        uses_aligned_peers=True,
        uses_peer_stabilizer=True,
    ),
}

ARM_DEFINITIONS: Final = MappingProxyType(_ARM_DEFINITIONS)


@dataclass(frozen=True, slots=True)
class FrozenCoordinateAudit:
    """Byte-level evidence that unsupported coordinates did not change."""

    exact: bool
    frozen_coordinate_count: int
    changed_frozen_coordinate_count: int

    def to_dict(self) -> dict[str, bool | int]:
        return {
            "exact": self.exact,
            "frozen_coordinate_count": self.frozen_coordinate_count,
            "changed_frozen_coordinate_count": self.changed_frozen_coordinate_count,
        }


def require_arm(name: str) -> ArmDefinition:
    """Return a declared arm or fail with a stable, user-facing error."""

    try:
        return ARM_DEFINITIONS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(ARM_DEFINITIONS))
        raise ValueError(
            f"unknown control arm {name!r}; expected one of: {valid}"
        ) from exc


def support_size(n_parameters: int, fraction: float) -> int:
    """Convert a positive fractional budget to a non-empty exact TopK size."""

    if n_parameters <= 0:
        raise ValueError("n_parameters must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return min(n_parameters, max(1, int(np.ceil(n_parameters * fraction))))


def deterministic_topk_mask(
    scores: NDArray[np.floating],
    k: int,
) -> NDArray[np.bool_]:
    """Select exactly ``k`` finite scores with stable flat-index tie breaking."""

    score_array = np.asarray(scores)
    if score_array.size == 0:
        raise ValueError("scores must be non-empty")
    if not 1 <= k <= score_array.size:
        raise ValueError("k must be between one and the number of scores")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must all be finite")

    flat = score_array.reshape(-1)
    chosen = np.argsort(-flat, kind="stable")[:k]
    mask = np.zeros(flat.shape, dtype=np.bool_)
    mask[chosen] = True
    return mask.reshape(score_array.shape)


def deterministic_random_mask(
    shape: tuple[int, ...],
    k: int,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """Draw an exact-size uniform support without replacement."""

    n_parameters = int(np.prod(shape, dtype=np.int64))
    if not 1 <= k <= n_parameters:
        raise ValueError("k must be between one and the number of parameters")
    selected = rng.choice(n_parameters, size=k, replace=False)
    mask: NDArray[np.bool_] = np.zeros(n_parameters, dtype=np.bool_)
    mask[selected] = True
    return mask.reshape(shape)


def normalize_update_on_support(
    update: NDArray[np.float64],
    support: NDArray[np.bool_],
    target_l2: float,
) -> NDArray[np.float64]:
    """Zero unsupported entries and match an exact parameter-space L2 budget."""

    update_array = np.asarray(update, dtype=np.float64)
    support_array = np.asarray(support, dtype=np.bool_)
    if update_array.shape != support_array.shape:
        raise ValueError("update and support must have identical shapes")
    if target_l2 < 0.0 or not np.isfinite(target_l2):
        raise ValueError("target_l2 must be finite and non-negative")

    matched = np.zeros_like(update_array)
    active = update_array[support_array]
    norm = float(np.linalg.vector_norm(active))
    if target_l2 == 0.0:
        return matched
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("cannot budget-match a zero or non-finite active update")
    matched[support_array] = active * (target_l2 / norm)
    return matched


def bitwise_frozen_audit(
    before: NDArray[np.floating],
    after: NDArray[np.floating],
    support: NDArray[np.bool_],
) -> FrozenCoordinateAudit:
    """Compare each unsupported floating-point value as raw bytes."""

    before_array = np.asarray(before)
    after_array = np.asarray(after)
    support_array = np.asarray(support, dtype=np.bool_)
    if (
        before_array.shape != after_array.shape
        or before_array.shape != support_array.shape
    ):
        raise ValueError("before, after, and support must have identical shapes")
    if before_array.dtype != after_array.dtype:
        raise ValueError("before and after must have identical dtypes")

    before_frozen = np.ascontiguousarray(before_array[~support_array])
    after_frozen = np.ascontiguousarray(after_array[~support_array])
    byte_dtype = np.dtype((np.void, before_array.dtype.itemsize))
    before_bits = before_frozen.view(byte_dtype).reshape(-1)
    after_bits = after_frozen.view(byte_dtype).reshape(-1)
    changed = int(np.count_nonzero(before_bits != after_bits))
    return FrozenCoordinateAudit(
        exact=changed == 0,
        frozen_coordinate_count=int(before_frozen.size),
        changed_frozen_coordinate_count=changed,
    )


def support_digest(support: NDArray[np.bool_]) -> str:
    """Hash shape plus packed support bits for compact fixed-plan evidence."""

    support_array = np.asarray(support, dtype=np.bool_)
    shape_bytes = np.asarray(support_array.shape, dtype=np.int64).tobytes()
    packed = np.packbits(support_array.reshape(-1), bitorder="little").tobytes()
    return sha256(shape_bytes + packed).hexdigest()


def sparse_comparison_arms(arms: tuple[str, ...]) -> tuple[str, ...]:
    """Return declared non-dense arms that must share the same update budget."""

    return tuple(name for name in arms if not require_arm(name).full_support_reference)
