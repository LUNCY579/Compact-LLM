"""Negative controls — brief sections 36 and 37.

Every positive result in this lab is a difference between a treatment and *something*.
These are the somethings. Each one is matched to the treatment on a dimension that a
sceptic would otherwise blame, and differs only on the dimension the hypothesis names:

============================  ===========================  ==========================
control                       matched on                   destroys
============================  ===========================  ==========================
``uniform_middle_boost``      dose, shape, direction       layer/head structure
``random_position_matrix``    every per-head value          which position holds them
``random_head_assignment``    every per-head curve          which head holds which
``sign_flipped``              magnitude, row mean           the sign of the claim
``frozen_gates``              per-head mean gate            positional variation
``random_gates``              per-head mean and sd          any structure at all
``repeat_parallel_stage``     FLOPs, recurrence, alphas     the head *ordering*
``random_sequential``         FLOPs, recurrence, alphas     the ordering hypothesis
``reversed_functional_...``   FLOPs, recurrence, alphas     the ordering *direction*
============================  ===========================  ==========================

Two of these deserve their reasoning spelled out rather than assumed.

**``repeat_parallel_stage`` is the compute-matched, no-reordering arm.** It runs the
same ``n_stages`` recurrence with the same alphas and the same K/V policy — so it pays
the same FLOPs and the same numerical instability as the treatment — but its stages are
contiguous blocks of the *identity* head order, which encodes no hypothesis. Any gap
between it and a significance-ordered arm is attributable to the ordering. A control
that merely re-ran parallel attention ``n`` times would not be matched on the
recurrence, and the recurrence is the thing under test.

**``sign_flipped`` negates the deviation, not the matrix.** ``S' = 2*mu - S`` with
``mu`` the per-``(layer, kv_head)`` mean over bins. Negating outright would also flip
the row mean, turning an attention *reallocation* into an attention *suppression* and
confounding two claims. Reflecting about the row mean keeps the mean, the magnitude and
the dynamic range, and inverts only the position-to-strength mapping — which is the
claim. It is an involution: applying it twice returns the original.

``all_controls_for`` exists so that an experiment script cannot forget one:
:data:`MANDATORY_CONTROLS` lists the full set and
:func:`assert_controls_complete` fails loudly on a mapping that is missing any of them.

DEVIATION: the brief writes ``all_controls_for(matrix, seed)``. Three of the nine
controls are *arms*, not matrices, and an arm cannot be built without knowing the query
head count, so ``arch`` is a required keyword-only argument here. The positional part of
the signature is as the brief writes it.

DEVIATION: ``SequentialArm`` is imported inside the functions that need it. Importing it
at module scope would execute ``sequential_attention/__init__.py``, which imports torch —
permitted there by SPEC section 2, but not here (see ``sgsa.py``'s note).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from rineya_psa.interventions.psa import SignificanceTable, coerce_significance
from rineya_psa.interventions.sgsa import alpha_schedule, build_sgsa_arm
from rineya_psa.types import Arch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rineya_psa.sequential_attention.arms import SequentialArm

__all__ = [
    "CONTROL_KINDS",
    "DEFAULT_CONTROL_SEED",
    "DEFAULT_CONTROL_STAGES",
    "MANDATORY_CONTROLS",
    "Control",
    "all_controls_for",
    "assert_controls_complete",
    "frozen_gates",
    "random_gates",
    "random_head_assignment",
    "random_position_matrix",
    "random_sequential",
    "repeat_parallel_stage",
    "reversed_functional_ordering",
    "sign_flipped",
    "uniform_middle_boost",
]

#: Seed of record for the control family. Recorded in every ``Control.meta``.
DEFAULT_CONTROL_SEED: int = 20260828

#: Stage count the arm-level controls are built at unless a caller says otherwise.
DEFAULT_CONTROL_STAGES: int = 4

CONTROL_KINDS: tuple[str, ...] = ("matrix", "arm")

#: Every control of sections 36-37. ``all_controls_for`` returns exactly these keys.
MANDATORY_CONTROLS: tuple[str, ...] = (
    "uniform_middle_boost",
    "random_position_matrix",
    "random_head_assignment",
    "sign_flipped",
    "frozen_gates",
    "random_gates",
    "repeat_parallel_stage",
    "random_sequential",
    "reversed_functional_ordering",
)


@dataclass(frozen=True, eq=False)
class Control:
    """A named control plus the reason it is the right one.

    ``eq=False``: the payload may be a :class:`SignificanceTable` holding a numpy array,
    and dataclass equality would return an array instead of a bool.
    """

    name: str
    kind: str
    payload: Any  # SignificanceTable | SequentialArm
    rationale: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in CONTROL_KINDS:
            raise ValueError(f"unknown control kind {self.kind!r}; known: {CONTROL_KINDS}")

    def describe(self) -> dict[str, Any]:
        payload = (
            self.payload.describe() if hasattr(self.payload, "describe") else str(self.payload)
        )
        return {
            "name": self.name,
            "kind": self.kind,
            "rationale": self.rationale,
            "payload": payload,
            "meta": dict(self.meta),
        }


def _as_table(matrix: Any, *, n_kv_heads: int | None = None) -> SignificanceTable:
    return coerce_significance(matrix, n_kv_heads=n_kv_heads)


def _rng(seed: int, salt: str) -> np.random.Generator:
    """A per-control stream derived from one seed.

    ``SeedSequence`` spawning rather than ``seed + hash(salt)``: two controls drawn from
    ``default_rng(seed)`` and ``default_rng(seed + 1)`` are *correlated* for small
    offsets, and a correlated pair of negative controls is not two controls.
    """
    entropy = np.random.SeedSequence(
        entropy=int(seed), spawn_key=(int.from_bytes(salt.encode("utf-8"), "big"),)
    )
    return np.random.default_rng(entropy)


# ---------------------------------------------------------------------------
# matrix controls (section 36)
# ---------------------------------------------------------------------------


def uniform_middle_boost(
    reference: Any,
    *,
    amplitude: float | None = None,
    width: float = 1.0 / 3.0,
    n_kv_heads: int | None = None,
) -> SignificanceTable:
    """One middle-of-context bump, identical in every layer and every head.

    The "your matrix is just 'the middle is hard'" control. ``amplitude`` defaults to the
    reference's root-mean-square deviation from its own per-row bin mean, so the boost
    carries the same dose as the matrix it stands in for; the resulting profile is
    centred over bins, like a standardised significance profile, so it is a
    reallocation and not a global gain.
    """
    table = _as_table(reference, n_kv_heads=n_kv_heads)
    n_bins = table.n_bins
    if not 0.0 < float(width) <= 1.0:
        raise ValueError(f"width must lie in (0, 1], got {width}")

    centres = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
    half = float(width) / 2.0
    bump = (np.abs(centres - 0.5) <= half).astype(np.float64)
    if not bump.any():  # a width narrower than one bin still has to name a bin
        bump[int(np.argmin(np.abs(centres - 0.5)))] = 1.0
    profile = bump - bump.mean()

    if amplitude is None:
        dev = table.values - table.values.mean(axis=-1, keepdims=True)
        amplitude = float(np.sqrt(np.mean(dev**2))) if dev.size else 1.0
    peak = float(np.abs(profile).max())
    if peak > 0.0:
        profile = profile * (float(amplitude) / peak)

    values = np.broadcast_to(profile, table.shape).copy()
    return SignificanceTable(
        values=values,
        binning=table.binning,
        name=f"{table.name}:uniform_middle_boost",
        source="control",
        meta={
            "control": "uniform_middle_boost",
            "amplitude": float(amplitude),
            "width": float(width),
            "boosted_bins": [int(b) for b in np.flatnonzero(bump)],
        },
    )


def random_position_matrix(
    matrix: Any, seed: int = DEFAULT_CONTROL_SEED, *, n_kv_heads: int | None = None
) -> SignificanceTable:
    """Shuffle the bin axis independently inside every ``(layer, kv_head)`` profile.

    Matched scale and statistics in the strongest available sense: each row's *multiset*
    of values is untouched, so its mean, sd, min, max and every quantile are identical
    to the real matrix. Only the map from position to strength is destroyed.
    """
    table = _as_table(matrix, n_kv_heads=n_kv_heads)
    rng = _rng(seed, "random_position_matrix")
    values = table.values.copy()
    for layer in range(table.n_layers):
        for head in range(table.n_kv_heads):
            values[layer, head] = rng.permutation(values[layer, head])
    return SignificanceTable(
        values=values,
        binning=table.binning,
        name=f"{table.name}:random_position",
        source="control",
        meta={"control": "random_position_matrix", "seed": int(seed)},
    )


def random_head_assignment(
    matrix: Any, seed: int = DEFAULT_CONTROL_SEED, *, n_kv_heads: int | None = None
) -> SignificanceTable:
    """Right curve, wrong head: permute the KV-head axis within each layer.

    Every positional curve the real matrix contains is still present, at full strength,
    in the same layer — it is simply attached to a different head. A treatment that
    survives this control is exploiting *position*, not head specialisation. The
    permutation is redrawn until it is not the identity (when more than one head exists),
    because an identity draw would silently make the control a duplicate of the
    treatment.
    """
    table = _as_table(matrix, n_kv_heads=n_kv_heads)
    rng = _rng(seed, "random_head_assignment")
    values = table.values.copy()
    perms: list[list[int]] = []
    for layer in range(table.n_layers):
        perm = rng.permutation(table.n_kv_heads)
        tries = 0
        while table.n_kv_heads > 1 and np.array_equal(perm, np.arange(table.n_kv_heads)):
            perm = rng.permutation(table.n_kv_heads)
            tries += 1
            if tries > 64:  # unreachable for H >= 2; a guard, not a fallback
                raise RuntimeError("could not draw a non-identity head permutation")
        values[layer] = table.values[layer][perm]
        perms.append([int(p) for p in perm])
    return SignificanceTable(
        values=values,
        binning=table.binning,
        name=f"{table.name}:random_heads",
        source="control",
        meta={"control": "random_head_assignment", "seed": int(seed), "permutations": perms},
    )


def sign_flipped(matrix: Any, *, n_kv_heads: int | None = None) -> SignificanceTable:
    """Reflect each profile about its own bin mean: ``S' = 2*mu - S``.

    See the module docstring for why this is not a plain negation. If the treatment
    helps and this hurts by a similar magnitude, the effect is in the matrix's shape; if
    both help, the effect is in the perturbation's mere existence.
    """
    table = _as_table(matrix, n_kv_heads=n_kv_heads)
    mu = table.values.mean(axis=-1, keepdims=True)
    return SignificanceTable(
        values=2.0 * mu - table.values,
        binning=table.binning,
        name=f"{table.name}:sign_flipped",
        source="control",
        meta={"control": "sign_flipped"},
    )


def frozen_gates(matrix: Any, *, n_kv_heads: int | None = None) -> SignificanceTable:
    """Replace every profile by its own bin mean — a gate that does not vary with position.

    Section 37's frozen gate. Identical average gate strength per head, zero positional
    information. It is the control for "the significance matrix is doing something" as
    opposed to "any per-head rescale is doing something".
    """
    table = _as_table(matrix, n_kv_heads=n_kv_heads)
    mu = table.values.mean(axis=-1, keepdims=True)
    return SignificanceTable(
        values=np.broadcast_to(mu, table.shape).copy(),
        binning=table.binning,
        name=f"{table.name}:frozen_gates",
        source="control",
        meta={"control": "frozen_gates"},
    )


def random_gates(
    matrix: Any, seed: int = DEFAULT_CONTROL_SEED, *, n_kv_heads: int | None = None
) -> SignificanceTable:
    """Gaussian gates, rescaled to match each profile's mean and sd *exactly*.

    Section 37's random gate. Drawing and reporting "approximately matched moments"
    would leave a reviewer to wonder whether a 3% sd difference explains the gap, so the
    draw is affinely corrected to hit the target mean and sd to floating-point
    precision. A row whose real sd is zero stays constant, which is the only faithful
    match available for it.
    """
    table = _as_table(matrix, n_kv_heads=n_kv_heads)
    rng = _rng(seed, "random_gates")
    values = np.empty_like(table.values)
    for layer in range(table.n_layers):
        for head in range(table.n_kv_heads):
            row = table.values[layer, head]
            target_mean = float(row.mean())
            target_sd = float(row.std())
            draw = rng.standard_normal(table.n_bins)
            draw_sd = float(draw.std())
            if target_sd <= 0.0 or draw_sd <= 0.0 or table.n_bins < 2:
                values[layer, head] = target_mean
                continue
            values[layer, head] = (draw - draw.mean()) * (target_sd / draw_sd) + target_mean
    return SignificanceTable(
        values=values,
        binning=table.binning,
        name=f"{table.name}:random_gates",
        source="control",
        meta={"control": "random_gates", "seed": int(seed)},
    )


# ---------------------------------------------------------------------------
# arm controls (sections 36-37)
# ---------------------------------------------------------------------------


def _arm_alphas(
    n_stages: int, schedule: str, alpha_total: float, alphas: Sequence[float] | None
) -> tuple[float, ...]:
    if alphas is not None:
        return tuple(float(a) for a in alphas)
    return alpha_schedule(schedule, n_stages, total=alpha_total)


def repeat_parallel_stage(
    arch: Arch,
    n_stages: int = DEFAULT_CONTROL_STAGES,
    *,
    schedule: str = "uniform",
    alpha_total: float = 1.0,
    alphas: Sequence[float] | None = None,
    recompute_kv: bool = True,
) -> SequentialArm:
    """Compute-matched sequential arm with **no reordering**: identity head order.

    Same stage count, same alphas, same K/V recomputation policy as the treatment, so
    ``compute.estimate_flops`` returns the identical number for both. The only
    difference is that its stages carry no hypothesis about which heads should act first.
    """
    return build_sgsa_arm(
        arch,
        n_stages,
        order=None,
        schedule=schedule,
        alpha_total=alpha_total,
        alphas=_arm_alphas(n_stages, schedule, alpha_total, alphas),
        recompute_kv=recompute_kv,
        name="repeat_parallel_stage",
    )


def random_sequential(
    arch: Arch,
    n_stages: int = DEFAULT_CONTROL_STAGES,
    seed: int = DEFAULT_CONTROL_SEED,
    *,
    schedule: str = "uniform",
    alpha_total: float = 1.0,
    alphas: Sequence[float] | None = None,
    recompute_kv: bool = True,
) -> SequentialArm:
    """Random head grouping: the ordering hypothesis replaced by a coin flip.

    Identical in every measurable respect to the treatment arm except *which* heads
    share a stage. This is the control that separates "staging heads by significance
    helps" from "staging heads at all helps".
    """
    rng = _rng(seed, "random_sequential")
    order = [int(h) for h in rng.permutation(arch.num_q_heads)]
    arm = build_sgsa_arm(
        arch,
        n_stages,
        order=order,
        schedule=schedule,
        alpha_total=alpha_total,
        alphas=_arm_alphas(n_stages, schedule, alpha_total, alphas),
        recompute_kv=recompute_kv,
        name="random_sequential",
    )
    return arm


def reversed_functional_ordering(
    reference: SequentialArm | Arch,
    *,
    arch: Arch | None = None,
    n_stages: int = DEFAULT_CONTROL_STAGES,
    schedule: str = "uniform",
    alpha_total: float = 1.0,
    recompute_kv: bool = True,
) -> SequentialArm:
    """The treatment arm's head order, reversed. Same heads, same stages, wrong direction.

    Accepts either the arm to reverse (the usual call: reverse the *treatment*) or an
    ``Arch``, in which case the identity order is reversed. If the ordering hypothesis is
    "most significant heads act first", this arm asserts the opposite while holding
    every other quantity — partition sizes, alphas, FLOPs — fixed.
    """
    from rineya_psa.sequential_attention.arms import SequentialArm

    if isinstance(reference, SequentialArm):
        if arch is not None:
            reference.validate_against(arch)
        return SequentialArm(
            name=f"reversed_{reference.name}",
            n_stages=reference.n_stages,
            groups=_partition_like(tuple(reversed(reference.order)), reference.groups),
            alphas=reference.alphas,
            recompute_kv=reference.recompute_kv,
            stage_significance=None,  # the stage->score map is exactly what was reversed
            query_refinement=reference.query_refinement,
            alpha_mode="custom",
        )
    target = reference
    order = tuple(reversed(range(target.num_q_heads)))
    return build_sgsa_arm(
        target,
        n_stages,
        order=order,
        schedule=schedule,
        alpha_total=alpha_total,
        recompute_kv=recompute_kv,
        name="reversed_functional_ordering",
    )


def _partition_like(
    order: Sequence[int], groups: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    """Cut ``order`` into blocks of the same sizes as ``groups``, in the same sequence.

    Reusing the sizes rather than recomputing them keeps the reversed arm matched to its
    treatment even when the treatment used an uneven partition.
    """
    out: list[tuple[int, ...]] = []
    cursor = 0
    for group in groups:
        size = len(group)
        out.append(tuple(int(h) for h in order[cursor : cursor + size]))
        cursor += size
    if cursor != len(order):
        raise ValueError("group sizes do not cover the head order")
    return tuple(out)


# ---------------------------------------------------------------------------
# the complete set
# ---------------------------------------------------------------------------


def all_controls_for(
    matrix: Any,
    seed: int = DEFAULT_CONTROL_SEED,
    *,
    arch: Arch,
    n_stages: int = DEFAULT_CONTROL_STAGES,
    treatment_arm: SequentialArm | None = None,
    schedule: str = "uniform",
    alpha_total: float = 1.0,
    recompute_kv: bool = True,
) -> dict[str, Control]:
    """Every control of sections 36-37, keyed by name. Nothing optional, nothing missing.

    ``treatment_arm`` is the arm the study is actually running; when given, the
    reversed-ordering control reverses *it* (which is the meaningful comparison) and the
    compute-matched controls copy its stage count, alphas and K/V policy so the FLOPs
    match exactly. Without it the controls are built at ``n_stages`` on the identity
    order, which is still internally consistent but is matched to a treatment the caller
    has not named.
    """
    table = coerce_significance(matrix, n_kv_heads=arch.num_kv_heads)
    table.assert_matches(arch)

    if treatment_arm is not None:
        treatment_arm.validate_against(arch)
        stages = treatment_arm.n_stages
        alphas: Sequence[float] | None = treatment_arm.alphas
        kv_policy = treatment_arm.recompute_kv
    else:
        stages = int(n_stages)
        alphas = None
        kv_policy = bool(recompute_kv)

    reference_arm = treatment_arm or repeat_parallel_stage(
        arch,
        stages,
        schedule=schedule,
        alpha_total=alpha_total,
        alphas=alphas,
        recompute_kv=kv_policy,
    )

    controls: dict[str, Control] = {
        "uniform_middle_boost": Control(
            name="uniform_middle_boost",
            kind="matrix",
            payload=uniform_middle_boost(table),
            rationale=(
                "same dose and direction, no layer or head structure: separates 'the "
                "middle is hard' from 'these heads handle the middle'"
            ),
        ),
        "random_position_matrix": Control(
            name="random_position_matrix",
            kind="matrix",
            payload=random_position_matrix(table, seed),
            rationale=(
                "per-head value multiset preserved exactly, bin order shuffled: isolates "
                "the position-to-strength mapping"
            ),
            meta={"seed": int(seed)},
        ),
        "random_head_assignment": Control(
            name="random_head_assignment",
            kind="matrix",
            payload=random_head_assignment(table, seed),
            rationale="right curve, wrong heads: isolates head specialisation",
            meta={"seed": int(seed)},
        ),
        "sign_flipped": Control(
            name="sign_flipped",
            kind="matrix",
            payload=sign_flipped(table),
            rationale=(
                "profile reflected about its own bin mean: same magnitude and row mean, "
                "opposite claim"
            ),
        ),
        "frozen_gates": Control(
            name="frozen_gates",
            kind="matrix",
            payload=frozen_gates(table),
            rationale=(
                "per-head mean gate held constant across positions: no positional information"
            ),
        ),
        "random_gates": Control(
            name="random_gates",
            kind="matrix",
            payload=random_gates(table, seed),
            rationale="per-head mean and sd matched exactly, all structure destroyed",
            meta={"seed": int(seed)},
        ),
        "repeat_parallel_stage": Control(
            name="repeat_parallel_stage",
            kind="arm",
            payload=repeat_parallel_stage(
                arch,
                stages,
                schedule=schedule,
                alpha_total=alpha_total,
                alphas=alphas,
                recompute_kv=kv_policy,
            ),
            rationale="compute-matched recurrence with no head reordering",
        ),
        "random_sequential": Control(
            name="random_sequential",
            kind="arm",
            payload=random_sequential(
                arch,
                stages,
                seed,
                schedule=schedule,
                alpha_total=alpha_total,
                alphas=alphas,
                recompute_kv=kv_policy,
            ),
            rationale="compute-matched recurrence with a random head grouping",
            meta={"seed": int(seed)},
        ),
        "reversed_functional_ordering": Control(
            name="reversed_functional_ordering",
            kind="arm",
            payload=reversed_functional_ordering(reference_arm, arch=arch),
            rationale="the treatment's own ordering, reversed: same heads, wrong direction",
            meta={"reversed_arm": reference_arm.name},
        ),
    }
    assert_controls_complete(controls)
    return controls


def assert_controls_complete(controls: Mapping[str, Control]) -> None:
    """Raise unless every control of :data:`MANDATORY_CONTROLS` is present.

    Call this at the top of an experiment script. A missing negative control is not a
    style problem; it is a result that cannot be defended.
    """
    missing = [name for name in MANDATORY_CONTROLS if name not in controls]
    if missing:
        raise ValueError(
            f"missing mandatory controls {missing}; sections 36-37 require all of "
            f"{list(MANDATORY_CONTROLS)}"
        )
