"""Significance-Guided Sequential Attention — brief section 31.

The recurrence, with ``X`` the post-``input_layernorm`` attention input, ``G_s`` the
``o_proj`` image of stage ``s``'s query heads and ``S`` a significance matrix::

    X_0     = X
    X_{s+1} = X_s + alpha_s * G_s(X_s ; S[l, s, h, p])

Three things are composed here and nothing else:

1. a :class:`~rineya_psa.sequential_attention.arms.SequentialArm` — the head partition
   and the ordering hypothesis;
2. the **stage-dependent residual strength** of brief section 27 — ``alpha_s`` from a
   named schedule rather than a magic tuple (:data:`ALPHA_SCHEDULES`);
3. the **per-stage significance** ``S[l, s, h, p]``, in one of the three matrix
   conditions of the section-28 factorial (:data:`MATRIX_CONDITIONS`).

Matrix conditions
-----------------
``none``
    No matrix at all. The arm runs as ``sequential_attention/`` defines it. This is the
    cell that isolates sequentiality from significance.
``global``
    One matrix, identical at every stage. Sequentiality plus a *static* positional
    prior — the cell that asks whether the matrix has to be stage-dependent.
``sequential``
    ``S`` genuinely varies with ``s``, built by gating a base matrix with a
    :data:`STAGE_PROFILES` profile. This is SGSA proper.

How the matrix enters the stage operator
----------------------------------------
Through SPEC 5.3's **Mode 2**: ``V'[h, j, :] = S[l, s, h, b(j)] * V[h, j, :]``, applied
to the full ``[B, H_kv, T, D]`` value tensor *before* the stage selects its heads and
before ``repeat_kv`` — exactly the tensor and exactly the point SPEC names. Mode 2 is
chosen over Mode 1 because it is exact, allocation-free and unrestricted in sequence
length, whereas a per-stage Mode-1 bias would materialise a ``[B, H_stage, T, T]`` mask
*per stage* and SPEC 12.3 already caps that construction at T <= 2048 for a single
application. Mode 3 is not offered because a per-stage constant head gain is what
``alpha_s`` already is, and having two knobs for one quantity makes the factorial
unreadable.

The value scale is ``1 + lam * w[s, b] * S_hat[l, h, b]`` clamped at zero, so ``lam = 0``
and ``condition="none"`` are both *bit-exact* identities against the plain arm (a
float multiply by exactly 1.0 is exact), which is what makes a lambda sweep a
dose-response curve anchored on the sequential baseline rather than on a nearby model.

DEVIATION: :class:`~rineya_psa.sequential_attention.wrapper.SequentialQwen3Attention`
refuses to run with a live ``PSAContext`` attached to a *partial* stage, because the
kernel-level PSA modes index heads and a partial stage renumbers them. That refusal is
correct and is not worked around here: SGSA subclasses the wrapper and applies the
matrix to ``value`` before head selection, where the head axis is still the model's own
KV-head axis. No hook, no ``psa_session``, no renumbering.

DEVIATION: SPEC section 2 permits an import-time ``torch`` only under ``hooks/``,
``loading.py``, ``sequential_attention/``, ``activation_analysis/`` and
``interventions/runtime*.py``. This module is none of those, so ``torch`` and the
wrapper are imported inside functions and the attention subclass is built lazily by
:func:`_sgsa_attention_class`. Config construction — arms, alphas, stage tables — stays
usable on a torch-free interpreter, which is where the factorial is planned.

DEVIATION: ``sequential_attention.arms`` is itself torch-free, but importing it executes
``sequential_attention/__init__.py``, which imports ``stability`` and therefore torch —
legally, since SPEC section 2 exempts that package. Importing it at module scope here
would launder torch straight back into ``interventions``, so ``SequentialArm`` and
``partition_heads`` are imported inside the functions that need them and appear at module
scope only under ``TYPE_CHECKING``.
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from rineya_psa.interventions.psa import (
    SignificanceTable,
    coerce_significance,
    standardize_significance,
)
from rineya_psa.types import Arch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rineya_psa.sequential_attention.arms import SequentialArm

__all__ = [
    "ALPHA_SCHEDULES",
    "DEFAULT_STAGE_PROFILE",
    "MATRIX_CONDITIONS",
    "SGSA_LAMBDA_GRID",
    "STAGE_PROFILES",
    "SGSAConfig",
    "alpha_schedule",
    "build_sgsa",
    "build_sgsa_arm",
    "install_sgsa",
    "sgsa_factorial",
    "sgsa_session",
    "sgsa_value_tables",
    "significance_bin_order",
    "significance_head_order",
    "stage_gates",
]

#: Brief section 28's three matrix conditions, in factorial order.
MATRIX_CONDITIONS: tuple[str, ...] = ("none", "global", "sequential")

#: Dose grid, anchored at 0 so every curve starts on the plain sequential arm.
SGSA_LAMBDA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


# ---------------------------------------------------------------------------
# section 27: stage-dependent residual strength
# ---------------------------------------------------------------------------


def _uniform_weights(n: int) -> list[float]:
    return [1.0] * n


def _increasing_weights(n: int) -> list[float]:
    return [float(s + 1) for s in range(n)]


def _decreasing_weights(n: int) -> list[float]:
    return [float(n - s) for s in range(n)]


def _middle_heavy_weights(n: int) -> list[float]:
    """A symmetric bump peaking on the middle stage(s), strictly positive everywhere.

    ``sin`` on half-open stage centres rather than on stage *edges*: with edges the
    first and last stage would get weight exactly 0, which silently deletes them from
    the recurrence and turns an ``n``-stage arm into an ``n-2``-stage one.
    """
    return [math.sin(math.pi * (s + 0.5) / n) for s in range(n)]


#: Section 27's four named schedules. Each maps ``n_stages -> unnormalised weights``.
ALPHA_SCHEDULES: dict[str, Any] = {
    "uniform": _uniform_weights,
    "increasing": _increasing_weights,
    "decreasing": _decreasing_weights,
    "middle_heavy": _middle_heavy_weights,
}


def alpha_schedule(name: str, n_stages: int, *, total: float = 1.0) -> tuple[float, ...]:
    """Residual coefficients for a named schedule, renormalised to sum to ``total``.

    ``total`` is the whole point of the normalisation: holding ``sum_s alpha_s`` fixed
    means the schedules differ only in *where* the residual write is spent, not in how
    much of it there is, so a difference between them is attributable to stage timing.
    ``total=1.0`` reproduces ``arms.alphas_for(n, "compute_matched")`` for the uniform
    schedule; ``total=n_stages`` reproduces ``"naive"``. Both mandated conventions are
    therefore reachable from here without a second code path.
    """
    n = int(n_stages)
    if n < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")
    try:
        factory = ALPHA_SCHEDULES[name]
    except KeyError:
        raise ValueError(
            f"unknown alpha schedule {name!r}; known: {sorted(ALPHA_SCHEDULES)}"
        ) from None
    weights = [float(w) for w in factory(n)]
    if any(w <= 0.0 for w in weights):
        raise ValueError(f"schedule {name!r} produced a non-positive weight: {weights}")
    scale = float(total) / sum(weights)
    return tuple(w * scale for w in weights)


# ---------------------------------------------------------------------------
# per-stage significance
# ---------------------------------------------------------------------------


def significance_bin_order(table: SignificanceTable) -> tuple[int, ...]:
    """Position bins ranked by descending mean ``|S|`` over layers and KV heads.

    This is the "priority" order the staged profiles consume: bin 0 of the order is the
    position band the matrix says matters most, not the earliest one.
    """
    strength = np.abs(np.asarray(table.values, dtype=np.float64)).mean(axis=(0, 1))
    # Stable sort on the negated strength so ties keep index order and the ranking is
    # reproducible across numpy versions.
    return tuple(int(b) for b in np.argsort(-strength, kind="stable"))


def significance_head_order(table: SignificanceTable, arch: Arch) -> tuple[int, ...]:
    """Query heads ranked by descending mean ``|S|`` of the KV head they read.

    The ordering hypothesis a significance-ordered arm encodes: strong heads act first,
    later stages see a hidden state they have already shaped. Query heads inside one KV
    group are tied by construction (they share a row of the matrix) and are emitted in
    ascending index order, so the result is a deterministic permutation.
    """
    table.assert_matches(arch)
    strength = np.abs(np.asarray(table.values, dtype=np.float64)).mean(axis=(0, 2))  # [H_kv]
    kv_rank = np.argsort(-strength, kind="stable")
    return tuple(
        int(q) for kv in kv_rank for q in arch.q_heads_for_kv(int(kv))
    )


def _replicate_gate(n_stages: int, n_bins: int, order: Sequence[int]) -> np.ndarray:
    return np.ones((n_stages, n_bins), dtype=np.float64)


def _ramp_gate(n_stages: int, n_bins: int, order: Sequence[int]) -> np.ndarray:
    """Same bins every stage, rising dose. Separates *when* from *where*."""
    doses = (np.arange(n_stages, dtype=np.float64) + 1.0) / float(n_stages)
    return np.repeat(doses[:, None], n_bins, axis=1)


def _cumulative_gate(n_stages: int, n_bins: int, order: Sequence[int]) -> np.ndarray:
    """Stage ``s`` sees the top ``(s+1)/n`` of the priority order; the last sees all."""
    gate = np.zeros((n_stages, n_bins), dtype=np.float64)
    for s in range(n_stages):
        take = -(-(s + 1) * n_bins // n_stages)  # ceil, in integer arithmetic
        gate[s, list(order[:take])] = 1.0
    return gate


def _partition_gate(n_stages: int, n_bins: int, order: Sequence[int]) -> np.ndarray:
    """Disjoint bands: stage ``s`` sees only its own block of the priority order."""
    if n_stages > n_bins:
        raise ValueError(
            f"the 'partition' profile needs at least one bin per stage; "
            f"{n_stages} stages, {n_bins} bins"
        )
    from rineya_psa.sequential_attention.arms import partition_heads

    gate = np.zeros((n_stages, n_bins), dtype=np.float64)
    for s, block in enumerate(partition_heads(order, n_stages)):
        gate[s, list(block)] = 1.0
    return gate


#: Named gates ``w[s, b]`` that turn one matrix into a stage-dependent family.
STAGE_PROFILES: dict[str, Any] = {
    "replicate": _replicate_gate,
    "ramp": _ramp_gate,
    "cumulative": _cumulative_gate,
    "partition": _partition_gate,
}

#: Coarse-to-fine over the significance ranking: the hypothesis SGSA exists to test.
DEFAULT_STAGE_PROFILE: str = "cumulative"


def stage_gates(
    n_stages: int,
    n_bins: int,
    profile: str = DEFAULT_STAGE_PROFILE,
    *,
    bin_order: Sequence[int] | None = None,
) -> np.ndarray:
    """``[n_stages, n_bins]`` gate weights for a named profile.

    ``bin_order`` is the priority ranking the partitioning profiles cut up; it defaults
    to plain index order, which makes ``partition`` a left-to-right sweep of the
    context. Pass :func:`significance_bin_order` to rank by the matrix instead.
    """
    s = int(n_stages)
    b = int(n_bins)
    if s < 1 or b < 1:
        raise ValueError(f"need n_stages >= 1 and n_bins >= 1, got {n_stages}, {n_bins}")
    try:
        factory = STAGE_PROFILES[profile]
    except KeyError:
        raise ValueError(
            f"unknown stage profile {profile!r}; known: {sorted(STAGE_PROFILES)}"
        ) from None
    order = tuple(range(b)) if bin_order is None else tuple(int(x) for x in bin_order)
    if sorted(order) != list(range(b)):
        raise ValueError(f"bin_order is not a permutation of range({b}): {order}")
    gate = np.asarray(factory(s, b, order), dtype=np.float64)
    if gate.shape != (s, b):
        raise ValueError(f"profile {profile!r} returned {gate.shape}, expected {(s, b)}")
    return gate


def sgsa_value_tables(
    matrix: Any,
    n_stages: int,
    *,
    lam: float,
    condition: str = "sequential",
    profile: str = DEFAULT_STAGE_PROFILE,
    bin_order: Sequence[int] | None = None,
    standardize: bool = True,
    arch: Arch | None = None,
    clamp_nonnegative: bool = True,
) -> np.ndarray | None:
    """``[L, n_stages, H_kv, n_bins]`` Mode-2 value scales, or ``None`` for ``"none"``.

    The scale is ``1 + lam * w[s, b] * S_hat[l, h, b]``. ``clamp_nonnegative`` stops a
    large ``lam`` from flipping the sign of a head's value contribution, which would be
    a different intervention wearing this one's label; it can only bite at ``lam > 1``
    on a standardised matrix, so the identity at ``lam = 0`` is untouched.
    """
    if condition not in MATRIX_CONDITIONS:
        raise ValueError(f"unknown matrix condition {condition!r}; known: {MATRIX_CONDITIONS}")
    if condition == "none":
        return None

    table = coerce_significance(
        matrix, n_kv_heads=None if arch is None else arch.num_kv_heads
    )
    if arch is not None:
        table.assert_matches(arch)
    prepared = standardize_significance(table) if standardize else table

    s = int(n_stages)
    if s < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")
    gate = (
        np.ones((s, prepared.n_bins), dtype=np.float64)
        if condition == "global"
        else stage_gates(s, prepared.n_bins, profile, bin_order=bin_order)
    )
    # [L, 1, H, B] * [1, S, 1, B] -> [L, S, H, B]
    scaled = prepared.values[:, None, :, :] * gate[None, :, None, :]
    tables = 1.0 + float(lam) * scaled
    if clamp_nonnegative:
        tables = np.maximum(tables, 0.0)
    return np.ascontiguousarray(tables)


# ---------------------------------------------------------------------------
# the arm
# ---------------------------------------------------------------------------


def build_sgsa_arm(
    arch: Arch,
    n_stages: int,
    *,
    order: Sequence[int] | None = None,
    schedule: str = "uniform",
    alpha_total: float = 1.0,
    alphas: Sequence[float] | None = None,
    recompute_kv: bool = True,
    head_scores: Any = None,
    name: str | None = None,
) -> SequentialArm:
    """A :class:`SequentialArm` whose alphas come from a section-27 named schedule.

    ``order`` carries the ordering hypothesis (see :func:`significance_head_order`);
    the partition is contiguous in order space, exactly as ``arms.partition_heads``
    defines it, so the arm is directly comparable with the Arms A-E family.
    """
    resolved = (
        tuple(range(arch.num_q_heads)) if order is None else tuple(int(h) for h in order)
    )
    if sorted(resolved) != list(range(arch.num_q_heads)):
        raise ValueError(
            f"head order is not a permutation of range({arch.num_q_heads}): {resolved}"
        )
    coefficients = (
        alpha_schedule(schedule, int(n_stages), total=alpha_total)
        if alphas is None
        else tuple(float(a) for a in alphas)
    )
    from rineya_psa.sequential_attention.arms import (
        SequentialArm,
        partition_heads,
        stage_significance_from_head_scores,
    )

    groups = partition_heads(resolved, int(n_stages))
    arm = SequentialArm(
        name=name or f"sgsa_{int(n_stages)}stage",
        n_stages=int(n_stages),
        groups=groups,
        alphas=coefficients,
        recompute_kv=bool(recompute_kv),
        stage_significance=(
            None
            if head_scores is None
            else stage_significance_from_head_scores(groups, head_scores)
        ),
        query_refinement=False,
        alpha_mode="custom",
    )
    arm.validate_against(arch)
    return arm


# ---------------------------------------------------------------------------
# the configuration object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SGSAConfig:
    """One cell of the section-28 factorial: an arm, a condition, a dose.

    ``eq=False`` because ``tables`` is a numpy array and dataclass equality would call
    ``ndarray.__eq__``, which returns an array and silently poisons ``==`` and ``in``.
    """

    label: str
    arm: SequentialArm
    condition: str
    lam: float
    tables: np.ndarray | None  # [L, n_stages, H_kv, n_bins]
    binning: Any = None
    profile: str = DEFAULT_STAGE_PROFILE
    schedule: str = "uniform"
    matrix_name: str = "significance"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.condition not in MATRIX_CONDITIONS:
            raise ValueError(
                f"unknown matrix condition {self.condition!r}; known: {MATRIX_CONDITIONS}"
            )
        if self.condition == "none":
            if self.tables is not None:
                raise ValueError("condition='none' must carry no tables")
            return
        if self.tables is None:
            raise ValueError(f"condition={self.condition!r} needs a table stack")
        t = np.ascontiguousarray(np.asarray(self.tables, dtype=np.float64))
        if t.ndim != 4:
            raise ValueError(f"tables must be [L, n_stages, H_kv, n_bins]; got {t.shape}")
        if t.shape[1] != self.arm.n_stages:
            raise ValueError(
                f"tables carry {t.shape[1]} stages but arm {self.arm.name!r} has "
                f"{self.arm.n_stages}"
            )
        if self.binning is not None and int(getattr(self.binning, "n_bins", -1)) != t.shape[3]:
            raise ValueError(
                f"tables have {t.shape[3]} bins but "
                f"{getattr(self.binning, 'name', self.binning)} has "
                f"{getattr(self.binning, 'n_bins', '?')}"
            )
        object.__setattr__(self, "tables", t)

    @property
    def n_stages(self) -> int:
        return self.arm.n_stages

    @property
    def n_bins(self) -> int:
        return 0 if self.tables is None else int(self.tables.shape[3])

    def is_identity_matrix(self) -> bool:
        """True when the matrix cannot move a value — the anchor of the lambda sweep."""
        return self.tables is None or bool(np.all(self.tables == 1.0))

    def resolve_binning(self) -> Any:
        """The binning, defaulting to ``AbsoluteBinning(n_bins)`` (imported lazily)."""
        if self.binning is not None:
            return self.binning
        from rineya_psa.hooks.binning import AbsoluteBinning

        return AbsoluteBinning(self.n_bins)

    def validate_against(self, arch: Arch) -> None:
        self.arm.validate_against(arch)
        if self.tables is None:
            return
        want = (arch.num_layers, self.arm.n_stages, arch.num_kv_heads)
        got = tuple(int(x) for x in self.tables.shape[:3])
        if got != want:
            raise ValueError(f"SGSA tables are {got} but the model needs {want}")

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "condition": self.condition,
            "lam": self.lam,
            "profile": None if self.condition != "sequential" else self.profile,
            "schedule": self.schedule,
            "matrix": self.matrix_name,
            "is_identity_matrix": self.is_identity_matrix(),
            "table_shape": None if self.tables is None else list(self.tables.shape),
            "binning": (
                None if self.binning is None else self.binning.describe()
            ),
            "arm": self.arm.describe(),
            "meta": dict(self.meta),
        }

    def install(self, model: Any, layers: Iterable[int] | str | None = "all") -> Any:
        return install_sgsa(model, layers, self)

    def session(self, model: Any, layers: Iterable[int] | str | None = "all") -> Any:
        return sgsa_session(model, layers, self)


def build_sgsa(
    arch: Arch,
    matrix: Any = None,
    *,
    n_stages: int = 4,
    condition: str = "sequential",
    lam: float = 1.0,
    order: Sequence[int] | None = None,
    schedule: str = "uniform",
    alpha_total: float = 1.0,
    alphas: Sequence[float] | None = None,
    recompute_kv: bool = True,
    profile: str = DEFAULT_STAGE_PROFILE,
    bin_order: Sequence[int] | None = None,
    rank_bins_by_significance: bool = True,
    standardize: bool = True,
    binning: Any = None,
    label: str | None = None,
) -> SGSAConfig:
    """Assemble one factorial cell.

    ``matrix`` may be ``None`` only for ``condition="none"``. With
    ``rank_bins_by_significance`` (the default) the stage profiles cut the *significance
    ranking* of the bins rather than their index order, which is what makes the
    ``cumulative`` profile a coarse-to-fine sweep rather than a left-to-right one.
    """
    if condition not in MATRIX_CONDITIONS:
        raise ValueError(f"unknown matrix condition {condition!r}; known: {MATRIX_CONDITIONS}")
    arm = build_sgsa_arm(
        arch,
        n_stages,
        order=order,
        schedule=schedule,
        alpha_total=alpha_total,
        alphas=alphas,
        recompute_kv=recompute_kv,
    )

    tables: np.ndarray | None = None
    resolved_binning = binning
    matrix_name = "none"
    if condition != "none":
        if matrix is None:
            raise ValueError(f"condition={condition!r} needs a significance matrix")
        table = coerce_significance(matrix, binning=binning, n_kv_heads=arch.num_kv_heads)
        table.assert_matches(arch)
        matrix_name = table.name
        if resolved_binning is None:
            resolved_binning = table.binning
        order_for_bins = bin_order
        if order_for_bins is None and rank_bins_by_significance:
            order_for_bins = significance_bin_order(table)
        tables = sgsa_value_tables(
            table,
            arm.n_stages,
            lam=lam,
            condition=condition,
            profile=profile,
            bin_order=order_for_bins,
            standardize=standardize,
            arch=arch,
        )

    span = "custom" if order is not None else "index"
    name = label or (
        f"sgsa:{condition}:lam{float(lam):g}:{arm.n_stages}stage:{schedule}:{span}"
    )
    return SGSAConfig(
        label=name,
        arm=arm,
        condition=condition,
        lam=float(lam),
        tables=tables,
        binning=resolved_binning,
        profile=profile,
        schedule=schedule,
        matrix_name=matrix_name,
        meta={"alpha_total": float(alpha_total), "head_order": list(arm.order)},
    )


def sgsa_factorial(
    arch: Arch,
    matrix: Any,
    *,
    n_stages: int = 4,
    conditions: Iterable[str] = MATRIX_CONDITIONS,
    lams: Iterable[float] = (1.0,),
    schedules: Iterable[str] = ("uniform",),
    order: Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[SGSAConfig, ...]:
    """Every (condition, lambda, alpha schedule) cell, de-duplicated.

    ``condition="none"`` ignores ``lam`` — there is no matrix for a dose to scale — so
    it contributes one cell per schedule rather than one per (schedule, lambda). Emitting
    the duplicates would inflate every mean over the factorial by the size of the lambda
    grid, silently.
    """
    out: list[SGSAConfig] = []
    lam_list = [float(x) for x in lams]
    for condition in conditions:
        for schedule in schedules:
            doses = [0.0] if condition == "none" else lam_list
            for lam in doses:
                out.append(
                    build_sgsa(
                        arch,
                        matrix,
                        n_stages=n_stages,
                        condition=condition,
                        lam=lam,
                        order=order,
                        schedule=schedule,
                        **kwargs,
                    )
                )
    return tuple(out)


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _sgsa_attention_class() -> type:
    """Build the wrapper subclass on first use.

    Defined inside a function so that importing ``interventions.sgsa`` does not import
    torch (see the module docstring). ``lru_cache`` keeps it a singleton, so
    ``isinstance`` checks across separate installs still work.
    """
    from rineya_psa.sequential_attention.wrapper import SequentialQwen3Attention

    class SGSAAttention(SequentialQwen3Attention):
        """Staged attention with a per-stage Mode-2 value scale.

        The scale is applied to the *full* ``[B, H_kv, T, D]`` value tensor before the
        stage selects its query heads, so the head axis is still the model's own KV-head
        axis and the significance row for KV head ``h`` lands on KV head ``h``.
        """

        def __init__(self, attn: Any, arm: SequentialArm, tables: Any, binning: Any) -> None:
            super().__init__(attn, arm)
            self.stage_tables = None if tables is None else np.ascontiguousarray(tables)
            self.stage_binning = binning
            self._scale_cache: dict[tuple[int, int, str, Any], Any] = {}

        def _stage_scale(self, stage: int, value: Any) -> Any:
            if self.stage_tables is None:
                return None
            import torch

            kv_len = int(value.shape[-2])
            key = (int(stage), kv_len, str(value.device), value.dtype)
            cached = self._scale_cache.get(key)
            if cached is None:
                row = torch.as_tensor(
                    self.stage_tables[stage], dtype=value.dtype, device=value.device
                )  # [H_kv, n_bins]
                cached = self.stage_binning.gather_source(row, kv_len)  # [H_kv, T]
                self._scale_cache[key] = cached
            return cached

        def _stage_attention(
            self,
            stage: int,
            query: Any,
            key: Any,
            value: Any,
            attention_mask: Any,
            kwargs: dict[str, Any],
        ) -> Any:
            scale = self._stage_scale(stage, value)
            if scale is not None:
                value = value * scale.unsqueeze(0).unsqueeze(-1)
            return super()._stage_attention(stage, query, key, value, attention_mask, kwargs)

        def extra_repr(self) -> str:
            has = "none" if self.stage_tables is None else str(self.stage_tables.shape)
            return f"{super().extra_repr()}, stage_tables={has}"

    return SGSAAttention


def _resolve_layers(model: Any, layers: Iterable[int] | str | None) -> tuple[int, ...]:
    """Layer indices to install on. Mirrors ``wrapper._resolve_layers``, negatives included.

    Reimplemented rather than imported because the upstream helper is private; copying
    six lines is cheaper than pinning a private name across module boundaries.
    """
    from rineya_psa.sequential_attention.wrapper import decoder_layers

    n = len(decoder_layers(model))
    if layers is None or layers == "all":
        return tuple(range(n))
    resolved = tuple(sorted({int(i) if int(i) >= 0 else n + int(i) for i in layers}))
    for i in resolved:
        if not 0 <= i < n:
            raise IndexError(f"layer {i} out of range [0,{n})")
    return resolved


def install_sgsa(
    model: Any, layers: Iterable[int] | str | None, config: SGSAConfig
) -> Any:
    """Swap ``layers[i].self_attn`` for an SGSA wrapper. Returns the wrapper's handle.

    The handle is ``sequential_attention.wrapper.InstalledSequential``, so ``uninstall``,
    ``describe`` and the context-manager spelling behave exactly as they do for a plain
    sequential install.
    """
    from rineya_psa.loading import arch_from_config
    from rineya_psa.sequential_attention.wrapper import (
        InstalledSequential,
        SequentialQwen3Attention,
        decoder_layers,
    )

    config.validate_against(arch_from_config(model.config))
    binning = config.resolve_binning() if config.tables is not None else None

    blocks = decoder_layers(model)
    indices = _resolve_layers(model, layers)
    cls = _sgsa_attention_class()
    originals: dict[int, Any] = {}
    wrappers: dict[int, Any] = {}
    try:
        for idx in indices:
            current = blocks[idx].self_attn
            if isinstance(current, SequentialQwen3Attention):
                raise RuntimeError(
                    f"layer {idx} already carries a sequential wrapper; uninstall first"
                )
            per_layer = None if config.tables is None else config.tables[idx]
            wrapper = cls(current, config.arm, per_layer, binning)
            originals[idx] = current
            wrappers[idx] = wrapper
            blocks[idx].self_attn = wrapper
    except Exception:
        for idx, original in originals.items():
            blocks[idx].self_attn = original
        raise
    return InstalledSequential(model, indices, config.arm, wrappers, originals)


@contextlib.contextmanager
def sgsa_session(
    model: Any, layers: Iterable[int] | str | None, config: SGSAConfig
) -> Iterator[Any]:
    """Install for the duration of the block, restore on exit including on exception."""
    handle = install_sgsa(model, layers, config)
    try:
        yield handle
    finally:
        handle.uninstall()
