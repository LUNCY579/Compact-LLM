"""Arm configurations for the sequential-attention study (brief section 22).

An *arm* is a partition of a layer's query heads into ordered stages plus the
per-stage residual coefficients. ``wrapper.SequentialQwen3Attention`` consumes one
and nothing else: the arm is the whole experimental condition, so it is frozen,
hashable-by-value and ``describe()``-able straight into a provenance stamp.

The recurrence, with ``X`` the post-``input_layernorm`` attention input and ``G_s``
the ``o_proj`` image of stage ``s``'s heads::

    X_0 = X
    X_s = X_{s-1} + alpha_s * G_s(X_{s-1})        for s = 1..n_stages
    attn_output = X_n - X_0 = sum_s alpha_s G_s(X_{s-1})

Arm A is the ``n_stages == 1`` special case and is therefore exactly the stock
parallel attention, which is what makes it a usable baseline rather than an
approximate one.

Residual scaling, honestly
--------------------------
The brief asks for ``alphas = 1/n_stages`` ("compute_matched") and ``alphas = 1``
("naive"), both measured, neither silently preferred. Both are provided and
:func:`alphas_for` refuses to guess. One analytic fact belongs next to the choice
rather than in a footnote: because the stages **partition** the heads, each
``G_s`` carries only its own heads' share of the output, and

    sum_s G_s(X)  ==  o_proj(concat_g head_g(X))  ==  the parallel output

exactly (``o_proj`` is linear and Qwen3 sets ``attention_bias=False``). So at
``alphas = 1`` and zero recurrence the write to the residual stream has *exactly*
the parallel magnitude, and ``alphas = 1/n_stages`` shrinks the attention write by
``1/n``. The brief's premise — "adding s stage outputs where the original added
one changes activation scale" — holds for a *replicated* stage design, not for the
head partition specified here. Both variants are still run, because the recurrence
(stage ``s`` reads ``X_{s-1}``, not ``X_0``) breaks the identity above and the size
of that break is an empirical question. ``stability.py`` measures it.

Arm E (``query_refinement``) is the exception: it does not move the hidden state at
all, so its ``alphas`` scale the query update instead. See
:class:`rineya_psa.sequential_attention.wrapper.SequentialQwen3Attention`.

DEVIATION: the brief's ``SequentialArm(name, n_stages, groups, alphas,
recompute_kv, stage_significance)`` is implemented with two additional fields that
both default to a value reproducing the brief's behaviour — ``query_refinement``
(False) selects Arm E's evolving-query recurrence, and ``alpha_mode`` ("custom")
records which of the two mandated scaling conventions produced ``alphas`` so the
provenance stamp does not have to re-derive it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rineya_psa.types import Arch

#: Arm identifiers, in brief order A..E.
ARM_NAMES: tuple[str, ...] = (
    "parallel",
    "two_stage",
    "four_stage",
    "fully_sequential",
    "query_refinement",
)

#: ``compute_matched`` = 1/n_stages, ``naive`` = 1.0. ``custom`` marks a hand-set tuple.
ALPHA_MODES: tuple[str, ...] = ("compute_matched", "naive", "custom")

#: The two conventions the brief requires to be measured side by side.
MANDATED_ALPHA_MODES: tuple[str, ...] = ("compute_matched", "naive")

#: Arm E's default stage count. Four matches Arm C so the two are directly comparable.
DEFAULT_REFINEMENT_STAGES: int = 4


def alphas_for(n_stages: int, mode: str) -> tuple[float, ...]:
    """Residual coefficients for one of the two mandated scaling conventions."""
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")
    if mode == "compute_matched":
        return tuple([1.0 / n_stages] * n_stages)
    if mode == "naive":
        return tuple([1.0] * n_stages)
    raise ValueError(
        f"alpha mode {mode!r} is not one of {MANDATED_ALPHA_MODES}; pass an explicit "
        "`alphas` tuple instead of asking this function to invent one"
    )


def partition_heads(order: Sequence[int], n_stages: int) -> tuple[tuple[int, ...], ...]:
    """Cut a head ordering into ``n_stages`` contiguous, near-equal, non-empty groups.

    Contiguity in *order space* is the point: the ordering carries the hypothesis
    (significance-descending, role-guided, ...) and the partition must not scramble
    it. Remainder heads go to the earliest stages, so with 16 heads in 3 stages the
    sizes are (6, 5, 5) rather than (5, 5, 6).
    """
    heads = tuple(int(h) for h in order)
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")
    if n_stages > len(heads):
        raise ValueError(
            f"cannot split {len(heads)} heads into {n_stages} non-empty stages"
        )
    base, extra = divmod(len(heads), n_stages)
    groups: list[tuple[int, ...]] = []
    cursor = 0
    for s in range(n_stages):
        size = base + (1 if s < extra else 0)
        groups.append(heads[cursor : cursor + size])
        cursor += size
    return tuple(groups)


@dataclass(frozen=True)
class SequentialArm:
    """One experimental condition: a staged head partition plus its coefficients."""

    name: str
    n_stages: int
    groups: tuple[tuple[int, ...], ...]
    alphas: tuple[float, ...]
    recompute_kv: bool = True
    stage_significance: tuple[float, ...] | None = None
    query_refinement: bool = False
    alpha_mode: str = "custom"

    def __post_init__(self) -> None:
        if self.n_stages != len(self.groups):
            raise ValueError(
                f"n_stages={self.n_stages} but {len(self.groups)} groups were given"
            )
        if len(self.alphas) != self.n_stages:
            raise ValueError(
                f"{len(self.alphas)} alphas for {self.n_stages} stages"
            )
        if self.stage_significance is not None and len(self.stage_significance) != self.n_stages:
            raise ValueError(
                f"{len(self.stage_significance)} significance values for "
                f"{self.n_stages} stages"
            )
        if self.alpha_mode not in ALPHA_MODES:
            raise ValueError(f"unknown alpha_mode {self.alpha_mode!r}; known: {ALPHA_MODES}")
        seen: set[int] = set()
        for s, group in enumerate(self.groups):
            if not group:
                raise ValueError(f"stage {s} of arm {self.name!r} is empty")
            for head in group:
                if int(head) < 0:
                    raise ValueError(f"negative query head {head} in stage {s}")
                if int(head) in seen:
                    raise ValueError(
                        f"query head {head} appears in more than one stage of "
                        f"arm {self.name!r}; stages must partition the heads"
                    )
                seen.add(int(head))
        if self.query_refinement and self.recompute_kv:
            # Arm E never moves the hidden state, so "recompute K/V from the updated
            # hidden state" has no updated hidden state to recompute from. Silently
            # honouring the flag would bill the study for work that changes nothing.
            raise ValueError(
                "query_refinement arms hold the hidden state fixed, so recompute_kv=True "
                "is meaningless; construct them with recompute_kv=False"
            )

    @property
    def order(self) -> tuple[int, ...]:
        """The head ordering this arm was built from, recovered from the partition."""
        return tuple(head for group in self.groups for head in group)

    @property
    def n_heads(self) -> int:
        return sum(len(g) for g in self.groups)

    @property
    def is_parallel(self) -> bool:
        """True when this arm is provably identical to stock attention."""
        return self.n_stages == 1 and self.alphas[0] == 1.0

    def covers(self, num_q_heads: int) -> bool:
        return set(self.order) == set(range(num_q_heads))

    def validate_against(self, arch: Arch) -> None:
        """Every query head in exactly one stage — checked against the real arch."""
        if not self.covers(arch.num_q_heads):
            missing = sorted(set(range(arch.num_q_heads)) - set(self.order))
            extra = sorted(set(self.order) - set(range(arch.num_q_heads)))
            raise ValueError(
                f"arm {self.name!r} does not partition the {arch.num_q_heads} query "
                f"heads: missing={missing}, out_of_range={extra}"
            )

    def kv_heads_for_stage(self, stage: int, arch: Arch) -> tuple[int, ...]:
        """KV heads read by stage ``stage``, via the single sanctioned GQA mapping."""
        return tuple(sorted({arch.kv_head_for_q(int(g)) for g in self.groups[stage]}))

    def kv_projection_cost(self) -> int:
        """K/V projections per forward, relative to parallel attention (which is 1).

        The scientific object is ``recompute_kv=True``: heads at stage ``s`` re-read the
        context through keys and values derived from ``X_{s-1}``, which is what makes
        the arm sequential rather than a re-parenthesised parallel layer. It costs
        ``n_stages`` K/V projections, K-norm applications and RoPE applications per
        layer instead of one. ``recompute_kv=False`` is the cheap control: it reuses
        the stage-0 K/V and isolates the effect of the evolving *query*.
        """
        return self.n_stages if self.recompute_kv else 1

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_stages": self.n_stages,
            "groups": [list(g) for g in self.groups],
            "alphas": list(self.alphas),
            "alpha_mode": self.alpha_mode,
            "recompute_kv": self.recompute_kv,
            "query_refinement": self.query_refinement,
            "stage_significance": (
                None if self.stage_significance is None else list(self.stage_significance)
            ),
            "order": list(self.order),
            "kv_projection_cost": self.kv_projection_cost(),
        }


def stage_significance_from_head_scores(
    groups: Sequence[Sequence[int]],
    head_scores: Mapping[int, float] | Sequence[float],
) -> tuple[float, ...]:
    """Mean per-head score of each stage, for reporting which stage got the strong heads."""
    def score(head: int) -> float:
        if isinstance(head_scores, Mapping):
            return float(head_scores[head])
        return float(head_scores[head])

    return tuple(
        sum(score(int(h)) for h in group) / len(group) for group in groups
    )


def _resolved_order(arch: Arch, order: Sequence[int] | None) -> tuple[int, ...]:
    if order is None:
        return tuple(range(arch.num_q_heads))
    resolved = tuple(int(h) for h in order)
    if sorted(resolved) != list(range(arch.num_q_heads)):
        raise ValueError(
            f"head order is not a permutation of range({arch.num_q_heads}): {resolved}"
        )
    return resolved


def _build(
    *,
    name: str,
    arch: Arch,
    n_stages: int,
    order: Sequence[int] | None,
    alpha_mode: str,
    alphas: Sequence[float] | None,
    recompute_kv: bool,
    head_scores: Mapping[int, float] | Sequence[float] | None,
    query_refinement: bool = False,
) -> SequentialArm:
    resolved = _resolved_order(arch, order)
    groups = partition_heads(resolved, n_stages)
    if alphas is None:
        coefficients = alphas_for(n_stages, alpha_mode)
        mode = alpha_mode
    else:
        coefficients = tuple(float(a) for a in alphas)
        mode = "custom"
    arm = SequentialArm(
        name=name,
        n_stages=n_stages,
        groups=groups,
        alphas=coefficients,
        recompute_kv=recompute_kv,
        stage_significance=(
            None
            if head_scores is None
            else stage_significance_from_head_scores(groups, head_scores)
        ),
        query_refinement=query_refinement,
        alpha_mode=mode,
    )
    arm.validate_against(arch)
    return arm


def arm_parallel(
    arch: Arch,
    *,
    order: Sequence[int] | None = None,
    head_scores: Mapping[int, float] | Sequence[float] | None = None,
) -> SequentialArm:
    """Arm A. One stage, every head, alpha 1 — bit-identical to stock attention.

    Mandatory baseline: every other arm is reported as a difference against this,
    and ``tests/test_sequential_attention.py`` pins the identity so that difference
    cannot acquire a silent offset. ``order`` is accepted and ignored for the
    partition (there is only one group) but is preserved in ``arm.order`` so a sweep
    can carry the same ordering label across all five arms.
    """
    resolved = _resolved_order(arch, order)
    arm = SequentialArm(
        name="parallel",
        n_stages=1,
        groups=(resolved,),
        alphas=(1.0,),
        recompute_kv=False,
        stage_significance=(
            None
            if head_scores is None
            else stage_significance_from_head_scores((resolved,), head_scores)
        ),
        query_refinement=False,
        # 1/1 == 1, so the two mandated conventions coincide here and the label is honest.
        alpha_mode="compute_matched",
    )
    arm.validate_against(arch)
    return arm


def arm_two_stage(
    arch: Arch,
    *,
    order: Sequence[int] | None = None,
    alpha_mode: str = "compute_matched",
    alphas: Sequence[float] | None = None,
    recompute_kv: bool = True,
    head_scores: Mapping[int, float] | Sequence[float] | None = None,
) -> SequentialArm:
    """Arm B. ``X1 = X + a1 G1(X)``; ``Y = X1 + a2 G2(X1)``."""
    return _build(
        name="two_stage",
        arch=arch,
        n_stages=2,
        order=order,
        alpha_mode=alpha_mode,
        alphas=alphas,
        recompute_kv=recompute_kv,
        head_scores=head_scores,
    )


def arm_four_stage(
    arch: Arch,
    *,
    order: Sequence[int] | None = None,
    alpha_mode: str = "compute_matched",
    alphas: Sequence[float] | None = None,
    recompute_kv: bool = True,
    head_scores: Mapping[int, float] | Sequence[float] | None = None,
) -> SequentialArm:
    """Arm C. Same recurrence as Arm B with four groups."""
    return _build(
        name="four_stage",
        arch=arch,
        n_stages=4,
        order=order,
        alpha_mode=alpha_mode,
        alphas=alphas,
        recompute_kv=recompute_kv,
        head_scores=head_scores,
    )


def arm_fully_sequential(
    arch: Arch,
    *,
    order: Sequence[int] | None = None,
    alpha_mode: str = "compute_matched",
    alphas: Sequence[float] | None = None,
    recompute_kv: bool = True,
    head_scores: Mapping[int, float] | Sequence[float] | None = None,
) -> SequentialArm:
    """Arm D. One query head per stage — the minimal sequential unit on this arch.

    Sixteen stages at ``recompute_kv=True`` means sixteen K/V projections per layer.
    That is the most expensive arm by a wide margin and the reason the brief calls
    for it on a restricted layer set rather than all 28.
    """
    return _build(
        name="fully_sequential",
        arch=arch,
        n_stages=arch.num_q_heads,
        order=order,
        alpha_mode=alpha_mode,
        alphas=alphas,
        recompute_kv=recompute_kv,
        head_scores=head_scores,
    )


def arm_query_refinement(
    arch: Arch,
    *,
    order: Sequence[int] | None = None,
    n_stages: int = DEFAULT_REFINEMENT_STAGES,
    alpha_mode: str = "compute_matched",
    alphas: Sequence[float] | None = None,
    recompute_kv: bool = False,
    head_scores: Mapping[int, float] | Sequence[float] | None = None,
) -> SequentialArm:
    """Arm E. Hidden state fixed; the *query* evolves across stages.

    ``recompute_kv`` exists only so a sweep can pass the same keyword to all five
    factories; ``True`` raises rather than being quietly ignored, because Arm E has
    no updated hidden state to recompute K/V from (see
    :class:`SequentialArm.__post_init__`). The parameter-free refinement operator is
    documented on the wrapper.
    """
    if recompute_kv:
        raise ValueError(
            "query_refinement holds the hidden state fixed, so recompute_kv=True would "
            "bill n_stages K/V projections that all produce the same tensors"
        )
    return _build(
        name="query_refinement",
        arch=arch,
        n_stages=n_stages,
        order=order,
        alpha_mode=alpha_mode,
        alphas=alphas,
        recompute_kv=False,
        head_scores=head_scores,
        query_refinement=True,
    )


_FACTORIES = {
    "parallel": arm_parallel,
    "two_stage": arm_two_stage,
    "four_stage": arm_four_stage,
    "fully_sequential": arm_fully_sequential,
    "query_refinement": arm_query_refinement,
}


def build_arm(name: str, arch: Arch, **kwargs: Any) -> SequentialArm:
    """Dispatch by arm name. ``kwargs`` are forwarded to the matching factory."""
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise KeyError(f"unknown arm {name!r}; known: {ARM_NAMES}") from None
    if name == "parallel":
        kwargs = {k: v for k, v in kwargs.items() if k in {"order", "head_scores"}}
    return factory(arch, **kwargs)


def all_arms(
    arch: Arch,
    *,
    order: Sequence[int] | None = None,
    alpha_mode: str = "compute_matched",
    recompute_kv: bool = True,
    head_scores: Mapping[int, float] | Sequence[float] | None = None,
) -> tuple[SequentialArm, ...]:
    """Arms A-E at one ordering and one scaling convention.

    Arm E is always built with ``recompute_kv=False`` regardless of the argument:
    it is the one arm for which the policy is not a free choice.
    """
    return tuple(
        build_arm(
            name,
            arch,
            order=order,
            alpha_mode=alpha_mode,
            recompute_kv=False if name == "query_refinement" else recompute_kv,
            head_scores=head_scores,
        )
        for name in ARM_NAMES
    )


def arm_grid(
    arch: Arch,
    *,
    order: Sequence[int] | None = None,
    names: Sequence[str] = ARM_NAMES,
    alpha_modes: Sequence[str] = MANDATED_ALPHA_MODES,
    recompute_kv: Sequence[bool] = (True, False),
    head_scores: Mapping[int, float] | Sequence[float] | None = None,
) -> tuple[SequentialArm, ...]:
    """Every (arm, alpha convention, K/V policy) cell, de-duplicated.

    Arm A collapses to a single cell (one stage, alpha 1, nothing to recompute) and
    Arm E to one cell per alpha convention, so the grid is not a naive product.
    """
    out: list[SequentialArm] = []
    seen: set[tuple[Any, ...]] = set()
    for name in names:
        for mode in alpha_modes:
            for recompute in recompute_kv:
                if name == "query_refinement" and recompute:
                    continue
                arm = build_arm(
                    name,
                    arch,
                    order=order,
                    alpha_mode=mode,
                    recompute_kv=recompute,
                    head_scores=head_scores,
                )
                key = (arm.name, arm.groups, arm.alphas, arm.recompute_kv, arm.query_refinement)
                if key in seen:
                    continue
                seen.add(key)
                out.append(arm)
    return tuple(out)
