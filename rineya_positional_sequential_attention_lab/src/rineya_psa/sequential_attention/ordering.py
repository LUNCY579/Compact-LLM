"""Head orderings for the staged arms, and the evidence rule behind the role labels
(brief section 23).

An *ordering* is a permutation of ``range(arch.num_q_heads)``. ``arms.partition_heads``
cuts it into contiguous stages, so the ordering *is* the hypothesis about which heads
should compute first. Five families are provided:

``original``
    Query-head index order. The null hypothesis: index order carries no meaning, so
    any effect of the other orderings is an effect of the ordering rule, not of
    staging per se.
``random_<i>``
    ``n`` seeded permutations. These are the control distribution the informative
    orderings must beat; without them a significance-descending win is unfalsifiable.
``significance_desc``
    Heads sorted by their causal significance, largest first, from a
    :class:`~rineya_psa.significance.matrices.SignificanceMatrix`.
``role_guided`` / ``role_reverse``
    Sorted by an assigned processing role, forwards (retrieval first) and backwards.
    The pair is the actual test: if the forward order helps and the reverse hurts,
    the ordering carries directional information rather than merely being non-random.

Roles are **derived, not asserted**
-----------------------------------
:func:`assign_roles` implements one explicit rule over measured per-head statistics
and returns its inputs next to its outputs, so a report can print the working. Every
criterion is a z-score across the head population, which makes the thresholds
dimensionless and comparable across models. A head that clears no rule is labelled
``None`` — the function refuses to guess, and unlabelled heads are ordered last in
both role orderings so they cannot silently drive the contrast.

Torch-free on purpose: this module is pure bookkeeping over numbers another phase
measured, so it stays importable on the CPU-only analysis path (SPEC section 2).

DEVIATION: a :class:`SignificanceMatrix` is indexed by **KV** head, and GQA makes the
``kv_group_size`` query heads inside one group indistinguishable to any KV-indexed
statistic. :func:`head_scores_from_significance` therefore assigns every query head in
a group its KV head's score and breaks the resulting ties by query-head index; the
degeneracy is recorded in ``HeadOrder.meta['gqa_degenerate']`` rather than hidden.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rineya_psa.types import Arch

__all__ = [
    "DEFAULT_ROLE_THRESHOLDS",
    "HeadOrder",
    "HeadStats",
    "ROLE_SEQUENCE",
    "RoleAssignment",
    "RoleThresholds",
    "all_orders",
    "assign_roles",
    "head_scores_from_significance",
    "is_permutation",
    "original_order",
    "random_orders",
    "role_guided_order",
    "role_reverse_order",
    "significance_descending_order",
]

#: The hypothesised processing pipeline, in forward order. ``role_guided`` runs heads
#: in this order; ``role_reverse`` runs it backwards. Nothing in the code assumes the
#: pipeline is real — that is what the two orderings are for.
ROLE_SEQUENCE: tuple[str, ...] = ("retrieval", "association", "integration", "verification")


# ---------------------------------------------------------------------------
# orderings
# ---------------------------------------------------------------------------


def is_permutation(order: Sequence[int], n: int) -> bool:
    return sorted(int(x) for x in order) == list(range(n))


@dataclass(frozen=True)
class HeadOrder:
    """A named permutation of the query heads, carrying why it looks like that."""

    name: str
    order: tuple[int, ...]
    rule: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        order = tuple(int(h) for h in self.order)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "meta", dict(self.meta))
        if len(set(order)) != len(order):
            raise ValueError(f"ordering {self.name!r} repeats a head: {order}")
        if any(h < 0 for h in order):
            raise ValueError(f"ordering {self.name!r} contains a negative head: {order}")

    def validate_against(self, arch: Arch) -> None:
        if not is_permutation(self.order, arch.num_q_heads):
            raise ValueError(
                f"ordering {self.name!r} is not a permutation of "
                f"range({arch.num_q_heads}): {self.order}"
            )

    def reversed(self, name: str | None = None) -> HeadOrder:
        return HeadOrder(
            name=name or f"{self.name}_reversed",
            order=tuple(reversed(self.order)),
            rule=f"reverse of: {self.rule}",
            meta={**self.meta, "reversed_from": self.name},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "order": list(self.order),
            "rule": self.rule,
            "meta": dict(self.meta),
        }


def original_order(arch: Arch) -> HeadOrder:
    """Arm-agnostic baseline ordering: query-head index, ascending."""
    return HeadOrder(
        name="original",
        order=tuple(range(arch.num_q_heads)),
        rule="query-head index ascending",
    )


def random_orders(arch: Arch, n: int = 8, seed: int = 0) -> tuple[HeadOrder, ...]:
    """``n`` seeded permutations, each recording the seed that produced it.

    Seeds are ``seed + i`` rather than draws from one stream so a single ordering can
    be reproduced without replaying the ones before it.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    out: list[HeadOrder] = []
    for i in range(n):
        rng = np.random.default_rng(int(seed) + i)
        perm = tuple(int(x) for x in rng.permutation(arch.num_q_heads))
        out.append(
            HeadOrder(
                name=f"random_{i}",
                order=perm,
                rule="uniform random permutation",
                meta={"seed": int(seed) + i, "index": i},
            )
        )
    return tuple(out)


def head_scores_from_significance(
    matrix: Any,
    arch: Arch,
    *,
    layer: int | None = None,
    reduce: str = "sum",
) -> dict[int, float]:
    """Per-**query**-head importance from a KV-head-indexed significance matrix.

    ``layer=None`` averages over every layer the matrix carries, which is what a
    single ordering applied to many layers needs. ``reduce`` selects how the position
    bins collapse: ``"sum"`` (total importance of the head), ``"mean"``, or ``"max"``
    (the head's single most important position bin).
    """
    importance = np.asarray(matrix.importance(), dtype=np.float64)  # [L, H_kv, B]
    if layer is None:
        per_layer = importance
    else:
        try:
            row = tuple(matrix.layers).index(int(layer))
        except ValueError:
            raise KeyError(
                f"layer {layer} is not in the matrix (has {list(matrix.layers)})"
            ) from None
        per_layer = importance[row : row + 1]

    if reduce == "sum":
        per_head = np.nansum(per_layer, axis=2)
    elif reduce == "mean":
        per_head = np.nanmean(per_layer, axis=2)
    elif reduce == "max":
        per_head = np.nanmax(per_layer, axis=2)
    else:
        raise ValueError(f"unknown reduce {reduce!r}; expected sum|mean|max")
    per_head = np.nanmean(per_head, axis=0)  # [H_kv]

    scores: dict[int, float] = {}
    for column, kv_head in enumerate(matrix.kv_heads):
        for q_head in arch.q_heads_for_kv(int(kv_head)):
            scores[int(q_head)] = float(per_head[column])
    missing = sorted(set(range(arch.num_q_heads)) - set(scores))
    if missing:
        raise ValueError(
            f"significance matrix covers KV heads {list(matrix.kv_heads)}, leaving query "
            f"heads {missing} unscored; an ordering must rank every head"
        )
    return scores


def significance_descending_order(
    matrix: Any,
    arch: Arch,
    *,
    layer: int | None = None,
    reduce: str = "sum",
    name: str = "significance_desc",
) -> HeadOrder:
    """Most causally significant query heads first."""
    scores = head_scores_from_significance(matrix, arch, layer=layer, reduce=reduce)
    order = tuple(
        sorted(range(arch.num_q_heads), key=lambda h: (-scores[h], h))
    )
    order_obj = HeadOrder(
        name=name,
        order=order,
        rule=(
            f"causal significance ({matrix.objective}) reduced over bins by {reduce!r}"
            f"{'' if layer is None else f' at layer {layer}'}, descending; "
            "ties broken by query-head index"
        ),
        meta={
            "scores": {int(h): float(scores[h]) for h in range(arch.num_q_heads)},
            "objective": getattr(matrix, "objective", None),
            "model_key": getattr(matrix, "model_key", None),
            "matrix_sha256": matrix.content_hash() if hasattr(matrix, "content_hash") else None,
            "layer": layer,
            "reduce": reduce,
            "gqa_degenerate": arch.kv_group_size > 1,
            "kv_group_size": arch.kv_group_size,
        },
    )
    order_obj.validate_against(arch)
    return order_obj


# ---------------------------------------------------------------------------
# roles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadStats:
    """Measured inputs to the role rule, for one ``(layer, q_head)``.

    Every field is a quantity another phase already produces: ``gold_span_attn`` and
    ``entropy`` from ``hooks.capture``, ``middle_mass`` from the position-binned
    attention profile, ``ablation_effect`` from the Mode-3 head knockout on gold NLL,
    ``margin_effect`` from the same knockout on the answer margin. All are "larger =
    more of this property"; sign conventions are the caller's responsibility and are
    recorded in :class:`RoleAssignment`.
    """

    layer: int
    q_head: int
    gold_span_attn: float = float("nan")
    ablation_effect: float = float("nan")
    middle_mass: float = float("nan")
    entropy: float = float("nan")
    margin_effect: float = float("nan")

    @classmethod
    def from_row(cls, row: Any, **overrides: Any) -> HeadStats:
        """Adapt an ``activation_analysis.head_stats.HeadStatRow`` (or a mapping, or
        any object with the right attributes) to this rule's five inputs.

        Extra columns are ignored and two spellings are accepted per field, because the
        producing module names things for its own purposes: ``middle_attn`` is this
        rule's ``middle_mass`` and ``attn_entropy`` is its ``entropy``.

        ``margin_effect`` is deliberately **not** aliased to ``patch_effect``. A patch
        effect and an effect on the answer margin are different measurements, and
        quietly substituting one would make the verification label mean something other
        than what :func:`assign_roles` documents. Supply it via ``overrides`` (typically
        from a second ablation pass against a ``margin``-objective significance matrix)
        or leave it unmeasured and let the rule decline to label verification heads.
        """
        def pick(*names: str) -> float:
            for name in names:
                if isinstance(row, Mapping):
                    if name in row:
                        return float(row[name])
                elif hasattr(row, name):
                    return float(getattr(row, name))
            return float("nan")

        values = {
            "layer": int(pick("layer")),
            "q_head": int(pick("q_head", "head")),
            "gold_span_attn": pick("gold_span_attn"),
            "ablation_effect": pick("ablation_effect"),
            "middle_mass": pick("middle_mass", "middle_attn"),
            "entropy": pick("entropy", "attn_entropy"),
            "margin_effect": pick("margin_effect"),
        }
        values.update(overrides)
        return cls(**values)

    def to_json(self) -> dict[str, Any]:
        return {
            "layer": int(self.layer),
            "q_head": int(self.q_head),
            "gold_span_attn": float(self.gold_span_attn),
            "ablation_effect": float(self.ablation_effect),
            "middle_mass": float(self.middle_mass),
            "entropy": float(self.entropy),
            "margin_effect": float(self.margin_effect),
        }


@dataclass(frozen=True)
class RoleThresholds:
    """The rule's free parameters, in one place so a report can quote them.

    ``z`` is in population standard deviations across all supplied heads; ``late_frac``
    is a fraction of network depth. The defaults are deliberately mild: at ``z = 0.5``
    roughly a third of heads clear any single criterion, which leaves the *conjunctive*
    rules (retrieval needs two, integration and verification need depth as well) doing
    the discriminating rather than an aggressive cut on one noisy statistic.
    """

    z: float = 0.5
    late_frac: float = 0.6

    def to_json(self) -> dict[str, float]:
        return {"z": float(self.z), "late_frac": float(self.late_frac)}


DEFAULT_ROLE_THRESHOLDS = RoleThresholds()


def _zscore(values: np.ndarray) -> np.ndarray:
    """z across the head population, NaN-safe. A constant column scores 0, not inf."""
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if not finite.any():
        return out
    sample = values[finite]
    sd = float(sample.std(ddof=0))
    if sd <= 0.0:
        out[finite] = 0.0
        return out
    out[finite] = (sample - float(sample.mean())) / sd
    return out


#: ``role -> (z-scored statistics that must all clear the threshold, needs late layer)``.
#: This table *is* the rule; :func:`assign_roles` contains no other role knowledge.
ROLE_RULE: dict[str, tuple[tuple[str, ...], bool]] = {
    "retrieval": (("gold_span_attn", "ablation_effect"), False),
    "association": (("middle_mass",), False),
    "integration": (("entropy",), True),
    "verification": (("margin_effect",), True),
}


@dataclass(frozen=True, eq=False)
class RoleAssignment:
    """Labels plus the numbers that produced them. ``label is None`` means 'refused'.

    ``eq=False`` for the reason ``significance.matrices`` gives: the payload includes
    arrays, and a generated ``__eq__`` would return an array and poison ``in``.
    """

    stats: tuple[HeadStats, ...]
    labels: tuple[str | None, ...]
    scores: tuple[float, ...]
    z: Mapping[str, np.ndarray]
    layer_frac: np.ndarray
    thresholds: RoleThresholds
    n_layers: int

    @property
    def n_labelled(self) -> int:
        return sum(1 for label in self.labels if label is not None)

    @property
    def unmeasured(self) -> tuple[str, ...]:
        """Statistics with no finite value anywhere in the population.

        A role whose criteria are all unmeasured can never be assigned, and a report
        that shows zero verification heads should say *why*: "no head cleared the bar"
        and "the bar was never measured" are different findings.
        """
        return tuple(key for key, arr in self.z.items() if not np.isfinite(arr).any())

    @property
    def unevaluable_roles(self) -> tuple[str, ...]:
        missing = set(self.unmeasured)
        return tuple(
            role for role, (keys, _) in ROLE_RULE.items() if missing.intersection(keys)
        )

    def rows(self) -> list[dict[str, Any]]:
        """One dict per head: inputs, z-scores, per-role scores, final label.

        This is what the report prints to show the rule's work; it is deliberately
        wide rather than pre-summarised.
        """
        out: list[dict[str, Any]] = []
        for i, head in enumerate(self.stats):
            row = head.to_json()
            row["layer_frac"] = float(self.layer_frac[i])
            for key, arr in self.z.items():
                row[f"z_{key}"] = float(arr[i])
            for role in ROLE_SEQUENCE:
                row[f"score_{role}"] = float(self._role_score(role, i))
                row[f"passes_{role}"] = bool(self._passes(role, i))
            row["role"] = self.labels[i]
            row["role_score"] = float(self.scores[i])
            out.append(row)
        return out

    def _passes(self, role: str, i: int) -> bool:
        keys, needs_late = ROLE_RULE[role]
        if needs_late and not self.layer_frac[i] >= self.thresholds.late_frac:
            return False
        for key in keys:
            value = self.z[key][i]
            if not np.isfinite(value) or value < self.thresholds.z:
                return False
        return True

    def _role_score(self, role: str, i: int) -> float:
        keys, _ = ROLE_RULE[role]
        values = [self.z[key][i] for key in keys]
        if any(not np.isfinite(v) for v in values):
            return float("nan")
        return float(np.mean(values))

    def head_roles(self, layer: int | None = None) -> dict[int, tuple[str | None, float]]:
        """Aggregate to one label per **query-head index**.

        The staged arms apply one head partition to every installed layer, so a
        role-guided ordering needs a per-index label, not a per-``(layer, head)`` one.
        With ``layer=None`` each index takes the role for which its mean score across
        layers is highest among the roles it passes somewhere; heads that pass nothing
        anywhere stay ``None``. With an explicit ``layer`` only that layer is used.
        """
        by_head: dict[int, dict[str, list[float]]] = {}
        for i, head in enumerate(self.stats):
            if layer is not None and int(head.layer) != int(layer):
                continue
            bucket = by_head.setdefault(int(head.q_head), {})
            for role in ROLE_SEQUENCE:
                if self._passes(role, i):
                    bucket.setdefault(role, []).append(self._role_score(role, i))
        out: dict[int, tuple[str | None, float]] = {}
        for q_head, bucket in by_head.items():
            if not bucket:
                out[q_head] = (None, float("nan"))
                continue
            means = {role: float(np.mean(vals)) for role, vals in bucket.items()}
            best = max(means, key=lambda r: (means[r], -ROLE_SEQUENCE.index(r)))
            out[q_head] = (best, means[best])
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "thresholds": self.thresholds.to_json(),
            "rule": {role: {"z_keys": list(keys), "late_layer": late}
                     for role, (keys, late) in ROLE_RULE.items()},
            "role_sequence": list(ROLE_SEQUENCE),
            "n_heads": len(self.stats),
            "n_labelled": self.n_labelled,
            "unmeasured_statistics": list(self.unmeasured),
            "unevaluable_roles": list(self.unevaluable_roles),
            "n_layers": int(self.n_layers),
            "rows": self.rows(),
        }


def assign_roles(
    head_stats: Sequence[HeadStats] | Sequence[Mapping[str, Any]] | Sequence[Any],
    *,
    thresholds: RoleThresholds = DEFAULT_ROLE_THRESHOLDS,
    n_layers: int | None = None,
) -> RoleAssignment:
    """Label heads by processing role from measured statistics, or refuse to.

    The rule, in full (``z`` = z-score across the supplied head population,
    ``late`` = ``layer / (n_layers - 1) >= late_frac``):

    ===============  ==================================================
    retrieval        ``z(gold_span_attn) >= t`` **and** ``z(ablation_effect) >= t``
    association      ``z(middle_mass) >= t``
    integration      ``z(entropy) >= t`` **and** ``late``
    verification     ``z(margin_effect) >= t`` **and** ``late``
    ===============  ==================================================

    A head may clear several rules; it takes the one with the highest mean z over that
    rule's own criteria, ties resolved by :data:`ROLE_SEQUENCE` position (earlier wins),
    which keeps the assignment deterministic. A head clearing nothing is labelled
    ``None``: the population is small and the statistics are noisy, so refusing is the
    honest outcome and a report that shows twelve labels out of sixteen is more useful
    than one that shows sixteen.
    """
    stats = [
        entry if isinstance(entry, HeadStats) else HeadStats.from_row(entry)
        for entry in head_stats
    ]
    if not stats:
        raise ValueError("assign_roles needs at least one head's statistics")

    depth = int(n_layers) if n_layers is not None else max(int(s.layer) for s in stats) + 1
    denominator = max(depth - 1, 1)
    layer_frac = np.array([float(s.layer) / denominator for s in stats], dtype=np.float64)

    keys = ("gold_span_attn", "ablation_effect", "middle_mass", "entropy", "margin_effect")
    z = {
        key: _zscore(np.array([float(getattr(s, key)) for s in stats], dtype=np.float64))
        for key in keys
    }

    assignment = RoleAssignment(
        stats=tuple(stats),
        labels=tuple([None] * len(stats)),
        scores=tuple([float("nan")] * len(stats)),
        z=z,
        layer_frac=layer_frac,
        thresholds=thresholds,
        n_layers=depth,
    )

    labels: list[str | None] = []
    scores: list[float] = []
    for i in range(len(stats)):
        passing = [role for role in ROLE_SEQUENCE if assignment._passes(role, i)]
        if not passing:
            labels.append(None)
            scores.append(float("nan"))
            continue
        best = max(passing, key=lambda r: (assignment._role_score(r, i), -ROLE_SEQUENCE.index(r)))
        labels.append(best)
        scores.append(assignment._role_score(best, i))

    return RoleAssignment(
        stats=assignment.stats,
        labels=tuple(labels),
        scores=tuple(scores),
        z=z,
        layer_frac=layer_frac,
        thresholds=thresholds,
        n_layers=depth,
    )


def _role_sorted(
    assignment: RoleAssignment,
    arch: Arch,
    sequence: Sequence[str],
    *,
    layer: int | None,
) -> tuple[tuple[int, ...], dict[int, tuple[str | None, float]]]:
    roles = assignment.head_roles(layer=layer)
    rank = {role: i for i, role in enumerate(sequence)}

    def key(head: int) -> tuple[int, float, int]:
        label, score = roles.get(head, (None, float("nan")))
        if label is None:
            # Unlabelled heads go last in *both* directions, so the forward/reverse
            # contrast is a contrast between roles and not between "has a role" and not.
            return (len(sequence), 0.0, head)
        return (rank[label], -float(score), head)

    return tuple(sorted(range(arch.num_q_heads), key=key)), roles


def _role_meta(
    assignment: RoleAssignment,
    roles: Mapping[int, tuple[str | None, float]],
    arch: Arch,
    layer: int | None,
) -> dict[str, Any]:
    return {
        "roles": {
            int(h): roles.get(h, (None, float("nan")))[0] for h in range(arch.num_q_heads)
        },
        "role_scores": {
            int(h): float(roles.get(h, (None, float("nan")))[1])
            for h in range(arch.num_q_heads)
        },
        "n_unlabelled": sum(
            1 for h in range(arch.num_q_heads) if roles.get(h, (None, 0.0))[0] is None
        ),
        "thresholds": assignment.thresholds.to_json(),
        "layer": layer,
    }


def role_guided_order(
    assignment: RoleAssignment,
    arch: Arch,
    *,
    layer: int | None = None,
    sequence: Sequence[str] = ROLE_SEQUENCE,
    name: str = "role_guided",
) -> HeadOrder:
    """Retrieval heads first, verification last; unlabelled heads trail."""
    order, roles = _role_sorted(assignment, arch, sequence, layer=layer)
    out = HeadOrder(
        name=name,
        order=order,
        rule=(
            "assign_roles labels, ordered " + " -> ".join(sequence)
            + "; within a role by descending role score; unlabelled heads last"
        ),
        meta=_role_meta(assignment, roles, arch, layer),
    )
    out.validate_against(arch)
    return out


def role_reverse_order(
    assignment: RoleAssignment,
    arch: Arch,
    *,
    layer: int | None = None,
    sequence: Sequence[str] = ROLE_SEQUENCE,
    name: str = "role_reverse",
) -> HeadOrder:
    """The role sequence run backwards, unlabelled heads still last.

    This is the falsifier for :func:`role_guided_order`: reversing the *pipeline*
    while holding the "labelled before unlabelled" structure fixed isolates the
    direction of the hypothesised information flow.
    """
    order, roles = _role_sorted(assignment, arch, tuple(reversed(tuple(sequence))), layer=layer)
    out = HeadOrder(
        name=name,
        order=order,
        rule=(
            "assign_roles labels, ordered "
            + " -> ".join(reversed(tuple(sequence)))
            + "; within a role by descending role score; unlabelled heads last"
        ),
        meta=_role_meta(assignment, roles, arch, layer),
    )
    out.validate_against(arch)
    return out


def all_orders(
    arch: Arch,
    *,
    matrix: Any | None = None,
    roles: RoleAssignment | None = None,
    n_random: int = 8,
    seed: int = 0,
    layer: int | None = None,
) -> tuple[HeadOrder, ...]:
    """Every ordering the available evidence supports, ``original`` first.

    Missing evidence drops the corresponding ordering rather than substituting a
    fabricated one: no significance matrix means no ``significance_desc`` row in the
    results table, which is the correct thing for the report to show.
    """
    out: list[HeadOrder] = [original_order(arch)]
    out.extend(random_orders(arch, n=n_random, seed=seed))
    if matrix is not None:
        out.append(significance_descending_order(matrix, arch, layer=layer))
    if roles is not None:
        out.append(role_guided_order(roles, arch, layer=layer))
        out.append(role_reverse_order(roles, arch, layer=layer))
    for order in out:
        order.validate_against(arch)
    return tuple(out)
