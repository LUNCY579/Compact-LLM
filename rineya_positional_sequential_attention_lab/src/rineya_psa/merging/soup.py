"""Brief section 35 -- positional fingerprints, complementarity, and a position-aware soup.

The chain the brief lays out, in order, and implemented in that order here:

1. **Positional fingerprint** ``P_m = [A_m(p_1) ... A_m(p_k)]`` -- one model reduced to its
   score at each point of the position grid (:class:`PositionalFingerprint`).
2. **Complementarity** between fingerprints: do two models fail in *different* places, so
   that a blend could be better than either? (:func:`complementarity`).
3. **Global-alpha interpolation** -- one coefficient for the whole stack
   (:func:`global_alpha_search`).
4. **Layer-wise alpha** -- coordinate ascent over the broad blocks of
   :mod:`rineya_psa.merging.arithmetic` (:func:`layerwise_alpha_search`).
5. **Greedy position-aware soup** -- add donors one at a time, keeping a move only when an
   explicit acceptance predicate says so (:func:`greedy_position_aware_soup`).

The acceptance rule is the brief's: *improves worst-position accuracy or the overall
long-context score, without unacceptable regression elsewhere*. It is a first-class object
(:class:`AcceptanceRule`) that returns an :class:`AcceptanceDecision` listing every
criterion, its measured value, its threshold and whether it passed -- so a rejected merge
can be read off the record instead of re-derived from source. No acceptance logic is
written as an inline ``if`` anywhere in this module.

Orientation is handled once, in :class:`PositionalFingerprint`. SPEC section 13.1 makes
gold NLL the primary metric for every mechanistic phase, and NLL is lower-is-better while
accuracy is higher-is-better; the fingerprint carries ``higher_is_better`` and exposes
``oriented`` values where larger is always better, so the rule and the search objective are
single-signed and never have to ask which metric they are looking at.

Every search is driven by a caller-supplied ``evaluate(recipe) -> PositionalFingerprint``.
That keeps this module free of the GPU: the searches here are pure bookkeeping over
whatever the caller measures, and :func:`apply_soup_recipe` is the only function that
touches weights.

DEVIATION: the brief says "soup" without pinning the ingredient count. Recipes here are
``theta' = theta_base + sum_m alpha_(m,block) * Delta_m`` -- a *sum* of donor deltas with
per-block coefficients, not a normalised average. Two donors at alpha 1.0 therefore
extrapolate rather than average, which matches the alpha=1.2 row of
:data:`~rineya_psa.merging.arithmetic.ALPHA_GRID` being an extrapolation too.
``SoupRecipe(normalize=True)`` divides by the ingredient count for callers who want the
convex-hull behaviour, and ``describe()`` records which convention was used.

DEVIATION: acceptance during the greedy stage is judged against the **incumbent** (does
adding this donor improve what we already hold?), not against the base. The decision
against the base is computed as well and stored on the candidate as ``decision_vs_base``,
because the guard "no unacceptable regression elsewhere" is only meaningful relative to the
starting point. Both are in ``to_json()``.

DEVIATION: torch is imported lazily inside :func:`apply_soup_recipe` and its helper, as in
:mod:`rineya_psa.merging.arithmetic`; SPEC section 2 does not whitelist ``merging/`` for a
module-scope torch import, and everything else in this file is numpy-only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from rineya_psa.evaluation.metrics import (
    edge_middle_gap,
    position_robustness_index,
    ushape_fit,
)
from rineya_psa.merging.arithmetic import (
    ALPHA_GRID,
    DEFAULT_CHUNK_BYTES,
    BlockAlphas,
    LayerBlock,
)
from rineya_psa.types import UShapeFit

__all__ = [
    "AcceptanceDecision",
    "AcceptanceRule",
    "AppliedSoup",
    "ComplementarityReport",
    "Criterion",
    "PositionalFingerprint",
    "SoupCandidate",
    "SoupObjective",
    "SoupPipelineResult",
    "SoupRecipe",
    "SoupSearchResult",
    "apply_soup_recipe",
    "complementarity",
    "complementarity_matrix",
    "complementarity_table",
    "default_acceptance_rule",
    "fingerprint_from_curve",
    "fingerprint_from_records",
    "global_alpha_search",
    "greedy_position_aware_soup",
    "layerwise_alpha_search",
    "run_soup_pipeline",
]

#: Metric-name fragments whose scores are lower-is-better. SPEC section 13.1 promotes gold
#: NLL to the primary metric, so getting this wrong would invert every acceptance test.
_LOWER_IS_BETTER_HINTS: tuple[str, ...] = ("nll", "loss", "perplexity", "ppl", "error")

#: Position keys are rounded before grouping so 0.5 and 0.5000000001 are one grid point.
_POSITION_DECIMALS: int = 6


def _metric_is_higher_better(metric: str) -> bool:
    lowered = str(metric).lower()
    return not any(hint in lowered for hint in _LOWER_IS_BETTER_HINTS)


def _field(record: Any, name: str) -> Any:
    """Read ``name`` off a dataclass record or a mapping row, without caring which."""
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


# --------------------------------------------------------------------------------------
# 1. Positional fingerprints
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionalFingerprint:
    """``P_m = [A_m(p_1) ... A_m(p_k)]``: one model (or merge) reduced to its position curve.

    ``higher_is_better`` is the single place orientation is decided. Everything downstream
    -- the acceptance rule, the search objective, complementarity -- reads ``oriented``,
    where larger is always better, so no other code has to know whether it is holding
    accuracy or NLL.
    """

    label: str
    positions: tuple[float, ...]
    values: tuple[float, ...]
    metric: str = "accuracy"
    higher_is_better: bool = True
    support: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        positions = tuple(float(p) for p in self.positions)
        values = tuple(float(v) for v in self.values)
        if not positions:
            raise ValueError(f"{self.label}: a fingerprint needs at least one position")
        if len(positions) != len(values):
            raise ValueError(
                f"{self.label}: {len(positions)} positions but {len(values)} values"
            )
        if list(positions) != sorted(positions):
            raise ValueError(f"{self.label}: positions must be sorted ascending")
        if len(set(positions)) != len(positions):
            raise ValueError(f"{self.label}: duplicate positions {positions}")
        if any(not math.isfinite(v) for v in values):
            raise ValueError(f"{self.label}: non-finite value in {values}")
        support = tuple(int(s) for s in self.support)
        if support and len(support) != len(positions):
            raise ValueError(f"{self.label}: support must be empty or one count per position")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "support", support)

    # -- orientation ---------------------------------------------------------------

    @property
    def oriented(self) -> np.ndarray:
        """Values re-signed so that larger is better, whatever the metric is."""
        arr = np.asarray(self.values, dtype=np.float64)
        return arr if self.higher_is_better else -arr

    @property
    def worst_index(self) -> int:
        """Index of the weakest position; ties resolve to the smaller position."""
        return int(np.argmin(self.oriented))

    @property
    def worst(self) -> float:
        """The raw metric value at the weakest position (readable, not re-signed)."""
        return float(self.values[self.worst_index])

    @property
    def worst_at(self) -> float:
        return float(self.positions[self.worst_index])

    @property
    def oriented_worst(self) -> float:
        return float(self.oriented.min())

    @property
    def overall(self) -> float:
        """Unweighted mean over the grid -- the 'overall long-context score'."""
        return float(np.mean(self.values))

    @property
    def oriented_overall(self) -> float:
        return float(self.oriented.mean())

    # -- shape ---------------------------------------------------------------------

    def as_mapping(self) -> dict[float, float]:
        return dict(zip(self.positions, self.values, strict=True))

    def edge_middle_gap(self) -> float:
        return edge_middle_gap(self.as_mapping())

    def robustness(self) -> float | None:
        """Position-robustness index, or ``None`` when the metric is not an accuracy.

        ``metrics.position_robustness_index`` is a ratio and is only bounded on [0, 1]
        scores; returning ``None`` rather than raising keeps ``describe()`` usable for an
        NLL fingerprint, where the quantity simply does not exist.
        """
        if not self.higher_is_better:
            return None
        arr = np.asarray(self.values, dtype=np.float64)
        if arr.min() < 0.0 or arr.max() > 1.0:
            return None
        return position_robustness_index(self.as_mapping())

    def ushape(self) -> UShapeFit:
        return ushape_fit(self.positions, self.values)

    # -- comparison ----------------------------------------------------------------

    def oriented_delta(self, other: PositionalFingerprint) -> np.ndarray:
        """``self - other`` per position, in oriented units (positive = self is better)."""
        self._require_same_grid(other)
        return self.oriented - other.oriented

    def _require_same_grid(self, other: PositionalFingerprint) -> None:
        if len(self.positions) != len(other.positions) or not np.allclose(
            self.positions, other.positions, atol=1e-9
        ):
            raise ValueError(
                f"{self.label} and {other.label} are on different position grids: "
                f"{self.positions} vs {other.positions}"
            )
        if self.higher_is_better != other.higher_is_better:
            raise ValueError(
                f"{self.label} and {other.label} disagree on metric orientation; "
                "comparing them would flip the sign of every conclusion"
            )

    def relabel(self, label: str) -> PositionalFingerprint:
        return PositionalFingerprint(
            label=label,
            positions=self.positions,
            values=self.values,
            metric=self.metric,
            higher_is_better=self.higher_is_better,
            support=self.support,
        )

    # -- serialisation -------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "positions": list(self.positions),
            "values": list(self.values),
            "support": list(self.support),
        }

    def describe(self) -> dict[str, Any]:
        out = self.to_json()
        out.update(
            {
                "overall": self.overall,
                "worst": self.worst,
                "worst_at": self.worst_at,
                "edge_middle_gap": self.edge_middle_gap(),
                "robustness": self.robustness(),
            }
        )
        return out


def fingerprint_from_curve(
    label: str,
    acc_by_pos: Mapping[float, float],
    *,
    metric: str = "accuracy",
    higher_is_better: bool | None = None,
) -> PositionalFingerprint:
    """Build a fingerprint from an already-aggregated ``{position: score}`` curve."""
    if not acc_by_pos:
        raise ValueError(f"{label}: empty positional curve")
    positions = tuple(sorted(float(p) for p in acc_by_pos))
    values = tuple(float(acc_by_pos[p]) for p in positions)
    return PositionalFingerprint(
        label=label,
        positions=positions,
        values=values,
        metric=metric,
        higher_is_better=(
            _metric_is_higher_better(metric) if higher_is_better is None else bool(higher_is_better)
        ),
    )


def fingerprint_from_records(
    records: Iterable[Any],
    *,
    label: str,
    metric: str = "correct",
    higher_is_better: bool | None = None,
    position_field: str = "target_positions",
    position_reducer: Callable[[Sequence[float]], float] | None = None,
    min_support: int = 1,
) -> PositionalFingerprint:
    """Aggregate ``runner.RunRecord`` rows (or plain dicts) into a positional fingerprint.

    Grouping is on the **requested** position, not the measured one: ``actual_positions``
    is continuous and would give one group per item. Multi-hop items carry two targets, so
    ``position_reducer`` decides which one indexes the row; the default takes the first,
    which is the position the item's construction varied.

    Rows whose metric is ``None`` -- an accuracy field on a teacher-forced-only run, say --
    are dropped rather than counted as zero, and the surviving count per position is
    recorded in ``support`` so a thin cell is visible instead of implied.
    """
    reducer = position_reducer or (lambda p: float(p[0]))
    buckets: dict[float, list[float]] = {}
    for record in records:
        value = _field(record, metric)
        if value is None:
            continue
        raw = _field(record, position_field)
        if raw is None:
            raise ValueError(f"{label}: record has no {position_field!r} to group on")
        targets = [float(x) for x in raw] if isinstance(raw, (list, tuple)) else [float(raw)]
        if not targets:
            raise ValueError(f"{label}: record has an empty {position_field!r}")
        key = round(float(reducer(targets)), _POSITION_DECIMALS)
        buckets.setdefault(key, []).append(float(value))

    kept = {p: v for p, v in buckets.items() if len(v) >= min_support}
    if not kept:
        raise ValueError(
            f"{label}: no position had at least {min_support} usable {metric!r} values"
        )
    positions = tuple(sorted(kept))
    return PositionalFingerprint(
        label=label,
        positions=positions,
        values=tuple(float(np.mean(kept[p])) for p in positions),
        metric=metric,
        higher_is_better=(
            _metric_is_higher_better(metric) if higher_is_better is None else bool(higher_is_better)
        ),
        support=tuple(len(kept[p]) for p in positions),
    )


# --------------------------------------------------------------------------------------
# 2. Complementarity
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ComplementarityReport:
    """Whether two fingerprints fail in *different* places, and what a blend could buy.

    ``oracle_gain`` is the headline: the mean of the per-position best minus the best
    single model's mean. It is an upper bound on what any position-independent blend can
    achieve, so a near-zero value is a reason not to spend GPU hours on a merge sweep.
    """

    label_a: str
    label_b: str
    metric: str
    positions: tuple[float, ...]
    oriented_delta: tuple[float, ...]
    correlation: float
    crossovers: int
    oracle_gain: float
    worst_gain: float
    best_by_position: tuple[str, ...]
    complementary: bool
    tolerance: float

    def to_json(self) -> dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "metric": self.metric,
            "positions": list(self.positions),
            "oriented_delta": list(self.oriented_delta),
            "correlation": self.correlation,
            "crossovers": self.crossovers,
            "oracle_gain": self.oracle_gain,
            "worst_gain": self.worst_gain,
            "best_by_position": list(self.best_by_position),
            "complementary": self.complementary,
            "tolerance": self.tolerance,
        }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, or NaN when either side is constant (r is undefined, not zero)."""
    if x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def complementarity(
    a: PositionalFingerprint, b: PositionalFingerprint, *, tolerance: float = 0.0
) -> ComplementarityReport:
    """Per-position comparison of two fingerprints on a shared grid.

    ``complementary`` is deliberately conservative: it needs both a *crossover* (each model
    wins somewhere) and an oracle gain above ``tolerance``. A model that dominates
    everywhere is not complementary no matter how large the average gap is.
    """
    a._require_same_grid(b)
    oa, ob = a.oriented, b.oriented
    delta = oa - ob
    signs = np.sign(delta[np.abs(delta) > tolerance])
    crossovers = int(np.count_nonzero(np.diff(signs))) if signs.size > 1 else 0

    oracle = np.maximum(oa, ob)
    oracle_gain = float(oracle.mean() - max(oa.mean(), ob.mean()))
    worst_gain = float(oracle.min() - max(oa.min(), ob.min()))
    best = tuple(a.label if d >= 0 else b.label for d in delta)

    return ComplementarityReport(
        label_a=a.label,
        label_b=b.label,
        metric=a.metric,
        positions=a.positions,
        oriented_delta=tuple(float(d) for d in delta),
        correlation=_pearson(oa, ob),
        crossovers=crossovers,
        oracle_gain=oracle_gain,
        worst_gain=worst_gain,
        best_by_position=best,
        complementary=bool(crossovers >= 1 and oracle_gain > tolerance),
        tolerance=float(tolerance),
    )


def complementarity_table(
    fingerprints: Sequence[PositionalFingerprint], *, tolerance: float = 0.0
) -> list[dict[str, Any]]:
    """One row per unordered pair; tidy enough to hand straight to ``pandas.DataFrame``."""
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(fingerprints):
        for b in fingerprints[i + 1 :]:
            rows.append(complementarity(a, b, tolerance=tolerance).to_json())
    return rows


def complementarity_matrix(
    fingerprints: Sequence[PositionalFingerprint], *, tolerance: float = 0.0
) -> tuple[tuple[str, ...], np.ndarray]:
    """``(labels, oracle_gain_matrix)``. Symmetric, zero on the diagonal by construction."""
    labels = tuple(fp.label for fp in fingerprints)
    n = len(labels)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            gain = complementarity(fingerprints[i], fingerprints[j], tolerance=tolerance)
            matrix[i, j] = matrix[j, i] = gain.oracle_gain
    return labels, matrix


# --------------------------------------------------------------------------------------
# Acceptance: an inspectable predicate, not an inline if
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Criterion:
    """One clause of the acceptance rule, with the number that decided it."""

    name: str
    role: str  # "improvement" | "guard"
    measured: float
    threshold: float
    comparison: str  # ">" | ">=" | "<="
    passed: bool
    description: str

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "measured": self.measured,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "passed": self.passed,
            "description": self.description,
        }


_COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    ">": lambda m, t: m > t,
    ">=": lambda m, t: m >= t,
    "<=": lambda m, t: m <= t,
    "<": lambda m, t: m < t,
}


def _criterion(
    name: str,
    role: str,
    measured: float,
    threshold: float,
    comparison: str,
    description: str,
) -> Criterion:
    compare = _COMPARATORS[comparison]
    return Criterion(
        name=name,
        role=role,
        measured=float(measured),
        threshold=float(threshold),
        comparison=comparison,
        passed=bool(compare(float(measured), float(threshold))),
        description=description,
    )


@dataclass(frozen=True)
class AcceptanceDecision:
    """The full evaluation of an :class:`AcceptanceRule` against one candidate."""

    accepted: bool
    rule_name: str
    require: str
    baseline_label: str
    candidate_label: str
    criteria: tuple[Criterion, ...]
    improvements_met: tuple[str, ...]
    guards_failed: tuple[str, ...]
    reason: str

    def criterion(self, name: str) -> Criterion:
        for c in self.criteria:
            if c.name == name:
                return c
        raise KeyError(f"no criterion named {name!r}; have {[c.name for c in self.criteria]}")

    def to_json(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rule": self.rule_name,
            "require": self.require,
            "baseline": self.baseline_label,
            "candidate": self.candidate_label,
            "criteria": [c.to_json() for c in self.criteria],
            "improvements_met": list(self.improvements_met),
            "guards_failed": list(self.guards_failed),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AcceptanceRule:
    """The brief's section 35 rule, as data.

    *Accept when the candidate improves worst-position score **or** the overall
    long-context score, and regresses nothing else unacceptably.*

    Every threshold is a field, every clause becomes a :class:`Criterion` in the returned
    decision, and ``describe()`` round-trips the whole rule into the provenance record --
    so "why was this merge rejected" is answered by reading the artifact.

    Thresholds are in **oriented** units (larger is better), which for an NLL fingerprint
    means a gain of 0.05 is 0.05 nats of improvement.
    """

    min_worst_gain: float = 0.0
    min_overall_gain: float = 0.0
    max_position_regression: float = 0.02
    max_overall_regression: float = 0.01
    min_preservation: float | None = None
    require: str = "any"
    name: str = "brief_s35_position_aware"

    def __post_init__(self) -> None:
        if self.require not in {"any", "all"}:
            raise ValueError(f"require must be 'any' or 'all', got {self.require!r}")
        for field_name in ("max_position_regression", "max_overall_regression"):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} is a tolerance and must be non-negative")

    # -- the predicate ---------------------------------------------------------------

    def evaluate(
        self,
        baseline: PositionalFingerprint,
        candidate: PositionalFingerprint,
        *,
        preservation: float | None = None,
    ) -> AcceptanceDecision:
        delta = candidate.oriented_delta(baseline)
        worst_gain = candidate.oriented_worst - baseline.oriented_worst
        overall_gain = candidate.oriented_overall - baseline.oriented_overall
        position_regression = float(max(0.0, -delta.min()))
        overall_regression = float(max(0.0, -overall_gain))

        criteria: list[Criterion] = [
            _criterion(
                "worst_position_gain",
                "improvement",
                worst_gain,
                self.min_worst_gain,
                ">",
                "worst-position score improves (the brief's first acceptance route)",
            ),
            _criterion(
                "overall_gain",
                "improvement",
                overall_gain,
                self.min_overall_gain,
                ">",
                "mean score over the position grid improves (the second route)",
            ),
            _criterion(
                "position_regression",
                "guard",
                position_regression,
                self.max_position_regression,
                "<=",
                "largest single-position loss stays inside tolerance",
            ),
            _criterion(
                "overall_regression",
                "guard",
                overall_regression,
                self.max_overall_regression,
                "<=",
                "mean score does not fall by more than tolerance",
            ),
        ]
        if self.min_preservation is not None:
            criteria.append(
                _criterion(
                    "preservation",
                    "guard",
                    float("nan") if preservation is None else preservation,
                    self.min_preservation,
                    ">=",
                    "general-capability preservation score stays above the floor",
                )
            )

        improvements = tuple(c.name for c in criteria if c.role == "improvement" and c.passed)
        n_improvements = sum(1 for c in criteria if c.role == "improvement")
        guards_failed = tuple(c.name for c in criteria if c.role == "guard" and not c.passed)

        improved = (
            len(improvements) >= 1 if self.require == "any" else len(improvements) == n_improvements
        )
        accepted = bool(improved and not guards_failed)
        return AcceptanceDecision(
            accepted=accepted,
            rule_name=self.name,
            require=self.require,
            baseline_label=baseline.label,
            candidate_label=candidate.label,
            criteria=tuple(criteria),
            improvements_met=improvements,
            guards_failed=guards_failed,
            reason=_acceptance_reason(accepted, improvements, guards_failed, self.require),
        )

    __call__ = evaluate

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "require": self.require,
            "min_worst_gain": self.min_worst_gain,
            "min_overall_gain": self.min_overall_gain,
            "max_position_regression": self.max_position_regression,
            "max_overall_regression": self.max_overall_regression,
            "min_preservation": self.min_preservation,
            "statement": (
                "accept when worst-position or overall score improves and no guard is "
                "breached"
            ),
        }


def _acceptance_reason(
    accepted: bool,
    improvements: Sequence[str],
    guards_failed: Sequence[str],
    require: str,
) -> str:
    if accepted:
        return f"improved {', '.join(improvements)}; all guards held"
    if guards_failed:
        return f"guard(s) breached: {', '.join(guards_failed)}"
    if require == "all":
        return "not every improvement criterion was met"
    return "neither worst-position nor overall score improved"


def default_acceptance_rule(metric: str = "accuracy") -> AcceptanceRule:
    """The rule the phase-8 scripts use, with tolerances scaled to the metric.

    Accuracy lives on [0, 1] so a two-point tolerance is 0.02; gold NLL is measured in nats
    with an edge-middle gap of ~0.27 (SPEC section 13.1), so the equivalent tolerance is an
    order of magnitude larger in absolute terms.
    """
    if _metric_is_higher_better(metric):
        return AcceptanceRule(name=f"brief_s35_{metric}")
    return AcceptanceRule(
        max_position_regression=0.05,
        max_overall_regression=0.02,
        name=f"brief_s35_{metric}",
    )


# --------------------------------------------------------------------------------------
# Recipes, objectives, candidates
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SoupRecipe:
    """``theta' = theta_base + sum_m alpha_(m,block) * Delta_m`` -- a soup as data.

    Immutable: ``with_ingredient`` and ``without_ingredient`` return new recipes, so a
    search history is a list of recipes that can be replayed rather than a mutated object.
    """

    base_key: str
    alphas: Mapping[str, BlockAlphas] = field(default_factory=dict)
    normalize: bool = False

    def __post_init__(self) -> None:
        resolved: dict[str, BlockAlphas] = {}
        for key, value in dict(self.alphas).items():
            if not isinstance(value, BlockAlphas):
                raise TypeError(f"{key}: expected BlockAlphas, got {type(value).__name__}")
            resolved[str(key)] = value
        if str(self.base_key) in resolved:
            raise ValueError(f"{self.base_key!r} is the base and cannot also be an ingredient")
        object.__setattr__(self, "alphas", MappingProxyType(resolved))

    @property
    def ingredients(self) -> tuple[str, ...]:
        return tuple(self.alphas)

    @property
    def n_ingredients(self) -> int:
        return len(self.alphas)

    def is_base(self) -> bool:
        """True when the recipe provably reproduces the base checkpoint."""
        return all(ba.is_identity() for ba in self.alphas.values())

    def effective_alphas(self) -> dict[str, BlockAlphas]:
        """Coefficients as actually applied, after the ``normalize`` convention."""
        if not self.normalize or not self.alphas:
            return dict(self.alphas)
        n = float(len(self.alphas))
        return {
            key: BlockAlphas(
                blocks=ba.blocks, alphas={name: a / n for name, a in ba.alphas.items()}
            )
            for key, ba in self.alphas.items()
        }

    def with_ingredient(self, key: str, alphas: BlockAlphas) -> SoupRecipe:
        updated = dict(self.alphas)
        updated[str(key)] = alphas
        return SoupRecipe(base_key=self.base_key, alphas=updated, normalize=self.normalize)

    def without_ingredient(self, key: str) -> SoupRecipe:
        updated = dict(self.alphas)
        updated.pop(str(key), None)
        return SoupRecipe(base_key=self.base_key, alphas=updated, normalize=self.normalize)

    def slug(self) -> str:
        """Deterministic, filesystem-safe, and the cache key for repeated evaluation."""
        if not self.alphas:
            return f"{self.base_key}-base"
        parts = [f"{key}_{self.alphas[key].slug()}" for key in sorted(self.alphas)]
        tag = "-norm" if self.normalize else ""
        return f"{self.base_key}+{'+'.join(parts)}{tag}"

    def describe(self) -> dict[str, Any]:
        return {
            "base_key": self.base_key,
            "normalize": self.normalize,
            "ingredients": list(self.ingredients),
            "alphas": {key: ba.describe() for key, ba in self.alphas.items()},
            "effective_alphas": {
                key: dict(ba.alphas) for key, ba in self.effective_alphas().items()
            },
            "slug": self.slug(),
            "is_base": self.is_base(),
        }

    to_json = describe


@dataclass(frozen=True)
class SoupObjective:
    """Scalar the searches maximise. Position-aware because ``worst_weight`` is not zero.

    ``score = worst_weight * oriented_worst + overall_weight * oriented_overall``. With the
    default 1/1 a merge that lifts the worst position by 0.1 is worth as much as one that
    lifts every position by 0.1, which is the trade the brief's section 35 asks for.
    """

    worst_weight: float = 1.0
    overall_weight: float = 1.0
    name: str = "worst_plus_overall"

    def score(self, fingerprint: PositionalFingerprint) -> float:
        return float(
            self.worst_weight * fingerprint.oriented_worst
            + self.overall_weight * fingerprint.oriented_overall
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "worst_weight": self.worst_weight,
            "overall_weight": self.overall_weight,
        }


@dataclass(frozen=True)
class SoupCandidate:
    """One evaluated recipe, its fingerprint, and why it was kept or dropped."""

    stage: str
    recipe: SoupRecipe
    fingerprint: PositionalFingerprint
    score: float
    decision: AcceptanceDecision | None = None
    decision_vs_base: AcceptanceDecision | None = None
    selected: bool = False

    @property
    def accepted(self) -> bool:
        return self.decision is None or self.decision.accepted

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "recipe": self.recipe.describe(),
            "fingerprint": self.fingerprint.describe(),
            "score": self.score,
            "selected": self.selected,
            "decision": None if self.decision is None else self.decision.to_json(),
            "decision_vs_base": (
                None if self.decision_vs_base is None else self.decision_vs_base.to_json()
            ),
        }


@dataclass(frozen=True)
class SoupSearchResult:
    """Outcome of one search stage: the winner, the trail, and the settings that produced it."""

    stage: str
    baseline: PositionalFingerprint
    best: SoupCandidate
    history: tuple[SoupCandidate, ...]
    rule: AcceptanceRule
    objective: SoupObjective
    n_evaluations: int

    @property
    def improved(self) -> bool:
        """True when the stage moved off its baseline recipe."""
        return not self.best.recipe.is_base() or self.best.score > self.objective.score(
            self.baseline
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "baseline": self.baseline.describe(),
            "best": self.best.to_json(),
            "history": [c.to_json() for c in self.history],
            "rule": self.rule.describe(),
            "objective": self.objective.describe(),
            "n_evaluations": self.n_evaluations,
        }


@dataclass(frozen=True)
class SoupPipelineResult:
    """The whole section-35 chain: fingerprints, complementarity, three search stages."""

    baseline: PositionalFingerprint
    fingerprints: tuple[PositionalFingerprint, ...]
    complementarity: tuple[dict[str, Any], ...]
    global_stage: tuple[SoupSearchResult, ...]
    layerwise_stage: tuple[SoupSearchResult, ...]
    greedy_stage: SoupSearchResult
    best_recipe: SoupRecipe
    best_fingerprint: PositionalFingerprint
    rule: AcceptanceRule
    objective: SoupObjective
    n_evaluations: int

    def to_json(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.describe(),
            "fingerprints": [fp.describe() for fp in self.fingerprints],
            "complementarity": [dict(row) for row in self.complementarity],
            "global_stage": [r.to_json() for r in self.global_stage],
            "layerwise_stage": [r.to_json() for r in self.layerwise_stage],
            "greedy_stage": self.greedy_stage.to_json(),
            "best_recipe": self.best_recipe.describe(),
            "best_fingerprint": self.best_fingerprint.describe(),
            "rule": self.rule.describe(),
            "objective": self.objective.describe(),
            "n_evaluations": self.n_evaluations,
        }


# --------------------------------------------------------------------------------------
# Evaluation plumbing
# --------------------------------------------------------------------------------------

EvaluateFn = Callable[[SoupRecipe], PositionalFingerprint]


class _Evaluator:
    """Memoised ``evaluate`` keyed on the recipe slug.

    Coordinate ascent revisits the incumbent constantly and every evaluation is a full
    benchmark pass; caching turns a stage from quadratic to linear in GPU time, and the hit
    count is reported so a caller can see how much was saved.
    """

    def __init__(self, evaluate: EvaluateFn) -> None:
        self._evaluate = evaluate
        self._cache: dict[str, PositionalFingerprint] = {}
        self.n_calls = 0
        self.n_hits = 0

    def __call__(self, recipe: SoupRecipe) -> PositionalFingerprint:
        key = recipe.slug()
        cached = self._cache.get(key)
        if cached is not None:
            self.n_hits += 1
            return cached
        fingerprint = self._evaluate(recipe)
        if not isinstance(fingerprint, PositionalFingerprint):
            raise TypeError(
                "evaluate must return a PositionalFingerprint, got "
                f"{type(fingerprint).__name__}"
            )
        self.n_calls += 1
        self._cache[key] = fingerprint
        return fingerprint


def _zero_alphas(blocks: Sequence[LayerBlock]) -> BlockAlphas:
    return BlockAlphas.uniform(blocks, 0.0)


def _base_recipe(base_key: str, *, normalize: bool = False) -> SoupRecipe:
    return SoupRecipe(base_key=base_key, alphas={}, normalize=normalize)


def _candidate(
    stage: str,
    recipe: SoupRecipe,
    fingerprint: PositionalFingerprint,
    objective: SoupObjective,
    *,
    decision: AcceptanceDecision | None = None,
    decision_vs_base: AcceptanceDecision | None = None,
    selected: bool = False,
) -> SoupCandidate:
    return SoupCandidate(
        stage=stage,
        recipe=recipe,
        fingerprint=fingerprint,
        score=objective.score(fingerprint),
        decision=decision,
        decision_vs_base=decision_vs_base,
        selected=selected,
    )


def _select(candidates: Sequence[SoupCandidate], incumbent: SoupCandidate) -> SoupCandidate:
    """Highest-scoring *accepted* candidate, or the incumbent when none is accepted."""
    accepted = [c for c in candidates if c.accepted and c.score > incumbent.score]
    if not accepted:
        return incumbent
    return max(accepted, key=lambda c: c.score)


def _mark_selected(
    history: Sequence[SoupCandidate], winner: SoupCandidate
) -> tuple[SoupCandidate, ...]:
    return tuple(
        SoupCandidate(
            stage=c.stage,
            recipe=c.recipe,
            fingerprint=c.fingerprint,
            score=c.score,
            decision=c.decision,
            decision_vs_base=c.decision_vs_base,
            selected=c is winner,
        )
        for c in history
    )


# --------------------------------------------------------------------------------------
# 3. Global alpha
# --------------------------------------------------------------------------------------


def global_alpha_search(
    evaluate: EvaluateFn,
    ingredient: str,
    blocks: Sequence[LayerBlock],
    *,
    base_key: str = "base",
    grid: Sequence[float] = ALPHA_GRID,
    rule: AcceptanceRule | None = None,
    objective: SoupObjective | None = None,
    baseline: PositionalFingerprint | None = None,
    normalize: bool = False,
) -> SoupSearchResult:
    """One coefficient for the whole stack: plain task-arithmetic interpolation.

    ``alpha = 0`` is the base and is used as the baseline fingerprint when the caller does
    not supply one; it is evaluated rather than assumed, so a non-deterministic harness
    shows up as a non-zero baseline drift instead of silently biasing every comparison.
    """
    rule = rule or default_acceptance_rule()
    objective = objective or SoupObjective()
    ev = _Evaluator(evaluate)

    base = _base_recipe(base_key, normalize=normalize)
    baseline = baseline if baseline is not None else ev(base)
    incumbent = _candidate("global_alpha", base, baseline, objective)

    history: list[SoupCandidate] = [incumbent]
    for alpha in grid:
        alphas = BlockAlphas.uniform(blocks, float(alpha))
        if alphas.is_identity():
            continue  # alpha = 0 *is* the baseline; re-measuring it would double-count it
        recipe = base.with_ingredient(ingredient, alphas)
        fingerprint = ev(recipe)
        history.append(
            _candidate(
                "global_alpha",
                recipe,
                fingerprint,
                objective,
                decision=rule.evaluate(baseline, fingerprint),
            )
        )

    winner = _select(history[1:], incumbent)
    return SoupSearchResult(
        stage="global_alpha",
        baseline=baseline,
        best=winner,
        history=_mark_selected(history, winner),
        rule=rule,
        objective=objective,
        n_evaluations=ev.n_calls,
    )


# --------------------------------------------------------------------------------------
# 4. Layer-wise alpha
# --------------------------------------------------------------------------------------


def layerwise_alpha_search(
    evaluate: EvaluateFn,
    ingredient: str,
    blocks: Sequence[LayerBlock],
    *,
    start: BlockAlphas | None = None,
    base_key: str = "base",
    grid: Sequence[float] = ALPHA_GRID,
    rule: AcceptanceRule | None = None,
    objective: SoupObjective | None = None,
    baseline: PositionalFingerprint | None = None,
    rounds: int = 2,
    normalize: bool = False,
) -> SoupSearchResult:
    """Coordinate ascent over the broad blocks, one block moved at a time.

    Coordinate ascent rather than the full ``|grid|^|blocks|`` product: four blocks by four
    alphas is 256 full benchmark passes, and ``arithmetic.full_block_grid`` refuses that
    scale for the same reason. Two rounds is usually enough -- the second round is what
    catches a block whose best alpha depends on a block fixed later in the first.
    """
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    rule = rule or default_acceptance_rule()
    objective = objective or SoupObjective()
    ev = _Evaluator(evaluate)

    base = _base_recipe(base_key, normalize=normalize)
    baseline = baseline if baseline is not None else ev(base)

    current = start if start is not None else _zero_alphas(blocks)
    incumbent_recipe = (
        base if current.is_identity() else base.with_ingredient(ingredient, current)
    )
    incumbent_fp = ev(incumbent_recipe)
    incumbent = _candidate(
        "layerwise_alpha",
        incumbent_recipe,
        incumbent_fp,
        objective,
        # The starting point is not a move, so it carries no verdict; giving the base a
        # decision would record it as "rejected" for failing to improve on itself.
        decision=None
        if incumbent_recipe.is_base()
        else rule.evaluate(baseline, incumbent_fp),
    )
    history: list[SoupCandidate] = [incumbent]

    for _ in range(rounds):
        improved_this_round = False
        for block in blocks:
            for alpha in grid:
                if current.alphas[block.name] == float(alpha):
                    continue
                trial = current.with_alpha(block.name, float(alpha))
                recipe = base if trial.is_identity() else base.with_ingredient(ingredient, trial)
                fingerprint = ev(recipe)
                candidate = _candidate(
                    "layerwise_alpha",
                    recipe,
                    fingerprint,
                    objective,
                    decision=rule.evaluate(baseline, fingerprint),
                )
                history.append(candidate)
            round_best = _select(history, incumbent)
            if round_best is not incumbent:
                incumbent = round_best
                current = _alphas_of(round_best.recipe, ingredient, blocks)
                improved_this_round = True
        if not improved_this_round:
            break

    return SoupSearchResult(
        stage="layerwise_alpha",
        baseline=baseline,
        best=incumbent,
        history=_mark_selected(history, incumbent),
        rule=rule,
        objective=objective,
        n_evaluations=ev.n_calls,
    )


def _alphas_of(
    recipe: SoupRecipe, ingredient: str, blocks: Sequence[LayerBlock]
) -> BlockAlphas:
    return recipe.alphas.get(ingredient) or _zero_alphas(blocks)


# --------------------------------------------------------------------------------------
# 5. Greedy position-aware soup
# --------------------------------------------------------------------------------------


def greedy_position_aware_soup(
    evaluate: EvaluateFn,
    ingredient_alphas: Mapping[str, BlockAlphas],
    *,
    base_key: str = "base",
    rule: AcceptanceRule | None = None,
    objective: SoupObjective | None = None,
    baseline: PositionalFingerprint | None = None,
    max_ingredients: int | None = None,
    normalize: bool = False,
    preservation: Mapping[str, float] | None = None,
) -> SoupSearchResult:
    """Add donors one at a time, keeping only moves the acceptance rule allows.

    Ordering is by each donor's *standalone* objective score, so the donor that most helps
    the worst position is offered first -- that is what makes the soup position-aware rather
    than a plain greedy average. Each addition is judged against the incumbent (would this
    donor improve the soup we already hold?) while ``decision_vs_base`` records the same
    verdict against the base, because "no unacceptable regression elsewhere" is a statement
    about the starting point.

    ``preservation`` supplies the general-capability score per candidate slug for rules that
    set ``min_preservation``; an absent entry makes the guard fail rather than pass, since a
    missing measurement is not evidence of preservation.
    """
    rule = rule or default_acceptance_rule()
    objective = objective or SoupObjective()
    ev = _Evaluator(evaluate)
    preservation = dict(preservation or {})

    base = _base_recipe(base_key, normalize=normalize)
    baseline = baseline if baseline is not None else ev(base)
    incumbent = _candidate("greedy_soup", base, baseline, objective, selected=True)
    history: list[SoupCandidate] = [incumbent]

    # Standalone pass: score every donor on its own, and order the greedy offers by it.
    standalone: list[tuple[float, str]] = []
    for key, alphas in ingredient_alphas.items():
        recipe = base.with_ingredient(key, alphas)
        fingerprint = ev(recipe)
        candidate = _candidate(
            "greedy_soup_standalone",
            recipe,
            fingerprint,
            objective,
            decision=rule.evaluate(
                baseline, fingerprint, preservation=preservation.get(recipe.slug())
            ),
        )
        history.append(candidate)
        standalone.append((candidate.score, key))
    order = [key for _, key in sorted(standalone, key=lambda pair: (-pair[0], pair[1]))]

    budget = len(order) if max_ingredients is None else int(max_ingredients)
    for key in order:
        if incumbent.recipe.n_ingredients >= budget:
            break
        if key in incumbent.recipe.ingredients:
            continue
        recipe = incumbent.recipe.with_ingredient(key, ingredient_alphas[key])
        fingerprint = ev(recipe)
        pres = preservation.get(recipe.slug())
        candidate = _candidate(
            "greedy_soup",
            recipe,
            fingerprint,
            objective,
            decision=rule.evaluate(incumbent.fingerprint, fingerprint, preservation=pres),
            decision_vs_base=rule.evaluate(baseline, fingerprint, preservation=pres),
        )
        history.append(candidate)
        if candidate.accepted and candidate.score > incumbent.score:
            incumbent = candidate

    return SoupSearchResult(
        stage="greedy_soup",
        baseline=baseline,
        best=incumbent,
        history=_mark_selected(history, incumbent),
        rule=rule,
        objective=objective,
        n_evaluations=ev.n_calls,
    )


def run_soup_pipeline(
    evaluate: EvaluateFn,
    ingredients: Sequence[str],
    blocks: Sequence[LayerBlock],
    *,
    base_key: str = "base",
    grid: Sequence[float] = ALPHA_GRID,
    rule: AcceptanceRule | None = None,
    objective: SoupObjective | None = None,
    donor_fingerprints: Sequence[PositionalFingerprint] = (),
    rounds: int = 2,
    max_ingredients: int | None = None,
    normalize: bool = False,
) -> SoupPipelineResult:
    """Sections 35's whole chain in one call: global alpha, then layer-wise, then greedy.

    Stages 3 and 4 run per ingredient and are independent; stage 5 combines their winners.
    A shared :class:`_Evaluator` is threaded through all of them, so the base recipe -- which
    every stage needs as its baseline -- is measured exactly once.

    Note that a sub-stage's ``n_evaluations`` counts the distinct recipes *that stage* asked
    for, some of which the shared cache answered; :attr:`SoupPipelineResult.n_evaluations` is
    the number that actually reached the caller's ``evaluate``, and it is the GPU cost.
    """
    rule = rule or default_acceptance_rule()
    objective = objective or SoupObjective()
    ev = _Evaluator(evaluate)

    base = _base_recipe(base_key, normalize=normalize)
    baseline = ev(base)

    global_stage: list[SoupSearchResult] = []
    layerwise_stage: list[SoupSearchResult] = []
    tuned: dict[str, BlockAlphas] = {}
    for ingredient in ingredients:
        g = global_alpha_search(
            ev,
            ingredient,
            blocks,
            base_key=base_key,
            grid=grid,
            rule=rule,
            objective=objective,
            baseline=baseline,
            normalize=normalize,
        )
        global_stage.append(g)
        start = _alphas_of(g.best.recipe, ingredient, blocks)
        lw = layerwise_alpha_search(
            ev,
            ingredient,
            blocks,
            start=start,
            base_key=base_key,
            grid=grid,
            rule=rule,
            objective=objective,
            baseline=baseline,
            rounds=rounds,
            normalize=normalize,
        )
        layerwise_stage.append(lw)
        best_alphas = _alphas_of(lw.best.recipe, ingredient, blocks)
        if not best_alphas.is_identity():
            tuned[ingredient] = best_alphas

    greedy = greedy_position_aware_soup(
        ev,
        tuned,
        base_key=base_key,
        rule=rule,
        objective=objective,
        baseline=baseline,
        max_ingredients=max_ingredients,
        normalize=normalize,
    )

    fingerprints = tuple(donor_fingerprints)
    table = tuple(complementarity_table(fingerprints)) if len(fingerprints) > 1 else ()
    return SoupPipelineResult(
        baseline=baseline,
        fingerprints=fingerprints,
        complementarity=table,
        global_stage=tuple(global_stage),
        layerwise_stage=tuple(layerwise_stage),
        greedy_stage=greedy,
        best_recipe=greedy.best.recipe,
        best_fingerprint=greedy.best.fingerprint,
        rule=rule,
        objective=objective,
        n_evaluations=ev.n_calls,
    )


# --------------------------------------------------------------------------------------
# Applying a recipe to a live model
# --------------------------------------------------------------------------------------


@dataclass
class AppliedSoup:
    """Handle for an in-place soup, and the means to undo it exactly.

    The mirror of ``arithmetic.AppliedMerge``, extended to more than one donor.
    ``restore()`` writes ``base_state`` back rather than replaying a saved copy, so it is
    exact and costs no extra memory. Usable as a context manager.
    """

    model: Any
    base_state: Mapping[str, Any]
    recipe: SoupRecipe
    touched: tuple[str, ...]
    skipped: tuple[str, ...]
    chunk_bytes: int = DEFAULT_CHUNK_BYTES
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        _write_soup_blend(
            self.model, self.base_state, {}, {}, self.touched, chunk_bytes=self.chunk_bytes
        )
        self.restored = True

    def __enter__(self) -> AppliedSoup:
        return self

    def __exit__(self, *exc: object) -> None:
        self.restore()

    def describe(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe.describe(),
            "n_touched": len(self.touched),
            "n_skipped": len(self.skipped),
            "skipped": list(self.skipped[:20]),
            "restored": self.restored,
        }


def _write_soup_blend(
    model: Any,
    base_state: Mapping[str, Any],
    deltas_by_ingredient: Mapping[str, Mapping[str, Any]],
    alphas_by_ingredient: Mapping[str, BlockAlphas],
    names: Sequence[str],
    *,
    chunk_bytes: int,
) -> None:
    """``param <- base + sum_m alpha_m(name) * delta_m[name]``, chunked, float32 accumulation.

    HELPER, not in a core module: ``arithmetic._write_blend`` accumulates one delta dict and
    the soup needs an arbitrary number, so the loop is re-implemented here rather than
    widening a signature another module already imports. Chunking matters for the same
    reason it does there -- ``embed_tokens`` is 622 MB in float32 on an 8.59 GB card.
    """
    import torch

    tensors = dict(model.state_dict(keep_vars=True))
    elements = max(1, chunk_bytes // 4)
    with torch.no_grad():
        for name in names:
            param = tensors[name]
            flat_p = param.reshape(-1)
            flat_b = base_state[name].reshape(-1)
            n = flat_p.numel()
            if flat_b.numel() != n:
                raise ValueError(
                    f"{name}: base snapshot has {flat_b.numel()} elements, model has {n}"
                )
            contributions = [
                (alphas_by_ingredient[key].for_tensor(name), deltas[name])
                for key, deltas in deltas_by_ingredient.items()
                if name in deltas and alphas_by_ingredient[key].for_tensor(name) != 0.0
            ]
            for start in range(0, n, elements):
                stop = min(start + elements, n)
                # copy=True is load-bearing: ``.to()`` is a no-op returning *self* when the
                # device and dtype already match, and add_ would then corrupt base_state.
                buf = flat_b[start:stop].to(
                    device=param.device, dtype=torch.float32, copy=bool(contributions)
                )
                for alpha, delta in contributions:
                    buf.add_(
                        delta.reshape(-1)[start:stop].to(
                            device=param.device, dtype=torch.float32
                        ),
                        alpha=float(alpha),
                    )
                flat_p[start:stop].copy_(buf)


def apply_soup_recipe(
    model: Any,
    base_state: Mapping[str, Any],
    deltas_by_ingredient: Mapping[str, Mapping[str, Any]],
    recipe: SoupRecipe,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> AppliedSoup:
    """Set ``model``'s weights to the recipe in place; ``AppliedSoup.restore()`` undoes it.

    The screening path, as in :func:`~rineya_psa.merging.arithmetic.apply_blockwise_delta`:
    nothing hits the disk. Every name in ``base_state`` is rewritten, including the ones
    with no delta, so the model's state afterwards depends only on
    ``(base_state, deltas, recipe)`` and not on what it held before -- which is what makes a
    sweep over dozens of recipes safe to run on one loaded model.
    """
    unknown = sorted(set(recipe.ingredients) - set(deltas_by_ingredient))
    if unknown:
        raise KeyError(
            f"recipe names ingredients with no delta dict: {unknown}; "
            f"have {sorted(deltas_by_ingredient)}"
        )
    tensors = dict(model.state_dict(keep_vars=True))
    names = [n for n in base_state if n in tensors]
    if not names:
        raise ValueError(
            "base_state shares no tensor names with the model; "
            "arithmetic.snapshot_state(model) is the intended source of base_state"
        )

    effective = recipe.effective_alphas()
    used = {key: deltas_by_ingredient[key] for key in recipe.ingredients}
    skipped = sorted(
        {n for deltas in used.values() for n in deltas if n not in tensors}
        | (set(base_state) - set(tensors))
    )
    _write_soup_blend(
        model, base_state, used, effective, names, chunk_bytes=chunk_bytes
    )
    return AppliedSoup(
        model=model,
        base_state=base_state,
        recipe=recipe,
        touched=tuple(names),
        skipped=tuple(skipped),
        chunk_bytes=chunk_bytes,
    )
