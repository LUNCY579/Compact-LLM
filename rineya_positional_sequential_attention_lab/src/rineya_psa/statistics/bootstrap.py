"""SPEC section 7: clustered paired bootstrap, BCa intervals, exact McNemar.

The unit of independence in this lab is the **scenario**, not the item: every positional
variant of one source example shares a ``scenario_id`` and they are strongly correlated.
Resampling items would therefore understate the standard error by roughly the square root
of the number of variants per scenario. Every routine here resamples *scenarios* and takes
all of their items, and the jackknife that feeds the BCa acceleration is likewise
leave-one-scenario-out.

Intervals are BCa (bias-corrected and accelerated), which is worth the jackknife on binary
accuracy differences where the bootstrap distribution is skewed and discrete. When BCa is
undefined -- a degenerate bias correction (every resample on one side of the estimate) or a
zero jackknife spread -- the routine falls back to the percentile interval and says so in
``result.method`` rather than silently reporting a different quantity.

Everything is seeded and the seed is returned in the result.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_N_RESAMPLES",
    "DEFAULT_SEED",
    "BootstrapResult",
    "InteractionContrastResult",
    "McNemarResult",
    "PairedBootstrapResult",
    "bootstrap_ci_of_statistic",
    "interaction_contrast",
    "mcnemar_exact",
    "paired_bootstrap",
    "summarize_paired_binary",
]

DEFAULT_N_RESAMPLES = 10_000
DEFAULT_SEED = 20260828

_NORMAL = NormalDist()
_CHUNK = 512  # resamples per vectorised block; keeps the index matrix small


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with a clustered bootstrap interval."""

    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    sd: float  # bootstrap standard deviation, i.e. the standard error of ``estimate``
    method: str
    n_resamples: int
    n_scenarios: int
    n_items: int
    seed: int

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedBootstrapResult(BootstrapResult):
    """``paired_bootstrap`` output: ``estimate`` is ``mean(b - a)`` over all items."""

    mean_a: float
    mean_b: float
    sd_diff: float
    cohens_dz: float
    cohens_dz_item: float


@dataclass(frozen=True)
class InteractionContrastResult(BootstrapResult):
    """Brief section 48 difference-in-differences with its two component effects."""

    seq_effect: float
    par_effect: float


@dataclass(frozen=True)
class McNemarResult:
    """Exact binomial McNemar over paired binary outcomes."""

    n_pairs: int
    a_only_correct: int
    b_only_correct: int
    discordant: int
    p_value: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------


def _cluster_index(
    n_items: int, cluster_ids: Sequence[Any] | None
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return ``(codes, members)``: each item's cluster code and each cluster's item indices.

    ``cluster_ids=None`` means every item is its own cluster, i.e. a plain item bootstrap.
    """
    if cluster_ids is None:
        codes = np.arange(n_items, dtype=np.int64)
        return codes, [np.array([i], dtype=np.int64) for i in range(n_items)]
    labels = np.asarray([str(c) for c in cluster_ids], dtype=object)
    if labels.shape[0] != n_items:
        raise ValueError(f"cluster_ids has {labels.shape[0]} entries for {n_items} items")
    unique, codes = np.unique(labels, return_inverse=True)
    members = [np.flatnonzero(codes == k) for k in range(unique.size)]
    return codes.astype(np.int64), members


def _bca_interval(
    theta_hat: float,
    boot: np.ndarray,
    jack: np.ndarray | None,
    ci: float,
) -> tuple[float, float, str]:
    """BCa endpoints, degrading to the percentile interval when BCa is undefined."""
    alpha = (1.0 - ci) / 2.0
    lo_pct = float(np.quantile(boot, alpha))
    hi_pct = float(np.quantile(boot, 1.0 - alpha))

    # Mid-p bias correction: ties are the norm for differences of binary accuracies.
    below = float(np.count_nonzero(boot < theta_hat))
    equal = float(np.count_nonzero(boot == theta_hat))
    p = (below + 0.5 * equal) / boot.size
    if not 0.0 < p < 1.0:
        return lo_pct, hi_pct, "percentile_degenerate_bias"
    z0 = _NORMAL.inv_cdf(p)

    acceleration = 0.0
    if jack is not None and jack.size >= 2:
        centred = jack.mean() - jack
        denom = 6.0 * float((centred**2).sum()) ** 1.5
        if denom > 0.0 and math.isfinite(denom):
            acceleration = float((centred**3).sum()) / denom
    if not math.isfinite(acceleration):
        acceleration = 0.0

    endpoints: list[float] = []
    for z in (_NORMAL.inv_cdf(alpha), _NORMAL.inv_cdf(1.0 - alpha)):
        adjusted = z0 + z
        denominator = 1.0 - acceleration * adjusted
        if denominator <= 0.0 or not math.isfinite(denominator):
            return lo_pct, hi_pct, "percentile_degenerate_acceleration"
        endpoints.append(_NORMAL.cdf(z0 + adjusted / denominator))
    lo_q = min(max(endpoints[0], 1e-6), 1.0 - 1e-6)
    hi_q = min(max(endpoints[1], 1e-6), 1.0 - 1e-6)
    if lo_q > hi_q:  # pragma: no cover - only reachable with an absurd acceleration
        lo_q, hi_q = hi_q, lo_q
    return float(np.quantile(boot, lo_q)), float(np.quantile(boot, hi_q)), "bca"


def _clustered_mean_bootstrap(
    diff: np.ndarray,
    members: Sequence[np.ndarray],
    *,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap and jackknife distributions of the item-weighted mean difference.

    Resampling draws ``K`` scenarios with replacement and keeps *all* items of each drawn
    scenario, so the ratio of sums below is exactly ``mean(diff over the resampled items)``.
    """
    sums = np.array([float(diff[m].sum()) for m in members], dtype=np.float64)
    counts = np.array([float(m.size) for m in members], dtype=np.float64)
    k = sums.size

    rng = np.random.default_rng(seed)
    boot = np.empty(n, dtype=np.float64)
    done = 0
    while done < n:
        block = min(_CHUNK, n - done)
        idx = rng.integers(0, k, size=(block, k))
        boot[done : done + block] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        done += block

    total_sum = sums.sum()
    total_count = counts.sum()
    remaining = total_count - counts
    with np.errstate(invalid="ignore", divide="ignore"):
        jack = np.where(remaining > 0, (total_sum - sums) / remaining, np.nan)
    jack = jack[np.isfinite(jack)]
    return boot, jack


def _as_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def paired_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    scenario_ids: Sequence[Any] | None = None,
    *,
    n: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    ci: float = 0.95,
) -> PairedBootstrapResult:
    """Paired mean difference ``b - a`` with a scenario-clustered BCa interval.

    ``cohens_dz`` is computed on the **scenario-level** mean differences, because those are
    the independent units; ``cohens_dz_item`` is the naive item-level version and is
    reported alongside so the inflation is visible rather than hidden.
    """
    a = _as_vector(values_a, "values_a")
    b = _as_vector(values_b, "values_b")
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must be the same length: {a.size} vs {b.size}")
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must lie in (0, 1)")

    diff = b - a
    _, members = _cluster_index(diff.size, scenario_ids)
    estimate = float(diff.mean())

    boot, jack = _clustered_mean_bootstrap(diff, members, n=n, seed=seed)
    lo, hi, method = _bca_interval(estimate, boot, jack, ci)

    cluster_means = np.array([float(diff[m].mean()) for m in members], dtype=np.float64)
    cluster_sd = float(cluster_means.std(ddof=1)) if cluster_means.size > 1 else 0.0
    item_sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    return PairedBootstrapResult(
        estimate=estimate,
        ci_low=lo,
        ci_high=hi,
        ci_level=float(ci),
        sd=float(boot.std(ddof=1)),
        method=method,
        n_resamples=int(n),
        n_scenarios=len(members),
        n_items=int(diff.size),
        seed=int(seed),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        sd_diff=item_sd,
        cohens_dz=float(cluster_means.mean() / cluster_sd) if cluster_sd > 0 else 0.0,
        cohens_dz_item=float(estimate / item_sd) if item_sd > 0 else 0.0,
    )


def mcnemar_exact(
    a_correct: Sequence[bool], b_correct: Sequence[bool]
) -> McNemarResult:
    """Two-sided exact (binomial) McNemar test for paired binary outcomes.

    Exact rather than chi-square because the discordant count is routinely small on a
    64-item showcase, where the asymptotic test is anti-conservative. Note that the exact
    test treats items as exchangeable; when several items share a scenario, read it
    alongside the clustered bootstrap rather than instead of it.
    """
    left = [bool(x) for x in a_correct]
    right = [bool(x) for x in b_correct]
    if not left or len(left) != len(right):
        raise ValueError("McNemar inputs must be non-empty and the same length")

    a_only = sum(1 for x, y in zip(left, right, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(left, right, strict=True) if y and not x)
    discordant = a_only + b_only

    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(a_only, b_only)
        # log-space binomial so a few thousand discordant pairs do not overflow.
        log_half = discordant * math.log(0.5)
        cumulative = sum(
            math.exp(
                math.lgamma(discordant + 1)
                - math.lgamma(k + 1)
                - math.lgamma(discordant - k + 1)
                + log_half
            )
            for k in range(tail + 1)
        )
        p_value = min(1.0, 2.0 * cumulative)

    return McNemarResult(
        n_pairs=len(left),
        a_only_correct=a_only,
        b_only_correct=b_only,
        discordant=discordant,
        p_value=float(p_value),
    )


def interaction_contrast(
    seq_sm: Sequence[float],
    seq: Sequence[float],
    par_sm: Sequence[float],
    par: Sequence[float],
    scenario_ids: Sequence[Any] | None = None,
    *,
    n: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    ci: float = 0.95,
) -> InteractionContrastResult:
    """``mean[(seq_sm - seq) - (par_sm - par)]`` with a scenario-clustered BCa interval.

    The brief's section 48 quantity: does the sequential-memory manipulation buy more on
    sequential items than the same manipulation buys on parallel items? All four vectors
    must be aligned item-for-item on the same scenarios.

    DEVIATION: the brief writes ``interaction_contrast(seq_sm, seq, par_sm, par)`` with no
    clustering argument, yet asks for a clustered CI. ``scenario_ids`` is therefore a fifth
    positional parameter; leaving it ``None`` degrades to an item-level bootstrap and is
    only correct when there is one item per scenario.
    """
    vectors = [
        _as_vector(seq_sm, "seq_sm"),
        _as_vector(seq, "seq"),
        _as_vector(par_sm, "par_sm"),
        _as_vector(par, "par"),
    ]
    sizes = {v.size for v in vectors}
    if len(sizes) != 1:
        raise ValueError(f"all four arms must be the same length, got sizes {sorted(sizes)}")
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must lie in (0, 1)")

    seq_delta = vectors[0] - vectors[1]
    par_delta = vectors[2] - vectors[3]
    diff = seq_delta - par_delta

    _, members = _cluster_index(diff.size, scenario_ids)
    estimate = float(diff.mean())
    boot, jack = _clustered_mean_bootstrap(diff, members, n=n, seed=seed)
    lo, hi, method = _bca_interval(estimate, boot, jack, ci)
    return InteractionContrastResult(
        estimate=estimate,
        ci_low=lo,
        ci_high=hi,
        ci_level=float(ci),
        sd=float(boot.std(ddof=1)),
        method=method,
        n_resamples=int(n),
        n_scenarios=len(members),
        n_items=int(diff.size),
        seed=int(seed),
        seq_effect=float(seq_delta.mean()),
        par_effect=float(par_delta.mean()),
    )


def bootstrap_ci_of_statistic(
    fn: Callable[[list[Any]], float],
    data: Sequence[Any],
    *,
    cluster_ids: Sequence[Any] | None = None,
    n: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    ci: float = 0.95,
) -> BootstrapResult:
    """Clustered BCa interval for any statistic of a record list.

    This is how derived metrics get intervals: ``edge_middle_gap`` and ``ushape_fit.beta2``
    are functions of a whole set of records, not per-item numbers, so they cannot go through
    :func:`paired_bootstrap`.

    ``fn`` receives a plain ``list`` of resampled elements and returns a float. It is called
    ``n + n_scenarios + 1`` times (bootstrap, leave-one-scenario-out jackknife for the BCa
    acceleration, and once on the full data), so a slow ``fn`` wants a smaller ``n``.
    Resamples where ``fn`` is undefined (it raises, or returns a non-finite value) are
    dropped and counted in ``method``.
    """
    rows = list(data)
    if not rows:
        raise ValueError("data must be non-empty")
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must lie in (0, 1)")

    _, members = _cluster_index(len(rows), cluster_ids)
    k = len(members)

    def evaluate(indices: np.ndarray) -> float:
        try:
            value = float(fn([rows[i] for i in indices]))
        except (ValueError, ZeroDivisionError, KeyError, IndexError):
            return float("nan")
        return value if math.isfinite(value) else float("nan")

    estimate = evaluate(np.arange(len(rows), dtype=np.int64))
    if not math.isfinite(estimate):
        raise ValueError("statistic is undefined on the full data")

    rng = np.random.default_rng(seed)
    raw = np.empty(n, dtype=np.float64)
    for i in range(n):
        drawn = rng.integers(0, k, size=k)
        raw[i] = evaluate(np.concatenate([members[j] for j in drawn]))
    boot = raw[np.isfinite(raw)]
    if boot.size < 2:
        raise ValueError(
            f"statistic was undefined on {n - boot.size} of {n} resamples; "
            "the metric needs a wider grid or more scenarios"
        )

    jack_values: list[float] = []
    if k >= 2:
        all_idx = np.arange(len(rows), dtype=np.int64)
        for j in range(k):
            keep = np.setdiff1d(all_idx, members[j], assume_unique=False)
            if keep.size:
                jack_values.append(evaluate(keep))
    jack = np.asarray([v for v in jack_values if math.isfinite(v)], dtype=np.float64)

    lo, hi, method = _bca_interval(estimate, boot, jack if jack.size else None, ci)
    if boot.size < n:
        method = f"{method}_partial_{boot.size}of{n}"
    return BootstrapResult(
        estimate=estimate,
        ci_low=lo,
        ci_high=hi,
        ci_level=float(ci),
        sd=float(boot.std(ddof=1)),
        method=method,
        n_resamples=int(boot.size),
        n_scenarios=k,
        n_items=len(rows),
        seed=int(seed),
    )


def summarize_paired_binary(
    a_correct: Sequence[bool],
    b_correct: Sequence[bool],
    scenario_ids: Sequence[Any] | None = None,
    *,
    n: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    ci: float = 0.95,
) -> Mapping[str, Any]:
    """Accuracy delta reported both ways: clustered bootstrap CI and exact McNemar."""
    bootstrap = paired_bootstrap(
        [float(x) for x in a_correct],
        [float(x) for x in b_correct],
        scenario_ids,
        n=n,
        seed=seed,
        ci=ci,
    )
    return {
        "bootstrap": bootstrap.to_json(),
        "mcnemar": mcnemar_exact(a_correct, b_correct).to_json(),
    }
