"""Cluster-robust paired comparison for RRP-002. MQ-001 supplies no statistics layer at all.

Why this is not a t-test. The banks' rows are not independent, and the dependence is severe and
MEASURED, not assumed:

  RRP2-PRES-DEV (b)  192 rows -> 16 digit-blinded question templates  (CL-038)
  RRP2-PRES-DEV (c)  192 rows -> 189 clusters, but 40 are minimal PAIRS (CL-039)
  RRP2-DEV-V1/V2     384/336 rows -> 8 families

At k = 8 or k = 16 clusters, every asymptotic interval is invalid and a bootstrap is shaky. So
the PRIMARY test here is an exact cluster-level sign-flip permutation test, enumerated in full
whenever k is small enough to enumerate, which is exactly the regime this programme lives in.

The load-bearing guard is `resolution_floor`. With k clusters the smallest two-sided p-value the
sign-flip test can ever produce is 2/2^k. At k = 4 that is 0.125, so the test CANNOT reject at
0.05 no matter how large the effect. A layer that returned "p = 0.125, not significant" there
would be lying by omission: the correct output is that the design has no resolution, and this
module refuses the significance claim instead of reporting one.

Everything here is deterministic. Monte Carlo paths take an explicit seed and record it.
Torch-free by contract.
"""
from __future__ import annotations

import collections
import itertools
import math
import random

# Cluster unit per bank, with the rationale, so no analysis can quietly pick a looser one.
#
# The Forge banks nest: family contains group_id contains row. Worlds in a family share the
# grammar and the solver; worlds in a group_id are variants of the SAME case. The conservative
# unit is therefore `family`, and it is primary. `group_id` is reported as a secondary, more
# powerful and less conservative read -- never as the headline.
CLUSTER_UNITS = {
    "RRP2-DEV-V1": {"primary": "family", "secondary": "group_id",
                    "primary_k": 8,
                    "note": "8 families only. The family-level resolution floor is 2/2^8 = "
                            "0.0078, so cross-family claims have very little power and an "
                            "arm must show a large effect or argue within-family."},
    "RRP2-DEV-V2": {"primary": "family", "secondary": "group_id",
                    "primary_k": 8,
                    "note": "as v1: 8 families, floor 0.0078."},
    "RRP2-PRES-DEV": {"primary": "cluster_id", "secondary": "sub_bank",
                      "primary_k": 589,
                      "note": "cluster_id is the digit-blinded last user turn, carried on every "
                              "row at build time (CL-038). Sub-bank (b) collapses to 16 of "
                              "these, so per-sub-bank results must report their own k."},
}

ALPHA = 0.05
# Enumerate exactly at or below this many clusters: 2^22 = 4.2M sign assignments is seconds of
# CPU and removes Monte Carlo error from the regime that matters here.
EXACT_MAX_CLUSTERS = 22
DEFAULT_DRAWS = 100_000
DEFAULT_BOOTSTRAP = 10_000


class Cluster:
    """One cluster's paired outcome. `diff` is arm minus baseline, in accuracy units."""

    __slots__ = ("key", "n", "baseline_correct", "arm_correct")

    def __init__(self, key, n, baseline_correct, arm_correct):
        self.key = key
        self.n = n
        self.baseline_correct = baseline_correct
        self.arm_correct = arm_correct

    @property
    def diff(self) -> float:
        return (self.arm_correct - self.baseline_correct) / self.n

    def as_dict(self) -> dict:
        return {"cluster": self.key, "rows": self.n,
                "baseline_correct": self.baseline_correct, "arm_correct": self.arm_correct,
                "diff": round(self.diff, 6)}


def paired_clusters(rows, baseline, arm, *, cluster_key) -> list[Cluster]:
    """Group a paired run into clusters. Both arms must cover the same rows in the same order."""
    if not (len(rows) == len(baseline) == len(arm)):
        raise ValueError("rows, baseline and arm must be the same length; a mismatched "
                         "denominator is not something to infer")
    if not rows:
        raise ValueError("no rows to compare")
    buckets = collections.defaultdict(lambda: [0, 0, 0])
    for row, b, a in zip(rows, baseline, arm):
        if cluster_key not in row:
            raise ValueError(f"row {row.get('id')!r} carries no {cluster_key!r}; the cluster "
                             f"unit must be on the data, not chosen at analysis time")
        cell = buckets[row[cluster_key]]
        cell[0] += 1
        cell[1] += bool(b)
        cell[2] += bool(a)
    return [Cluster(key, n, bc, ac) for key, (n, bc, ac) in sorted(buckets.items())]


def resolution_floor(k: int, *, alpha: float = ALPHA) -> dict:
    """What the design can and cannot show, before any data is looked at.

    The sign-flip test's most extreme two-sided outcome has probability 2/2^k, so that is the
    smallest p-value obtainable. If it exceeds alpha the design cannot reject at alpha at all,
    and no amount of effect size changes that.
    """
    smallest = 2.0 / (2 ** k) if k < 60 else 0.0
    can = smallest <= alpha
    return {"clusters": k, "smallest_two_sided_p": smallest,
            "alpha": alpha, "can_reject_at_alpha": can,
            "verdict": ("the design can reach alpha" if can else
                        f"REFUSED: with {k} clusters the smallest achievable two-sided p is "
                        f"{smallest:.4f} > {alpha}, so no effect of any size can be called "
                        f"significant. Report the estimate and the interval, never a "
                        f"significance claim.")}


def _statistic(diffs) -> float:
    return sum(diffs) / len(diffs)


def cluster_permutation_test(clusters, *, alpha: float = ALPHA, shift: float = 0.0,
                             one_sided: bool = False, draws: int = DEFAULT_DRAWS,
                             seed: int = 20260915) -> dict:
    """Exact-when-possible cluster sign-flip test on the mean per-cluster difference.

    Null: each cluster's difference is symmetric about `shift`. `shift` is what makes
    non-inferiority testable -- see `non_inferiority`.
    """
    diffs = [c.diff - shift for c in clusters]
    k = len(diffs)
    observed = _statistic(diffs)
    # A cluster whose difference is exactly zero contributes nothing to the statistic AND
    # nothing to the permutation distribution: flipping its sign changes neither. So the
    # resolution is set by the number of clusters that actually MOVED, not by the number that
    # exist. Measured consequence: 589 clusters with 6 moving has the resolution of k = 6
    # (smallest two-sided p 0.031), not of k = 589. Reporting the raw count here would
    # overstate the design's power by orders of magnitude.
    effective = sum(1 for d in diffs if abs(d) > 1e-12)
    floor = resolution_floor(effective, alpha=alpha)

    def extreme(value: float) -> bool:
        return (value >= observed - 1e-12) if one_sided else (abs(value) >= abs(observed) - 1e-12)

    nonzero = [d for d in diffs if abs(d) > 1e-12]
    if effective <= EXACT_MAX_CLUSTERS:
        total = 2 ** effective
        hits = 0
        for signs in itertools.product((1, -1), repeat=effective):
            if extreme(sum(s * d for s, d in zip(signs, nonzero)) / k):
                hits += 1
        p = hits / total
        method, draws_used = "exact_enumeration", total
    else:
        rng = random.Random(seed)
        hits = 0
        for _ in range(draws):
            flipped = []
            for d in diffs:
                flipped.append(d if rng.getrandbits(1) else -d)
            if extreme(_statistic(flipped)):
                hits += 1
        p = (hits + 1) / (draws + 1)   # add-one, so a Monte Carlo p is never reported as 0
        method, draws_used = "monte_carlo_mt19937", draws

    return {
        "test": "cluster sign-flip permutation on the mean per-cluster difference",
        "one_sided": one_sided, "shift": shift,
        "clusters": k,
        "effective_clusters_nonzero": effective,
        "effective_cluster_note": ("resolution is set by the clusters that MOVED; zero-difference "
                                   "clusters contribute nothing to the permutation distribution"),
        "observed_statistic": round(observed, 6),
        "p_value": p, "method": method, "draws": draws_used,
        "seed": seed if method == "monte_carlo_mt19937" else None,
        "null": "each cluster's paired difference is symmetric about the shift",
        "resolution": floor,
        "significant_at_alpha": bool(p <= alpha and floor["can_reject_at_alpha"]),
        "significance_refused": not floor["can_reject_at_alpha"],
    }


def cluster_bootstrap(clusters, *, draws: int = DEFAULT_BOOTSTRAP, alpha: float = ALPHA,
                      seed: int = 20260915) -> dict:
    """Percentile interval from resampling CLUSTERS with replacement, not rows.

    Flagged unreliable below 30 clusters. It is reported there anyway, because a wide honest
    interval is more informative than no interval, but it must not be read as exact.
    """
    k = len(clusters)
    diffs = [c.diff for c in clusters]
    weights = [c.n for c in clusters]
    rng = random.Random(seed)
    row_w, cluster_w = [], []
    for _ in range(draws):
        picks = []
        for _ in range(k):
            # LCG low bits modulo a power of two cycle over every cluster exactly
            # once, producing a zero-width "bootstrap". Draw with replacement.
            picks.append(rng.randrange(k))
        cluster_w.append(sum(diffs[i] for i in picks) / k)
        total = sum(weights[i] for i in picks)
        row_w.append(sum(diffs[i] * weights[i] for i in picks) / total if total else 0.0)

    def interval(values):
        values = sorted(values)
        lo = values[max(0, int(math.floor((alpha / 2) * len(values))))]
        hi = values[min(len(values) - 1, int(math.ceil((1 - alpha / 2) * len(values))) - 1)]
        return [round(lo, 6), round(hi, 6)]

    return {
        "method": "nonparametric cluster bootstrap, percentile", "draws": draws, "seed": seed,
        "rng": "python_random_mt19937_randrange",
        "clusters": k, "alpha": alpha,
        "cluster_weighted_interval": interval(cluster_w),
        "row_weighted_interval": interval(row_w),
        "reliable": k >= 30,
        "caveat": None if k >= 30 else (
            f"only {k} clusters: the percentile interval is reported but is NOT reliable at "
            f"this cluster count. The exact permutation p-value is the primary inference."),
    }


def point_estimates(clusters) -> dict:
    """Both weightings, always both.

    They disagree whenever clusters are unbalanced, and RRP2-PRES-DEV's are very unbalanced
    (one cluster holds 18 rows, 570 hold one). Reporting only the row-weighted number lets a
    handful of templated rows drive the headline; reporting only the cluster-weighted one
    discards real rows. So: both, side by side.
    """
    rows = sum(c.n for c in clusters)
    base = sum(c.baseline_correct for c in clusters)
    arm = sum(c.arm_correct for c in clusters)
    return {
        "rows": rows, "clusters": len(clusters),
        "baseline_accuracy": round(base / rows, 6), "arm_accuracy": round(arm / rows, 6),
        "row_weighted_difference": round((arm - base) / rows, 6),
        "cluster_weighted_difference": round(sum(c.diff for c in clusters) / len(clusters), 6),
        "baseline_correct": base, "arm_correct": arm,
        "largest_cluster_rows": max(c.n for c in clusters),
        "clusters_with_no_change": sum(1 for c in clusters if c.arm_correct == c.baseline_correct),
    }


def compare(rows, baseline, arm, *, cluster_key, bank=None, margin: float = 0.0,
            alpha: float = ALPHA, seed: int = 20260915) -> dict:
    """Does the gain exceed the declared practical margin at the cluster level?

    Test the boundary at ``margin``, under the sign-flip test's symmetry assumption,
    and require the observed effect to be above it. Retaining the two-sided test
    makes this directional decision conservative and preserves the margin=0
    procedure. Rejecting a zero-effect null while the point estimate barely clears
    a positive margin does not establish superiority over that margin.

    The shifted null requires independent cluster signs to be exchangeable about
    a common location/constant shift of ``margin``. This assumption is not proved
    by the recorded outcomes. Arbitrary heterogeneous, asymmetric cluster effects
    do not give a distribution-free test of population mean superiority here.
    """
    if (isinstance(margin, bool) or not isinstance(margin, (int, float))
            or not math.isfinite(margin) or not 0 <= margin <= 1):
        raise ValueError("superiority margin must be a finite accuracy difference in [0, 1]")
    clusters = paired_clusters(rows, baseline, arm, cluster_key=cluster_key)
    est = point_estimates(clusters)
    test = cluster_permutation_test(clusters, alpha=alpha, seed=seed, shift=margin)
    boot = cluster_bootstrap(clusters, alpha=alpha, seed=seed)
    practical = est["cluster_weighted_difference"] > margin
    return {
        "contrast": "superiority", "bank": bank, "cluster_unit": cluster_key,
        "cluster_unit_rationale": (CLUSTER_UNITS.get(bank) or {}).get("note"),
        "practical_margin": margin,
        "hypothesis": ("At the null boundary, cluster effects are symmetric about the "
                       "declared margin. Superiority requires rejecting that boundary "
                       "with the two-sided sign-flip test and an observed gain above it."),
        "validity_assumptions": ("Independent cluster sign exchangeability under a symmetric "
                                 "location/constant-shift null about the margin; not a "
                                 "distribution-free test of arbitrary population mean superiority."),
        "estimates": est, "permutation": test, "bootstrap": boot,
        "exceeds_practical_margin": practical,
        "claim_supported": bool(practical and test["significant_at_alpha"]),
        "why_not_supported": None if (practical and test["significant_at_alpha"]) else (
            "the design cannot reach alpha (see permutation.resolution)"
            if test["significance_refused"] else
            "the estimate does not exceed the pre-declared practical margin" if not practical
            else "the difference is not distinguishable from the declared margin at the cluster level"),
        "clusters": [c.as_dict() for c in clusters][:64],
        "quality_claim": False,
    }


def non_inferiority(rows, baseline, arm, *, cluster_key, margin: float, bank=None,
                    alpha: float = ALPHA, seed: int = 20260915) -> dict:
    """Preservation: is the arm NOT worse than the baseline by more than `margin`?

    Tested as a one-sided shifted permutation. H0 is that the true loss is at least `margin`
    (i.e. diff <= -margin); rejecting it is what licenses the non-inferiority claim. Note the
    asymmetry on purpose: failing to reject is NOT evidence of preservation, and with few
    clusters that failure is the expected outcome.
    """
    clusters = paired_clusters(rows, baseline, arm, cluster_key=cluster_key)
    est = point_estimates(clusters)
    test = cluster_permutation_test(clusters, alpha=alpha, shift=-margin, one_sided=True,
                                    seed=seed)
    boot = cluster_bootstrap(clusters, alpha=alpha, seed=seed)
    lower = boot["cluster_weighted_interval"][0]
    return {
        "contrast": "non_inferiority", "bank": bank, "cluster_unit": cluster_key,
        "margin": margin,
        "hypothesis": f"H0: the arm is worse by at least {margin} in accuracy; "
                      f"rejecting H0 licenses the preservation claim",
        "estimates": est, "permutation": test, "bootstrap": boot,
        "bootstrap_lower_bound": lower,
        "non_inferiority_supported": bool(test["significant_at_alpha"]
                                          and est["cluster_weighted_difference"] > -margin),
        "note": ("Failure to reject is not evidence of preservation. With few clusters it is "
                 "the default outcome, and it must be reported as 'preservation not "
                 "demonstrated', never as 'preservation confirmed'."),
        "quality_claim": False,
    }


def minimal_pair_contrast(rows, predictions, *, abstained) -> dict:
    """The strictest abstention read available: within-pair evidence sensitivity.

    Each complete pair differs by the decisive fact alone, so the pair is its own control. A
    model that abstains on both, or answers both, is insensitive to evidence rather than merely
    inaccurate -- and the row-level accuracy cannot tell those apart.

    `abstained` maps row index -> bool. The pair unit is `pair_id`; only complete pairs count.
    """
    pairs = collections.defaultdict(dict)
    for index, row in enumerate(rows):
        if row.get("pair_id"):
            pairs[row["pair_id"]][bool(row["answerable"])] = abstained[index]
    complete = {k: v for k, v in pairs.items() if len(v) == 2}
    tally = collections.Counter()
    for members in complete.values():
        on_unanswerable, on_answerable = members[False], members[True]
        if on_unanswerable and not on_answerable:
            tally["evidence_sensitive"] += 1
        elif on_unanswerable and on_answerable:
            tally["abstains_regardless"] += 1
        elif not on_unanswerable and not on_answerable:
            tally["answers_regardless"] += 1
        else:
            tally["inverted"] += 1
    k = len(complete)
    return {
        "unit": "complete minimal pair (pair_id)", "complete_pairs": k,
        "incomplete_pairs_excluded": len(pairs) - k,
        **dict(tally),
        "evidence_sensitivity_rate": round(tally["evidence_sensitive"] / k, 6) if k else None,
        "resolution": resolution_floor(k),
        "reading": ("`evidence_sensitive` is the only outcome that shows the model responded to "
                    "the presence of the decisive fact. The other three are all failures of "
                    "calibration, and `inverted` is the worst: it abstained where the fact was "
                    "present and answered where it was absent."),
        "quality_claim": False,
    }


def holm(pvalues: dict, *, alpha: float = ALPHA) -> dict:
    """Holm-Bonferroni over the PRE-DECLARED primary contrasts only.

    Applied to the predeclared set and nothing else. Running it over every contrast computed
    would be a different -- and much weaker -- procedure than the one registered, and choosing
    the family after seeing the p-values is how multiplicity control gets quietly abandoned.
    """
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out, still = {}, True
    for index, (name, p) in enumerate(ordered):
        threshold = alpha / (m - index)
        if still and p <= threshold:
            out[name] = {"p": p, "threshold": threshold, "rejected": True}
        else:
            still = False
            out[name] = {"p": p, "threshold": threshold, "rejected": False}
    return {"alpha": alpha, "contrasts": m, "procedure": "Holm-Bonferroni, step-down",
            "results": out,
            "rejected": sorted(k for k, v in out.items() if v["rejected"])}
