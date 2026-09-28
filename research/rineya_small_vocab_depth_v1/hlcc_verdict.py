"""HLCC-001 -- the confirmatory verdict for linear compensation, EXACT and distribution-free.

SMALLVD-001's H-LC verdict (a percentile bootstrap on a category-mean KL ratio) could not be certified:
the realised per-row log-ratios had a spread of 1.25 and a skewed, heavy-tailed shape, outside every
calibrated family. HLCC-001 therefore tests a different estimand with an exact test:

  d_i = log(plain_i / lc_i) per row (KL per position, or answer nats), d > 0 = compensation better.
  PRIMARY TEST: one-sided exact SIGN test, H0 median(d) = 0 (P(d > 0) = 1/2), rows independent (one row
  per template family). Exact rows with d == 0 carry no sign and are dropped (counted).
  SIZE: the distribution-free lower confidence bound on median(d) from order statistics -- exp() of it
  is the minimum per-row MEDIAN reduction factor.

No resampling, no calibration family, no shape assumption. The estimand is the PER-ROW MEDIAN, not the
§4 category-mean objective ratio; that ratio is reported beside it, descriptively.

R2 (cold review, before any run): THE REGISTERED TEST IS ROW-LEVEL, with a POOL-CONDITIONAL H0. Every
category was drawn as a content-independent hash-ordered simple random sample from a fixed pool, and T
is fixed from SELECT; conditional on the pools and T, the sampled rows' signs are i.i.d. by
construction, so the row-level sign test is exact (conservative under stratification and sampling
without replacement). It licenses claims about THESE POOLS, not about new books, generators or
packages -- the panel shares most of its generators with the fit set, so no unit rule could license
that. Two unit analyses are reported beside it:
  * the all-axis connected components (HLCC_ALLAXIS_UNIT_MAP.json, pinned), with a MAJORITY-SIGN unit
    statistic -- the registered SENSITIVITY analysis (direction only);
  * the 267cec9b unit rule (`unit_of` + unit MEDIAN), descriptive only. It was withdrawn because it
    held alpha under neither reading: it collapsed 2 of the panel's 8 shared-generator axes, and the
    median of an EVEN-sized unit is the MEAN of its middle rows, which under a skewed null is positive
    more than half the time (0.635 for two LogNormal(0,1.5)-1 draws).
"""
from __future__ import annotations

import math

ALPHA_EACH = 0.0125          # two co-primary tests (KL, answer bpb) at one-sided 0.025 total (Bonferroni)
LARGE_BAR = math.log(1.25)   # size annotation: median per-row reduction of at least 20 %


def unit_of(row: dict) -> str:
    """The independence unit of a CONFIRM_PANEL row (registered in HLCC001_PREREG.md)."""
    rid = row["id"]
    if row["kind"] == "text":
        src = rid[len("confirm:"):] if rid.startswith("confirm:") else rid
        if src.startswith(("story:", "md:")):        # <prefix>:<document>:<chunk index>
            return "doc:" + src.rsplit(":", 1)[0]
        return "doc:" + src                          # py:/js: ids are one file each
    if row["source"] == "cc0:validation":            # rineya_v3_sft__<task>__validation_<k>__<item>
        return "task:" + rid.split("__")[1]
    return "row:" + rid


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def collapse(d_by_id: dict[str, float], unit_by_id: dict[str, str]) -> dict[str, float]:
    """Unit -> median of its rows' d. Every row must have a unit (a missing one is an error, not a drop)."""
    groups: dict[str, list[float]] = {}
    for i, x in d_by_id.items():
        groups.setdefault(unit_by_id[i], []).append(x)
    return {u: median(v) for u, v in sorted(groups.items())}


def majority_sign_collapse(d_by_id: dict[str, float], unit_by_id: dict[str, str]) -> dict[str, int]:
    """Unit -> sign(#rows with d > 0 - #rows with d < 0), in {-1, 0, +1}; 0 is a tie (dropped by the sign
    test). Unlike the median of an even-sized unit it never averages, so under a null where every
    row's sign is a fair coin each unit's sign is symmetric whatever the shape of d."""
    acc: dict[str, int] = {}
    for i, x in d_by_id.items():
        u = unit_by_id[i]
        acc[u] = acc.get(u, 0) + (x > 0) - (x < 0)
    return {u: (v > 0) - (v < 0) for u, v in sorted(acc.items())}


def binom_upper_tail(n: int, k: int) -> float:
    """P(Binom(n, 1/2) >= k), exact (integer arithmetic)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def sign_test(d: list[float]) -> dict:
    nz = [x for x in d if x != 0]
    n, k = len(nz), sum(1 for x in nz if x > 0)
    return {"rows": len(d), "ties_dropped": len(d) - n, "n": n, "positive": k,
            "p_greater": binom_upper_tail(n, k) if n else 1.0,     # compensation better
            "p_less": binom_upper_tail(n, n - k) if n else 1.0}    # compensation worse


def median_lower_bound(d: list[float], alpha: float = ALPHA_EACH):
    """Distribution-free one-sided (1 - alpha) lower confidence bound for median(d): the order
    statistic d_(k) with the largest k such that P(Binom(n, 1/2) <= k - 1) <= alpha. None if even
    d_(1) cannot be certified at this n."""
    s = sorted(d)
    n = len(s)
    best, cum = None, 0.0
    for k in range(1, n + 1):
        cum += math.comb(n, k - 1) / 2 ** n            # P(B <= k - 1)
        if cum <= alpha:
            best = k
        else:
            break
    return None if best is None else s[best - 1]


def verdict(d_kl: list[float], d_bpb: list[float], alpha: float = ALPHA_EACH, with_size: bool = True) -> dict:
    """CONFIRMED  both sign tests reject H0 toward compensation-better at alpha each
       PARTIAL    exactly one does
       HARMFUL    either rejects toward compensation-WORSE at alpha (checked first)
       NOT_CONFIRMED otherwise
    size_large: both median lower bounds >= log(1.25) (annotation, CONFIRMED only). with_size=False
    for inputs that are SIGNS (majority-sign units): an order statistic of +-1 is not a size."""
    tk, tb = sign_test(d_kl), sign_test(d_bpb)
    lk, lb = (median_lower_bound(d_kl, alpha), median_lower_bound(d_bpb, alpha)) if with_size else (None, None)
    ck, cb = tk["p_greater"] < alpha, tb["p_greater"] < alpha
    if tk["p_less"] < alpha or tb["p_less"] < alpha:
        v = "HARMFUL"
    elif ck and cb:
        v = "CONFIRMED"
    elif ck or cb:
        v = "PARTIAL"
    else:
        v = "NOT_CONFIRMED"
    large = (v == "CONFIRMED" and lk is not None and lb is not None and lk >= LARGE_BAR and lb >= LARGE_BAR)
    return {"verdict": v, "alpha_each": alpha, "kl": {**tk, "median_lower_bound": lk,
                                                      "min_median_reduction_factor": None if lk is None else math.exp(lk)},
            "bpb": {**tb, "median_lower_bound": lb,
                    "min_median_reduction_factor": None if lb is None else math.exp(lb)},
            "size_large": (large if v == "CONFIRMED" else None) if with_size else None,
            "large_bar_factor": math.exp(LARGE_BAR), "with_size": with_size}


def power_sign_test(n: int, p_true: float, alpha: float = ALPHA_EACH) -> float:
    """Exact power of the one-sided sign test at n non-tied rows when P(d > 0) = p_true."""
    kcrit = next(k for k in range(n + 2) if binom_upper_tail(n, k) < alpha)
    return sum(math.comb(n, i) * p_true ** i * (1 - p_true) ** (n - i) for i in range(kcrit, n + 1))
