"""SMALLVD-001 Amendment 2/4 -- the H-LC verdict, pure Python (no torch), so its power can be simulated
without a model and without host RAM.

obj(rows) is the §4 objective exactly as `run_layer_study.category_mean_kl(layer_ablation.summarise(rows))`
computes it: per category the POSITION-weighted mean KL (sum kl*n / sum n), then the unweighted mean
over categories. tests/test_linear_compensation.py asserts the two agree.
"""
from __future__ import annotations

import math
import random

SUPPORT_BAR = 0.20
PRECISE_LOWER = 0.10      # Amendment 4: SUPPORTED with a lower bound below this is flagged size-imprecise
MIN_ROWS_PER_CATEGORY = 5
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 2000, 17


def cat_mean_kl(rows: list[dict]) -> float:
    by = {}
    for r in rows:
        a = by.setdefault(r["category"], [0.0, 0])
        a[0] += r["kl"] * r["n"]
        a[1] += r["n"]
    return sum(s / n for s, n in by.values()) / len(by)


def diagnostics(plain_rows: list[dict], comp_rows: list[dict]) -> dict:
    """Observable after the fit: the spread of per-row log(LC/plain) KL, the spread of log plain KL,
    and the minimum per-category effective n (sum w)^2 / sum w^2 of the plain weights w = kl * n.
    Calibration (Amendment 5) found the null rate driven by the log-ratio spread even at FIXED
    effective n, so both are recorded and the spreads gate the verdict."""
    lr = [math.log(max(c["kl"], 1e-12) / max(p["kl"], 1e-12)) for p, c in zip(plain_rows, comp_rows)]
    lk = [math.log(max(p["kl"], 1e-12)) for p in plain_rows]

    def sd(x):
        m = sum(x) / len(x)
        return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))
    w = {}
    for r in plain_rows:
        w.setdefault(r["category"], []).append(r["kl"] * r["n"])
    return {"s_ratio_hat": sd(lr), "s_kl_hat": sd(lk),
            "min_eff_n": min(sum(x) ** 2 / sum(y * y for y in x) for x in w.values())}


def verdict(plain: list[dict], comp: list[dict], ids: list[str], support_bar: float = SUPPORT_BAR,
            resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """r = 1 - obj(LC)/obj(plain) on the given rows; 95 % interval by a category-stratified PAIRED
    bootstrap (the same resampled rows for both arms).
      SUPPORTED       r >= support_bar and lower bound > 0   (a usefulness bar: a tiny gain is not
                                                             worth a new mechanism in the student)
      REAL_BUT_SMALL  lower bound > 0 but r < support_bar
      HARMFUL         upper bound < 0                        (no size floor, deliberately: plain
                                                             pruning is the free null, so ANY
                                                             reliable harm disqualifies the method)
      INCONCLUSIVE    otherwise
      UNDERPOWERED    any category has < MIN_ROWS_PER_CATEGORY rows (a guard for subsampled dry
                      runs; the real REPORT has 7-18 per category and never reaches it)
    size_precise (Amendment 4): for SUPPORTED, whether the lower bound also clears PRECISE_LOWER. The
    power simulation (design/HLC_POWER_SIM.json) shows that under extreme per-row heterogeneity a
    TRUE r of 0.10 still reaches SUPPORTED 25-28 % of the time: the effect is real (lower bound > 0)
    but the ">= 0.20" size is noise. The flag says so instead of letting the verdict imply it.
    The interval is CONDITIONAL on these categories (they are fixed, not sampled)."""
    P, C = {r["id"]: r for r in plain}, {r["id"]: r for r in comp}
    by_cat = {}
    for i in ids:
        by_cat.setdefault(P[i]["category"], []).append(i)

    def r_of(sample):
        return 1 - cat_mean_kl([C[i] for i in sample]) / cat_mean_kl([P[i] for i in sample])
    r = r_of(ids)
    rng = random.Random(seed)
    boots = sorted(r_of([rng.choice(v) for v in by_cat.values() for _ in v]) for _ in range(resamples))
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    v = ("UNDERPOWERED" if min(len(x) for x in by_cat.values()) < MIN_ROWS_PER_CATEGORY else
         "SUPPORTED" if r >= support_bar and lo > 0 else "REAL_BUT_SMALL" if lo > 0 else
         "HARMFUL" if hi < 0 else "INCONCLUSIVE")
    return {"rows": len(ids), "obj_plain": cat_mean_kl([P[i] for i in ids]), "obj_lc": cat_mean_kl([C[i] for i in ids]),
            "relative_reduction_r": r, "bootstrap_95": [lo, hi], "support_bar": support_bar,
            "rows_per_category": {c: len(x) for c, x in sorted(by_cat.items())},
            "resamples": resamples, "seed": seed, "verdict": v,
            "size_precise": (lo >= PRECISE_LOWER) if v == "SUPPORTED" else None,
            "ci_width": hi - lo,
            # Amendment 5: one-sided bootstrap tail fractions, for a CALIBRATED level alpha*
            "boot_frac_le_0": sum(b <= 0 for b in boots) / len(boots),
            "boot_frac_le_precise": sum(b <= PRECISE_LOWER for b in boots) / len(boots),
            **diagnostics([P[i] for i in ids], [C[i] for i in ids])}
