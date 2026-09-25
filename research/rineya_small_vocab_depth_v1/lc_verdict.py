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


# ---------------------------------------------------------------------------------------------------
# Amendment 5 (revised): the REGISTERED verdict is calibrated (design/HLC_CALIBRATION.json, from
# design/hlc_calibration_sim.py + design/hlc_calibration_analysis.py, 36,300 committed draws).
# CLAIM, ONE-SIDED: at every simulated cell, P(false certification) <= 0.025, accepted only when the
# worst cell's one-sided 95 % WILSON UPPER BOUND is <= 0.025 -- the claim holds within Monte-Carlo
# error, explicitly. The raw rule (lower 2.5 % bound > 0) was anticonservative under the mean-ratio
# null (up to 0.168), driven by the per-row log-ratio spread -- NOT by effective n.
#   benefit  per band of the OBSERVED spread s_ratio_hat, calibrated at the band's upper edge (the
#            benefit null rate rises with the spread in every measured row)
#   size     SUPPORTED only, calibrated at the r = 0.10 boundary
#   harm     ONE uniform level: harm's null rate is NOT monotone in the spread -- it peaks at the
#            INTERIOR band 0.45 (0.027 / 0.030 / 0.015 / 0.012) -- so no edge rule is safe and the only
#            correct target is the maximum; the minimum over bands is band 0.45's own level
#            (calibration AT the measured maximum, not a fallback)
#   floor    the smallest level is 1/2000: band 0.8 sits ON it, i.e. certifies only when at most ONE
#            bootstrap draw falls at or below 0 (a named fragility); above 0.8 no level exists at this
#            resolution, so the edge is FORCED by the resample count, not chosen
# As applied (band from the noisy observed spread), worst generative cell, [Wilson upper]:
#   false benefit 0.016 [0.0223]   false harm 0.014 [0.0199]   false size 0.0125 [0.0190]
#   false benefit under the harmful alternative 0.  Harm POWER at r = -0.20: 0.96-1.00 (spread <= 0.2),
#   0.50-0.85 (0.45), 0.03-0.09 (0.8). A missed harm cannot cause adoption: adoption needs SUPPORTED.
# BINDING: valid only for BOOTSTRAP_SEED = 17, BOOTSTRAP_RESAMPLES = 2000 and the REPORT category
# sizes it was simulated at; any other combination returns UNCALIBRATED.
# (upper edge of s_ratio_hat, alpha_benefit, alpha_size)
CALIBRATION = ((0.2, 0.004, 0.025), (0.45, 0.002, 0.002), (0.8, 0.0005, 0.0005), (1.2, None, None))
ALPHA_HARM = 0.01
# Harm POWER at r = -0.20 by band (minimum over s_kl, as applied; HLC_CALIBRATION.json
# harm_power_r_minus020_min_by_band). A non-HARMFUL verdict where this is low means harm was
# UNDETECTABLE, not absent: benefit is certifiable to 0.8, harm is meaningfully detectable only to 0.45.
HARM_POWER_R020_BY_BAND = {0.2: 0.96, 0.45: 0.5, 0.8: 0.02666666666666667, 1.2: 0.0}
CALIBRATED_FOR = {"seed": 17, "resamples": 2000,
                  "rows_per_category": {"abstention_unknown": 14, "concise_reasoning": 16, "conversation": 18,
                                        "grounded_extraction": 13, "instruction_writing": 14, "long_context": 15,
                                        "text_english_prose": 9, "text_javascript": 8, "text_markdown": 10,
                                        "text_python": 7}}


def calibrated(v: dict) -> dict:
    """The REGISTERED H-LC verdict from a raw `verdict()` result.
      UNDERPOWERED   raw guard (subsampled dry runs only)
      UNCALIBRATED   seed / resamples / category sizes differ from the calibration, or the observed
                     spread is above 0.8: r and its interval are reported, nothing is certified
      SUPPORTED      boot_frac_le_0 < alpha_benefit(band) and r >= 0.20
      REAL_BUT_SMALL boot_frac_le_0 < alpha_benefit(band) and r < 0.20
      HARMFUL        bootstrap mass above 0 < ALPHA_HARM
      INCONCLUSIVE   otherwise
    min_eff_n is an INFORMATION diagnostic only: calibration was measured to follow the spread, not
    effective n, so refusing a calibrated verdict for low effective n would be a too-strict gate."""
    out = {**v, "verdict_uncalibrated": v["verdict"], "size_precise_uncalibrated": v["size_precise"],
           "calibration_band": None, "alpha_benefit": None, "alpha_size": None, "alpha_harm": None,
           "size_precise": None, "size_claim_calibrated": False}
    if v["verdict"] == "UNDERPOWERED":
        return out
    out.update(harm_detectability(v["s_ratio_hat"]))
    binding = (v["seed"] == CALIBRATED_FOR["seed"] and v["resamples"] == CALIBRATED_FOR["resamples"]
               and v["rows_per_category"] == CALIBRATED_FOR["rows_per_category"])
    band = next((b for b in CALIBRATION if v["s_ratio_hat"] <= b[0]), None)
    if not binding or band is None or band[1] is None:
        return {**out, "verdict": "UNCALIBRATED", "calibration_band": None if band is None else band[0],
                "uncalibrated_reason": ("seed/resamples/category sizes differ from the calibration" if not binding
                                        else "observed spread above the calibrated range")}
    _, a_b, a_s = band
    benefit = v["boot_frac_le_0"] < a_b
    harm = (1 - v["boot_frac_le_0"]) < ALPHA_HARM
    vc = ("SUPPORTED" if benefit and v["relative_reduction_r"] >= v["support_bar"] else
          "REAL_BUT_SMALL" if benefit else "HARMFUL" if harm else "INCONCLUSIVE")
    return {**out, "verdict": vc, "calibration_band": band[0], "alpha_benefit": a_b, "alpha_size": a_s,
            "alpha_harm": ALPHA_HARM,
            # a band with no calibratable size level certifies NO size claim (None, never False)
            "size_precise": (v["boot_frac_le_precise"] < a_s) if (vc == "SUPPORTED" and a_s is not None) else None,
            "size_claim_calibrated": a_s is not None}


def harm_detectability(s_ratio_hat: float) -> dict:
    """Pre-declared (Amendment 5, A5.7): what a NON-HARMFUL verdict means at this observed spread."""
    band = next((b for b in sorted(HARM_POWER_R020_BY_BAND) if s_ratio_hat <= b), None)
    power = 0.0 if band is None else HARM_POWER_R020_BY_BAND[band]
    level = "YES" if power >= 0.9 else "PARTIAL" if power >= 0.5 else "NO"
    return {"harm_detectability": level, "harm_power_r_minus020_at_band": power,
            "harm_reading": {"YES": "harm at r = -0.20 would be detected with power >= 0.9",
                             "PARTIAL": "harm at r = -0.20 would be detected only about half the time",
                             "NO": "a non-HARMFUL verdict here means harm was UNDETECTABLE, not absent"}[level]}
