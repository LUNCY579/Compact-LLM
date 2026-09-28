"""SMALLVD-001 Amendment 5 -- derive the calibrated levels from the raw calibration draws.

The claim is ONE-SIDED: at every simulated cell, the probability of a false certification is at
most CLAIM = 0.025. A level is accepted only if the worst cell's one-sided 95 % WILSON UPPER BOUND
on that probability is <= CLAIM -- so the claim holds within Monte-Carlo error, explicitly, rather
than through an ad-hoc tighter target (the first version used a 0.02 point target; the review showed
harm's worst cell then had an upper bound of 0.0266).

Per spread band (simulated at its UPPER edge):
  alpha_benefit  largest level in ALPHAS whose null false-benefit bound is <= CLAIM at every s_kl
                 (the benefit null rate rises with the spread in every measured row, so the upper
                 edge is the conservative point)
  alpha_size     the same for the size claim at its r = 0.10 boundary (SUPPORTED only)
  alpha_harm     per-band design level, then ONE UNIFORM level = the minimum over bands. Harm's null
                 rate is NOT monotone in the spread: it peaks at an INTERIOR band, so no edge rule
                 is safe and the only correct target is the maximum -- which the minimum over bands
                 selects (calibration AT the measured maximum, not a fallback).
The smallest level is 1 / resamples: a band that needs less is UNCALIBRATED, and a band AT that
level certifies only when at most one bootstrap draw falls at or below 0 (a named fragility).
usage: python hlc_calibration_analysis.py <draws *.jsonl ...> <out.json>
"""
import json
import math
import sys
from collections import defaultdict

CLAIM = 0.025
Z = 1.6448536269514722          # one-sided 95 %
RESAMPLES = 2000
ALPHAS = (0.025, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.004, 0.003, 0.002, 0.001, 1 / RESAMPLES)


def wilson_upper(k: int, n: int) -> float:
    p = k / n
    c = Z * Z / n
    return (p + c / 2 + Z * math.sqrt(p * (1 - p) / n + c / (4 * n))) / (1 + c)


def main():
    *ins, out = sys.argv[1:]
    rows = [json.loads(l) for p in ins for l in open(p, encoding="utf-8") if l.strip()]
    g = defaultdict(list)
    for r in rows:
        g[(r["band_s_ratio"], r["s_kl"], r["kind"])].append(r)
    bands = sorted({r["band_s_ratio"] for r in rows})
    s_kls = sorted({r["s_kl"] for r in rows})
    count = lambda xs, f: sum(1 for x in xs if f(x))
    res = {"claim_one_sided": CLAIM, "criterion": "worst-cell one-sided 95% Wilson upper bound <= claim",
           "resamples": RESAMPLES, "alphas_tried": ALPHAS, "total_draws": len(rows), "bands": []}

    def pick(b, kind, event):
        for a in ALPHAS:
            worst = max(wilson_upper(count(g[(b, k, kind)], lambda x: event(x, a)), len(g[(b, k, kind)]))
                        for k in s_kls)
            if worst <= CLAIM:
                return a, worst
        return None, None

    ben_ev = lambda x, a: x["boot_frac_le_0"] < a
    size_ev = lambda x, a: x["relative_reduction_r"] >= 0.20 and x["boot_frac_le_precise"] < a
    harm_ev = lambda x, a: x["boot_frac_le_0"] > 1 - a
    for b in bands:
        ab, wb = pick(b, "null", ben_ev)
        asz, wsz = pick(b, "boundary", size_ev)
        ah, wh = pick(b, "null", harm_ev)
        res["bands"].append({"band_upper_s_ratio": b, "alpha_benefit": ab, "worst_upper_null_false_benefit": wb,
                             "alpha_size": asz, "worst_upper_boundary_false_size": wsz,
                             "alpha_harm_per_band_design": ah, "worst_upper_null_false_harm_design": wh,
                             "at_resolution_floor": ab == 1 / RESAMPLES,
                             "null_harm_rate_uncalibrated_0.025": max(
                                 count(g[(b, k, "null")], lambda x: harm_ev(x, 0.025)) / len(g[(b, k, "null")])
                                 for k in s_kls)})
    ah_min = min(c["alpha_harm_per_band_design"] for c in res["bands"] if c["alpha_harm_per_band_design"] is not None)
    for c in res["bands"]:
        c["alpha_harm"] = ah_min
    res["alpha_harm_uniform"] = ah_min
    res["harm_null_rate_profile_by_band"] = {c["band_upper_s_ratio"]: c["null_harm_rate_uncalibrated_0.025"]
                                             for c in res["bands"]}

    # ---- the rule AS APPLIED (band chosen from the NOISY observed spread), per generative cell
    table = [(c["band_upper_s_ratio"], c["alpha_benefit"], c["alpha_size"], c["alpha_harm"]) for c in res["bands"]]

    def applied(x):
        band = next((t for t in table if x["s_ratio_hat"] <= t[0]), None)
        if band is None or band[1] is None:
            return "UNCALIBRATED", False
        ben = x["boot_frac_le_0"] < band[1]
        sup = ben and x["relative_reduction_r"] >= 0.20
        size = sup and band[2] is not None and x["boot_frac_le_precise"] < band[2]
        harm = x["boot_frac_le_0"] > 1 - band[3]
        return ("SUPPORTED" if sup else "REAL_BUT_SMALL" if ben else "HARMFUL" if harm else "OTHER"), size
    oc = []
    for b in bands:
        for k in s_kls:
            cell = {"generative_s_ratio": b, "s_kl": k}
            for kind, key, ev in (("null", "null_false_benefit", lambda v: v[0] in ("SUPPORTED", "REAL_BUT_SMALL")),
                                  ("null", "null_false_SUPPORTED", lambda v: v[0] == "SUPPORTED"),
                                  ("null", "null_false_HARMFUL", lambda v: v[0] == "HARMFUL"),
                                  ("boundary", "boundary_false_size_claim", lambda v: v[1]),
                                  ("power", "power_r030_SUPPORTED", lambda v: v[0] == "SUPPORTED"),
                                  ("power", "power_r030_uncalibrated", lambda v: v[0] == "UNCALIBRATED"),
                                  ("harm_power_020", "harm_power_r_minus020", lambda v: v[0] == "HARMFUL"),
                                  ("harm_power_030", "harm_power_r_minus030", lambda v: v[0] == "HARMFUL"),
                                  # ADOPTION UNDER HARM: P(SUPPORTED | r < 0) -- not the same quantity as the
                                  # controlled P(SUPPORTED | r = 0); measured, not argued (review e4b502f)
                                  ("harm_power_020", "adoption_under_harm_r_minus020", lambda v: v[0] == "SUPPORTED"),
                                  ("harm_power_030", "adoption_under_harm_r_minus030", lambda v: v[0] == "SUPPORTED"),
                                  ("n2_harm", "n2_false_benefit", lambda v: v[0] in ("SUPPORTED", "REAL_BUT_SMALL"))):
                xs = g.get((b, k, kind), [])
                if not xs:
                    cell[key] = None
                    continue
                kk = sum(1 for x in xs if ev(applied(x)))
                cell[key], cell[key + "_n"] = kk / len(xs), len(xs)
                if key in ("null_false_benefit", "null_false_HARMFUL", "boundary_false_size_claim",
                           "adoption_under_harm_r_minus020", "adoption_under_harm_r_minus030"):
                    cell[key + "_wilson_upper"] = wilson_upper(kk, len(xs))
            oc.append(cell)
    res["operating_characteristic_as_applied"] = oc
    for key in ("null_false_benefit", "null_false_HARMFUL", "boundary_false_size_claim",
                "adoption_under_harm_r_minus020", "adoption_under_harm_r_minus030"):
        res[f"max_{key}_as_applied"] = max(o[key] for o in oc)
        res[f"max_{key}_wilson_upper_as_applied"] = max(o[key + "_wilson_upper"] for o in oc)
    # the names the canary test reads
    res["max_null_false_harm_as_applied"] = res["max_null_false_HARMFUL_as_applied"]
    res["max_boundary_false_size_as_applied"] = res["max_boundary_false_size_claim_as_applied"]
    res["claim_holds_within_mc_error"] = all(res[f"max_{k}_wilson_upper_as_applied"] <= CLAIM
                                             for k in ("null_false_benefit", "null_false_HARMFUL",
                                                       "boundary_false_size_claim",
                                                       "adoption_under_harm_r_minus020",
                                                       "adoption_under_harm_r_minus030"))
    # harm DETECTABILITY by band, from the measured harm power at r = -0.20 (minimum over s_kl)
    res["harm_power_r_minus020_min_by_band"] = {
        b: min(o["harm_power_r_minus020"] for o in oc if o["generative_s_ratio"] == b) for b in bands}
    json.dump(res, open(out, "w", encoding="utf-8", newline=chr(10)), indent=1)
    print("draws", len(rows), "| harm profile", res["harm_null_rate_profile_by_band"], "| uniform harm", ah_min)
    for c in res["bands"]:
        print(" band", c["band_upper_s_ratio"], "alpha_benefit", c["alpha_benefit"], "alpha_size", c["alpha_size"],
              "harm design", c["alpha_harm_per_band_design"], "floor" if c["at_resolution_floor"] else "")
    for k in ("null_false_benefit", "null_false_HARMFUL", "boundary_false_size_claim",
              "adoption_under_harm_r_minus020", "adoption_under_harm_r_minus030"):
        print(" AS APPLIED", k, "worst", round(res[f"max_{k}_as_applied"], 5),
              "wilson upper", round(res[f"max_{k}_wilson_upper_as_applied"], 5))
    print(" claim holds within MC error:", res["claim_holds_within_mc_error"])
    print(" harm power r=-0.20 (min over s_kl) by band:", res["harm_power_r_minus020_min_by_band"])
    for o in oc:
        print("  ", o["generative_s_ratio"], o["s_kl"], "pow0.3", o["power_r030_SUPPORTED"],
              "harm-0.2", o["harm_power_r_minus020"], "harm-0.3", o["harm_power_r_minus030"])


if __name__ == "__main__":
    main()
