"""SMALLVD-001 Amendment 5 -- derive the calibrated levels from the raw calibration draws.

Per spread band (simulated at its UPPER edge, the conservative point):
  alpha_benefit = the largest level in ALPHAS whose null false-benefit rate, P(boot_frac_le_0 < alpha |
                  r = 0, N1), is <= TARGET at EVERY s_kl; none -> the band is UNCALIBRATED
  alpha_size    = the same for the size claim at its boundary: P(boot_frac_le_precise < alpha | r = 0.10)
  checks        = power at r = 0.30 (P(r >= 0.20 and boot_frac_le_0 < alpha_benefit)), the harm-side null
                  rate P(boot_frac_le_0 > 0.975 | r = 0) for the unchanged HARMFUL rule, and false benefit
                  under the HARMFUL alternative N2
usage: python hlc_calibration_analysis.py <band_*.jsonl ...> <out.json>
"""
import json
import sys
from collections import defaultdict

TARGET = 0.025
ALPHAS = (0.025, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0005)   # 2000 resamples: 0.0005 = none <= 0


def main():
    global TARGET
    args = sys.argv[1:]
    if args[0].startswith("--target="):
        TARGET = float(args.pop(0).split("=")[1])
    *ins, out = args
    rows = [json.loads(l) for p in ins for l in open(p, encoding="utf-8") if l.strip()]
    g = defaultdict(list)
    for r in rows:
        g[(r["band_s_ratio"], r["s_kl"], r["kind"])].append(r)
    bands = sorted({r["band_s_ratio"] for r in rows})
    s_kls = sorted({r["s_kl"] for r in rows})
    rate = lambda xs, f: sum(1 for x in xs if f(x)) / len(xs)
    res = {"target": TARGET, "alphas_tried": ALPHAS, "bands": []}
    for b in bands:
        def pick(kind, field):
            for a in ALPHAS:
                worst = max(rate(g[(b, k, kind)], lambda x: x[field] < a) for k in s_kls)
                if worst <= TARGET:
                    return a, worst
            return None, None
        ab, wb = pick("null", "boot_frac_le_0")
        asz, wsz = pick("boundary", "boot_frac_le_precise")
        # harm: certified when the bootstrap mass ABOVE 0 is below alpha, i.e. frac_le_0 > 1 - alpha
        ah, wh = None, None
        for a in ALPHAS:
            worst = max(rate(g[(b, k, "null")], lambda x: x["boot_frac_le_0"] > 1 - a) for k in s_kls)
            if worst <= TARGET:
                ah, wh = a, worst
                break
        cell = {"band_upper_s_ratio": b, "alpha_benefit": ab, "worst_null_false_benefit_at_alpha": wb,
                "alpha_size": asz, "worst_boundary_false_size_at_alpha": wsz,
                "alpha_harm": ah, "worst_null_false_harm_at_alpha": wh, "by_s_kl": {}}
        for k in s_kls:
            nul, bnd, pw, n2 = (g[(b, k, t)] for t in ("null", "boundary", "power", "n2_harm"))
            cell["by_s_kl"][str(k)] = {
                "n": {"null": len(nul), "boundary": len(bnd), "power": len(pw), "n2": len(n2)},
                "uncalibrated_null_false_benefit_0.025": rate(nul, lambda x: x["boot_frac_le_0"] < 0.025),
                "null_false_benefit_at_alpha": None if ab is None else rate(nul, lambda x: x["boot_frac_le_0"] < ab),
                "null_false_harm_unchanged_rule": rate(nul, lambda x: x["boot_frac_le_0"] > 0.975),
                "power_r030_SUPPORTED": None if ab is None else rate(
                    pw, lambda x: x["relative_reduction_r"] >= 0.20 and x["boot_frac_le_0"] < ab),
                "power_r030_size_precise": None if (ab is None or asz is None) else rate(
                    pw, lambda x: x["relative_reduction_r"] >= 0.20 and x["boot_frac_le_0"] < ab
                    and x["boot_frac_le_precise"] < asz),
                "n2_false_benefit_at_alpha": None if ab is None else rate(n2, lambda x: x["boot_frac_le_0"] < ab),
                "median_s_ratio_hat_null": sorted(x["s_ratio_hat"] for x in nul)[len(nul) // 2],
                "median_min_eff_n_null": sorted(x["min_eff_n"] for x in nul)[len(nul) // 2]}
        res["bands"].append(cell)
    # HARM's null rate FALLS as the spread rises -- the opposite direction to benefit -- so "calibrate
    # at the band's upper edge" is anticonservative for harm (measured: 0.031 as applied with per-band
    # harm levels). One UNIFORM harm level, the minimum over bands, is conservative in either direction.
    ah_min = min(c["alpha_harm"] for c in res["bands"] if c["alpha_harm"] is not None)
    for c in res["bands"]:
        c["alpha_harm_per_band_design"] = c["alpha_harm"]
        c["alpha_harm"] = ah_min
    res["alpha_harm_uniform"] = ah_min

    # ---- pass 2: the rule AS APPLIED. The verdict picks its band from the OBSERVED spread, which is
    # noisy, so a draw generated at one spread can land in a looser band. Apply the complete banded
    # rule to every draw and report the realised operating characteristic per generative cell.
    table = [(c["band_upper_s_ratio"], c["alpha_benefit"], c["alpha_size"], c["alpha_harm"]) for c in res["bands"]]

    def applied(x):
        band = next((t for t in table if x["s_ratio_hat"] <= t[0]), None)
        if band is None or band[1] is None:
            return "UNCALIBRATED", False
        ben = x["boot_frac_le_0"] < band[1]
        # the size claim attaches to SUPPORTED only, exactly as lc_verdict.calibrated does (the first
        # version also counted REAL_BUT_SMALL draws; the code-vs-analysis canary test caught it)
        size = ben and x["relative_reduction_r"] >= 0.20 and band[2] is not None and x["boot_frac_le_precise"] < band[2]
        harm = band[3] is not None and x["boot_frac_le_0"] > 1 - band[3]
        return ("SUPPORTED" if ben and x["relative_reduction_r"] >= 0.20 else
                "REAL_BUT_SMALL" if ben else "HARMFUL" if harm else "OTHER"), size
    oc = []
    for b in bands:
        for k in s_kls:
            nul, bnd, pw, n2 = (g[(b, k, t)] for t in ("null", "boundary", "power", "n2_harm"))
            ap = lambda xs: [applied(x) for x in xs]
            an, abn, apw, an2 = ap(nul), ap(bnd), ap(pw), ap(n2)
            oc.append({"generative_s_ratio": b, "s_kl": k,
                       "null_false_benefit": sum(v[0] in ("SUPPORTED", "REAL_BUT_SMALL") for v in an) / len(an),
                       "null_false_SUPPORTED": sum(v[0] == "SUPPORTED" for v in an) / len(an),
                       "null_false_HARMFUL": sum(v[0] == "HARMFUL" for v in an) / len(an),
                       "boundary_false_size_claim": sum(v[1] for v in abn) / len(abn),
                       "power_r030_SUPPORTED": sum(v[0] == "SUPPORTED" for v in apw) / len(apw),
                       "power_r030_uncalibrated": sum(v[0] == "UNCALIBRATED" for v in apw) / len(apw),
                       "n2_false_benefit": sum(v[0] in ("SUPPORTED", "REAL_BUT_SMALL") for v in an2) / len(an2)})
    res["operating_characteristic_as_applied"] = oc
    res["max_null_false_benefit_as_applied"] = max(o["null_false_benefit"] for o in oc)
    res["max_boundary_false_size_as_applied"] = max(o["boundary_false_size_claim"] for o in oc)
    res["max_null_false_harm_as_applied"] = max(o["null_false_HARMFUL"] for o in oc)
    json.dump(res, open(out, "w", encoding="utf-8", newline=chr(10)), indent=1)
    print("AS APPLIED: max null false benefit", res["max_null_false_benefit_as_applied"],
          "| max boundary false size", res["max_boundary_false_size_as_applied"],
          "| max null false harm", res["max_null_false_harm_as_applied"])
    for o in oc:
        print(" ", o)
    for c in res["bands"]:
        print(c["band_upper_s_ratio"], "alpha_benefit", c["alpha_benefit"], "alpha_size", c["alpha_size"],
              "alpha_harm", c["alpha_harm"],
              {k: (v["uncalibrated_null_false_benefit_0.025"], v["null_false_benefit_at_alpha"],
                   v["power_r030_SUPPORTED"], v["n2_false_benefit_at_alpha"]) for k, v in c["by_s_kl"].items()})


if __name__ == "__main__":
    main()
