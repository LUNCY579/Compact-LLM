"""ARMPERP-001 scoring, exactly as registered in ARMPERP_PREREG.md section 6.

PRIMARY: boundary (n=112), exact paired two-sided McNemar vs EACH comparator, Holm across the
two. P2 resolution check: discordance < 6 cannot reach p<0.05, so that comparison is
INDETERMINATE against that comparator, NOT agreement.

CANARIES. The comparator boundary counts 85 / 68 / 69 are recorded in INDEP005_RESULT.md's
four-arm table. This script recomputes them from per_row and ASSERTS them. If they do not
reproduce, the extractor is wrong and nothing else it prints means anything -- four wrong
extractors in one day were caught this way and none by reading code.
"""
import json
from math import comb

REC = r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records"
ARMS = {
    "arm-perp":    (f"{REC}/armperp001-20260923/evidence/HSKILL2_armperp001_perp.json",     None),
    "h025 (dH)":   (f"{REC}/indep003-20260921/evidence/HSKILL2_indep003_s17_h025.json",      85),
    "a0554 (same)":(f"{REC}/indep005-20260921/evidence/HSKILL2_indep005_same_a0554.json",    68),
    "control":     (f"{REC}/indep001-20260921/evidence/HSKILL2_indep001_t100h000.json",      69),
}


def rows(path, kind):
    j = json.load(open(path, encoding="utf-8"))
    return {k: v["credited"] for k, v in j["per_row"].items() if v["row_type"] == kind}


def mcnemar(A, B):
    ks = sorted(set(A) & set(B))
    b = sum(1 for k in ks if A[k] and not B[k])
    c = sum(1 for k in ks if B[k] and not A[k])
    n = b + c
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return b, c, n, p, min(1.0, 2 / 2 ** n) if n else 1.0, len(ks)


def main():
    bnd, ans = {}, {}
    print("CANARY -- comparator boundary counts must reproduce from per_row:")
    for name, (path, expect) in ARMS.items():
        bnd[name] = rows(path, "boundary"); ans[name] = rows(path, "answerable")
        got = sum(bnd[name].values())
        if expect is None:
            print(f"  {name:14s} boundary {got:3d}/112   (arm under test, no recorded value)")
        else:
            ok = got == expect
            print(f"  {name:14s} boundary {got:3d}/112   recorded {expect}   {'MATCH' if ok else 'MISMATCH'}")
            assert ok, f"CANARY FAILED on {name}: {got} vs recorded {expect}"
    print("  all comparators reproduce -- extractor validated by collision\n")

    A = bnd["arm-perp"]; nA = sum(A.values())
    print(f"PRIMARY -- boundary, arm-perp = {nA}/112, exact paired two-sided McNemar\n")
    print(f"  {'vs':16s} {'their':>6s} {'net':>5s} {'b':>3s} {'c':>3s} {'disc':>5s} {'p':>11s} {'Holm':>11s} {'verdict':>18s}")
    res = {}
    for name in ("h025 (dH)", "a0554 (same)", "control"):
        b, c, n, p, minp, k = mcnemar(A, bnd[name])
        res[name] = dict(their=sum(bnd[name].values()), net=nA - sum(bnd[name].values()),
                         b=b, c=c, disc=n, p=p, minp=minp)
    # Holm across the TWO registered comparisons only (control is descriptive)
    reg = ["h025 (dH)", "a0554 (same)"]
    order = sorted(reg, key=lambda x: res[x]["p"])
    for i, name in enumerate(order):
        res[name]["holm"] = min(1.0, res[name]["p"] * (len(reg) - i))
    for i in range(1, len(order)):
        res[order[i]]["holm"] = max(res[order[i]]["holm"], res[order[i-1]]["holm"])
    res["control"]["holm"] = None

    for name in ("h025 (dH)", "a0554 (same)", "control"):
        r = res[name]
        if r["disc"] < 6:
            v = "INDETERMINATE (P2)"
        elif r["holm"] is None:
            v = "descriptive"
        else:
            v = "SIG DIFFERENT" if r["holm"] < 0.05 else "not sig"
        h = f"{r['holm']:.3e}" if r["holm"] is not None else "        --"
        print(f"  {name:16s} {r['their']:6d} {r['net']:+5d} {r['b']:3d} {r['c']:3d} {r['disc']:5d} "
              f"{r['p']:11.3e} {h:>11s} {v:>18s}")

    h, a = res["h025 (dH)"], res["a0554 (same)"]
    # P2 IS A PRECONDITION ON READING A LIMB AT ALL, NOT A FOOTNOTE ON THE VERDICT.
    # A comparison with discordance < 6 cannot reach p<0.05, so "not significant" there is NOT
    # "not different" -- it is UNRESOLVED. Letting it satisfy a "NOT significantly different"
    # clause is precisely what P2 was registered to forbid. The first version of this function
    # did exactly that and printed CONTENT; a peer scoring the artifacts cold caught it.
    resolved = lambda r: r["disc"] >= 6
    sig = lambda r: resolved(r) and r["holm"] is not None and r["holm"] < 0.05
    nsig = lambda r: resolved(r) and not sig(r)          # RESOLVED *and* not significant
    print("\nREGISTERED OUTCOME (P2 applied as a precondition on each limb):")
    print(f"  a0554 limb: discordance {a['disc']:2d} -> {'RESOLVED' if resolved(a) else 'UNRESOLVED (P2)'}")
    print(f"  h025  limb: discordance {h['disc']:2d} -> {'RESOLVED' if resolved(h) else 'UNRESOLVED (P2)'}")
    if sig(a) and nsig(h):
        # ERRATUM 2026-09-23: the rank figure quoted here was min 21 / median 23 / max 27, which
        # was a 21-of-140 sample that sorted-key order confined to layers 10-12 ONLY. Measured on
        # the excluded layers: L8-9 gives min 20 median 22.5 max 27, L26-27 gives min 16 median
        # 24.0 max 26. Over all three groups (49 tensors, 7 layers): MIN 16, MEDIAN 24, MAX 27.
        # The reported min was too high and the true min equals the comparators' rank exactly.
        print("  GEOMETRY. MUST be reported as confounded with rank "
              "(median 24 vs comparators 16, over 49 tensors spanning L8-12 and L26-27; "
              "the earlier median-23 figure sampled L10-12 only).")
    elif nsig(a) and sig(h):
        print("  CONTENT. Rank confound does not apply -- it explains a gain, not its absence.")
    elif sig(a) and sig(h):
        print("  INTERMEDIATE -- significantly different from BOTH. Claim neither.")
    elif resolved(a) and resolved(h):
        print("  INDETERMINATE -- both limbs resolved, neither significant. Claim nothing.")
    else:
        print("  *** NO REGISTERED CELL FITS: one limb resolved, one UNRESOLVED by P2. ***")
        print("  The four-outcome table assumed both limbs would be readable. They were not.")
        if sig(h):
            print(f"  LICENSED: arm-perp is SIGNIFICANTLY BELOW delta_H ({h['their']}) -- "
                  f"net {h['net']:+d}, b={h['b']} c={h['c']}, Holm {h['holm']:.3e}.")
            print("            ORTHOGONALITY ALONE DOES NOT BUY THE BOUNDARY GAIN.")
        print("  NOT LICENSED: any claim that arm-perp EQUALS the same-skill arm. That needs an")
        print("            equivalence test registered in advance, not this one reread.")
    # SECOND defect in the same table, found independently: the CONTENT and INERT readings have
    # reference points 1 row apart (a0554 68, control 69) at discordance 1. No arm could ever
    # have separated them, so even a clean CONTENT could not have excluded INERT.
    ctl = res["control"]
    print(f"\n  and the two readings' REFERENCE POINTS are themselves unresolvable:")
    print(f"    a0554 (CONTENT ref) {a['their']} vs control (INERT ref) {ctl['their']} "
          f"-- 1 row apart, discordance 1. CONTENT and INERT were never separable.")

    print(f"\nSECONDARY (descriptive, no prediction): answerable")
    for name in ARMS:
        print(f"  {name:16s} {sum(ans[name].values()):3d}/112")
    acq = json.load(open(f"{REC}/armperp001-20260923/evidence/PROCESS_armperp001_perp.json",
                        encoding="utf-8"))
    print(f"\n  acquisition {acq['aggregate']['canonical_correct']}/{acq['views']}  "
          f"bank {str(acq.get('views_sha256',''))[:8]}  mnt {acq.get('max_new_tokens')}  "
          f"(h025 235, a0554 241, control 235)")
    assert str(acq.get("views_sha256", ""))[:8] == "7dd53f2a" and acq.get("max_new_tokens") == 96, \
        "acquisition leg is NOT comparable -- wrong bank or decode budget"
    print("  comparability asserted: bank 7dd53f2a, max_new_tokens 96")


if __name__ == "__main__":
    raise SystemExit(main())
