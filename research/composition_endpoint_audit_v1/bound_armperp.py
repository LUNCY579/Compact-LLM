"""POST-HOC bounds for ARMPERP-001. NOT a registered test. Read the header before the numbers.

WHY THIS EXISTS. The registered primary produced a state the outcome table has no cell for:
one limb resolved, one blocked by P2. Three separate attempts -- two mine, one a peer's -- then
tried to rescue a conclusion by reading an UNRESOLVED cell (discordance < 6) as a null. P2 was
registered precisely to forbid that, and it forbids it in the rescue as much as in the verdict.

A BOUND IS A DIFFERENT OBJECT AND IT IS AVAILABLE. |net| is bounded above by the number of
DISCORDANT pairs, so an exact upper limit on the discordance RATE bounds the effect. That is a
claim an unresolved McNemar can support, because it does not assert absence -- it asserts
smallness, with a stated width.

    LICENSED      "arm-perp does not have delta_H's boundary effect"
    NOT LICENSED  "arm-perp has no boundary effect"    -- the bound is 7.6 rows wide, not zero

WHERE THE BETWEEN-DRAW ISSUE ACTUALLY LIVES -- and it is NOT a noise-asymmetry argument.

An earlier version of this header argued that cross-draw noise "inflates an observed difference
and does not shrink one", so a small observed difference is conservative. THAT IS FALSE and it
is withdrawn. Writing D = delta + eps with E[eps] = 0, eps can take the opposite sign to delta
and MASK it, so a small |D| can understate |delta| badly on a single draw. Jensen gives
E|D| >= |E[D]| = |delta|, which is a statement about the AVERAGE OVER DRAWS, not a bound on the
one draw in hand. (Second asymmetry argument this lane has got wrong; a peer caught both.)

The bound does not need it. The CP limit is computed on the DISCORDANCE BETWEEN TWO SPECIFIC
ARMS, row by row, over shared rows. It is a within-pair statement about two objects on disk and
draw variability is not inside it. The draw issue lives one level up, in GENERALISATION:

    LICENSED      THIS arm-perp does not have delta_H's boundary effect. A property of the
                  built arm; no draw argument required.
    NOT LICENSED  Orthogonalised same-skill content IN GENERAL does not have delta_H's
                  boundary effect. That quantifies over draws of s23 -- a different s23 gives
                  a different d_perp and a different arm. The between-draw floor bites HERE,
                  and no bound on one realisation settles it.

The registered question was whether THIS construction manufactures the gain, so the scoped
claim is the one the experiment asked for.

THIS IS POST-HOC. No equivalence test was registered in ARMPERP_PREREG.md. These are additional
descriptive statistics on the registered data, not a reinterpretation of the registered test.
"""
import json
from math import comb

REC = r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records"
H = {"perp":   f"{REC}/armperp001-20260923/evidence/HSKILL2_armperp001_perp.json",
     "h025":   f"{REC}/indep003-20260921/evidence/HSKILL2_indep003_s17_h025.json",
     "a0554":  f"{REC}/indep005-20260921/evidence/HSKILL2_indep005_same_a0554.json",
     "control":f"{REC}/indep001-20260921/evidence/HSKILL2_indep001_t100h000.json"}
DELTA_H_CERTIFIED = 16          # h025 vs control, +16, nd=24, p=1.544e-03, FLOORLESS


def rows(p):
    j = json.load(open(p, encoding="utf-8"))
    return {k: v["credited"] for k, v in j["per_row"].items() if v["row_type"] == "boundary"}


def cp_upper(k, n, alpha=0.05):
    """Exact Clopper-Pearson upper limit: smallest p with P(X<=k | n,p) <= alpha.

    CONVENTION MATTERS AND MUST BE STATED. alpha=0.05 is the ONE-SIDED 95% upper limit;
    alpha=0.025 is the upper limit of the TWO-SIDED 95% interval. A reader recomputing with
    the other convention gets a different number and cannot tell which of you is wrong.
    Both are reported below; the conclusion is identical under either.
    """
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if sum(comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1)) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    R = {k: rows(v) for k, v in H.items()}
    assert len({len(v) for v in R.values()}) == 1 == len({112}), "boundary row counts differ"
    print("BOUNDARY, n=112. Exact CP 95% upper bound on the DISCORDANCE RATE, and the worst-case")
    print("net it permits (every discordant pair falling one way).\n")
    print(f"  {'contrast':18s} {'A':>4s} {'B':>4s} {'b':>3s} {'c':>3s} {'nd':>3s} "
          f"{'1-sided':>9s} {'|net|<=':>8s}  {'2-sided':>9s} {'|net|<=':>8s}")
    out = {}
    for a, b in (("perp", "control"), ("perp", "a0554"), ("a0554", "control"), ("h025", "control")):
        ks = sorted(set(R[a]) & set(R[b]))
        x = sum(1 for k in ks if R[a][k] and not R[b][k])
        y = sum(1 for k in ks if R[b][k] and not R[a][k])
        u1, u2 = cp_upper(x + y, len(ks), 0.05), cp_upper(x + y, len(ks), 0.025)
        out[(a, b)] = (u1 * len(ks), u2 * len(ks))
        print(f"  {a+' vs '+b:18s} {sum(R[a][k] for k in ks):4d} {sum(R[b][k] for k in ks):4d} "
              f"{x:3d} {y:3d} {x+y:3d} {u1:9.4f} {u1*len(ks):7.2f}r  {u2:9.4f} {u2*len(ks):7.2f}r")

    w1, w2 = out[("perp", "control")]
    print(f"\nTHE ONE THAT CARRIES THE CONCLUSION:")
    print(f"  arm-perp's boundary effect vs the control is within +/- {w1:.2f} rows (one-sided 95%)")
    print(f"                                              or +/- {w2:.2f} rows (two-sided 95%).")
    print(f"  delta_H's CERTIFIED effect is +{DELTA_H_CERTIFIED}.")
    verdict = max(w1, w2) < DELTA_H_CERTIFIED
    print(f"  Both below {DELTA_H_CERTIFIED} -> THE BOUND EXCLUDES delta_H's EFFECT under EITHER convention."
          if verdict else "  bound does NOT exclude it under both conventions; claim nothing.")
    assert verdict, "bound fails to exclude the certified effect -- the conclusion does not hold"
    print("\n  LICENSED:     THIS arm-perp does not have delta_H's boundary effect.")
    print("  NOT LICENSED: arm-perp has no boundary effect (the bound is ~8 rows wide, not 0),")
    print("                nor anything about orthogonalised same-skill content IN GENERAL,")
    print("                which would quantify over draws of s23.")


if __name__ == "__main__":
    raise SystemExit(main())
