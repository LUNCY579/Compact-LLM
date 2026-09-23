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

AND THE BETWEEN-DRAW FLOOR DOES NOT THREATEN A BOUND. The floor exists because cross-draw
comparisons can show SPURIOUS DIFFERENCES; it cannot manufacture a spurious SIMILARITY.
Cross-draw variation inflates an observed difference rather than shrinking one, so a small
observed difference from a cross-draw comparison is CONSERVATIVE. (The author got an adjacent
asymmetry argument wrong earlier in this lane -- that one invoked a cell that was not a null.
This one is about the direction of a bias on a bound, which is a different claim.)

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
    """Exact Clopper-Pearson upper limit: smallest p with P(X<=k | n,p) <= alpha."""
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
          f"{'CP95 rate':>10s} {'|net| <=':>9s}")
    out = {}
    for a, b in (("perp", "control"), ("perp", "a0554"), ("a0554", "control"), ("h025", "control")):
        ks = sorted(set(R[a]) & set(R[b]))
        x = sum(1 for k in ks if R[a][k] and not R[b][k])
        y = sum(1 for k in ks if R[b][k] and not R[a][k])
        u = cp_upper(x + y, len(ks))
        out[(a, b)] = u * len(ks)
        print(f"  {a+' vs '+b:18s} {sum(R[a][k] for k in ks):4d} {sum(R[b][k] for k in ks):4d} "
              f"{x:3d} {y:3d} {x+y:3d} {u:10.4f} {u*len(ks):8.1f}r")

    w = out[("perp", "control")]
    print(f"\nTHE ONE THAT CARRIES THE CONCLUSION:")
    print(f"  arm-perp's TRUE boundary effect vs the control is within +/- {w:.1f} rows at 95%.")
    print(f"  delta_H's CERTIFIED effect is +{DELTA_H_CERTIFIED}.")
    verdict = w < DELTA_H_CERTIFIED
    print(f"  {w:.1f} < {DELTA_H_CERTIFIED} -> THE BOUND EXCLUDES delta_H's EFFECT."
          if verdict else f"  {w:.1f} >= {DELTA_H_CERTIFIED} -> bound does NOT exclude it; claim nothing.")
    assert verdict, "bound fails to exclude the certified effect -- the conclusion does not hold"
    print("\n  LICENSED:     arm-perp does not have delta_H's boundary effect.")
    print("  NOT LICENSED: arm-perp has no boundary effect.")


if __name__ == "__main__":
    raise SystemExit(main())
