"""HLOC-001 scoring, exactly as registered in HLOC_PREREG.md section 3 + Amendment 1.

WRITTEN BEFORE THE BATTERY FINISHED, deliberately. The ARMPERP scorer was written after its
numbers existed and its verdict logic negated straight past its own P2 guard; writing the
decision procedure while the outcome is still unknown removes that degree of freedom.

P2 IS A PRECONDITION, NOT A FOOTNOTE. A comparison too small to reach significance is
UNRESOLVED and CANNOT satisfy any negative clause.

TWO CORRECTIONS after the first scoring, both found by a peer scoring cold, neither changing
the verdict (INDETERMINATE either way, and more firmly):
  (a) P2 registered nd < 6, which is the threshold for an UNADJUSTED exact test. This design
      applies Holm across m=2, so the smallest resolvable discordance is nd >= 7. DERIVED
      below from m and alpha rather than hardcoded.
  (b) `not above(x)` is TRUE for an UNRESOLVED limb, so 4 of 16 limb-state pairs printed a
      substantive verdict with a limb that could not be read. Fixed by HOISTING a
      both-resolved guard ABOVE every branch -- patching individual clauses is what failed
      in ARMPERP, and this is the third occurrence of that shape.

AMENDMENT 1 split the old single H_DOSE cell, which absorbed two patterns that read oppositely:
  both ablations indistinguishable from their scalars -> location does not matter  (H_DOSE)
  one or both significantly BELOW their scalars       -> restriction is harmful    (H_RESTRICTION)
"""
import json
from math import comb

REC = r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records"
H = f"{REC}/hloc001-20260923"
ARMS = {
    "hattn":    (f"{H}/HSKILL2_hloc001_hattn.json",     None),
    "hattn_s":  (f"{H}/HSKILL2_hloc001_hattn_s.json",   None),
    "hmlp":     (f"{H}/HSKILL2_hloc001_hmlp.json",      None),
    "hmlp_s":   (f"{H}/HSKILL2_hloc001_hmlp_s.json",    None),
    # CANARIES -- recorded values that must reproduce, or the extractor is wrong
    "h025":     (f"{REC}/indep003-20260921/evidence/HSKILL2_indep003_s17_h025.json", 85),
    "control":  (f"{REC}/indep001-20260921/evidence/HSKILL2_indep001_t100h000.json", 69),
}
REGISTERED = [("hattn", "hattn_s"), ("hmlp", "hmlp_s")]


def rows(p):
    j = json.load(open(p, encoding="utf-8"))
    return {k: v["credited"] for k, v in j["per_row"].items() if v["row_type"] == "boundary"}


def mcnemar(A, B):
    ks = sorted(set(A) & set(B))
    b = sum(1 for k in ks if A[k] and not B[k])
    c = sum(1 for k in ks if B[k] and not A[k])
    n = b + c
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return sum(A[k] for k in ks), sum(B[k] for k in ks), b, c, n, p, (min(1.0, 2 / 2 ** n) if n else 1.0)


def MIN_ND_FOR(m, alpha=0.05):
    """Smallest discordance whose MINIMUM attainable two-sided p survives Holm over m tests."""
    return next(n for n in range(1, 60) if (2 / 2 ** n) * m < alpha)


def main():
    R = {}
    print("CANARIES -- recorded comparator counts must reproduce from per_row:")
    for name, (path, exp) in ARMS.items():
        try:
            R[name] = rows(path)
        except FileNotFoundError:
            print(f"  {name:10s} ARTIFACT MISSING -- battery incomplete"); return 2
        got = sum(R[name].values())
        if exp is None:
            print(f"  {name:10s} boundary {got:3d}/112   (arm under test)")
        else:
            print(f"  {name:10s} boundary {got:3d}/112   recorded {exp}   {'MATCH' if got == exp else 'MISMATCH'}")
            assert got == exp, f"CANARY FAILED on {name}"
    print("  extractor validated by collision\n")

    res = {}
    for a, b in REGISTERED:
        A, B, x, y, n, p, mp = mcnemar(R[a], R[b])
        res[(a, b)] = dict(A=A, B=B, b=x, c=y, nd=n, p=p, minp=mp)
    order = sorted(REGISTERED, key=lambda k: res[k]["p"])
    for i, k in enumerate(order):
        res[k]["holm"] = min(1.0, res[k]["p"] * (len(REGISTERED) - i))
    for i in range(1, len(order)):
        res[order[i]]["holm"] = max(res[order[i]]["holm"], res[order[i - 1]]["holm"])

    print("PRIMARY -- boundary, ablation vs its OWN norm-matched scalar control")
    print(f"  {'comparison':22s} {'abl':>4s} {'scalar':>7s} {'net':>5s} {'b':>3s} {'c':>3s} "
          f"{'nd':>3s} {'p':>10s} {'Holm':>10s}  {'limb':>12s}")
    for a, b in REGISTERED:
        r = res[(a, b)]
        limb = ("UNRESOLVED" if r["nd"] < MIN_ND_FOR(len(REGISTERED))
                else ("SIG" if r["holm"] < 0.05 else "resolved null"))
        print(f"  {a + ' vs ' + b:22s} {r['A']:4d} {r['B']:7d} {r['A']-r['B']:+5d} {r['b']:3d} "
              f"{r['c']:3d} {r['nd']:3d} {r['p']:10.3e} {r['holm']:10.3e}  {limb:>12s}")

    # --- P2 THRESHOLD, DERIVED FROM THE CORRECTION ACTUALLY APPLIED, not assumed ---------
    # P2 registered nd < 6 as UNRESOLVED. That is the threshold for an UNADJUSTED exact test.
    # This design applies Holm across m comparisons, so the smallest-p limb must clear
    # alpha/m. Derive it instead of hardcoding 6:
    m, ALPHA = len(REGISTERED), 0.05
    MIN_ND = MIN_ND_FOR(m, ALPHA)
    resolved = lambda k: res[k]["nd"] >= MIN_ND

    # --- STRUCTURAL GUARD: no substantive verdict while ANY limb is unresolved ------------
    # `not above(x)` is TRUE for an unresolved limb, so an unresolved limb SATISFIES A
    # NEGATIVE CLAUSE and four of sixteen limb-state pairs printed a substantive verdict.
    # Identical in form to ARMPERP's `not sig(a)`; THIRD occurrence of this shape. Patching
    # each branch is what failed twice, so the guard is hoisted above all of them.
    both_resolved = all(resolved(k) for k in REGISTERED)

    above = lambda k: resolved(k) and res[k]["holm"] < ALPHA and res[k]["A"] > res[k]["B"]
    below = lambda k: resolved(k) and res[k]["holm"] < ALPHA and res[k]["A"] < res[k]["B"]
    nsig = lambda k: resolved(k) and res[k]["holm"] >= ALPHA
    ka, km = REGISTERED[0], REGISTERED[1]

    print(f"\nREGISTERED OUTCOME (P2 as a precondition; threshold DERIVED for Holm m={m}):")
    print(f"  smallest resolvable discordance under Holm x{m}: nd >= {MIN_ND}"
          f"   (nd={MIN_ND-1} gives min attainable p {2/2**(MIN_ND-1):.5f} -> {m*2/2**(MIN_ND-1):.5f})")
    for k in REGISTERED:
        print(f"  {k[0]:8s} limb: nd {res[k]['nd']:2d} -> {'RESOLVED' if resolved(k) else 'UNRESOLVED (P2)'}")
    if not both_resolved:
        print("  => INDETERMINATE -- at least one limb is UNRESOLVED, so NO substantive outcome\n"
              "     is readable. Guard hoisted above the branches: an unresolved limb must not\n"
              "     be able to satisfy a negative clause.")
        v = None
    elif above(ka) and not above(km):
        v = ("H_ATTN -- the gain is ATTENTION-carried. Contract-in-attention on a second skill.")
    elif above(km) and not above(ka):
        v = ("H_MLP -- MLP-carried. Contradicts the sibling lane's localisation.")
    elif above(ka) and above(km):
        v = ("H_BOTH -- distributed. Report both; claim no single carrier.")
    elif (below(ka) or below(km)) and not (above(ka) or above(km)):
        v = ("H_RESTRICTION -- restriction to a single module family is HARMFUL whichever "
             "family it is.\n  The carrier question is malformed for this endpoint: the "
             "manipulation damages something\n  the matched scalar does not.")
    elif nsig(ka) and nsig(km):
        v = ("H_DOSE -- both limbs RESOLVED and null. Location does not matter at fixed norm; "
             "the ablation axis is closed for this endpoint.")
    else:
        v = ("INDETERMINATE -- a limb needed for every registered outcome is UNRESOLVED. "
             "Claim nothing;\n  report min_attainable_p. Registered so an underpowered result "
             "cannot be read as H_DOSE.")
    if v is not None:
        print(f"  => {v}")

    print("\nDESCRIPTIVE, no prediction registered (norms differ ~10%):")
    A, B, x, y, n, p, mp = mcnemar(R["hattn"], R["hmlp"])
    print(f"  hattn vs hmlp   {A} vs {B}   net {A-B:+d}   nd {n}   p {p:.3e}   min_p {mp:.2e}")
    print(f"  sibling lane's equality was 68 vs 67 across a 50% energy gap; mine is a ~10% gap.")
    print("\nFOR CONTEXT ONLY -- NOT a registered comparison (total norms differ):")
    print(f"  h025 (full delta_H) {sum(R['h025'].values())}   control {sum(R['control'].values())}")


if __name__ == "__main__":
    raise SystemExit(main())
