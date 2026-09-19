"""Is the `low` position null SELECTION, or is it real?

    python analysis/position_selection.py --root <records>/alias001-20260918 \
        --out <records>/alias001-20260918/POSITION_SELECTION.json

Amendment 18 found the delta damages late-answer rows more, scored on the source-correct
denominator, in `high` and `equal` but **not** in `low`.

The peer session then proposed that the `low` null is an artifact of the **damage rule itself**.
Their evidence: the source's own accuracy falls with position (rho = -0.296 in `low`, -0.227 in
`equal`, -0.150 ns in `high`) -- the mirror image of the delta's damage gradient. So conditioning
on source-correct removes late-answer rows from the denominator **exactly where the source
already fails on them**, and the test in `low` may simply have no rows left to speak with. They
flagged this as a third route to an uninformative null -- **selection**, alongside wrong endpoint
class and thin denominator.

That is a sharp hypothesis and it is checkable. Five checks, in the order that makes the answer
hard to argue with:

  * **A. Unconditional sensitivity.** Drop the damage rule entirely: correlate position with
    each arm's raw accuracy over all 120 rows. The delta's *excess* over the source is what the
    selection account says is hidden in `low`.
  * **C. Discordant pairs.** The test that would settle it by including source-wrong rows via
    the repair cell -- **not runnable here**, and the reason is itself a result.
  * **D. Were the cells actually emptied?** The account requires the late bins in `low` to be
    depleted of source-correct rows. Bin position into quartiles and look.
  * **E. Power.** A null over populated cells still needs the power to have seen a real effect.
    Ask what `low` *would* have shown had it carried a `high`-sized gradient.
  * **F. Matched support.** Amendment 18 ordered the conditions by rho without checking that
    they span the same position range. They do not -- `high` has one source-correct row past
    0.75 and `low` has 18. Recompute on common support.

F is a check on my own prior claim, not on the peer's, and it is included because the binned
table in D is what exposed the problem.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st  # noqa: F401  (kept: medians used in ad-hoc reruns of this receipt)
from math import comb
from pathlib import Path

CONDITIONS = ["low", "equal", "high"]
QUARTILES = [0.0, 0.25, 0.50, 0.75, 1.01]
MATCHED_SUPPORT = 0.75
PERM_ITERS = 20000


def norm(v) -> str:
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def spearman(x, y) -> float:
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[o[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def perm_p(x, y, seed: int, iters: int = PERM_ITERS) -> float:
    """Permutation p for |rho|. Deterministic given seed."""
    obs = abs(spearman(x, y))
    rnd = random.Random(seed)
    yy = list(y)
    c = 0
    for _ in range(iters):
        rnd.shuffle(yy)
        if abs(spearman(x, yy)) >= obs - 1e-12:
            c += 1
    return c / iters


def fisher(a: int, b: int, c: int, d: int) -> float:
    n, r1, c1 = a + b + c + d, a + b, a + c

    def pr(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    p0 = pr(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 + 1e-12)


def load(root: Path, sub: str, rows: dict) -> dict:
    f = root / sub / "predictions.jsonl"
    if not f.exists():
        raise SystemExit(f"missing predictions: {f}")
    p = {r["id"]: r.get("prediction", "") for r in
         (json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip())}
    miss = [i for i in rows if i not in p]
    if miss:
        raise SystemExit(f"{sub} misses {len(miss)} rows")
    return {i: norm(p[i]) == rows[i]["reference_normalised"] for i in rows}


def mentions(r: dict) -> list[float]:
    c, ref = r["context"], r["reference"]
    out, i = [], c.find(ref)
    while i >= 0:
        out.append(i / len(c))
        i = c.find(ref, i + 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = {r["id"]: r for r in
            (json.loads(x) for x in
             (a.root / "ALIAS_PROBE.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())}
    src = load(a.root, "gen-B", rows)
    lanes = {"native_a1": load(a.root, "gen-a10", rows),
             "lwf_l0": load(a.root, "gen-lwf_l0", rows)}

    def pos(i):
        return mentions(rows[i])[0]

    def ids_of(c):
        return [i for i in rows if rows[i]["condition"] == c]

    out: dict = {
        "record": "POSITION_SELECTION",
        "class": "MECHANISM SCOUT -- not a confirmation, one bank, one draw",
        "question": ("is Amendment 18's `low` position null an artifact of the damage rule "
                     "(the peer session's selection hypothesis), or is it real?"),
        "perm_iters": PERM_ITERS,
    }

    # --- A. unconditional -------------------------------------------------------------
    A = {}
    for c in CONDITIONS:
        ids = ids_of(c)
        p = [pos(i) for i in ids]
        b = [1 if src[i] else 0 for i in ids]
        rb = spearman(p, b)
        per_lane = {}
        for lab, arm in lanes.items():
            v = [1 if arm[i] else 0 for i in ids]
            r = spearman(p, v)
            per_lane[lab] = {"n_correct": sum(v), "rho": round(r, 4),
                             "p": perm_p(p, v, 11), "excess_over_source": round(r - rb, 4)}
        A[c] = {"n": len(ids), "source": {"n_correct": sum(b), "rho": round(rb, 4),
                                          "p": perm_p(p, b, 11)}, "lanes": per_lane}
    out["A_unconditional"] = {
        "what": "position vs raw accuracy, all rows, NO damage rule -- so no selection can act",
        "predicted_by_selection_account": ("in `low` the delta should be MORE position-sensitive "
                                           "than the source; the excess is what conditioning "
                                           "hides"),
        "by_condition": A,
    }

    # --- C. discordant pairs ----------------------------------------------------------
    C = {}
    for lab, arm in lanes.items():
        C[lab] = {c: {"damage": sum(1 for i in ids_of(c) if src[i] and not arm[i]),
                      "repair": sum(1 for i in ids_of(c) if not src[i] and arm[i])}
                  for c in CONDITIONS}
    out["C_discordant_pairs"] = {
        "what": ("the test that WOULD settle it -- compare damage-cell to repair-cell positions, "
                 "so source-wrong rows enter via repairs and the damage rule excludes nothing"),
        "cells": C,
        "runnable": False,
        "why_not": ("the repair cell is empty (0-1 rows per condition per lane). the delta on "
                    "this bank is essentially purely subtractive, so delta-correct is NESTED in "
                    "source-correct and there is no second cell to compare against"),
        "why_that_matters": ("it means conditioning on source-correct discards no information "
                             "about the delta -- on source-wrong rows the delta is almost never "
                             "right either, so those rows are dead for both arms, not withheld "
                             "from one"),
    }

    # --- D. were the cells emptied? ---------------------------------------------------
    D = {}
    for c in CONDITIONS:
        ids = ids_of(c)
        bins = []
        for k in range(4):
            b = [i for i in ids if QUARTILES[k] <= pos(i) < QUARTILES[k + 1]]
            ok = [i for i in b if src[i]]
            dm = {lab: sum(1 for i in ok if not arm[i]) for lab, arm in lanes.items()}
            bins.append({
                "bin": [QUARTILES[k], min(QUARTILES[k + 1], 1.0)],
                "rows": len(b), "source_correct": len(ok),
                "source_rate": round(len(ok) / len(b), 4) if b else None,
                "damaged": dm,
                "damage_rate": {lab: (round(v / len(ok), 4) if ok else None)
                                for lab, v in dm.items()},
            })
        D[c] = bins
    out["D_cell_occupancy"] = {
        "what": "position quartiles: did the damage rule actually EMPTY the late bins in `low`?",
        "by_condition": D,
    }

    # --- E. power ---------------------------------------------------------------------
    def qcells(c, k, lab):
        ids = [i for i in ids_of(c) if QUARTILES[k] <= pos(i) < QUARTILES[k + 1] and src[i]]
        return sum(1 for i in ids if not lanes[lab][i]), len(ids)

    E = {}
    for lab in lanes:
        d1, n1 = qcells("low", 0, lab)
        d4, n4 = qcells("low", 3, lab)
        mdd = next((k for k in range(d1, n4 + 1)
                    if fisher(d1, n1 - d1, k, n4 - k) < 0.05), None)
        hi_lo, hi_n_lo = qcells("high", 0, lab)
        hi_hi, hi_n_hi = qcells("high", 2, lab)
        hi_rate = hi_hi / hi_n_hi if hi_n_hi else 0.0
        cf = round(hi_rate * n4)
        E[lab] = {
            "low_Q1": f"{d1}/{n1}", "low_Q4": f"{d4}/{n4}",
            "low_Q1_vs_Q4_fisher_p": round(fisher(d1, n1 - d1, d4, n4 - d4), 4),
            "high_Q1": f"{hi_lo}/{hi_n_lo}", "high_Q3": f"{hi_hi}/{hi_n_hi}",
            "high_Q1_vs_Q3_fisher_p": fisher(hi_lo, hi_n_lo - hi_lo, hi_hi, hi_n_hi - hi_hi),
            "min_detectable_low_Q4_count": mdd,
            "min_detectable_low_Q4_rate": round(mdd / n4, 4) if mdd else None,
            "counterfactual_low_Q4_at_high_rate": f"{cf}/{n4}",
            "counterfactual_fisher_p": fisher(d1, n1 - d1, cf, n4 - cf),
        }
    out["E_power"] = {
        "what": "a null over populated cells still needs power -- what WOULD `low` have shown?",
        "by_lane": E,
    }

    # --- F. matched support -----------------------------------------------------------
    F = {}
    for lab, arm in lanes.items():
        per_c = {}
        for c in CONDITIONS:
            ok = [i for i in ids_of(c) if src[i]]
            mid = [i for i in ok if pos(i) < MATCHED_SUPPORT]
            fp, fd = [pos(i) for i in ok], [0 if arm[i] else 1 for i in ok]
            mp, md = [pos(i) for i in mid], [0 if arm[i] else 1 for i in mid]
            per_c[c] = {
                "full": {"n": len(ok), "damaged": sum(fd), "rho": round(spearman(fp, fd), 4),
                         "p": perm_p(fp, fd, 31)},
                "matched": {"n": len(mid), "damaged": sum(md), "rho": round(spearman(mp, md), 4),
                            "p": perm_p(mp, md, 31)},
            }
        F[lab] = per_c
    out["F_matched_support"] = {
        "what": f"recompute on common support (position < {MATCHED_SUPPORT})",
        "why": ("Amendment 18 ordered the conditions by rho without matching position support. "
                "`high` has 1 source-correct row past 0.75 and `low` has 18, so the ordering "
                "could have been a support difference. This is a check on MY claim, not the "
                "peer's"),
        "by_lane": F,
    }

    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------------------------
    print("A. UNCONDITIONAL -- no damage rule, so selection cannot act")
    print(f"   {'cond':<7}{'src rho':>9}{'p':>8}   {'delta rho':>10}{'p':>8}{'excess':>9}")
    for c in CONDITIONS:
        s = A[c]["source"]
        n = A[c]["lanes"]["native_a1"]
        print(f"   {c:<7}{s['rho']:>+9.3f}{s['p']:>8.4f}   {n['rho']:>+10.3f}{n['p']:>8.4f}"
              f"{n['excess_over_source']:>+9.3f}")
    print("\nD. CELL OCCUPANCY (native) -- were `low`'s late bins emptied?")
    for c in CONDITIONS:
        cells = " | ".join(f"{b['source_correct']:>3} ok/{b['damaged']['native_a1']:>2} dmg"
                           for b in D[c])
        print(f"   {c:<7}{cells}")
    print("\nE. POWER (native, low Q1 vs Q4)")
    e = E["native_a1"]
    print(f"   observed {e['low_Q1']} vs {e['low_Q4']}, p={e['low_Q1_vs_Q4_fisher_p']:.4f}; "
          f"a high-sized {e['counterfactual_low_Q4_at_high_rate']} would give "
          f"p={e['counterfactual_fisher_p']:.2e}")
    print(f"   minimum detectable in low Q4: {e['min_detectable_low_Q4_count']}/{e['low_Q4'][-2:]}")
    print("\nF. MATCHED SUPPORT")
    for lab in lanes:
        print(f"   {lab}")
        for c in CONDITIONS:
            v = F[lab][c]
            print(f"     {c:<7}full {v['full']['rho']:>+.3f} (p={v['full']['p']:.4f}, "
                  f"n={v['full']['n']})   matched {v['matched']['rho']:>+.3f} "
                  f"(p={v['matched']['p']:.4f}, n={v['matched']['n']})")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
