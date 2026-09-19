"""MPOS-001 scorer — the arm x band interaction on matched pairs.

    python probe/score_mpos.py --root <records>/mpos001-20260919 \
        --out <records>/mpos001-20260919/MPOS_RESULT.json

**Written before any arm was generated and before any prediction existed.** Registered in
`probe/MPOS001_PREREG.md`.

## What it tests

Each scenario appears in BOTH bands built from identical material — same names, distractors,
question and sentences, differing only in arrangement. So for each pair and each arm:

    d_arm = 1[arm correct on EARLY] - 1[arm correct on LATE]      in {-1, 0, +1}

and the primary statistic is **`d_delta - d_source` per pair**, tested against zero with an
**exact sign test** over the pairs where it is non-zero. Bonferroni over 4 = **0.0125**
(2 clean cells x 2 lanes).

**Direction registered: positive** — moving the answer late costs the DELTA more than the
SOURCE. A significant negative value falsifies the positional account as an intervention.

## Why the interaction rather than the damage rate

A deterministic answer-agnostic rule emits the same string for both arms, so it contributes
equally to `d_source` and `d_delta` and **cancels in the difference**. The absolute band
comparison has no such protection: it is between rows, and a band-dependent cue changes which
rows are solvable at all. On this bank `argmin` of each candidate's mean mention position scores
0.812-0.938 in `early` and 0.000 in `late` **in the d=3 cells** — which is why only `k1d1` and
`k3d1`, where that swing is exactly 0.000, are scored here.

## The denominator that bounds the estimate

**Discordant pairs**, not the 48 pairs per cell. An exact sign test needs >= 6 discordant pairs
for p < 0.05 and more under correction, so the realised discordant count is printed whatever the
outcome, and a thin one is reported as *"not powered to distinguish"* rather than as a null.
That is the denominator this programme has got wrong before.

## Secondary, registered: the cue-exploitation differential ON THIS BANK

The one confound the interaction cannot cancel is a cue whose *exploitability* differs between
arms. Bounded on ALIAS-001 at 0.211 (Newcombe method 10, two-sided 95% upper). Re-measured here
on this bank's own errors, against its **declared candidate pool** -- never a regex over
capitalised tokens, which is how the first attempt at this measurement selected a venue name
present in 360/360 rows. Rows where the cue IS the reference are excluded: an error cannot match
the cue there, so conditioning on error already selects against the heuristic succeeding.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

CLEAN_CELLS = ("k1d1", "k3d1")
BANDS = ("early", "late")
LANES = {"native_a100": "gen-a100", "lwf_l0": "gen-lwf_l0"}
SOURCE = "gen-B"
BONFERRONI = 0.05 / 4


def norm(v) -> str:
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided sign test: P(|X - n/2| >= |k - n/2|) under Binomial(n, 1/2)."""
    if n == 0:
        return 1.0
    obs = abs(k - n / 2)
    return sum(comb(n, x) for x in range(n + 1) if abs(x - n / 2) >= obs - 1e-12) / 2 ** n


def min_discordant_for_significance(alpha: float) -> int:
    """Smallest all-one-way discordant count whose exact two-sided p clears alpha."""
    n = 1
    while n < 200:
        if binom_two_sided(n, n) < alpha:
            return n
        n += 1
    return -1


def load_preds(root: Path, sub: str, ids: set) -> dict:
    f = root / sub / "predictions.jsonl"
    if not f.exists():
        raise SystemExit(f"missing predictions: {f}")
    p = {r["id"]: r.get("prediction", "") for r in
         (json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip())}
    miss = [i for i in ids if i not in p]
    if miss:
        raise SystemExit(f"{sub} misses {len(miss)} of {len(ids)} scored rows, e.g. {miss[:3]}")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(x) for x in
            (a.root / "MPOS_PROBE.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    scored = [r for r in rows if r["cell"] in CLEAN_CELLS]
    ids = {r["id"] for r in scored}
    by_id = {r["id"]: r for r in scored}

    # pairs must be complete, or the interaction is not defined
    pairs: dict = {}
    for r in scored:
        pairs.setdefault(r["pair_id"], {})[r["band"]] = r
    bad = [k for k, v in pairs.items() if set(v) != set(BANDS)]
    if bad:
        raise SystemExit(f"{len(bad)} incomplete pairs, e.g. {bad[:3]}")

    preds = {"source": load_preds(a.root, SOURCE, ids)}
    for lane, sub in LANES.items():
        preds[lane] = load_preds(a.root, sub, ids)

    def ok(arm, rid):
        return norm(preds[arm][rid]) == by_id[rid]["reference_normalised"]

    mde = min_discordant_for_significance(BONFERRONI)
    results: dict = {}
    for cell in CLEAN_CELLS:
        cp = {k: v for k, v in pairs.items() if v["early"]["cell"] == cell}
        per_lane = {}
        for lane in LANES:
            pos = neg = zero = 0
            ds_sum = dd_sum = 0
            detail = []
            for k, v in sorted(cp.items()):
                e, l = v["early"]["id"], v["late"]["id"]
                d_src = int(ok("source", e)) - int(ok("source", l))
                d_dlt = int(ok(lane, e)) - int(ok(lane, l))
                ds_sum += d_src
                dd_sum += d_dlt
                diff = d_dlt - d_src
                if diff > 0:
                    pos += 1
                elif diff < 0:
                    neg += 1
                else:
                    zero += 1
                detail.append({"pair": k, "d_source": d_src, "d_delta": d_dlt, "diff": diff})
            disc = pos + neg
            p = binom_two_sided(pos, disc)
            n_pairs = len(cp)
            per_lane[lane] = {
                "n_pairs": n_pairs,
                "discordant": disc, "pos": pos, "neg": neg, "tied": zero,
                "source_position_cost": round(ds_sum / n_pairs, 4),
                "delta_position_cost": round(dd_sum / n_pairs, 4),
                "interaction_mean": round((dd_sum - ds_sum) / n_pairs, 4),
                "sign_test_p": p,
                "clears_bonferroni": p < BONFERRONI,
                "powered": disc >= mde,
                "verdict": (
                    "not powered to distinguish -- an all-one-way result needs "
                    f">= {mde} discordant pairs to clear {BONFERRONI:.4f}, and there are {disc}"
                    if disc < mde else
                    ("SUPPORTS the registered direction" if p < BONFERRONI and pos > neg else
                     "FALSIFIES the registered direction" if p < BONFERRONI else
                     "no significant interaction at the corrected bound")),
                "pairs": detail,
            }
        results[cell] = per_lane

    # --- secondary: cue exploitation on THIS bank's own errors --------------------------
    pool = sorted({r["reference"] for r in rows})

    def earliest_person(ctx):
        pres = {c: ctx.find(c) for c in pool if c in ctx}
        return min(pres, key=pres.get) if pres else None

    cue = {}
    for arm in ["source", *LANES]:
        hit = scorable = errors = 0
        for r in scored:
            if ok(arm, r["id"]):
                continue
            errors += 1
            c = earliest_person(r["context"])
            if c is None or c == r["reference"]:
                continue
            scorable += 1
            if norm(preds[arm][r["id"]]) == norm(c):
                hit += 1
        cue[arm] = {"errors": errors, "scorable": scorable, "cue_match": hit,
                    "rate": round(hit / scorable, 4) if scorable else None}

    out = {
        "record": "MPOS_RESULT",
        "class": "MECHANISM SCOUT -- one bank, one draw. Not a confirmation.",
        "prereg": "probe/MPOS001_PREREG.md",
        "scored_cells": list(CLEAN_CELLS),
        "excluded_cells": {
            "k1d3": "not readable -- argmin cue swing 0.812, manufacturable 0.171 vs predicted "
                    "ceiling 0.08",
            "k3d3": "not readable -- argmin cue swing 0.938, manufacturable 0.198 vs predicted "
                    "0.10-0.30",
        },
        "bonferroni": BONFERRONI,
        "min_discordant_for_significance": mde,
        "primary": results,
        "secondary_cue_exploitation": cue,
        "caveats": [
            "the interaction cancels a cue available to BOTH arms; it does NOT cancel a cue "
            "whose exploitability differs between them. that residual is bounded at 0.211 on "
            "ALIAS-001 and re-measured here as a secondary",
            "absolute per-band rates are not reported: they are the quantity the cue "
            "contaminates, and reporting them beside the interaction invites the comparison "
            "this design exists to avoid",
            "one bank, one draw, two cells, two lanes -- nothing here promotes anything",
        ],
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"MPOS-001 -- interaction d_delta - d_source, exact sign test, "
          f"Bonferroni {BONFERRONI:.4f}")
    print(f"(an all-one-way result needs >= {mde} discordant pairs to clear that bound)\n")
    print(f"{'cell':<7}{'lane':<13}{'pairs':>6}{'disc':>6}{'+':>4}{'-':>4}"
          f"{'src cost':>10}{'dlt cost':>10}{'interaction':>13}{'p':>9}")
    for cell in CLEAN_CELLS:
        for lane in LANES:
            v = results[cell][lane]
            print(f"{cell:<7}{lane:<13}{v['n_pairs']:>6}{v['discordant']:>6}{v['pos']:>4}"
                  f"{v['neg']:>4}{v['source_position_cost']:>10.4f}"
                  f"{v['delta_position_cost']:>10.4f}{v['interaction_mean']:>13.4f}"
                  f"{v['sign_test_p']:>9.4f}")
    print()
    for cell in CLEAN_CELLS:
        for lane in LANES:
            print(f"  {cell}/{lane}: {results[cell][lane]['verdict']}")
    print("\nsecondary -- cue exploitation on this bank's own errors "
          "(declared pool, cue!=reference):")
    for arm, v in cue.items():
        r = f"{v['rate']:.4f}" if v["rate"] is not None else "--"
        print(f"  {arm:<13}{v['cue_match']:>4}/{v['scorable']:<5} = {r}   "
              f"({v['errors']} errors total)")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
