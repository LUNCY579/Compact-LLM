"""Measure differential cue exploitation on MPOS's OWN arms — replacing an extrapolation.

    python analysis/cue_differential.py --root <records>/mpos002-20260919 \
        --out <records>/mpos002-20260919/CUE_DIFFERENTIAL.json

**Written before the primary was read.** Amendment 1b registered that this analysis would exist
and that it would be committed separately from the registered path, so that nothing in the
primary moves once predictions exist. This file is that separate analysis.

## The problem it fixes

Amendment 1b bounds the manufacturable interaction as `differential x band_swing`, using a
differential of **0.211** measured on ALIAS-001 for the **earliest-person** cue. It then applies
it to a **sentence-index** cue with a band swing of 0.771. That extrapolation is unsound, and
both sessions now have evidence for why:

  * the differential is **cue-specific and not sign-stable**. Across ten cues on ALIAS-001 it
    ranges roughly -0.07 to +0.09 — on some cues the delta exploits more, on others the SOURCE
    does. It is a property of the arm-pair-AND-cue, not of the arm pair.
  * so a bound measured on one cue does not transfer to a cue it was not measured on.

It is nonetheless *conservative* in this instance, because 0.211 sits near the maximum of the
observed upper bounds. **But "conservative extrapolation" is still an extrapolation, and the fix
is cheap: measure the differential for the cue that actually matters, on the arms that actually
ran.** That is what this does.

## What it measures

For each cue, restricted to rows where the cue does NOT already name the reference (an error
cannot match the cue there, and conditioning on error already selects against the heuristic
succeeding):

    rate(arm) = P(arm's WRONG answer == the cue's answer)
    differential = rate(delta) - rate(source)

with a Fisher exact p, a Newcombe method 10 two-sided 95% upper bound, and the minimum
detectable differential at the realised denominators. The cues swept are the ones the systematic
rule sweep found to be most band-dependent in the scored cells, in **both** directions, plus the
earliest-person cue for comparability with ALIAS-001.

## Tie-breaking, which is not a detail

`most_frequent` ties whenever the answer and a distractor share a mention count, and Python's
`max()` over a set breaks ties by arbitrary iteration order. Two sessions measuring "the same"
rule got differentials of +0.033 and -0.054 on that cue for exactly this reason. **Every cue here
resolves ties deterministically and explicitly** (by earliest occurrence, then by string), and
any row whose cue is tied is additionally reported so the count is visible rather than silent.
"""
from __future__ import annotations

import argparse
import json
import re
from math import comb
from pathlib import Path

CLEAN_CELLS = ("k1d1", "k3d1")
LANES = {"native_a100": "gen-a100", "lwf_l0": "gen-lwf_l0"}
SOURCE = "gen-B"


def norm(v) -> str:
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def fisher(a: int, b: int, c: int, d: int) -> float:
    n, r1, c1 = a + b + c + d, a + b, a + c
    if not (r1 and c1 and n):
        return 1.0

    def pr(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    p0 = pr(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 + 1e-12)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


def occurrences(ctx, c):
    o, i = [], ctx.find(c)
    while i >= 0:
        o.append(i)
        i = ctx.find(c, i + 1)
    return o


def cue_answer(row, pool, kind):
    """Deterministic, explicitly tie-broken. Returns (pick, was_tied)."""
    ctx = row["context"]
    pres = {c: occurrences(ctx, c) for c in pool if c in ctx}
    if not pres:
        return None, False

    def pick_by(keyfn, reverse=False):
        vals = {c: keyfn(c) for c in pres}
        best = max(vals.values()) if reverse else min(vals.values())
        tied = sorted(c for c in vals if vals[c] == best)
        # tie-break: earliest first occurrence, then string order
        return sorted(tied, key=lambda c: (pres[c][0], c))[0], len(tied) > 1

    if kind == "earliest_person":
        return pick_by(lambda c: pres[c][0])
    if kind == "earliest_mean":
        return pick_by(lambda c: sum(pres[c]) / len(pres[c]))
    if kind == "latest_mean":
        return pick_by(lambda c: sum(pres[c]) / len(pres[c]), reverse=True)
    if kind.startswith("in_sentence_"):
        si = int(kind.rsplit("_", 1)[1])
        ss = sentences(ctx)
        if si >= len(ss):
            return None, False
        here = sorted(c for c in pres if c in ss[si])
        if len(here) != 1:
            return (here[0] if here else None), len(here) > 1
        return here[0], False
    raise SystemExit(f"unknown cue {kind}")


def load(root: Path, sub: str, ids: set) -> dict:
    f = root / sub / "predictions.jsonl"
    if not f.exists():
        raise SystemExit(f"missing predictions: {f}")
    p = {r["id"]: r.get("prediction", "") for r in
         (json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip())}
    miss = [i for i in ids if i not in p]
    if miss:
        raise SystemExit(f"{sub} misses {len(miss)} rows")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cues", nargs="*", default=[
        "earliest_person", "earliest_mean", "latest_mean",
        "in_sentence_5", "in_sentence_11", "in_sentence_3", "in_sentence_13"])
    a = ap.parse_args(argv)

    rows = [json.loads(x) for x in
            (a.root / "MPOS_PROBE.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    pool = sorted({r["reference"] for r in rows})
    scored = [r for r in rows if r["cell"] in CLEAN_CELLS]
    ids = {r["id"] for r in scored}
    by_id = {r["id"]: r for r in scored}

    preds = {"source": load(a.root, SOURCE, ids)}
    for lane, sub in LANES.items():
        preds[lane] = load(a.root, sub, ids)

    def ok(arm, rid):
        return norm(preds[arm][rid]) == by_id[rid]["reference_normalised"]

    out = {
        "record": "CUE_DIFFERENTIAL",
        "class": "post-primary analysis, committed separately from the registered path",
        "why": ("replaces an extrapolated differential (0.211, measured on a DIFFERENT cue on a "
                "DIFFERENT bank) with a measurement on the cue and arms that matter"),
        "by_cell": {},
    }

    for cell in CLEAN_CELLS:
        sub_rows = [r for r in scored if r["cell"] == cell]
        per_cue = {}
        for kind in a.cues:
            arms = {}
            ties = 0
            for arm in ["source", *LANES]:
                hit = scorable = errors = 0
                for r in sub_rows:
                    if ok(arm, r["id"]):
                        continue
                    errors += 1
                    c, tied = cue_answer(r, pool, kind)
                    if arm == "source" and tied:
                        ties += 1
                    if c is None or c == r["reference"]:
                        continue
                    scorable += 1
                    if norm(preds[arm][r["id"]]) == norm(c):
                        hit += 1
                arms[arm] = {"errors": errors, "scorable": scorable, "cue_match": hit,
                             "rate": round(hit / scorable, 4) if scorable else None}
            b = arms["source"]
            diffs = {}
            for lane in LANES:
                v = arms[lane]
                if not (v["scorable"] and b["scorable"]):
                    diffs[lane] = {"differential": None, "why": "empty denominator"}
                    continue
                d = v["rate"] - b["rate"]
                p = fisher(v["cue_match"], v["scorable"] - v["cue_match"],
                           b["cue_match"], b["scorable"] - b["cue_match"])
                l1, _ = wilson(b["cue_match"], b["scorable"])
                _, u2 = wilson(v["cue_match"], v["scorable"])
                upper = d + ((u2 - v["rate"]) ** 2 + (b["rate"] - l1) ** 2) ** 0.5
                mdd = None
                for k in range(v["cue_match"], v["scorable"] + 1):
                    if fisher(k, v["scorable"] - k, b["cue_match"],
                              b["scorable"] - b["cue_match"]) < 0.05:
                        mdd = round(k / v["scorable"] - b["rate"], 4)
                        break
                diffs[lane] = {"differential": round(d, 4), "fisher_p": round(p, 4),
                               "newcombe_m10_upper95": round(upper, 4),
                               "min_detectable_differential": mdd}
            per_cue[kind] = {"arms": arms, "differentials": diffs, "tied_cue_rows": ties}
        out["by_cell"][cell] = per_cue

    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("CUE DIFFERENTIAL on MPOS's own arms — measured, not extrapolated\n")
    for cell in CLEAN_CELLS:
        print(f"--- {cell} ---")
        print(f"  {'cue':<18}{'src':>10}{'native':>10}{'lwf':>10}"
              f"{'d(nat)':>9}{'d(lwf)':>9}{'ties':>6}")
        for kind, v in out["by_cell"][cell].items():
            ar = v["arms"]
            f = lambda x: f"{x:.3f}" if x is not None else "--"  # noqa: E731
            dn = v["differentials"]["native_a100"].get("differential")
            dl = v["differentials"]["lwf_l0"].get("differential")
            print(f"  {kind:<18}{f(ar['source']['rate']):>10}"
                  f"{f(ar['native_a100']['rate']):>10}{f(ar['lwf_l0']['rate']):>10}"
                  f"{(f'{dn:+.3f}' if dn is not None else '--'):>9}"
                  f"{(f'{dl:+.3f}' if dl is not None else '--'):>9}{v['tied_cue_rows']:>6}")
        print()
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
