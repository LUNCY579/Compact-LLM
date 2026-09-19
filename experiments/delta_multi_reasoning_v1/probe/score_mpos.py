"""MPOS-002 scorer — the arm x band interaction on matched pairs.

    python probe/score_mpos.py --root <records>/mpos002-20260919 \
        --out <records>/mpos002-20260919/MPOS_RESULT.json

**Written before any arm was generated and before any prediction existed.** Registered in
`probe/MPOS001_PREREG.md` (amendment 1 records the redesign this file implements).

## What changed from the first draft, and why

A pre-run adversarial review found the first scorer's primary **could not reach its own
registered prediction**. Both defects were confirmed by direct arithmetic before any GPU was
spent:

  * **The exact sign test discards magnitude.** At 48 pairs per cell it can declare significance
    only from a net of 8, i.e. |interaction| >= 8/48 = **0.1667** — and the registered `k1d1`
    band was +0.03 to +0.15, **entirely below that**. The cell was unfalsifiable as specified.
  * **The minimum detectable effect was computed from alpha alone**, as a flat
    "discordant >= 8". The required net actually GROWS with the realised discordant count —
    8 at D=8, 12 at D=20, 16 at D=30 — so a null could be stamped "powered" with an MDE 2x too
    small. That is the flat-threshold-on-a-noisy-estimate defect, which this programme has
    already made once.

So the primary is now a **sign-flip permutation test on the MEAN** of the per-pair statistic,
which uses magnitude as well as direction, and the bank was rebuilt at **144 pairs per cell**.
Measured power at that size, assuming a 0.30 discordance rate: **0.99** for a +0.20 effect,
**0.26** for +0.09. `k3d1` is therefore powered for its registered band and `k1d1` is **not**,
which is stated here rather than discovered afterwards.

## The statistic

Each scenario appears in BOTH bands, built from identical material — same names, distractors,
question and sentences, differing only in arrangement. Per pair and arm:

    d_arm = 1[arm correct on EARLY] - 1[arm correct on LATE]      in {-1, 0, +1}
    x     = d_delta - d_source                                     in {-2 ... +2}

Primary: **mean(x)**, sign-flip permutation p, per cell x lane, **Bonferroni 0.05/4 = 0.0125**.
Registered direction positive. The sign test is retained as a secondary for continuity with the
original registration, never as the primary.

## Why the interaction rather than the damage rate

A deterministic answer-agnostic rule emits the same string for both arms, so it contributes
equally to `d_source` and `d_delta` and **cancels in the difference**. The absolute band
comparison has no such protection.

**A cue that is NOT cancelled:** one whose exploitability differs between arms. Bounded at
**0.211** on ALIAS-001 (Newcombe method 10, two-sided 95% upper) and re-measured here as a
secondary, **with its differential and that differential's minimum detectable value** — the
first draft reported only per-arm rates, which is the very null-without-an-MDE failure the
primary was fixed for.

## The denominator that bounds the estimate

**Discordant pairs**, not the 144 pairs per cell. Reported always, alongside the minimum |mean|
this run could have detected at the realised discordance — computed by simulation from the
observed structure, not from alpha alone.
"""
from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path

CLEAN_CELLS = ("k1d1", "k3d1")
BANDS = ("early", "late")
LANES = {"native_a100": "gen-a100", "lwf_l0": "gen-lwf_l0",
         "titan_sparse": "gen-sparse"}
SOURCE = "gen-B"
#: derived, not hard-coded: adding an arm must widen the family automatically. Addendum 2
#: registers 0.05/6 as governing any statement that compares Sparse to the other arms, and
#: choosing the smaller family after seeing a number is the move this programme has already
#: caught twice.
BONFERRONI = 0.05 / (len(LANES) * 2)
PERM_ITERS = 20000
BOOT_ITERS = 10000
#: registered bands, for reporting the estimate AGAINST the prediction rather than only testing
REGISTERED = {"k1d1": (0.03, 0.15), "k3d1": (0.10, 0.30)}
#: Addendum 2: Sparse is predicted POSITIVE but SMALLER than native's +0.2847.
REGISTERED_SPARSE = {"k1d1": (0.00, 0.12), "k3d1": (0.05, 0.20)}


def norm(v) -> str:
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def signflip_p(x, seed: int, iters: int = PERM_ITERS) -> float:
    """Two-sided sign-flip permutation p for mean(x). Exact null: x_i symmetric about 0."""
    n = len(x)
    if n == 0 or all(v == 0 for v in x):
        return 1.0
    obs = abs(sum(x) / n)
    rng = random.Random(seed)
    c = 0
    for _ in range(iters):
        s = sum(v if rng.random() < 0.5 else -v for v in x)
        if abs(s / n) >= obs - 1e-12:
            c += 1
    return c / iters


def boot_ci(x, seed: int, iters: int = BOOT_ITERS, lo=2.5, hi=97.5):
    n = len(x)
    rng = random.Random(seed)
    means = sorted(sum(x[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (means[int(lo / 100 * iters)], means[min(iters - 1, int(hi / 100 * iters))])


def mde_analytic(x, alpha: float, power: float = 0.80) -> float:
    """Smallest |mean| detectable at the REALISED per-pair spread, at `alpha` and `power`.

    Replaces a simulation that was O(100 effect sizes x 300 reps x 600 permutations) and hung on
    a 144-pair cell -- found by smoke-testing this scorer on synthetic predictions BEFORE the
    real arms existed. It would have hung on the real data too.

    Under the sign-flip null the mean has sd = sqrt(sum(x^2))/n exactly, so the normal
    approximation to the permutation test gives

        MDE = (z_{alpha/2} + z_{power}) * sqrt(sum(x_i^2)) / n

    This is an approximation to the exact permutation test and is labelled as one in the receipt.
    At n = 144 with values in {-2..2} the two agree closely; it is reported alongside the exact
    permutation p, never instead of it.
    """
    n = len(x)
    if n == 0:
        return float("inf")
    ss = sum(v * v for v in x)
    if ss == 0:
        return float("inf")
    sd = (ss ** 0.5) / n
    z = {0.05: 1.959964, 0.025: 2.241403, 0.0125: 2.497705, 0.00625: 2.734368}
    za = z.get(round(alpha, 5), 2.497705)
    zb = 0.841621  # 80% power
    return (za + zb) * sd


def sign_test_p(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    o = abs(k - n / 2)
    return sum(comb(n, x) for x in range(n + 1) if abs(x - n / 2) >= o - 1e-12) / 2 ** n


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

    results: dict = {}
    for cell in CLEAN_CELLS:
        cp = {k: v for k, v in pairs.items() if v["early"]["cell"] == cell}
        per_lane = {}
        for lane in LANES:
            xs, ds, dd = [], [], []
            for k, v in sorted(cp.items()):
                e, l = v["early"]["id"], v["late"]["id"]
                d_src = int(ok("source", e)) - int(ok("source", l))
                d_dlt = int(ok(lane, e)) - int(ok(lane, l))
                ds.append(d_src)
                dd.append(d_dlt)
                xs.append(d_dlt - d_src)
            n = len(xs)
            mean = sum(xs) / n
            disc = sum(1 for v in xs if v != 0)
            pos = sum(1 for v in xs if v > 0)
            neg = sum(1 for v in xs if v < 0)
            p_perm = signflip_p(xs, 101)
            ci = boot_ci(xs, 202)
            mde = mde_analytic(xs, BONFERRONI)
            lo, hi = (REGISTERED_SPARSE if lane == "titan_sparse"
                      else REGISTERED)[cell]
            per_lane[lane] = {
                "n_pairs": n, "discordant": disc, "pos": pos, "neg": neg,
                "tied": n - disc,
                "magnitude_2_pairs": sum(1 for v in xs if abs(v) == 2),
                "source_position_cost": round(sum(ds) / n, 4),
                "delta_position_cost": round(sum(dd) / n, 4),
                "interaction_mean": round(mean, 4),
                "interaction_ci95": [round(ci[0], 4), round(ci[1], 4)],
                "perm_p": p_perm,
                "clears_bonferroni": p_perm < BONFERRONI,
                "sign_test_p_secondary": sign_test_p(pos, disc),
                "mde_at_realised_spread": (None if mde == float("inf") else round(mde, 4)),
                "mde_method": "normal approximation to the sign-flip null; exact perm p reported separately",
                "registered_band": [lo, hi],
                "estimate_vs_band": (
                    "inside" if lo <= mean <= hi else "above" if mean > hi else "below"),
                "band_reachable": (mde is not None and mde != float("inf") and mde <= hi),
                "verdict": (
                    "NOT POWERED -- every pair is tied; no effect size was detectable"
                    if mde == float("inf") else
                    f"SUPPORTS the registered direction (p={p_perm:.4f})"
                    if p_perm < BONFERRONI and mean > 0 else
                    f"FALSIFIES the registered direction (p={p_perm:.4f})"
                    if p_perm < BONFERRONI and mean < 0 else
                    f"not powered to distinguish -- only |mean| >= {mde:.3f} was detectable "
                    f"here, and the registered band is {lo}-{hi}"),
            }
        results[cell] = per_lane

    # --- secondary: cue exploitation, WITH the differential and its MDE ------------------
    pool = sorted({r["reference"] for r in rows})

    def earliest_person(ctx):
        pres = {c: ctx.find(c) for c in pool if c in ctx}
        return min(pres, key=pres.get) if pres else None

    raw = {}
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
        raw[arm] = {"errors": errors, "scorable": scorable, "cue_match": hit,
                    "rate": (round(hit / scorable, 4) if scorable else None)}

    b = raw["source"]
    diffs = {}
    for lane in LANES:
        v = raw[lane]
        if not (v["scorable"] and b["scorable"]):
            diffs[lane] = {"differential": None, "why": "empty denominator"}
            continue
        d = v["rate"] - b["rate"]
        p = fisher(v["cue_match"], v["scorable"] - v["cue_match"],
                   b["cue_match"], b["scorable"] - b["cue_match"])
        l1, u1 = wilson(b["cue_match"], b["scorable"])
        l2, u2 = wilson(v["cue_match"], v["scorable"])
        p1, p2 = b["rate"], v["rate"]
        newc_hi = d + ((u2 - p2) ** 2 + (p1 - l1) ** 2) ** 0.5
        # smallest differential detectable against the source arm at these denominators
        mdd = None
        for k in range(v["cue_match"], v["scorable"] + 1):
            if fisher(k, v["scorable"] - k, b["cue_match"],
                      b["scorable"] - b["cue_match"]) < 0.05:
                mdd = round(k / v["scorable"] - b["rate"], 4)
                break
        diffs[lane] = {
            "differential": round(d, 4), "fisher_p": round(p, 4),
            "newcombe_m10_upper95": round(newc_hi, 4),
            "min_detectable_differential": mdd,
            "reading": ("a POSITIVE differential means the delta exploits the positional cue "
                        "MORE than the source, which is the residual the interaction cannot "
                        "cancel"),
        }

    out = {
        "record": "MPOS_RESULT",
        "class": "MECHANISM SCOUT -- one bank, one draw. Not a confirmation.",
        "prereg": "probe/MPOS001_PREREG.md (+ amendment 1)",
        "bank": str(a.root),
        "scored_cells": list(CLEAN_CELLS),
        "bonferroni": BONFERRONI,
        "primary_test": "sign-flip permutation on mean(d_delta - d_source), magnitude-aware",
        "primary": results,
        "secondary_cue_exploitation": {"per_arm": raw, "differentials": diffs,
                                       "alias001_bound": 0.211},
        "caveats": [
            "the interaction cancels a cue available to BOTH arms; it does NOT cancel a cue "
            "whose exploitability differs between them -- that residual is the secondary above",
            "a positional cue (candidate named in the PENULTIMATE sentence) swings across "
            "bands in both scored cells and favours LATE, so it biases the statistic AGAINST "
            "the registered positive direction: a positive result survives it, a null does not",
            "absolute per-band rates are not reported -- they are the quantity the cue "
            "contaminates",
            "one bank, one draw, two cells, two lanes -- nothing here promotes anything",
        ],
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"MPOS-002 -- mean(d_delta - d_source), sign-flip permutation, "
          f"Bonferroni {BONFERRONI:.4f}\n")
    print(f"{'cell':<7}{'lane':<13}{'pairs':>6}{'disc':>6}{'+':>4}{'-':>4}"
          f"{'mean':>9}{'95% CI':>18}{'MDE':>7}{'p':>9}")
    for cell in CLEAN_CELLS:
        for lane in LANES:
            v = results[cell][lane]
            ci = f"[{v['interaction_ci95'][0]:+.3f},{v['interaction_ci95'][1]:+.3f}]"
            mde = "--" if v["mde_at_realised_spread"] is None else \
                f"{v['mde_at_realised_spread']:.3f}"
            print(f"{cell:<7}{lane:<13}{v['n_pairs']:>6}{v['discordant']:>6}{v['pos']:>4}"
                  f"{v['neg']:>4}{v['interaction_mean']:>9.4f}{ci:>18}{mde:>7}"
                  f"{v['perm_p']:>9.4f}")
    print()
    for cell in CLEAN_CELLS:
        for lane in LANES:
            print(f"  {cell}/{lane}: {results[cell][lane]['verdict']}")
    print("\nsecondary -- cue exploitation (declared pool, cue != reference):")
    for arm, v in raw.items():
        r = f"{v['rate']:.4f}" if v["rate"] is not None else "--"
        print(f"  {arm:<13}{v['cue_match']:>4}/{v['scorable']:<5} = {r}   "
              f"({v['errors']} errors)")
    for lane, v in diffs.items():
        if v.get("differential") is None:
            print(f"  {lane:<13}differential: {v['why']}")
        else:
            print(f"  {lane:<13}differential {v['differential']:+.4f}  "
                  f"p={v['fisher_p']:.4f}  Newcombe upper {v['newcombe_m10_upper95']:.4f}  "
                  f"min detectable {v['min_detectable_differential']}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
