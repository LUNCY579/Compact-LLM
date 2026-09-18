"""Upper-bound EVERY selector that reads a given feature set, immune to statistical power.

WHY THIS AND NOT A FEATURE-ASSOCIATION TEST
-------------------------------------------
The first attempt tested whether individual request features *associate* with the items where a
weaker arm beats the best arm. Nothing was significant — but with 19 such items in 128, that
test could not have detected a 1.8x effect in a 15-item subgroup, so "no feature found" was a
statement about power, not about the world.

The partition oracle sidesteps that entirely. Partition the items by the features, then take
the BEST ARM IN EACH CELL, chosen with labels. That is the ceiling for *every possible*
selector reading only those features — including rules neither of us thought of — because no
label-blind rule can beat the label-aware best choice within the cells it can distinguish. It
is a bound, not an estimate, so it does not depend on sample size.

THE CONTROL THAT MAKES IT MEAN ANYTHING
---------------------------------------
The oracle rises with granularity for free: shuffle items into enough arbitrary cells and
"best arm per cell" starts fitting noise. So every scheme is compared against random
partitions **matched on exact cell sizes**, not merely cell count.

Size-matching is the tighter and the correct control. The per-cell max is convex in cell size,
so leaving cell sizes free inflates the baseline and can hide a real effect. That difference is
not hypothetical: against a count-matched null the 17-cell scheme's baseline came out ~247.6
and every partition looked null; against this size-matched null it is ~244.0 and the `status`
cell shows at +2.9 SD.

WHAT THIS FOUND, on the seven alpha/k arms over SELECT-VIEWS-001
----------------------------------------------------------------
    partition                       cells  oracle  vs best   size-matched random      z
    family only                         3     238       +3   238.6 +/- 1.6        -0.41
    kind                                7     240       +5   241.4 +/- 1.7        -0.81
    status                              3     242       +7   237.8 +/- 1.5        +2.91
    kind x status                      11     245      +10   242.7 +/- 1.8        +1.25
    kind x status x sign               17     247      +12   244.0 +/- 1.9        +1.54
    family x kind x status x sign      17     247      +12   244.0 +/- 1.9        +1.56

Two things worth keeping:

1. `family x kind x status x sign` equals `kind x status x sign` EXACTLY. Once the event
   features are known, family adds literally nothing — an independent confirmation, from a
   different direction, of the +3 ceiling on family-level routing.
2. `status` is the only cell with signal: +2.91 SD, which clears a Bonferroni threshold over
   six schemes (z ~ 2.64). **It still does not license building anything**: its oracle gain is
   +7, exactly REP-001's single-draw floor, so the LABEL-AWARE ceiling sits at the noise floor
   and any deployable selector can only do worse. "Distinguishable from a random partition" and
   "worth building" are different bars; this clears the first and fails the second.

Read-only. No model is run. Writes one JSON receipt.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import random
import statistics as st
from pathlib import Path

R = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
ARM_LABELS = {"int001_a025": "a0.25", "int001_a050": "a0.50", "int001_a075": "a0.75",
              "int001_a100": "a1.00", "int001_s10d": "k0.10", "int001_s25d": "k0.25",
              "int001_s50d": "k0.50"}
#: REP-001's single-draw acquisition spread. A gain at or below this is not a direction.
DRAW_FLOOR = 7


def load_arms():
    arms, fam = {}, {}
    for p in glob.glob(str(R / "**" / "PROCESS*.json"), recursive=True):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        lbl = d.get("label")
        if lbl not in ARM_LABELS or "predictions" not in d:
            continue
        pr = d["predictions"]
        it = pr.items() if isinstance(pr, dict) else ((r["id"], r) for r in pr)
        m = {}
        for i, r in it:
            m[i] = bool(r.get("canonical"))
            if r.get("family"):
                fam[i] = r["family"]
        arms[ARM_LABELS[lbl]] = m
    return arms, fam


def event_of(views, i):
    pr = views[i]["prompt"]
    msgs = pr if isinstance(pr, list) else ast.literal_eval(pr)
    user = [m for m in msgs if m.get("role") == "user"][-1]["content"]
    try:
        return json.loads(user).get("event", {})
    except Exception:
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)

    arms, fam = load_arms()
    if len(arms) != 7:
        raise SystemExit(f"expected 7 arms, found {sorted(arms)}")
    views = {v["id"]: v for v in (json.loads(l) for l in
             (R / "select-views-001" / "SELECT_VIEWS.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip())}
    ids = sorted(set.intersection(*(set(v) for v in arms.values())))
    ev = {i: event_of(views, i) for i in ids}

    best_arm = max(arms, key=lambda k: sum(arms[k][i] for i in ids))
    best = sum(arms[best_arm][i] for i in ids)
    per_item_oracle = sum(1 for i in ids if any(arms[k][i] for k in arms))

    def cell_max(members):
        return max(sum(arms[k][i] for i in members) for k in arms)

    schemes = {
        "family only": lambda i: fam.get(i),
        "kind": lambda i: ev[i].get("kind"),
        "status": lambda i: ev[i].get("status"),
        "kind x status": lambda i: (ev[i].get("kind"), ev[i].get("status")),
        "kind x status x sign": lambda i: (ev[i].get("kind"), ev[i].get("status"),
                                           (ev[i].get("amount") or 0) < 0),
        "family x kind x status x sign": lambda i: (fam.get(i), ev[i].get("kind"),
                                                    ev[i].get("status"),
                                                    (ev[i].get("amount") or 0) < 0),
    }

    rng = random.Random(a.seed)
    rows = []
    for name, keyfn in schemes.items():
        cells: dict = {}
        for i in ids:
            cells.setdefault(keyfn(i), []).append(i)
        sizes = [len(v) for v in cells.values()]
        oracle = sum(cell_max(v) for v in cells.values())
        sims = []
        for _ in range(a.trials):
            sh = ids[:]
            rng.shuffle(sh)
            pos, tot = 0, 0
            for sz in sizes:                       # SIZE-matched, not count-matched
                tot += cell_max(sh[pos:pos + sz])
                pos += sz
            sims.append(tot)
        mu, sd = st.mean(sims), st.stdev(sims)
        rows.append({
            "partition": name, "cells": len(cells), "cell_sizes": sorted(sizes, reverse=True),
            "oracle": oracle, "vs_best_single_arm": oracle - best,
            "random_size_matched_mean": round(mu, 2), "random_size_matched_sd": round(sd, 2),
            "z": round((oracle - mu) / sd, 2) if sd else None,
            "beats_random": bool(oracle > mu + 2 * sd),
            "gain_vs_draw_floor": ("AT_OR_BELOW_FLOOR" if oracle - best <= DRAW_FLOOR
                                   else "ABOVE_FLOOR"),
        })

    by = {r["partition"]: r for r in rows}
    receipt = {
        "schema": 1, "record": "DMRF_FEATURE_PARTITION_CEILING",
        "what_this_is": (
            "Upper bound on every selector reading a given feature set: best arm per cell, "
            "chosen with labels. A bound rather than an estimate, so it does not depend on "
            "statistical power -- which is why it supersedes the earlier feature-association "
            "test that could not have detected a 1.8x effect on n=15."),
        "control": ("random partitions matched on EXACT CELL SIZES, not cell count. The per-cell "
                    "max is convex in cell size, so count-matching inflates the baseline and can "
                    "hide a real effect; measured here as ~247.6 vs ~244.0 on the 17-cell scheme."),
        "arms": sorted(arms), "n_items": len(ids), "trials": a.trials, "seed": a.seed,
        "best_single_arm": best, "best_arm_label": best_arm,
        "per_item_oracle_all_arms": per_item_oracle,
        "rep001_draw_floor": DRAW_FLOOR,
        "partitions": rows,
        "family_adds_nothing_once_event_features_known": (
            by["family x kind x status x sign"]["oracle"] == by["kind x status x sign"]["oracle"]),
        "findings": {
            "status_is_the_only_signal": by["status"]["z"],
            "status_oracle_gain": by["status"]["vs_best_single_arm"],
            "status_verdict": (
                "Clears a Bonferroni threshold over six schemes (z ~ 2.64) but its oracle gain "
                "is +7, exactly the draw floor. The LABEL-AWARE ceiling sits at the noise floor, "
                "so any deployable selector can only do worse. Distinguishable from random and "
                "not worth building are both true."),
            "every_partition_at_or_below_random": False,
            "correction_note": (
                "An earlier summary stated every partition sits at or below a random partition "
                "of equal granularity. That is not supported under a size-matched null: `status` "
                "is +2.9 SD. The conclusion is unchanged because the gain is at the floor."),
        },
        "recommendation": "Do not build a selector for this delta.",
        "model_calls": 0, "optimizer_steps": 0,
        "quality_claim": False, "release_permitted": False,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in
                      ("best_single_arm", "per_item_oracle_all_arms",
                       "family_adds_nothing_once_event_features_known", "findings")}, indent=2))
    for r in rows:
        print(f"  {r['partition']:<32}{r['cells']:>4}{r['oracle']:>7}{r['vs_best_single_arm']:>+6}"
              f"   {r['random_size_matched_mean']:>7.1f} +/- {r['random_size_matched_sd']:.1f}"
              f"  z={r['z']:+.2f}  {r['gain_vs_draw_floor']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
