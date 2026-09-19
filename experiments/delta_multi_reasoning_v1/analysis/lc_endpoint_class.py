"""Classify each long-context stratum by what the UNROTATED delta does to it, then split the
rotated arms' damage by that class.

    python analysis/lc_endpoint_class.py --bank .../all_prompts.jsonl --evidence <dir> \
        --source supa=LONGCTX_supa.json --baseline l0=LONGCTX_lwf_l0_lc.json \
        --arm l025=LONGCTX_lwf_l025_lc.json --arm ord23=LONGCTX_lwf_ord23_lc.json \
        --out <records>/LC_ENDPOINT_CLASS.json

**REPORT-ONLY** — the 256-row suite is SPENT. Zero model calls.

## Why this is run here rather than taken on trust

The peer session used this stratification to **retract their own "damage reallocation"
headline**. A retraction is exactly as load-bearing as a claim and deserves the same
verification, so this recomputes it from the raw per-row evidence on this side.

## The argument being checked

Classify each stratum by `Δ = λ=0 − Supa`, using **only the source and the unrotated delta**.
No rotated-arm data enters the classification, so it is a stratifier and not selection on the
outcome — which is the property that makes the split legitimate at all.

  * **PRESERVED** — the delta leaves it alone. Damage here is genuine forgetting.
  * **IMPROVED** — the delta *gains* here. Damage here is **the delta's own gain being given
    back**, which is an ACQUISITION cost, not forgetting.
  * **DAMAGED** — the delta already costs this. A proper preservation endpoint.
  * **BELOW GATE** — the source cannot do it; nothing to preserve, nothing readable.

Then split each rotated arm's losses by class. **If the rotated arms' damage sits mostly on
IMPROVED strata, "rotation relocates forgetting" is the wrong description** — the arm is giving
back acquisition it never securely had, and on the process bank `ord23` already costs 28
acquisition rows. Those would be the same fact measured twice, not two findings.

## The denominator, unchanged

Losses are scored over the rows **λ=0 got right**, never over all 256. Answerable rows only,
since the unanswerable variant measures abstention rather than retrieval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--source", required=True, metavar="NAME=FILE")
    ap.add_argument("--baseline", required=True, metavar="NAME=FILE")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=FILE")
    ap.add_argument("--answerable-only", action="store_true", default=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = {r["id"]: r for r in
            (json.loads(l) for l in a.bank.read_text(encoding="utf-8").splitlines() if l.strip())}

    def load(spec):
        n, _, f = spec.partition("=")
        j = json.loads((a.evidence / f).read_text(encoding="utf-8"))
        return n, {k: bool(v) for k, v in j["per_row_correct"].items()}

    sname, scor = load(a.source)
    bname, bcor = load(a.baseline)
    arms = [load(s) for s in a.arm]

    def answerable(i):
        r = rows[i]
        return str(r.get("answerable", "True")).lower() == "true" and \
            r.get("variant") != "unanswerable"

    ids = [i for i in rows if not a.answerable_only or answerable(i)]
    strata = sorted({rows[i]["task_type"] for i in ids})

    per, klass = {}, {}
    for s in strata:
        sub = [i for i in ids if rows[i]["task_type"] == s]
        sc = sum(1 for i in sub if scor.get(i))
        bc = sum(1 for i in sub if bcor.get(i))
        d = bc - sc
        # classification uses ONLY source and unrotated delta
        if sc <= 0.20 * len(sub):
            k = "BELOW_GATE"
        elif d >= 2:
            k = "IMPROVED"
        elif d <= -2:
            k = "DAMAGED"
        else:
            k = "PRESERVED"
        klass[s] = k
        per[s] = {"n": len(sub), f"{sname}": sc, f"{bname}": bc, "delta": d, "class": k}

    by_class = {}
    for n, cor in arms:
        tally = {}
        for s in strata:
            sub = [i for i in ids if rows[i]["task_type"] == s and bcor.get(i)]
            lost = sum(1 for i in sub if not cor.get(i))
            per[s].setdefault("arm_losses", {})[n] = lost
            per[s].setdefault("arm_denominator", len(sub))
            t = tally.setdefault(klass[s], {"losses": 0, "denominator": 0})
            t["losses"] += lost
            t["denominator"] += len(sub)
        by_class[n] = tally

    out = {
        "record": "LC_ENDPOINT_CLASS",
        "class": "REPORT-ONLY -- the 256-row suite is SPENT; zero model calls",
        "verifies": ("the peer session's stratification, used by them to RETRACT their own "
                     "'damage reallocation' headline"),
        "classification_inputs": f"{sname} and {bname} only -- no rotated-arm data",
        "answerable_only": a.answerable_only,
        "per_stratum": per,
        "damage_by_class": by_class,
        "reading": ("damage on an IMPROVED stratum is the delta's own gain being given back -- "
                    "an ACQUISITION cost, not forgetting"),
        "caveats": [
            "SPENT instrument, report-only, non-confirmatory",
            "the classification threshold (|delta| <= 1 = PRESERVED) is a choice; the strata "
            "here are far from it in both directions, but a borderline stratum would be "
            "sensitive to it",
            "BELOW_GATE is assigned at source <= 20% and is reported, never carried",
        ],
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'stratum':<26}{'n':>4}{sname:>6}{bname:>5}{'delta':>7}  class"
          + "".join(f"{n:>9}" for n, _ in arms))
    for s in strata:
        v = per[s]
        print(f"{s:<26}{v['n']:>4}{v[sname]:>6}{v[bname]:>5}{v['delta']:>+7}  {v['class']:<11}"
              + "".join(f"{v['arm_losses'][n]:>9}" for n, _ in arms))
    print(f"\n{'':<26}damage by class (losses / lambda=0-correct rows)")
    for n, _ in arms:
        parts = "   ".join(f"{k}: {v['losses']}/{v['denominator']}"
                           for k, v in sorted(by_class[n].items()))
        print(f"  {n:<10}{parts}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
