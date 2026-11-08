"""LWF α=0.75 on the alias bank — the last open cell in the frontier.

    python probe/score_a075_alias.py --root <records>/alias001-20260918 \
        --out <records>/lwf001-20260918/A075_ALIAS_RESULT.json

**Written before the predictions were read.** Prereg: `probe/LWF_A075_ALIAS_PREREG.md`, which
registers a **point** prediction of **216.5 / 360 (band 210–222)**, interpolated from the
already-measured collinear ray. The peer session's bands are **≥210 transfers / 203–209 partial
/ <203 native-specific**.

The arm is the peer's `ctrl-own-alpha0.75` — their build, their bytes, not rebuilt here.

## Why every contrast is reported split, and net figures are not quoted alone

The peer reported α=0.75's long context as **13 gains / 13 losses, net zero**, then decomposed it
and found **+8 on properly-classed reading and −5 on abstention** — a real trade hidden inside a
net of nothing, by pooling endpoint classes their own prereg says to separate.

This bank's class distinction is **salience condition**: `low` (answer named once, distractors
three times), `equal` (2/2), `high` (3/1). **Absolute rates in `high` are not readable** — the
answer is the most frequent person there by construction and
`most_frequent_candidate_in_context` scores 1.000. **Paired contrasts are readable in all
three**, because a deterministic answer-agnostic rule emits the same string for both arms and
cannot create a discordant pair.

Three contrasts are reported, because they answer different questions:

  * **vs λ=0** — does interpolation recover alias at all on this lane;
  * **vs `ownA05` (α=0.50)** — the discriminating one. The peer found α=0.75 vs α=0.50 is +7 at
    **p = 0.281** on acquisition, so the lanes' two interior points are not separable there and
    **alias decides which is the better operating point**;
  * **vs Supa** — how much of the source's competence is back.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from paired import paired  # noqa: E402

CONDITIONS = ["low", "equal", "high"]
ARMS = {"supa": "gen-B", "l0": "gen-lwf_l0", "owna05": "gen-lwf_owna05",
        "a075": "gen-lwf_owna075"}
POINT_PREDICTION, BAND = 216.5, (210, 222)
PEER_BANDS = {"transfers": 210, "partial": 203}


def norm(v) -> str:
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            (a.root / "ALIAS_PROBE.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ref = {r["id"]: r["reference_normalised"] for r in rows}
    cond = {r["id"]: r["condition"] for r in rows}

    correct = {}
    for lab, d in ARMS.items():
        f = a.root / d / "predictions.jsonl"
        if not f.exists():
            raise SystemExit(f"missing predictions for {lab}: {f}")
        p = {r["id"]: r.get("prediction", "") for r in
             (json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip())}
        if [i for i in ref if i not in p]:
            raise SystemExit(f"{lab} does not cover all {len(ref)} rows")
        correct[lab] = {i: norm(p[i]) == ref[i] for i in ref}

    n = len(ref)
    acc = {k: sum(v.values()) for k, v in correct.items()}
    hit = acc["a075"]

    contrasts = {}
    for base in ("l0", "owna05", "supa"):
        pr = paired(base, "a075", correct[base], correct["a075"])
        by_c = {}
        for c in CONDITIONS:
            ids = [i for i in ref if cond[i] == c]
            q = paired(base, "a075", {i: correct[base][i] for i in ids},
                       {i: correct["a075"][i] for i in ids})
            by_c[c] = {"n": len(ids), "base": q.baseline_correct, "a075": q.candidate_correct,
                       "a075_gains": q.repair, "a075_losses": q.damage,
                       "p": q.as_dict()["mcnemar_exact_two_sided_p"],
                       "absolute_readable": c != "high"}
        contrasts[base] = {
            "base_correct": pr.baseline_correct, "a075_correct": pr.candidate_correct,
            "a075_gains": pr.repair, "a075_losses": pr.damage,
            "net": pr.candidate_correct - pr.baseline_correct,
            "p": pr.as_dict()["mcnemar_exact_two_sided_p"],
            "by_condition": by_c,
        }

    in_band = BAND[0] <= hit <= BAND[1]
    peer = ("transfers" if hit >= PEER_BANDS["transfers"]
            else "partial" if hit >= PEER_BANDS["partial"] else "native_specific")
    better_than_a050 = contrasts["owna05"]["net"]

    out = {
        "record": "A075_ALIAS_RESULT",
        "class": "MECHANISM SCOUT -- not a confirmation",
        "prereg": "probe/LWF_A075_ALIAS_PREREG.md",
        "arm": "peer session's lwf001-20260918/ctrl-own-alpha0.75 (their bytes, not rebuilt)",
        "n": n, "a075_correct": hit, "a075_rate": round(hit / n, 4),
        "arm_totals": acc,
        "my_point_prediction": POINT_PREDICTION, "my_band": list(BAND),
        "my_prediction_hit": in_band,
        "miss_rows": round(hit - POINT_PREDICTION, 1),
        "peer_band_fired": peer,
        "contrasts": contrasts,
        "discriminating_contrast": {
            "vs_owna05_net": better_than_a050,
            "reading": ("the peer found alpha=0.75 vs alpha=0.50 is +7 at p = 0.281 on "
                        "acquisition, so alias is what separates the lane's two interior "
                        "points"),
        },
        "caveats": [
            "absolute rates in `high` are not readable -- the answer is the most frequent "
            "person there by construction and the extraction floor is 1.000",
            "paired contrasts ARE readable in every condition",
            "gains and losses are never netted; the peer's 13/13 hid a +8/-5",
            "one draw, one lane, a mechanism scout -- nothing here promotes anything",
        ],
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"alpha=0.75 alias: {hit}/{n} = {hit/n:.4f}")
    print(f"  my point prediction {POINT_PREDICTION} (band {BAND[0]}-{BAND[1]}): "
          f"{'HIT' if in_band else 'MISS'}  ({hit - POINT_PREDICTION:+.1f} rows)")
    print(f"  peer band: {peer}")
    print(f"\n{'contrast':<12}{'base':>6}{'a075':>6}{'gains':>7}{'losses':>8}{'net':>6}{'p':>11}")
    for b in ("l0", "owna05", "supa"):
        v = contrasts[b]
        print(f"{'vs '+b:<12}{v['base_correct']:>6}{v['a075_correct']:>6}{v['a075_gains']:>7}"
              f"{v['a075_losses']:>8}{v['net']:>+6}{v['p']:>11.4f}")
    for b in ("l0", "owna05"):
        print(f"\n  split by salience -- vs {b}")
        for c in CONDITIONS:
            x = contrasts[b]["by_condition"][c]
            print(f"    {c:<7}n={x['n']:<5}gains {x['a075_gains']:>3}  losses {x['a075_losses']:>3}"
                  f"  p={x['p']:.4f}  {'' if x['absolute_readable'] else '(abs not readable)'}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
