"""Score ALIAS-001 and decide between uniform weakening and salience.

    python probe/score_alias_probe.py --root <records>/alias001-20260918 \
        --gen B=<...>/gen-B/predictions.jsonl --gen D=<...>/gen-D/predictions.jsonl \
        --out <...>/ALIAS_PROBE_RESULT.json

**Written before the generations finished**, so the analysis cannot be shaped by the numbers.
MECHANISM SCOUT: openly spendable, re-runnable, and not a confirmation of anything.

## The registered predictions

    UNIFORM WEAKENING   the B-D gap is roughly constant across low / equal / high
    SALIENCE            the gap shrinks monotonically low -> equal -> high

They diverge on `equal`, which removes the answer's salience disadvantage without making it
extractable. The verdict rule is fixed here:

  * **SALIENCE** if the damage count falls monotonically across the three conditions AND the
    `low` vs `high` contrast is significant;
  * **UNIFORM** if damage is present in every condition and `low` vs `high` is not significant;
  * **NEITHER** otherwise, reported as such rather than forced into one of them.

## What is and is not readable

The paired B-D contrast is readable in every condition: a deterministic rule emits the same
string for both systems, so it cannot manufacture a difference. **Absolute rates in `high` are
not readable** -- the answer is the most frequent person there by construction and
`most_frequent_candidate_in_context` scores 1.000. That is stated in the build manifest and is
repeated here so nobody reads a ceiling as competence.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method"))
from floors import gate  # noqa: E402
from paired import fisher_exact_2x2, paired  # noqa: E402

CONDITIONS = ["low", "equal", "high"]


def norm(v):
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gen", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            (a.root / "ALIAS_PROBE.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    build = json.loads((a.root / "ALIAS_BUILD.json").read_text(encoding="utf-8"))
    preds = {}
    for spec in a.gen:
        lab, _, path = spec.partition("=")
        preds[lab] = {r["id"]: r.get("prediction", "") for r in
                      (json.loads(l) for l in
                       Path(path).read_text(encoding="utf-8").splitlines() if l.strip())}
    pb, pd = preds["B"], preds["D"]

    per = {}
    for c in CONDITIONS:
        sub = [r for r in rows if r["condition"] == c]
        cb = {r["id"]: norm(pb.get(r["id"], "")) == r["reference_normalised"] for r in sub}
        cd = {r["id"]: norm(pd.get(r["id"], "")) == r["reference_normalised"] for r in sub}
        p = paired("B", "D", cb, cd)
        fl = build["floors"][c]["binding_floor_rate"]
        per[c] = {
            "n": len(sub), "B": p.baseline_correct, "D": p.candidate_correct,
            "B_rate": round(p.baseline_correct / len(sub), 4),
            "D_rate": round(p.candidate_correct / len(sub), 4),
            "repair": p.repair, "damage": p.damage,
            "gap": p.baseline_correct - p.candidate_correct,
            "p_mcnemar": p.as_dict()["mcnemar_exact_two_sided_p"],
            # The denominator rule: a condition where the baseline is near the floor cannot
            # show damage, so a shrinking RAW gap could just be a shrinking baseline.
            "damage_rate_given_B_correct": (round(p.damage / p.baseline_correct, 4)
                                            if p.baseline_correct else None),
            "published_floor": fl,
            "B_gate": gate(p.baseline_correct, len(sub), fl)["verdict"],
            "absolute_rates_readable": c != "high",
        }

    lo, hi = per["low"], per["high"]
    # Is the damage differently distributed between the extreme conditions? Conditioned on the
    # rows B got right, which is the only set on which a loss was possible.
    fisher_cond = (fisher_exact_2x2(lo["damage"], lo["B"] - lo["damage"],
                                    hi["damage"], hi["B"] - hi["damage"])
                   if lo["B"] and hi["B"] else None)
    damages = [per[c]["damage"] for c in CONDITIONS]
    rates = [per[c]["damage_rate_given_B_correct"] for c in CONDITIONS]
    monotone_counts = damages[0] >= damages[1] >= damages[2] and damages[0] > damages[2]
    monotone_rates = (all(x is not None for x in rates)
                      and rates[0] >= rates[1] >= rates[2] and rates[0] > rates[2])
    sig = fisher_cond is not None and fisher_cond <= 0.05

    if monotone_rates and sig:
        verdict = "SALIENCE"
    elif all(d > 0 for d in damages) and not sig:
        verdict = "UNIFORM_WEAKENING"
    else:
        verdict = "NEITHER_CLEANLY"

    out = {
        "record": "ALIAS_001_RESULT", "class": "MECHANISM SCOUT -- not a confirmation",
        "per_condition": per,
        "monotone_in_counts": monotone_counts,
        "monotone_in_rates_given_B_correct": monotone_rates,
        "fisher_low_vs_high_conditioned": fisher_cond,
        "verdict": verdict,
        "verdict_rule": ("SALIENCE needs monotone damage rates AND a significant low-vs-high "
                         "contrast; UNIFORM needs damage everywhere and no significant "
                         "contrast; anything else is reported as NEITHER rather than forced"),
        "caveats": [
            "absolute rates in `high` are not readable -- the answer is the most frequent "
            "person there by construction and the extraction floor is 1.000",
            "the paired contrast is readable in every condition: a deterministic rule emits "
            "the same string for both systems and cannot create a discordant pair",
            "one training draw of one delta; 40 packs x 3 variants per condition",
        ],
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'condition':<8}{'n':>5}{'B':>5}{'D':>5}{'rep':>5}{'dam':>5}{'gap':>5}"
          f"{'p':>10}{'dam|B':>8}{'floor':>7}  {'B gate':<26}absolute?")
    for c in CONDITIONS:
        v = per[c]
        dr = "n/a" if v["damage_rate_given_B_correct"] is None else \
            f"{v['damage_rate_given_B_correct']:.3f}"
        print(f"{c:<8}{v['n']:>5}{v['B']:>5}{v['D']:>5}{v['repair']:>5}{v['damage']:>5}"
              f"{v['gap']:>5}{v['p_mcnemar']:>10.3e}{dr:>8}{v['published_floor']:>7.3f}  "
              f"{v['B_gate']:<26}{'yes' if v['absolute_rates_readable'] else 'NO'}")
    fc = "n/a" if fisher_cond is None else f"{fisher_cond:.4f}"
    print(f"\nmonotone in counts {monotone_counts}; in rates|B-correct {monotone_rates}; "
          f"Fisher low vs high (conditioned) {fc}")
    print(f"\nVERDICT: {verdict}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
