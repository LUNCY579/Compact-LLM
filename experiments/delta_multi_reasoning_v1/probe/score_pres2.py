"""PRES-002 Stage 1 — which strata are ELIGIBLE to serve as preservation endpoints?

    python probe/score_pres2.py --root <records>/pres002-20260919 \
        --baseline supa=gen-supa --candidate l0=gen-l0 \
        --out <records>/pres002-20260919/PRES2_STAGE1.json

**Written before either prediction file was read**, as `probe/PRES002_PREREG.md` registers.
MECHANISM SCOUT. This stage decides eligibility and **is not a result**.

A stratum qualifies only if BOTH hold:

  1. **Supa clears the stratum's published floor** by `gate()` — otherwise the source cannot do
     the task and there is nothing to preserve;
  2. **the candidate damages it**, exact two-sided McNemar at p <= 0.05, damage scored over
     **the rows the baseline got right**.

A stratum failing either is **reported and DROPPED**. That is the discipline the arithmetic
panel should have had: an endpoint the delta *improves* cannot show sparing, and carrying it
would repeat that mistake in a new coordinate.

## The power column, and why it is registered rather than computed afterwards

On alias, lambda=0.25 recovered **39%** of the damage lambda=0 did (28 of 71 rows). That
fraction is the benchmark. For each stratum this reports the number of rows a 39% recovery would
be, and the smallest McNemar p reachable with that many discordant pairs. **A stratum that
cannot reach p <= 0.05 even with a perfect 39% recovery is UNDERPOWERED, not negative** — the
exact error corrected in Amendment 7 for `references_retained`, and it is flagged here rather
than discovered later.
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
from paired import paired  # noqa: E402

BENCHMARK_RECOVERY = 0.39   # the alias effect, registered in PRES002_PREREG.md


def norm(v) -> str:
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def mcnemar_floor(n_disc: int) -> float:
    """Smallest two-sided exact p reachable with n_disc discordant pairs, all one way."""
    if n_disc <= 0:
        return 1.0
    return min(1.0, 2.0 * (0.5 ** n_disc))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--baseline", required=True, metavar="NAME=DIR")
    ap.add_argument("--candidate", required=True, metavar="NAME=DIR")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            (a.root / "PRES2_PROBE.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    build = json.loads((a.root / "PRES2_BUILD.json").read_text(encoding="utf-8"))
    ref = {r["id"]: r["reference_normalised"] for r in rows}
    stratum = {r["id"]: r["stratum"] for r in rows}
    strata = build["strata"]

    def load(spec):
        name, _, d = spec.partition("=")
        f = a.root / d / "predictions.jsonl"
        if not f.exists():
            raise SystemExit(f"missing predictions: {f}")
        p = {r["id"]: r.get("prediction", "") for r in
             (json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip())}
        miss = [i for i in ref if i not in p]
        if miss:
            raise SystemExit(f"{name} missing {len(miss)} of {len(ref)} rows")
        return name, {i: norm(p[i]) == ref[i] for i in ref}

    bname, bcor = load(a.baseline)
    cname, ccor = load(a.candidate)

    per, qualifying = {}, []
    for s in strata:
        ids = [i for i in ref if stratum[i] == s]
        n = len(ids)
        fl = build["floors_per_stratum"][s]["binding_floor_rate"]
        b = {i: bcor[i] for i in ids}
        c = {i: ccor[i] for i in ids}
        pr = paired(bname, cname, b, c)
        g = gate(pr.baseline_correct, n, fl)
        p = pr.as_dict()["mcnemar_exact_two_sided_p"]

        # power: could a 39% recovery of this damage even be seen?
        bench_rows = int(round(BENCHMARK_RECOVERY * pr.damage))
        reach = mcnemar_floor(bench_rows)
        powered = reach <= 0.05

        floor_ok = not str(g["verdict"]).startswith("AT_OR_BELOW")
        damaged = pr.damage > pr.repair and p <= 0.05
        ok = bool(floor_ok and damaged)
        if ok:
            qualifying.append(s)
        per[s] = {
            "n": n, "published_floor": fl,
            f"{bname}_correct": pr.baseline_correct,
            f"{cname}_correct": pr.candidate_correct,
            f"{bname}_rate": round(pr.baseline_correct / n, 4),
            f"{cname}_rate": round(pr.candidate_correct / n, 4),
            "baseline_gate": g["verdict"],
            "baseline_clears_floor": floor_ok,
            "repair": pr.repair, "damage": pr.damage,
            "damage_rate_given_baseline_correct": (
                round(pr.damage / pr.baseline_correct, 4) if pr.baseline_correct else None),
            "p_mcnemar": p, "damaged_significantly": damaged,
            "benchmark_recovery_rows": bench_rows,
            "smallest_reachable_p_at_benchmark": round(reach, 4),
            "powered_for_benchmark_effect": powered,
            "QUALIFIES": ok,
            "why_not": (None if ok else
                        ("baseline at or below floor" if not floor_ok else
                         "not significantly damaged -- dropped, as the arithmetic panel should "
                         "have been")),
        }

    pooled = paired(bname, cname, bcor, ccor)
    if not qualifying:
        stage2 = ("CANCELLED -- no stratum qualifies; these three skills are not damaged by this "
                  "delta, and Amendment 7's one-skill narrowing stands unchallenged AND untested")
    else:
        under = [s for s in qualifying if not per[s]["powered_for_benchmark_effect"]]
        stage2 = (f"RUNS on {qualifying}"
                  + (f"; UNDERPOWERED for a benchmark-size effect: {under}" if under else ""))

    out = {
        "record": "PRES_002_STAGE1",
        "class": "MECHANISM SCOUT -- eligibility screen, NOT a result",
        "prereg": "probe/PRES002_PREREG.md",
        "baseline": bname, "candidate": cname,
        "benchmark_recovery_fraction": BENCHMARK_RECOVERY,
        "per_stratum": per,
        "pooled": {"n": len(ref), f"{bname}_correct": pooled.baseline_correct,
                   f"{cname}_correct": pooled.candidate_correct,
                   "repair": pooled.repair, "damage": pooled.damage,
                   "p": pooled.as_dict()["mcnemar_exact_two_sided_p"]},
        "qualifying_strata": qualifying,
        "stage2": stage2,
        "caveats": [
            "eligibility only; no claim about direction is made or implied here",
            "a stratum the delta does not damage is DROPPED, not carried",
            "a stratum too small to resolve a 39% recovery is UNDERPOWERED, not negative",
        ],
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'stratum':<24}{'n':>5}{'floor':>7}{'supa':>7}{'cand':>7}{'rep':>5}{'dam':>5}"
          f"{'p':>10}{'bench':>7}{'minp':>8}  gate / qualifies")
    for s in strata:
        v = per[s]
        print(f"{s:<24}{v['n']:>5}{v['published_floor']:>7.3f}{v[f'{bname}_correct']:>7}"
              f"{v[f'{cname}_correct']:>7}{v['repair']:>5}{v['damage']:>5}{v['p_mcnemar']:>10.4f}"
              f"{v['benchmark_recovery_rows']:>7}{v['smallest_reachable_p_at_benchmark']:>8.3f}  "
              f"{v['baseline_gate']} / {'YES' if v['QUALIFIES'] else 'no'}")
    pd = out["pooled"]
    print(f"\npooled  n={pd['n']}  {bname} {pd[f'{bname}_correct']}  "
          f"{cname} {pd[f'{cname}_correct']}  repair {pd['repair']} damage {pd['damage']}  "
          f"p = {pd['p']:.3e}")
    print(f"\nqualifying strata: {qualifying or 'NONE'}")
    print(f"STAGE 2: {stage2}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
