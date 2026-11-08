"""Is the damage the PRICE OF ACQUISITION, or the price of MAGNITUDE?

    python analysis/coupling_ladder.py --alias <records>/alias001-20260918 \
        --frontier <records>/int001-20260917/FRONTIER.json \
        --out <records>/alias001-20260918/COUPLING_LADDER.json

**This is a LOOK, not a TEST, and it is labelled that way in the output record.** Every number
it consumes was already on disk before the question was asked: the alias dose ladder (ALIAS-002,
five arms, 360 rows each) and INT-001's acquisition ladder on SELECT-VIEWS-001, which happen to
sit on the *same five alphas* and the *same build recipe*. Nothing here was registered in
advance, so it can generate a hypothesis and it cannot confirm one. The registered successor is
`probe/SATURATION_PREREG.md`.

## The question a monotone dose response could not answer

ALIAS-002 established that alias accuracy falls monotonically with alpha, rho = -1.000 in every
salience condition. It was registered at the time that **monotone decline is necessary for the
capacity account and not sufficient for it**, because any mechanism scaling with magnitude
predicts the same curve. This script asks the question that *does* separate them, using data
that already existed.

    STRONG CAPACITY   the damage is the price paid for the new skill. Marginal damage should
                      therefore track MARGINAL ACQUISITION: where acquisition stops rising,
                      damage should stop accruing.

    MAGNITUDE         the damage is the price of moving the weights at all. Marginal damage
                      tracks ||delta|| = alpha * ||delta_1||, i.e. it is roughly linear in alpha
                      and is INDIFFERENT to whether acquisition has saturated.

These diverge sharply on the top of the ladder, because acquisition is known to saturate there
(234 -> 235 of 384 over the last quarter-dose) while alpha keeps rising.

## The statistic that expresses the difference

**Marginal damage per acquisition row gained**, computed on each of the four intervals:

    ratio(i) = (alias_acc[i-1] - alias_acc[i]) / (acquisition[i] - acquisition[i-1])

Under strong capacity this is roughly CONSTANT -- each row of new skill costs about the same
amount of old skill. Under magnitude it EXPLODES at the top, because the denominator goes to
zero while the numerator does not.

Reported alongside are the two normalised trajectories -- fraction of total acquisition gained
and fraction of total alias damage incurred, by dose -- which is the same fact stated so it can
be read off directly, and the paired McNemar on each adjacent pair of arms, which is what says
whether any individual rung moved at all.

## The apples-and-oranges caveat, which came from the peer session and is correct

Comparing an ACQUISITION endpoint's rise against a PRESERVATION endpoint's fall is not a like
for like comparison of levels, and no claim here rests on their relative percentages. The
comparison being made is one of SHAPE over a shared x-axis: both curves are measured on the same
five hash-bound arms, so "acquisition has stopped moving while alias has not" is a statement
about two trajectories on one ladder, not about two magnitudes on one scale.

`references_retained` (n = 47) is carried as a second preservation reference precisely because
it is the endpoint the peer trusts, and its noise floor at that n is stated rather than hidden.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired import paired  # noqa: E402

#: arm directory -> alpha. Same build recipe and same 360 rendered prompts throughout.
ARMS = [("gen-B", 0.00), ("gen-a025", 0.25), ("gen-a05", 0.50),
        ("gen-a075", 0.75), ("gen-a10", 1.00)]
CONDITIONS = ["low", "equal", "high"]


def norm(v) -> str:
    """Identical to probe/score_alias_probe.py::norm -- scoring convention must not drift."""
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alias", type=Path, required=True)
    ap.add_argument("--frontier", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            (a.alias / "ALIAS_PROBE.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ref = {r["id"]: r["reference_normalised"] for r in rows}
    cond = {r["id"]: r["condition"] for r in rows}

    correct: dict[float, dict[str, bool]] = {}
    for d, alpha in ARMS:
        p = {r["id"]: r.get("prediction", "")
             for r in (json.loads(l) for l in
                       (a.alias / d / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                       if l.strip())}
        missing = [i for i in ref if i not in p]
        if missing:
            raise SystemExit(f"arm {d} is missing {len(missing)} of {len(ref)} rows")
        correct[alpha] = {i: norm(p[i]) == ref[i] for i in ref}

    fr = json.loads(a.frontier.read_text(encoding="utf-8"))
    acq = {0.0: 14}   # alpha = 0 is Supa, recorded in INT001_RESULTS.md as a prior measurement
    ret = {0.0: 47}
    ari = {0.0: 47}
    for arm in fr["arms"]:
        acq[float(arm["alpha"])] = int(arm["acquisition"])
        ret[float(arm["alpha"])] = int(arm["retained"])
        ari[float(arm["alpha"])] = int(arm["arithmetic_permissive"])

    alphas = [x for _, x in ARMS]
    n = len(ref)
    acc = {al: sum(correct[al].values()) / n for al in alphas}

    # ---- adjacent-rung paired tests -------------------------------------------------------
    rungs = []
    for lo, hi in zip(alphas, alphas[1:]):
        pr = paired(f"a{lo}", f"a{hi}", correct[lo], correct[hi])
        d_acq = acq[hi] - acq[lo]
        d_alias = acc[lo] - acc[hi]
        rungs.append({
            "from": lo, "to": hi,
            "alias_correct_from": sum(correct[lo].values()),
            "alias_correct_to": sum(correct[hi].values()),
            "alias_repair": pr.repair, "alias_damage": pr.damage,
            "alias_p_mcnemar": pr.as_dict()["mcnemar_exact_two_sided_p"],
            "alias_drop_pp": round(d_alias * 100, 2),
            "acquisition_from": acq[lo], "acquisition_to": acq[hi], "acquisition_gain": d_acq,
            # The statistic. Percentage points of alias accuracy surrendered per acquisition
            # row gained. Constant under strong capacity; explodes at the top under magnitude.
            "pp_alias_per_acq_row": (round(d_alias * 100 / d_acq, 4) if d_acq > 0 else None),
        })

    total_acq = acq[1.0] - acq[0.0]
    total_dam = acc[0.0] - acc[1.0]
    traj = [{
        "alpha": al,
        "alias_acc": round(acc[al], 4),
        "acquisition": acq[al],
        "frac_acquisition_gained": round((acq[al] - acq[0.0]) / total_acq, 4),
        "frac_alias_damage_incurred": round((acc[0.0] - acc[al]) / total_dam, 4),
        "references_retained": ret[al],
        "arithmetic_permissive": ari[al],
    } for al in alphas]

    # ---- the saturation region, which is where the accounts disagree ----------------------
    sat = next(r for r in rungs if r["from"] == 0.75)
    by_cond = {}
    for c in CONDITIONS:
        ids = [i for i in ref if cond[i] == c]
        pr = paired("a0.75", "a1.0",
                    {i: correct[0.75][i] for i in ids}, {i: correct[1.0][i] for i in ids})
        by_cond[c] = {"n": len(ids), "from": pr.baseline_correct, "to": pr.candidate_correct,
                      "repair": pr.repair, "damage": pr.damage,
                      "p": pr.as_dict()["mcnemar_exact_two_sided_p"]}

    ratios = [r["pp_alias_per_acq_row"] for r in rungs if r["pp_alias_per_acq_row"] is not None]
    spread = (max(ratios) / min(ratios)) if ratios and min(ratios) > 0 else None

    out = {
        "record": "COUPLING_LADDER",
        "class": "LOOK -- hypothesis-generating, not a test; every input predated the question",
        "registered_successor": "probe/SATURATION_PREREG.md",
        "question": ("does damage track MARGINAL ACQUISITION (strong capacity) or MAGNITUDE "
                     "(alpha), on the interval where acquisition has saturated"),
        "trajectory": traj,
        "rungs": rungs,
        "marginal_ratio_spread_max_over_min": (round(spread, 1) if spread else None),
        "saturation_region_0p75_to_1p00": {
            "acquisition_gain_rows": sat["acquisition_gain"],
            "acquisition_gain_frac_of_total": round(sat["acquisition_gain"] / total_acq, 4),
            "alias_drop_pp": sat["alias_drop_pp"],
            "alias_drop_frac_of_total_damage": round(
                (acc[0.75] - acc[1.0]) / total_dam, 4),
            "alias_p_mcnemar": sat["alias_p_mcnemar"],
            "by_condition": by_cond,
        },
        "caveats": [
            "LOOK, not a test: the hypothesis was generated by placing two existing tables side "
            "by side, so nothing here is confirmatory and no account is settled by it",
            "acquisition rise vs preservation fall is not a like-for-like comparison of LEVELS; "
            "the claim made is about the SHAPE of two trajectories over one shared alpha axis",
            "references_retained is n = 47, so its last-rung +1 is inside its own noise and is "
            "reported rather than interpreted",
            "arithmetic_permissive RISES across the ladder, so it is a partially-exercised "
            "endpoint and is not a damage curve; it is carried for completeness",
            "one training draw of one delta; the alias bank is a mechanism scout with no seal",
        ],
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'alpha':>6}{'alias':>8}{'acq/384':>9}{'%acq':>7}{'%dmg':>7}{'ret/47':>8}{'arith':>7}")
    for t in traj:
        print(f"{t['alpha']:>6.2f}{t['alias_acc']:>8.3f}{t['acquisition']:>9}"
              f"{t['frac_acquisition_gained']*100:>6.1f}%{t['frac_alias_damage_incurred']*100:>6.1f}%"
              f"{t['references_retained']:>8}{t['arithmetic_permissive']:>7}")

    print(f"\n{'rung':>14}{'d_acq':>7}{'d_alias':>9}{'pp/row':>9}{'rep':>5}{'dam':>5}{'p':>10}")
    for r in rungs:
        pr = "n/a" if r["pp_alias_per_acq_row"] is None else f"{r['pp_alias_per_acq_row']:.3f}"
        print(f"{str(r['from'])+'->'+str(r['to']):>14}{r['acquisition_gain']:>7}"
              f"{r['alias_drop_pp']:>8.1f}p{pr:>9}{r['alias_repair']:>5}{r['alias_damage']:>5}"
              f"{r['alias_p_mcnemar']:>10.4f}")
    if spread:
        print(f"\nmarginal cost per acquisition row spans {spread:.0f}x across the ladder")
    print(f"\nsaturation region 0.75 -> 1.00: acquisition +{sat['acquisition_gain']}/384 "
          f"({sat['acquisition_gain']/total_acq*100:.1f}% of total gain), "
          f"alias {sat['alias_drop_pp']:.1f}pp "
          f"({(acc[0.75]-acc[1.0])/total_dam*100:.1f}% of total damage), "
          f"McNemar p = {sat['alias_p_mcnemar']:.4f}")
    for c in CONDITIONS:
        v = by_cond[c]
        print(f"    {c:<6} {v['from']:>3} -> {v['to']:>3}  repair {v['repair']:>2} "
              f"damage {v['damage']:>2}  p = {v['p']:.4f}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
