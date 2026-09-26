"""ABST-001 -- null calibration and power of the REGISTERED statistic (abst_verdict.verdict), CPU only.

Every cell draws 64 rows per family WITHOUT replacement from finite pools of the real fresh sizes
(112/109/109/102), with skewed heavy-tailed NLL log-ratios, and feeds them through verdict() itself.
  N1a/N1b H1 null: pools half worse (odd pools rounded DOWN / UP -- review: 'exactly half' is not
          possible at 109 rows, so both tilts are measured)
  N2/N3   heterogeneous H1 nulls (2/3,2/3,2/3,0) and its mirror
  N4      UNKNOWN-anchor null (teacher at the log-midpoint of plain/lc + symmetric noise): any step-2
          rejection counts as false
  D1      TEACHER-BLIND GENERIC DAMAGE (review): LC inflates NLL on UNKNOWN and known rows regardless of
          the teacher, with the spent geometry (UNKNOWN: teacher 1.14-2.85x above plain; known: teacher at
          0.3x plain) -> FIDELITY must stay <= 0.025 (step 2 alone says TOWARD most of the time)
  N5/N6   Holm at the split alpha 0.0125, false rejections among TRUE nulls only (sign nulls / contrast nulls)
  P1-P6   power: lead shape; homogeneous 0.60 / 0.65; FIDELITY truth; Holm DEPTH truth
  S1      superpopulation family random effect -> the row test is anticonservative for NEW families
usage: python design/abst_null_sim.py --out design/ABST_NULL_SIM.json [--sims 20000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import abst_verdict as av  # noqa: E402

POOLS = (112, 109, 109, 102)
PER = 64
TOWARD = ("FIDELITY", "TOWARD_TEACHER_OVERSHOOT", "TOWARD_TEACHER_NOT_CORROBORATED")


def draw_signs(rng, p_by_family, rounding=round):
    out = []
    for size, p in zip(POOLS, p_by_family):
        k = int(rounding(p * size))
        out += rng.sample([True] * k + [False] * (size - k), PER)
    return out


def nll_pair(rng, worse):
    a = math.exp(rng.gauss(1.4, 0.6))
    r = math.exp(abs(rng.lognormvariate(0, 1.0) - 0.5) + 1e-9)
    return a, (a * r if worse else a / r)


def cell_from_signs(rng, worse):
    pl, lc = zip(*(nll_pair(rng, w) for w in worse))
    return list(pl), list(lc)


def midpoint_teacher(rng, pl, lc):
    return [math.exp((math.log(p) + math.log(q)) / 2 + rng.gauss(0, 1.0)) for p, q in zip(pl, lc)]


def beyond_teacher(pl, lc):
    return [q * math.exp(0.5 * math.log(q / p)) for p, q in zip(pl, lc)]


def known(rng, toward=True):
    kp = [math.exp(rng.gauss(1.0, 0.4)) for _ in range(len(POOLS) * PER)]
    return {"A_known": {"plain": kp, "lc": [p * (0.7 if toward else 1.3) for p in kp]},
            "T4_known": {"teacher": [p * 0.3 for p in kp]}}


def rate(k, n):
    return {"rate": k / n, "k": k, "n": n, "upper_3se": k / n + 3 * math.sqrt(max(k / n * (1 - k / n), 1e-12) / n)}


def bound(alpha, n):
    return alpha + 3 * math.sqrt(alpha * (1 - alpha) / n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260926)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (create-only)")
    rng = random.Random(a.seed)
    S, S4 = a.sims, a.sims // 4
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (HERE / "abst_verdict.py", HERE / "hlcc_verdict.py", Path(__file__).resolve())}
    res = {"schema": 1, "record": "ABST_NULL_SIM", "code_sha256": code, "seed": a.seed, "sims": S,
           "alpha": av.ALPHA, "alpha_anchor_tail": av.ALPHA_ANCHOR_TAIL, "alpha_after_anchor": av.ALPHA_AFTER_ANCHOR,
           "pools": POOLS, "per_family": PER, "cells": {}}
    C = res["cells"]
    for name, fam, rnd in (("N1a_half_floor", (0.5,) * 4, math.floor), ("N1b_half_ceil", (0.5,) * 4, math.ceil),
                           ("N2_hetero", (2 / 3, 2 / 3, 2 / 3, 0.0), round), ("N3_mirror_hetero", (1 / 3, 1 / 3, 1 / 3, 1.0), round)):
        low = high = 0
        for _ in range(S):
            pl, lc = cell_from_signs(rng, draw_signs(rng, fam, rnd))
            lab = av.verdict({"A": {"plain": pl, "lc": lc}, "T4": {"teacher": pl}})["H1"]["label"]
            low += lab == "LC_LOWERS_UNKNOWN"
            high += lab == "LC_RAISES_UNKNOWN"
        C[name] = {"false_LC_LOWERS": rate(low, S), "false_LC_RAISES": rate(high, S)}
    k = 0
    for _ in range(S):
        pl, lc = cell_from_signs(rng, draw_signs(rng, (0.8, 0.1, 0.8, 0.8)))
        v = av.verdict({"A": {"plain": pl, "lc": lc}, "T4": {"teacher": midpoint_teacher(rng, pl, lc)}, **known(rng)})
        k += v["H2"]["label"] in TOWARD + ("AWAY_FROM_TEACHER",)
    C["N4_unknown_anchor_null"] = {"false_step2_rejection": rate(k, S)}
    fid = tow = 0
    for _ in range(S):
        pl = [math.exp(rng.gauss(1.4, 0.3)) for _ in range(len(POOLS) * PER)]
        te = [p * rng.uniform(1.14, 2.85) for p in pl]
        lc = [p * math.exp(abs(rng.gauss(0, 0.4))) if rng.random() < 0.7 else p / math.exp(abs(rng.gauss(0, 0.4))) for p in pl]
        v = av.verdict({"A": {"plain": pl, "lc": lc}, "T4": {"teacher": te}, **known(rng, toward=False)})
        tow += v["H2"]["label"] in TOWARD
        fid += v["H2"]["label"] == "FIDELITY"
    C["D1_teacher_blind_generic_damage"] = {"false_FIDELITY": rate(fid, S), "step2_toward_rate": tow / S,
                                            "reading": "step 2 alone cannot tell; step 2b keeps FIDELITY at alpha"}
    for name, kind in (("N5_holm_sign_nulls", "sign"), ("N6_holm_contrast_nulls", "contrast")):
        fwer = reached = 0
        for _ in range(S4):
            pl, lc = cell_from_signs(rng, draw_signs(rng, (0.8, 0.1, 0.8, 0.8)))
            cells = {"A": {"plain": pl, "lc": lc}, "T4": {"teacher": beyond_teacher(pl, lc)}, **known(rng)}
            cells["D"] = dict(zip(("plain", "lc"), cell_from_signs(rng, draw_signs(rng, (0.5,) * 4))))
            if kind == "sign":
                for X in ("B", "C"):
                    cells[X] = dict(zip(("plain", "lc"), cell_from_signs(rng, draw_signs(rng, (0.5,) * 4))))
                nulls = ("D+", "D-", "B+", "B-", "C+", "C-")
            else:
                for X in ("B", "C"):
                    cells[X] = dict(zip(("plain", "lc"), cell_from_signs(rng, draw_signs(rng, (0.8, 0.1, 0.8, 0.8)))))
                nulls = ("D+", "D-", "dV", "dD")
            v = av.verdict(cells)
            if v["H3"].get("holm_rejected"):
                reached += 1
                fwer += any(v["H3"]["holm_rejected"][h] for h in nulls)
        C[name] = {"fwer_over_true_nulls": rate(fwer, S4), "step3_reached": reached / S4, "true_nulls": list(nulls)}
    for name, fam in (("P1_lead_shape", (0.8, 0.1, 0.8, 0.8)), ("P2_homog_060", (0.6,) * 4), ("P3_homog_065", (0.65,) * 4)):
        k = 0
        for _ in range(S4):
            pl, lc = cell_from_signs(rng, draw_signs(rng, fam))
            k += av.verdict({"A": {"plain": pl, "lc": lc}, "T4": {"teacher": pl}})["H1"]["label"] == "LC_LOWERS_UNKNOWN"
        C[name] = {"power_LC_LOWERS": k / S4}
    k = 0
    for _ in range(S4):
        pl, lc = cell_from_signs(rng, draw_signs(rng, (0.8, 0.1, 0.8, 0.8)))
        k += av.verdict({"A": {"plain": pl, "lc": lc}, "T4": {"teacher": beyond_teacher(pl, lc)}, **known(rng)})["H2"]["label"] == "FIDELITY"
    C["P4_fidelity_truth"] = {"power_FIDELITY": k / S4}
    k = 0
    for _ in range(S4):
        pl, lc = cell_from_signs(rng, draw_signs(rng, (0.8, 0.1, 0.8, 0.8)))
        cells = {"A": {"plain": pl, "lc": lc}, "T4": {"teacher": beyond_teacher(pl, lc)}, **known(rng),
                 "D": dict(zip(("plain", "lc"), cell_from_signs(rng, [not w for w in draw_signs(rng, (0.9,) * 4)]))),
                 "B": dict(zip(("plain", "lc"), cell_from_signs(rng, draw_signs(rng, (0.8,) * 4)))),
                 "C": dict(zip(("plain", "lc"), cell_from_signs(rng, [not w for w in draw_signs(rng, (0.8,) * 4)])))}
        k += av.verdict(cells)["H3"]["label"] == "DEPTH"
    C["P5_holm_depth_truth"] = {"power_DEPTH_label": k / S4}
    k = 0
    for _ in range(S4):
        fam = [min(max(0.5 + rng.gauss(0, 0.15), 0.0), 1.0) for _ in range(4)]
        pl, lc = cell_from_signs(rng, draw_signs(rng, fam))
        k += av.verdict({"A": {"plain": pl, "lc": lc}, "T4": {"teacher": pl}})["H1"]["label"] == "LC_LOWERS_UNKNOWN"
    C["S1_superpopulation_family_effect_sd015"] = {"false_LC_LOWERS_for_new_families": rate(k, S4),
                                                   "reading": "above alpha: the row-level test licenses the POOLS, not new families"}
    checks = {
        "N1-N3 each directional false rate <= 0.025 + 3SE": all(
            C[n][t]["rate"] <= bound(av.ALPHA, S) for n in ("N1a_half_floor", "N1b_half_ceil", "N2_hetero", "N3_mirror_hetero")
            for t in ("false_LC_LOWERS", "false_LC_RAISES")),
        "N4 false step-2 rejection <= 0.025 + 3SE": C["N4_unknown_anchor_null"]["false_step2_rejection"]["rate"] <= bound(0.025, S),
        "D1 false FIDELITY under teacher-blind damage <= 0.025 + 3SE": C["D1_teacher_blind_generic_damage"]["false_FIDELITY"]["rate"] <= bound(0.025, S),
        "N5/N6 Holm FWER <= 0.0125 + 3SE": all(C[n]["fwer_over_true_nulls"]["rate"] <= bound(av.ALPHA_AFTER_ANCHOR, S4)
                                              for n in ("N5_holm_sign_nulls", "N6_holm_contrast_nulls")),
        "P1 power >= 0.95": C["P1_lead_shape"]["power_LC_LOWERS"] >= 0.95,
        "P4 FIDELITY power >= 0.95": C["P4_fidelity_truth"]["power_FIDELITY"] >= 0.95,
        "P5 DEPTH power >= 0.90": C["P5_holm_depth_truth"]["power_DEPTH_label"] >= 0.90,
    }
    res["checks"], res["all_pass"] = checks, all(checks.values())
    with open(a.out, "x", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"checks": checks, "all_pass": res["all_pass"]}, indent=1))
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
