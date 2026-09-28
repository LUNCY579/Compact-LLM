"""ABST-001 -- null calibration and power of the REGISTERED statistic (abst_verdict.verdict), CPU only.

Every cell draws 64 rows per family WITHOUT replacement from finite pools of the real fresh sizes
(112/109/109/102), with skewed heavy-tailed NLL log-ratios, and feeds them through verdict() itself.
  N1a/N1b H1 null: pools half worse (odd pools rounded DOWN / UP -- review: 'exactly half' is not
          possible at 109 rows, so both tilts are measured)
  N2/N3   heterogeneous H1 nulls (2/3,2/3,2/3,0) and its mirror
  N4      UNKNOWN-anchor null (teacher at the log-midpoint of plain/lc + symmetric noise): any step-2
          rejection counts as false
  D1      R1 (cold review M1): TEACHER-BLIND ANSWER-TYPE SHIFT -- LC moves probability mass from UNKNOWN to
          the digits regardless of the teacher (UNKNOWN NLL up on 70% of rows, gold-digit NLL down on 85%, noisy),
          with the spent geometry (UNKNOWN: teacher 1.14-2.85x plain; known: teacher 0.11-0.57x plain). REPORTED,
          not checked: TEACHER_CONSISTENT_DIRECTION fires here too -- the label is direction-consistent, not
          teacher-specific (this is why no outcome licenses abstention). The rate is a property of THIS
          alternative's strength, not of the design.
  D2      TEACHER-BLIND GENERIC DAMAGE: MOST NLLs up (70% of UNKNOWN rows, 85% of known rows), noisy, with the
          spent known-row geometry (plain ABOVE the teacher on every known row) -> TEACHER_CONSISTENT_DIRECTION
          must stay <= 0.025. R1 review: under generic damage a known row can move toward the teacher only where
          plain sits BELOW the teacher, so this check holds BY CONSTRUCTION of the geometry; the guarantee on real
          data is the registered count rule (prereg §9: generic damage is excluded only if fewer than 147 known
          rows have plain NLL below the teacher's). D2r measures the dependence:
  D2r     the same generic damage with 60% of known rows reversed (plain below the teacher): REPORTED
  N5/N6   Holm at the split alpha 0.0125, false rejections among TRUE nulls only (sign nulls / contrast nulls)
  P1-P5   power: lead shape; homogeneous 0.60 / 0.65; TEACHER_CONSISTENT_DIRECTION truth; Holm DEPTH truth
  P4b     TEACHER_CONSISTENT_DIRECTION power when the known-row effect is only 0.60 toward (REPORTED; R1 review:
          P4's 1.0 is a property of its extreme alternative)
  S1      superpopulation family random effect -> the row test is anticonservative for NEW families
Draws: --sims for N1-N4, D1, D2, D2r; --sims/4 for N5, N6, P1-P5, P4b, S1.
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
TOWARD = ("TEACHER_CONSISTENT_DIRECTION", "TOWARD_TEACHER_OVER_SUPPRESSION", "TOWARD_TEACHER_NOT_CORROBORATED")
TCD = "TEACHER_CONSISTENT_DIRECTION"


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


def known(rng, toward=True, p_toward=0.85, reversed_share=0.0):
    """Known rows, NOISY (R1: a first version was deterministic -- LC 0.7x/1.3x and teacher 0.3x on every row).
    Teacher 0.11-0.57x plain (the spent range), except a fraction `reversed_share` of rows where the teacher sits
    ABOVE plain (1.75-9.1x); LC moves the digit NLL down on a fraction p_toward of rows (toward=True) or up on
    that fraction (toward=False), by a random factor."""
    n = len(POOLS) * PER
    kp = [math.exp(rng.gauss(1.0, 0.4)) for _ in range(n)]
    kt = [p * (rng.uniform(1.75, 9.1) if reversed_share and rng.random() < reversed_share else rng.uniform(0.11, 0.57))
          for p in kp]
    frac = p_toward if toward else 1 - p_toward
    kl = [p * (math.exp(-abs(rng.gauss(0, 0.3))) if rng.random() < frac else math.exp(abs(rng.gauss(0, 0.3)))) for p in kp]
    return {"A_known": {"plain": kp, "lc": kl}, "T4_known": {"teacher": kt}}


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
    for name, known_toward, rev, reading in (
            ("D1_teacher_blind_answer_type_shift", True, 0.0, "REPORTED: the label is not teacher-specific"),
            ("D2_teacher_blind_generic_damage", False, 0.0,
             "must stay <= 0.025; holds by construction of the spent known-row geometry (plain above the teacher)"),
            ("D2r_generic_damage_60pct_known_rows_reversed", False, 0.6,
             "REPORTED: with most known rows reversed, step 2b no longer excludes generic damage")):
        fid = tow = 0
        for _ in range(S):
            pl = [math.exp(rng.gauss(1.4, 0.3)) for _ in range(len(POOLS) * PER)]
            te = [p * rng.uniform(1.14, 2.85) for p in pl]
            lc = [p * math.exp(abs(rng.gauss(0, 0.4))) if rng.random() < 0.7 else p / math.exp(abs(rng.gauss(0, 0.4))) for p in pl]
            v = av.verdict({"A": {"plain": pl, "lc": lc}, "T4": {"teacher": te},
                            **known(rng, toward=known_toward, reversed_share=rev)})
            tow += v["H2"]["label"] in TOWARD
            fid += v["H2"]["label"] == TCD
        C[name] = {"TEACHER_CONSISTENT_DIRECTION_rate": rate(fid, S), "step2_toward_rate": tow / S, "reading": reading}
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
        k += av.verdict({"A": {"plain": pl, "lc": lc}, "T4": {"teacher": beyond_teacher(pl, lc)}, **known(rng)})["H2"]["label"] == TCD
    C["P4_teacher_consistent_truth"] = {"power_TEACHER_CONSISTENT_DIRECTION": k / S4,
                                        "note": "UNKNOWN rows all closer, known rows 85% toward: an extreme alternative"}
    k = 0
    for _ in range(S4):
        pl, lc = cell_from_signs(rng, draw_signs(rng, (0.8, 0.1, 0.8, 0.8)))
        k += av.verdict({"A": {"plain": pl, "lc": lc}, "T4": {"teacher": beyond_teacher(pl, lc)},
                         **known(rng, p_toward=0.60)})["H2"]["label"] == TCD
    C["P4b_teacher_consistent_known_060"] = {"power_TEACHER_CONSISTENT_DIRECTION": k / S4, "reading": "REPORTED"}
    k = 0
    for _ in range(S4):
        pl, lc = cell_from_signs(rng, draw_signs(rng, (0.8, 0.1, 0.8, 0.8)))
        cells = {"A": {"plain": pl, "lc": lc}, "T4": {"teacher": beyond_teacher(pl, lc)}, **known(rng),
                 "D": dict(zip(("plain", "lc"), cell_from_signs(rng, [not w for w in draw_signs(rng, (0.9,) * 4)]))),
                 "B": dict(zip(("plain", "lc"), cell_from_signs(rng, draw_signs(rng, (0.8,) * 4)))),
                 "C": dict(zip(("plain", "lc"), cell_from_signs(rng, [not w for w in draw_signs(rng, (0.8,) * 4)])))}
        k += av.verdict(cells)["H3"]["label"] == "SIGN_FOLLOWS_DEPTH"
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
        "D2 TEACHER_CONSISTENT_DIRECTION under teacher-blind generic damage <= 0.025 + 3SE":
            C["D2_teacher_blind_generic_damage"]["TEACHER_CONSISTENT_DIRECTION_rate"]["rate"] <= bound(0.025, S),
        "N5/N6 Holm FWER <= 0.0125 + 3SE": all(C[n]["fwer_over_true_nulls"]["rate"] <= bound(av.ALPHA_AFTER_ANCHOR, S4)
                                              for n in ("N5_holm_sign_nulls", "N6_holm_contrast_nulls")),
        "P1 power >= 0.95": C["P1_lead_shape"]["power_LC_LOWERS"] >= 0.95,
        "P4 TEACHER_CONSISTENT_DIRECTION power >= 0.95": C["P4_teacher_consistent_truth"]["power_TEACHER_CONSISTENT_DIRECTION"] >= 0.95,
        "P5 DEPTH power >= 0.90": C["P5_holm_depth_truth"]["power_DEPTH_label"] >= 0.90,
    }
    res["checks"], res["all_pass"] = checks, all(checks.values())
    with open(a.out, "x", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"checks": checks, "all_pass": res["all_pass"]}, indent=1))
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
