"""ROTATION-002 — is `delta-compression-is-magnitude` true on a preservation endpoint?

    python probe/score_rotation2.py --root <records>/alias001-20260918 \
        --norms <records>/ROTATION_NORMS.json --arm-norms <records>/ROT2_NORMS.json \
        --out <records>/ROTATION2_RESULT.md.json

**Written before the arm finished generating**, as `probe/ROTATION001_PREREG.md` Addendum 1
registers. MECHANISM SCOUT.

`k = 0.25` (a coordinate mask) and the α = 0.836382 arm (a scalar) have **identical norm** and
sit at **cosine 0.836382** to each other — ROTATION-001 showed the mask is a projection, so its
cosine to the ray equals its norm ratio exactly. This is therefore a clean same-norm,
different-direction paired contrast on 360 rows.

The programme's standing conclusion is that mask, rank truncation and scalar scaling are **one
lever acting through magnitude**. That was established on acquisition, plus preservation
endpoints of n = 47 where a 2.7 pp difference is a single row. This bank is n = 360.

**Primary: exact two-sided McNemar, mask vs scalar. One comparison, no multiplicity correction**
— Holm over a family of one is the identity. Registered direction: mask > scalar.

**Secondary: the interpolation error.** ROTATION-001's +2.7 pp residual assumed the α ladder is
linear between α = 0.75 (0.633) and α = 1.00 (0.603), giving ≈ 0.623 at 0.836. The measured
value replaces it, and the difference is reported whatever it is — if the interpolation was badly
wrong, every residual in this programme computed against an α-ladder interpolation needs
re-deriving, and that is the most consequential thing this arm could say.

Absolute rates in the `high` salience condition are not readable by construction; paired
contrasts are readable in every condition. Repair and damage are never netted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from paired import paired  # noqa: E402

CONDITIONS = ["low", "equal", "high"]
#: the alpha ladder points bracketing 0.836, from INT-001 + ALIAS-002, for the interpolation check
LADDER = {0.75: 0.6333, 1.00: 0.6028}
INTERPOLATED_AT_0836 = 0.6228


def norm(v) -> str:
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--norms", type=Path, required=True)
    ap.add_argument("--arm-norms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            (a.root / "ALIAS_PROBE.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ref = {r["id"]: r["reference_normalised"] for r in rows}
    cond = {r["id"]: r["condition"] for r in rows}
    n = len(ref)

    arms = {"mask_k025": "gen-D", "scalar_a0836": "gen-rot2_a0836",
            "supa": "gen-B", "alpha100": "gen-a10", "alpha075": "gen-a075"}
    correct = {}
    for lab, d in arms.items():
        f = a.root / d / "predictions.jsonl"
        if not f.exists():
            raise SystemExit(f"missing predictions for {lab}: {f}")
        p = {r["id"]: r.get("prediction", "") for r in
             (json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip())}
        if [i for i in ref if i not in p]:
            raise SystemExit(f"arm {lab} does not cover all {n} rows")
        correct[lab] = {i: norm(p[i]) == ref[i] for i in ref}

    rn = json.loads(a.norms.read_text(encoding="utf-8"))["norms"]
    an = json.loads(a.arm_norms.read_text(encoding="utf-8"))["norms"]
    target = rn["k025"]["fro"]
    achieved = an["rot2"]["fro"]
    err = abs(achieved - target)
    if err > 1e-4:
        raise SystemExit(f"norm match failed: built {achieved:.6f} vs target {target:.6f} "
                         f"(err {err:.2e}); refusing to score")

    acc = {k: sum(v.values()) / n for k, v in correct.items()}
    pr = paired("scalar_a0836", "mask_k025", correct["scalar_a0836"], correct["mask_k025"])
    p = pr.as_dict()["mcnemar_exact_two_sided_p"]

    by_c = {}
    for c in CONDITIONS:
        ids = [i for i in ref if cond[i] == c]
        q = paired("scalar_a0836", "mask_k025",
                   {i: correct["scalar_a0836"][i] for i in ids},
                   {i: correct["mask_k025"][i] for i in ids})
        by_c[c] = {"n": len(ids), "scalar": q.baseline_correct, "mask": q.candidate_correct,
                   "mask_gains": q.repair, "mask_losses": q.damage,
                   "p": q.as_dict()["mcnemar_exact_two_sided_p"],
                   "absolute_readable": c != "high"}

    diff_pp = (acc["mask_k025"] - acc["scalar_a0836"]) * 100
    interp_err_pp = (acc["scalar_a0836"] - INTERPOLATED_AT_0836) * 100
    band = 2.5
    if diff_pp > band and p <= 0.05:
        verdict = "ONE_LEVER_FAILS_ON_PRESERVATION -- the mask preserves more at identical norm"
    elif abs(diff_pp) <= band and p > 0.05:
        verdict = ("ONE_LEVER_HOLDS_ON_PRESERVATION -- extends the standing conclusion to a "
                   "well-powered preservation endpoint; ROTATION-001 section 4 flag withdrawn")
    elif diff_pp < -band and p <= 0.05:
        verdict = "REVERSED -- the scalar preserves more; no account predicts this"
    else:
        verdict = "INCONCLUSIVE -- reported as such, not forced"

    out = {
        "record": "ROTATION_002_RESULT",
        "class": "MECHANISM SCOUT -- not a confirmation; no read budget was spent",
        "prereg": "probe/ROTATION001_PREREG.md Addendum 1",
        "question": ("does delta-compression-is-magnitude hold on a preservation endpoint with "
                     "adequate power (n = 360, not n = 47)"),
        "norm_match": {"target_k025_fro": target, "achieved_scalar_fro": achieved,
                       "abs_err": err, "cosine_between_them": 0.836382,
                       "why_cosine_equals_norm_ratio": (
                           "ROTATION-001: a coordinate mask is a PROJECTION, so the masked-away "
                           "part is exactly orthogonal to the retained part")},
        "accuracies": {k: round(v, 4) for k, v in acc.items()},
        "primary": {
            "comparison": "mask k=0.25 vs scalar alpha=0.836382, identical norm",
            "mask_acc": round(acc["mask_k025"], 4),
            "scalar_acc": round(acc["scalar_a0836"], 4),
            "difference_pp": round(diff_pp, 2),
            "mask_gains": pr.repair, "mask_losses": pr.damage,
            "p_mcnemar_exact_two_sided": p,
            "multiplicity": "none applied -- one comparison, Holm over a family of one is the identity",
            "by_condition": by_c,
        },
        "interpolation_check": {
            "alpha_ladder_bracket": LADDER,
            "interpolated_at_0836": INTERPOLATED_AT_0836,
            "measured_at_0836": round(acc["scalar_a0836"], 4),
            "interpolation_error_pp": round(interp_err_pp, 2),
            "rotation001_residual_was_pp": round(
                (acc["mask_k025"] - INTERPOLATED_AT_0836) * 100, 2),
            "rotation001_residual_becomes_pp": round(diff_pp, 2),
            "consequence_if_large": ("every residual in this programme computed against an "
                                     "alpha-ladder interpolation would need re-deriving"),
        },
        "verdict": verdict,
        "caveats": [
            "registered direction was mask > scalar, and that is the direction already expected "
            "from the interpolation -- weaker than a blind prediction, and said so in the prereg",
            "absolute rates in `high` are not readable by construction; paired contrasts are",
            "one delta, one bank, one draw; a mechanism scout promotes nothing",
        ],
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"norm match: target {target:.6f}  achieved {achieved:.6f}  err {err:.2e}")
    print(f"\nPRIMARY -- same norm, cosine 0.836382 apart")
    print(f"  mask   k=0.25          {acc['mask_k025']:.4f}")
    print(f"  scalar alpha=0.836382  {acc['scalar_a0836']:.4f}")
    print(f"  difference {diff_pp:+.1f}pp   mask gains {pr.repair} / loses {pr.damage}   "
          f"p = {p:.4f}")
    print(f"\n{'cond':<7}{'n':>5}{'scalar':>8}{'mask':>7}{'gain':>6}{'loss':>6}{'p':>9}  readable")
    for c in CONDITIONS:
        v = by_c[c]
        print(f"{c:<7}{v['n']:>5}{v['scalar']:>8}{v['mask']:>7}{v['mask_gains']:>6}"
              f"{v['mask_losses']:>6}{v['p']:>9.4f}  {'yes' if v['absolute_readable'] else 'NO'}")
    print(f"\nINTERPOLATION CHECK")
    print(f"  alpha ladder assumed {INTERPOLATED_AT_0836:.4f} at 0.836; "
          f"measured {acc['scalar_a0836']:.4f}  ({interp_err_pp:+.1f}pp)")
    print(f"  ROTATION-001's residual {(acc['mask_k025']-INTERPOLATED_AT_0836)*100:+.1f}pp "
          f"becomes {diff_pp:+.1f}pp")
    print(f"\nVERDICT: {verdict}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
