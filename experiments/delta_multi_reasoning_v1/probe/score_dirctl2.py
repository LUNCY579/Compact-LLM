"""DIRCTL-002 — is the alias sparing geometric, or specific to the anchor?

    python probe/score_dirctl2.py --root <records>/alias001-20260918 \
        --ord23-norms <records>/lwf001-20260918/ORD23_NORMS.json \
        --twin-norms <records>/lwf001-20260918/D2TWIN_NORMS.json \
        --dirctl <records>/lwf001-20260918/DIRCTL_RESULT.json \
        --out <records>/lwf001-20260918/DIRCTL2_RESULT.json

**Written before `ord23` finished generating**, as `probe/LWF_ALIAS_PREREG.md` Addendum 3
registers. MECHANISM SCOUT.

DIRCTL-001 showed the anchored arms spare alias binding beyond their magnitude (+7.2 and
+9.4 pp, Holm p = 2.4e-05 and 4.1e-06). **Both were produced by the same anchor loss**, so the
supported claim is "this anchor spares alias", not "direction spares alias".

`arm-lambda0-ord23` is the peer session's control: λ=0 rotated by **row order alone** — same 384
rows, same pure-CE objective, no anchor. Measured here at ‖δ‖ = 0.849139 and cos 0.593855 to
λ=0, i.e. **53.6 degrees off the ray at 1.02x the norm**.

Its collinear twin is an **extrapolation** past λ=0 at `a = 1.020245875`, built by
`analysis/build_scaled_arm.py` with the achieved norm verified before writing. DIRCTL-001
measured the ray as APPROXIMATELY_LINEAR, so a 2% extension is mild — but it is extrapolation
and is labelled as such.

    GEOMETRY         ord23 shows a positive residual comparable to the anchored arms
    ANCHOR_SPECIFIC  ord23's residual is ~0; Amendment 4 narrows in the record
    REVERSED         negative; no account predicts it

**Primary: exact two-sided McNemar, ord23 vs twin, 360 paired rows. One comparison, so no
multiplicity correction** — Holm over a family of one is the identity.

The registered confound: **ord23 is not acquisition-matched** (170 vs λ=0.25's 197). Tolerable
here only because the coupling analysis and DIRCTL-001 both found damage indifferent to
acquisition; it would not be tolerable in an experiment that had not established that first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from paired import paired  # noqa: E402

CONDITIONS = ["low", "equal", "high"]


def norm(v) -> str:
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ord23-norms", type=Path, required=True)
    ap.add_argument("--twin-norms", type=Path, required=True)
    ap.add_argument("--dirctl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            (a.root / "ALIAS_PROBE.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ref = {r["id"]: r["reference_normalised"] for r in rows}
    cond = {r["id"]: r["condition"] for r in rows}
    n = len(ref)

    correct = {}
    for lab, d in [("ord23", "gen-ord23"), ("twin", "gen-d2twin"), ("l0", "gen-lwf_l0")]:
        f = a.root / d / "predictions.jsonl"
        if not f.exists():
            raise SystemExit(f"missing predictions for {lab}: {f}")
        p = {r["id"]: r.get("prediction", "") for r in
             (json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip())}
        if [i for i in ref if i not in p]:
            raise SystemExit(f"arm {lab} does not cover all {n} rows")
        correct[lab] = {i: norm(p[i]) == ref[i] for i in ref}

    on = json.loads(a.ord23_norms.read_text(encoding="utf-8"))
    tn = json.loads(a.twin_norms.read_text(encoding="utf-8"))
    fro_o = on["norms"]["ord23"]["fro"]
    fro_t = tn["norms"]["twin"]["fro"]
    cos_o = on["pairs"]["l0:ord23"]["cosine"]
    err = abs(fro_o - fro_t)
    if err > 1e-4:
        raise SystemExit(f"norm match failed: ord23 {fro_o:.6f} vs twin {fro_t:.6f} "
                         f"(err {err:.2e}); refusing to score")

    acc = {k: sum(v.values()) / n for k, v in correct.items()}
    pr = paired("twin", "ord23", correct["twin"], correct["ord23"])
    p = pr.as_dict()["mcnemar_exact_two_sided_p"]
    resid = (acc["ord23"] - acc["twin"]) * 100

    by_c = {}
    for c in CONDITIONS:
        ids = [i for i in ref if cond[i] == c]
        q = paired("twin", "ord23", {i: correct["twin"][i] for i in ids},
                   {i: correct["ord23"][i] for i in ids})
        by_c[c] = {"n": len(ids), "twin": q.baseline_correct, "ord23": q.candidate_correct,
                   "ord23_gains": q.repair, "ord23_losses": q.damage,
                   "p": q.as_dict()["mcnemar_exact_two_sided_p"],
                   "absolute_readable": c != "high"}

    d1 = json.loads(a.dirctl.read_text(encoding="utf-8"))["primary"]
    comparison = [
        {"arm": "ord23", "mechanism": "row order (no anchor)", "angle_deg": 53.6,
         "norm_ratio_to_l0": round(fro_o / 0.832289, 4), "residual_pp": round(resid, 2),
         "p": p, "acquisition": 170},
        {"arm": "lambda=0.25", "mechanism": "anchor loss", "angle_deg": 58.4,
         "norm_ratio_to_l0": 0.8924, "residual_pp": d1["l025"]["measured_residual_pp"],
         "p": d1["l025"]["p_holm"], "acquisition": 197},
        {"arm": "lambda=0.5", "mechanism": "anchor loss", "angle_deg": 69.6,
         "norm_ratio_to_l0": 0.8171, "residual_pp": d1["l05"]["measured_residual_pp"],
         "p": d1["l05"]["p_holm"], "acquisition": 125},
    ]

    band = 2.5
    if resid > band and p <= 0.05:
        verdict = ("GEOMETRY -- a non-anchor rotation also spares alias; 'direction beats dose' "
                   "survives with a second mechanism behind it")
    elif abs(resid) <= band and p > 0.05:
        verdict = ("ANCHOR_SPECIFIC -- Amendment 4 NARROWS in the record to 'an anchor loss buys "
                   "alias preservation that its magnitude reduction does not explain'")
    elif resid < -band and p <= 0.05:
        verdict = "REVERSED -- a rotation that COSTS alias; no account predicts this"
    else:
        verdict = "INCONCLUSIVE -- reported as such, not forced"

    out = {
        "record": "DIRCTL_002_RESULT",
        "class": "MECHANISM SCOUT -- not a confirmation; no read budget was spent",
        "prereg": "probe/LWF_ALIAS_PREREG.md Addendum 3",
        "arm_provenance": "arm-lambda0-ord23 was built by the peer session, not by this one",
        "geometry": {"ord23_fro": fro_o, "twin_fro": fro_t, "norm_abs_err": err,
                     "cos_ord23_to_l0": cos_o,
                     "twin_is_extrapolation_past_parent": True,
                     "ray_shape_from_DIRCTL001": "APPROXIMATELY_LINEAR"},
        "accuracies": {k: round(v, 4) for k, v in acc.items()},
        "primary": {"ord23_acc": round(acc["ord23"], 4), "twin_acc": round(acc["twin"], 4),
                    "residual_pp": round(resid, 2),
                    "ord23_gains": pr.repair, "ord23_losses": pr.damage,
                    "p_mcnemar_exact_two_sided": p,
                    "multiplicity": "none -- one comparison, Holm over a family of one is identity",
                    "by_condition": by_c},
        "mechanism_comparison": comparison,
        "verdict": verdict,
        "caveats": [
            "ord23 is NOT acquisition-matched (170 vs 197); tolerable only because damage was "
            "already established indifferent to acquisition",
            "the twin is an EXTRAPOLATION 2% past the parent, not an interpolant",
            "no angle->residual ordering is reported: the peer's ord23 acquisition result and "
            "ROTATION-001's projection identity both argue against one",
            "absolute rates in `high` are not readable by construction; paired contrasts are",
        ],
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"geometry: ord23 ||d|| {fro_o:.6f}   twin ||d|| {fro_t:.6f}   err {err:.2e}")
    print(f"          cos(ord23, l0) = {cos_o:.6f}  ({53.6:.1f} deg off the ray)")
    print(f"\nPRIMARY -- ord23 vs its norm-matched collinear twin")
    print(f"  ord23 {acc['ord23']:.4f}   twin {acc['twin']:.4f}   residual {resid:+.1f}pp")
    print(f"  ord23 gains {pr.repair} / loses {pr.damage}   p = {p:.4e}")
    print(f"\n{'cond':<7}{'n':>5}{'twin':>7}{'ord23':>7}{'gain':>6}{'loss':>6}{'p':>9}  readable")
    for c in CONDITIONS:
        v = by_c[c]
        print(f"{c:<7}{v['n']:>5}{v['twin']:>7}{v['ord23']:>7}{v['ord23_gains']:>6}"
              f"{v['ord23_losses']:>6}{v['p']:>9.4f}  {'yes' if v['absolute_readable'] else 'NO'}")
    print(f"\nMECHANISM COMPARISON (no ordering claimed)")
    print(f"{'arm':<13}{'mechanism':<24}{'deg':>6}{'norm':>7}{'resid':>8}{'acq':>6}")
    for c in comparison:
        print(f"{c['arm']:<13}{c['mechanism']:<24}{c['angle_deg']:>6.1f}"
              f"{c['norm_ratio_to_l0']:>7.3f}{c['residual_pp']:>7.1f}p{c['acquisition']:>6}")
    print(f"\nVERDICT: {verdict}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
