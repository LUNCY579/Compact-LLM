"""Score the LWF anchor arms on ALIAS-001 against a WITHIN-LANE magnitude calibration.

    python probe/score_lwf_alias.py --root <records>/alias001-20260918 \
        --norms <records>/lwf001-20260918/DELTA_NORMS.json \
        --out <records>/lwf001-20260918/LWF_ALIAS_RESULT.json

**Written before any prediction file was read**, as `probe/LWF_ALIAS_PREREG.md` registers. The
generations were launched first because generation is answer-neutral; scoring is not.

MECHANISM SCOUT. No seal, no read budget. Joint experiment with the peer session: the arms are
theirs, the binding instrument is ours, and Gate v2 has no binding endpoint.

## The confound this exists to remove

An anchor loss reduces ‖δ‖ as a side effect. ALIAS-002 established that alias accuracy falls
monotonically with ‖δ‖ (ρ = −1.000 in every salience condition), so **"the anchored arm loses
less alias" is exactly what a pure magnitude account predicts with no protection of anything.**

## The calibration, and why it is unusually clean here

`ctrl-own-alpha0.5` is `arm-lambda0`'s own finished delta multiplied by 0.5. So the three arms

    Supa  ->  ctrl-own-alpha0.5  ->  arm-lambda0

lie on **one ray**: the same direction at doses 0, 0.5 and 1.0 of the LWF lane's own update.
That is a three-point within-lane dose ladder, calibrated inside the lane rather than against
RRP-015's direction, and being three points it can show curvature rather than assuming a line.

The anchored arms are then placed on that ladder **by their measured ‖δ‖**:

    residual(arm) = alias_observed(arm) − alias_interpolated_at(‖δ(arm)‖)

A positive residual is alias accuracy the anchor bought that its magnitude reduction does not
explain. A residual at zero means the anchor is a norm knob.

## Verdict rule, fixed here and not after the numbers

  * **ANCHOR_PROTECTS** — both anchored residuals positive, increasing in λ, and the larger one
    significant against its norm-nearest ladder arm by exact paired McNemar.
  * **MAGNITUDE_ONLY** — residuals inside the noise band and not significant.
  * **ANCHOR_COSTS** — residuals negative. No account predicts this; it would be reported as
    unexplained rather than absorbed.
  * **NEITHER_CLEANLY** — anything else, reported as such.

`own-α0.5` vs `λ=0.5` is reported separately and is the directly actionable comparison: the peer
has already shown own-α wins acquisition 208 vs 125 and retention 40 vs 39.

## Readability

Paired contrasts are readable in every salience condition — a deterministic answer-agnostic rule
emits the same string for both arms and cannot create a discordant pair. **Absolute rates in
`high` are not readable**: the answer is the most frequent person there by construction and
`most_frequent_candidate_in_context` scores 1.000. Damage and repair are never netted, and every
subgroup rate is scored over the rows the BASELINE got right.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from paired import paired  # noqa: E402

CONDITIONS = ["low", "equal", "high"]
#: label -> (generation directory, lane dose if it lies on the calibration ray else None)
ARMS = [
    ("supa",    "gen-B",           0.0),
    ("owna05",  "gen-lwf_owna05",  0.5),
    ("l0",      "gen-lwf_l0",      1.0),
    ("l025",    "gen-lwf_l025",    None),
    ("l05",     "gen-lwf_l05",     None),
]
LAMBDA = {"l0": 0.0, "l025": 0.25, "l05": 0.5}


def norm(v) -> str:
    """Identical to probe/score_alias_probe.py::norm -- scoring convention must not drift."""
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def interp(x: float, pts: list[tuple[float, float]]) -> float:
    """Piecewise-linear on a sorted (norm, accuracy) ladder; clamped outside its range."""
    pts = sorted(pts)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--norms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            (a.root / "ALIAS_PROBE.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ref = {r["id"]: r["reference_normalised"] for r in rows}
    cond = {r["id"]: r["condition"] for r in rows}
    n = len(ref)

    correct: dict[str, dict[str, bool]] = {}
    for label, d, _ in ARMS:
        f = a.root / d / "predictions.jsonl"
        if not f.exists():
            raise SystemExit(f"missing predictions for {label}: {f}")
        p = {r["id"]: r.get("prediction", "") for r in
             (json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip())}
        if [i for i in ref if i not in p]:
            raise SystemExit(f"arm {label} does not cover all {n} rows")
        correct[label] = {i: norm(p[i]) == ref[i] for i in ref}

    nz = json.loads(a.norms.read_text(encoding="utf-8"))
    fro = {k: v["fro"] for k, v in nz["norms"].items()}
    fro["supa"] = 0.0
    if nz.get("self_check") and not str(nz["self_check"]["verdict"]).startswith("PASS"):
        raise SystemExit("DELTA_NORMS self-check FAILED; refusing to score against it")

    acc = {lab: sum(correct[lab].values()) / n for lab, _, _ in ARMS}

    per_arm = {}
    for lab, _, dose in ARMS:
        by_c = {}
        for c in CONDITIONS:
            ids = [i for i in ref if cond[i] == c]
            pr = paired("supa", lab, {i: correct["supa"][i] for i in ids},
                        {i: correct[lab][i] for i in ids})
            by_c[c] = {"n": len(ids), "correct": pr.candidate_correct,
                       "acc": round(pr.candidate_correct / len(ids), 4),
                       "repair_vs_supa": pr.repair, "damage_vs_supa": pr.damage,
                       "damage_rate_given_supa_correct": (
                           round(pr.damage / pr.baseline_correct, 4)
                           if pr.baseline_correct else None),
                       "absolute_rate_readable": c != "high"}
        pr = paired("supa", lab, correct["supa"], correct[lab])
        per_arm[lab] = {
            "gen_dir": dict((x, y) for x, y, _ in [(l, d, z) for l, d, z in ARMS])[lab],
            "lane_dose_on_calibration_ray": dose,
            "lambda": LAMBDA.get(lab),
            "delta_fro": fro.get(lab),
            "n": n, "correct": pr.candidate_correct, "acc": round(acc[lab], 4),
            "repair_vs_supa": pr.repair, "damage_vs_supa": pr.damage,
            "p_vs_supa": pr.as_dict()["mcnemar_exact_two_sided_p"],
            "by_condition": by_c,
        }

    # ---- the within-lane calibration ------------------------------------------------------
    ladder = sorted((fro[lab], acc[lab]) for lab, _, dose in ARMS if dose is not None)
    residuals = {}
    for lab in ("l025", "l05"):
        pred = interp(fro[lab], ladder)
        # significance against the norm-nearest calibration arm, which is the fairest single
        # paired comparison available -- an interpolation has no paired partner of its own
        nearest = min((lab2 for lab2, _, dose in ARMS if dose is not None),
                      key=lambda L: abs(fro[L] - fro[lab]))
        pr = paired(nearest, lab, correct[nearest], correct[lab])
        residuals[lab] = {
            "delta_fro": fro[lab], "observed": round(acc[lab], 4),
            "predicted_from_norm": round(pred, 4),
            "residual_pp": round((acc[lab] - pred) * 100, 2),
            "norm_nearest_calibration_arm": nearest,
            "nearest_delta_fro": fro[nearest],
            "repair_vs_nearest": pr.repair, "damage_vs_nearest": pr.damage,
            "p_vs_nearest": pr.as_dict()["mcnemar_exact_two_sided_p"],
        }

    r25, r50 = residuals["l025"]["residual_pp"], residuals["l05"]["residual_pp"]
    band = 2.5  # pp; ~1 SE of a proportion at n=360 near 0.7, registered as the noise band
    sig = residuals["l05"]["p_vs_nearest"] <= 0.05
    if r25 > band and r50 > band and r50 >= r25 and sig:
        verdict = "ANCHOR_PROTECTS"
    elif abs(r25) <= band and abs(r50) <= band and not sig:
        verdict = "MAGNITUDE_ONLY"
    elif r25 < -band and r50 < -band:
        verdict = "ANCHOR_COSTS"
    else:
        verdict = "NEITHER_CLEANLY"

    # own-alpha vs the anchor at comparable intent: the directly actionable comparison
    pr_own = paired("l05", "owna05", correct["l05"], correct["owna05"])
    head_to_head = {
        "comparison": "ctrl-own-alpha0.5 vs arm-lambda0.5",
        "own_alpha_acc": round(acc["owna05"], 4), "lambda05_acc": round(acc["l05"], 4),
        "own_alpha_delta_fro": fro["owna05"], "lambda05_delta_fro": fro["l05"],
        "own_alpha_gains": pr_own.repair, "own_alpha_losses": pr_own.damage,
        "p": pr_own.as_dict()["mcnemar_exact_two_sided_p"],
        "peer_reported_elsewhere": {"acquisition": {"own_alpha": 208, "lambda05": 125},
                                    "retained_47": {"own_alpha": 40, "lambda05": 39}},
    }

    out = {
        "record": "LWF_ALIAS_RESULT",
        "class": "MECHANISM SCOUT -- not a confirmation; no read budget was spent",
        "prereg": "probe/LWF_ALIAS_PREREG.md",
        "joint_with": "peer session (LWF-001 arms)",
        "standing_prediction_under_test": "the anchor helps most on skills absent from the "
                                          "training mixture",
        "calibration_ray": ("Supa -> ctrl-own-alpha0.5 -> arm-lambda0 are the SAME direction at "
                            "doses 0, 0.5 and 1.0 of the LWF lane's own update"),
        "calibration_points_norm_acc": ladder,
        "per_arm": per_arm,
        "residuals": residuals,
        "noise_band_pp": band,
        "head_to_head_own_alpha_vs_anchor": head_to_head,
        "verdict": verdict,
        "verdict_rule": ("ANCHOR_PROTECTS needs both residuals above the band, increasing in "
                         "lambda, and significance against the norm-nearest calibration arm; "
                         "MAGNITUDE_ONLY needs both inside the band and not significant; "
                         "negatives are ANCHOR_COSTS; anything else is reported as NEITHER"),
        "caveats": [
            "LWF-001 is already FALSIFIED on its own axes; a protective effect here would be "
            "SELECTIVE, surviving an otherwise-dominated intervention",
            "absolute rates in the `high` condition are not readable -- the answer is the most "
            "frequent person there by construction and the extraction floor is 1.000",
            "paired contrasts ARE readable in every condition: a deterministic rule emits the "
            "same string for both arms and cannot create a discordant pair",
            "the calibration is within-lane, so it does not confound direction with magnitude, "
            "but three points is still a dose ladder of three points",
            "one training draw per arm, one seed; a mechanism scout promotes nothing",
        ],
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'arm':<9}{'lam':>6}{'||d||':>9}{'acc':>8}{'rep':>5}{'dam':>5}{'p vs supa':>11}"
          f"{'low':>8}{'equal':>8}{'high':>8}")
    for lab, _, _ in ARMS:
        v = per_arm[lab]
        lam = "-" if v["lambda"] is None else f"{v['lambda']:.2f}"
        fr = "-" if v["delta_fro"] is None else f"{v['delta_fro']:.4f}"
        print(f"{lab:<9}{lam:>6}{fr:>9}{v['acc']:>8.3f}{v['repair_vs_supa']:>5}"
              f"{v['damage_vs_supa']:>5}{v['p_vs_supa']:>11.4f}"
              + "".join(f"{v['by_condition'][c]['acc']:>8.3f}" for c in CONDITIONS))

    print(f"\ncalibration ray (||delta||, acc): "
          + "  ".join(f"({x:.4f}, {y:.3f})" for x, y in ladder))
    print(f"\n{'arm':<7}{'||d||':>9}{'obs':>8}{'pred':>8}{'resid':>8}{'nearest':>10}"
          f"{'gain':>6}{'loss':>6}{'p':>9}")
    for lab in ("l025", "l05"):
        r = residuals[lab]
        print(f"{lab:<7}{r['delta_fro']:>9.4f}{r['observed']:>8.3f}"
              f"{r['predicted_from_norm']:>8.3f}{r['residual_pp']:>7.1f}p"
              f"{r['norm_nearest_calibration_arm']:>10}{r['repair_vs_nearest']:>6}"
              f"{r['damage_vs_nearest']:>6}{r['p_vs_nearest']:>9.4f}")

    h = head_to_head
    print(f"\nown-alpha0.5 vs lambda0.5 on alias: {h['own_alpha_acc']:.3f} vs "
          f"{h['lambda05_acc']:.3f}, own-alpha gains {h['own_alpha_gains']} / "
          f"loses {h['own_alpha_losses']}, p = {h['p']:.4f}")
    print(f"  (peer reports elsewhere: acquisition 208 vs 125, retained 40 vs 39)")
    print(f"\nVERDICT: {verdict}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
