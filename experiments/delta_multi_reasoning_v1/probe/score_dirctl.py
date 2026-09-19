"""DIRCTL-001 — does the direction residual survive when the interpolation is removed?

    python probe/score_dirctl.py --root <records>/alias001-20260918 \
        --norms <records>/lwf001-20260918/DELTA_NORMS.json \
        --dirctl-norms <records>/lwf001-20260918/DIRCTL_NORMS.json \
        --out <records>/lwf001-20260918/DIRCTL_RESULT.json

**Written before either arm finished generating**, as `probe/DIRCTL001_PREREG.md` registers.
MECHANISM SCOUT: no seal, no read budget.

Amendment 4 measured the anchored arms' residual against a piecewise-linear interpolation of a
three-point ray, and both anchored arms sat inside a wide unmeasured gap. This replaces the
interpolated value at each anchored arm's norm with a **real arm at exactly that norm on exactly
that direction**, so the primary becomes a direct paired contrast between two arms.

## The collinear ladder, now five points

All of these lie on **one ray** — `theta_supa + a*(theta_lambda0 - theta_supa)` — so the only
thing varying across them is magnitude:

    Supa 0.000  ->  own-a0.5 0.416  ->  sc-m05 0.680  ->  sc-m025 0.743  ->  lambda=0 0.832

The two anchored arms sit at 0.680 and 0.743, **off** this ray (cosine 0.349 and 0.524), and are
compared to the collinear arm at their own norm.

## What is tested

  * **Primary** — exact two-sided McNemar, anchored vs norm-matched collinear, for each pair,
    **Holm-corrected over the family of two**. Registered direction: anchored > collinear.
  * **Secondary 1** — curvature of the collinear ladder against its endpoint chord. This is the
    first direct read of the magnitude curve's shape in this lane and it says how wrong the old
    interpolation was, in percentage points, at each anchored arm's norm.
  * **Secondary 2** — Spearman over the five collinear points with an exact permutation p,
    testing whether ALIAS-002's monotone dose response replicates on an independently trained
    delta's ray.

## Readability

Paired contrasts are readable in every salience condition: a deterministic answer-agnostic rule
emits the same string for both arms and cannot create a discordant pair. **Absolute rates in
`high` are not readable** — the answer is the most frequent person there by construction.
Repair and damage are never netted.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from paired import paired  # noqa: E402

CONDITIONS = ["low", "equal", "high"]
#: label -> generation dir. The first five are COLLINEAR; the last two are the anchored arms.
COLLINEAR = [("supa", "gen-B"), ("owna05", "gen-lwf_owna05"), ("sc_m05", "gen-dirctl_m05"),
             ("sc_m025", "gen-dirctl_m025"), ("l0", "gen-lwf_l0")]
ANCHORED = [("l05", "gen-lwf_l05"), ("l025", "gen-lwf_l025")]
#: anchored arm -> the collinear arm built to match its norm exactly
MATCH = {"l025": "sc_m025", "l05": "sc_m05"}
#: what Amendment 4's three-point interpolation predicted, for the record
OLD_INTERP = {"l025": 0.576, "l05": 0.591}


def norm(v) -> str:
    """Identical to probe/score_alias_probe.py::norm -- scoring convention must not drift."""
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def spearman(x, y) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def perm_p(x, y) -> float:
    """Exact two-sided permutation p over all n! orderings. n = 5 gives a floor of 1/60."""
    obs = spearman(x, y)
    perms = list(itertools.permutations(y))
    hits = sum(1 for p in perms if abs(spearman(x, list(p))) >= abs(obs) - 1e-12)
    return hits / len(perms)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--norms", type=Path, required=True)
    ap.add_argument("--dirctl-norms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            (a.root / "ALIAS_PROBE.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ref = {r["id"]: r["reference_normalised"] for r in rows}
    cond = {r["id"]: r["condition"] for r in rows}
    n = len(ref)

    correct: dict[str, dict[str, bool]] = {}
    for lab, d in COLLINEAR + ANCHORED:
        f = a.root / d / "predictions.jsonl"
        if not f.exists():
            raise SystemExit(f"missing predictions for {lab}: {f}")
        p = {r["id"]: r.get("prediction", "") for r in
             (json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip())}
        if [i for i in ref if i not in p]:
            raise SystemExit(f"arm {lab} does not cover all {n} rows")
        correct[lab] = {i: norm(p[i]) == ref[i] for i in ref}

    nz = json.loads(a.norms.read_text(encoding="utf-8"))["norms"]
    dz = json.loads(a.dirctl_norms.read_text(encoding="utf-8"))
    fro = {"supa": 0.0, "owna05": nz["owna05"]["fro"], "l0": nz["l0"]["fro"],
           "l025": nz["l025"]["fro"], "l05": nz["l05"]["fro"],
           "sc_m025": dz["norms"]["sc_m025"]["fro"], "sc_m05": dz["norms"]["sc_m05"]["fro"]}

    # ---- the build assertion: did the scalars actually land on their targets? --------------
    match_err = {k: abs(fro[v] - fro[k]) for k, v in MATCH.items()}
    collinearity = {s: dz["pairs"][s]["cosine"] for s in dz.get("pairs", {})}
    if max(match_err.values()) > 1e-5:
        raise SystemExit(f"norm match failed: {match_err}; refusing to score")

    acc = {lab: sum(correct[lab].values()) / n for lab in correct}

    # ---- PRIMARY: anchored vs its norm-matched collinear twin ------------------------------
    primary, praw = {}, {}
    for anc, sc in MATCH.items():
        pr = paired(sc, anc, correct[sc], correct[anc])
        by_c = {}
        for c in CONDITIONS:
            ids = [i for i in ref if cond[i] == c]
            q = paired(sc, anc, {i: correct[sc][i] for i in ids},
                       {i: correct[anc][i] for i in ids})
            by_c[c] = {"n": len(ids), "collinear": q.baseline_correct,
                       "anchored": q.candidate_correct,
                       "anchored_gains": q.repair, "anchored_losses": q.damage,
                       "p": q.as_dict()["mcnemar_exact_two_sided_p"],
                       "absolute_readable": c != "high"}
        p = pr.as_dict()["mcnemar_exact_two_sided_p"]
        praw[anc] = p
        primary[anc] = {
            "anchored_arm": anc, "collinear_twin": sc,
            "norm_anchored": fro[anc], "norm_collinear": fro[sc],
            "norm_match_abs_err": match_err[anc],
            "anchored_acc": round(acc[anc], 4), "collinear_acc": round(acc[sc], 4),
            "measured_residual_pp": round((acc[anc] - acc[sc]) * 100, 2),
            "amendment4_interpolated_residual_pp": round(
                (acc[anc] - OLD_INTERP[anc]) * 100, 2),
            "old_interpolation_value": OLD_INTERP[anc],
            "interpolation_error_pp": round((acc[sc] - OLD_INTERP[anc]) * 100, 2),
            "anchored_gains": pr.repair, "anchored_losses": pr.damage,
            "p_mcnemar": p, "by_condition": by_c,
        }
    padj = holm(praw)
    for k, v in padj.items():
        primary[k]["p_holm"] = v

    # ---- SECONDARY 1: curvature of the collinear ladder ------------------------------------
    ladder = sorted(((fro[lab], acc[lab], lab) for lab, _ in COLLINEAR))
    x0, y0, _ = ladder[0]
    x1, y1, _ = ladder[-1]
    curve = []
    for x, y, lab in ladder:
        chord = y0 + (y1 - y0) * ((x - x0) / (x1 - x0)) if x1 != x0 else y0
        curve.append({"arm": lab, "norm": x, "acc": round(y, 4),
                      "chord": round(chord, 4), "dev_from_chord_pp": round((y - chord) * 100, 2)})
    interior = [c["dev_from_chord_pp"] for c in curve[1:-1]]
    shape = ("ABOVE_CHORD -- falls slower than linear early" if all(d > 0.5 for d in interior)
             else "BELOW_CHORD -- falls faster than linear early" if all(d < -0.5 for d in interior)
             else "APPROXIMATELY_LINEAR")

    # ---- SECONDARY 2: dose response on this ray --------------------------------------------
    xs = [c["norm"] for c in curve]
    ys = [c["acc"] for c in curve]
    rho = spearman(xs, ys)
    pp = perm_p(xs, ys)

    # ---- verdict ---------------------------------------------------------------------------
    band = 2.5
    res = {k: primary[k]["measured_residual_pp"] for k in primary}
    sig = {k: primary[k]["p_holm"] <= 0.05 for k in primary}
    if all(res[k] > band for k in res) and all(sig.values()):
        verdict = "DIRECTION_SURVIVES"
    elif all(abs(res[k]) <= band for k in res) and not any(sig.values()):
        verdict = "INTERPOLATION_ARTIFACT -- Amendment 4 headline RETRACTED"
    elif all(res[k] < -band for k in res):
        verdict = "REVERSED -- the anchored arms are WORSE at matched norm"
    else:
        verdict = "MIXED -- reported as such, not forced"

    out = {
        "record": "DIRCTL_001_RESULT",
        "class": "MECHANISM SCOUT -- not a confirmation; no read budget was spent",
        "prereg": "probe/DIRCTL001_PREREG.md",
        "accuracies": {k: round(v, 4) for k, v in acc.items()},
        "norms": fro,
        "collinearity_check": collinearity,
        "primary": primary,
        "holm_family_size": len(praw),
        "curvature": {"ladder": curve, "shape": shape,
                      "note": ("deviation from the chord between Supa and lambda=0; the two "
                               "interior points at 0.680 and 0.743 are what Amendment 4 had to "
                               "interpolate")},
        "dose_response_on_this_ray": {
            "n_points": len(xs), "spearman_rho": round(rho, 4),
            "p_exact_permutation": round(pp, 4),
            "replicates_ALIAS002": bool(rho < 0 and all(
                ys[i] >= ys[i + 1] for i in range(len(ys) - 1))),
        },
        "verdict": verdict,
        "verdict_rule": ("DIRECTION_SURVIVES needs both measured residuals above the band AND "
                         "both Holm-adjusted p <= 0.05; INTERPOLATION_ARTIFACT needs both inside "
                         "the band and neither significant; both negative is REVERSED; anything "
                         "else is MIXED and is reported rather than forced"),
        "caveats": [
            "the scalar arms are post-hoc edits of a trained delta while the anchored arms were "
            "trained under a different objective, so they differ in more than direction -- "
            "named in the prereg before the build",
            "this does NOT resolve LWF_ALIAS_PREREG Addendum 2: even a surviving residual "
            "cannot separate 'the anchor protects' from 'that direction damages less'",
            "absolute rates in `high` are not readable by construction; paired contrasts are",
            "one training draw, one seed, one bank; a mechanism scout promotes nothing",
        ],
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("COLLINEAR LADDER (one ray, magnitude only)")
    print(f"{'arm':<9}{'||d||':>9}{'acc':>8}{'chord':>8}{'dev':>8}")
    for c in curve:
        print(f"{c['arm']:<9}{c['norm']:>9.4f}{c['acc']:>8.3f}{c['chord']:>8.3f}"
              f"{c['dev_from_chord_pp']:>7.1f}p")
    print(f"  shape: {shape}")
    print(f"  dose response on this ray: rho = {rho:.4f}, exact permutation p = {pp:.4f}")

    print("\nPRIMARY -- anchored vs its NORM-MATCHED collinear twin")
    print(f"{'pair':<16}{'||d||':>8}{'anch':>8}{'colin':>8}{'resid':>8}"
          f"{'gain':>6}{'loss':>6}{'p':>10}{'p_holm':>10}")
    for k, v in primary.items():
        print(f"{k+' vs '+v['collinear_twin']:<16}{v['norm_anchored']:>8.4f}"
              f"{v['anchored_acc']:>8.3f}{v['collinear_acc']:>8.3f}"
              f"{v['measured_residual_pp']:>7.1f}p{v['anchored_gains']:>6}"
              f"{v['anchored_losses']:>6}{v['p_mcnemar']:>10.2e}{v['p_holm']:>10.2e}")
    print("\nHOW WRONG WAS THE INTERPOLATION?")
    for k, v in primary.items():
        print(f"  {k}: Amendment 4 assumed {v['old_interpolation_value']:.3f}, "
              f"measured {v['collinear_acc']:.3f} "
              f"({v['interpolation_error_pp']:+.1f}pp)  ->  residual "
              f"{v['amendment4_interpolated_residual_pp']:+.1f}pp becomes "
              f"{v['measured_residual_pp']:+.1f}pp")
    print(f"\nVERDICT: {verdict}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
