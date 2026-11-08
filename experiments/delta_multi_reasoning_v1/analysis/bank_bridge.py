"""BRIDGE-001 — calibrate the dose bank against select-views, so D2/D4 can be tested.

    python analysis/bank_bridge.py --evidence <...>/retention-design-v1 \
        --out <records>/bridge001-20260919/BANK_BRIDGE.json

**Written and committed before the bridge runs finished**, per `probe/BRIDGE001_PREREG.md`.
An offset computed after seeing the arms it is meant to license is not a calibration.

## The problem

Two acquisition tables, two banks, nothing but the hash to tell them apart:

  * `7dd53f2a751b54c3…` — select-views-001. Every alpha/k/r/lambda/rotation/LoRA arm. 47 labels.
  * `45674251e38e5742…` — process-transition-probe-001. The RRP-008 dose ladder,
    D1 239 -> D2 276 -> D3 289 -> D4 304. 28 labels.

Same 384 views, same 3 x 128 family split, **same implementation digest**. A LoRA-at-D2 arm
scored on one has no comparator on the other.

## What this computes, and the two rules that constrain it

A bridge point is **one model scored on both banks**. The banks hold DIFFERENT items, so this is
an **unpaired two-proportion** comparison — McNemar is inapplicable and using it would overstate
precision by treating independent items as paired.

  1. **Per-arm offsets are primary. Pooling requires a heterogeneity check first.** Pooling a
     strength-dependent offset into one number is the pooled-floor failure in a new coordinate:
     a single figure that hides a split. `heterogeneity()` runs Cochran's Q over the per-arm
     differences and the caller must consult it before quoting any pooled number.
  2. **The offset is licensed only over the range it was measured on.** D4 scores 304 on the
     dose bank; the strongest bridge arm available is ~238 on the select-views scale.
     `licensed_range()` reports the interval, and `extrapolation_guard()` refuses to correct any
     score outside it. **This is registered in the prereg precisely so it cannot be waved through
     later** — applying a quantity measured over one range to a range it was never measured on is
     the cue-differential error, which this programme has already had to withdraw once.

## Byte identity

A bridge point requires *the same model*, not the same construction. Two traps, both hit:

  * `arm-rebuild-s10` is not `arm-s10` -- `out_weight_sha256` differs and the retained-norm
    differs at ~1.2e-9. Different bytes are a different model.
  * **`model_dir` is not the model.** `process_probe.py` supports `--checkpoint` overlays, so the
    single base directory `.../Supa` hosts **13 distinct arms** differing only by the checkpoint
    applied on top. A first version of this module matched on `model_dir` alone, silently paired
    `rrp015_control` (select-views 234) with `supa_rrp007_a1` (dose 289), and produced a **+55**
    bridge point that drove the entire heterogeneity verdict. Pairs are therefore matched on
    `(model_dir, checkpoint.manifest_sha256)` and the module reports the field it matched on.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from math import sqrt

DOSE = "45674251e38e574229f4ea98d04c30fe623ec6ac58ad308ad491661c84afab7e"
SELECT = "7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3"
Z = 1.96


def collect(roots: list[str]) -> dict:
    """Every PROCESS artifact, keyed by (model_dir, bank). Later files never clobber earlier."""
    out: dict = {}
    for root in roots:
        for f in glob.glob(os.path.join(root, "**", "PROCESS_*.json"), recursive=True):
            try:
                d = json.load(open(f, encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(d, dict):
                continue
            bank, md = d.get("views_sha256"), d.get("model_dir")
            agg = d.get("aggregate") or {}
            if bank not in (DOSE, SELECT) or not md or agg.get("canonical_correct") is None:
                continue
            # model_dir is NOT the model. process_probe.py supports --checkpoint overlays, so a
            # single base directory (".../Supa") hosts 13 distinct arms that differ only by the
            # checkpoint applied on top of it. Matching on model_dir alone silently paired
            # rrp015_control (sv 234) with supa_rrp007_a1 (dose 289) and manufactured a +55
            # bridge point that drove the entire heterogeneity verdict.
            ck = d.get("checkpoint") or {}
            ident = ck.get("manifest_sha256") or "NO_CHECKPOINT"
            key = ((os.path.normcase(os.path.normpath(md)), ident), bank)
            out.setdefault(key, {"label": d.get("label"), "n": agg.get("n"),
                                 "correct": agg.get("canonical_correct"), "file": os.path.basename(f)})
    return out


def pairs(found: dict) -> list[dict]:
    """Models present on BOTH banks. Matching is on model_dir -- byte identity, not construction."""
    models = {m for m, _ in found}
    out = []
    for m in sorted(models):
        a, b = found.get((m, DOSE)), found.get((m, SELECT))
        if not (a and b):
            continue
        n1, n2 = a["n"], b["n"]
        p1, p2 = a["correct"] / n1, b["correct"] / n2
        se = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        d = p1 - p2
        out.append({
            "model_dir": m[0], "checkpoint_manifest": m[1],
            "matched_on": "model_dir + checkpoint.manifest_sha256",
            "dose_label": a["label"], "select_label": b["label"],
            "dose": a["correct"], "select": b["correct"], "n": n1,
            "offset_rows": a["correct"] - b["correct"],
            "offset_ci_rows": [round((d - Z * se) * n1, 1), round((d + Z * se) * n1, 1)],
            "se_rows": round(se * n1, 2),
            "z": round(d / se, 3) if se else None,
        })
    return sorted(out, key=lambda r: r["select"])


def heterogeneity(ps: list[dict]) -> dict:
    """Cochran's Q over per-arm offsets. Consult BEFORE quoting any pooled number."""
    usable = [p for p in ps if p["se_rows"] > 0]
    if len(usable) < 2:
        return {"applicable": False, "why": "fewer than two usable points"}
    w = [1.0 / p["se_rows"] ** 2 for p in usable]
    y = [p["offset_rows"] for p in usable]
    mean = sum(wi * yi for wi, yi in zip(w, y)) / sum(w)
    Q = sum(wi * (yi - mean) ** 2 for wi, yi in zip(w, y))
    df = len(usable) - 1
    # I^2: share of variance beyond sampling error.
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    return {"applicable": True, "Q": round(Q, 3), "df": df, "I2": round(I2, 3),
            "inverse_variance_mean_rows": round(mean, 2),
            "pooling_permitted": bool(Q <= df * 2 and I2 < 0.5),
            "why": ("pooling is permitted only when the per-arm offsets are consistent. If Q "
                    "greatly exceeds its df, or I2 >= 0.5, the offset DEPENDS ON ARM STRENGTH "
                    "and the bridge is a curve, not a number -- quoting one figure would hide "
                    "a split, which is the pooled-floor failure in a new coordinate.")}


def licensed_range(ps: list[dict]) -> dict:
    if not ps:
        return {"min_select": None, "max_select": None}
    lo = min(p["select"] for p in ps)
    hi = max(p["select"] for p in ps)
    return {"min_select": lo, "max_select": hi,
            "why": "the offset is measured only over this span of select-views scores"}


def extrapolation_guard(score: int, rng: dict) -> dict:
    """Refuse to correct a score outside the measured range. Registered in the prereg."""
    lo, hi = rng.get("min_select"), rng.get("max_select")
    if lo is None:
        return {"permitted": False, "why": "no bridge points"}
    ok = lo <= score <= hi
    return {"permitted": ok, "score": score, "range": [lo, hi],
            "why": ("inside the measured range" if ok else
                    f"REFUSED: {score} lies outside the measured span [{lo}, {hi}]. Applying an "
                    "offset measured over one range to a range it was never measured on is the "
                    "cue-differential error; this refusal was registered before the numbers "
                    "existed.")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-score", type=int, action="append", default=[],
                    help="a dose-bank score to test against the extrapolation guard, e.g. 276, 304")
    a = ap.parse_args(argv)

    found = collect(a.evidence)
    ps = pairs(found)
    if not ps:
        print("REFUSED: no model appears on both banks. Nothing written.")
        return 2

    het = heterogeneity(ps)
    rng = licensed_range(ps)
    result = {
        "record": "BANK_BRIDGE_001",
        "class": "MECHANISM SCOUT / instrument calibration. REPORT-ONLY -- the dose bank declares "
                 "CANDIDATE_NOT_ADMITTED_OR_TRAINED and 'never for selection'. Nothing here may "
                 "select, promote or rank an artifact.",
        "banks": {"dose": DOSE, "select_views": SELECT},
        "comparison_kind": "unpaired two-proportion (the banks hold DIFFERENT items; McNemar is "
                           "inapplicable and would overstate precision)",
        "points": ps, "heterogeneity": het, "licensed_range": rng,
        "extrapolation_checks": [extrapolation_guard(s, rng) for s in a.check_score],
    }

    print("BRIDGE-001 — dose bank vs select-views\n")
    print(f"  {'select':>7}{'dose':>7}{'offset':>8}{'95% CI rows':>20}   model")
    for p in ps:
        ci = f"[{p['offset_ci_rows'][0]:+.1f}, {p['offset_ci_rows'][1]:+.1f}]"
        print(f"  {p['select']:>7}{p['dose']:>7}{p['offset_rows']:>+8}{ci:>20}   "
              f"{os.path.basename(p['model_dir'])} [{p['checkpoint_manifest'][:8]}]")
    if het.get("applicable"):
        print(f"\n  heterogeneity: Q={het['Q']} df={het['df']} I2={het['I2']}  "
              f"pooling_permitted={het['pooling_permitted']}")
        if het["pooling_permitted"]:
            print(f"  inverse-variance mean offset: {het['inverse_variance_mean_rows']:+.2f} rows")
        else:
            print("  POOLING REFUSED — the offset depends on arm strength; the bridge is a curve.")
    print(f"\n  licensed range (select-views scale): "
          f"[{rng['min_select']}, {rng['max_select']}]")
    for c in result["extrapolation_checks"]:
        print(f"    check {c['score']}: {'OK' if c['permitted'] else 'REFUSED'} — {c['why'][:110]}")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
