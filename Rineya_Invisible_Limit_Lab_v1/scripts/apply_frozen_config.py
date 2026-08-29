#!/usr/bin/env python3
"""Fit the frozen configuration on DEV and emit gated predictions for a target split.

The frozen Stage 8 candidate, selected on DEV before TEST was scored:

    Rineya-Supa-1.7.0-devfreeze  +  linear hidden-state abstention probe
    one forward pass, compute-eligible (budget is 6)

Two DEV findings drove that choice:

  * Routing buys nothing. Every eligible multi-arm policy tied or lost against
    the single probe-gated core, because the arms share a base and adapter and so
    agree almost everywhere. Oracle union across five arms is 223/256 while the
    best answer-blind policy reaches 175 -- the complementarity exists but is not
    reachable without the key.
  * The baked S4 logit bias is counterproductive once a probe is present. Probe
    on the ungated core scores 175/256; the same probe on Supa-A1, which carries
    bias 5.15625 in config, scores 170. The bias makes false abstentions the
    probe would not make. So the frozen config uses the UNGATED core.

Protocol: the probe is fit on all of DEV, then applied once to the target split.
Nothing about the target split informs the fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def features(sig: dict) -> np.ndarray:
    return np.concatenate([np.asarray(v, dtype=np.float32)
                           for _, v in sorted(sig["hidden"].items())])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-signals", type=Path, required=True)
    ap.add_argument("--dev-key", type=Path, required=True)
    ap.add_argument("--target-signals", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()

    dev_sig = {r["id"]: r for r in load_jsonl(args.dev_signals)}
    dev_key = {r["id"]: r for r in load_jsonl(args.dev_key)}
    ids = sorted(set(dev_sig) & set(dev_key))

    X = np.asarray([features(dev_sig[i]) for i in ids])
    y = np.asarray([0 if dev_key[i]["answerable"] else 1 for i in ids])
    groups = np.asarray([dev_key[i]["scenario_id"] for i in ids])

    # Threshold is selected by cross-validation WITHIN DEV, then the model is
    # refit on all of DEV. The target split never touches either step.
    thresholds = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))
        clf.fit(X[tr], y[tr])
        s = clf.predict_proba(X[tr])[:, 1]
        best_t, best = 0.5, -1
        for t in np.unique(np.round(s, 3)):
            f = s >= t
            ok = np.zeros(len(tr), dtype=bool)
            ok[f & (y[tr] == 1)] = True
            ok[~f & (y[tr] == 0)] = True
            if ok.sum() > best:
                best, best_t = ok.sum(), float(t)
        thresholds.append(best_t)
    threshold = float(np.median(thresholds))

    final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))
    final.fit(X, y)
    print(f"probe fit on {len(ids)} DEV rows; threshold={threshold:.4f} "
          f"(median of {[round(t,3) for t in thresholds]})")

    tgt = load_jsonl(args.target_signals)
    Xt = np.asarray([features(r) for r in tgt])
    scores = final.predict_proba(Xt)[:, 1]
    fires = scores >= threshold

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for r, s, fire in zip(tgt, scores, fires):
            f.write(json.dumps({
                "id": r["id"],
                "status": "ok",
                "prediction": "UNKNOWN" if fire else r["prediction"],
                "abstained": bool(fire),
                "probe_score": round(float(s), 6),
                "ungated_prediction": r["prediction"],
            }, ensure_ascii=False) + "\n")

    print(f"wrote {len(tgt)} rows -> {args.output}  (abstained on {int(fires.sum())})")

    if args.manifest:
        args.manifest.write_text(json.dumps({
            "config": "Rineya-Supa-1.7.0-devfreeze + linear hidden-state abstention probe",
            "forward_passes_per_item": 1,
            "compute_budget": 6,
            "compute_eligible": True,
            "probe": {
                "features": "final-position hidden state, layers -1/-8/-15 concatenated",
                "dim": int(X.shape[1]),
                "model": "StandardScaler + LogisticRegression(C=0.05, max_iter=3000)",
                "threshold": threshold,
                "fit_rows": len(ids),
                "fit_split": "DEV",
            },
            "target_signals": str(args.target_signals),
            "target_rows": len(tgt),
            "abstentions": int(fires.sum()),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
