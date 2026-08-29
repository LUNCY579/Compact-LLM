#!/usr/bin/env python3
"""Stages 1-3: fit and honestly evaluate abstention gates over captured signals.

All three stages are offline computations over one capture file, so they share
this tool:

  Stage 1  S1  single normalised signal, threshold swept
  Stage 2  S2  multi-signal logistic gate over decorrelated logit features
  Stage 3  P   probe on the final-position hidden state

The decisive question is Stage 3's: the published gate reads a *scalar* off the
logit layer and reaches AUC 0.79. The logit layer is a 151,936-way projection of
a 1024-dim state and is enormously lossy for this purpose. If a probe on the
hidden state clears 0.79 by a real margin the deficit is readout and can be
fixed by packaging; if it plateaus, the representation is the limit and training
is required. That is the fork the whole programme turns on.

Honesty machinery
-----------------
Every number here is cross-validated with **GroupKFold on scenario_id**. The four
variants of a scenario share a context, so a plain split would put near-identical
rows in train and test and inflate every estimate. Grouping by scenario is not
optional.

Thresholds are chosen inside each training fold, never on the fold being scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Logit-layer features. Chosen because the published diagnostics showed these
# families are only partially correlated: UNKNOWN-mass (AUC ~0.79), entropy
# (~0.72) and margin (~0.67) measure different things.
LOGIT_FEATURES = [
    "unknown_exact_minus_top1",
    "unknown_exact_logprob",
    "unknown_variant_minus_top1",
    "unknown_variant_logprob",
    "entropy",
    "logprob_margin",
    "top1_logprob",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def normalize(text: str) -> str:
    import re
    t = str(text).strip().lower()
    t = re.sub(r"^\s*(final\s+answer|answer)\s*:\s*", "", t)
    t = re.sub(r"[“”\"'`]", "", t)
    t = re.sub(r"[^a-z0-9\s-]", " ", t)
    t = re.sub(r"\b(the|a|an)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def build(signals: list[dict], key: list[dict]) -> dict[str, Any]:
    key_by_id = {r["id"]: r for r in key}
    ids, X_logit, X_hidden, y_unans, base_ok, groups = [], [], [], [], [], []
    for s in signals:
        k = key_by_id.get(s["id"])
        if k is None:
            continue
        ft = s["first_token"]
        ids.append(s["id"])
        X_logit.append([float(ft[f]) for f in LOGIT_FEATURES])
        hid = s.get("hidden") or {}
        X_hidden.append(np.concatenate([np.asarray(v, dtype=np.float32)
                                        for _, v in sorted(hid.items())]) if hid else np.zeros(1))
        y_unans.append(0 if k["answerable"] else 1)
        accepted = {normalize(a) for a in k["accepted_answers"]}
        base_ok.append(normalize(s["prediction"]) in accepted)
        groups.append(k["scenario_id"])
    return {
        "ids": ids,
        "X_logit": np.asarray(X_logit, dtype=np.float64),
        "X_hidden": np.asarray(X_hidden, dtype=np.float32),
        "y": np.asarray(y_unans, dtype=int),
        "base_ok": np.asarray(base_ok, dtype=bool),
        "groups": np.asarray(groups),
        "key_by_id": key_by_id,
    }


def score_with_gate(d: dict, fires: np.ndarray) -> dict[str, Any]:
    """Apply an abstain decision and recompute correctness.

    Firing rewrites the prediction to UNKNOWN: correct on an unanswerable row,
    wrong on an answerable one. This is the trade the invisible-limit hypothesis
    predicts, measured directly.
    """
    unans = d["y"] == 1
    ok = d["base_ok"].copy()
    ok[fires & unans] = True
    ok[fires & ~unans] = False
    gained = int((fires & unans & ~d["base_ok"]).sum())
    lost = int((fires & ~unans & d["base_ok"]).sum())
    tp = int((fires & unans).sum())
    fp = int((fires & ~unans).sum())
    return {
        "correct": int(ok.sum()),
        "n": len(ok),
        "gained": gained,
        "lost": lost,
        "abstentions": int(fires.sum()),
        "abstention_precision": (tp / (tp + fp)) if (tp + fp) else 1.0,
        "abstention_recall": tp / max(1, int(unans.sum())),
        "ok": ok,
    }


def sweep_threshold(scores_tr: np.ndarray, d_tr: dict) -> float:
    """Pick the threshold maximising corrected score inside a training fold."""
    best_t, best_c = float("inf"), -1
    cands = np.unique(np.round(scores_tr, 4))
    for t in cands:
        fires = scores_tr >= t
        unans = d_tr["y"] == 1
        ok = d_tr["base_ok"].copy()
        ok[fires & unans] = True
        ok[fires & ~unans] = False
        c = int(ok.sum())
        if c > best_c:
            best_c, best_t = c, float(t)
    return best_t


def cv_gate(d: dict, feature_key: str, model_factory, n_splits: int = 5) -> dict[str, Any]:
    """Cross-validated gate. Returns pooled out-of-fold predictions and score."""
    X = d[feature_key]
    y, groups = d["y"], d["groups"]
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y), dtype=float)
    fires = np.zeros(len(y), dtype=bool)
    for tr, te in gkf.split(X, y, groups):
        clf = model_factory()
        clf.fit(X[tr], y[tr])
        s_tr = clf.predict_proba(X[tr])[:, 1]
        s_te = clf.predict_proba(X[te])[:, 1]
        d_tr = {"y": y[tr], "base_ok": d["base_ok"][tr]}
        t = sweep_threshold(s_tr, d_tr)
        oof[te] = s_te
        fires[te] = s_te >= t
    return {"auc": float(roc_auc_score(y, oof)), "scores": oof, "fires": fires}


def cv_single_signal(d: dict, name: str, n_splits: int = 5) -> dict[str, Any]:
    """Cross-validated single-signal threshold gate (the S1/S4 family)."""
    idx = LOGIT_FEATURES.index(name)
    s = d["X_logit"][:, idx]
    y, groups = d["y"], d["groups"]
    gkf = GroupKFold(n_splits=n_splits)
    fires = np.zeros(len(y), dtype=bool)
    for tr, te in gkf.split(s.reshape(-1, 1), y, groups):
        t = sweep_threshold(s[tr], {"y": y[tr], "base_ok": d["base_ok"][tr]})
        fires[te] = s[te] >= t
    return {"auc": float(roc_auc_score(y, s)), "scores": s, "fires": fires}


def per_variant(d: dict, ok: np.ndarray) -> dict[str, str]:
    out: dict[str, list[int]] = {}
    for i, rid in enumerate(d["ids"]):
        v = d["key_by_id"][rid]["variant"]
        acc = out.setdefault(v, [0, 0])
        acc[0] += int(ok[i]); acc[1] += 1
    return {k: f"{v[0]}/{v[1]}" for k, v in sorted(out.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    d = build(load_jsonl(args.signals), load_jsonl(args.key))
    n = len(d["ids"])
    print(f"rows={n}  unanswerable={int(d['y'].sum())}  "
          f"scenarios={len(set(d['groups']))}  hidden_dim={d['X_hidden'].shape[1]}")
    print()

    results: dict[str, Any] = {}
    base_ok = d["base_ok"]
    print(f"{'configuration':38s} {'AUC':>6s} {'correct':>9s} {'gain':>5s} {'lost':>5s} {'prec':>6s}")
    print("-" * 78)
    print(f"{'ungated baseline':38s} {'-':>6s} {int(base_ok.sum()):>4d}/{n:<4d} {'-':>5s} {'-':>5s} {'-':>6s}")

    # --- Stage 1: single-signal gates (the published S1 / S4 family) ---
    for name in ("unknown_variant_logprob", "unknown_exact_minus_top1"):
        r = cv_single_signal(d, name, args.folds)
        s = score_with_gate(d, r["fires"])
        label = f"S1 {name}" if "variant" in name else f"S4 {name}"
        print(f"{label[:38]:38s} {r['auc']:>6.3f} {s['correct']:>4d}/{n:<4d} "
              f"{s['gained']:>5d} {s['lost']:>5d} {s['abstention_precision']:>6.3f}")
        results[label] = {"auc": r["auc"], **{k: v for k, v in s.items() if k != "ok"},
                          "per_variant": per_variant(d, s["ok"])}

    # --- Stage 2: multi-signal calibrated logistic gate ---
    r2 = cv_gate(d, "X_logit",
                 lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
                 args.folds)
    s2 = score_with_gate(d, r2["fires"])
    print(f"{'S2 multi-signal logistic (logits)':38s} {r2['auc']:>6.3f} {s2['correct']:>4d}/{n:<4d} "
          f"{s2['gained']:>5d} {s2['lost']:>5d} {s2['abstention_precision']:>6.3f}")
    results["S2_multisignal"] = {"auc": r2["auc"], **{k: v for k, v in s2.items() if k != "ok"},
                                 "per_variant": per_variant(d, s2["ok"])}

    # --- Stage 3: hidden-state probes (the decisive experiment) ---
    if d["X_hidden"].shape[1] > 1:
        r3 = cv_gate(d, "X_hidden",
                     lambda: make_pipeline(StandardScaler(),
                                           LogisticRegression(max_iter=3000, C=0.05)),
                     args.folds)
        s3 = score_with_gate(d, r3["fires"])
        print(f"{'P-linear probe (hidden state)':38s} {r3['auc']:>6.3f} {s3['correct']:>4d}/{n:<4d} "
              f"{s3['gained']:>5d} {s3['lost']:>5d} {s3['abstention_precision']:>6.3f}")
        results["P_linear"] = {"auc": r3["auc"], **{k: v for k, v in s3.items() if k != "ok"},
                               "per_variant": per_variant(d, s3["ok"])}

        r4 = cv_gate(d, "X_hidden",
                     lambda: make_pipeline(StandardScaler(),
                                           MLPClassifier(hidden_layer_sizes=(64,), max_iter=800,
                                                         alpha=1.0, random_state=17)),
                     args.folds)
        s4 = score_with_gate(d, r4["fires"])
        print(f"{'P-mlp probe (hidden state)':38s} {r4['auc']:>6.3f} {s4['correct']:>4d}/{n:<4d} "
              f"{s4['gained']:>5d} {s4['lost']:>5d} {s4['abstention_precision']:>6.3f}")
        results["P_mlp"] = {"auc": r4["auc"], **{k: v for k, v in s4.items() if k != "ok"},
                            "per_variant": per_variant(d, s4["ok"])}

        # Combined: hidden state + logit features
        Xc = np.hstack([d["X_hidden"], d["X_logit"].astype(np.float32)])
        d2 = dict(d); d2["X_comb"] = Xc
        r5 = cv_gate(d2, "X_comb",
                     lambda: make_pipeline(StandardScaler(),
                                           LogisticRegression(max_iter=3000, C=0.05)),
                     args.folds)
        s5 = score_with_gate(d, r5["fires"])
        print(f"{'P+S combined (hidden + logits)':38s} {r5['auc']:>6.3f} {s5['correct']:>4d}/{n:<4d} "
              f"{s5['gained']:>5d} {s5['lost']:>5d} {s5['abstention_precision']:>6.3f}")
        results["P_plus_S"] = {"auc": r5["auc"], **{k: v for k, v in s5.items() if k != "ok"},
                               "per_variant": per_variant(d, s5["ok"])}

    print()
    print("per-variant correct (the uniformity requirement):")
    print(f"  {'ungated':30s} {per_variant(d, base_ok)}")
    for k, v in results.items():
        print(f"  {k:30s} {v['per_variant']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signals": str(args.signals), "key": str(args.key),
            "rows": n, "unanswerable": int(d["y"].sum()),
            "folds": args.folds, "grouping": "GroupKFold on scenario_id",
            "ungated": {"correct": int(base_ok.sum()), "n": n,
                        "per_variant": per_variant(d, base_ok)},
            "gates": results,
        }
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
