"""Analyze matched-vs-within-class-mismatched reply NLL margins."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def ci(v, rng, n=10000):
    v = np.asarray(v, float); idx = rng.integers(0, len(v), size=(n, len(v)))
    x = v[idx].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.quantile(x, .025)), float(np.quantile(x, .975))]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--matched", type=Path, required=True); ap.add_argument("--mismatch", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    m = json.loads(args.matched.read_text()); x = json.loads(args.mismatch.read_text())
    M = {a["arm"]: {r["id"]: r for r in a["rows_detail"]} for a in m["results"]}
    X = {a["arm"]: {r["id"]: r for r in a["rows"]} for a in x["results"]}
    rng = np.random.default_rng(20260923); out = {"schema": 1, "instrument": x["instrument"], "matched_sha256": hashlib.sha256(args.matched.read_bytes()).hexdigest(), "mismatch_sha256": hashlib.sha256(args.mismatch.read_bytes()).hexdigest(), "bootstrap": {"n": 10000, "seed": 20260923}, "arms": {}}
    for arm in sorted(X):
        ids = sorted(X[arm]); margin = np.array([X[arm][i]["mean_nll"] - M[arm][i]["mean_nll"] for i in ids], float)
        out["arms"][arm] = {"matched_mean_nll": float(np.mean([M[arm][i]["mean_nll"] for i in ids])), "mismatch_mean_nll": float(np.mean([X[arm][i]["mean_nll"] for i in ids])), "mismatch_minus_matched": ci(margin, rng), "rows": len(ids)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["arms"], indent=2))


if __name__ == "__main__": main()
