"""Analyze the full held-out Opallo reply-content NLL matrix."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def mean_ci(v, rng, n=10000):
    v = np.asarray(v, float)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    x = v[idx].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.quantile(x, .025)), float(np.quantile(x, .975))]}


def ratio_ci(num, den, rng, n=10000):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), size=(n, len(num)))
    x = num[idx].mean(1) / den[idx].mean(1)
    return {"point": float(num.mean() / den.mean()), "ci95": [float(np.quantile(x, .025)), float(np.quantile(x, .975))],
            "mean_num": float(num.mean()), "mean_den": float(den.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    records = {x["arm"]: {r["id"]: r for r in x["rows_detail"]} for x in raw["results"]}
    ids = sorted(records["BASE"])
    rng = np.random.default_rng(20260923)

    def vals(arm, key):
        return np.array([records[arm][i][key] for i in ids], float)

    base_nll, op_nll = vals("BASE", "mean_nll"), vals("OPALLO", "mean_nll")
    base_z, op_z = vals("BASE", "mean_logodds"), vals("OPALLO", "mean_logodds")
    out = {"schema": 1, "instrument": "OPALLO-HELDOUT-REPLY-CONTENT-NLL-001",
           "raw_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
           "bootstrap": {"n": 10000, "seed": 20260923}, "rows": len(ids), "arms": {}}
    for arm in records:
        nll = vals(arm, "mean_nll")
        z = vals(arm, "mean_logodds")
        nll_gain = base_nll - nll
        z_gain = z - base_z
        out["arms"][arm] = {
            "nll_per_token": mean_ci(nll, rng),
            "logodds_per_token": mean_ci(z, rng),
            "nll_gain_vs_base": mean_ci(nll_gain, rng),
            "logodds_gain_vs_base": mean_ci(z_gain, rng),
            "P_O_nll": ratio_ci(nll_gain, base_nll - op_nll, rng),
            "P_O_logodds": ratio_ci(z_gain, op_z - base_z, rng),
        }
        classes = sorted({records[arm][i]["reply_class"] for i in ids})
        out["arms"][arm]["by_class_nll"] = {}
        for c in classes:
            sel = np.array([records[arm][i]["reply_class"] == c for i in ids])
            out["arms"][arm]["by_class_nll"][c] = {"rows": int(sel.sum()),
                                                       "nll_per_token": float(nll[sel].mean()),
                                                       "logodds_per_token": float(z[sel].mean())}
    out["specificity"] = {
        "SPARSE_P_O_nll": out["arms"]["SPARSE"]["P_O_nll"],
        "ADD100_P_O_nll": out["arms"]["ADD-100"]["P_O_nll"],
        "SPARSE_P_O_logodds": out["arms"]["SPARSE"]["P_O_logodds"],
        "ADD100_P_O_logodds": out["arms"]["ADD-100"]["P_O_logodds"],
    }
    out["dose_ladder"] = {arm: {"nll_per_token": out["arms"][arm]["nll_per_token"],
                                 "logodds_per_token": out["arms"][arm]["logodds_per_token"]}
                          for arm in ("OPALLO", "ADD-O75", "ADD-100", "ADD-S75", "AVG", "ADD-75")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"specificity": out["specificity"], "dose_ladder": out["dose_ladder"]}, indent=2))


if __name__ == "__main__":
    main()
