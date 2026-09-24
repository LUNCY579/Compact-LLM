"""Analyze the full nine-arm teacher-forced matrix."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np


COEFF = {"BASE": (0, 0), "SPARSE": (1, 0), "OPALLO": (0, 1), "AVG": (.5, .5),
         "ADD-S75": (.75, 1), "ADD-O75": (1, .75), "ADD-75": (.75, .75),
         "ADD-100": (1, 1), "CONFLICT-LOCAL": (1, 1)}


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
    arms = {a["arm"]: {r["id"]: r for r in a["rows"]} for a in raw["results"]}
    ids = sorted(arms["BASE"])
    rng = np.random.default_rng(20260923)

    def vals(arm, part, readout):
        return np.array([arms[arm][i][part][readout] for i in ids], float)

    out = {"schema": 1, "instrument": raw["instrument"],
           "raw_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
           "bootstrap": {"n": 10000, "seed": 20260923}, "coefficients": COEFF,
           "readouts": {}}
    for readout in ("clipped_logodds", "unclipped_logprob"):
        out["readouts"][readout] = {"per_arm": {}, "interaction": {}, "P_S": {}}
        for arm in COEFF:
            out["readouts"][readout]["per_arm"][arm] = {
                "content": mean_ci(vals(arm, "content", readout), rng),
                "closing": mean_ci(vals(arm, "closing", readout), rng),
            }
        s = vals("SPARSE", "content", readout) - vals("BASE", "content", readout)
        o = vals("OPALLO", "content", readout) - vals("BASE", "content", readout)
        i = vals("ADD-100", "content", readout) - vals("SPARSE", "content", readout) - vals("OPALLO", "content", readout) + vals("BASE", "content", readout)
        out["readouts"][readout]["interaction"]["content"] = {"kappa": ratio_ci(i, s, rng), "raw": mean_ci(i, rng)}
        sc = vals("SPARSE", "closing", readout) - vals("BASE", "closing", readout)
        ic = vals("ADD-100", "closing", readout) - vals("SPARSE", "closing", readout) - vals("OPALLO", "closing", readout) + vals("BASE", "closing", readout)
        out["readouts"][readout]["interaction"]["closing"] = {"kappa": ratio_ci(ic, sc, rng), "raw": mean_ci(ic, rng)}
        for arm in COEFF:
            gain = vals(arm, "content", readout) - vals("BASE", "content", readout)
            out["readouts"][readout]["P_S"][arm] = ratio_ci(gain, s, rng)
        rho = vals("ADD-O75", "content", readout) - (.25 * vals("SPARSE", "content", readout) + .75 * vals("ADD-100", "content", readout))
        out["readouts"][readout]["beta_curvature_content"] = mean_ci(rho, rng)
        conflict = vals("CONFLICT-LOCAL", "content", readout) - vals("ADD-100", "content", readout)
        out["readouts"][readout]["conflict_local_minus_add100_content"] = mean_ci(conflict, rng)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["readouts"]["clipped_logodds"], indent=2))


if __name__ == "__main__":
    main()
