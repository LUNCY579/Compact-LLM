"""Fixed analysis for FUNCTIONAL-ADDITIVITY-001."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


ARM_NAMES = ["BASE", "SPARSE", "OPALLO", "ADD-O75", "ADD-100"]
READOUTS = ["clipped_logodds", "unclipped_logprob", "decision_logodds"]


def ci_ratio(num, den, rng, n=10000):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), size=(n, len(num)))
    means_num = num[idx].mean(axis=1)
    means_den = den[idx].mean(axis=1)
    ratios = means_num / means_den
    return {"point": float(num.mean() / den.mean()),
            "ci95": [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))],
            "mean_num": float(num.mean()), "mean_den": float(den.mean())}


def ci_mean(values, rng, n=10000):
    values = np.asarray(values, float)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return {"mean": float(values.mean()),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}


def outcome(k):
    lo, hi = k["ci95"]
    point = k["point"]
    if lo > -0.10 and hi < 0.10:
        return "ADDITIVE"
    if hi < 0 and point <= -0.10:
        return "SUB-ADDITIVE"
    if lo > 0 and point >= 0.10:
        return "SUPER-ADDITIVE"
    return "UNRESOLVED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    arms = {a["arm"]: {r["id"]: r for r in a["rows"]} for a in raw["arms"]}
    if set(arms) != set(ARM_NAMES):
        raise SystemExit(f"analysis requires exactly {ARM_NAMES}")
    ids = sorted(arms["BASE"])
    families = {rid: arms["BASE"][rid]["family"] for rid in ids}
    gates = raw["gates"]
    rng = np.random.default_rng(20260923)
    decision = {}
    for rid in ids:
        decision[rid] = any(p < 0.9 for arm in ARM_NAMES for p in arms[arm][rid]["content"]["token_probability"])
    for arm in ARM_NAMES:
        # Decision readout is filled using the symmetric rule above.
        for rid in ids:
            row = arms[arm][rid]
            row["content"]["decision_logodds"] = row["content"]["clipped_logodds"] if decision[rid] else 0.0
    def values(arm, part, key):
        return np.array([arms[arm][rid][part][key] for rid in ids], float)
    def family_result(readout, part="content"):
        out = {}
        for fam in sorted(set(families.values())):
            sel = np.array([families[rid] == fam for rid in ids])
            g = values("SPARSE", part, readout)[sel] - values("BASE", part, readout)[sel]
            i = (values("ADD-100", part, readout)[sel] - values("SPARSE", part, readout)[sel]
                 - values("OPALLO", part, readout)[sel] + values("BASE", part, readout)[sel])
            out[fam] = {"kappa": ci_ratio(i, g, rng), "interaction": ci_mean(i, rng), "n": int(sel.sum())}
        return out
    summary = {"schema": 1, "instrument": "FUNCTIONAL-ADDITIVITY-001",
               "raw_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
               "bootstrap": {"n": 10000, "seed": 20260923}, "gates": gates,
               "readouts": {}, "quality_claim": False, "release_permitted": False}
    all_gates = all(g.get("passed", False) for g in gates.values())
    summary["gates"]["G-D-denominator"] = {}
    for readout in READOUTS:
        den = values("SPARSE", "content", readout) - values("BASE", "content", readout)
        summary["gates"]["G-D-denominator"][readout] = ci_mean(den, rng)
    summary["gates"]["G-D-denominator"]["passed"] = all(
        x["ci95"][0] > 0 for k, x in summary["gates"]["G-D-denominator"].items() if k != "passed")
    all_gates = all_gates and summary["gates"]["G-D-denominator"]["passed"]
    for readout in READOUTS:
        for part in ("content", "closing"):
            if readout == "decision_logodds" and part == "closing":
                continue
            g = values("SPARSE", part, readout) - values("BASE", part, readout)
            i = values("ADD-100", part, readout) - values("SPARSE", part, readout) - values("OPALLO", part, readout) + values("BASE", part, readout)
            k = ci_ratio(i, g, rng)
            summary["readouts"].setdefault(readout, {})[part] = {
                "kappa": k, "outcome": outcome(k), "interaction": ci_mean(i, rng),
                "family": family_result(readout, part),
            }
        z = values("ADD-100", "content", readout) - values("BASE", "content", readout)
        s = values("SPARSE", "content", readout) - values("BASE", "content", readout)
        o = values("OPALLO", "content", readout) - values("BASE", "content", readout)
        ps = ci_ratio(z, s, rng)
        ps["additive_prediction"] = float(1 + o.mean() / s.mean())
        summary["readouts"][readout]["P_S"] = ps
        rho_c = values("ADD-O75", "content", readout) - (0.25 * values("SPARSE", "content", readout) + 0.75 * values("ADD-100", "content", readout))
        curvature = {"content": ci_mean(rho_c, rng)}
        if readout != "decision_logodds":
            rho_e = values("ADD-O75", "closing", readout) - (0.25 * values("SPARSE", "closing", readout) + 0.75 * values("ADD-100", "closing", readout))
            curvature["closing"] = ci_mean(rho_e, rng)
        summary["readouts"][readout]["beta_curvature"] = curvature
    # Registered row-level descriptive set: SPARSE period-tolerant correct and ADD-100 not.
    bank = json.loads((Path(__file__).resolve().parent.parent / "fresh_dual_skill_instrument_v1" / "T_BANK_V2.json").read_text())
    refs = {r["id"]: r["reference"] for r in bank["rows"]}
    # Raw generation records are in V2_ARMS; use their registered period-tolerant sets.
    fresh_root = Path(__file__).resolve().parent.parent / "fresh_dual_skill_instrument_v1"
    v2 = json.loads((fresh_root / "V2_ARMS.json").read_text())
    cal = json.loads((fresh_root / "CALIBRATION_V2.json").read_text())
    tolerant = {}
    for a in cal["arms"] + v2["arms"]:
        tolerant[a["arm"]] = {p["id"] for p in a["predictions"] if p["generation"] in (refs[p["id"]], refs[p["id"]] + ".")}
    rows_special = sorted(tolerant.get("SPARSE", set()) - tolerant.get("ADD-100", set()))
    summary["row_level"] = {"sparse_tolerant_correct_add100_tolerant_wrong": rows_special,
                             "n": len(rows_special),
                             "mean_interaction_content_special": float(np.mean([arms["ADD-100"][r]["content"]["clipped_logodds"] - arms["SPARSE"][r]["content"]["clipped_logodds"] - arms["OPALLO"][r]["content"]["clipped_logodds"] + arms["BASE"][r]["content"]["clipped_logodds"] for r in rows_special])) if rows_special else None,
                             "mean_interaction_content_other": float(np.mean([arms["ADD-100"][r]["content"]["clipped_logodds"] - arms["SPARSE"][r]["content"]["clipped_logodds"] - arms["OPALLO"][r]["content"]["clipped_logodds"] + arms["BASE"][r]["content"]["clipped_logodds"] for r in ids if r not in rows_special]))}
    summary["status"] = "ANALYZED" if all_gates else "GATE_FAILED_ANALYSIS_STOPPED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "gates": summary["gates"],
                      "primary": summary["readouts"].get("clipped_logodds", {}).get("content", {})}, indent=2))
    if not all_gates:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
