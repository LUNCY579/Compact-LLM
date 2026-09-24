"""FUNCTIONAL-ADDITIVITY-001 Amendment 1 analysis (POST-HOC; see PLAN.md). Writes AMENDMENT_1.json.

Nothing here is registered. Outcome labels are what the REGISTERED rule would give when applied to
post-hoc readouts; they do not replace the registered primary (ADDITIVE, analysed in ../ANALYSIS.json).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXACT = HERE / "TF_EXACT.json"
OUT = HERE / "AMENDMENT_1.json"
ARMS = ["BASE", "SPARSE", "OPALLO", "ADD-O75", "ADD-100"]
REGISTERED_PRIMARY_KAPPA = -0.063952  # ../ANALYSIS.json, 6 d.p.


def label(point, lo, hi):
    if lo > -0.10 and hi < 0.10:
        return "ADDITIVE"
    if hi < 0 and point <= -0.10:
        return "SUB-ADDITIVE"
    if lo > 0 and point >= 0.10:
        return "SUPER-ADDITIVE"
    return "UNRESOLVED"


def kappa(Z):
    """Z: dict arm -> per-row array. Returns point, CI, label, mean I, mean g_S (fresh rng per call)."""
    I = Z["ADD-100"] - Z["SPARSE"] - Z["OPALLO"] + Z["BASE"]
    gS = Z["SPARSE"] - Z["BASE"]
    point = I.mean() / gS.mean()
    rng = np.random.default_rng(20260923)
    idx = rng.integers(0, len(I), size=(10000, len(I)))
    boot = I[idx].mean(axis=1) / gS[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"kappa": float(point), "ci95": [float(lo), float(hi)], "label_under_registered_rule": label(point, lo, hi),
            "mean_I": float(I.mean()), "mean_gS": float(gS.mean()), "rows_I_negative": int((I < 0).sum())}


def main() -> int:
    d = json.loads(EXACT.read_text(encoding="utf-8"))
    if not d["canary_all_passed"]:
        raise SystemExit("canary failed: amendment not read")
    rows = {a["arm"]: a["rows"] for a in d["arms"]}
    ids = [r["id"] for r in rows["BASE"]]
    for a in ARMS:
        assert [r["id"] for r in rows[a]] == ids
    fam = np.array([r["family"] for r in rows["BASE"]])

    def content_z(a, clip):
        out = []
        for r in rows[a]:
            z = np.array(r["z_unclipped"][:-1], dtype=float)  # content = gold tokens 0..G-2
            if clip is not None:
                z = np.clip(z, -clip, clip)
            out.append(z)
        return out

    res = {"schema": 1, "instrument": d["instrument"], "registered": False,
           "tf_exact_sha256": hashlib.sha256(EXACT.read_bytes()).hexdigest(),
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "bootstrap": {"n": 10000, "rng": "numpy default_rng(20260923), fresh per readout", "unit": "row"}}

    # 1. clip sweep on content log-odds
    sweep = {}
    for clip in [5, 8, 10, 12, 15, 20, None]:
        Z = {a: np.array([z.sum() for z in content_z(a, clip)]) for a in ARMS}
        k = kappa(Z)
        caps = {a: float(np.mean(np.concatenate([np.abs(z) >= clip for z in content_z(a, None)]))) if clip else 0.0
                for a in ("SPARSE", "ADD-100")}
        k["fraction_content_positions_at_or_beyond_clip"] = caps
        sweep["none" if clip is None else str(clip)] = k
    res["clip_sweep_content_logodds"] = sweep
    res["canary_primary_reproduced"] = abs(sweep["10"]["kappa"] - REGISTERED_PRIMARY_KAPPA) < 5e-7

    # 2. secondary (b) per POSITION, as PREREG line 71 defines it
    lp = {a: [np.array(r["logprob"][:-1], dtype=float) for r in rows[a]] for a in ARMS}
    decision = [np.any(np.stack([np.exp(lp[a][i]) < 0.9 for a in ARMS]), axis=0) if len(lp["BASE"][i]) else
                np.zeros(0, dtype=bool) for i in range(len(ids))]
    res["decision_positions"] = {"qualifying": int(sum(m.sum() for m in decision)),
                                 "content_positions": int(sum(len(m) for m in decision))}
    for tag, clip in (("clip10", 10), ("unclipped", None)):
        Zc = {a: content_z(a, clip) for a in ARMS}
        Z = {a: np.array([Zc[a][i][decision[i]].sum() for i in range(len(ids))]) for a in ARMS}
        res[f"secondary_b_per_position_{tag}"] = kappa(Z)

    # 3. per-family kappa (clip 10 and unclipped)
    pf = {}
    for f in sorted(set(fam)):
        m = fam == f
        pf[f] = {}
        for tag, clip in (("clip10", 10), ("unclipped", None)):
            Z = {a: np.array([z.sum() for z in content_z(a, clip)])[m] for a in ARMS}
            pf[f][tag] = kappa(Z)
        pf[f]["n"] = int(m.sum())
    res["per_family"] = pf

    # 4. secondary 7: period propensity at the convention site
    prop = {}
    for a in ARMS:
        vals, wins, merged_n, int_n = [], 0, 0, 0
        for r in rows[a]:
            if r["merged_id"] is not None:      # "}" -> "}." : the period acts at the final gold position
                v = r["final_pos_logprob_merged"]
                gold = r["logprob"][-1]
                merged_n += 1
            else:                                # integer rows: "." competes with EOS at the EOS position
                v = r["eos_pos_logprob_period"]
                gold = r["eos_logprob"]
                int_n += 1
            vals.append(v)
            wins += int(v > gold)
        prop[a] = {"mean_logprob_period_variant": float(np.mean(vals)),
                   "rows_period_variant_beats_gold_continuation": wins, "merged_rows": merged_n, "integer_rows": int_n}
    res["secondary_7_period_propensity"] = prop

    OUT.write_bytes((json.dumps(res, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({k: res[k] for k in ("canary_primary_reproduced", "decision_positions")}, indent=1))
    for c, k in sweep.items():
        print(f"clip {c:>4}: kappa {k['kappa']:+.5f} CI [{k['ci95'][0]:+.5f},{k['ci95'][1]:+.5f}] "
              f"{k['label_under_registered_rule']:<13} capped SPARSE {k['fraction_content_positions_at_or_beyond_clip']['SPARSE']:.3f} "
              f"ADD-100 {k['fraction_content_positions_at_or_beyond_clip']['ADD-100']:.3f}")
    for tag in ("clip10", "unclipped"):
        k = res[f"secondary_b_per_position_{tag}"]
        print(f"(b) per position {tag}: kappa {k['kappa']:+.5f} CI [{k['ci95'][0]:+.5f},{k['ci95'][1]:+.5f}] {k['label_under_registered_rule']}")
    for f, v in pf.items():
        print(f"family {f:<18} n={v['n']} clip10 {v['clip10']['kappa']:+.4f} [{v['clip10']['ci95'][0]:+.4f},{v['clip10']['ci95'][1]:+.4f}]"
              f"  unclipped {v['unclipped']['kappa']:+.4f} [{v['unclipped']['ci95'][0]:+.4f},{v['unclipped']['ci95'][1]:+.4f}] gS={v['unclipped']['mean_gS']:.2f}")
    for a, v in prop.items():
        print(f"period propensity {a:<8} mean logp {v['mean_logprob_period_variant']:+.3f}  beats gold {v['rows_period_variant_beats_gold_continuation']}/192")
    print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
