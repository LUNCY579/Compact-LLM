"""CONFIRM-002 Stage A analysis (PREREG §4; AMENDMENT_1). Written before any Stage A arm was scored.

1. HARNESS CANARY (a gate): my scorer's BASE and SPARSE on HSKILL_VIEWS_V2 against the recorded native-probe
   artifacts HSKILL2_supa.json and HSKILL2_tsparse.json. PASS requires every count (boundary abstain,
   answerable attempt, answerable correct) within +/-2 rows AND per-row credited decisions agreeing on
   >= 216/224 rows, for both arms.
2. DEVELOPMENT ESTIMATES on exposed banks: T skill on T_BANK_V3X (BASE, SPARSE from CONFIRM-001's CONF_GATES.json;
   H25, TH25 and THFULL from Stage A), and the boundary skill on HSKILL_VIEWS_V2 (all five arms, my scorer).
3. POWER at Stage B sizes (T 864, boundary 336, answerable 109): 1,000 outer simulations resample the development
   per-row outcomes, JOINTLY across arms, with replacement to Stage B size. Each applies analyse_stage_b's own
   decision functions (1,000 inner bootstrap draws), giving P(pass) per limb.
   Rule: P < 0.80 for C1 means extend T; P < 0.80 for C2, C3 or C4 means extend the boundary half (AMENDMENT_1 §1).
   P < 0.20 for C1 or C2 raises the operator's spend decision. Arms, coefficients, margins and rules never change.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyse_stage_b as B  # noqa: E402

HV = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hskill-views-002")
sys.path.insert(0, str(HV))
import boundary_rule as BR  # noqa: E402

V3X = HERE.parent / "composition_confirmation_v1" / "T_BANK_V3X.json"
V3X_SHA = "173aa372466d5e3cc8efbebe4cc7e7ba09a8eb34de4598ce8f9d96a2b8e6557c"
VIEWS = HV / "HSKILL_VIEWS_V2.jsonl"
VIEWS_SHA = "ba1f2982fd5d08aac861d4547b1dadb33d20e6022ff9aa06934e1ad2e7154723"
RECORDED = {"BASE": HV / "scored" / "HSKILL2_supa.json", "SPARSE": HV / "scored" / "HSKILL2_tsparse.json"}
SIZES = {"T": 864, "boundary": 336, "answerable": 109}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def views_rows():
    assert sha(VIEWS) == VIEWS_SHA
    return {json.loads(l)["id"]: json.loads(l) for l in VIEWS.read_text(encoding="utf-8").splitlines() if l.strip()}


def recorded_decisions(path, rows):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    assert d["instrument_sha256"] == VIEWS_SHA
    out = {}
    for p in d["predictions"]:
        r = rows[p["id"]]; g = p["prediction"]
        out[p["id"]] = {"credited": bool(BR.score_row(r["row_type"], g, r["reference"])[1]),
                        "attempt": (not BR.abstains(BR.first_line(g))) if r["row_type"] == "answerable" else None, "text": g}
    return out


def counts(dec, rows):
    b = [i for i in rows if rows[i]["row_type"] == "boundary"]; a = [i for i in rows if rows[i]["row_type"] == "answerable"]
    return {"boundary_abstain": sum(dec[i]["credited"] for i in b), "answerable_attempt": sum(dec[i]["attempt"] for i in a),
            "answerable_correct": sum(dec[i]["credited"] for i in a)}


def mine(files, rows):
    out = {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        assert d["bank_sha256"] == VIEWS_SHA and d["runtime"]["max_new_tokens"] == 48, f
        for a in d["arms"]:
            out[a["arm"]] = {p["id"]: {"credited": p["credited"], "attempt": p.get("attempt"), "text": p["prediction"]} for p in a["predictions"]}
    return out


def canary(M, rows):
    res, ok = {}, True
    for arm, path in RECORDED.items():
        rec = recorded_decisions(path, rows); cm, cr = counts(M[arm], rows), counts(rec, rows)
        agree = sum(M[arm][i]["credited"] == rec[i]["credited"] for i in rows)
        text_eq = sum(M[arm][i]["text"] == rec[i]["text"] for i in rows)
        within = all(abs(cm[k] - cr[k]) <= 2 for k in cm)
        res[arm] = {"mine": cm, "recorded": cr, "per_row_credited_agreement": agree, "text_identical": text_eq,
                    "PASS": within and agree >= 216}
        ok &= res[arm]["PASS"]
    res["PASS"] = ok
    return res


def t_arrays(files):
    assert sha(V3X) == V3X_SHA
    ref = {r["id"]: r["reference"] for r in json.loads(V3X.read_text(encoding="utf-8"))["rows"]}; ids = sorted(ref)
    out = {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        assert d["bank_sha256"] == V3X_SHA and d["runtime"]["max_new_tokens"] == 48, f
        for a in d["arms"]:
            if a["arm"] in ("BASE", "SPARSE", "H25", "TH25", "THFULL"):
                g = {p["id"]: p["generation"] for p in a["predictions"]}
                out[a["arm"]] = np.array([B.prim(g[i], ref[i]) for i in ids])
    return out


def power(T, Hb, Ha, sims=1000, inner=1000, seed=7):
    rng = np.random.default_rng(seed)
    tally = {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "ALL4": 0, "C5": 0, "gates_T": 0, "G_H": 0}
    nT, nB, nA = len(T["SPARSE"]), len(Hb["SPARSE"]), len(Ha["SPARSE"])
    for _ in range(sims):
        it, ib, ia = rng.integers(0, nT, SIZES["T"]), rng.integers(0, nB, SIZES["boundary"]), rng.integers(0, nA, SIZES["answerable"])
        t = {k: v[it] for k, v in T.items()}; hb = {k: v[ib] for k, v in Hb.items()}; ha = {k: v[ia] for k, v in Ha.items()}
        gt = B.t_gates(t["BASE"], t["SPARSE"]); gt["ALL"] = gt["G1"] and gt["G2"] and gt["G3"]
        gh = B.g_h(hb["BASE"], hb["H25"])
        tally["gates_T"] += gt["ALL"]; tally["G_H"] += gh["PASS"]
        r = B.confirmatory(t, hb, ha, gt, gh, np.random.default_rng(int(rng.integers(1 << 31))), boot=inner)
        for k in ("C1", "C2", "C3", "C4"):
            tally[k] += bool(r[k].get("PASS"))
        tally["ALL4"] += all(r[k].get("PASS") for k in ("C1", "C2", "C3", "C4"))
        tally["C5"] += B.c5(t["SPARSE"], t["THFULL"])["PASS"]
    return {k: v / sims for k, v in tally.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t-runs", nargs="+", required=True, help="CONF_GATES.json (BASE, SPARSE on V3X) + Stage A T scores")
    ap.add_argument("--h-runs", nargs="+", required=True, help="Stage A boundary scores on HSKILL_VIEWS_V2 (all five arms)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sims", type=int, default=1000)
    a = ap.parse_args()
    rows = views_rows()
    M = mine(a.h_runs, rows)
    can = canary(M, rows)
    T = t_arrays(a.t_runs)
    bid = sorted(i for i in rows if rows[i]["row_type"] == "boundary"); aid = sorted(i for i in rows if rows[i]["row_type"] == "answerable")
    Hb = {k: np.array([M[k][i]["credited"] for i in bid]) for k in M}
    Ha = {k: np.array([M[k][i]["attempt"] for i in aid]) for k in M}
    dev = {"T_V3X_primary": {k: int(v.sum()) for k, v in T.items()},
           "T_TH25_vs_SPARSE": B.paired(T["SPARSE"], T["TH25"]), "T_THFULL_vs_SPARSE": B.paired(T["SPARSE"], T["THFULL"]),
           "VIEWS_boundary": {k: int(v.sum()) for k, v in Hb.items()}, "VIEWS_attempt": {k: int(v.sum()) for k, v in Ha.items()},
           "VIEWS_TH25_vs_SPARSE_boundary": B.paired(Hb["SPARSE"], Hb["TH25"]), "VIEWS_H25_vs_BASE_boundary": B.paired(Hb["BASE"], Hb["H25"])}
    pw = power(T, Hb, Ha, sims=a.sims)
    rec = {"extend_T": pw["C1"] < 0.80, "extend_boundary": min(pw["C2"], pw["C3"], pw["C4"]) < 0.80,
           "spend_decision_flag": pw["C1"] < 0.20 or pw["C2"] < 0.20}
    out = {"stage": "A", "canary": can, "development": dev, "power_at_stage_b_sizes": pw, "rules": rec,
           "note": "Stage A may only gate the harness, enlarge the fresh banks by extension, or raise a spend decision."}
    a.output.write_bytes((json.dumps(out, indent=2) + "\n").encode())
    print(json.dumps({"canary_PASS": can["PASS"], "canary": {k: v for k, v in can.items() if k != "PASS"}, "power": pw, "rules": rec}, indent=1))
    return 0 if can["PASS"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
