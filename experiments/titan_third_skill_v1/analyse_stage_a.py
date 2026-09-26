"""TRISKILL-001 Stage A analysis: the registered canaries and rules A1-A3 (PREREG §6), nothing else.

    python analyse_stage_a.py a1        # donor gate only (the runner calls this after C100 is scored)
    python analyse_stage_a.py full      # canaries, A1, A2, A3 -> STAGE_A_RESULT.json (create-only)

Reads only files under stage_a/: c_preds/<ARM>__<dev|cal>.jsonl, t/<ARM>.json (score_arms.py
output on T_BANK_V3X), h/<ARM>.json (score_boundary.py output on HSKILL_VIEWS_V2).
- C content = SIF semantic correctness (sif.scoring), on the 192 JSON-contract rows of DEV+CAL.
- T primary = exact match after stripping whitespace and at most one trailing '.'.
- H = boundary_abstain from score_boundary (boundary_rule 416cfb6a).
The power projection at the end is descriptive: PREREG §6 forbids it changing any Stage B rule.
"""
from __future__ import annotations

import json, random, sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SA = HERE / "stage_a"
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/Rineya-Dataset/Rineya_Skill_Integration_Foundry_v1/src")
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hskill-views-002")
from sif.scoring import score_records  # noqa: E402
import boundary_rule as BR  # noqa: E402

V3X = EXP / "composition_confirmation_v1" / "T_BANK_V3X.json"
REF_T = EXP / "composition_confirmation_v2" / "STAGE_A_T_V3X.json"
REF_H = EXP / "composition_confirmation_v2" / "STAGE_A_H_VIEWS_NEW.json"
PHASE0_BASE_DEV = HERE / "phase0_predictions" / "BASE__atomic_C.jsonl"
TT_NAME = "Rineya-Titan-Transition-1.0.0-devfreeze"
BETAS = {"TTC25": 0.25, "TTC50": 0.50, "TTC100": 1.00}
A1_MIN_NET, A1_MAX_P = 30, 0.01
A2_T_LOSS, A2_H_LOSS, A2_MIN_G = 8, 5, 20
A3_FRACTION = 0.8


def mcnemar_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def mcnemar_one_sided(gain: int, loss: int) -> float:
    """P(X >= gain) for X ~ Bin(gain+loss, 1/2): the exact one-sided test that TTC beats TT."""
    n = gain + loss
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(gain, n + 1)) / 2 ** n


def c_rows(arm: str) -> dict:
    """id -> scored row, over DEV+CAL (256 rows); callers filter to the 192 JSON rows."""
    sys.path.insert(0, str(HERE))
    from score_c import load
    out = {}
    for bank in ("dev", "cal"):
        _, keys = load(bank)
        p = SA / "c_preds" / f"{arm}__{bank}.jsonl"
        preds = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        _, rows = score_records(keys, preds)
        comp = {x["id"]: x["completion"] for x in preds}
        for r in rows:
            r["refusal"] = bool(BR.abstains(BR.first_line(comp[r["id"]])))
            out[r["id"]] = r
    return out


def json192(rows: dict) -> dict:
    return {k: v for k, v in rows.items() if v["variant"] != "alternate_format"}


def paired(a: dict, b: dict, key: str):
    """(b_gains_over_a, b_losses_vs_a) on a shared id set."""
    assert set(a) == set(b)
    g = sum(b[i][key] and not a[i][key] for i in a)
    l = sum(a[i][key] and not b[i][key] for i in a)
    return g, l


def t_rows(arm: str) -> dict:
    ref = {r["id"]: r["reference"] for r in json.loads(V3X.read_text(encoding="utf-8"))["rows"]}
    d = json.loads((SA / "t" / f"{arm}.json").read_text(encoding="utf-8"))
    preds = d["arms"][0]["predictions"]
    out = {}
    for p in preds:
        g = p["generation"].strip()
        prim = g[:-1] if g.endswith(".") else g
        out[p["id"]] = {"primary": prim == ref[p["id"]], "strict": g == ref[p["id"]], "generation": p["generation"],
                        "trailing_period": g.endswith("."), "refusal": bool(BR.abstains(BR.first_line(p["generation"])))}
    if len(out) != 864:
        raise SystemExit(f"{arm}: T predictions {len(out)} != 864")
    return out


def h_rows(arm: str) -> dict:
    d = json.loads((SA / "h" / f"{arm}.json").read_text(encoding="utf-8"))
    return {p["id"]: p for p in d["arms"][0]["predictions"]}


def a1() -> dict:
    base, c100 = json192(c_rows("BASE")), json192(c_rows("C100"))
    g, l = paired(base, c100, "semantic_correct")
    p = mcnemar_two_sided(g, l)
    return {"rule": f"C100 - BASE on C semantic (192 JSON rows) >= +{A1_MIN_NET}, exact McNemar p < {A1_MAX_P}",
            "base_semantic": sum(r["semantic_correct"] for r in base.values()),
            "c100_semantic": sum(r["semantic_correct"] for r in c100.values()),
            "c100_strict": sum(r["strict_correct"] for r in c100.values()),
            "gained": g, "lost": l, "net": g - l, "p_two_sided": p,
            "pass": (g - l) >= A1_MIN_NET and p < A1_MAX_P}


def canaries() -> dict:
    k1 = (SA / "c_preds" / "BASE__dev.jsonl").read_bytes() == PHASE0_BASE_DEV.read_bytes()
    ref_t = {p["id"]: p["generation"] for p in next(a for a in json.loads(REF_T.read_text(encoding="utf-8"))["arms"] if a["arm"] == "TH25")["predictions"]}
    tt_t = t_rows("TT")
    k2 = {i: tt_t[i]["generation"] for i in tt_t} == ref_t
    ref_h = {p["id"]: p["prediction"] for p in next(a for a in json.loads(REF_H.read_text(encoding="utf-8"))["arms"] if a["arm"] == "TH25")["predictions"]}
    tt_h = {i: p["prediction"] for i, p in h_rows("TT").items()}
    k3 = tt_h == ref_h
    tt_strict = sum(r["strict"] for r in tt_t.values())
    tt_boundary = sum(p["credited"] for p in h_rows("TT").values() if p["row_type"] == "boundary")
    return {"K1_base_dev_C_bytes_equal_phase0": k1, "K2_tt_T_generations_equal_confirm002_TH25": k2,
            "K3_tt_H_generations_equal_confirm002_TH25": k3, "tt_T_strict": tt_strict, "tt_boundary": tt_boundary,
            "expected": {"tt_T_strict": 444, "tt_boundary": 86},
            "pass": k1 and k2 and k3 and tt_strict == 444 and tt_boundary == 86}


def power_projection(c_tt, c_x, t_tt, t_x, h_tt, h_x, draws=2000, seed=20260926) -> dict:
    """Descriptive only. Resample Stage A paired rows to Stage B sizes; normal-approx bounds."""
    rng = random.Random(seed)

    def lb(pairs, n):   # one-sided 95% lower bound of sum(x - tt) over n resampled rows
        s = [rng.choice(pairs) for _ in range(n)]
        d = [x - t for t, x in s]; m = sum(d) / n
        var = sum((v - m) ** 2 for v in d) / max(1, n - 1)
        return n * m - 1.645 * (var * n) ** 0.5, sum(t for t, _ in s)
    cp = [(int(c_tt[i]["semantic_correct"]), int(c_x[i]["semantic_correct"])) for i in c_tt]
    tp = [(int(t_tt[i]["primary"]), int(t_x[i]["primary"])) for i in t_tt]
    hp = [(int(h_tt[i]["credited"]), int(h_x[i]["credited"])) for i in h_tt if h_tt[i]["row_type"] == "boundary"]
    ok = {"C1": 0, "C2": 0, "C3": 0}
    for _ in range(draws):
        b1, t1 = lb(tp, 864); ok["C1"] += b1 > -0.05 * t1
        b2, t2 = lb(hp, 336); ok["C2"] += b2 > -0.05 * t2
        s = [rng.choice(cp) for _ in range(336)]
        g = sum(x and not t for t, x in s); l = sum(t and not x for t, x in s)
        ok["C3"] += (g - l) >= 34 and mcnemar_one_sided(g, l) < 0.05
    return {k: round(v / draws, 3) for k, v in ok.items()} | {"draws": draws, "note": "descriptive; changes no Stage B rule"}


def full() -> int:
    out_path = HERE / "STAGE_A_RESULT.json"
    if out_path.exists():
        raise SystemExit("refusing to overwrite STAGE_A_RESULT.json")
    can = canaries()
    res = {"schema": 1, "experiment": "TRISKILL-001", "stage": "A", "registered_in": "PREREG.md (0d2b64b4) section 6",
           "canaries": can, "quality_claim": False, "release_permitted": False}
    if not can["pass"]:
        res["verdict"] = "STOP_CANARY_FAILED"
        out_path.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"); print(json.dumps(res, indent=1)); return 2
    res["A1"] = a1()
    if not res["A1"]["pass"]:
        res["verdict"] = "STOP_A1_DONOR_GATE_FAILED"
        out_path.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"); print(json.dumps(res, indent=1)); return 0
    c_tt = json192(c_rows("TT")); t_tt = t_rows("TT"); h_tt = h_rows("TT")
    tt_T = sum(r["primary"] for r in t_tt.values())
    tt_H = sum(p["credited"] for p in h_tt.values() if p["row_type"] == "boundary")
    tt_C = sum(r["semantic_correct"] for r in c_tt.values())
    a2, table = {}, {"TT": {"C_semantic": tt_C, "C_strict": sum(r["strict_correct"] for r in c_tt.values()), "T_primary": tt_T, "H_boundary": tt_H}}
    for arm, beta in BETAS.items():
        c = json192(c_rows(arm)); t = t_rows(arm); h = h_rows(arm)
        cg, cl = paired(c_tt, c, "semantic_correct")
        tg, tl = paired(t_tt, t, "primary")
        hb = {i: p for i, p in h.items() if p["row_type"] == "boundary"}; hb_tt = {i: p for i, p in h_tt.items() if p["row_type"] == "boundary"}
        hg, hl = paired(hb_tt, hb, "credited")
        T = sum(r["primary"] for r in t.values()); H = sum(p["credited"] for p in hb.values()); C = sum(r["semantic_correct"] for r in c.values())
        g = C - tt_C
        a2[arm] = {"beta": beta, "g_C_semantic_gain": g, "C_gained": cg, "C_lost": cl,
                   "T_loss": tt_T - T, "T_gained": tg, "T_lost": tl, "H_loss": tt_H - H, "H_gained": hg, "H_lost": hl,
                   "eligible": (tt_T - T) <= A2_T_LOSS and (tt_H - H) <= A2_H_LOSS and g >= A2_MIN_G}
        table[arm] = {"C_semantic": C, "C_strict": sum(r["strict_correct"] for r in c.values()), "T_primary": T,
                      "T_strict": sum(r["strict"] for r in t.values()), "T_trailing_periods": sum(r["trailing_period"] for r in t.values()),
                      "H_boundary": H, "H_answerable_attempts": sum(p.get("attempt", False) for p in h.values() if p["row_type"] == "answerable"),
                      "refusals_on_C": sum(r["refusal"] for r in c.values()), "refusals_on_T": sum(r["refusal"] for r in t.values())}
    for arm in ("BASE", "C25", "C50", "C100"):
        c = json192(c_rows(arm))
        table[arm] = {"C_semantic": sum(r["semantic_correct"] for r in c.values()), "C_strict": sum(r["strict_correct"] for r in c.values())}
    elig = {a: v for a, v in a2.items() if v["eligible"]}
    res["A2"], res["table"] = a2, table
    if not elig:
        res["verdict"] = "STOP_NO_ELIGIBLE_BETA"
        res["A3"] = None
    else:
        gmax = max(v["g_C_semantic_gain"] for v in elig.values())
        pick = min((v["beta"], a) for a, v in elig.items() if v["g_C_semantic_gain"] >= A3_FRACTION * gmax)
        res["A3"] = {"g_max_eligible": gmax, "threshold": A3_FRACTION * gmax, "beta_star": pick[0], "arm": pick[1]}
        res["verdict"] = "PROCEED_TO_STAGE_B"
        x = pick[1]
        res["power_projection_at_stage_b_sizes"] = power_projection(c_tt, json192(c_rows(x)), t_tt, t_rows(x), h_tt, h_rows(x))
    out_path.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: res[k] for k in res if k in ("canaries", "A1", "A2", "A3", "verdict", "power_projection_at_stage_b_sizes")}, indent=1))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("a1", "full"):
        raise SystemExit("usage: analyse_stage_a.py {a1|full}")
    if sys.argv[1] == "a1":
        r = a1(); print(json.dumps(r)); raise SystemExit(0 if r["pass"] else 3)
    raise SystemExit(full())
