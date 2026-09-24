"""CONFIRM-001 registered analysis. Written BEFORE any arm was scored on T_BANK_V3 or T_BANK_V3X.

Implements PREREG.md as amended by AMENDMENT_1.md (both committed before any arm was scored).

One point the preregistration left implicit is fixed here, still before any data: how C3's margin
test enters the Holm family. Two readings are computed and BOTH are reported; the STRICTER governs
the confirmatory verdict, so the choice cannot be made after seeing which one passes.

  C3-a (95% bound, as literally written): the lower 95% bootstrap bound (5th percentile) of
        (ADD-100 - SPARSE) exceeds -10% of SPARSE.
  C3-b (Holm-level): one-sided bootstrap p = P*(diff <= -margin) enters Holm with C1 and C2.
  The 2.5th percentile is reported as well. It is descriptive and does not govern.

Amendment 1: the governing bank is T_BANK_V3X (864 rows = T_BANK_V3's 288 verbatim + 576 new).
Row thresholds scale with n (gates to their stated percentages; C5/C6 bands by n/288). The
original 288 rows are ALSO analysed with the original thresholds and reported as a descriptive
secondary that does not govern. C6 (RAND_MASK vs SCALAR_P) is new and outside the Holm family.

Primary rule: exact match after stripping whitespace and AT MOST ONE trailing '.'.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BANK = HERE / "T_BANK_V3X.json"
BANK_SHA = "__FILLED_AFTER_GENERATION__"
V3_PREFIX = "CF1-"                      # T_BANK_V3's own rows; the extension uses CF1X-
SEED = 20260924
BOOT = 10_000
ALPHA = 0.05


def primary(gen: str, ref: str) -> bool:
    g = gen.strip()
    if g.endswith("."):
        g = g[:-1]
    return g == ref


def strict(gen: str, ref: str) -> bool:
    return gen.strip() == ref


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(run_files):
    import hashlib
    raw = BANK.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BANK_SHA:
        raise SystemExit(f"{BANK.name} does not match its committed sha256")
    bank = {r["id"]: r for r in json.loads(raw)["rows"]}
    arms = {}
    for f in run_files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if d["bank_sha256"] != BANK_SHA:
            raise SystemExit(f"{f} was scored on a different bank")
        if d["runtime"]["max_new_tokens"] != 48:
            raise SystemExit(f"{f} was decoded at max_new_tokens={d['runtime']['max_new_tokens']}, not 48")
        for a in d["arms"]:
            g = {p["id"]: p["generation"] for p in a["predictions"]}
            if set(g) != set(bank):
                raise SystemExit(f"arm {a['arm']} row set differs from the bank")
            if a["arm"] in arms:
                raise SystemExit(f"arm {a['arm']} appears in two run files")
            arms[a["arm"]] = g
    return bank, arms


def thresholds(n: int, original: bool) -> dict:
    if original:                          # PREREG.md as committed in c3a6f8a0, n = 288
        assert n == 288
        return {"G1_max": 57, "G2_min": 115, "G3_min": 86, "C5_effect": 10, "C5_null": 6}
    return {"G1_max": (2 * n) // 10,      # BASE <= 20%
            "G2_min": -(-4 * n // 10),    # SPARSE >= 40%
            "G3_min": -(-3 * n // 10),    # separation >= 30pp
            "C5_effect": round(10 * n / 288), "C5_null": round(6 * n / 288)}


def vec(bank, ids, arms, name, rule):
    return np.array([rule(arms[name][i], bank[i]["reference"]) for i in ids], dtype=bool)


def paired(x, y):
    b = int((x & ~y).sum()); c = int((y & ~x).sum())
    return {"x": int(x.sum()), "y": int(y.sum()), "net_y_minus_x": c - b, "lost": b, "gained": c, "p": mcnemar(b, c)}


def holm(ps: dict) -> dict:
    order = sorted(ps, key=lambda k: ps[k])
    m, out, still = len(order), {}, True
    for i, k in enumerate(order):
        thr = ALPHA / (m - i)
        still = still and ps[k] <= thr
        out[k] = {"p": ps[k], "threshold": thr, "significant": still}
    return out


def band_verdict(diff: int, p: float, t: dict, pos: str, neg: str, null: str) -> str:
    if diff >= t["C5_effect"] and p < ALPHA:
        return pos
    if diff <= -t["C5_effect"] and p < ALPHA:
        return neg
    if abs(diff) <= t["C5_null"]:
        return null
    return "INCONCLUSIVE"


def gates(bank, ids, arms, t) -> dict:
    base_p, spar_p = vec(bank, ids, arms, "BASE", primary), vec(bank, ids, arms, "SPARSE", primary)
    g = {"thresholds": {k: t[k] for k in ("G1_max", "G2_min", "G3_min")},
         "G1_base_floor": int(base_p.sum()) <= t["G1_max"],
         "G2_donor_ceiling": int(spar_p.sum()) >= t["G2_min"],
         "G3_separation": int(spar_p.sum()) - int(base_p.sum()) >= t["G3_min"],
         "BASE_primary": int(base_p.sum()), "SPARSE_primary": int(spar_p.sum()),
         "BASE_strict": int(vec(bank, ids, arms, "BASE", strict).sum()),
         "SPARSE_strict": int(vec(bank, ids, arms, "SPARSE", strict).sum())}
    g["ALL_PASS"] = g["G1_base_floor"] and g["G2_donor_ceiling"] and g["G3_separation"]
    return g


def analyse(bank, ids, arms, t) -> dict:
    n = len(ids)
    out = {"n": n, "gates": gates(bank, ids, arms, t)}
    V = {k: vec(bank, ids, arms, k, primary) for k in arms}
    S, AVG, ADD = V["SPARSE"], V["AVG"], V["ADD-100"]
    sparse_n = int(S.sum()); margin = 0.10 * sparse_n

    c1 = paired(S, AVG); c1["attenuation_rows"] = -c1["net_y_minus_x"]; c1["magnitude_ok"] = c1["attenuation_rows"] >= margin
    c2 = paired(AVG, ADD); c2["direction_ok"] = c2["net_y_minus_x"] > 0
    rng = np.random.default_rng(SEED)
    d = ADD.astype(int) - S.astype(int)
    boots = np.array([d[rng.integers(0, n, n)].sum() for _ in range(BOOT)])
    c3 = {"diff_ADD_minus_SPARSE": int(d.sum()), "margin_rows": -margin,
          "retained_fraction": (sparse_n + int(d.sum())) / sparse_n if sparse_n else None,
          "lower95": float(np.percentile(boots, 5)), "lower97_5_descriptive": float(np.percentile(boots, 2.5)),
          "one_sided_p": float((boots <= -margin).mean())}
    c3["C3a_bound_ok"] = c3["lower95"] > -margin
    H = holm({"C1": c1["p"], "C2": c2["p"], "C3": c3["one_sided_p"]})
    c3["C3b_holm_ok"] = H["C3"]["significant"]
    out["confirmatory"] = {
        "C1_attenuation": {**c1, "holm": H["C1"], "PASS": c1["magnitude_ok"] and H["C1"]["significant"]},
        "C2_add100_beats_avg": {**c2, "holm": H["C2"], "PASS": c2["direction_ok"] and H["C2"]["significant"]},
        "C3_interference_small": {**c3, "holm": H["C3"],
                                  "PASS": c3["C3a_bound_ok"] and c3["C3b_holm_ok"],
                                  "governing_reading": "stricter of C3-a and C3-b (both must hold)"},
    }
    per = lambda k: sum(arms[k][i].strip().endswith(".") for i in ids)
    out["C4_convention_descriptive"] = {"ADD-100_periods": per("ADD-100"), "SPARSE_periods": per("SPARSE"),
                                        "AVG_periods": per("AVG"),
                                        "meets_registered_pattern": per("ADD-100") >= 0.5 * n and per("SPARSE") <= 0.02 * n and per("AVG") <= 0.02 * n}
    if "RAND_MASK" in V and "SPARSE_ATTN" in V:
        c5 = paired(V["RAND_MASK"], V["SPARSE_ATTN"])        # net = SPARSE_ATTN - RAND_MASK
        diff = c5["net_y_minus_x"]
        verdict = band_verdict(diff, c5["p"], t,
                               "LOCATION EFFECT REPLICATES (attention carries more than random elements)",
                               "REVERSED (random elements carry more than attention)",
                               f"NO LOCATION EFFECT (|diff| <= {t['C5_null']})")
        out["C5_location"] = {**c5, "SPARSE_ATTN_minus_RAND_MASK": diff, "verdict": verdict,
                              "bands": {"effect": t["C5_effect"], "null": t["C5_null"]},
                              "reference_SCALAR_MATCHED": int(V["SCALAR_MATCHED"].sum()) if "SCALAR_MATCHED" in V else None}
    if "RAND_MASK" in V and "SCALAR_P" in V:
        c6 = paired(V["SCALAR_P"], V["RAND_MASK"])           # net = RAND_MASK - SCALAR_P
        diff = c6["net_y_minus_x"]
        verdict = band_verdict(diff, c6["p"], t,
                               "MASK EXCEEDS ITS EXPECTATION (RAND_MASK > SCALAR_P)",
                               "MASK FALLS SHORT OF ITS EXPECTATION (RAND_MASK < SCALAR_P)",
                               f"MASK ACTS LIKE ITS EXPECTATION (|diff| <= {t['C5_null']})")
        out["C6_mask_vs_expectation"] = {**c6, "RAND_MASK_minus_SCALAR_P": diff, "verdict": verdict,
                                         "bands": {"effect": t["C5_effect"], "null": t["C5_null"]}}
    out["per_arm_primary"] = {k: int(v.sum()) for k, v in sorted(V.items())}
    out["per_arm_strict"] = {k: int(vec(bank, ids, arms, k, strict).sum()) for k in sorted(arms)}
    out["per_arm_by_family_primary"] = {
        k: {fam: int(sum(v[j] for j, i in enumerate(ids) if bank[i]["family"] == fam))
            for fam in sorted({bank[i]["family"] for i in ids})} for k, v in sorted(V.items())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    bank, arms = load(a.runs)
    ids = sorted(bank)
    t = thresholds(len(ids), original=False)
    res = {"instrument": "CONFIRM-001", "bank": BANK.name, "bank_sha256": BANK_SHA, "n": len(ids),
           "rule_primary": "strip <=1 trailing '.'", "governing": "full T_BANK_V3X (Amendment 1)"}
    g = gates(bank, ids, arms, t)
    res["gates"] = g
    if a.gates_only or not g["ALL_PASS"]:
        print(json.dumps(res, indent=2))
        if a.output:
            a.output.write_bytes((json.dumps(res, indent=2) + "\n").encode())
        return 0 if g["ALL_PASS"] else 3

    res["governing_result"] = analyse(bank, ids, arms, t)
    v3_ids = [i for i in ids if i.startswith(V3_PREFIX)]
    res["secondary_original_v3_288_descriptive"] = {
        "note": "T_BANK_V3's own 288 rows under the ORIGINAL thresholds. Reported regardless; does not govern.",
        **analyse(bank, v3_ids, arms, thresholds(len(v3_ids), original=True))}
    print(json.dumps(res, indent=2))
    if a.output:
        a.output.write_bytes((json.dumps(res, indent=2) + "\n").encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
