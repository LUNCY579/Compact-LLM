"""CONFIRM-001 registered analysis. Written BEFORE any arm was scored on T_BANK_V3.

Implements PREREG.md exactly. One point the preregistration left implicit is fixed here, still
before any data: how C3's margin test enters the Holm family. Two readings are computed and BOTH
are reported; the STRICTER governs the confirmatory verdict, so the choice cannot be made after
seeing which one passes.

  C3-a (95% bound, as literally written): the lower 95% bootstrap bound of
        (ADD-100 - SPARSE) exceeds -10% of SPARSE.
  C3-b (Holm-level): one-sided bootstrap p = P*(diff <= -margin) enters Holm with C1 and C2.

Primary rule: exact match after stripping whitespace and AT MOST ONE trailing '.'.
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BANK = HERE / "T_BANK_V3.json"
BANK_SHA = "20597c2247eff91b4197d18e63fc02012673b06fb80ffb694d07fa0adfc7507a"
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
        raise SystemExit("T_BANK_V3.json does not match its committed sha256")
    bank = {r["id"]: r for r in json.loads(raw)["rows"]}
    arms = {}
    for f in run_files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if d["bank_sha256"] != BANK_SHA:
            raise SystemExit(f"{f} was scored on a different bank")
        for a in d["arms"]:
            g = {p["id"]: p["generation"] for p in a["predictions"]}
            if set(g) != set(bank):
                raise SystemExit(f"arm {a['arm']} row set differs from the bank")
            arms[a["arm"]] = g
    return bank, arms


def vec(bank, arms, name, rule):
    ids = sorted(bank)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    bank, arms = load(a.runs)
    n = len(bank)
    res = {"instrument": "CONFIRM-001", "bank_sha256": BANK_SHA, "n": n, "rule_primary": "strip <=1 trailing '.'"}

    base_p, spar_p = vec(bank, arms, "BASE", primary), vec(bank, arms, "SPARSE", primary)
    g = {"G1_base_le_57": int(base_p.sum()) <= 57, "G2_sparse_ge_115": int(spar_p.sum()) >= 115,
         "G3_sep_ge_86": int(spar_p.sum()) - int(base_p.sum()) >= 86,
         "BASE_primary": int(base_p.sum()), "SPARSE_primary": int(spar_p.sum()),
         "BASE_strict": int(vec(bank, arms, "BASE", strict).sum()),
         "SPARSE_strict": int(vec(bank, arms, "SPARSE", strict).sum())}
    g["ALL_PASS"] = g["G1_base_le_57"] and g["G2_sparse_ge_115"] and g["G3_sep_ge_86"]
    res["gates"] = g
    if a.gates_only or not g["ALL_PASS"]:
        print(json.dumps(res, indent=2))
        if a.output:
            a.output.write_bytes((json.dumps(res, indent=2) + "\n").encode())
        return 0 if g["ALL_PASS"] else 3

    V = {k: vec(bank, arms, k, primary) for k in arms}
    S, AVG, ADD = V["SPARSE"], V["AVG"], V["ADD-100"]
    sparse_n = int(S.sum()); margin = 0.10 * sparse_n

    c1 = paired(S, AVG); c1["attenuation_rows"] = -c1["net_y_minus_x"]; c1["magnitude_ok"] = c1["attenuation_rows"] >= margin
    c2 = paired(AVG, ADD); c2["direction_ok"] = c2["net_y_minus_x"] > 0
    rng = np.random.default_rng(SEED)
    d = ADD.astype(int) - S.astype(int)
    boots = np.array([d[rng.integers(0, n, n)].sum() for _ in range(BOOT)])
    c3 = {"diff_ADD_minus_SPARSE": int(d.sum()), "margin_rows": -margin,
          "lower95": float(np.percentile(boots, 5)), "one_sided_p": float((boots <= -margin).mean())}
    c3["C3a_bound_ok"] = c3["lower95"] > -margin
    H = holm({"C1": c1["p"], "C2": c2["p"], "C3": c3["one_sided_p"]})
    c3["C3b_holm_ok"] = H["C3"]["significant"]
    res["confirmatory"] = {
        "C1_attenuation": {**c1, "holm": H["C1"], "PASS": c1["magnitude_ok"] and H["C1"]["significant"]},
        "C2_add100_beats_avg": {**c2, "holm": H["C2"], "PASS": c2["direction_ok"] and H["C2"]["significant"]},
        "C3_interference_small": {**c3, "holm": H["C3"],
                                  "PASS": c3["C3a_bound_ok"] and c3["C3b_holm_ok"],
                                  "governing_reading": "stricter of C3-a and C3-b (both must hold)"},
    }
    per = lambda k: sum(arms[k][i].strip().endswith(".") for i in bank)
    res["C4_convention_descriptive"] = {"ADD-100_periods": per("ADD-100"), "SPARSE_periods": per("SPARSE"),
                                        "AVG_periods": per("AVG"),
                                        "meets_registered_pattern": per("ADD-100") >= 0.5 * n and per("SPARSE") <= 0.02 * n and per("AVG") <= 0.02 * n}
    if "RAND_MASK" in V and "SPARSE_ATTN" in V:
        c5 = paired(V["RAND_MASK"], V["SPARSE_ATTN"])        # net = SPARSE_ATTN - RAND_MASK
        diff = c5["net_y_minus_x"]
        if diff >= 10 and c5["p"] < ALPHA:
            verdict = "LOCATION EFFECT REPLICATES (attention carries more than random elements)"
        elif diff <= -10 and c5["p"] < ALPHA:
            verdict = "REVERSED (random elements carry more than attention)"
        elif abs(diff) <= 6:
            verdict = "NO LOCATION EFFECT (|diff| <= 6)"
        else:
            verdict = "INCONCLUSIVE"
        res["C5_location"] = {**c5, "SPARSE_ATTN_minus_RAND_MASK": diff, "verdict": verdict,
                              "reference_SCALAR_MATCHED": int(V["SCALAR_MATCHED"].sum()) if "SCALAR_MATCHED" in V else None}
    res["per_arm_primary"] = {k: int(v.sum()) for k, v in sorted(V.items())}
    res["per_arm_strict"] = {k: int(vec(bank, arms, k, strict).sum()) for k in sorted(arms)}
    print(json.dumps(res, indent=2))
    if a.output:
        a.output.write_bytes((json.dumps(res, indent=2) + "\n").encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
