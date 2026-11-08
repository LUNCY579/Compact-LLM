"""INTERFERENCE-MECH-001 step 1 analysis, written before any step-1 arm was built or scored.

Exploratory, on T_BANK_V3X. CONFIRM-001 has spent that bank's confirmatory read, so it is now
exposed. The reference arms are CONFIRM-001's own recorded predictions on the same rows
(SPARSE, ADD-100). Step-1 arms come from IM_SCORES.json. Primary rule as CONFIRM-001: strip, then at
most one trailing '.'.

Predictions are in STEP1_PREREG.md. Bands reuse CONFIRM-001's n = 864 bands: "same" means
|diff| <= 18, "different" means |diff| >= 30 with exact McNemar p < 0.05, and anything else is
inconclusive.
"""
from __future__ import annotations

import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CC = HERE.parent / "composition_confirmation_v1"
BANK = CC / "T_BANK_V3X.json"
BANK_SHA = "173aa372466d5e3cc8efbebe4cc7e7ba09a8eb34de4598ce8f9d96a2b8e6557c"
SAME, DIFF = 18, 30


def prim(g, r):
    s = g.strip()
    return (s[:-1] if s.endswith(".") else s) == r


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(files):
    raw = BANK.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BANK_SHA
    ref = {r["id"]: r["reference"] for r in json.loads(raw)["rows"]}
    gen = {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        assert d["bank_sha256"] == BANK_SHA and d["runtime"]["max_new_tokens"] == 48, f
        for a in d["arms"]:
            g = {p["id"]: p["generation"] for p in a["predictions"]}
            assert set(g) == set(ref), a["arm"]
            gen[a["arm"]] = g
    ids = sorted(ref)
    return ids, {a: np.array([prim(gen[a][i], ref[i]) for i in ids]) for a in gen}


def band(x, y):
    """y relative to x."""
    b = int((x & ~y).sum()); c = int((y & ~x).sum()); d = c - b; p = mcnemar(b, c)
    v = "SAME" if abs(d) <= SAME else ("HIGHER" if d >= DIFF and p < 0.05 else ("LOWER" if d <= -DIFF and p < 0.05 else "INCONCLUSIVE"))
    return {"x": int(x.sum()), "y": int(y.sum()), "diff": d, "lost": b, "gained": c, "p": p, "band": v}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default=str(HERE / "IM_SCORES.json"))
    ap.add_argument("--output", type=Path, default=HERE / "STEP1_RESULT.json")
    a = ap.parse_args()
    ids, V = load([CC / "CONF_GATES.json", CC / "CONF_COMPOSE.json", a.scores])
    out = {"bank": BANK.name, "exploratory": True, "n": len(ids), "per_arm_primary": {k: int(v.sum()) for k, v in sorted(V.items())}}
    out["P1_DARE-O50_vs_ADD-100"] = band(V["ADD-100"], V["DARE-O50"])
    out["P2_DARE-O10_vs_ADD-100"] = band(V["ADD-100"], V["DARE-O10"])
    out["P3_DISJOINT-DARE_vs_ADD-100"] = band(V["ADD-100"], V["DISJOINT-DARE"])
    out["P4_SPARSE-125_vs_SPARSE"] = band(V["SPARSE"], V["SPARSE-125"])
    # P5: does raising the skill coefficient shrink the interference? I = (ADD-A125 - SPARSE-125) - (ADD-100 - SPARSE), per row
    i_row = (V["ADD-A125"].astype(int) - V["SPARSE-125"].astype(int)) - (V["ADD-100"].astype(int) - V["SPARSE"].astype(int))
    rng = np.random.default_rng(20260925)
    boots = np.array([i_row[rng.integers(0, len(ids), len(ids))].sum() for _ in range(10000)])
    out["P5_interference_shrinks_at_alpha_1.25"] = {
        "interference_at_1.00": int(V["ADD-100"].sum() - V["SPARSE"].sum()),
        "interference_at_1.25": int(V["ADD-A125"].sum() - V["SPARSE-125"].sum()),
        "I_rows": int(i_row.sum()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "prediction": "I >= +10 with the CI excluding 0",
        "met": bool(i_row.sum() >= 10 and np.percentile(boots, 2.5) > 0)}
    out["P6_ADD-A125_vs_SPARSE"] = band(V["SPARSE"], V["ADD-A125"])
    args_out = a.output
    args_out.write_bytes((json.dumps(out, indent=1) + "\n").encode())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
