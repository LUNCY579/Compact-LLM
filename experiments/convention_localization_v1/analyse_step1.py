"""CONVENTION-LOC-001 step 1 analysis. Written before any step-1 arm was built or scored.

Implements the predictions registered in RESULTS_STEP0B.md (commit 96ebaaa5) and STEP1_PREREG.md
(commit 3c5e125d), on T_BANK_V2. Exploratory: that bank is exposed.

Reference arms come from recorded files: SPARSE from CALIBRATION_V2.json, ADD-100 from V2_ARMS.json.
Step-1 arms come from STEP1_SCORES.json. "Period row" means generation.strip().endswith('.').
Tolerant (primary) rule: strip, then at most one trailing '.'. The strict rule is strip only.

The beta arms' per-row predictions come from the linear-logit model in linear_conflict.py:
m_hat(beta) = (1 - beta)*m(SPARSE) + beta*m(ADD-100), with m taken from TF_EXACT.json; a period is
predicted where m_hat > 0.
"""
from __future__ import annotations

import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FDS = EXP / "fresh_dual_skill_instrument_v1"
sys.path.insert(0, str(HERE))
from linear_conflict import margins, TF  # noqa: E402  (the committed step-0b margin definition)

V2_SHA = "fc094136a5b581cb9cc03393b1e5959ed8f69247638c81e37cf7f68fbf132e01"
BETA_PRED = {"ADD-O50": (0.5, 3, (0, 8)), "ADD-O60": (0.6, 12, (7, 17)), "ADD-O90": (0.9, 127, (95, 159))}
AGREE_MIN, TOLERANT_MIN = 173, 92


def prim(g, r):
    s = g.strip()
    return (s[:-1] if s.endswith(".") else s) == r


def strict(g, r):
    return g.strip() == r


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default=str(HERE / "STEP1_SCORES.json"))
    ap.add_argument("--output", type=Path, default=HERE / "STEP1_RESULT.json")
    a = ap.parse_args()
    raw = (FDS / "T_BANK_V2.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == V2_SHA
    ref = {r["id"]: r["reference"] for r in json.loads(raw)["rows"]}
    ids = sorted(ref)
    gen = {}
    for f in (FDS / "CALIBRATION_V2.json", FDS / "V2_ARMS.json", Path(a.scores)):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        assert d["bank_sha256"] == V2_SHA and d["runtime"]["max_new_tokens"] == 48, f
        for arm in d["arms"]:
            g = {p["id"]: p["generation"] for p in arm["predictions"]}
            assert set(g) == set(ref), arm["arm"]
            gen[arm["arm"]] = g
    per = {k: np.array([gen[k][i].strip().endswith(".") for i in ids]) for k in gen}
    T = {k: np.array([prim(gen[k][i], ref[i]) for i in ids]) for k in gen}
    S = {k: np.array([strict(gen[k][i], ref[i]) for i in ids]) for k in gen}

    tf = json.loads(TF.read_text(encoding="utf-8"))
    A = {x["arm"]: {r["id"]: r for r in x["rows"]} for x in tf["arms"]}
    assert sorted(A["SPARSE"]) == ids
    mS = margins([A["SPARSE"][i] for i in ids])[0]
    mA = margins([A["ADD-100"][i] for i in ids])[0]

    out = {"bank": "T_BANK_V2 (exposed)", "n": len(ids), "exploratory": True,
           "periods": {k: int(v.sum()) for k, v in sorted(per.items())},
           "tolerant": {k: int(v.sum()) for k, v in sorted(T.items())},
           "strict": {k: int(v.sum()) for k, v in sorted(S.items())}, "beta_arms": {}}
    for arm, (beta, pred_n, (lo, hi)) in BETA_PRED.items():
        pred = ((1 - beta) * mS + beta * mA) > 0
        agree = int((pred == per[arm]).sum())
        out["beta_arms"][arm] = {
            "beta": beta, "predicted_period_rows": int(pred.sum()), "registered_prediction": pred_n,
            "observed_period_rows": int(per[arm].sum()), "band": [lo, hi],
            "count_in_band": lo <= int(per[arm].sum()) <= hi,
            "per_row_agreement": agree, "agreement_ok": agree >= AGREE_MIN,
            "tolerant": int(T[arm].sum()), "tolerant_ok": int(T[arm].sum()) >= TOLERANT_MIN,
            "PASS": lo <= int(per[arm].sum()) <= hi and agree >= AGREE_MIN and int(T[arm].sum()) >= TOLERANT_MIN}
    assert int((((1 - 0.5) * mS + 0.5 * mA) > 0).sum()) == 3        # the registered numbers are reproduced
    assert int((((1 - 0.9) * mS + 0.9 * mA) > 0).sum()) == 127

    # H1 / H1c: the audit session's P1-P3, restated for n = 192, plus my additive prediction
    h1p, h1cp = int(per["H1"].sum()), int(per["H1c"].sum())
    p1_h1 = "PASS" if h1p <= 5 else ("FALSIFIED" if h1p > 19 else "BETWEEN")
    p1_h1c = "SUFFICIENT" if h1cp >= 96 else ("NOT_SUFFICIENT" if h1cp <= 19 else "INCONCLUSIVE")
    d = T["H1"].astype(int) - T["SPARSE"].astype(int)
    rng = np.random.default_rng(20260924)
    boots = np.array([d[rng.integers(0, len(ids), len(ids))].sum() for _ in range(10000)])
    margin2 = 0.10 * int(T["SPARSE"].sum())
    b = int((S["ADD-100"] & ~S["H1"]).sum()); c = int((S["H1"] & ~S["ADD-100"]).sum())
    need3 = 0.30 * int(S["SPARSE"].sum())
    out["H1"] = {
        "P1": {"H1_periods": h1p, "H1_verdict": p1_h1, "H1c_periods": h1cp, "H1c_verdict": p1_h1c},
        "P2": {"diff_tolerant_H1_minus_SPARSE": int(d.sum()), "lower95_one_sided": float(np.percentile(boots, 5)),
               "margin": -margin2, "PASS": float(np.percentile(boots, 5)) > -margin2},
        "P3": {"strict_H1": int(S["H1"].sum()), "strict_ADD-100": int(S["ADD-100"].sum()), "gain": c - b,
               "required": need3, "p": mcnemar(b, c), "PASS": (c - b) >= need3 and mcnemar(b, c) < 0.05},
        "additive_prediction": {"H1c_ge_150": h1cp >= 150, "sum_periods": h1p + h1cp,
                                "sum_within_178_pm_25": abs(h1p + h1cp - 178) <= 25}}
    a.output.write_bytes((json.dumps(out, indent=1) + "\n").encode())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
