"""CONFIRM-001 pre-read power of the registered decision rule, from the EXPOSED V2 development data.

For each candidate bank size n, resample n rows with replacement from V2's clean rows (the 11
content-contaminated ids removed), then apply the registered rule exactly: gates, C1, C2, C3-a,
C3-b and Holm, using the functions imported from analyse_confirm.py. The output is the probability
that each limb passes IF the fresh bank behaves like V2 at its point estimate. Uncertainty in that
point estimate is NOT propagated, so these are optimistic for a true effect nearer the margin.

No CONFIRM-001 arm output is read. Deterministic (seed 1).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FDS = HERE.parent / "fresh_dual_skill_instrument_v1"
sys.path.insert(0, str(HERE))
from analyse_confirm import primary, mcnemar, holm  # registered functions, reused verbatim

V2_SHA = "fc094136a5b581cb9cc03393b1e5959ed8f69247638c81e37cf7f68fbf132e01"


def load():
    bank = {r["id"]: r for r in json.loads((FDS / "T_BANK_V2.json").read_text(encoding="utf-8"))["rows"]}
    bad = set(json.loads((FDS / "CONTENT_CONTAMINATED_ROWS.json").read_text(encoding="utf-8")))
    gens = {}
    for f in ("CALIBRATION_V2.json", "V2_ARMS.json"):
        d = json.loads((FDS / f).read_text(encoding="utf-8"))
        assert d["bank_sha256"] == V2_SHA, f
        assert d["runtime"]["max_new_tokens"] == 48, f
        for a in d["arms"]:
            gens[a["arm"]] = {p["id"]: p["generation"] for p in a["predictions"]}
    ids = sorted(i for i in bank if i not in bad)
    assert len(bad & set(bank)) == 11 and len(ids) == 181
    V = {k: np.array([primary(gens[k][i], bank[i]["reference"]) for i in ids])
         for k in ("BASE", "SPARSE", "AVG", "ADD-100")}
    return ids, V


def simulate(V, n, sims=2000, inner=2000, seed=1):
    rng = np.random.default_rng(seed)
    m_rows = len(V["BASE"])
    c = dict(gates=0, C1=0, C2=0, C3a=0, C3b=0, C3=0, all3=0)
    diffs, margins = [], []
    g1, g2, g3 = int(0.2 * n), -(-4 * n // 10), -(-3 * n // 10)     # <=20%, >=40%, >=30pp
    for _ in range(sims):
        idx = rng.integers(0, m_rows, n)
        B, S, A, D = (V[k][idx] for k in ("BASE", "SPARSE", "AVG", "ADD-100"))
        if not (B.sum() <= g1 and S.sum() >= g2 and S.sum() - B.sum() >= g3):
            continue
        c["gates"] += 1
        margin = 0.10 * S.sum()
        b1, c1 = int((S & ~A).sum()), int((A & ~S).sum())
        b2, c2 = int((A & ~D).sum()), int((D & ~A).sum())
        d = D.astype(int) - S.astype(int)
        boots = d[rng.integers(0, n, (inner, n))].sum(axis=1)
        H = holm({"C1": mcnemar(b1, c1), "C2": mcnemar(b2, c2), "C3": float((boots <= -margin).mean())})
        r1 = (b1 - c1) >= margin and H["C1"]["significant"]
        r2 = c2 > b2 and H["C2"]["significant"]
        r3a = np.percentile(boots, 5) > -margin
        r3b = H["C3"]["significant"]
        c["C1"] += r1; c["C2"] += r2; c["C3a"] += r3a; c["C3b"] += r3b
        c["C3"] += r3a and r3b; c["all3"] += r1 and r2 and r3a and r3b
        diffs.append(int(d.sum())); margins.append(-margin)
    return c, sims, diffs, margins


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[288, 864])
    a = ap.parse_args()
    ids, V = load()
    print(f"V2 clean rows {len(ids)}: " + ", ".join(f"{k} {int(v.sum())}" for k, v in V.items()))
    for n in a.n:
        c, sims, diffs, margins = simulate(V, n)
        g = c["gates"]
        print(f"--- n = {n}: P(gates) {g / sims:.3f}; given gates: " +
              ", ".join(f"P({k}) {c[k] / g:.3f}" for k in ("C1", "C2", "C3a", "C3b", "C3", "all3")))
        print(f"    ADD-100 - SPARSE: median {np.median(diffs):.0f} rows, 5-95% "
              f"[{np.percentile(diffs, 5):.0f}, {np.percentile(diffs, 95):.0f}]; margin median {np.median(margins):.1f}")


if __name__ == "__main__":
    main()
