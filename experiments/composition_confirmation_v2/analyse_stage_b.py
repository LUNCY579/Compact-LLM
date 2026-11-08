"""CONFIRM-002 Stage B registered analysis. Written BEFORE any arm was scored on T_BANK_V4 or HSKILL_SOUP_V3.

Implements PREREG.md §5 (779cd431) as amended by AMENDMENT_1.md (both pre-data):
  Gates: G1 BASE <= 20%, G2 SPARSE >= 40%, G3 separation >= 30pp (T_BANK_V4, primary rule), which guard C1
         and C5; G_H: H25 - BASE boundary abstentions >= +17 with exact McNemar p < 0.05, which guards C3.
  Holm family at alpha 0.05 over the evaluable members of {C1, C2, C3, C4}:
    C1  T kept:          one-sided 95% paired-bootstrap lower bound of (TH25 - SPARSE) on T > -5% of SPARSE;
                         Holm p = P*(boot <= -margin). BOTH must hold.
    C2  H arrives:       TH25 - SPARSE boundary abstentions >= +17, exact McNemar p (Holm).
    C3  H kept:          lower 95% bound of P_H = (TH25_b - SPARSE_b) / (H25_b - BASE_b), paired row bootstrap,
                         > 0.75; Holm p = P*(P_H <= 0.75). Draws whose denominator is <= 0 count as <= 0.75.
    C4  no over-abstain: lower 95% bound of (TH25 - SPARSE) answerable attempts > -5 rows; Holm p = P*(boot <= -5).
  C5 (own alpha):        SPARSE - THFULL >= 10% of SPARSE on T, with exact McNemar p < 0.05.
Descriptive: C1 at a 10% margin; THFULL boundary/answerable; per-class boundary gains; answerable correct;
strict T scores; a question-core cluster bootstrap for C2 and C3 (core = user text lower-cased with digit runs
blanked). Bootstrap: 10,000 resamples, seed 20260925.
"""
from __future__ import annotations

import argparse, hashlib, json, math, re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
T_BANK = HERE / "T_BANK_V4.json"
T_SHA = "82f0a6423dd6b58b81319708e7fbda7c5880b9734742323ea11574ff354c93c0"
H_BANK = HERE / "HSKILL_SOUP_V3.jsonl"
H_SHA = "686df45b94fa000be8df8d15e1f026a7f74e22deadfa4aacda96f5c6244c8ad5"
SEED, BOOT, ALPHA = 20260925, 10_000, 0.05
C3_RATIO, C4_ROWS = 0.75, 5


def min_gain(n_boundary: int) -> int:
    """C2 and G_H: 5% of the boundary rows, rounded up (17 at 336; AMENDMENT_1 section 1 scales it on extension)."""
    return -(-5 * n_boundary // 100)


def prim(g, r):
    s = g.strip()
    return (s[:-1] if s.endswith(".") else s) == r


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(x, y):
    b = int((x & ~y).sum()); c = int((y & ~x).sum())
    return {"x": int(x.sum()), "y": int(y.sum()), "net": c - b, "lost": b, "gained": c, "p": mcnemar(b, c)}


def holm(ps):
    order = sorted(ps, key=ps.get); out, still = {}, True
    for i, k in enumerate(order):
        thr = ALPHA / (len(order) - i); still = still and ps[k] <= thr
        out[k] = {"p": ps[k], "threshold": thr, "significant": still}
    return out


# ---------------- decision functions (also used by Stage A's power projection) ----------------
def t_gates(base_t, spar_t):
    n = len(base_t)
    g1, g2, g3 = (2 * n) // 10, -(-4 * n // 10), -(-3 * n // 10)
    return {"G1": int(base_t.sum()) <= g1, "G2": int(spar_t.sum()) >= g2, "G3": int(spar_t.sum() - base_t.sum()) >= g3,
            "thresholds": [g1, g2, g3], "BASE": int(base_t.sum()), "SPARSE": int(spar_t.sum())}


def g_h(base_b, h25_b):
    p = paired(base_b, h25_b)
    need = min_gain(len(base_b))
    return {**p, "required": need, "PASS": p["net"] >= need and p["p"] < ALPHA}


def c1(spar_t, th_t, rng, boot=BOOT, margin_frac=0.05):
    d = th_t.astype(int) - spar_t.astype(int); n = len(d); m = margin_frac * int(spar_t.sum())
    bs = d[rng.integers(0, n, (boot, n))].sum(axis=1)
    return {"diff": int(d.sum()), "margin": -m, "lower95": float(np.percentile(bs, 5)), "p": float((bs <= -m).mean()),
            "bound_ok": float(np.percentile(bs, 5)) > -m}


def c2(spar_b, th_b):
    p = paired(spar_b, th_b)
    need = min_gain(len(spar_b))
    return {**p, "required": need, "gain_ok": p["net"] >= need}


def c3(base_b, spar_b, h25_b, th_b, rng, boot=BOOT):
    n = len(base_b); idx = rng.integers(0, n, (boot, n))
    num = th_b[idx].sum(axis=1).astype(float) - spar_b[idx].sum(axis=1)
    den = h25_b[idx].sum(axis=1).astype(float) - base_b[idx].sum(axis=1)
    ratio = np.where(den > 0, num / np.where(den > 0, den, 1), -np.inf)
    point_den = float(h25_b.sum() - base_b.sum())
    point = float(th_b.sum() - spar_b.sum()) / point_den if point_den > 0 else None
    return {"P_H": point, "lower95": float(np.percentile(ratio, 5)), "p": float((ratio <= C3_RATIO).mean()),
            "bound_ok": float(np.percentile(ratio, 5)) > C3_RATIO}


def c4(spar_a, th_a, rng, boot=BOOT):
    d = th_a.astype(int) - spar_a.astype(int); n = len(d)
    bs = d[rng.integers(0, n, (boot, n))].sum(axis=1)
    return {"diff": int(d.sum()), "lower95": float(np.percentile(bs, 5)), "p": float((bs <= -C4_ROWS).mean()),
            "bound_ok": float(np.percentile(bs, 5)) > -C4_ROWS}


def c5(spar_t, thf_t):
    p = paired(thf_t, spar_t)          # net = SPARSE - THFULL
    need = 0.10 * int(spar_t.sum())
    return {**p, "loss": p["net"], "required": need, "PASS": p["net"] >= need and p["p"] < ALPHA}


def confirmatory(T, Hb, Ha, gates_t, gate_h, rng, boot=BOOT):
    """T: dict arm -> bool array on T bank; Hb/Ha: arm -> bool arrays (boundary credited / answerable attempt)."""
    res, ps = {}, {}
    if gates_t["ALL"]:
        res["C1"] = c1(T["SPARSE"], T["TH25"], rng, boot=boot); ps["C1"] = res["C1"]["p"]
    else:
        res["C1"] = {"status": "REJECTED_BY_T_GATES"}
    res["C2"] = c2(Hb["SPARSE"], Hb["TH25"]); ps["C2"] = res["C2"]["p"]
    if gate_h["PASS"]:
        res["C3"] = c3(Hb["BASE"], Hb["SPARSE"], Hb["H25"], Hb["TH25"], rng, boot=boot); ps["C3"] = res["C3"]["p"]
    else:
        res["C3"] = {"status": "NOT_EVALUABLE_G_H_FAILED"}
    res["C4"] = c4(Ha["SPARSE"], Ha["TH25"], rng, boot=boot); ps["C4"] = res["C4"]["p"]
    H = holm(ps)
    for k in ps:
        res[k]["holm"] = H[k]
        cond = {"C1": res[k].get("bound_ok"), "C2": res[k].get("gain_ok"), "C3": res[k].get("bound_ok"), "C4": res[k].get("bound_ok")}[k]
        res[k]["PASS"] = bool(cond and H[k]["significant"])
    return res


# ---------------- loading ----------------
def load_t(files):
    raw = T_BANK.read_bytes(); assert hashlib.sha256(raw).hexdigest() == T_SHA, "T_BANK_V4 hash"
    ref = {r["id"]: r["reference"] for r in json.loads(raw)["rows"]}; ids = sorted(ref)
    gen = {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        assert d["bank_sha256"] == T_SHA and d["runtime"]["max_new_tokens"] == 48, f
        for a in d["arms"]:
            g = {p["id"]: p["generation"] for p in a["predictions"]}
            assert set(g) == set(ref) and a["arm"] not in gen, a["arm"]
            gen[a["arm"]] = g
    P = {a: np.array([prim(g[i], ref[i]) for i in ids]) for a, g in gen.items()}
    S = {a: np.array([g[i].strip() == ref[i] for i in ids]) for a, g in gen.items()}
    return ids, P, S


def load_h(files):
    raw = H_BANK.read_bytes(); assert hashlib.sha256(raw).hexdigest() == H_SHA, "HSKILL_SOUP_V3 hash"
    rows = {json.loads(l)["id"]: json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()}
    bid = sorted(i for i, r in rows.items() if r["row_type"] == "boundary")
    aid = sorted(i for i, r in rows.items() if r["row_type"] == "answerable")
    Hb, Ha, Hc = {}, {}, {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        assert d["bank_sha256"] == H_SHA and d["runtime"]["max_new_tokens"] == 48, f
        for a in d["arms"]:
            p = {x["id"]: x for x in a["predictions"]}
            assert set(p) == set(rows) and a["arm"] not in Hb, a["arm"]
            Hb[a["arm"]] = np.array([p[i]["credited"] for i in bid])
            Ha[a["arm"]] = np.array([p[i]["attempt"] for i in aid])
            Hc[a["arm"]] = np.array([p[i]["credited"] for i in aid])
    fam = np.array([rows[i]["family"] for i in bid])
    core = np.array([re.sub(r"\d+", "#", " ".join(m["content"] for m in rows[i]["prompt"] if m["role"] == "user").lower()) for i in bid])
    return Hb, Ha, Hc, fam, core


def cluster_boot(core, fn, rng, boot=2000):
    cores = np.unique(core); groups = [np.where(core == c)[0] for c in cores]
    vals = []
    for _ in range(boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        vals.append(fn(idx))
    vals = np.array([v for v in vals if v is not None and np.isfinite(v)])
    return {"n_cores": len(cores), "p5": float(np.percentile(vals, 5)) if len(vals) else None,
            "median": float(np.median(vals)) if len(vals) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t-runs", nargs="+", required=True)
    ap.add_argument("--h-runs", nargs="+", required=True)
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    ids, T, TS = load_t(a.t_runs)
    Hb, Ha, Hc, fam, core = load_h(a.h_runs)
    gt = t_gates(T["BASE"], T["SPARSE"]); gt["ALL"] = gt["G1"] and gt["G2"] and gt["G3"]
    gh = g_h(Hb["BASE"], Hb["H25"])
    res = {"instrument": "CONFIRM-002", "t_bank_sha256": T_SHA, "h_bank_sha256": H_SHA, "gates_T": gt, "gate_H": gh}
    if not a.gates_only:
        rng = np.random.default_rng(SEED)
        res["confirmatory"] = confirmatory(T, Hb, Ha, gt, gh, rng)
        res["C5_full_strength"] = c5(T["SPARSE"], T["THFULL"]) if gt["ALL"] else {"status": "REJECTED_BY_T_GATES"}
        res["descriptive"] = {
            "C1_at_10pct_margin": c1(T["SPARSE"], T["TH25"], np.random.default_rng(SEED), margin_frac=0.10),
            "T_primary": {k: int(v.sum()) for k, v in T.items()}, "T_strict": {k: int(v.sum()) for k, v in TS.items()},
            "boundary_abstain": {k: int(v.sum()) for k, v in Hb.items()}, "answerable_attempt": {k: int(v.sum()) for k, v in Ha.items()},
            "answerable_correct": {k: int(v.sum()) for k, v in Hc.items()},
            "per_class_TH25_minus_SPARSE": {f: int(Hb["TH25"][fam == f].sum() - Hb["SPARSE"][fam == f].sum()) for f in sorted(set(fam))},
            "cluster_C2_gain": cluster_boot(core, lambda i: float(Hb["TH25"][i].sum() - Hb["SPARSE"][i].sum()), np.random.default_rng(SEED)),
            "cluster_C3_ratio": cluster_boot(core, lambda i: (float(Hb["TH25"][i].sum() - Hb["SPARSE"][i].sum()) / d) if (d := float(Hb["H25"][i].sum() - Hb["BASE"][i].sum())) > 0 else None,
                                             np.random.default_rng(SEED))}
    a.output.write_bytes((json.dumps(res, indent=2) + "\n").encode())
    print(json.dumps(res, indent=2)[:6000])
    ok = gt["ALL"] or gh["PASS"]
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
