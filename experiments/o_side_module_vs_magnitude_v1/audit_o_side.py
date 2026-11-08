"""OSIDE-001 independent audit (post-run). It recomputes the registered quantities from the raw score files WITHOUT
calling analyse_o_side's statistics: its own content split, its own class-resampling bootstrap (explicit class
indices, a different RNG and seed), its own null cell (different seed and sizes), a provenance re-check of every score
file against the registered/BUILD hashes, and a per-class breakdown of the contrast. Output: AUDIT_O_SIDE.json.
CPU only; reads files only."""
from __future__ import annotations

import hashlib, json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ("BASE", "OPALLO", "OPALLO-BF16", "SPARSE", "ADD-100", "ADD-O50", "H1", "H1c", "SEED23", "SEED29", "SEED41")
REG = {"BASE": "75b0a92cd28eb90d121d72aa7618c497a6ff1b4522b73bdf01cabf515c30ad93",
       "OPALLO": "5ff1c9a5cf3674ace09eaf33bfc21431ea63718cff3cbf5b52701fd3de1416f8",
       "OPALLO-BF16": "5ff1c9a5cf3674ace09eaf33bfc21431ea63718cff3cbf5b52701fd3de1416f8",
       "SPARSE": "6ae85f5bdab9437edc06b20716e6e7008d8aa402ec95a0a65ebc7bf8da1e94ed",
       "ADD-100": "95a7576725721e8143c1be8b21448326d6d87dc69087da39ea378c47fadf0d5b",
       "ADD-O50": "bc69a4ec403b74fb09de815ec55738a7126da345802a7b45e520f205e3c68051",
       "H1": "3ff2aa596ccdc8234dae053362c1cabc8f5c3e274f3ea78591a9a7b219729fc1",
       "H1c": "ea46496c1a713c971589f3575532d0dd7f64e85cf7eaf8dd410c4b206c82043e"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    man = json.loads((HERE / "ROWS_MANIFEST.json").read_text(encoding="utf-8"))
    res = json.loads((HERE / "O_SIDE_RESULT.json").read_text(encoding="utf-8"))
    rows = man["rows"]
    n = len(rows)
    out = {"schema": 1, "experiment": "OSIDE-001", "what": "independent post-run audit", "problems": []}

    # ---- A. provenance and file identity
    prov = {}
    for a in ARMS:
        p = HERE / "scores" / f"{a}.json"
        s = json.loads(p.read_text(encoding="utf-8"))
        want = REG.get(a) or json.loads((HERE / "arms" / a / "BUILD.json").read_text(encoding="utf-8"))["weight_sha256"]
        ok = {"file_sha_matches_result": sha(p) == res["score_files"][a],
              "weight_sha_expected": s["weight_sha256"] == want,
              "cast_flag": bool(s["cast_bf16"]) == (a == "OPALLO-BF16"),
              "limit_none": s.get("limit") is None, "rows": len(s["rows"]) == n,
              "ids_in_order": [x["id"] for x in s["rows"]] == [m["id"] for m in rows],
              "runtime": (s["runtime"]["python"], s["runtime"]["torch"], s["runtime"]["transformers"], s["runtime"]["device"])
                         == ("3.11.15", "2.6.0+cu124", "5.14.1", "NVIDIA GeForce RTX 4070 Laptop GPU"),
              "inputs": all(s["input_sha256"][k] == man[k] for k in ("o_reply_nll_py_sha256", "source_train_sha256", "consumed_arm_dose_sha256"))}
        if a in ("ADD-O50", "H1", "H1c"):
            b = json.loads((HERE / "arms" / a / "BUILD.json").read_text(encoding="utf-8"))
            ok["rebuild_match_registered"] = b.get("match") is True and b["weight_sha256"] == REG[a]
        prov[a] = ok
        for k, v in ok.items():
            if not v:
                out["problems"].append(f"{a}: {k}")
        s["_c"] = None
        # ---- content / final per row, computed here from the per-token lists
        c = np.empty(n); f = np.full(n, np.nan)
        for i, (x, m) in enumerate(zip(s["rows"], rows)):
            v = np.asarray(x["nll"], dtype=np.float64)
            if m["terminal"]:
                c[i] = v[:-1].mean(); f[i] = v[-1]
            else:
                c[i] = v.mean()
        prov[a]["_content"], prov[a]["_final"] = c, f
    out["provenance"] = {a: {k: v for k, v in d.items() if not k.startswith("_")} for a, d in prov.items()}

    C = {a: prov[a]["_content"] for a in ARMS}
    classes = sorted({m["reply_class"] for m in rows})
    cls = np.array([classes.index(m["reply_class"]) for m in rows])
    members = [np.where(cls == k)[0] for k in range(len(classes))]

    def P(arm, idx):
        b, o, x = C["BASE"][idx].mean(), C["OPALLO"][idx].mean(), C[arm][idx].mean()
        return (b - x) / (b - o)

    allidx = np.arange(n)
    D = P("H1", allidx) - P("ADD-O50", allidx)
    reported = res["registered_comparison"]
    out["point_estimates"] = {"D": D, "D_reported": reported["D_point"], "P_H1": P("H1", allidx), "P_ADD-O50": P("ADD-O50", allidx),
                              "M_content": {a: float(C[a].mean()) for a in ARMS}}
    if abs(D - reported["D_point"]) > 1e-12:
        out["problems"].append(f"D differs from the reported value: {D} vs {reported['D_point']}")
    for a in ARMS:
        if abs(C[a].mean() - res["descriptive"]["content_M"][a]) > 1e-12:
            out["problems"].append(f"content mean of {a} differs from the reported value")

    # ---- B. an independent class bootstrap: explicit resampled class lists, PCG64 seed 7, 4000 draws
    rng = np.random.Generator(np.random.PCG64(7))
    K = len(classes)
    ds = []
    for _ in range(4000):
        pick = rng.integers(0, K, K)
        idx = np.concatenate([members[k] for k in pick])
        ds.append(P("H1", idx) - P("ADD-O50", idx))
    ds = np.array(ds)
    out["independent_bootstrap"] = {"draws": 4000, "seed": 7, "se": float(ds.std(ddof=1)),
                                    "ci95": [float(np.quantile(ds, .025)), float(np.quantile(ds, .975))],
                                    "ci98": [float(np.quantile(ds, .01)), float(np.quantile(ds, .99))],
                                    "reported_ci95": reported["ci95_percentile"], "reported_decision": reported["decision_interval"]}

    # ---- C. an independent null cell: shift H1 so the class-population D is 0; 600 outer x 600 inner, seed 11
    shift = C["ADD-O50"].mean() - C["H1"].mean()
    H1n = C["H1"] + shift
    base, opal, a50 = C["BASE"], C["OPALLO"], C["ADD-O50"]
    sums = np.array([[base[m].sum(), opal[m].sum(), H1n[m].sum(), a50[m].sum(), len(m)] for m in members])
    rng2 = np.random.Generator(np.random.PCG64(11))
    excl = {0.05: 0, 0.02: 0}
    OUT, IN = 600, 600
    for _ in range(OUT):
        samp = sums[rng2.integers(0, K, K)]
        w = rng2.multinomial(K, np.full(K, 1 / K), size=IN).astype(float)
        agg = w @ samp
        mb, mo, mh, ma = (agg[:, j] / agg[:, 4] for j in range(4))
        d = (mb - mh) / (mb - mo) - (mb - ma) / (mb - mo)
        for al in excl:
            lo, hi = np.quantile(d, [al / 2, 1 - al / 2])
            excl[al] += int(lo > 0 or hi < 0)
    out["independent_null_cell"] = {"outer": OUT, "inner": IN, "seed": 11,
                                    "rate": {str(k): v / OUT for k, v in excl.items()},
                                    "reported": reported["calibration"]["exclusion_rate_by_alpha"]}

    # ---- D. per-class breakdown and row signs of the contrast (content NLL: ADD-O50 minus H1; positive = H1 better)
    diff = C["ADD-O50"] - C["H1"]
    per_class = []
    for k, m in enumerate(members):
        per_class.append({"class": classes[k], "rows": int(len(m)), "mean_nats_ADD-O50_minus_H1": float(diff[m].mean()),
                          "rows_H1_better": int((diff[m] > 0).sum()), "D_class": float(P("H1", m) - P("ADD-O50", m))})
    per_class.sort(key=lambda r: r["D_class"])
    out["per_class"] = per_class
    out["row_signs"] = {"H1_better": int((diff > 0).sum()), "ADD-O50_better": int((diff < 0).sum()), "ties": int((diff == 0).sum())}
    out["classes_D_positive"] = sum(r["D_class"] > 0 for r in per_class)
    out["PASS"] = not out["problems"]
    (HERE / "AUDIT_O_SIDE.json").write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({k: out[k] for k in ("problems", "point_estimates", "independent_bootstrap", "independent_null_cell",
                                          "row_signs", "classes_D_positive", "PASS")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
