"""OSIDE-001 offline tests: no GPU, no score of any registered arm.

Synthetic data prove that every registered outcome and every gate limb can both fail and pass, that the canary
catches a 1e-4 drift, and that the content/final split drops exactly the terminal token. Two tests read the real
source files (headers and first chunks only) to show SeekST decodes exactly what verify_independent.ST decodes.
Run: python -B test_o_side_offline.py
"""
from __future__ import annotations

import sys, traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyse_o_side as A  # noqa: E402

K, PER_CLASS = 24, 60
P_DEFAULT = {"OPALLO": 1.0, "OPALLO-BF16": 1.0, "SPARSE": 0.15, "ADD-100": 0.97, "ADD-O50": 0.80, "H1": 0.80,
             "H1c": 0.30, "SEED23": 1.0, "SEED29": 1.0, "SEED41": 1.0}


def synth(p=None, tau=0.02, cast_offset=0.0, seed=0):
    """Per-token NLLs: arm X lowers each content token by p_X(class) * gain(class); the terminal token by p_X * 0.8."""
    p = {**P_DEFAULT, **(p or {})}
    rng = np.random.default_rng(seed)
    gain_k = 0.5 + 0.1 * rng.standard_normal(K)
    eps = {a: tau * rng.standard_normal(K) for a in p}
    rows, toks = [], {a: [] for a in ("BASE",) + tuple(p)}
    for k in range(K):
        for j in range(PER_CLASS):
            L = int(rng.integers(2, 11))
            term = bool(rng.random() < 0.97)
            rows.append({"id": f"r{k}_{j}", "reply_class": f"c{k:02d}", "n_tokens": L, "final_token_id": 13,
                         "terminal": term, "replicate_consumed": bool(rng.random() < 0.01)})
            base = 3.0 + 0.3 * rng.standard_normal(L)
            g = gain_k[k] + 0.05 * rng.standard_normal(L)
            toks["BASE"].append(base)
            for a, pa in p.items():
                pk = pa + (0.0 if a in ("OPALLO", "OPALLO-BF16") else eps[a][k])
                v = base - pk * g
                if term:
                    v[-1] = base[-1] - pk * 0.8
                if a == "OPALLO-BF16":
                    v = v + cast_offset
                toks[a].append(v)
    man = {"rows": rows}
    scores = {a: {"arm": a, "_sha256": "synthetic",
                  "rows": [{"id": r["id"], "nll": t.tolist(), "mean_nll_gpu": float(np.mean(t))} for r, t in zip(rows, toks[a])]}
              for a in toks}
    recorded = {"runtime": {"synthetic": True},
                "results": [{"arm": a, "rows_detail": [{"id": r["id"], "mean_nll": float(np.mean(t)), "tokens": len(t)}
                                                       for r, t in zip(rows, toks[a])]} for a in A.CANARY_ARMS]}
    return man, scores, recorded


def run(**kw):
    man, scores, recorded = synth(**kw)
    return A.analyse(man, scores, recorded)


def test_better():
    r = run(p={"H1": 0.90, "ADD-O50": 0.60})
    assert r["denominator_gate"]["PASS"], r["denominator_gate"]
    assert r["registered_comparison"]["verdict"] == "MODULE_ROUTE_BETTER", r["registered_comparison"]["verdict"]


def test_worse():
    r = run(p={"H1": 0.60, "ADD-O50": 0.90})
    assert r["registered_comparison"]["verdict"] == "MODULE_ROUTE_WORSE"


def test_null_reports_mde():
    r = run(p={"H1": 0.80, "ADD-O50": 0.80}, tau=0.05, seed=3)
    c = r["registered_comparison"]
    assert c["verdict"] == "NULL_MODULE_ROUTE_BUYS_NOTHING", (c["verdict"], c["decision_interval"])
    assert c["mde"] > 0 and c["mde_at_decision_level"] >= c["mde"] - 1e-12


def test_dg1_fails_without_a_gain():
    r = run(p={"OPALLO": 0.0, "OPALLO-BF16": 0.0})
    assert not r["denominator_gate"]["DG1_content_gain"]["PASS"]
    assert r["registered_comparison"]["verdict"] == "NOT_ISSUED_DENOMINATOR_GATE_FAILED"


def test_dg2_cast():
    man, scores, rec = synth()
    gain = float(np.mean([np.mean(np.asarray(s["nll"][:-1] if m["terminal"] else s["nll"]))
                          for s, m in zip(scores["BASE"]["rows"], man["rows"])]) -
                 np.mean([np.mean(np.asarray(s["nll"][:-1] if m["terminal"] else s["nll"]))
                          for s, m in zip(scores["OPALLO"]["rows"], man["rows"])]))
    ok = run(cast_offset=0.01 * gain)
    bad = run(cast_offset=0.10 * gain)
    assert ok["denominator_gate"]["DG2_cast"]["PASS"] and not bad["denominator_gate"]["DG2_cast"]["PASS"]
    assert bad["registered_comparison"]["verdict"] == "NOT_ISSUED_DENOMINATOR_GATE_FAILED"


def test_dg3a_replicate_without_gain():
    r = run(p={"SEED29": 0.0})
    assert not r["denominator_gate"]["DG3a_every_replicate_gains"]["PASS"]


def test_dg3b_seed_spread():
    bad = run(p={"SEED23": 0.2, "SEED29": 1.8, "SEED41": 1.0})
    assert bad["denominator_gate"]["DG3a_every_replicate_gains"]["PASS"]
    assert not bad["denominator_gate"]["DG3b_gain_vs_seed_sd"]["PASS"]
    ok = run(p={"SEED23": 0.9, "SEED29": 1.1, "SEED41": 1.0})
    assert ok["denominator_gate"]["DG3b_gain_vs_seed_sd"]["PASS"]


def test_canary_catches_drift():
    man, scores, rec = synth()
    rec["results"][1]["rows_detail"][5]["mean_nll"] += 1e-4          # OPALLO, one row
    r = A.analyse(man, scores, rec)
    assert r["status"] == "HARNESS_FAILURE" and not r["canary"]["arms"]["OPALLO"]["PASS"]
    man, scores, rec = synth()
    assert A.analyse(man, scores, rec)["canary"]["PASS"]


def test_content_split():
    man = {"rows": [{"id": "a", "n_tokens": 3, "terminal": True}, {"id": "b", "n_tokens": 2, "terminal": False}]}
    s = {"arm": "X", "rows": [{"id": "a", "nll": [1.0, 2.0, 9.0]}, {"id": "b", "nll": [4.0, 6.0]}]}
    r = A.row_arrays(s, man)
    assert r["c_mean"].tolist() == [1.5, 5.0] and r["c_n"].tolist() == [2, 2]
    assert r["final"].tolist() == [9.0, 0.0] and r["full"].tolist() == [4.0, 5.0]


def test_null_cell_detects_undercoverage_structure():
    man, scores, rec = synth(p={"H1": 0.8, "ADD-O50": 0.8}, tau=0.08)
    rows = man["rows"]
    ci = np.array([int(r["reply_class"][1:]) for r in rows])
    arr = {a: A.row_arrays(scores[a], man) for a in ("BASE", "OPALLO", "H1", "ADD-O50")}
    ro = A.Readout({a: v["c_mean"] for a, v in arr.items()}, np.ones(len(rows)), ci, K)
    cal = A.null_cell(ro, outer=400, inner=400)
    assert abs(cal["population_D"]) < 1e-9
    rates = list(cal["exclusion_rate_by_alpha"].values())
    assert all(rates[i] >= rates[i + 1] - 0.02 for i in range(len(rates) - 1)), rates   # wider interval, fewer exclusions


def test_seekst_matches_st_on_real_files():
    sys.path.insert(0, str(HERE.parent / "titan_model_soup_mini_v1" / "local-runs"))
    from verify_independent import ST
    from seekst import SeekST
    import build_o_arms as B
    for path in (B.SUPA, B.SV / "OPALLO" / "model.safetensors"):
        a, b = ST(path), SeekST(path)
        try:
            assert a.keys() == b.keys()
            for k in a.keys()[:3] + a.keys()[-3:]:
                assert a.info(k) == b.info(k)
                hi = min(4, a.info(k)[1][0])
                x, y = a.rows(k, 0, hi), b.rows(k, 0, hi)
                assert x.shape == y.shape and (x.view(np.uint32) == y.view(np.uint32)).all(), k
        finally:
            a.close(); b.close()


def test_outcome_boundaries():
    assert A.outcome([0.001, 0.2]) == "MODULE_ROUTE_BETTER"
    assert A.outcome([-0.2, -0.001]) == "MODULE_ROUTE_WORSE"
    assert A.outcome([-0.01, 0.01]) == "NULL_MODULE_ROUTE_BUYS_NOTHING"
    assert A.outcome([0.0, 0.1]) == "NULL_MODULE_ROUTE_BUYS_NOTHING"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in tests:
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                f()
            print(f"PASS {n}")
        except Exception:
            bad += 1
            print(f"FAIL {n}")
            traceback.print_exc()
    print(f"{len(tests) - bad}/{len(tests)} passed")
    raise SystemExit(1 if bad else 0)
