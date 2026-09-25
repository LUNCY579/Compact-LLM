"""OSIDE-001 offline tests: no GPU, no score of any registered arm.

Synthetic data prove that every registered outcome and every gate limb can both fail and pass, that the canary
catches a 1e-4 drift, and that the content/final split drops exactly the terminal token. Two tests read the real
source files (headers and first chunks only) to show SeekST decodes exactly what verify_independent.ST decodes.
Run: python -B test_o_side_offline.py
"""
from __future__ import annotations

import json, sys, traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyse_o_side as A  # noqa: E402

K, PER_CLASS = 24, 60
SYN_SHA = {**A.REGISTERED_SHA, "SEED23": "23" * 32, "SEED29": "29" * 32, "SEED41": "41" * 32}
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
    man = {"rows": rows, **{k: k[:6] * 8 for k in A.INPUT_KEYS}}
    rt = dict(A.REGISTERED_RUNTIME)
    scores = {a: {"arm": a, "_sha256": "synthetic", "weight_sha256": SYN_SHA[a], "cast_bf16": a == "OPALLO-BF16",
                  "limit": None, "runtime": dict(rt), "input_sha256": {k: man[k] for k in A.INPUT_KEYS},
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


# ---------------------------------------------------------------- AMENDMENT_1 guards (each must fail AND pass)
def test_a1_nan_never_becomes_a_verdict():
    man, scores, rec = synth(p={"H1": 0.9, "ADD-O50": 0.6})
    scores["H1"]["rows"][5]["nll"][0] = float("nan")
    try:
        A.analyse(man, scores, rec)
        raise AssertionError("a NaN in a contrast arm was analysed")
    except SystemExit as e:
        assert "non-finite" in str(e)
    assert A.outcome([float("nan"), float("nan")]) == "INDETERMINATE_NONFINITE"


def test_a1_canary_sees_nan_anywhere():
    man, scores, rec = synth()
    scores["OPALLO"]["rows"][1234 % len(scores["OPALLO"]["rows"])]["mean_nll_gpu"] = float("nan")
    c = A.canary(scores, rec, A.CANARY_ARMS)
    assert not c["arms"]["OPALLO"]["PASS"] and not c["PASS"]
    man, scores, rec = synth()
    scores["BASE"]["rows"][7]["nll"][1] = float("inf")
    assert not A.canary(scores, rec, A.CANARY_ARMS)["arms"]["BASE"]["PASS"]


def test_a1_provenance_fails_and_passes():
    man, scores, rec = synth()
    ok = A.provenance(scores, man, SYN_SHA)
    assert ok["PASS"], ok
    cases = [("H1", lambda s: s["runtime"].__setitem__("torch", "2.7.1")),
             ("ADD-O50", lambda s: s.__setitem__("weight_sha256", "0" * 64)),
             ("OPALLO-BF16", lambda s: s.__setitem__("cast_bf16", False)),
             ("SEED29", lambda s: s["input_sha256"].__setitem__("source_train_sha256", "f" * 64)),
             ("H1c", lambda s: s.__setitem__("limit", 30)),
             ("SPARSE", lambda s: s["runtime"].__setitem__("device", "another GPU"))]
    for arm, mutate in cases:
        man, scores, rec = synth()
        mutate(scores[arm])
        prov = A.provenance(scores, man, SYN_SHA)
        assert not prov["PASS"], arm
        r = A.analyse(man, scores, rec, prov=prov)
        assert r["status"] == "HARNESS_FAILURE" and "registered_comparison" not in r, arm


def test_a1_p4_label_needs_the_gate():
    bad = run(p={"SEED23": 0.2, "SEED29": 1.8, "SEED41": 1.0, "H1": 0.9})
    assert not bad["denominator_gate"]["PASS"]
    assert bad["descriptive"]["P4_reading_on_H1"] == "UNDEFINED_DENOMINATOR_GATE_FAILED"
    ok = run(p={"H1": 0.9})
    assert ok["denominator_gate"]["PASS"] and ok["descriptive"]["P4_reading_on_H1"] == "A_content_carried_by_MLP"


def test_a1_by_token_covers_every_arm():
    r = run()
    want = {a for a in A.ALL_ARMS if a not in ("BASE", "OPALLO")}
    assert set(r["descriptive"]["P_content_by_token"]) == want == set(r["descriptive"]["P_content"])


def _sandbox():
    """run_o_side with SCORES/FAILED/ARMS (and analyse's HERE) redirected into a temp dir, wait_ram stubbed, and a
    stub run() that really creates what each step creates: a build dir with BUILD.json + weights, a score file with
    the scorer's fields over all manifest rows (sb.defect makes the next score file fail provenance), a deletion."""
    import tempfile, types
    import run_o_side as R
    import build_o_arms as B
    tmp = Path(tempfile.mkdtemp(prefix="oside_sbx_"))
    old = (R.SCORES, R.FAILED, B.ARMS, R.run, R.wait_ram, A.HERE)
    R.SCORES, R.FAILED, B.ARMS, A.HERE = tmp / "scores", tmp / "scores_failed", tmp / "arms", tmp
    R.SCORES.mkdir(); B.ARMS.mkdir()
    man = A.load_manifest()
    sb = types.SimpleNamespace(R=R, B=B, tmp=tmp, calls=[], defect=None)

    def stub_run(cmd, ok=(0,)):
        c = [str(x) for x in cmd]
        sb.calls.append(c)
        j = " ".join(c)
        if "build_o_arms.py" in j and ("rebuild" in c or "seed" in c):
            arm = c[-1] if "rebuild" in c else f"SEED{c[-1]}"
            d = B.ARMS / arm
            d.mkdir()
            (d / "model.safetensors").write_bytes(b"w")
            sha = A.REGISTERED_SHA.get(arm, arm.lower() * 8)
            (d / "BUILD.json").write_bytes(json.dumps({"weight_sha256": sha, "match": True}).encode())
        elif "build_o_arms.py" in j and "delete" in c:
            d = B.ARMS / c[-1]
            (d / "model.safetensors").unlink()
            (d / "WEIGHTS_DELETED.json").write_bytes(b"{}")
        elif "score_o_tokens.py" in j:
            arm, exp, out = c[c.index("--arm") + 1], c[c.index("--expect-sha") + 1], Path(c[c.index("--output") + 1])
            rows = [{"id": m["id"], "nll": [1.0] * m["n_tokens"], "mean_nll_gpu": 1.0} for m in man["rows"]]
            if sb.defect == "rows":
                rows = rows[:5]
            res = {"arm": arm, "weight_sha256": exp, "cast_bf16": "--cast-bf16" in c, "limit": None,
                   "runtime": dict(A.REGISTERED_RUNTIME), "input_sha256": {k: man[k] for k in A.INPUT_KEYS}, "rows": rows}
            if sb.defect == "device":
                res["runtime"]["device"] = "another GPU"
            sb.defect = None
            out.write_bytes(json.dumps(res).encode())
        return 0

    R.run = stub_run
    R.wait_ram = lambda *a, **k: None

    def restore():
        R.SCORES, R.FAILED, B.ARMS, R.run, R.wait_ram, A.HERE = old
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    sb.restore = restore
    return sb


def _kinds(calls):
    out = []
    for c in calls:
        j = " ".join(c)
        out.append("score" if "score_o_tokens.py" in j else "delete" if "delete" in c else
                   "build" if "build_o_arms.py" in j else "other")
    return out


def test_a1_driver_incomplete_score_is_moved_not_trusted():
    sb = _sandbox()
    try:
        (sb.R.SCORES / "BASE.json").write_bytes(b"\x00" * 100)          # zero-filled after a power loss
        assert sb.R.score_problems("BASE", A.REGISTERED_SHA["BASE"])[0].startswith("unreadable")
        sb.R.score("BASE", Path("x"), A.REGISTERED_SHA["BASE"])
        moved = list(sb.R.FAILED.rglob("BASE.json"))
        assert len(moved) == 1 and _kinds(sb.calls) == ["score"]
        assert sb.R.score_problems("BASE", A.REGISTERED_SHA["BASE"]) == []   # the rescored file is complete
    finally:
        sb.restore()


def test_a1_driver_canary_failure_moves_stage_files():
    sb = _sandbox()
    try:
        for a in ("BASE", "OPALLO"):
            (sb.R.SCORES / f"{a}.json").write_bytes(b"{}")
        sb.R.run = lambda cmd, ok=(0,): 3
        try:
            sb.R.canary_stage(("BASE", "OPALLO"), ("BASE", "OPALLO"))
            raise AssertionError("canary failure did not stop the run")
        except SystemExit as e:
            assert "canary FAILED" in str(e)
        assert not list(sb.R.SCORES.iterdir()) and len(list(sb.R.FAILED.rglob("*.json"))) == 2
        sb.R.run = lambda cmd, ok=(0,): 0
        sb.R.canary_stage(("OPALLO-BF16",), ("BASE",))                   # rc 0: nothing moves
    finally:
        sb.restore()


def test_a1_driver_rebuilt_arm_guards():
    sb = _sandbox()
    try:
        d = sb.B.ARMS / "H1"
        d.mkdir(); (d / "model.safetensors.partial").write_bytes(b"x")    # crash mid-build
        sb.R.built_arm("H1", ["rebuild", "H1"])
        assert _kinds(sb.calls) == ["build", "score", "delete"], _kinds(sb.calls)
        sc = [c for c in sb.calls if "score_o_tokens.py" in " ".join(c)][0]
        assert sc[sc.index("--expect-sha") + 1] == A.REGISTERED_SHA["H1"]
        assert (d / "WEIGHTS_DELETED.json").exists() and not (d / "model.safetensors").exists()
        sb.calls.clear()
        sb.R.built_arm("H1", ["rebuild", "H1"])                            # complete: nothing happens
        assert sb.calls == []
        (sb.R.SCORES / "H1.json").unlink()
        import shutil
        shutil.rmtree(d)
        d.mkdir(); (d / "model.safetensors").write_bytes(b"x")
        (d / "BUILD.json").write_bytes(json.dumps({"weight_sha256": "0" * 64, "match": False}).encode())
        try:
            sb.R.built_arm("H1", ["rebuild", "H1"])
            raise AssertionError("a mismatched rebuild was scored")
        except SystemExit as e:
            assert "does not certify" in str(e)
        (d / "REBUILD_MISMATCH.json").write_bytes(b"{}")
        try:
            sb.R.built_arm("H1", ["rebuild", "H1"])
            raise AssertionError("REBUILD_MISMATCH was ignored")
        except SystemExit as e:
            assert "REBUILD_MISMATCH" in str(e)
    finally:
        sb.restore()


def test_a1b_torn_build_json_is_rebuilt_not_a_crash():
    sb = _sandbox()
    try:
        for arm, cmd, torn in (("H1", ["rebuild", "H1"], b""), ("SEED23", ["seed", "23"], b"\x00" * 536)):
            d = sb.B.ARMS / arm
            d.mkdir(); (d / "model.safetensors").write_bytes(b"x"); (d / "BUILD.json").write_bytes(torn)
            if arm == "SEED23":
                assert A.expected_hashes()["SEED23"] is None and sb.R.expected_sha("SEED23") is None
            sb.calls.clear()
            sb.R.built_arm(arm, cmd)
            assert _kinds(sb.calls) == ["build", "score", "delete"], (arm, _kinds(sb.calls))
        assert A.expected_hashes()["SEED23"] == "seed23" * 8
    finally:
        sb.restore()


def test_a1b_fresh_score_checked_before_delete():
    for defect in ("rows", "device"):
        sb = _sandbox()
        try:
            sb.defect = defect
            try:
                sb.R.built_arm("SEED29", ["seed", "29"])
                raise AssertionError(f"{defect}: a failing fresh score was accepted")
            except SystemExit as e:
                assert "fails provenance" in str(e), str(e)
            d = sb.B.ARMS / "SEED29"
            assert (d / "model.safetensors").exists() and "delete" not in _kinds(sb.calls)   # weights kept
            sb.calls.clear()
            sb.R.built_arm("SEED29", ["seed", "29"])                       # the permitted rerun
            assert _kinds(sb.calls) == ["score", "delete"], _kinds(sb.calls)
            assert len(list(sb.R.FAILED.rglob("SEED29.json"))) == 1
        finally:
            sb.restore()


def test_a1_driver_imports_no_torch():
    import subprocess
    code = ("import sys; sys.path.insert(0, r'%s'); import run_o_side; "
            "print('torch' in sys.modules)") % str(HERE)
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=str(HERE))
    assert out.returncode == 0 and out.stdout.strip() == "False", (out.stdout, out.stderr[-500:])


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
