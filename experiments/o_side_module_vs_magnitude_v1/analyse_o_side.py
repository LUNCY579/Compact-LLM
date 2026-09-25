"""OSIDE-001 analysis (PREREG.md sections 3-7): canary, denominator gate, the registered H1 vs ADD-O50
comparison with its null-cell calibration, and the descriptive O-side table.

Inputs: ROWS_MANIFEST.json, scores/<ARM>.json (score_o_tokens.py) and, for the canary only, the recorded
row-level run titan_opallo_skill_vector_v1/full_scale/O_REPLY_NLL.json. Output: O_SIDE_RESULT.json.
numpy only. `--canary-only ARM ...` is the driver's early stop (exit 3 on failure).

Readouts share one form: a per-row numerator U[r, arm] and a per-row weight N[r], aggregated per reply class,
so that a class-cluster bootstrap draw is a vector of class multiplicities w and a mean is (w @ S) / (w @ W).
  content (primary)   U = mean NLL over the row's content tokens, N = 1          (row-weighted, as P_O was)
  content, by token   U = summed content NLL,                     N = #content tokens
  final               U = NLL of the terminal token,              N = 1 on terminal rows, 0 elsewhere
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
MANIFEST = HERE / "ROWS_MANIFEST.json"
MANIFEST_SHA = "19834d340aff2df12dc5f2333b7e9354832f3c916989c1d9ab459986586dd63b"
RECORDED = EXP / "titan_opallo_skill_vector_v1" / "full_scale" / "O_REPLY_NLL.json"
SCORES = HERE / "scores"
OUT = HERE / "O_SIDE_RESULT.json"

CANARY_ARMS = ("BASE", "OPALLO", "SPARSE", "ADD-100")
SEEDS = ("SEED23", "SEED29", "SEED41")
ALL_ARMS = ("BASE", "OPALLO", "OPALLO-BF16", "SPARSE", "ADD-100", "ADD-O50", "H1", "H1c") + SEEDS
BOOT_B, BOOT_SEED = 10_000, 20260925
CAL_OUTER, CAL_INNER, CAL_SEED = 2000, 1000, 20260926
ALPHA = 0.05
ALPHA_GRID = (0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.002)
CAL_MAX_RATE = 0.075            # registered percentile interval kept if the null-cell rate is at most this
CANARY_TOL = 1e-5
CAST_FRAC = 0.05                # DG2: |OPALLO-BF16 - OPALLO| <= 5% of the content gain
SEED_SNR = 3.0                  # DG3b: content gain >= 3 x between-seed SD
Z_POWER = NormalDist().inv_cdf(0.80)
MDE_MULT = NormalDist().inv_cdf(1 - ALPHA / 2) + Z_POWER   # two-sided 5%, power 80%, normal approximation
READ_A, READ_B = 0.85, 0.60     # FOLLOWUP_PROPOSAL P4 readings on P_O^content(H1)


# ---------------------------------------------------------------- readouts and the class-cluster bootstrap
class Readout:
    """Per-class sums of a per-row numerator for several arms, over an optional row mask."""

    def __init__(self, U: dict, N: np.ndarray, cls_index: np.ndarray, K: int, mask: np.ndarray | None = None):
        m = np.ones(len(N), bool) if mask is None else mask
        self.arms = list(U)
        self.K = K
        self.S = np.stack([np.bincount(cls_index[m], weights=np.asarray(U[a], float)[m], minlength=K)
                           for a in self.arms], axis=1)                          # K x A
        self.W = np.bincount(cls_index[m], weights=np.asarray(N, float)[m], minlength=K)
        if (self.W <= 0).any():
            raise ValueError("a class has no weight under this mask")

    def means(self, w: np.ndarray | None = None, S=None, W=None) -> dict:
        S = self.S if S is None else S
        W = self.W if W is None else W
        if w is None:
            return dict(zip(self.arms, S.sum(0) / W.sum()))
        M = (w @ S) / (w @ W)[:, None]
        return {a: M[:, j] for j, a in enumerate(self.arms)}

    def shifted(self, arm: str, delta: float) -> "Readout":
        """Every row of `arm` moved by delta (row-weighted readouts only: N = 1 per row)."""
        c = object.__new__(Readout)
        c.arms, c.K, c.W, c.S = self.arms, self.K, self.W, self.S.copy()
        c.S[:, self.arms.index(arm)] += delta * self.W
        return c


def draws(rng, K: int, B: int) -> np.ndarray:
    return rng.multinomial(K, np.full(K, 1.0 / K), size=B).astype(float)


def P(M: dict, arm: str):
    return (M["BASE"] - M[arm]) / (M["BASE"] - M["OPALLO"])


def D_stat(M: dict, a1: str = "H1", a2: str = "ADD-O50"):
    return P(M, a1) - P(M, a2)


def boot(readout: Readout, fn, seed: int = BOOT_SEED, B: int = BOOT_B, alpha: float = ALPHA) -> dict:
    rng = np.random.default_rng(seed)
    x = fn(readout.means(draws(rng, readout.K, B)))
    lo, hi = np.quantile(x, [alpha / 2, 1 - alpha / 2])
    return {"point": float(fn(readout.means())), "ci": [float(lo), float(hi)], "se": float(x.std(ddof=1)),
            "level": 1 - alpha, "B": B, "seed": seed, "_draws": x}


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def null_cell(readout: Readout, a1: str = "H1", a2: str = "ADD-O50", outer: int = CAL_OUTER, inner: int = CAL_INNER,
              seed: int = CAL_SEED) -> dict:
    """Coverage of the percentile interval when the true D is exactly 0 but the class heterogeneity is the observed
    one: a1's rows are shifted so that the pseudo-population has P(a1) = P(a2); samples of K classes are drawn
    from it, each gets its own percentile interval, and we count how often 0 is excluded."""
    M = readout.means()
    pop = readout.shifted(a1, float(M[a2] - M[a1]))
    d0 = float(D_stat(pop.means(), a1, a2))
    rng = np.random.default_rng(seed)
    K = readout.K
    excl = np.zeros(len(ALPHA_GRID))
    for _ in range(outer):
        idx = rng.integers(0, K, K)
        S_s, W_s = pop.S[idx], pop.W[idx]
        d = D_stat(pop.means(draws(rng, K, inner), S_s, W_s), a1, a2)
        for j, al in enumerate(ALPHA_GRID):
            lo, hi = np.quantile(d, [al / 2, 1 - al / 2])
            excl[j] += (lo > 0) or (hi < 0)
    rates = excl / outer
    ok = [al for al, r in zip(ALPHA_GRID, rates) if r <= 0.05]
    return {"population_D": d0, "outer": outer, "inner": inner, "seed": seed,
            "exclusion_rate_by_alpha": {str(al): float(r) for al, r in zip(ALPHA_GRID, rates)},
            "rate_at_registered_alpha": float(rates[0]),
            "registered_interval_calibrated": bool(rates[0] <= CAL_MAX_RATE),
            "calibrated_alpha": (max(ok) if ok else None)}


def outcome(ci) -> str:
    lo, hi = ci
    if lo > 0:
        return "MODULE_ROUTE_BETTER"
    if hi < 0:
        return "MODULE_ROUTE_WORSE"
    return "NULL_MODULE_ROUTE_BUYS_NOTHING"


# ---------------------------------------------------------------- inputs
def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST, check: bool = True) -> dict:
    if check and sha256(path) != MANIFEST_SHA:
        raise SystemExit("ROWS_MANIFEST.json does not match its registered hash")
    return json.loads(path.read_text(encoding="utf-8"))


def row_arrays(score: dict, man: dict) -> dict:
    """content mean, content sum, content count, final NLL and full mean per row, in manifest order."""
    rows = man["rows"]
    if [r["id"] for r in score["rows"]] != [m["id"] for m in rows]:
        raise SystemExit(f"{score['arm']}: row order differs from the manifest")
    n = len(rows)
    out = {k: np.zeros(n) for k in ("c_mean", "c_sum", "c_n", "final", "full")}
    for i, (s, m) in enumerate(zip(score["rows"], rows)):
        v = np.asarray(s["nll"], float)
        if len(v) != m["n_tokens"]:
            raise SystemExit(f"{score['arm']} row {m['id']}: {len(v)} tokens, manifest says {m['n_tokens']}")
        c = v[:-1] if m["terminal"] else v
        out["c_mean"][i], out["c_sum"][i], out["c_n"][i] = c.mean(), c.sum(), len(c)
        out["final"][i] = v[-1] if m["terminal"] else 0.0
        out["full"][i] = v.mean()
    return out


def canary(scores: dict, recorded: dict, arms) -> dict:
    rec = {x["arm"]: {r["id"]: r for r in x["rows_detail"]} for x in recorded["results"]}
    out = {"tolerance": CANARY_TOL, "arms": {}}
    for a in arms:
        s, r = scores[a], rec[a]
        ids_equal = sorted(x["id"] for x in s["rows"]) == sorted(r)
        diffs, internal, tok_bad = [], [], 0
        for x in s["rows"]:
            y = r.get(x["id"])
            if y is None:
                continue
            diffs.append(abs(x["mean_nll_gpu"] - y["mean_nll"]))
            internal.append(abs(float(np.mean(x["nll"])) - x["mean_nll_gpu"]))
            tok_bad += int(len(x["nll"]) != y["tokens"])
        mx, mi = (max(diffs) if diffs else float("inf")), (max(internal) if internal else float("inf"))
        out["arms"][a] = {"rows_compared": len(diffs), "ids_equal": ids_equal, "token_count_mismatches": tok_bad,
                          "max_abs_diff_vs_recorded": mx, "max_abs_internal_mean_diff": mi,
                          "PASS": bool(ids_equal and tok_bad == 0 and mx <= CANARY_TOL and mi <= CANARY_TOL)}
    out["runtime_recorded"] = recorded["runtime"]
    out["PASS"] = all(v["PASS"] for v in out["arms"].values())
    return out


# ---------------------------------------------------------------- the registered analysis
def analyse(man: dict, scores: dict, recorded: dict) -> dict:
    res = {"schema": 1, "experiment": "OSIDE-001", "manifest_sha256": MANIFEST_SHA,
           "score_files": {a: scores[a]["_sha256"] for a in scores}, "quality_claim": False}
    res["canary"] = canary(scores, recorded, CANARY_ARMS)
    if not res["canary"]["PASS"]:
        res["status"] = "HARNESS_FAILURE"
        return res
    rows = man["rows"]
    classes = sorted({r["reply_class"] for r in rows})
    ci = np.array([classes.index(r["reply_class"]) for r in rows])
    K = len(classes)
    term = np.array([r["terminal"] for r in rows], float)
    rep_free = ~np.array([r["replicate_consumed"] for r in rows])
    arr = {a: row_arrays(scores[a], man) for a in ALL_ARMS}
    c_n = arr["BASE"]["c_n"]
    content = Readout({a: arr[a]["c_mean"] for a in ALL_ARMS}, np.ones(len(rows)), ci, K)
    by_token = Readout({a: arr[a]["c_sum"] for a in ALL_ARMS}, c_n, ci, K)
    final = Readout({a: arr[a]["final"] for a in ALL_ARMS}, term, ci, K)
    content_r = Readout({a: arr[a]["c_mean"] for a in ALL_ARMS}, np.ones(len(rows)), ci, K, mask=rep_free)
    M = content.means()

    # ---- denominator gate
    g1 = boot(content, lambda m: m["BASE"] - m["OPALLO"])
    gain = g1["point"]
    cast = abs(float(M["OPALLO-BF16"] - M["OPALLO"]))
    Mr = content_r.means()
    seed_gain = {s: strip(boot(content_r, lambda m, s=s: m["BASE"] - m[s])) for s in SEEDS}
    gains_r = [float(Mr["BASE"] - Mr["OPALLO"])] + [float(Mr["BASE"] - Mr[s]) for s in SEEDS]
    sd_seed = float(np.std(gains_r, ddof=1))
    gate = {
        "DG1_content_gain": {**strip(g1), "PASS": bool(g1["ci"][0] > 0)},
        "DG2_cast": {"abs_diff": cast, "limit": CAST_FRAC * gain, "fraction_of_gain": cast / gain if gain else None,
                     "PASS": bool(gain > 0 and cast <= CAST_FRAC * gain)},
        "DG3a_every_replicate_gains": {"rows": int(rep_free.sum()), "by_seed": seed_gain,
                                       "PASS": all(v["ci"][0] > 0 for v in seed_gain.values())},
        "DG3b_gain_vs_seed_sd": {"rows": int(rep_free.sum()), "gains_seed17_23_29_41": gains_r, "sd": sd_seed,
                                 "ratio": gains_r[0] / sd_seed if sd_seed > 0 else float("inf"), "min_ratio": SEED_SNR,
                                 "PASS": bool(gains_r[0] >= SEED_SNR * sd_seed)},
    }
    gate["PASS"] = all(v["PASS"] for v in gate.values())
    res["denominator_gate"] = gate

    # ---- the registered comparison (always computed; the verdict is issued only behind the gate)
    d = boot(content, D_stat)
    cal = null_cell(content)
    if cal["registered_interval_calibrated"]:
        interval, basis = d["ci"], "registered 95% class-cluster percentile interval (null-cell rate within limit)"
    elif cal["calibrated_alpha"] is not None:
        al = cal["calibrated_alpha"]
        interval = [float(x) for x in np.quantile(d["_draws"], [al / 2, 1 - al / 2])]
        basis = f"calibrated percentile interval at level {1 - al} (the 95% interval's null-cell rate exceeded {CAL_MAX_RATE})"
    else:
        interval, basis = None, "no grid level reaches a 5% null-cell rate"
    al_used = ALPHA if cal["registered_interval_calibrated"] else cal["calibrated_alpha"]
    mde_dec = None if al_used is None else (NormalDist().inv_cdf(1 - al_used / 2) + Z_POWER) * d["se"]
    comp = {"D_point": d["point"], "ci95_percentile": d["ci"], "se": d["se"], "mde": MDE_MULT * d["se"],
            "mde_at_decision_level": mde_dec, "decision_alpha": al_used,
            "calibration": cal, "decision_interval": interval, "decision_basis": basis,
            "P_content_H1": strip(boot(content, lambda m: P(m, "H1"))),
            "P_content_ADD-O50": strip(boot(content, lambda m: P(m, "ADD-O50")))}
    if not gate["PASS"]:
        comp["verdict"] = "NOT_ISSUED_DENOMINATOR_GATE_FAILED"
    elif interval is None:
        comp["verdict"] = "INDETERMINATE_UNCALIBRATABLE_INTERVAL"
    else:
        comp["verdict"] = outcome(interval)
    res["registered_comparison"] = comp

    # ---- secondary, descriptive
    ph1 = comp["P_content_H1"]["point"]
    desc = {
        "P4_reading_on_H1": ("A_content_carried_by_MLP" if ph1 >= READ_A else
                             "B_mostly_attention_carried_style" if ph1 <= READ_B else "inconclusive"),
        "raw_content_nats_ADD-O50_minus_H1": strip(boot(content, lambda m: m["ADD-O50"] - m["H1"])),
        "content_M": {a: float(M[a]) for a in ALL_ARMS},
        "P_content": {a: strip(boot(content, lambda m, a=a: P(m, a))) for a in ALL_ARMS if a not in ("BASE", "OPALLO")},
        "P_final": {a: strip(boot(final, lambda m, a=a: P(m, a))) for a in ALL_ARMS if a not in ("BASE", "OPALLO")},
        "final_M": {a: float(v) for a, v in final.means().items()},
        "P_content_by_token": {a: strip(boot(by_token, lambda m, a=a: P(m, a))) for a in ("H1", "ADD-O50", "H1c", "ADD-100", "SPARSE")},
        "D_by_token": strip(boot(by_token, D_stat)),
        "additivity_H1_plus_H1c": strip(boot(content, lambda m: P(m, "H1") + P(m, "H1c"))),
        "specificity_SPARSE_P_content": strip(boot(content, lambda m: P(m, "SPARSE"))),
        "D_with_seed_mean_denominator": float((Mr["H1"] - Mr["ADD-O50"]) * -1 / np.mean(gains_r)),
        "full_mean_M": {a: float(arr[a]["full"].mean()) for a in ALL_ARMS},
    }
    res["descriptive"] = desc
    res["status"] = "VERDICT" if gate["PASS"] and interval is not None else "NO_VERDICT"
    return res


def load_scores(arms) -> dict:
    out = {}
    for a in arms:
        p = SCORES / f"{a}.json"
        out[a] = json.loads(p.read_text(encoding="utf-8"))
        out[a]["_sha256"] = sha256(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canary-only", nargs="+", metavar="ARM")
    ap.add_argument("--output", type=Path, default=OUT)
    a = ap.parse_args()
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))
    if a.canary_only:
        c = canary(load_scores(a.canary_only), recorded, a.canary_only)
        print(json.dumps({k: v for k, v in c.items() if k != "runtime_recorded"}, indent=2))
        return 0 if c["PASS"] else 3
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    res = analyse(load_manifest(), load_scores(ALL_ARMS), recorded)
    a.output.write_bytes((json.dumps(res, indent=2) + "\n").encode("utf-8"))
    comp = res.get("registered_comparison", {})
    print(json.dumps({"status": res["status"], "canary": res["canary"]["PASS"],
                      "gate": res.get("denominator_gate", {}).get("PASS"), "verdict": comp.get("verdict"),
                      "D": comp.get("D_point"), "interval": comp.get("decision_interval"), "mde": comp.get("mde")}, indent=2))
    return 0 if res["status"] != "HARNESS_FAILURE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
