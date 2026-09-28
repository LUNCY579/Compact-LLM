"""CONFIRM-001 post-run audit. Independent of analyse_confirm.py: nothing is imported from it.

Written AFTER the registered analysis ran. It does not change any verdict. It checks the verdicts
four ways:
  A. recompute every registered number from the raw CONF_*.json predictions and compare it with
     CONFIRM_RESULT.json;
  B. test the robustness of C3 with a non-bootstrap interval;
  C. read the TEXT of the rows ADD-100 loses relative to SPARSE, and classify why;
  D. test whether the subgroup differences (bank part, family) exceed chance, as descriptive checks
     only.
Also E, provenance: bank hashes, runtime identity, row sets, preflight, build records, log order.
Writes AUDIT_INDEPENDENT.json. Read-only otherwise.
"""
from __future__ import annotations

import hashlib, json, math, re
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BANK = HERE / "T_BANK_V3X.json"
RUNS = ["CONF_GATES.json", "CONF_COMPOSE.json", "CONF_LOCATION.json", "CONF_RANDMASK.json"]
FAMS = ("ledger_replay", "prerequisite_plan", "register_machine")


def prim(g: str, ref: str) -> bool:
    s = g.strip()
    return (s[:-1] if s.endswith(".") else s) == ref


def strict(g: str, ref: str) -> bool:
    return g.strip() == ref


def binom_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    out = {"problems": []}
    raw = BANK.read_bytes()
    bank_sha = hashlib.sha256(raw).hexdigest()
    rows = json.loads(raw)["rows"]
    ref = {r["id"]: r["reference"] for r in rows}
    fam = {r["id"]: r["family"] for r in rows}
    copyb = {r["id"]: r["copy_baseline"] for r in rows}
    ids = sorted(ref)
    part = {i: ("V3" if i.startswith("CF1-") else "EXT") for i in ids}
    assert sum(v == "V3" for v in part.values()) == 288 and len(ids) == 864

    # ---------- E. provenance ----------
    gen, runtimes, declared = {}, {}, {}
    for f in RUNS:
        d = json.loads((HERE / f).read_text(encoding="utf-8"))
        if d["bank_sha256"] != bank_sha:
            out["problems"].append(f"{f}: bank_sha256 mismatch")
        runtimes[f] = d["runtime"]
        for a in d["arms"]:
            pid = [p["id"] for p in a["predictions"]]
            if len(pid) != 864 or len(set(pid)) != 864 or set(pid) != set(ids):
                out["problems"].append(f"{f}/{a['arm']}: row set is not exactly the bank")
            gen[a["arm"]] = {p["id"]: p["generation"] for p in a["predictions"]}
            declared[a["arm"]] = a["correct"]
    rt0 = json.dumps(runtimes[RUNS[0]], sort_keys=True)
    out["E_provenance"] = {
        "bank_sha256": bank_sha,
        "runtime_identical_across_files": all(json.dumps(r, sort_keys=True) == rt0 for r in runtimes.values()),
        "runtime": runtimes[RUNS[0]],
        "arms": sorted(gen),
    }
    pf = json.loads((HERE / "PREFLIGHT.json").read_text(encoding="utf-8"))
    out["E_provenance"]["preflight"] = {"torch": pf["torch"], "all_six_match": all(v["match"] for v in pf["arms"].values()),
                                        "n": len(pf["arms"])}
    bm = json.loads((HERE / "arms" / "RAND_MASK" / "BUILD.json").read_text(encoding="utf-8"))
    bs = json.loads((HERE / "arms" / "SCALAR_P" / "BUILD.json").read_text(encoding="utf-8"))
    out["E_provenance"]["builds"] = {
        "p_equal": bm["p"] == bs["p"], "p": bm["p"],
        "scalar_p_points_at_rand_mask": bs["rand_mask_weight_sha256"] == bm["weight_sha256"],
        "rand_mask_norm_rel_err": bm["relative_error"], "scalar_p_norm_rel_err": bs["relative_error"],
        "rand_mask_sha": bm["weight_sha256"], "scalar_p_sha": bs["weight_sha256"]}
    log = (HERE / "run_confirm.log").read_text(encoding="utf-8", errors="replace").splitlines()
    idx = lambda pat: next((k for k, l in enumerate(log) if pat in l), None)
    order = {"ALL_PASS": idx('"ALL_PASS": true'), "p0": idx("p0="), "verify_RAND_MASK": idx("verify RAND_MASK"),
             "scoring_compose": idx("scoring compose"), "released": idx("[gpu-lease] released"), "exit": idx("[wmi_runner] exit rc=0")}
    out["E_provenance"]["log_line_order"] = order
    vals = [order[k] for k in ("ALL_PASS", "p0", "verify_RAND_MASK", "scoring_compose", "released", "exit")]
    out["E_provenance"]["gates_read_before_any_build_and_all_steps_in_order"] = None not in vals and vals == sorted(vals)

    # ---------- A. recompute ----------
    P = {a: np.array([prim(gen[a][i], ref[i]) for i in ids]) for a in gen}
    S = {a: np.array([strict(gen[a][i], ref[i]) for i in ids]) for a in gen}
    for a in gen:
        if int(S[a].sum()) != declared[a]:
            out["problems"].append(f"{a}: strict {int(S[a].sum())} != scorer's own correct {declared[a]}")
    masks = {"ALL": np.ones(864, bool), "V3": np.array([part[i] == "V3" for i in ids]),
             "EXT": np.array([part[i] == "EXT" for i in ids])}
    counts = {m: {a: {"primary": int(P[a][mk].sum()), "strict": int(S[a][mk].sum())} for a in sorted(gen)} for m, mk in masks.items()}
    out["A_counts"] = counts

    def pair(x, y, mk):
        b = int((x & ~y & mk).sum()); c = int((y & ~x & mk).sum())
        return {"lost": b, "gained": c, "net": c - b, "p": binom_two_sided(b, c)}

    rng = np.random.default_rng(12345)
    res = {}
    for m, mk in (("ALL", masks["ALL"]), ("V3", masks["V3"])):
        n = int(mk.sum())
        c1 = pair(P["SPARSE"], P["AVG"], mk); c2 = pair(P["AVG"], P["ADD-100"], mk)
        sp = int(P["SPARSE"][mk].sum()); margin = 0.1 * sp
        d = (P["ADD-100"].astype(int) - P["SPARSE"].astype(int))[mk]
        boots = d[rng.integers(0, n, (10000, n))].sum(axis=1)
        g, l = int((d == 1).sum()), int((d == -1).sum())
        diff = g - l
        var = (g + l) - diff ** 2 / n
        se = math.sqrt(var)
        c3 = {"diff": diff, "margin": -margin, "retained": (sp + diff) / sp,
              "boot_p5": float(np.percentile(boots, 5)), "boot_p2_5": float(np.percentile(boots, 2.5)),
              "boot_P_le_margin": float((boots <= -margin).mean()),
              "normal_lower95_one_sided": diff - 1.6449 * se,
              "normal_p_one_sided": 0.5 * math.erfc(((diff + margin) / se) / math.sqrt(2)),
              "passes_under_bootstrap": float(np.percentile(boots, 5)) > -margin,
              "passes_under_normal": diff - 1.6449 * se > -margin,
              "retained_fraction_lower95_normal": (sp + diff - 1.6449 * se) / sp}
        ps = {"C1": c1["p"], "C2": c2["p"], "C3": c3["boot_P_le_margin"]}
        order_ = sorted(ps, key=ps.get)
        holm, still = {}, True
        for k_, key in enumerate(order_):
            thr = 0.05 / (3 - k_); still = still and ps[key] <= thr
            holm[key] = {"p": ps[key], "thr": thr, "sig": still}
        c1["PASS"] = (c1["lost"] - c1["gained"]) >= margin and holm["C1"]["sig"]
        c2["PASS"] = c2["net"] > 0 and holm["C2"]["sig"]
        c3["PASS"] = c3["passes_under_bootstrap"] and holm["C3"]["sig"]
        per = lambda a: int(sum(gen[a][i].strip().endswith(".") for i, k in zip(ids, mk) if k))
        c5 = pair(P["RAND_MASK"], P["SPARSE_ATTN"], mk); c6 = pair(P["SCALAR_P"], P["RAND_MASK"], mk)
        res[m] = {"n": n, "C1": c1, "C2": c2, "C3": c3, "holm": holm,
                  "C4": {"ADD-100": per("ADD-100"), "SPARSE": per("SPARSE"), "AVG": per("AVG")}, "C5": c5, "C6": c6}
    out["A_recomputed"] = res

    # compare with the registered result
    reg = json.loads((HERE / "CONFIRM_RESULT.json").read_text(encoding="utf-8"))
    cmp = []
    for m, key in (("ALL", "governing_result"), ("V3", "secondary_original_v3_288_descriptive")):
        g_ = reg[key]; r_ = res[m]; c = g_["confirmatory"]
        pairs_ = [
            ("C1 lost/gained", (c["C1_attenuation"]["lost"], c["C1_attenuation"]["gained"]), (r_["C1"]["lost"], r_["C1"]["gained"])),
            ("C1 PASS", c["C1_attenuation"]["PASS"], r_["C1"]["PASS"]),
            ("C2 lost/gained", (c["C2_add100_beats_avg"]["lost"], c["C2_add100_beats_avg"]["gained"]), (r_["C2"]["lost"], r_["C2"]["gained"])),
            ("C2 PASS", c["C2_add100_beats_avg"]["PASS"], r_["C2"]["PASS"]),
            ("C3 diff", c["C3_interference_small"]["diff_ADD_minus_SPARSE"], r_["C3"]["diff"]),
            ("C3 PASS", c["C3_interference_small"]["PASS"], r_["C3"]["PASS"]),
            ("C4 periods", (g_["C4_convention_descriptive"]["ADD-100_periods"], g_["C4_convention_descriptive"]["SPARSE_periods"], g_["C4_convention_descriptive"]["AVG_periods"]),
             (r_["C4"]["ADD-100"], r_["C4"]["SPARSE"], r_["C4"]["AVG"])),
            ("C5 lost/gained", (g_["C5_location"]["lost"], g_["C5_location"]["gained"]), (r_["C5"]["lost"], r_["C5"]["gained"])),
            ("C6 lost/gained", (g_["C6_mask_vs_expectation"]["lost"], g_["C6_mask_vs_expectation"]["gained"]), (r_["C6"]["lost"], r_["C6"]["gained"])),
            ("per-arm primary", g_["per_arm_primary"], {a: counts[m][a]["primary"] for a in sorted(gen)}),
            ("per-arm strict", g_["per_arm_strict"], {a: counts[m][a]["strict"] for a in sorted(gen)}),
        ]
        for name, a_, b_ in pairs_:
            same = json.dumps(a_, sort_keys=True) == json.dumps(b_, sort_keys=True)
            cmp.append({"scope": m, "item": name, "registered": a_, "recomputed": b_, "match": same})
            if not same:
                out["problems"].append(f"{m} {name}: registered {a_} vs recomputed {b_}")
    out["A_comparison"] = cmp

    # ---------- C. text of ADD-100's losses ----------
    def canon(s: str) -> str:
        return re.sub(r"\s+", "", s.strip()).rstrip(".")

    lost = [i for i in ids if P["SPARSE"][ids.index(i)] and not P["ADD-100"][ids.index(i)]]
    gained = [i for i in ids if P["ADD-100"][ids.index(i)] and not P["SPARSE"][ids.index(i)]]
    cls = {}
    for i in lost:
        g = gen["ADD-100"][i]
        if canon(g) == canon(ref[i]):
            cls[i] = "FORMAT_RESIDUE"
        elif canon(g) == canon(copyb[i]):
            cls[i] = "COPY_INPUT"
        else:
            cls[i] = "CONTENT_ERROR"
    ex = {k: [] for k in ("FORMAT_RESIDUE", "COPY_INPUT", "CONTENT_ERROR")}
    for i in lost:
        if len(ex[cls[i]]) < 6:
            ex[cls[i]].append({"id": i, "reference": ref[i], "SPARSE": gen["SPARSE"][i], "ADD-100": gen["ADD-100"][i], "AVG": gen["AVG"][i]})
    # the kind of content error: for JSON answers, how many keys differ; for integers, abs difference
    detail = Counter()
    for i in lost:
        if cls[i] != "CONTENT_ERROR":
            continue
        g = canon(gen["ADD-100"][i])
        try:
            a, b = json.loads(ref[i]), json.loads(g)
            if isinstance(a, dict) and isinstance(b, dict):
                nd = sum(a.get(k) != b.get(k) for k in set(a) | set(b))
                detail[f"json_{nd}_fields_differ"] += 1
            elif isinstance(a, int) and isinstance(b, int):
                detail["int_" + ("off_by_<=2" if abs(a - b) <= 2 else "off_by_>2")] += 1
            else:
                detail["type_mismatch"] += 1
        except Exception:
            detail["unparseable"] += 1
    avg_lost = {i for i in ids if P["SPARSE"][ids.index(i)] and not P["AVG"][ids.index(i)]}
    add_equals_avg = sum(canon(gen["ADD-100"][i]) == canon(gen["AVG"][i]) for i in lost if cls[i] == "CONTENT_ERROR")
    out["C_text"] = {
        "lost": len(lost), "gained": len(gained), "net": len(gained) - len(lost),
        "lost_by_class": dict(Counter(cls.values())),
        "lost_by_family": dict(Counter(fam[i] for i in lost)), "gained_by_family": dict(Counter(fam[i] for i in gained)),
        "lost_by_part": dict(Counter(part[i] for i in lost)), "gained_by_part": dict(Counter(part[i] for i in gained)),
        "content_error_detail": dict(detail),
        "content_errors_where_ADD100_output_equals_AVG_output": add_equals_avg,
        "overlap_ADD100_lost_with_AVG_lost": {"both": len(set(lost) & avg_lost), "ADD_only": len(set(lost) - avg_lost),
                                               "AVG_lost_total": len(avg_lost),
                                               "jaccard": len(set(lost) & avg_lost) / len(set(lost) | avg_lost)},
        "examples": ex,
        "ADD100_all_rows_trailing_structure": dict(Counter(
            "ends_}." if gen["ADD-100"][i].strip().endswith("}.") else
            "ends_." if gen["ADD-100"][i].strip().endswith(".") else "no_period" for i in ids)),
        "ADD100_rows_with_more_than_one_trailing_period": sum(gen["ADD-100"][i].strip().endswith("..") for i in ids),
        "ADD100_rows_with_internal_whitespace_vs_reference": sum(
            re.sub(r"\s+", "", gen["ADD-100"][i]) != gen["ADD-100"][i].strip() for i in ids),
    }

    # ---------- D. subgroup differences (descriptive; permutation tests) ----------
    d_all = P["ADD-100"].astype(int) - P["SPARSE"].astype(int)
    v3m = masks["V3"]
    stat = d_all[v3m].mean() - d_all[~v3m].mean()
    perm = np.random.default_rng(7)
    null = np.array([(lambda m: d_all[m].mean() - d_all[~m].mean())(perm.permutation(v3m)) for _ in range(20000)])
    famarr = np.array([fam[i] for i in ids])
    fam_means = {f: float(d_all[famarr == f].mean()) for f in FAMS}
    fstat = np.var([d_all[famarr == f].mean() for f in FAMS])
    fnull = np.array([np.var([d_all[p == f].mean() for f in FAMS]) for p in (perm.permutation(famarr) for _ in range(20000))])
    out["D_subgroups"] = {
        "V3_minus_EXT_mean_row_diff": float(stat), "perm_p_two_sided": float((np.abs(null) >= abs(stat)).mean()),
        "V3": {"diff": int(d_all[v3m].sum()), "sparse": int(P["SPARSE"][v3m].sum())},
        "EXT": {"diff": int(d_all[~v3m].sum()), "sparse": int(P["SPARSE"][~v3m].sum())},
        "family_mean_row_diff": fam_means,
        "family_diff_and_sparse": {f: {"diff": int(d_all[famarr == f].sum()), "sparse": int(P["SPARSE"][famarr == f].sum())} for f in FAMS},
        "family_heterogeneity_perm_p": float((fnull >= fstat).mean()),
    }

    # ---------- magnitude reading for C5/C6 ----------
    out["C5_C6_reading"] = {
        "SCALAR_P_0.402": counts["ALL"]["SCALAR_P"]["primary"], "RAND_MASK_norm_0.4401": counts["ALL"]["RAND_MASK"]["primary"],
        "SPARSE_ATTN_norm_0.4401": counts["ALL"]["SPARSE_ATTN"]["primary"], "SCALAR_MATCHED_0.634_norm_0.4401": counts["ALL"]["SCALAR_MATCHED"]["primary"],
        "SPARSE_1.0": counts["ALL"]["SPARSE"]["primary"],
        "SPARSE_ATTN_minus_SCALAR_MATCHED": pair(P["SCALAR_MATCHED"], P["SPARSE_ATTN"], masks["ALL"]),
        "SCALAR_MATCHED_minus_SPARSE": pair(P["SPARSE"], P["SCALAR_MATCHED"], masks["ALL"]),
        "SCALAR_P_minus_SCALAR_MATCHED": pair(P["SCALAR_MATCHED"], P["SCALAR_P"], masks["ALL"]),
        "RAND_MASK_vs_SCALAR_P_identical_generations": sum(gen["RAND_MASK"][i] == gen["SCALAR_P"][i] for i in ids),
    }
    (HERE / "AUDIT_INDEPENDENT.json").write_bytes((json.dumps(out, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({k: out[k] for k in ("problems", "E_provenance", "C_text", "D_subgroups", "C5_C6_reading")}, indent=1)[:12000])
    print("A_comparison mismatches:", [c for c in cmp if not c["match"]])
    for m in ("ALL", "V3"):
        r = res[m]
        print(m, "C1", r["C1"], "\n   C2", r["C2"], "\n   C3", {k: r["C3"][k] for k in r["C3"]}, "\n   C5", r["C5"], "C6", r["C6"])


if __name__ == "__main__":
    main()
