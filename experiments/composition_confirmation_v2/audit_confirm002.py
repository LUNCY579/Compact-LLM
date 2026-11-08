"""CONFIRM-002 post-run audit, independent of analyse_stage_b.py (nothing is imported from it).

A. Re-score every raw generation from its TEXT, with boundary_rule and the primary transition rule, rather
   than trusting the stored 'credited'/'attempt' flags. Compare with the stored flags and with
   CONFIRM002_RESULT.json.
B. Recompute C1-C5 with my own implementations: a different bootstrap seed, plus non-bootstrap intervals
   (a normal approximation for C1 and C4, and a delta method for C3).
C. Provenance: bank hashes, runtime identity across all Stage B files, arm hashes (B_PREFLIGHT vs BUILD.json),
   row sets.
D. Read the text: TH25's boundary gains and losses vs SPARSE, and T losses, with examples.
E. Question-core repetition under an alternative, coarser core key, because the registered key produced
   336 unique cores in 336 rows.
Writes AUDIT_CONFIRM002.json.
"""
from __future__ import annotations

import hashlib, json, math, re, sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
HV = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hskill-views-002")
sys.path.insert(0, str(HV))
import boundary_rule as BR  # noqa: E402

T_FILES = ["B_T_GATES.json", "B_T_COMPOSE.json"]
H_FILES = ["B_H_GATES_BASE.json", "B_H_GATES_H25.json", "B_H_SPARSE.json", "B_H_COMPOSE.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prim(g, r):
    s = g.strip()
    return (s[:-1] if s.endswith(".") else s) == r


def binom2(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    out = {"problems": []}
    tbank = json.loads((HERE / "T_BANK_V4.json").read_text(encoding="utf-8"))["rows"]
    tref = {r["id"]: r["reference"] for r in tbank}; tids = sorted(tref)
    hrows = {json.loads(l)["id"]: json.loads(l) for l in (HERE / "HSKILL_SOUP_V3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    bid = sorted(i for i, r in hrows.items() if r["row_type"] == "boundary"); aid = sorted(i for i, r in hrows.items() if r["row_type"] == "answerable")

    # ---- C. provenance ----
    rts, tg, hg = [], {}, {}
    for f in T_FILES:
        d = json.loads((HERE / f).read_text(encoding="utf-8"))
        if d["bank_sha256"] != sha(HERE / "T_BANK_V4.json"):
            out["problems"].append(f"{f}: T bank hash")
        rts.append(json.dumps(d["runtime"], sort_keys=True))
        for a in d["arms"]:
            g = {p["id"]: p["generation"] for p in a["predictions"]}
            if set(g) != set(tref):
                out["problems"].append(f"{f}/{a['arm']}: row set")
            tg[a["arm"]] = g
    hrts = []
    for f in H_FILES:
        d = json.loads((HERE / f).read_text(encoding="utf-8"))
        if d["bank_sha256"] != sha(HERE / "HSKILL_SOUP_V3.jsonl"):
            out["problems"].append(f"{f}: H bank hash")
        hrts.append(json.dumps(d["runtime"], sort_keys=True))
        for a in d["arms"]:
            p = {x["id"]: x for x in a["predictions"]}
            if set(p) != set(hrows):
                out["problems"].append(f"{f}/{a['arm']}: row set")
            hg[a["arm"]] = p
    pf = json.loads((HERE / "B_PREFLIGHT.json").read_text(encoding="utf-8"))
    builds = {}
    for n in ("H25", "TH25", "THFULL"):
        builds[n] = json.loads((HERE / "arms" / n / "BUILD.json").read_text(encoding="utf-8"))["weight_sha256"] == pf["arms"][n]["sha256"]
    out["C_provenance"] = {"T_runtime_identical": len(set(rts)) == 1, "H_runtime_identical": len(set(hrts)) == 1,
                           "T_runtime": json.loads(rts[0]), "H_runtime": json.loads(hrts[0]),
                           "preflight_all_arms_match": all(v["match"] for v in pf["arms"].values()), "c2_arm_build_hash_equals_preflight": builds}

    # ---- A. re-score from text ----
    T = {a: np.array([prim(g[i], tref[i]) for i in tids]) for a, g in tg.items()}
    rescored = {}
    for a, p in hg.items():
        b = np.array([bool(BR.score_row("boundary", p[i]["prediction"], hrows[i]["reference"])[1]) for i in bid])
        at = np.array([not BR.abstains(BR.first_line(p[i]["prediction"])) for i in aid])
        co = np.array([bool(BR.score_row("answerable", p[i]["prediction"], hrows[i]["reference"])[1]) for i in aid])
        stored_b = np.array([p[i]["credited"] for i in bid]); stored_at = np.array([p[i]["attempt"] for i in aid])
        if (b != stored_b).any() or (at != stored_at).any():
            out["problems"].append(f"{a}: stored flags differ from re-scoring the text ({int((b != stored_b).sum())} boundary, {int((at != stored_at).sum())} attempts)")
        rescored[a] = {"b": b, "at": at, "co": co}
    out["A_counts"] = {"T_primary": {a: int(v.sum()) for a, v in T.items()},
                       "boundary": {a: int(v["b"].sum()) for a, v in rescored.items()},
                       "attempt": {a: int(v["at"].sum()) for a, v in rescored.items()},
                       "correct": {a: int(v["co"].sum()) for a, v in rescored.items()}}
    reg = json.loads((HERE / "CONFIRM002_RESULT.json").read_text(encoding="utf-8"))
    rd = reg["descriptive"]
    for k, a, b in (("T_primary", rd["T_primary"], out["A_counts"]["T_primary"]), ("boundary", rd["boundary_abstain"], out["A_counts"]["boundary"]),
                    ("attempt", rd["answerable_attempt"], out["A_counts"]["attempt"]), ("correct", rd["answerable_correct"], out["A_counts"]["correct"])):
        if a != b:
            out["problems"].append(f"{k}: registered {a} vs re-scored {b}")

    # ---- B. recompute limbs, own seed + non-bootstrap intervals ----
    rng = np.random.default_rng(777)
    d = T["TH25"].astype(int) - T["SPARSE"].astype(int); n = len(d)
    g_, l_ = int((d == 1).sum()), int((d == -1).sum()); diff = g_ - l_
    se = math.sqrt((g_ + l_) - diff ** 2 / n)
    boots = d[rng.integers(0, n, (10000, n))].sum(axis=1)
    c1 = {"diff": diff, "lost": l_, "gained": g_, "boot_p5": float(np.percentile(boots, 5)), "normal_lower95": diff - 1.6449 * se,
          "margin": -0.05 * int(T["SPARSE"].sum())}
    Hb = {a: v["b"] for a, v in rescored.items()}
    b2, c2_ = int((Hb["SPARSE"] & ~Hb["TH25"]).sum()), int((Hb["TH25"] & ~Hb["SPARSE"]).sum())
    c2 = {"net": c2_ - b2, "lost": b2, "gained": c2_, "p": binom2(b2, c2_)}
    nb = len(bid); idx = rng.integers(0, nb, (10000, nb))
    num = Hb["TH25"][idx].sum(1).astype(float) - Hb["SPARSE"][idx].sum(1); den = Hb["H25"][idx].sum(1).astype(float) - Hb["BASE"][idx].sum(1)
    ratio = np.where(den > 0, num / np.where(den > 0, den, 1), -np.inf)
    # delta method on per-row differences u = TH25-SPARSE, v = H25-BASE
    u = Hb["TH25"].astype(float) - Hb["SPARSE"]; v = Hb["H25"].astype(float) - Hb["BASE"]
    R = u.sum() / v.sum(); var = (np.var(u - R * v, ddof=1) * nb) / v.sum() ** 2
    c3 = {"P_H": float(R), "boot_p5": float(np.percentile(ratio, 5)), "delta_lower95": float(R - 1.6449 * math.sqrt(var)),
          "boot_P_le_075": float((ratio <= 0.75).mean())}
    at = {a: v["at"] for a, v in rescored.items()}
    d4 = at["TH25"].astype(int) - at["SPARSE"].astype(int)
    c4 = {"diff": int(d4.sum()), "discordant": int((d4 != 0).sum())}
    b5, c5_ = int((T["SPARSE"] & ~T["THFULL"]).sum()), int((T["THFULL"] & ~T["SPARSE"]).sum())
    c5 = {"loss": b5 - c5_, "lost": b5, "gained": c5_, "p": binom2(b5, c5_), "required": 0.10 * int(T["SPARSE"].sum())}
    out["B_recomputed"] = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}
    rc = reg["confirmatory"]
    for name, cond in (("C1 diff", rc["C1"]["diff"] == c1["diff"]), ("C2 lost/gained", (rc["C2"]["lost"], rc["C2"]["gained"]) == (b2, c2_)),
                       ("C3 P_H", abs(rc["C3"]["P_H"] - c3["P_H"]) < 1e-12), ("C4 diff", rc["C4"]["diff"] == c4["diff"]),
                       ("C5 loss", reg["C5_full_strength"]["loss"] == c5["loss"])):
        if not cond:
            out["problems"].append(f"mismatch: {name}")

    # ---- D. read the text ----
    fam = {i: hrows[i]["family"] for i in bid}
    gains = [i for j, i in enumerate(bid) if Hb["TH25"][j] and not Hb["SPARSE"][j]]
    losses = [i for j, i in enumerate(bid) if Hb["SPARSE"][j] and not Hb["TH25"][j]]
    ex = lambda ids, k: [{"family": fam[i], "user": [m["content"] for m in hrows[i]["prompt"] if m["role"] == "user"][0][:140],
                          "SPARSE": hg["SPARSE"][i]["prediction"][:120], "TH25": hg["TH25"][i]["prediction"][:120]} for i in ids[:k]]
    tl = [i for j, i in enumerate(tids) if T["SPARSE"][j] and not T["TH25"][j]]
    out["D_text"] = {"boundary_gains_by_family": dict(Counter(fam[i] for i in gains)), "boundary_losses_by_family": dict(Counter(fam[i] for i in losses)),
                     "gain_examples": ex(gains, 6), "loss_examples": ex(losses, 6),
                     "T_lost_rows": len(tl), "T_lost_examples": [{"ref": tref[i], "SPARSE": tg["SPARSE"][i], "TH25": tg["TH25"][i]} for i in tl[:5]],
                     "THFULL_strict_vs_primary": {"strict": int(sum(tg["THFULL"][i].strip() == tref[i] for i in tids)), "primary": int(T["THFULL"].sum()),
                                                  "trailing_period_rows": int(sum(tg["THFULL"][i].strip().endswith(".") for i in tids))}}
    # ---- E. question cores, coarser key: first 6 words of the user text, lower-cased, digits blanked ----
    core2 = [" ".join(re.sub(r"\d+", "#", [m["content"] for m in hrows[i]["prompt"] if m["role"] == "user"][0].lower()).split()[:6]) for i in bid]
    cc = Counter(core2)
    out["E_cores"] = {"registered_key_unique": len(set(re.sub(r"\d+", "#", " ".join(m["content"] for m in hrows[i]["prompt"] if m["role"] == "user").lower()) for i in bid)),
                      "coarse_key_unique": len(cc), "rows_on_repeated_coarse_cores": sum(v for v in cc.values() if v > 1),
                      "largest_coarse_core": cc.most_common(3)}
    (HERE / "AUDIT_CONFIRM002.json").write_bytes((json.dumps(out, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({k: out[k] for k in ("problems", "C_provenance", "A_counts", "B_recomputed", "E_cores")}, indent=1, ensure_ascii=False))
    print(json.dumps({k: out["D_text"][k] for k in ("boundary_gains_by_family", "boundary_losses_by_family", "T_lost_rows", "THFULL_strict_vs_primary")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
