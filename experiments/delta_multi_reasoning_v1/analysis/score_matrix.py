"""Assemble O / T / REP / MR outcomes for one system and compute every registered table.

Gold enters HERE and nowhere upstream: the renderers and the selector policy never see a
reference. Candidate-oracle accuracy is computed in this file, after every generation is
recorded, and gets no deployment credit.

Produces, per system:
  * the four conditions' scores, overall and per stratum
  * paired repair / damage / net tables for every registered comparison
  * selector utilisation, role complementarity, and whether the roles did different things
  * call and token accounting, logical and deduplicated
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.answers import extract, is_correct         # noqa: E402
from analysis.paired import (                            # noqa: E402
    cluster_bootstrap_ci, noise_verdict, paired, role_complementarity,
    roles_perform_different_operations, selector_utilisation,
)
from inference.selector import Candidate, s0_agreement   # noqa: E402

SEP = "||"


def load_gen(d: Path) -> tuple[dict, dict]:
    meta = json.loads((d / "GENERATE.json").read_text(encoding="utf-8"))
    preds = {}
    for line in (d / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            preds[o["id"]] = o
    return meta, preds


def pick_from_selector_text(text: str, ids: list[str]) -> str | None:
    """Exactly one candidate id present, else None. Mirrors selector.model_selector."""
    hits = [i for i in ids if i in (text or "")]
    return hits[0] if len(hits) == 1 else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scout", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=("CAL", "CHECK"))
    ap.add_argument("--label", required=True, help="system label, e.g. D")
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--tokenmatch", type=Path, required=True)
    ap.add_argument("--rep-s2", type=Path, required=True, dest="rep_s2")
    ap.add_argument("--mr-s2", type=Path, required=True, dest="mr_s2")
    ap.add_argument("--mr-s1", type=Path, dest="mr_s1")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = {r["id"]: r for r in
            (json.loads(l) for l in a.scout.read_text(encoding="utf-8").splitlines() if l.strip())
            if r["split"] == a.split}

    sm, sp = load_gen(a.solver)
    tm, tp = load_gen(a.tokenmatch)
    rm, rp = load_gen(a.rep_s2)
    mm, mp = load_gen(a.mr_s2)
    s1m, s1p = (load_gen(a.mr_s1) if a.mr_s1 and a.mr_s1.exists() else ({}, {}))

    # condition -> {item_id: bool}
    outcome: dict[str, dict[str, bool]] = {c: {} for c in ("O", "T", "REP", "MR", "MR_S0")}
    per_item: list[dict] = []
    cand_correct: dict[str, dict[str, bool]] = {}
    role_texts: dict[str, dict[str, str]] = {}
    selected_mr: dict[str, str | None] = {}
    s1_outcome: dict[str, bool] = {}
    flags = {"selector_malformed_mr": 0, "selector_malformed_rep": 0,
             "mr_all_malformed": 0, "rep_all_malformed": 0, "mr_tie_broken": 0}
    toks = {"solver": 0, "tokenmatch": 0, "rep_s2": 0, "mr_s2": 0, "mr_s1": 0}
    calls = {"solver": 0, "tokenmatch": 0, "rep_s2": 0, "mr_s2": 0, "mr_s1": 0}

    for iid, row in sorted(rows.items()):
        direct_raw = sp.get(f"{iid}{SEP}direct", {}).get("prediction", "")
        direct_ans = extract(row, direct_raw)
        o_ok = is_correct(row, direct_ans)
        outcome["O"][iid] = o_ok

        t_raw = tp.get(f"{iid}{SEP}T", {}).get("prediction", "")
        outcome["T"][iid] = is_correct(row, extract(row, t_raw))

        # ---- REP candidates and its selector
        rep_cands = [Candidate("P0", direct_raw, direct_ans)]
        for i in (1, 2, 3):
            raw = sp.get(f"{iid}{SEP}rep{i}", {}).get("prediction", "")
            rep_cands.append(Candidate(f"P{i}", raw, extract(row, raw)))
        rep_use = [c for c in rep_cands if not c.malformed]
        if not rep_use:
            flags["rep_all_malformed"] += 1
            outcome["REP"][iid] = False
        else:
            txt = rp.get(f"{iid}{SEP}rep_s2", {}).get("prediction", "")
            pick = pick_from_selector_text(txt, [c.rid for c in rep_use])
            if pick is None:
                flags["selector_malformed_rep"] += 1
                sel = s0_agreement(rep_cands)
                ans = sel.answer
            else:
                ans = next(c.answer for c in rep_use if c.rid == pick)
            outcome["REP"][iid] = is_correct(row, ans)

        # ---- MR candidates, its selector, and S0
        mr_cands = [Candidate("R0", direct_raw, direct_ans)]
        texts = {"R0": direct_raw}
        for rid in ("R1", "R2", "R3"):
            raw = sp.get(f"{iid}{SEP}{rid}", {}).get("prediction", "")
            texts[rid] = raw
            mr_cands.append(Candidate(rid, raw, extract(row, raw)))
        role_texts[iid] = texts
        cand_correct[iid] = {c.rid: is_correct(row, c.answer) for c in mr_cands}

        mr_use = [c for c in mr_cands if not c.malformed]
        s0 = s0_agreement(mr_cands)
        outcome["MR_S0"][iid] = is_correct(row, s0.answer)
        if s0.flags.get("tie_broken"):
            flags["mr_tie_broken"] += 1
        if not mr_use:
            flags["mr_all_malformed"] += 1
            outcome["MR"][iid] = False
            selected_mr[iid] = None
        else:
            txt = mp.get(f"{iid}{SEP}mr_s2", {}).get("prediction", "")
            pick = pick_from_selector_text(txt, [c.rid for c in mr_use])
            if pick is None:
                flags["selector_malformed_mr"] += 1
                selected_mr[iid] = s0.selected_rid
                ans = s0.answer
            else:
                selected_mr[iid] = pick
                ans = next(c.answer for c in mr_use if c.rid == pick)
            outcome["MR"][iid] = is_correct(row, ans)

        if f"{iid}{SEP}mr_s1" in s1p and mr_use:
            txt = s1p[f"{iid}{SEP}mr_s1"].get("prediction", "")
            pick = pick_from_selector_text(txt, [c.rid for c in mr_use])
            ans = (next(c.answer for c in mr_use if c.rid == pick) if pick
                   else s0_agreement(mr_cands).answer)
            s1_outcome[iid] = is_correct(row, ans)

        per_item.append({
            "id": iid, "stratum": row["stratum"], "world_id": row["world_id"],
            "router_case": row.get("router_case"), "unanswerable": bool(row.get("unanswerable")),
            "O": o_ok, "T": outcome["T"][iid], "REP": outcome["REP"][iid],
            "MR": outcome["MR"][iid], "MR_S0": outcome["MR_S0"][iid],
            "candidates": cand_correct[iid], "selected_mr": selected_mr.get(iid),
            "raw": {k: v for k, v in texts.items()},
        })

    for key, (meta, preds) in (("solver", (sm, sp)), ("tokenmatch", (tm, tp)),
                               ("rep_s2", (rm, rp)), ("mr_s2", (mm, mp)),
                               ("mr_s1", (s1m, s1p))):
        calls[key] = len(preds)
        toks[key] = sum((p.get("generated_tokens") or 0) for p in preds.values())

    strata = sorted({r["stratum"] for r in rows.values()})

    def score(cond: str, stratum: str | None = None) -> int:
        return sum(1 for i, v in outcome[cond].items()
                   if v and (stratum is None or rows[i]["stratum"] == stratum))

    conditions = {
        c: {"total": score(c), "n": len(rows),
            "per_stratum": {s: {"correct": score(c, s),
                                "n": sum(1 for r in rows.values() if r["stratum"] == s)}
                            for s in strata}}
        for c in ("O", "T", "REP", "MR", "MR_S0")
    }

    pairs = {}
    for base, cand in (("O", "T"), ("O", "REP"), ("O", "MR"), ("REP", "MR"), ("MR_S0", "MR")):
        p = paired(base, cand, outcome[base], outcome[cand])
        d = p.as_dict()
        d["noise_verdict_vs_acquisition_spread"] = noise_verdict("acquisition", p.net_gain)
        pairs[f"{base}_vs_{cand}"] = d

    by_world: dict[str, list[float]] = {}
    for i, v in outcome["MR"].items():
        by_world.setdefault(rows[i]["world_id"], []).append(1.0 if v else 0.0)
    mr_ci = cluster_bootstrap_ci(by_world)

    report = {
        "schema": 1, "record": "DMRF_MATRIX_SCORE",
        "system": a.label, "split": a.split, "n_items": len(rows),
        "runtime": {"transformers": sm.get("transformers"), "dtype": sm.get("dtype"),
                    "decoding": sm.get("decoding"),
                    "rows_skipped_context_overflow": {
                        k: m.get("rows_skipped_context_overflow")
                        for k, m in (("solver", sm), ("tokenmatch", tm), ("rep_s2", rm),
                                     ("mr_s2", mm), ("mr_s1", s1m)) if m}},
        "conditions": conditions,
        "paired": pairs,
        "mr_world_clustered_ci": mr_ci,
        "selector_utilisation_mr_s2": selector_utilisation(cand_correct, selected_mr,
                                                           outcome["MR"]),
        "role_complementarity": role_complementarity(cand_correct),
        "roles_did_different_things": roles_perform_different_operations(role_texts),
        "s1_diagnostic": {
            "n": len(s1_outcome),
            "correct_with_findings_withheld": sum(1 for v in s1_outcome.values() if v),
            "correct_with_findings_shown_same_subset":
                sum(1 for i in s1_outcome if outcome["MR"].get(i)),
            "note": ("S1 withholds the derived checker findings on a prospectively fixed "
                     "subset. Its calls are accounted separately below."),
        },
        "flags": flags,
        "accounting": {
            "actual_calls_by_pass": calls,
            "actual_generated_tokens_by_pass": toks,
            "actual_total_calls": sum(calls.values()),
            "logical_deployment_calls": {
                "O": len(rows), "T": len(rows), "REP": 5 * len(rows), "MR": 5 * len(rows)},
            "registered_reuse": ("one `direct` generation serves O, REP[0] and MR R0; identical "
                                 "checkpoint, prompt, decoding, limit and policy"),
            "s1_extra_calls": calls["mr_s1"],
            "note": ("The deduplicated count is what was spent; the logical count is what a "
                     "deployment would pay. Reporting only one of them misleads in one "
                     "direction or the other."),
        },
        "gold_entered_here_only": True,
        "oracle_note": ("candidate-oracle accuracy in role_complementarity is diagnostic and "
                        "gets no deployment credit"),
        "quality_claim": False, "release_permitted": False,
        "per_item": per_item,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "system": a.label, "n": len(rows),
        "conditions": {c: f"{v['total']}/{v['n']}" for c, v in conditions.items()},
        "per_stratum_MR_vs_O": {
            s: f"O {conditions['O']['per_stratum'][s]['correct']} -> "
               f"MR {conditions['MR']['per_stratum'][s]['correct']} "
               f"of {conditions['O']['per_stratum'][s]['n']}" for s in strata},
        "paired": {k: {"repair": v["repair"], "damage": v["damage"], "net": v["net_gain"],
                       "p": round(v["mcnemar_exact_two_sided_p"], 4),
                       "noise": v["noise_verdict_vs_acquisition_spread"]}
                   for k, v in pairs.items()},
        "selector_utilisation": report["selector_utilisation_mr_s2"]["utilisation"],
        "oracle_minus_best_single": report["role_complementarity"]["oracle_minus_best_single"],
        "roles_verdict": report["roles_did_different_things"]["verdict"],
        "flags": flags,
        "actual_total_calls": sum(calls.values()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
