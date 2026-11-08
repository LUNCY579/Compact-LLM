"""Scoring for MRS-2. Gold is read here only, and only after generation.

The heavy statistics are REUSED from the MRS-1 analysis module rather than
reimplemented: `summarize`, `complementarity`, `factorial`, `paired_counts` and
`cluster_ci` are pure functions over the record shape MRS-2 deliberately preserves.

MRS-2 adds one thing MRS-1 did not have: a constant/majority-answer baseline reported
alongside every arm. MRS-FP-1 found a candidate that answered a single constant token
for every item, which scores above zero purely from the answer prior. Without the
baseline in view, such a model's accuracy is unreadable.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

from .core import (RECORDS, CLASSIFICATION, digest, file_hash, gold_labels, load_frozen,
                   new_dir, public_bank, read_json, source_identity, write_new)
from .parsing import answer
from . import prompts
from experiments.multi_reasoning_scout_v1.analysis import (
    complementarity, factorial, fraction, paired_counts, summarize)

SCORED_SPLIT = "CHECK"


def constant_baselines(rows, labels):
    """Accuracy of answering one fixed token for every item, plus the best such token.

    This is a property of the bank's answer prior, not of any model, and it is the
    floor every arm must clear before an accuracy number means anything.
    """
    gold = [labels[r["item"]["id"]]["answer"] for r in rows]
    counts = Counter(gold)
    per_token = {t: fraction(counts.get(t, 0), len(rows)) for t in ("0", "1", "2", "3", "UNKNOWN")}
    best = max(per_token, key=lambda t: per_token[t])
    return {"n": len(rows), "answer_prior": dict(counts), "per_constant_answer": per_token,
            "best_constant_answer": best, "best_constant_accuracy": per_token[best],
            "note": "Bank property, not a model result. Any arm at or below this is "
                    "indistinguishable from answering a fixed token."}


def degeneracy(rows):
    """Per-role input sensitivity. A role emitting one token for every item is not solving."""
    out = {}
    for role in ("R0", "R1", "R2", "R3", "D1", "D2", "D3"):
        values = [answer(r["outputs"][role]) for r in rows]
        counts = Counter(v for v in values if v is not None)
        out[role] = {"distinct_answers": len(counts), "distribution": dict(counts),
                     "constant": len(counts) <= 1,
                     "modal_share": fraction(max(counts.values()), len(values)) if counts else None}
    return out


def load_run(run_dir):
    run_dir = Path(run_dir)
    generation = read_json(run_dir / "generation.json")
    rows = [read_json(p) for p in sorted((run_dir / "raw").glob("*.item.json"))]
    bank = {i.id: i for i in public_bank()}
    for row in rows:
        item = row["item"]
        if item["id"] not in bank:
            raise ValueError("item outside the frozen bank: " + item["id"])
        source = bank[item["id"]]
        if item["problem"] != source.problem or item["split"] != source.split:
            raise ValueError("item text or split does not match the frozen bank: " + item["id"])
        for key, record in row["outputs"].items():
            if record["case_id"] != item["id"]:
                raise ValueError("candidate mapped to the wrong source item: " + item["id"])
    return generation, rows


def analyze_run(run_dir, out):
    freeze = load_frozen()
    generation, rows = load_run(run_dir)
    labels = gold_labels()
    scored = [r for r in rows if r["item"]["split"] == SCORED_SPLIT]
    cal = [r for r in rows if r["item"]["split"] == "CAL"]
    if not scored:
        raise ValueError("no scored CHECK cases")

    result = summarize(scored, labels)
    result["constant_baseline"] = constant_baselines(scored, labels)
    result["role_degeneracy"] = degeneracy(scored)
    best = result["constant_baseline"]["best_constant_accuracy"]
    result["above_constant_baseline"] = {
        arm: {"accuracy": s["accuracy"], "baseline": best,
              "margin": s["accuracy"] - best, "exceeds": s["accuracy"] > best}
        for arm, s in result["arms"].items()}

    # CAL is descriptive only and was never used to tune any selector.
    if cal:
        cal_summary = summarize(cal, labels)
        result["CAL_descriptive"] = {
            "n": len(cal), "arms": {k: v["accuracy"] for k, v in cal_summary["arms"].items()},
            "constant_baseline": constant_baselines(cal, labels),
            "note": "Descriptive. No selector was tuned on CAL; all selectors were frozen."}

    per_family = {}
    for family in sorted({r["item"]["family"] for r in scored}):
        subset = [r for r in scored if r["item"]["family"] == family]
        sub = summarize(subset, labels)
        per_family[family] = {"n": len(subset),
                              "arms": {k: v["accuracy"] for k, v in sub["arms"].items()},
                              "constant_baseline": constant_baselines(subset, labels),
                              "complementarity": complementarity(subset, labels)}
    result["per_family"] = per_family

    per_difficulty = {}
    for level in sorted({str(labels[r["item"]["id"]].get("difficulty")) for r in scored}):
        subset = [r for r in scored if str(labels[r["item"]["id"]].get("difficulty")) == level]
        if subset:
            sub = summarize(subset, labels)
            per_difficulty[level] = {"n": len(subset),
                                     "arms": {k: v["accuracy"] for k, v in sub["arms"].items()},
                                     "constant_baseline": constant_baselines(subset, labels)}
    result["per_difficulty"] = per_difficulty

    result["binding"] = generation["binding"]
    result["resources"] = generation["resources"]
    result["classification"] = CLASSIFICATION
    result["scored_split"] = SCORED_SPLIT
    result["efficacy_scope"] = ("OPEN DEVELOPMENT / MECHANISM SCOUT. Not confirmation, "
                                "not a release claim.")
    result["checker_boolean_fields"] = (
        "Absent by construction: the single-field contract carries no checker booleans, so the "
        "MRS-1 checker_errors counters are structurally zero here and are not evidence of "
        "reliable checking.")
    result["A1_caveat"] = (
        "Token-budget matching is vacuous under a single-field contract: the model terminates "
        "after the answer regardless of allowance. A1 measures that, not compute matching.")

    out = new_dir(Path(out))
    write_new(out / "RESULTS.json", result)
    write_new(out / "COMPLEMENTARITY_MATRIX.json", {
        "binding": generation["binding"]["model"]["id"],
        "diverse_roles_R0_R1_R2_R3": result["complementarity"],
        "repeated_direct_R0_D1_D2_D3": result["repeated_direct_complementarity"],
        "role_disagreement_rate": result["role_disagreement_rate"],
        "candidate_diversity_mean": result["candidate_diversity_mean"],
        "oracle_analysis_only": result["A4_oracle_analysis_only"],
        "per_family": {k: v["complementarity"] for k, v in per_family.items()},
        "role_degeneracy": result["role_degeneracy"]})
    write_new(out / "RESOURCE_REPORT.json", {
        "model": generation["binding"]["model"]["id"], "measured": generation["resources"],
        "per_arm_logical_cost": {k: v["logical_cost"] for k, v in result["arms"].items()},
        "cost_per_gross_repair": result["cost_per_gross_repair"],
        "cost_per_net_recovered_case": result["cost_per_net_recovered_case"],
        "limits_not_weakened": True})
    write_new(out / "receipt.json", {
        "stage": "MRS2_ANALYSIS", "status": "PASS", "run": str(run_dir),
        "freeze_sha256": file_hash(RECORDS / "FREEZE.json"),
        "prompts_sha256": prompts.manifest()["sha256"],
        "source": source_identity(), "scored_cases": len(scored),
        "gold_read_after_generation": True, "efficacy": False,
        "outputs": {p.name: file_hash(p) for p in sorted(out.iterdir())}})
    return result


def future_2x2(old_path, new_path, out):
    """Cells A/B (old checkpoint) against C/D (future trained checkpoint).

    Implemented and tested now, before any trained candidate exists, so the analysis
    cannot be designed after seeing C and D.
    """
    old, new = read_json(old_path), read_json(new_path)
    result = factorial(old, new)
    out = new_dir(Path(out))
    write_new(out / "FUTURE_2X2.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "future"))
    p.add_argument("--run", type=Path)
    p.add_argument("--old", type=Path)
    p.add_argument("--new", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "run":
        if not args.run:
            p.error("run requires --run")
        analyze_run(args.run, args.out)
    else:
        if not (args.old and args.new):
            p.error("future requires --old and --new")
        future_2x2(args.old, args.new, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
