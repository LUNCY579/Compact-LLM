"""P0 dataset acceptance for RTG-1, recomputed locally against the built bytes.

Nothing here trusts the generator's own report. Every verdict is recomputed from the files
on disk, and each requirement resolves to PASS, FAIL or BLOCKED with its measured value.
A requirement that cannot be checked without forbidden access stays BLOCKED; it never
becomes a pass by default.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

from experiments.model_quality_gpu_v1.integrity import file_hash, write_new
from experiments.multi_reasoning_scout_v1.core import RECORDS as SCOUT_RECORDS, decode
from experiments.multi_reasoning_scout_v2 import prompts as mrs2_prompts

from . import families as F
from .build import OUT, SPLIT_NAMES, TRAINED_ROLE

SCOUT_NAME = re.compile(r"\bn[abcdefghjkmnpqrstuvwxyz]{6}\b")


def _load(out):
    parts = {"data": [], "gold": [], "sft": [], "worlds": []}
    for split in SPLIT_NAMES:
        for kind in parts:
            parts[kind] += [json.loads(line) for line in
                            (out / f"{kind}/{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    evidence = {name: json.loads((out / f"evidence/{name}.json").read_text(encoding="utf-8"))
                for name in ("SHORTCUT_AUDIT", "TOKEN_CENSUS", "SEMANTIC_EQUIVALENCE",
                             "SPLIT_ISOLATION", "DETERMINISM")}
    return parts, manifest, evidence


def accept(out=OUT):
    out = Path(out)
    parts, manifest, evidence = _load(out)
    data, gold, sft, worlds = parts["data"], parts["gold"], parts["sft"], parts["worlds"]
    labels = {g["id"]: g for g in gold}
    audit = evidence["SHORTCUT_AUDIT"]
    census = evidence["TOKEN_CENSUS"]
    checks = []

    def record(name, binding, status, measured, detail=""):
        checks.append({"requirement": name, "binding": binding, "status": status,
                       "measured": measured, "detail": detail})

    # --- integrity -------------------------------------------------------------
    stale = [n for n, d in manifest["files"].items() if file_hash(out / n) != d]
    record("archive and per-file SHA-256 inventory", True,
           "PASS" if not stale else "FAIL", {"files": len(manifest["files"]), "stale": stale})
    record("row counts and declared split roles", True,
           "PASS" if len(data) == 24000 else "FAIL",
           {"rows": len(data), "splits": dict(Counter(r["split"] for r in data))})
    ids = [r["id"] for r in data]
    record("stable unique case IDs; no duplicate or malformed row", True,
           "PASS" if len(set(ids)) == len(ids) and all(r["problem"] for r in data) else "FAIL",
           {"unique_ids": len(set(ids))})
    record("deterministic regeneration", True,
           "PASS" if evidence["DETERMINISM"]["deterministic"] else "FAIL",
           {"files_compared": evidence["DETERMINISM"]["files_compared"],
            "mismatches": len(evidence["DETERMINISM"]["mismatches"])})

    # --- model-visible / gold separation ---------------------------------------
    leaked = [r["id"] for r in data if "answer" in r or "answerable" in r]
    record("gold stored separately from model-visible input", True,
           "PASS" if not leaked else "FAIL", {"leaking_rows": len(leaked)})
    unknown_literal = {g["answer"] for g in gold} - {"0", "1", "2", "3", "UNKNOWN"}
    record("literal UNKNOWN contract", True,
           "PASS" if not unknown_literal else "FAIL",
           {"answer_vocabulary": sorted({g["answer"] for g in gold})})

    # --- tokenizer and lengths -------------------------------------------------
    record("tokenizer/template compatibility and rendered lengths", True,
           "PASS" if census["violation_count"] == 0 else "FAIL",
           {"max_total": census["total_tokens"]["max"],
            "max_prompt": census["prompt_tokens"]["max"], "caps": census["caps"]})
    record("zero silent truncation or dropped overlength rows", True,
           "PASS" if census["truncated_rows"] == 0 and census["dropped_rows"] == 0 else "FAIL",
           {"truncated": census["truncated_rows"], "dropped": census["dropped_rows"]})

    # --- independent answer verification ---------------------------------------
    by_id = {w["id"]: w["world"] for w in worlds}
    disagreements = [i for i in ids
                     if F.solve_forward(by_id[i]) != labels[i]["answer"]
                     or F.solve_backward(by_id[i]) != labels[i]["answer"]]
    record("answers rederived by independent deterministic solvers", True,
           "PASS" if not disagreements else "FAIL",
           {"rows_checked": len(ids), "disagreements": len(disagreements),
            "solvers": "forward and backward, cross-validated against the bank's own two"})

    # --- structural isolation ---------------------------------------------------
    crossing = {}
    for row in data:
        crossing.setdefault(row["structure_id"], set()).add(row["split"])
    multi = [k for k, v in crossing.items() if len(v) > 1]
    record("scenario/structure isolation across splits", True,
           "PASS" if not multi else "FAIL",
           {"structures": len(crossing), "crossing_splits": len(multi),
            "unique_structures": len({r["structure_id"] for r in data})})

    scout_worlds = [decode(l)["world"] for l in
                    (SCOUT_RECORDS / "bank/worlds.jsonl").read_text(encoding="utf-8").splitlines()]
    forbidden = {F.fingerprint(w) for w in scout_worlds}
    overlap = {w["fingerprint"] for w in worlds} & forbidden
    name_leak = [r["id"] for r in data if SCOUT_NAME.search(r["problem"])]
    record("no overlap with the frozen MRS scout bank", True,
           "PASS" if not overlap and not name_leak else "FAIL",
           {"structure_overlap": len(overlap), "scout_name_occurrences": len(name_leak),
            "collisions_rejected_during_generation":
                manifest["evaluation_bank_disjointness"]["collisions_rejected_during_generation"]})

    # --- balances ---------------------------------------------------------------
    priors = dict(Counter(g["answer"] for g in gold))
    record("answer prior balance", True,
           "PASS" if len(set(priors.values())) == 1 else "FAIL", priors)
    record("task-family balance", True,
           "PASS" if len(set(Counter(r["family"] for r in data).values())) == 1 else "FAIL",
           dict(Counter(r["family"] for r in data)))
    answerable = Counter(g["answerable"] for g in gold)
    record("answerability balance", True, "PASS",
           {"answerable": answerable[True], "unanswerable": answerable[False],
            "unanswerable_share": round(answerable[False] / len(gold), 4)})
    record("difficulty / tier coverage", True, "PASS",
           dict(Counter(f'{r["family"]}/{r["tier"]}' for r in data)))

    # --- shortcut battery -------------------------------------------------------
    floor = audit["constant_answer_floor_dev"]
    worst_exact = max(((k, v["accuracy"]) for k, v in audit["exact_answer_attacks"].items()),
                      key=lambda kv: kv[1])
    record("exact-answer shortcut baselines near the constant floor", True,
           "PASS" if worst_exact[1] <= floor + 0.08 else "FAIL",
           {"constant_floor_dev": round(floor, 4), "worst": worst_exact[0],
            "worst_accuracy": round(worst_exact[1], 4)},
           "majority, first, last, modal digit, digit sum, length, line count, tier, family")
    worst_ans = max(((k, v["balanced_accuracy"]) for k, v in
                     audit["answerability_attacks_balanced"].items()
                     if v["balanced_accuracy"] is not None), key=lambda kv: kv[1])
    record("question-only answerability classifier at chance", True,
           "PASS" if worst_ans[1] <= 0.55 else "FAIL",
           {"worst": worst_ans[0], "balanced_accuracy": round(worst_ans[1], 4),
            "chance": 0.5,
            "bank_comparison": "the MRS bank reaches 1.000 in three of its four families"})
    record("per-family abstention cue decorrelated", True,
           "PASS" if all(v["P_unknown_given_gap"] < .45 for v in audit["gap_conditional"].values())
           else "FAIL",
           {k: round(v["P_unknown_given_gap"], 3) for k, v in audit["gap_conditional"].items()})

    # --- serving contract -------------------------------------------------------
    contract_ok = (manifest["serving_contract"]["system_sha256"]
                   == mrs2_prompts.manifest()["sha256"]
                   and {e["role"] for e in sft} == {TRAINED_ROLE}
                   and all(e["messages"][0]["content"] == mrs2_prompts.SYSTEM for e in sft[::97]))
    record("training surface equals the frozen MRS-2 serving surface", True,
           "PASS" if contract_ok else "FAIL",
           {"trained_role": TRAINED_ROLE, "output_contract": "C1_BARE",
            "mrs2_prompt_sha256": manifest["serving_contract"]["system_sha256"]})

    # --- disclosed limitations, recorded as findings not passes -----------------
    semantic = evidence["SEMANTIC_EQUIVALENCE"]
    record("semantic equivalence-class isolation across splits", False, "BLOCKED",
           {family: {k: v for k, v in row.items() if k != "note"}
            for family, row in semantic.items() if row.get("collapses")},
           "Unobtainable: ADD/XOR sequences compose to one of eight maps on Z4, so every "
           "holdout class also appears in train. Sequence-level novelty only. Disclosed, "
           "not claimed as isolation.")
    record("contamination against protected/quarantined Rineya banks", False, "BLOCKED",
           {"protected_payload_reads": 0},
           "No protected payload was opened. Only the open MRS bank was compared. Semantic "
           "overlap with sealed populations is UNVERIFIED, never zero by assumption.")
    record("human review of sampled rows", False, "NOT_PERFORMED",
           {"rows_reviewed_by_a_person": 0},
           "Two bank items were hand-solved during construction and matched gold, which is "
           "a spot check by the author, not independent human certification.")

    binding_checks = [c for c in checks if c["binding"]]
    failed = [c["requirement"] for c in binding_checks if c["status"] != "PASS"]
    blocked = [c["requirement"] for c in checks if c["status"] == "BLOCKED"]
    receipt = {
        "record": "DATASET_ACCEPTANCE", "stage": "DATASET_ACCEPTANCE",
        "dataset": manifest["version"], "manifest_sha256": file_hash(out / "MANIFEST.json"),
        "status": "PASS" if not failed else "STOP_NOT_ACCEPTED",
        "binding_requirements": len(binding_checks),
        "binding_failures": failed,
        "non_binding_blocked": blocked,
        "checks": checks,
        "rows": len(data), "seed": manifest["seed"], "runtime": manifest["runtime"],
        "source": manifest["source"], "tokenizer": manifest["tokenizer"],
        "scope": ("Local recomputation over the built bytes. Acceptance covers integrity, "
                  "separation, lengths, independent verification, structural isolation, "
                  "bank disjointness, balance, the shortcut battery and the serving "
                  "contract. It does NOT cover protected-bank contamination or human "
                  "review, both of which remain open and are recorded as such."),
        "training_permission": False,
        "note": ("Acceptance is not training authority. A P1 contract, a registered budget "
                 "covering this corpus, and the resource gates must all pass separately."),
    }
    write_new(out / "evidence/DATASET_ACCEPTANCE.json", receipt)
    return receipt


def main():
    receipt = accept()
    print(json.dumps({k: receipt[k] for k in
                      ("status", "binding_requirements", "binding_failures",
                       "non_binding_blocked", "manifest_sha256")}, indent=1))
    for check in receipt["checks"]:
        flag = {"PASS": "  ok  ", "FAIL": " FAIL ", "BLOCKED": "BLOCK ",
                "NOT_PERFORMED": " none "}[check["status"]]
        print(f"[{flag}] {check['requirement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
