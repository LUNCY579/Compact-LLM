"""Scoped research-policy v5 disposition; source/data only, no model imports."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import open_bank

C = Path(__file__).resolve().parent.parent
ROOT = C.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    destination = C / "reviews/research_policy_v5_receipt.json"
    require(not destination.exists(), "exclusive policy review already exists")
    policy_path = C / "RESEARCH_POLICY.jsonl"
    entries = [json.loads(line) for line in policy_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    proposals = [entry for entry in entries if entry.get("version") == 5 and entry.get("event") != "REVIEW_DISPOSITION"]
    require(len(proposals) == 1, "policy5 proposal identity differs")
    proposed = proposals[0]
    require(proposed["parent"] == 4 and proposed["status"] == "PROPOSED_REVIEW_PENDING"
            and proposed["empirical_research_allocation_improvement"] == "NOT_MEASURED", "policy lineage/claim differs")
    admission_path = C / "open_admission_v2.json"
    require(sha(admission_path) == "65e3e424d0d2b3c91905e99ec2dcb477e7d55ce4e9e56df623323c4b212b9c60", "approved OPEN admission changed")
    admission = read(admission_path)
    dispatch_review = read(C / "reviews/open_dispatch_receipt.json")
    require(dispatch_review["status"] == "PASS_OPEN_FP32_DISPATCH"
            and dispatch_review["admission_sha256"] == sha(admission_path), "OPEN review binding differs")
    require(len(admission["dispatch_hashes"]) == 78, "OPEN source closure differs")
    for relative, digest in admission["dispatch_hashes"].items():
        path = (ROOT / relative).resolve(strict=True)
        require(path.is_relative_to(ROOT) and sha(path) == digest, "reviewed OPEN closure changed: " + relative)
    bank_path = Path(admission["open_bank_path"])
    require(sha(bank_path) == admission["open_bank_sha256"], "approved bank changed")
    bank, audit = read(bank_path), read(bank_path.parent / "audit.json")
    require(sha(C / "open_bank.py") == admission["open_source_sha256"], "reviewed solver/scorer changed")
    positive = [open_bank.solve_prompt(row["prompt"]) for row in bank]
    require(len(bank) == 24 and all(open_bank.score_response(answer, row)["correct"] for row, answer in zip(bank, positive)), "prompt-only positive control differs")
    blind = Counter()
    for row in bank:
        for name, response in open_bank.blind_predictions(row["prompt"]).items():
            blind[name] += int(open_bank.score_response(response, row)["correct"])
    require(dict(blind) == audit["blind_correct_counts"] and max(blind.values()) <= 12, "own-bank floor gate differs")
    require(Counter(row["split"] for row in bank) == {"DEVELOPMENT": 16, "OPEN_VALIDATION": 8}, "split differs")
    families = {split: {row["generator_family"] for row in bank if row["split"] == split} for split in ("DEVELOPMENT", "OPEN_VALIDATION")}
    require(len(families["DEVELOPMENT"]) == len(families["OPEN_VALIDATION"]) == 4
            and not families["DEVELOPMENT"] & families["OPEN_VALIDATION"], "generator variants overlap")
    require(len({row["parent_id"] for row in bank}) == 24 and len({row["category"] for row in bank}) == 4
            and all("target" not in row["prompt"].lower() for row in bank), "world/target-marker gate differs")
    # A wrong canonical answer remains format-valid; gold only changes correctness.
    for row in bank:
        wrong = "NO" if row["answer_kind"] == "choice" and row["gold"] == "YES" else "YES" if row["answer_kind"] == "choice" else "99999999"
        result = open_bank.score_response(wrong, row)
        require(result["format_valid"] and not result["correct"], "gold-independent canonical formatting differs")
    prior = read(C / "reviews/gpu_analysis_review_attempt002_receipt.json")
    require(admission["static_original_A3_mean"] == prior["pool_mean"] == 1.1473425965417514, "static comparator differs from prior frozen reference mean")
    protocol_path = C / "OPEN_NATIVE_REGISTRATION.md"
    protocol = protocol_path.read_text(encoding="utf-8")
    clauses = {"six_constant_controls": "{.8,.9,1,1.1,1.2,1.1473425965417514}",
               "DEV_freeze_before_validation": "BEFORE any validation model forward",
               "validation_frozen_primary": "primary comparator is the frozenDEV alpha",
               "practical_margin": "Practical exploratory margin.05",
               "own_baselines": "data/scorer/output-cap intervention with its own baselines",
               "no_controller_fit": "No controller parameter revision is justified",
               "source_visible_limit": "exploratory transfer, not independent confirmation"}
    require(all(text in protocol for text in clauses.values()), "approved primary clause differs")
    require(admission["max_new_tokens"] == 8 and admission["max_open_input_tokens"] == 350
            and admission["hard_deadline_seconds"] == 14400 and admission["native_finalization_reserve_seconds"] == 1800
            and admission["infrastructure_retry_available"] is False
            and admission["parent_host_available_floor_bytes"] == 7*1024**3
            and admission["gpu_allocated_ceiling_bytes"] == 5*1024**3, "unchanged admitted shape/resource limits differ")
    require("torch" not in sys.modules, "policy apparatus imported native model runtime")
    evidence_paths = [policy_path, admission_path, protocol_path, C / "OPEN_BANK_DESIGN.md", bank_path,
                      bank_path.parent / "audit.json", bank_path.parent / "positive_controls.json",
                      C / "reviews/open_bank_review_receipt.json", C / "reviews/open_dispatch_receipt.json",
                      C / "reviews/gpu_result_review_002.md", C / "reviews/gpu_analysis_review_attempt002_receipt.json"]
    result = {"schema_version": "1.0", "reviewer": "/root/adversarial_review", "created_utc": datetime.now(timezone.utc).isoformat(),
              "version": 5, "parent": 4, "status": "ACCEPTED_FOR_SCOPED_PROCEDURE",
              "policy_record": proposed, "policy_record_sha256": hashlib.sha256(json.dumps(proposed, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "evidence_sha256": {str(path.relative_to(C)): sha(path) for path in evidence_paths},
              "review_script_sha256": sha(__file__), "verified_dispatch_hash_count": 78,
              "verified_checks": {"prompt_only_positive_controls": 24, "maximum_tested_blind_floor": max(blind.values()),
                  "tested_blind_floor_ceiling": 12, "blind_counts": dict(blind),
                  "generator_split": {name: sorted(values) for name, values in families.items()},
                  "gold_independent_format": True, "new_data_scorer_cap_own_baselines": True,
                  "fixed_constants": [.8, .9, 1., 1.1, 1.1473425965417514, 1.2],
                  "DEV_freeze_before_validation": True, "practical_margin": .05, "original_inference_guards_unchanged": True},
              "scope": "Prospectively prioritize task-bank discrimination and stronger fixed controls with unchanged candidate; preserve approved OPEN source/data/scorer/cap matrix and safety limits.",
              "empirical_research_allocation_improvement": "NOT_MEASURED", "empirical_allocation_efficiency": "NOT_MEASURED",
              "native_model_loads": 0, "native_calls": 0, "new_native_outcomes_read": False,
              "limitations": ["Flat reference constant curve/zero finite-grid headroom supports this next research choice, not general impossibility of adaptive schedules.",
                  "Tested blind predictors are a bounded floor set, not an optimal shortcut bound; global10/24 can hide family50percent floors.",
                  "Named variants share author/parser/code categories; source-visible validation remains exploratory, not independent confirmation.",
                  "The prior-reference A3 mean strengthens a prospective static comparator; it is not an OPEN label fit or a new controller revision.",
                  "This accepts procedural consistency; no measured improvement in research allocation efficiency, agent ability, model efficacy or deployment speed.",
                  "Native OPEN qualification/results remain pending; fresh resource floors and cumulative21600-second budget apply; no retry remains."],
              "unresolved_policy_blockers": []}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": result["status"], "version": 5, "verified_dispatch_hashes": 78, "positive_controls": 24, "blind_max": max(blind.values())}))


if __name__ == "__main__":
    main()
