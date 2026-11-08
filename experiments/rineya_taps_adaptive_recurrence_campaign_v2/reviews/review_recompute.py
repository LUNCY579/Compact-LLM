"""Source-only independent review receipt; stdlib, explicit DEV/SMOKE allowlist."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "initial_issue_receipt.json"
V1 = "experiments/rineya_taps_adaptive_recurrence_v1/"
ARMR = "experiments/anchor_routed_middle_recurrence_v1/"
LAB = "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/"


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def evidence(path, start, end):
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    return {"path": path, "sha256": digest(path), "start_line": start,
            "end_line": end, "source_excerpt": "\n".join(lines[start - 1:end])}


def issue(key, severity, title, classification, recommendation, *sources):
    return {"id": key, "severity": severity, "title": title,
            "classification": classification, "disposition": "OPEN_COORDINATOR_DECISION",
            "recommendation": recommendation, "evidence": list(sources)}


def main():
    registration_path = ARMR + "preregistrations/fixed_cpu_budget_v2.json"
    reg = json.loads((ROOT / registration_path).read_text(encoding="utf-8"))
    inherited = [{"path": p, "expected_sha256": h, "actual_sha256": digest(p),
                  "match": digest(p) == h} for p, h in reg["preserved_source_sha256"].items()]
    # Read only these exact authorized paths, never glob a dataset tree.
    dev_path = LAB + "data/evaluation/mergedev_fast.jsonl"
    smoke_path = LAB + "data/evaluation/supa_layer_ablation_smoke.jsonl"
    dev = [json.loads(l) for l in (ROOT / dev_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    smoke = [json.loads(l) for l in (ROOT / smoke_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    groups = {}
    for row in dev:
        groups.setdefault(row["scenario_id"], []).append(row["case_id"])
    scorer_path = LAB + "src/rineya_cross_scale/evaluation/scoring.py"
    spec = importlib.util.spec_from_file_location("review_pinned_scorer", ROOT / scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    # Each predictor below sees prompt only, never scorer/target/metadata.
    shortcuts = {"prompt_copy": lambda prompt: prompt, "constant_UNKNOWN": lambda prompt: "UNKNOWN",
                 "constant_YES": lambda prompt: "YES", "constant_NO": lambda prompt: "NO",
                 "empty_response": lambda prompt: ""}
    floors = {}
    for name, predict in shortcuts.items():
        rr = [{"case_id": c["case_id"], "score": scorer.score_response(predict(c["prompt"]), c["scorer"])["score"],
               "format_score": scorer.score_response(predict(c["prompt"]), c["format_scorer"])["score"]}
              for c in dev]
        floors[name] = {"input_contract": "prompt string only; no target access", "items": len(rr),
                        "correct": sum(r["score"] for r in rr),
                        "accuracy": sum(r["score"] for r in rr) / len(rr), "rows": rr}
    pool = (.8, .9, 1., 1.1, 1.2)
    rng_key = int.from_bytes(hashlib.sha256(f"17:{dev[0]['case_id']}".encode()).digest()[:8], "big")
    simulated = {str(nominal_seed): [random.Random(rng_key).choice(pool) for _prefix in range(5)]
                 for nominal_seed in (17, 29)}
    issues = [
        issue("R01", "HIGH", "Native A5 reset/seed contract degenerates to repeated per-example constants", "REQUIRED_FOR_H4",
              "Preserve legacy and restrict its interpretation; freeze and test corrected successor prefix RNG before H4 comparison.",
              evidence(V1 + "native.py", 106, 118), evidence(V1 + "integration.py", 59, 63),
              evidence(V1 + "scheduler.py", 135, 171)),
        issue("R02", "HIGH", "Between-forward resource checks require a parent hard supervisor", "REQUIRED_BEFORE_NATIVE_DISPATCH",
              "Enforce parent-owned deadline/RSS/exclusivity, reserve finalization, and terminate only admitted worker descendants.",
              evidence(V1 + "safety.py", 81, 85), evidence(V1 + "native.py", 116, 118)),
        issue("R03", "HIGH", "Native run does not bind smoke bytes and current experiment digest", "REQUIRED_BEFORE_NATIVE_DISPATCH",
              "Verify historical smoke SHA and freeze commit/uncommitted/source/scorer/config/review digests in campaign admission.",
              evidence(V1 + "native.py", 30, 58), evidence(V1 + "native.py", 81, 91)),
        issue("R04", "HIGH", "Native smoke identity lacks maximum-shape and exception restoration qualification", "REQUIRED_QUALIFICATION",
              "Add separately reviewed successor qualification; do not modify inherited identity/gates.",
              evidence(V1 + "native.py", 133, 148), evidence(V1 + "native.py", 177, 178),
              evidence(V1 + "native.py", 195, 202)),
        issue("R05", "HIGH", "Native output drops dependent scenario IDs", "REQUIRED_ANALYSIS_CONTRACT",
              "Bind metadata map; preserve parent units; aggregate repeats within item. Thirteen scenarios are an upper bound, not validated independent generators.",
              evidence(ARMR + "src/armr/evaluation.py", 91, 110)),
        issue("R06", "HIGH", "Complete six-hour run feasibility is not established by case limits", "REQUIRED_BEFORE_NATIVE_DISPATCH",
              "Forecast all 392 evaluations plus failures from qualified cost evidence, with ceiling/reserves and complete-matrix admission.",
              evidence(V1 + "native.py", 133, 176)),
        issue("R07", "MEDIUM", "Sixteen inspected DEV cases cannot adjudicate broad efficacy or tight preservation", "EXPLORATORY_LIMITATION",
              "Predeclare weights/margins/clusters/multiplicity, report paired counts, retain selection history; require independent authorized confirmation for efficacy.",
              evidence(ARMR + "evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json", 151, 274)),
        issue("R08", "MEDIUM", "Generation pool is prefix weighted and duplicate deterministic runs reweight it", "REQUIRED_POOL_DISCLOSURE",
              "Freeze pool provenance and weighting; do not call it task-conditional matched control. Keep teacher-forced diagnostic traces separate.",
              evidence(V1 + "native.py", 109, 114), evidence(V1 + "native.py", 149, 162)),
        issue("R09", "MEDIUM", "Instrumented native timing and dense FLOPs proxy do not support deployment speed", "EXPLORATORY_LIMITATION",
              "Report null unmeasured peak resources, account failures/qualification, and qualify paired minimal-instrumentation end-to-end timing before speed claims.",
              evidence(V1 + "integration.py", 89, 106), evidence(ARMR + "src/armr/evaluation.py", 113, 122)),
        issue("R10", "MEDIUM", "Prompt-copy scorer floors and one abstention item limit semantics", "REQUIRED_ENDPOINT_DISCLOSURE",
              "Report answer-blind floor, exact format, semantic/abstention/preservation endpoints separately; no scorer changes during scale comparison.",
              evidence(scorer_path, 62, 100)),
        issue("R11", "LOW", "Some dependency/early-path errors precede legacy terminal receipt", "REQUIRED_SUPERVISOR_RECEIPT",
              "Parent captures preflight/import/data/worker failures with status and costs; retain inherited refusal.",
              evidence(V1 + "native.py", 30, 33), evidence(V1 + "native.py", 74, 95),
              evidence(V1 + "safety.py", 62, 78)),
    ]
    result = {
        "schema_version": "1.0", "reviewer": "/root/adversarial_review", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "review_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "authority_sha256": hashlib.sha256(Path(r"C:\Users\clclc\.codex\attachments\6953c229-f698-4a9d-b293-c35ab386f0cb\pasted-text-1.txt").read_bytes()).hexdigest(),
        "native_model_loads": 0, "native_inference_runs": 0, "protected_or_test_reads": False,
        "scope": "Independent source review, two explicit allowed DEV/SMOKE files, source-only stdlib calculations.",
        "status": "CONDITIONAL_APPARATUS_PROCEED_NATIVE_DISPATCH_BLOCKED_PENDING_DISPOSITIONS",
        "issues": issues, "inherited_hash_checks": inherited,
        "source_passes": {
            "preflight_no_from_pretrained_or_forward": True,
            "preflight_imports_dependency_qwen_and_cuda_probes": True,
            "native_test_entrypoint_absent": True,
            "allowlist_before_content_read": evidence(ARMR + "src/armr/provenance.py", 47, 58),
            "gold_nll_guard": evidence(V1 + "diagnostics.py", 6, 23),
            "fixture_enumeration_cap_and_cost": evidence(V1 + "experiment.py", 218, 242),
        },
        "data_audit": {"dev_sha256": digest(dev_path), "smoke_sha256": digest(smoke_path),
                       "dev_items": len(dev), "smoke_items": len(smoke),
                       "family_count": len({r["family"] for r in dev}),
                       "scenario_count_upper_bound": len(groups), "scenario_groups": groups,
                       "missing_parent_ids": sum(not r.get("parent_id") for r in dev),
                       "generator_independence": "NOT_ESTABLISHED", "exposure": "PREVIOUSLY_INSPECTED_DEV"},
        "rng_source_reproduction": {"method": "Independently emulate source seed derivation and first choice after every reset; no Torch/model import",
                                    "config_seed": 17, "nominal_repetitions": simulated,
                                    "prefix_values_identical": len(set(simulated["17"])) == 1,
                                    "nominal_seed_values_identical": simulated["17"] == simulated["29"]},
        "statistical_feasibility": {"one_row_accuracy_increment": 1 / len(dev),
                                   "two_sided_exact_all_wins_p": {str(w): min(1., 2 ** (1 - w)) for w in range(1, 9)},
                                   "minimum_all_win_discordances_for_unadjusted_two_sided_p_le_05": 6,
                                   "conditional_historical_12_of_16_max_corrections_p": .125,
                                   "conditional_historical_13_of_16_max_corrections_p": .25,
                                   "illustrative_zero_harm_one_sided_95_upper_13_independent_units": 1 - .05 ** (1 / 13),
                                   "warning": "Conditional mathematical feasibility, not inference from a new model measurement; independence/exchangeability not established."},
        "legacy_complete_matrix_forecast": {"physical_case_evaluations": 392,
                                            "reused_oracle_summaries": 32,
                                            "max_generation_forwards_at_current_caps": 6348,
                                            "dev_case_cap_sum": sum(r["max_new_tokens"] for r in dev),
                                            "native_wall_seconds": None, "predicted_fit_under_six_hours": None},
        "answer_blind_floors": floors,
        "fixture_cost_reconciliation": {"ordinary_rollouts": 640, "oracle_extra_rollouts": 400,
                                        "physical_total": 1040, "oracle_core_calls": 1200,
                                        "core_calls_total": 2960, "extra_4_step_status": "NOT_ENUMERATED_CAP"},
        "limitations": ["No native qualification or efficacy measurement", "No independent validation", "No protected data access",
                        "No claim current dependencies/assets match exact runtime", "Coordinator must record each issue disposition"]}
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"receipt": str(OUT), "issues": len(issues), "hashes_match": all(r["match"] for r in inherited),
                      "scenario_count": len(groups), "floors": {k: v["accuracy"] for k, v in floors.items()}}))


if __name__ == "__main__":
    main()
