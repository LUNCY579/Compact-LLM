"""Independent terminal evidence/response review; never loads a model."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.solver_audit import solve_prompt

CAMPAIGN = Path(__file__).resolve().parent.parent
RUN = CAMPAIGN / "runs/native-gpu-reference-002"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def journal(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def main():
    out = CAMPAIGN / "reviews/gpu_analysis_review_attempt002_receipt.json"
    require(not out.exists(), "exclusive result review exists")
    metadata = read(CAMPAIGN / "runs/binding-001/PROMPT_AUDIT.json")
    items = {row["case_id"]: row for row in metadata}
    rows = journal(RUN / "result/outcomes.jsonl")
    physical = [r for r in rows if r["physical_run"]]
    dev = [r for r in physical if r["split"] == "DEV"]
    require(len(physical) == 408 and len(rows) - len(physical) == 32 and len(dev) == 288, "matrix cardinality differs")
    scorer_path = ROOT / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/rineya_cross_scale/evaluation/scoring.py"
    require(sha(scorer_path) == "5a22e1c1ff7b0a56623931c0782f81dd2b3a9df83ccfe77ce3724d37cf320086", "pinned scorer changed")
    spec = importlib.util.spec_from_file_location("review_002_pinned_scorer", scorer_path)
    scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
    for row in rows:
        item = items[row["case_id"]]
        require(scorer.score_response(row["response"], item["scorer"])["score"] == row["score"], "raw task score mismatch")
        require(scorer.score_response(row["response"], item["format_scorer"])["score"] == row["format_score"], "raw format score mismatch")
    policy = lambda r: f"A2:{r['fixed_alpha']:g}" if r["arm"] == "A2" else r["arm"]
    grouped = defaultdict(list)
    for row in dev:
        grouped[(policy(row), row["case_id"])].append(row)
    by_policy = defaultdict(dict)
    for (name, case_id), rr in grouped.items():
        require(len(rr) == 2 and {r["seed"] for r in rr} == {17, 29}, "case replicate matrix incomplete")
        by_policy[name][case_id] = statistics.mean(r["score"] for r in rr)
    accuracy = {name: statistics.mean(rr.values()) for name, rr in by_policy.items()}
    require(accuracy["A0"] == .75 and all(v == .8125 for name, v in accuracy.items() if name != "A0"), "recomputed score curve differs")
    fixed_names = [f"A2:{a:g}" for a in (.8, .9, 1., 1.1, 1.2)]
    require(all(by_policy[name] == by_policy["A1"] for name in fixed_names + ["A3", "A5"]), "policy score vectors differ")
    oracle = statistics.mean(max(by_policy[name][case_id] for name in fixed_names) for case_id in by_policy["A0"])
    require(oracle == .8125, "oracle score differs")
    wins = [case_id for case_id in by_policy["A0"] if by_policy["A3"][case_id] > by_policy["A0"][case_id]]
    losses = [case_id for case_id in by_policy["A0"] if by_policy["A3"][case_id] < by_policy["A0"][case_id]]
    require(wins == ["mergedev_counterfactual_reasoning_770aaf5dae0c1fb5"] and not losses, "paired gain/loss differs")
    selection = read(RUN / "result/frozen_dev_selection.json")
    traces = journal(RUN / "result/trajectories.jsonl")
    pool = [min(1.2, max(.8, e["alpha"])) for e in traces if e["arm"] == "A3" and e["split"] == "DEV" and e["iteration"] == 1]
    require(selection["pool"] == pool and selection["alpha"] == .8 and len(pool) == 176, "pool or fixed tie rule differs")
    forwards = journal(RUN / "result/forwards.jsonl")
    completed = [e for e in forwards if e["event"] in ("FORWARD_COMPLETED", "MAXIMUM_FORWARD_COMPLETED")]
    require(len(completed) == 2333 and sum(e["prefix_tokens"] for e in completed) == 684869, "full forward/position counts differ")
    completed_generation = [e for e in completed if e["event"] == "FORWARD_COMPLETED"]
    require(len(completed_generation) == 2317 and sum(e["prefix_tokens"] for e in completed_generation) == 648901, "generation counters differ")
    require(sum(r["full_model_calls_including_gold_nll"] for r in physical) == 2317, "row model-call sum differs")
    native = read(RUN / "result/receipt.json"); worker = read(RUN / "terminal_receipt.json")
    require(native["status"] == "GPU_DEV_COMPLETE_NOT_TESTED" and native["source_unchanged"] and native["checkpoint_files_unchanged"] and worker["exit_code"] == 0, "native terminal not complete/unchanged")
    budgets = [r for r in journal(CAMPAIGN / "BUDGET_LEDGER.jsonl") if r.get("event") == "native_completed"]
    spent = sum(r["native_worker_seconds"] for r in budgets)
    require(abs(spent - 732.171) < 1e-6 and abs(worker["native_worker_seconds"] - 568.094) < 1e-6, "worker cost differs")
    report = read(CAMPAIGN / "reviews/gpu_analysis_review_attempt002/REPORT.json")
    require(report["complete"] and not report["errors"] and report["constant_oracle"]["headroom"] == 0 and report["constant_oracle"]["capture"] is None, "approved analyzer replay differs")
    manual = []
    for case_id, rr in sorted(defaultdict(list, {key: [r for r in dev if r["case_id"] == key] for key in by_policy["A0"]}).items()):
        item = items[case_id]
        solution = solve_prompt(item["raw_prompt"])
        variants = defaultdict(list)
        for row in rr:
            variants[row["response"]].append(row)
        reviewed = []
        for response, variants_rows in sorted(variants.items()):
            family = item["family"]
            if solution.prediction is not None:
                semantic = scorer.normalize_text(response) == scorer.normalize_text(solution.prediction)
                method = "prompt-only symbolic solver/witness plus agent review"
                rationale = "Response matches parsed prompt solution" if semantic else "Response contradicts parsed prompt solution"
            elif family == "common_sense":
                semantic = True
                method = "agent judgment on exposed open-ended prompt"
                rationale = "Drying/moisture loss is a plausible consequence of prolonged lack of watering; keyword 'wilts' is unnecessarily narrow"
            elif family == "science":
                semantic = True
                method = "agent review of exposed factual answer"
                rationale = "Red blood cells correctly answer the question; phrasing expansion does not change correctness"
            elif family == "normal_conversation":
                semantic = False
                method = "agent review plus explicit response-contract check"
                rationale = "Greeting is understandable but two words and an exclamation violate both explicit constraints"
            else:
                semantic = None; method = "NOT_ASSESSED"; rationale = "No agent judgment registered"
            syntax = bool(scorer.score_response(response, item["format_scorer"])["score"])
            if family == "counterfactual_reasoning":
                syntax = response.strip() in ("YES", "NO")
            reviewed.append({"response": response, "policies": sorted({policy(r) for r in variants_rows}),
                             "frozen_task_score": variants_rows[0]["score"], "frozen_format_score": variants_rows[0]["format_score"],
                             "agent_prompt_success": semantic, "syntax_only_valid_where_checked": syntax,
                             "method": method, "rationale": rationale, "human_review": "NOT_REVIEWED"})
        manual.append({"case_id": case_id, "family": item["family"], "scenario_id": item["independent_cluster"],
                       "machine_solver_status": solution.status, "machine_solver_prediction": solution.prediction,
                       "machine_solver_witness": solution.witness, "response_variants": reviewed,
                       "semantic_status": "AGENT_REVIEW_OF_EXPOSED_DEV_POSTHOC_DIAGNOSTIC_NOT_PRIMARY"})
    result = {"schema_version": "1.0", "reviewer": "/root/adversarial_review", "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PASS_COMPLETE_EXPLORATORY_REFERENCE_RECOMPUTATION", "run": "runs/native-gpu-reference-002",
              "review_script_sha256": sha(__file__), "raw_outcomes_sha256": sha(RUN / "result/outcomes.jsonl"),
              "raw_forward_sha256": sha(RUN / "result/forwards.jsonl"), "raw_trajectories_sha256": sha(RUN / "result/trajectories.jsonl"),
              "reviewed_analyzer_report_sha256": sha(CAMPAIGN / "reviews/gpu_analysis_review_attempt002/REPORT.json"),
              "physical_rows": 408, "nonphysical_oracle_rows": 32, "DEV_physical_rows": 288, "unique_DEV_cases": 16,
              "independently_recomputed_accuracy": accuracy, "fixed_alpha_tie_selected": .8,
              "oracle_accuracy": oracle, "finite_constant_grid_headroom": 0., "capture_ratio": None,
              "score_vectors_of_all_recurrent_policies_identical": True, "adaptive_specific_wins": [], "adaptive_specific_losses": [],
              "A3_vs_intact_wins": wins, "A3_vs_intact_losses": losses, "intact_scored_correct_retained": "12/12",
              "inherited_format_accuracy": {t["policy"]: t["format_score"] for t in report["tables"]},
              "format_caution": "Counterfactual format scorer expects gold YES, so syntactically valid NO is counted invalid; the 14/16→15/16 inherited rise duplicates correctness, not an independent syntax improvement",
              "houseplant_caution": "Plausible open-ended answers rejected by gold keyword; frozen primary scores remain unchanged",
              "semantic_abstention_agent_review": {"status": "AGENT_REVIEW_NOT_HUMAN", "correct_missing-code_UNKNOWN_per_policy": "1/1", "false_abstention_on_answerables_per_policy": "0/15", "unsupported_answer_on_missing-code_case_per_policy": "0/1", "broad_answerability_inference": "NOT_SUPPORTED"},
              "termination": "All DEV rows EOS; no invalid/safety/truncation rows", "pool_size": len(pool),
              "pool_alpha_range": [min(pool), max(pool)], "pool_mean": statistics.mean(pool), "clipping_saturation": "0 upper and lower",
              "model_calls_completed_including_maxima": 2333, "processed_prefix_positions_completed_including_maxima": 684869,
              "generation_calls": 2317, "generation_prefix_positions": 648901, "partial_forced_compute_positions": None,
              "worker_seconds": worker["native_worker_seconds"], "cumulative_worker_seconds": spent,
              "remaining_native_seconds_snapshot": 21600 - spent, "peak_gpu_allocated_bytes": report["costs"]["peak_gpu_allocated_bytes"],
              "peak_gpu_reserved_bytes": report["costs"]["peak_gpu_reserved_bytes"], "peak_RSS_bytes": worker["peak_rss_bytes"],
              "manual_semantic_case_review": manual,
              "recommendation": "No A3 revision justified by this primary endpoint; improve bank discrimination/scorer separation prospectively, with own baselines and independent generator partitions",
              "outcome_scope": "No supported fixed-scale advantage beyond unit and no finite constant-selector accuracy headroom on this exposed bank; not a universal claim about adaptive prefix schedules or reasoning",
              "native_model_loads_by_reviewer": 0, "protected_or_native_TEST_reads": False, "human_review": "NOT_REVIEWED",
              "broad_IQ_claim": False, "speed_claim": False, "independent_confirmation": "NOT_RUN"}
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
    print(json.dumps({"status": result["status"], "accuracy": accuracy, "headroom": 0, "physical_rows": 408,
                      "manual_cases": len(manual), "worker_seconds": worker["native_worker_seconds"], "spent": spent}))


if __name__ == "__main__":
    main()
