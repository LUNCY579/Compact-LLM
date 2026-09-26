"""Audit the preregistered time-only rerun against its immutable predecessor."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parents[1] / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src")]
from armr.preregistration import budget_only_config
from armr.provenance import sha256, write_json
from armr.reporting import csv_file, mean
from run_fixed_recurrence import development_result

LOGIT_FIELDS = ("logit_cosine", "kl_base_to_arm", "max_abs_logit_delta",
                "top1_agreement", "topk_overlap", "margin_change",
                "gold_token", "gold_rank", "baseline_gold_rank")


def compare_case(previous: dict, current: dict) -> dict:
    old_ids, new_ids = previous["token_ids"], current["token_ids"]
    censored = previous["termination"] == "timeout"
    equal = new_ids[:len(old_ids)] == old_ids if censored else new_ids == old_ids
    deltas = [abs(current[k] - previous[k]) for k in LOGIT_FIELDS
              if current[k] is not None and previous[k] is not None]
    same_nulls = all((current[k] is None) == (previous[k] is None) for k in LOGIT_FIELDS)
    return {"arm": current["arm"], "case_id": current["case_id"],
            "prior_termination": previous["termination"], "new_termination": current["termination"],
            "prior_tokens": len(old_ids), "new_tokens": len(new_ids),
            "comparison": "prior_prefix" if censored else "full_generation",
            "token_identity": equal, "max_first_logit_metric_difference": max(deltas, default=0.),
            "logit_metric_identity": same_nulls and max(deltas, default=0.) <= 1e-7,
            "prior_correct": previous["score"], "new_correct": current["score"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to((ROOT / "evidence/02c_fixed_cpu_budget_v2").resolve()):
        raise ValueError("expected registered CPU-budget run")
    manifest = json.loads((run / "run_manifest.json").read_text())
    registration = json.loads((run / "preregistration_snapshot.json").read_text())
    prior = (ROOT / registration["prior_run"]).resolve()
    old_manifest = json.loads((prior / "run_manifest.json").read_text())
    budget_only_config(old_manifest["config"], manifest["config"])
    previous = [json.loads(s) for s in (prior / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()]
    current = [json.loads(s) for s in (run / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()]
    old_keys = [(r["arm"], r["case_id"]) for r in previous]
    new_keys = [(r["arm"], r["case_id"]) for r in current]
    repo = ROOT.parents[1]
    registration_path = Path(manifest["preregistration"])
    committed = subprocess.check_output(["git", "-C", str(repo), "show",
        manifest["git_commit"] + ":" + registration_path.relative_to(repo).as_posix()])
    checks = {
        "preregistration_committed_before_execution": json.loads(committed) == registration,
        "preregistration_file_hash": sha256(registration_path) == manifest["preregistration_sha256"],
        "prior_receipt_hash": sha256(prior / "run_manifest.json") == registration["prior_manifest_sha256"],
        "exact_matrix_case_order": old_keys == new_keys and len(new_keys) == 72 and len(set(new_keys)) == 72,
        "selection_unchanged": manifest["selected_configuration"] == old_manifest["selected_configuration"],
        "canonical_state_unchanged": manifest["canonical_unchanged"] and manifest["state_hash_before"] == old_manifest["state_hash_before"],
        "checkpoint_files_unchanged": manifest["checkpoint_files_before"] == old_manifest["checkpoint_files_before"],
        "no_training": manifest["trainable_parameter_count"] == manifest["parameters_with_gradients"] == 0,
        "restored_logits_identical": manifest["decision"]["restored_logits_bit_identical"],
    }
    for k in ("torch", "transformers", "python", "qwen_implementation_sha256", "device", "dtype", "dataset_sha256"):
        checks[f"unchanged_{k}"] = manifest[k] == old_manifest[k]
    checks["preserved_source_and_data_hashes"] = all(sha256(repo / p) == h for p,h in registration["preserved_source_sha256"].items())
    old_map = {(r["arm"],r["case_id"]): r for r in previous}
    paired = [compare_case(old_map[(r["arm"],r["case_id"])],r) for r in current]
    checks["uncensored_ids_and_censored_prefixes_reproduce"] = all(r["token_identity"] for r in paired)
    checks["first_logit_metrics_reproduce"] = all(r["logit_metric_identity"] for r in paired)
    grouped = defaultdict(list)
    for row in current:
        grouped[row["arm"]].append(row)
    base = grouped["A0_dev"]
    quality = {arm: development_result(grouped[arm],base) for arm in ("A1_dev","A3_dev")}
    coverage = {}
    dynamics = []
    for arm in ("A0_dev","A1_dev","A3_dev"):
        rows = grouped[arm]
        stops = [r for r in rows if r["safety_stopped"]]
        coverage[arm] = {
            "items":len(rows), "completed":len(rows)-len(stops),
            "correct_among_completed":sum(r["score"] for r in rows if not r["safety_stopped"]),
            "eos":sum(r["eos_emitted"] for r in rows), "token_caps":sum(r["truncated"] for r in rows),
            "stopped_case_ids":[r["case_id"] for r in stops],
            "wall_seconds":sum(r["wall_seconds"] for r in rows),
            "maximum_case_seconds":max(r["wall_seconds"] for r in rows),
        }
        dynamics.append({"arm":arm,"mean_cosine":mean([r["logit_cosine"] for r in rows]),
            "mean_kl":mean([r["kl_base_to_arm"] for r in rows]),
            "minimum_rms_ratio":min(r["trajectory_min_rms_ratio"] for r in rows),
            "maximum_rms_ratio":max(r["trajectory_max_rms_ratio"] for r in rows),
            "gold_nll_cases":sum(r["gold_nll"] is not None for r in rows),
            "mean_gold_nll":mean([r["gold_nll"] for r in rows if r["gold_nll"] is not None]),
            "prefill_seconds":sum(r["prefill_seconds"] for r in rows),
            "decode_seconds":sum(r["decode_seconds"] for r in rows),
            "dense_matmul_flops":sum(r["generation_dense_matmul_flops_estimate"] for r in rows)})
    fidelity = all(checks.values())
    all_complete = all(c["completed"]==16 for c in coverage.values())
    router = fidelity and all_complete and any(r["router_scaffold_licensed"] for r in quality.values())
    status = "FAIL" if not fidelity else "INCONCLUSIVE" if not all_complete else "PASS"
    decision = {"status":status,"fidelity_checks":checks,"all_development_cases_complete":all_complete,
                "coverage":coverage,"comparisons":quality,"router_scaffold_gate":router,
                "router_implemented":False,"training_performed":False,"peft_recovery":False,"sealed_evaluation":False}
    out = ROOT / "evidence/02d_cpu_budget_v2_audit" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"decision.json",decision)
    csv_file(out/"prior_generation_identity.csv",paired)
    csv_file(out/"dynamics_efficiency.csv",dynamics)
    write_json(out/"source_manifest.json",{"source_run":str(run),"source_files_sha256":{
        str(p.relative_to(run)):sha256(p) for p in sorted(run.rglob("*")) if p.is_file()},
        "prior_run":str(prior),"prior_raw_sha256":sha256(prior/"raw_results.jsonl"),
        "script_sha256":sha256(Path(__file__)),"model_loaded":False})
    table = "\n".join(f"| {arm} | {c['completed']}/{c['items']} | {c['correct_among_completed']:.0f} | {c['eos']} | {c['token_caps']} | {c['maximum_case_seconds']:.1f} |" for arm,c in coverage.items())
    comparisons = "\n".join(f"- {arm} vs A0: {len(q['wins'])} paired wins, {len(q['losses'])} paired losses; preservation={q['preservation_pass']}; router scaffold gate={q['router_scaffold_licensed']}." for arm,q in quality.items())
    (out/"report.md").write_text(f"""# Registered CPU-budget replication audit

Status: {status} for execution and fidelity. Existing router-scaffold development gate: {router}.

Question: does the identical A0/A1/A3 matrix complete and preserve quality when
the generation deadline alone increases from 90 to 900 seconds per case?
Preregistration was committed before execution. Setup: same 16 MERGEDEV-fast
cases, eight smoke cases per arm, selected layers, CPU FP32/eager, four threads,
seed 17, greedy token caps, no cache, scoring and existing gates. The source
manifest identifies the exact execution commit, runtime and checkpoint hashes.

| Arm | Completed | Correct among completed | EOS | Token cap | Max case seconds |
|---|---:|---:|---:|---:|---:|
{table}

{comparisons}

Fidelity checks: {sum(checks.values())}/{len(checks)} pass. The prior run had
66 non-censored case-runs and six timeout case-runs across smoke/development.
The identity CSV compares complete token IDs for the former and old prefixes
for the latter, plus first-token logit metrics for all 72. Configuration,
data, source, runtime, selected arms and checkpoint/state invariants are checked
independently; a mismatching audit cannot license escalation.

Canonical restoration and frozen-parameter checks are in decision.json. Hidden,
NLL, prefill/decode, FLOP and RSS evidence is retained in the source run and
dynamics_efficiency.csv. Deadlines remain cooperative between Torch forwards;
the reporting latency also includes optional gold NLL. Token-cap completions
retain the original scoring policy and are identified separately from EOS.

Interpretation: this resolves the registered time-budget question only if
coverage and fidelity pass. A passing existing development gate licenses a
future scaffold experiment, not a routing or intelligence claim. All 16 cases
are development data previously inspected in the original run. There is no
sealed, training, compute-matched efficacy or long-context reasoning result.
The prior failed/inconclusive evidence remains unchanged.

Scope complete at this matrix and audit: no automatic router implementation,
PEFT, additional layer selection, or sealed evaluation was performed.
""",encoding="utf-8")
    print(out)
    return 0 if fidelity else 1


if __name__ == "__main__":
    raise SystemExit(main())
