"""Read-only independent recomputation of saved metrics, scores, and gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LAB = REPO / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1"
sys.path.insert(0, str(LAB / "src"))
from rineya_cross_scale.evaluation.scoring import score_response


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonlines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def primary_gate(rows, baseline, c):
    ratios = [r[k] for r in rows for k in ("trajectory_min_rms_ratio", "trajectory_max_rms_ratio")]
    ratios += [h["rms_ratio_to_normal"] for r in rows for h in r["hidden"]]
    return (len(rows) == 8 and [r["case_id"] for r in rows] == [r["case_id"] for r in baseline]
            and not any(r["safety_stopped"] for r in rows)
            and c["hidden_rms_ratio_bounds"][0] <= min(ratios) <= max(ratios) <= c["hidden_rms_ratio_bounds"][1]
            and mean(r["logit_cosine"] for r in rows) >= c["minimum_mean_logit_cosine"]
            and mean(r["kl_base_to_arm"] for r in rows) <= c["maximum_mean_kl"]
            and mean(r["score"] for r in rows) >= mean(r["score"] for r in baseline)-c["maximum_score_drop"]
            and mean(r["invalid"] for r in rows) <= mean(r["invalid"] for r in baseline)+c["maximum_invalid_increase"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to((ROOT / "evidence").resolve()):
        raise ValueError("audit accepts only this experiment's evidence")
    m, d, c = read(run / "run_manifest.json"), read(run / "decision.json"), read(run / "protocol_snapshot.json")
    raw = jsonlines(run / "raw_results.jsonl")
    groups = defaultdict(list)
    for r in raw:
        groups[r["arm"]].append(r)
    checks = {"run_complete": m["status"] == d["status"] == "COMPLETE",
              "config_matches_started_and_finished": c == m["config"] == read(run / "started_manifest.json")["config"],
              "unique_arm_case_rows": len(raw) == len({(r["arm"], r["case_id"]) for r in raw}),
              "registered_source_hashes_match": all(sha(REPO/p) == h for p, h in m["sources"].items()),
              "dataset_hashes_match": all(sha(LAB/"data/evaluation"/p) == h for p, h in m["datasets"].items()),
              "supa_checkpoint_receipts_match": m["supa_files_before"] == m["supa_files_after"],
              "donor_checkpoint_receipts_match": m["donor_files_before"] == m["donor_files_after"],
              "loaded_states_match": m["supa_state_before"] == m["supa_state_after"] and m["donor_state_before"] == m["donor_state_after"],
              "no_backbone_training": m["trainable_parameters"] == m["gradient_tensors"] == 0,
              "peak_process_memory_within_budget": m["peak_rss_bytes"] <= c["max_rss_gib"] * 2**30,
              "restored_logits": m["restored_logits_bit_identical"],
              "native_supa_capture_exact": m["native_supa_capture_max_delta"] == 0,
              "identity_control_receipts": all(d["identity"].values()),
              "primary_gate_recomputed": primary_gate(groups["G_smoke"], groups["A0_smoke"], c) == d["primary_mechanical_pass"]}
    datasets = {stage: jsonlines(LAB / "data/evaluation" / file) for stage, file in
                (("smoke", "supa_layer_ablation_smoke.jsonl"), ("dev", "mergedev_fast.jsonl"))}
    case_maps = {stage: {r["case_id"]: r for r in rows} for stage, rows in datasets.items()}
    expected = [name+"_smoke" for name in ("A0", "Z", "I", "N", "R", "S", "G", "A1")]
    if d["development"]:
        expected += [name+"_dev" for name in ("A0", "G", "R", "S", "A1")]
    checks["registered_arm_order"] = list(groups) == expected
    checks["registered_case_order"] = all([r["case_id"] for r in rows] == [r["case_id"] for r in datasets[name.rsplit("_", 1)[1]]] for name, rows in groups.items())
    checks["outputs_equal_raw_receipts"] = all(read(run / (name+".json")) == rows for name, rows in groups.items())
    checks["scores_recomputed"] = all(score_response(r["response"], case_maps[name.rsplit("_", 1)[1]][r["case_id"]]["scorer"])["score"] == r["score"] for name, rows in groups.items() for r in rows)
    checks["format_scores_recomputed"] = all((score_response(r["response"], case_maps[name.rsplit("_", 1)[1]][r["case_id"]]["format_scorer"])["score"] if case_maps[name.rsplit("_", 1)[1]][r["case_id"]].get("format_scorer") else None) == r["format_score"] for name, rows in groups.items() for r in rows)
    checks["identity_outputs_and_logit_deltas"] = all([r["token_ids"] for r in groups[name+"_smoke"]] == [r["token_ids"] for r in groups["A0_smoke"]] and all(r["max_abs_logit_delta"] == 0 for r in groups[name+"_smoke"]) for name in ("Z", "I"))
    calibration = read(run / "calibration_inputs.json")
    train_scenarios = {r["scenario_id"] for r, train in zip(calibration["lengths"], calibration["train_prompt_mask"]) if train}
    hold_scenarios = {r["scenario_id"] for r, train in zip(calibration["lengths"], calibration["train_prompt_mask"]) if not train}
    checks["scenario_disjoint_holdout"] = not train_scenarios.intersection(hold_scenarios) and sorted(hold_scenarios) == calibration["holdout_scenarios"] and len(hold_scenarios) == c["holdout_scenarios"]
    checks["calibration_token_accounting"] = sum(r["retained_tokens"] for r in calibration["lengths"]) == calibration["train_tokens"]+calibration["heldout_tokens"]
    cal_rows = jsonlines(LAB / "data/evaluation/calibration.jsonl")
    train_ids = {r["case_id"] for r, train in zip(calibration["lengths"], calibration["train_prompt_mask"]) if train}
    cal_prompts = {r["prompt"] for r in cal_rows}
    overlap = {stage: {"exact_prompt_overlap_with_calibration": sum(r["prompt"] in cal_prompts for r in rows),
                       "cases_used_for_bridge_training": sum(r["case_id"] in train_ids for r in rows)}
               for stage, rows in datasets.items()}
    checks["development_prompts_not_calibration_prompts"] = overlap["dev"]["exact_prompt_overlap_with_calibration"] == 0
    fit = read(run / "bridge_fit.json")
    checks["local_bridge_and_activation_hashes"] = all(sha(ROOT/"local"/run.name/name) == h for name, h in fit["artifacts"].items())
    checks["development_gate_enforced"] = bool(d["development"]) == (d["primary_mechanical_pass"] and not any(r["safety_stopped"] for r in groups["A0_smoke"]))
    history = REPO / "experiments/anchor_routed_middle_recurrence_v1/evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z"
    historical_hashes = {}
    for name in ("A0_smoke", "A1_smoke", "A0_dev", "A1_dev"):
        if name in groups:
            path = history / "outputs" / (name+".json")
            prior, current = read(path), groups[name]
            historical_hashes[name] = sha(path)
            keys = ("case_id", "token_ids", "score", "logit_cosine", "kl_base_to_arm", "max_abs_logit_delta")
            checks[name+"_historical_identity"] = len(prior) == len(current) and all(all(a[k] == b[k] for k in keys) for a, b in zip(prior, current))
    for name, saved in d["development"].items():
        rows, baseline = groups[name+"_dev"], groups["A0_dev"]
        wins = [r["case_id"] for r, b in zip(rows, baseline) if r["score"] > b["score"]]
        losses = [r["case_id"] for r, b in zip(rows, baseline) if r["score"] < b["score"]]
        fam = defaultdict(list)
        for r, b in zip(rows, baseline):
            fam[r["family"]].append(r["score"]-b["score"])
        preserved = (sum(r["score"] for r in rows) >= sum(r["score"] for r in baseline)
                     and all(mean(v) >= -.1 for v in fam.values())
                     and sum(r["invalid"] for r in rows) <= sum(r["invalid"] for r in baseline)
                     and not any(r["safety_stopped"] for r in rows+baseline))
        checks[name+"_development_recomputed"] = (wins == saved["wins"] and losses == saved["losses"] and preserved == saved["preservation_pass"] and saved["correct"] == sum(r["score"] for r in rows))
    receipt = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
               "case_rows": len(raw), "arm_counts": dict(Counter(r["arm"] for r in raw)),
               "data_overlap": overlap,
               "historical_output_sha256": historical_hashes,
               "source_run": str(run), "auditor_sha256": sha(Path(__file__)),
               "source_manifest_sha256": sha(run/"run_manifest.json"),
               "raw_results_sha256": sha(run/"raw_results.jsonl")}
    audit = ROOT / "evidence" / ("audit_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    audit.mkdir(exist_ok=False)
    (audit / "audit.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print(f"AUDIT_DIR={audit}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
