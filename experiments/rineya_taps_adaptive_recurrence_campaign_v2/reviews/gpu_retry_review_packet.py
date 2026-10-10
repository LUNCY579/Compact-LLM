"""Source-only independent allocation-retry review; no native model invocation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = Path(__file__).resolve().parent.parent
EXPECTED = "62e14bc729b28ee8c3d47a48fc8a4cb1c868c5aedc6544cbf1589c014021e152"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    out = CAMPAIGN / "reviews/gpu_retry_dispatch_receipt.json"
    require(not out.exists(), "exclusive retry review exists")
    old = read(CAMPAIGN / "gpu_admission_v2.json")
    new_path = CAMPAIGN / "gpu_admission_v4.json"
    require(sha(new_path) == EXPECTED, "retry admission changed")
    new = read(new_path)
    allowed = {"dispatch_hashes", "forecast", "gpu_inventory_at_freeze", "hard_deadline_seconds", "infrastructure_retry_number",
               "registered_utc", "retry_lineage", "review_receipt"}
    differences = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
    require(differences <= allowed, "scientific setting differs: " + repr(differences - allowed))
    require(new["hard_deadline_seconds"] == 10800 and new["infrastructure_retry_number"] == 1,
            "not the sole 10800-second retry")
    require(new["review_receipt"] == "reviews/gpu_retry_dispatch_receipt.json", "retry review path differs")
    require(new["status"] == "READY_FOR_GPU_FP32_DISPATCH", "retry admission not ready")
    require(new["gpu_uuid"] == old["gpu_uuid"] and new["gpu_driver_version"] == old["gpu_driver_version"], "host identity changed")
    for key, value in old["forecast"].items():
        if key != "cost_estimate_status":
            require(new["forecast"][key] == value, "matrix forecast changes: " + key)
    doc = "experiments/rineya_taps_adaptive_recurrence_campaign_v2/GPU_HOST_REGISTRATION.md"
    amendment = "experiments/rineya_taps_adaptive_recurrence_campaign_v2/RESOURCE_AMENDMENT_001.md"
    for relative, digest in old["dispatch_hashes"].items():
        if relative != doc:
            require(new["dispatch_hashes"].get(relative) == digest, "scientific code/data source changed: " + relative)
    require(set(new["dispatch_hashes"]) - set(old["dispatch_hashes"]) == {amendment}, "unexpected retry source expansion")
    require(len(new["dispatch_hashes"]) == 71, "retry dispatch closure differs")
    for relative, digest in new["dispatch_hashes"].items():
        require(sha(ROOT / relative) == digest, "frozen retry source changed: " + relative)
    require(sha(CAMPAIGN / "gpu_native.py") == "4c1a05c6511ff267d790654475f2a22cb5a3b8fa02857b295aa1ab3ae6c20e6f", "worker changed")
    run = CAMPAIGN / "runs/native-gpu-reference-001"
    qualification = read(run / "result/qualification.json")
    forecast = read(run / "result/matrix_forecast.json")
    native = read(run / "result/receipt.json")
    worker = read(run / "terminal_receipt.json")
    rows = [json.loads(line) for line in (run / "result/outcomes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    forwards = [json.loads(line) for line in (run / "result/forwards.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == 112 and all(row["split"] == "SMOKE" and row["physical_run"] for row in rows), "DEV outcomes or unregistered partial rows exist")
    require(not (run / "result/frozen_dev_selection.json").exists(), "outcome-dependent fixed/pool selection exists")
    state = new["expected_loaded_state_sha256"]
    require(qualification["status"] == "QUALIFIED_GPU_FP32_REFERENCE_OPERATOR" and qualification["before_state_sha256"] == qualification["after_state_sha256"] == state,
            "native pre-DEV state qualification missing")
    require(qualification["checkpoint_files_unchanged"] and qualification["original_new_A3_identity"] == "PASS"
            and qualification["scaled_arms_smoke"] == "PASS" and qualification["A5_frozen_pool_qualification"] == "PENDING_DEV_POOL",
            "native/reference qualification or A5 scope differs")
    require(qualification["maximum_shape"][0]["identity"] == qualification["maximum_shape"][0]["forced_exception_restoration"] == qualification["maximum_shape"][0]["cache_rejection"] == "PASS",
            "native maximum/restoration/cache gate failed")
    require(native["native_model_loads"] == 1 and native["trainable_parameters"] == 0 and native["training"] is False,
            "native load/training contract differs")
    require("forecast does not fit" in native["error"] and forecast["admitted"] is False and worker["exit_code"] == 2,
            "failure not the identified allocation refusal")
    total = forecast["already_elapsed_worker_seconds"] + forecast["remaining_forecast_seconds"] + forecast["finalization_reserve_seconds"]
    require(abs(forecast["remaining_forecast_seconds"] - forecast["remaining_generation_and_maximum_calls_cap"] * forecast["max_observed_synchronized_forward_seconds"] * 1.5) < 1e-8,
            "qualified complete forecast arithmetic differs")
    require(forecast["finalization_reserve_seconds"] == 1800 and forecast["parent_deadline_seconds"] == 7200
            and total > 7200 and total < 10800, "allocation/reserve interpretation differs")
    completed = [json.loads(line) for line in (CAMPAIGN / "BUDGET_LEDGER.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    completed = [event for event in completed if event.get("event") == "native_completed"]
    require(len({r["run"] for r in completed}) == len(completed), "duplicate native cost charges")
    spent = sum(event["native_worker_seconds"] for event in completed)
    require(abs(spent - 164.077) < 1e-6 and 10800 + 1800 <= 21600 - spent, "remaining campaign budget does not fit retry")
    require(not (CAMPAIGN / "native_worker.lock").exists(), "live worker lock exists")
    identities = []
    for case_id in {r["case_id"] for r in rows}:
        for extra in (0, 1):
            a = next(r for r in rows if r["case_id"] == case_id and r["arm"] == "qualification_legacy" and r["extra_loops"] == extra)
            b = next(r for r in rows if r["case_id"] == case_id and r["arm"] == "qualification_new" and r["extra_loops"] == extra)
            require(a["token_ids"] == b["token_ids"] and a["response"] == b["response"] and a["termination"] == b["termination"], "raw legacy/new identity differs")
            identities.append([case_id, extra, "legacy_new"])
        a = next(r for r in rows if r["case_id"] == case_id and r["arm"] == "qualification_reference_A3")
        b = next(r for r in rows if r["case_id"] == case_id and r["arm"] == "qualification_successor_A3")
        require(a["token_ids"] == b["token_ids"] and a["response"] == b["response"] and a["termination"] == b["termination"], "raw reference/new A3 identity differs")
        identities.append([case_id, 1, "A3_reference_successor"])
    counts = {kind: sum(event.get("event") == kind for event in forwards) for kind in
              ("FORWARD_STARTED", "FORWARD_COMPLETED", "MAXIMUM_FORWARD_STARTED", "MAXIMUM_FORWARD_COMPLETED", "MAXIMUM_FORWARD_FAILED", "CACHE_REJECTION_STARTED", "CACHE_REJECTED")}
    require(counts["MAXIMUM_FORWARD_STARTED"] == 16 and counts["MAXIMUM_FORWARD_COMPLETED"] == 15
            and counts["MAXIMUM_FORWARD_FAILED"] == 1 and counts["CACHE_REJECTED"] == 2, "partial qualification attempt counters differ")
    result = {"schema_version": "1.0", "reviewer": "/root/adversarial_review", "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PASS_GPU_FP32_DISPATCH", "admission": "gpu_admission_v4.json", "admission_sha256": EXPECTED,
              "review_script_sha256": sha(__file__), "retry_number": 1,
              "source_checked_dispatch_hashes": new["dispatch_hashes"], "unchanged_scientific_source_contract": True,
              "allowed_changed_admission_keys": sorted(differences), "native_worker_source_sha256": sha(CAMPAIGN / "gpu_native.py"),
              "raw_native_qualification_receipt_sha256": sha(run / "result/qualification.json"),
              "raw_forecast_sha256": sha(run / "result/matrix_forecast.json"), "raw_worker_receipt_sha256": sha(run / "terminal_receipt.json"),
              "raw_outcomes_sha256": sha(run / "result/outcomes.jsonl"), "raw_forward_journal_sha256": sha(run / "result/forwards.jsonl"),
              "raw_identity_pairs_recomputed": len(identities), "raw_forward_attempt_counts": counts,
              "raw_complete_logit_bitwise_equality": "arrays unlogged; executed pinned source comparison plus qualification PASS, not independently byte-recomputed",
              "measured_qualification_status": "PASS_PRE_DEV_REFERENCE_MECHANICS_A5_ACTUAL_POOL_PENDING", "DEV_rows_before_amendment": 0,
              "native_worker_spent_seconds": spent, "remaining_campaign_seconds": 21600 - spent,
              "independently_recomputed_required_old_allocation_seconds": total, "old_allocation_shortfall_seconds": total - 7200,
              "new_worker_deadline_seconds": 10800, "finalization_reserve_seconds": 1800,
              "reason": "identified infrastructure/per-job resource allocation forecast refusal before DEV; scientific configuration/code/data/scorer/controller/endpoint unchanged",
              "research_policy_v4_review": {"status": "ACCEPT_BOUNDED_PROSPECTIVE_ALLOCATION_RULE", "acceptance": "164.077 spent and 7259.4413448 required independently reproduced; unchanged scientific hashes; 10800+1800 fits remaining allowance", "empirical_allocation_efficiency": "NOT_MEASURED"},
              "scope": "Sole infrastructure retry in a new exclusive run directory; requalify same exact GPU host and keep forecast factor/reserves/gates. No further CPU or GPU retry after this, and no numerical/scientific failure retry.",
              "native_model_loads_by_reviewer": 0, "real_gpu_inventory_calls_by_reviewer": 0, "protected_or_native_TEST_reads": False,
              "efficacy_claim": "NOT_RUN_NO_DEV_ROWS", "independent_replication": "NOT_RUN"}
    require(sha(new_path) == EXPECTED, "retry admission changed during review")
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
    print(json.dumps({"status": result["status"], "admission_sha256": EXPECTED, "spent_seconds": spent,
                      "old_required_seconds": total, "remaining_seconds": 21600 - spent, "identity_pairs": len(identities)}))


if __name__ == "__main__":
    main()
