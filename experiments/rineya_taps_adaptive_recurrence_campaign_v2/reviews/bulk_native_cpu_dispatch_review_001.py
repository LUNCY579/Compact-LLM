"""Independent prospective bulk dispatch closure; stdlib/hash only, no models."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

C = Path(__file__).resolve().parent.parent
ROOT = C.parents[1]
EXPECTED = "a303cd65c353a6d12e6544503bbf210c054ae919f795577d8b01e96224464aca"
GIB = 1024**3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""): h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(ok, why):
    if not ok: raise RuntimeError(why)


def tests(path, number):
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    totals = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    require(totals == {"tests": number, "failures": 0, "errors": 0, "skipped": 0}, "independent CPU tests failed")
    return {"counts": totals, "junit_sha256": sha(path)}


def main():
    target = C / "reviews/bulk_dispatch_receipt.json"
    require(not target.exists(), "exclusive dispatch receipt already exists")
    admission_path = C / "bulk_admission_v1.json"
    require(sha(admission_path) == EXPECTED, "bulk admission changed")
    a = read(admission_path)
    require(a["status"] == "READY_FOR_BULK_FP32_DISPATCH" and a["review_receipt"] == "reviews/bulk_dispatch_receipt.json", "wrong bulk dispatch contract")
    require(len(a["dispatch_hashes"]) == 84, "source closure differs")
    inherited = ("experiments/rineya_taps_adaptive_recurrence_v1/", "experiments/anchor_routed_middle_recurrence_v1/src/",
                 "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/")
    smoke_key = "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/supa_layer_ablation_smoke.jsonl"
    for relative, digest in a["dispatch_hashes"].items():
        path = (ROOT / relative).resolve(strict=True)
        allowed = C in path.parents or relative.endswith(".py") and relative.startswith(inherited) or relative == smoke_key
        require(allowed and path.is_relative_to(ROOT) and sha(path) == digest, "frozen bulk source mismatch: " + relative)
    require(not any("mergedev_fast.jsonl" in relative for relative in a["dispatch_hashes"]), "new worker should not bind/read original DEV data")
    require(sha(C / "bulk_native.py") == "1ed52ccbb546d826caec2bb48b3c7fa76dd5a136f1a1d7e3fae421b4871c46aa", "final native candidate changed")
    require(sha(C / "gpu_native.py") == a["gpu_helpers_sha256"] == "4c1a05c6511ff267d790654475f2a22cb5a3b8fa02857b295aa1ab3ae6c20e6f", "qualified forward helpers changed")
    original = read(C / "runs/binding-001/MODEL_DATA_BINDING.json")
    require(a["source"] == original["source_path"] and a["checkpoint_files"] == original["checkpoint_files"], "canonical model binding differs")
    for name, digest in a["checkpoint_files"].items():
        require(Path(name).name == name and sha(Path(a["source"]) / name) == digest, "canonical asset mismatch: " + name)
    design = read(C / "reviews/bulk_design_review_001.json")
    decoded = read(C / "reviews/bulk_binding_cpu_receipt_001.json")
    require(design["status"] == "PASS_BULK_SOURCE_DATA_APPARATUS_NOT_NATIVE_ADMITTED" and len(design["independent_prompt_only_witnesses"]) == 288,
            "independent bank/witness review missing")
    require(decoded["status"] == "PASS_BULK_NO_MODEL_BINDING_SOURCE_AND_RAW_ID_AUDIT" and decoded["independent_decoded_input_id_rows"] == 288,
            "independent rendered ID audit missing")
    bank_path, token_path = Path(a["bulk_bank_path"]), Path(a["bulk_token_binding_path"])
    bank, token = read(bank_path), read(token_path)
    require(sha(bank_path) == design["bank_sha256"] == a["bulk_bank_sha256"], "bank changed")
    require(sha(token_path) == decoded["binding_sha256"] == a["bulk_token_binding_sha256"] == "1a3723cdc918bb219119044aef5d209334f36c4b140674b8b93174f5361553bf", "final token binding changed")
    require(a["bulk_normalizer_source_sha256"] == token["normalizer_source_sha256"] == sha(C / "open_bank.py"), "normalizer closure differs")
    require(a["actual_task_shapes"] == token["shapes"] == {"batch": 1, "padding": False, "cache": False,
            "max_input_tokens": 232, "max_output_tokens": 8, "max_forward_prefix_tokens": 239}, "actual shapes differ")
    require(len(bank) == len(token["rows"]) == 288 and len({r["parent_id"] for r in bank}) == 144, "parent/case contract differs")
    forecast = {"scientific_physical_evaluations": 288*11, "SMOKE_physical_evaluations": 8*16,
                "physical_case_evaluations": 288*11+8*16, "maximum_generation_forwards": 288*11*8+8*16*24,
                "maximum_shape_forward_attempts": 18, "cache_rejection_attempts": 2,
                "maximum_model_API_attempts": 28416+18+2, "oracle_nonphysical_rows": 288,
                "frozen_validation_alias_nonphysical_rows": 96,
                "bulk_generation_prefix_positions": 11*sum(8*r["input_tokens"]+28 for r in token["rows"]),
                "maximum_shape_attempted_prefix_positions": 18*357}
    metadata = read(C / "runs/binding-001/PROMPT_AUDIT.json")
    smoke = [r for r in metadata if r["split"] == "SMOKE"]
    require(len(smoke) == 8, "SMOKE metadata differs")
    forecast["SMOKE_generation_prefix_positions"] = 16*sum(24*r["input_tokens"]+276 for r in smoke)
    forecast["maximum_generation_prefix_positions"] = forecast["bulk_generation_prefix_positions"] + forecast["SMOKE_generation_prefix_positions"]
    require(all(a["forecast"][key] == value for key, value in forecast.items()), "complete count/position forecast differs")
    require(forecast["maximum_generation_prefix_positions"] == 3609560, "generation position total differs")
    require(a["forecast"]["fresh_complete_forecast_required"] is True and a["forecast"]["forecast_factor"] == 1.5
            and a["maximum_shape_qualification_status"] == "REQUIRED_FRESH_NATIVE_NOT_RUN", "fresh qualification/forecast required")
    contract = {"device": "cuda:0", "dtype": "float32", "attn_implementation": "eager", "TF32": False,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "deterministic_algorithms": True,
                "training": False, "native_test": False, "optimizer_steps": 0, "controller_revisions": 0,
                "cloud_spend": 0, "infrastructure_retry_available": False,
                "max_new_tokens": 8, "max_bulk_input_tokens": 350, "qualified_maximum_prefix_tokens": 357,
                "hard_deadline_seconds": 18000, "idle_forward_deadline_seconds": 950,
                "native_finalization_reserve_seconds": 1800, "parent_host_available_floor_bytes": 7*GIB,
                "worker_host_available_floor_bytes": 6*GIB, "worker_rss_ceiling_bytes": 6*GIB,
                "gpu_initial_free_floor_bytes": 6*GIB, "gpu_allocated_ceiling_bytes": 5*GIB,
                "gpu_global_free_reserve_bytes": GIB, "validation_independent_world_units": 48}
    require(all(a[key] == value for key, value in contract.items()), "scientific/resource/budget scope differs")
    require(a["static_original_A3_mean"] == read(C / "reviews/gpu_analysis_review_attempt002_receipt.json")["pool_mean"], "sixth constant changed")
    all_events = [json.loads(line) for line in (C / "BUDGET_LEDGER.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    spent = sum(row["native_worker_seconds"] for row in all_events if row.get("event") == "native_completed")
    require(abs(spent - 732.171) < 1e-6 and spent + 18000 + 1800 <= 21600, "full native allocation exceeds grant")
    require(not (C / "native_worker.lock").exists(), "a native worker lock exists")
    cpu_initial = tests(C / "reviews/bulk_native_cpu_junit_001.xml", 7)
    cpu_final = tests(C / "reviews/bulk_native_cpu_junit_002.xml", 6)
    # Recompute the source-blind historical feasibility packet; timing selection
    # uses shape/completion only and excludes no quality outcomes.
    forward_path = C / "runs/native-gpu-reference-002/result/forwards.jsonl"
    journal = [json.loads(line) for line in forward_path.read_text().splitlines() if line.strip()]
    compatible = [event for event in journal if event.get("event") in ("FORWARD_COMPLETED", "MAXIMUM_FORWARD_COMPLETED")
                  and 1 <= event.get("prefix_tokens", 0) <= 357]
    require(len(compatible) == 2065 and max(event["prefix_tokens"] for event in compatible) == 88, "historical timing shape scope differs")
    first_cold = compatible[0]["forward_seconds"]
    warm_max = max(event["forward_seconds"] for event in compatible[1:])
    require(abs(first_cold-.5056918) < 1e-8 and abs(warm_max-.1467086) < 1e-8, "prior timing constants differ")
    provisional = 28436*warm_max*1.5 + first_cold + 200 + 1800
    require(provisional < 18000, "provisional complete forecast cannot fit")
    result = {"schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/adversarial_review",
        "status": "PASS_BULK_FP32_DISPATCH", "admission": admission_path.name, "admission_sha256": EXPECTED,
        "review_script_sha256": sha(__file__), "verified_dispatch_hashes": a["dispatch_hashes"], "verified_checkpoint_assets": a["checkpoint_files"],
        "independent_initial_native_parent_CPU_tests": cpu_initial, "independent_final_native_CPU_tests": cpu_final,
        "source_data_review_sha256": sha(C / "reviews/bulk_design_review_001.json"),
        "rendered_ID_review_sha256": sha(C / "reviews/bulk_binding_cpu_receipt_001.json"),
        "independently_recomputed_full_forecast": forecast,
        "source_blind_provisional_cost": {"historical_forward_journal_sha256": sha(forward_path), "compatible_completed_calls": 2065,
            "observed_prefix_range": [24, 88], "unmeasured_prefix_range": [89, 357], "first_cold_seconds": first_cold,
            "subsequent_maximum_seconds": warm_max, "complete_API_cap_factor": 1.5,
            "cold_setup_allowance_seconds": 200, "finalization_reserve_seconds": 1800, "provisional_total_seconds": provisional,
            "status": "FEASIBILITY_ONLY_FRESH_NATIVE_MAX357_FULL_MATRIX_FORECAST_MANDATORY"},
        "native_seconds_used": spent, "remaining_native_seconds": 21600-spent,
        "native_model_loads_by_reviewer": 0, "native_calls_by_reviewer": 0, "real_GPU_inventory_calls_by_reviewer": 0,
        "native_qualification": "NOT_RUN", "known_resources_at_freeze": {"available_GPU_bytes": a["gpu_inventory_at_freeze"]["free_bytes"],
            "required_parent_GPU_bytes": 6*GIB, "launch_allowed_from_that_snapshot": False},
        "scientific_limits": ["Only48 parent-world units validate theDEV-selected six-point comparator; source generators remain shared and exposed",
            "All288 case/variant harms remain visible; two seeds and variants average within parent, not additional independent sources",
            "Five-point practical margin is unchanged; small gains/tight preservation may remain unresolved; no confirmatory broad inference",
            "Finite per-case constant oracle and validation/family envelopes are diagnostic; capture is null for nonpositiveheadroom, otherwise unclipped",
            "A3 remains unchanged; no fitted controller/model training, cloud spend or empirical allocation-efficiency claim",
            "Late A5 phase and instrumented shared-GPU timing prevent deployment speed claims"],
        "conditions_before_scientific_measurements": ["Fresh exact UUID/driver and>=7GiBhost/6GiBGPU parent floors; failed resource guard must create no child",
            "Contained suspended bootstrap assigned to Windows Job before imports; sole-worker lock and18000hard/950idle deadlines",
            "Fresh max357/noextra/unit/originalA3/sixconstants/exception/cache/source/state/RMS/FP32/positions/anchors qualification",
            "Fresh fully completed warm/max357 timing must forecast the complete remaining matrix before anyDEV outcome; firstcold chargedonce inelapsed",
            "Exact DEV fixed/pool/source/parent freeze before anyvalidation forward; actual17/29 A5seed streams and frozen pool qualification",
            "All failures/partial work retained, cost missing fields null, no survivors/truncation/retry; separate analyzer review required before outcome use"],
        "unresolved_dispatch_source_blockers": []}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": result["status"], "admission_sha256": EXPECTED, "dispatch_hashes": 84,
                      "API_cap": 28436, "generation_positions": 3609560, "provisional_cost_seconds": provisional,
                      "resources_available_at_freeze": False}))


if __name__ == "__main__": main()
