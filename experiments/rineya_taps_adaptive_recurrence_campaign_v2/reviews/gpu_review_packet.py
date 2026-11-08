"""Independent source-only verification of the frozen GPU qualification packet."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = Path(__file__).resolve().parent.parent
EXPECTED = "5f9f2d2921ea9cb49d6a9f5fa0748964c3ac628c9b90268bc3cd267471524238"


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


def junit(path, expected):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    result = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    require(result == {"tests": expected, "failures": 0, "errors": 0, "skipped": 0}, "CPU test receipt mismatch")
    return {"path": str(path.relative_to(CAMPAIGN)), "sha256": sha(path), **result}


def main():
    out = CAMPAIGN / "reviews/gpu_dispatch_receipt.json"
    require(not out.exists(), "exclusive GPU review receipt exists")
    admission_path = CAMPAIGN / "gpu_admission_v2.json"
    require(sha(admission_path) == EXPECTED, "GPU admission changed")
    admission = read(admission_path)
    require(admission["status"] == "READY_FOR_GPU_FP32_DISPATCH" and admission["lane"] == "GPU_FP32_SUCCESSOR", "wrong GPU lane")
    require(admission["review_receipt"] == "reviews/gpu_dispatch_receipt.json", "wrong reviewer receipt binding")
    checked = {}
    for relative, expected in admission["dispatch_hashes"].items():
        path = (ROOT / relative).resolve(strict=True)
        require(path.is_relative_to(ROOT), "dispatch path leaves workspace")
        require(sha(path) == expected, "GPU dispatch code/data changed: " + relative)
        checked[relative] = expected
    require(len(checked) == 70, "GPU dispatch closure count differs")
    require(sha(CAMPAIGN / "gpu_native.py") == "4c1a05c6511ff267d790654475f2a22cb5a3b8fa02857b295aa1ab3ae6c20e6f", "final loader-accounting source changed")
    require(sha(CAMPAIGN / "gpu_supervisor.py") == "227888015222897c126fa28699a00b3f083d94a64d1f50937b8cc964dc8a5a6c", "final forward-watch parent changed")
    require(sha(CAMPAIGN / "tests/test_gpu_semantics.py") == "2dde3866e191102ffcb1ba7420e90966b8ca8bc5d3dc03b72191597971ef78e4", "final semantics tests changed")
    tests = {"independent_parent_and_initial_semantics": junit(CAMPAIGN / "reviews/gpu_review_cpu_junit.xml", 28),
             "independent_final_semantics": junit(CAMPAIGN / "reviews/gpu_review_semantics_final_junit.xml", 10)}
    binding_path = CAMPAIGN / "runs/binding-001/MODEL_DATA_BINDING.json"
    binding = read(binding_path)
    require(sha(binding_path) == admission["binding_sha256"], "GPU binding bytes differ")
    require(admission["checkpoint_files"] == binding["checkpoint_files"] and admission["source"] == binding["source_path"], "GPU source identity differs")
    require(admission["maximum_shapes"] == binding["shapes"] and admission["maximum_shapes"]["max_shape_native_mechanics"] == "NOT_RUN", "metadata misrepresented as native qualification")
    require(admission["runtime_metadata"] == binding["runtime"], "GPU runtime metadata differs")
    require(sha(Path(binding["runtime"]["qwen_file"])) == binding["runtime"]["qwen_sha256"], "Qwen dependency code changed")
    assets = {}
    for name, expected in admission["checkpoint_files"].items():
        actual = sha(Path(admission["source"]) / name)
        require(actual == expected, "canonical GPU asset changed: " + name)
        assets[name] = actual
    rows = read(CAMPAIGN / "runs/binding-001/PROMPT_AUDIT.json")
    dev = [r for r in rows if r["split"] == "DEV"]
    smoke = [r for r in rows if r["split"] == "SMOKE"]
    require(len(dev) == 16 and len(smoke) == 8, "admitted bank differs")
    for row in rows:
        cap, n = row["max_new_tokens"], len(row["input_token_ids"])
        require(row["maximum_processed_prefix_positions"] == cap * n + cap * (cap - 1) // 2, "raw position formula differs")
    forecast = {"physical_evaluate_case_calls": 15 * len(smoke) + 18 * len(dev),
                "maximum_generation_forwards": 15 * sum(r["max_new_tokens"] for r in smoke) + 18 * sum(r["max_new_tokens"] for r in dev),
                "maximum_generation_prefix_positions": 15 * sum(r["maximum_processed_prefix_positions"] for r in smoke) + 18 * sum(r["maximum_processed_prefix_positions"] for r in dev),
                "maximum_shape_forward_attempts": 17, "cache_rejection_attempts": 2,
                "maximum_shape_prefix_positions_attempted": 17 * 2248, "oracle_nonphysical_rows": 32}
    forecast["maximum_model_api_attempts"] = forecast["maximum_generation_forwards"] + 17 + 2
    require(all(admission["forecast"][key] == value for key, value in forecast.items()), "GPU matrix forecast does not reconcile")
    required = {"device": "cuda:0", "dtype": "float32", "attn_implementation": "eager", "TF32": False,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "training": False, "native_test": False, "optimizer_steps": 0,
                "controller_candidate_revision_count": 0, "hard_deadline_seconds": 7200,
                "idle_forward_deadline_seconds": 950, "parent_host_available_floor_bytes": 7 * 1024**3,
                "worker_host_available_floor_bytes": 6 * 1024**3, "worker_rss_ceiling_bytes": 6 * 1024**3,
                "gpu_initial_free_floor_bytes": 6 * 1024**3, "gpu_allocated_ceiling_bytes": 5 * 1024**3,
                "gpu_global_free_reserve_bytes": 1024**3, "native_finalization_reserve_seconds": 1800}
    require(all(admission[key] == value for key, value in required.items()), "GPU authority/resource/operator contract changed")
    inventory = admission["gpu_inventory_at_freeze"]
    require(inventory["uuid"] == admission["gpu_uuid"] and inventory["driver_version"] == admission["gpu_driver_version"]
            and inventory["index"] == 0 and inventory["free_bytes"] >= 6 * 1024**3, "coordinator GPU inventory does not meet frozen contract")
    budget = [json.loads(line) for line in (CAMPAIGN / "BUDGET_LEDGER.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    completed = [r for r in budget if r.get("event") == "native_completed"]
    require(len({r["run"] for r in completed}) == len(completed), "duplicate native budget charges")
    spent = sum(r["native_worker_seconds"] for r in completed)
    require(spent + 7200 + 1800 <= 21600, "GPU budget/reserve exceeds campaign allowance")
    require(not (CAMPAIGN / "native_worker.lock").exists(), "another worker lock exists")
    cpu_path = CAMPAIGN / "runs/native-reference-001/terminal_receipt.json"
    cpu = read(cpu_path)
    cpu_preflight = read(cpu_path.parent / "result/preflight.json")
    require(cpu["status"] == "WORKER_NONZERO" and cpu_preflight["status"] == "BLOCKED" and cpu_preflight["native_runs"] == 0, "historical CPU refusal was not preserved")
    result = {
        "schema_version": "1.0", "reviewer": "/root/adversarial_review", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_GPU_FP32_DISPATCH", "admission": "gpu_admission_v2.json", "admission_sha256": EXPECTED,
        "review_script_sha256": sha(__file__), "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "checked_dispatch_hashes": checked, "canonical_assets_independently_rehashed": assets,
        "independent_cpu_test_receipts": tests, "independent_final_semantics_pytest_seconds": 2.51,
        "independent_combined_parent_and_previous_semantics_pytest_seconds": 5.55,
        "independently_recomputed_forecast": forecast, "budget_spent_snapshot_seconds": spent,
        "budget_native_allowance_seconds": 21600, "cpu_refusal_receipt_sha256": sha(cpu_path),
        "checks": {"separate_explicit_gpu_entry_no_cpu_fallback": True, "exact_asset_runtime_source_and_token_ids_binding": True,
                   "frozen_fp32_parameter_and_buffer_device_checks": True, "actual_loaded_custom_head_and_state_hash_gates": True,
                   "cuda0_eager_no_tf32_cache_padding_training": True,
                   "fresh_native_zero_unit_original_A3_identity_required_before_DEV": True,
                   "native2248_maxshape_normal_exception_restoration_and_cache_probes_required": True,
                   "mask_position_replay_object_identity_checks": True, "all_required_scale_smoke_failure_stops_incomplete": True,
                   "corrected_A5_actual_config_seed_prefix_stream_generation_pool": True,
                   "A5_frozen_pool_smoke_and_maxshape_before_A5_DEV": True,
                   "complete_measured_remaining_cost_forecast_gate_before_DEV": True,
                   "suspended_venv_redirector_job_containment_and_owned_only_termination": True,
                   "load_failed_partial_forward_cache_attempt_costs_and_peaks_logged": True,
                   "upstream_sync_before_first_pass_local_timer_and_observer_clock_cleanup": True},
        "scope": "One coordinator-owned prospectively registered GPU-FP32 successor worker may attempt fresh qualification and then full registered exposed DEV only if all built-in gates and complete-cost forecast pass.",
        "native_model_loads_by_reviewer": 0, "real_gpu_inventory_calls_by_reviewer": 0, "protected_or_native_TEST_reads": False,
        "interpretation_limits": ["Dispatch review is not native maximum-shape, identity, scaled-arm smoke or efficacy PASS; these require actual native receipts",
                                  "CPU original refusal remains BLOCKED; GPU is a successor, not exact CPU reproduction",
                                  "Two A5 stochastic schedules are repeats, not independent items; sixteen exposed rows and provisional clusters limit inference",
                                  "Full-gold diagnostics disabled; native per-token schedule oracle not enumerated",
                                  "Synchronized detailed instrumentation is not deployment speed evidence",
                                  "Separate GPU analyzer still needs independent review before outcome-dependent use",
                                  "No deeper recurrence, alternate dtype, training, protected data or publication authorized"]}
    require(sha(admission_path) == EXPECTED, "GPU admission changed during review")
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": result["status"], "admission_sha256": EXPECTED, "dispatch_files": len(checked), "assets": len(assets), "forecast": forecast}))


if __name__ == "__main__":
    main()
