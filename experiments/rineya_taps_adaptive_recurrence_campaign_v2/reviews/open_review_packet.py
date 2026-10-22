"""Independent source/data/token/resource OPEN dispatch receipt; no model import."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = Path(__file__).resolve().parent.parent
EXPECTED = "65e3e424d0d2b3c91905e99ec2dcb477e7d55ce4e9e56df623323c4b212b9c60"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def main():
    out = CAMPAIGN / "reviews/open_dispatch_receipt.json"
    require(not out.exists(), "exclusive OPEN review receipt exists")
    admission_path = CAMPAIGN / "open_admission_v2.json"
    require(sha(admission_path) == EXPECTED, "OPEN admission changed")
    admission = read(admission_path)
    require(admission["status"] == "READY_FOR_OPEN_FP32_DISPATCH" and admission["review_receipt"] == "reviews/open_dispatch_receipt.json", "wrong OPEN dispatch contract")
    require(len(admission["dispatch_hashes"]) == 78, "dispatch closure differs")
    for relative, expected in admission["dispatch_hashes"].items():
        path = (ROOT / relative).resolve(strict=True)
        require(path.is_relative_to(ROOT) and sha(path) == expected, "OPEN frozen source changed: " + relative)
    require(sha(CAMPAIGN / "open_native.py") == "4794c1bbc2e70275e3b7eef5c180752bc76734b8d67e10152c1a0d9de55c07c5", "final OPEN worker differs")
    require(sha(CAMPAIGN / "gpu_native.py") == admission["gpu_helpers_sha256"] == "4c1a05c6511ff267d790654475f2a22cb5a3b8fa02857b295aa1ab3ae6c20e6f", "qualified GPU helpers changed")
    junit_path = CAMPAIGN / "reviews/open_dispatch_cpu_junit.xml"
    xml = ET.parse(junit_path).getroot()
    suites = list(xml.iter("testsuite"))
    totals = {k: sum(int(s.get(k, "0")) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    require(totals == {"tests": 10, "failures": 0, "errors": 0, "skipped": 0}, "independent OPEN CPU tests failed")
    bank_path = Path(admission["open_bank_path"])
    bank = read(bank_path)
    require(sha(bank_path) == admission["open_bank_sha256"] and len(bank) == 24, "OPEN bank changed")
    manifest = read(bank_path.parent / "manifest.json")
    for name, expected in manifest["files"].items():
        require(Path(name).name == name and sha(bank_path.parent / name) == expected, "OPEN manifest file mismatch")
    require(sha(CAMPAIGN / "open_bank.py") == manifest["source_sha256"] == admission["open_source_sha256"], "scorer/generator source mismatch")
    token_path = Path(admission["open_token_binding_path"])
    token = read(token_path)
    require(sha(token_path) == admission["open_token_binding_sha256"] == "0fb77896268645c32b7b0f8c1d4b750f461c5bdf35e26a9bee40f691c531e7eb", "OPEN token binding differs")
    require(token["bank_sha256"] == sha(bank_path) and token["native_model_loads"] == 0, "token metadata misbound/loads model")
    rows = token["rows"]
    require(len(rows) == 24 and {r["case_id"] for r in rows} == {r["id"] for r in bank}, "token population differs")
    for row in rows:
        n = len(row["input_token_ids"])
        require(row["input_tokens"] == n and 1 <= n <= 350 and row["max_new_tokens"] == 8
                and row["max_forward_prefix_tokens"] == n + 7 and row["max_processed_prefix_positions"] == 8 * n + 28,
                "OPEN token shape/position formula differs")
    actual = {"batch": 1, "padding": False, "cache": False, "max_input_tokens": max(r["input_tokens"] for r in rows),
              "max_forward_prefix_tokens": max(r["max_forward_prefix_tokens"] for r in rows), "max_output_tokens": 8}
    require(actual == admission["actual_task_shapes"] and actual["max_input_tokens"] == 118 and actual["max_forward_prefix_tokens"] == 125, "actual shapes differ")
    old_prompts = read(CAMPAIGN / "runs/binding-001/PROMPT_AUDIT.json")
    smoke = [r for r in old_prompts if r["split"] == "SMOKE"]
    forecast = {"physical_case_evaluations": 24 * 10 * 2 + 8 * 16,
                "OPEN_case_evaluations": 480, "SMOKE_case_evaluations": 128,
                "maximum_generation_forwards": 24 * 10 * 2 * 8 + 8 * 16 * 24,
                "maximum_generation_prefix_positions": 20 * sum(r["max_processed_prefix_positions"] for r in rows)
                                                     + 16 * sum(r["maximum_processed_prefix_positions"] for r in smoke),
                "maximum_shape_forward_attempts": 18, "cache_rejection_attempts": 2,
                "maximum_shape_attempted_prefix_positions": 18 * 357, "maximum_model_api_attempts": 6932}
    require(all(admission["forecast"][k] == v for k, v in forecast.items()), "OPEN complete forecast differs")
    require(forecast["maximum_generation_prefix_positions"] == 477088, "OPEN bound position total differs")
    require(admission["static_original_A3_mean"] == read(CAMPAIGN / "reviews/gpu_analysis_review_attempt002_receipt.json")["pool_mean"], "sixth fixed comparator is not frozen original mean")
    resource = {"device": "cuda:0", "dtype": "float32", "attn_implementation": "eager", "TF32": False,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "max_new_tokens": 8, "max_open_input_tokens": 350,
                "qualified_maximum_prefix_tokens": 357, "training": False, "native_test": False,
                "controller_candidate_revision_count": 0, "infrastructure_retry_available": False,
                "hard_deadline_seconds": 14400, "native_finalization_reserve_seconds": 1800,
                "parent_host_available_floor_bytes": 7 * 1024**3, "worker_host_available_floor_bytes": 6 * 1024**3,
                "worker_rss_ceiling_bytes": 6 * 1024**3, "gpu_allocated_ceiling_bytes": 5 * 1024**3,
                "gpu_initial_free_floor_bytes": 6 * 1024**3, "gpu_global_free_reserve_bytes": 1024**3}
    require(all(admission[k] == v for k, v in resource.items()), "resource/authority/operator contract differs")
    costs = [json.loads(l) for l in (CAMPAIGN / "BUDGET_LEDGER.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    spent = sum(r["native_worker_seconds"] for r in costs if r.get("event") == "native_completed")
    require(abs(spent - 732.171) < 1e-6 and spent + 14400 + 1800 <= 21600, "OPEN deadline and reserve exceed grant")
    projected = 6932 * admission["forecast"]["previous_measured_max_forward_seconds"] * 1.5
    require(abs(projected - admission["forecast"]["raw_all_calls_conservative_seconds"]) < 1e-8 and projected + 200 + 1800 < 14400, "conservative OPEN forecast does not fit")
    require(not (CAMPAIGN / "native_worker.lock").exists(), "live native lock exists")
    result = {"schema_version": "1.0", "reviewer": "/root/adversarial_review", "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PASS_OPEN_FP32_DISPATCH", "admission": "open_admission_v2.json", "admission_sha256": EXPECTED,
              "review_script_sha256": sha(__file__), "checked_dispatch_hashes": admission["dispatch_hashes"],
              "independent_CPU_tests": {"counts": totals, "pytest_seconds": 12.96, "junit_sha256": sha(junit_path)},
              "independently_recomputed_forecast": forecast, "actual_shapes": actual,
              "budget_spent_snapshot_seconds": spent, "remaining_native_seconds": 21600 - spent,
              "conservative_all_API_seconds": projected, "native_model_loads_by_reviewer": 0,
              "real_GPU_inventory_calls_by_reviewer": 0, "protected_or_native_TEST_reads": False,
              "checks": {"new_strict_data_scorer_cap_has_own_baselines": True,
                         "exact_packet_positive_control_and_token_ID_population": True,
                         "strict_answer_format_gold_independence": True,
                         "sixth_static_comparator_derived_from_prior_frozen_geometry_not_OPEN_labels": True,
                         "DEVELOPMENT192_unique_fixed_cells32_A3_generations_before_selection": True,
                         "generation_trace_case_seed_prefix_coverage_before_pool_freeze": True,
                         "immutable_DATA_code_token_config_freeze_before_validation_model_calls": True,
                         "fresh_inherited_smoke_and357_maxshape_identity_exception_cache_source_state_gates": True,
                         "actual_seed_corrected_A5_stream_smoke_maxshape_before_OPEN_A5": True,
                         "new_OPEN_bootstrap_status_and_reader_no_CPU_GPU_stage_fallback": True,
                         "contained_suspended_owned_tree_parent_deadline_resource_floors": True},
              "scope": "One coordinator-owned prospectively registered OPEN-data/scorer/output-cap intervention; fresh native qualification and complete-cost forecast must pass before own DEVELOPMENT/OPEN_VALIDATION matrix. No A3 parameter revision.",
              "limits": ["Dispatch PASS is not native OPEN qualification or efficacy",
                         "OPEN_VALIDATION is source-visible exploratory transfer, not independent confirmation",
                         "Eight generator names share four broad categories/code branches; only four variant clusters per split",
                         "Own-bank shortcut floors and exact response-format/answerability endpoints must accompany results",
                         "Original-bank score/format/semantic limitations remain preserved, not retroactively changed",
                         "Sole infrastructure retry was already used; no further native retry after another refusal/failure",
                         "Separate OPEN analyzer still requires review before outcome-dependent use"]}
    require(sha(admission_path) == EXPECTED, "OPEN admission changed during review")
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
    print(json.dumps({"status": result["status"], "admission_sha256": EXPECTED, "files": 78, "forecast": forecast, "spent": spent}))


if __name__ == "__main__":
    main()
