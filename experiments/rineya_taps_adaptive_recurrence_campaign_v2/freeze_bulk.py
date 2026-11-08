"""Coordinator source freeze for the prospective bulk matrix; no model imports."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .run_apparatus import PACKAGE, ROOT, sha


def main():
    target = PACKAGE / "bulk_admission_v1.json"
    if target.exists():
        raise FileExistsError("retain the prior freeze; use a new reviewed version")
    old = json.loads((PACKAGE / "open_admission_v2.json").read_text())
    bank_dir = PACKAGE / "runs/bulk-bank-apparatus-001"
    bank = json.loads((bank_dir / "bank.json").read_text())
    token_path = PACKAGE / "runs/bulk-tokenizer-binding-002/BULK_TOKENIZER_BINDING.json"
    token = json.loads(token_path.read_text())
    manifest = json.loads((bank_dir / "manifest.json").read_text())
    if (len(bank) != 288 or token["status"] != "BULK_TOKENIZER_BOUND_NATIVE_NOT_MEASURED"
            or manifest["status"] != "PASS_PREPARATION_ONLY"
            or token["bank_sha256"] != sha(bank_dir / "bank.json")
            or manifest["source_sha256"] != sha(PACKAGE / "bulk_bank.py")
            or sha(PACKAGE / "bulk_native.py") != "1ed52ccbb546d826caec2bb48b3c7fa76dd5a136f1a1d7e3fae421b4871c46aa"):
        raise RuntimeError("final author source/bank/token candidate differs")
    if json.loads((PACKAGE / "reviews/bulk_design_review_001.json").read_text())["status"].startswith("PASS") is not True:
        raise RuntimeError("bulk design independent review missing")
    # Keep exact inherited Python/SMOKE closure. The new worker does not read old DEV.
    prefixes = ("experiments/rineya_taps_adaptive_recurrence_v1/",
        "experiments/anchor_routed_middle_recurrence_v1/src/",
        "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/")
    smoke = "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/supa_layer_ablation_smoke.jsonl"
    dispatch = {name: value for name, value in old["dispatch_hashes"].items()
        if (name.endswith(".py") and name.startswith(prefixes)) or name == smoke}
    names = ["AUTHORITY.txt", "AUTHORITY_EXTENSION_001.md", "LARGE_BANK_DESIGN_001.md",
        "BULK_ANALYSIS_PLAN_001.md", "BULK_NATIVE_REGISTRATION_001.md", "BULK_NATIVE_DRAFT_001.md",
        "binding.py", "control.py", "gpu_native.py", "native_supervisor.py", "supervision.py", "run_apparatus.py",
        "open_bank.py", "bulk_bank.py", "bulk_binding.py", "bulk_entry.py", "bulk_supervisor.py", "bulk_native.py",
        "runs/binding-001/MODEL_DATA_BINDING.json", "runs/binding-001/PROMPT_AUDIT.json",
        "runs/bulk-bank-apparatus-001/bank.json", "runs/bulk-bank-apparatus-001/audit.json",
        "runs/bulk-bank-apparatus-001/positive_controls.json", "runs/bulk-bank-apparatus-001/manifest.json",
        "runs/bulk-tokenizer-binding-002/BULK_TOKENIZER_BINDING.json",
        "reviews/bulk_design_review_001.json", "reviews/bulk_binding_cpu_receipt_001.json",
        "analyses/gpu-reference-002/REPORT.json"]
    for name in names:
        path = PACKAGE / name
        dispatch[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, expected in dispatch.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("inherited/source freeze mismatch: " + name)
    keep = ["CUBLAS_WORKSPACE_CONFIG", "TF32", "attn_implementation", "checkpoint_files",
        "controller_candidate_revision_count", "deterministic_algorithms", "device", "dtype",
        "gpu_allocated_ceiling_bytes", "gpu_driver_version", "gpu_global_free_reserve_bytes",
        "gpu_helpers_sha256", "gpu_initial_free_floor_bytes", "gpu_uuid", "idle_forward_deadline_seconds",
        "infrastructure_retry_available", "interpreter", "native_finalization_reserve_seconds", "native_test",
        "parent_host_available_floor_bytes", "source", "static_original_A3_mean", "training",
        "worker_host_available_floor_bytes", "worker_rss_ceiling_bytes"]
    result = {key: old[key] for key in keep}
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits"], text=True, timeout=5).strip().splitlines()
    if len(inventory) != 1:
        raise RuntimeError("registered single GPU changed")
    values = [value.strip() for value in inventory[0].split(",")]
    gpu = {"index": int(values[0]), "uuid": values[1], "name": values[2],
        "total_bytes": int(values[3])*1024**2, "free_bytes": int(values[4])*1024**2,
        "utilization_percent": int(values[5]), "driver_version": values[6]}
    if gpu["uuid"] != result["gpu_uuid"] or gpu["driver_version"] != result["gpu_driver_version"]:
        raise RuntimeError("qualified GPU identity/driver changed; new review needed")
    prior_rows = json.loads((PACKAGE / "runs/binding-001/PROMPT_AUDIT.json").read_text())
    smoke_rows = [row for row in prior_rows if row["split"] == "SMOKE"]
    if len(smoke_rows) != 8:
        raise RuntimeError("complete inherited SMOKE shape packet required")
    bulk_positions = 11*sum(row["max_processed_prefix_positions"] for row in token["rows"])
    smoke_positions = 16*sum(24*row["input_tokens"] + 276 for row in smoke_rows)
    result.update({"id": "BULK-GPU-001", "status": "READY_FOR_BULK_FP32_DISPATCH",
        "lane": "BULK_GPU_FP32_INTERVENTION", "review_receipt": "reviews/bulk_dispatch_receipt.json",
        "registered_utc": datetime.now(timezone.utc).isoformat(), "hard_deadline_seconds": 18000,
        "max_new_tokens": 8, "max_bulk_input_tokens": 350, "qualified_maximum_prefix_tokens": 357,
        "maximum_shape_qualification_status": "REQUIRED_FRESH_NATIVE_NOT_RUN",
        "actual_task_shapes": token["shapes"], "gpu_inventory_at_freeze": gpu,
        "native_resources_available_at_freeze": gpu["free_bytes"] >= 6*1024**3,
        "bulk_bank_path": str(bank_dir / "bank.json"), "bulk_bank_sha256": sha(bank_dir / "bank.json"),
        "bulk_manifest_sha256": sha(bank_dir / "manifest.json"), "bulk_audit_sha256": sha(bank_dir / "audit.json"),
        "bulk_positive_controls_sha256": sha(bank_dir / "positive_controls.json"),
        "bulk_source_sha256": sha(PACKAGE / "bulk_bank.py"),
        "bulk_normalizer_source_sha256": sha(PACKAGE / "open_bank.py"),
        "bulk_token_binding_path": str(token_path), "bulk_token_binding_sha256": sha(token_path),
        "dispatch_hashes": dispatch, "partition": "parent-world holdout; shared generators",
        "validation_independent_world_units": 48, "data_exposure": "author-visible exploratory OPEN; no native outcomes",
        "forecast": {"scientific_physical_evaluations": 3168, "SMOKE_physical_evaluations": 128,
            "physical_case_evaluations": 3296, "maximum_generation_forwards": 28416,
            "maximum_shape_forward_attempts": 18, "cache_rejection_attempts": 2,
            "maximum_model_API_attempts": 28436, "oracle_nonphysical_rows": 288,
            "frozen_validation_alias_nonphysical_rows": 96,
            "bulk_generation_prefix_positions": bulk_positions, "SMOKE_generation_prefix_positions": smoke_positions,
            "maximum_generation_prefix_positions": bulk_positions+smoke_positions,
            "maximum_shape_attempted_prefix_positions": 6426, "forecast_factor": 1.5,
            "cost_scope": "whole matrix; firstcold/setup charged once; fresh warm/max357 gate beforeDEV",
            "historical_matching_prefix_range": [24, 88], "unmeasured_prefix_range": [89, 357],
            "fresh_complete_forecast_required": True},
        "controller_revisions": 0, "optimizer_steps": 0, "cloud_spend": 0})
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"admission": str(target), "sha256": sha(target), "dispatch_files": len(dispatch),
        "prefix_position_cap": bulk_positions+smoke_positions, "GPUfree_MiB_at_freeze": gpu["free_bytes"]//1024**2,
        "review_status": "REQUIRED_NOT_CREATED_BY_AUTHOR"}))


if __name__ == "__main__":
    main()
