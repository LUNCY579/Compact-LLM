"""Freeze explicit GPU FP32 successor admission before any model outcomes."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.run_apparatus import ROOT, PACKAGE, sha
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.gpu_supervisor import gpu_inventory

GIB = 1024**3


def main(name="gpu_admission_v1.json", deadline=7200):
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("admission filename only")
    if deadline not in (7200, 10800):
        raise ValueError("only independently preregistered allocations allowed")
    binding_path = PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json"
    binding = json.loads(binding_path.read_text())
    inventory = gpu_inventory()
    if inventory["index"] != 0 or inventory["total_bytes"] < 8 * GIB - 16 * 1024**2:
        raise RuntimeError("registered single8GiB GPU identity differs")
    dispatch = []
    for root in (ROOT / "experiments/rineya_taps_adaptive_recurrence_v1",
                 ROOT / "experiments/anchor_routed_middle_recurrence_v1/src",
                 ROOT / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src"):
        dispatch.extend(root.rglob("*.py"))
    dispatch.extend(PACKAGE / name for name in (
        "gpu_supervisor.py", "gpu_entry.py", "gpu_native.py", "supervision.py", "native_supervisor.py",
        "run_apparatus.py", "binding.py", "control.py", "GPU_HOST_REGISTRATION.md", "PREREGISTRATION.md",
        "AUTHORITY.txt", "runs/binding-001/MODEL_DATA_BINDING.json", "runs/binding-001/PROMPT_AUDIT.json"))
    dispatch.extend(Path(row["path"]) for row in binding["datasets"].values())
    if deadline == 10800:
        dispatch.append(PACKAGE / "RESOURCE_AMENDMENT_001.md")
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(dispatch))}
    meta = json.loads((binding_path.parent / "PROMPT_AUDIT.json").read_text())
    added_smoke_positions = 2 * sum(row["maximum_processed_prefix_positions"] for row in meta if row["split"] == "SMOKE")
    admission = {"id": "GPU-REF-V1-001", "status": "READY_FOR_GPU_FP32_DISPATCH",
                 "registered_utc": datetime.now(timezone.utc).isoformat(), "lane": "GPU_FP32_SUCCESSOR",
                 "source": binding["source_path"], "interpreter": sys.executable,
                 "binding_sha256": sha(binding_path), "dispatch_hashes": hashes,
                 "device": "cuda:0", "dtype": "float32", "attn_implementation": "eager",
                 "gpu_uuid": inventory["uuid"], "gpu_driver_version": inventory["driver_version"],
                 "gpu_inventory_at_freeze": inventory,
                 "gpu_allocated_ceiling_bytes": 5 * GIB, "gpu_global_free_reserve_bytes": GIB,
                 "gpu_initial_free_floor_bytes": 6 * GIB, "worker_host_available_floor_bytes": 6 * GIB,
                 "parent_host_available_floor_bytes": 7 * GIB, "worker_rss_ceiling_bytes": 6 * GIB,
                 "checkpoint_files": binding["checkpoint_files"], "maximum_shapes": binding["shapes"],
                 "expected_loaded_state_sha256": binding["historical_expected_loaded_fp32_state_sha256"],
                 "runtime_metadata": binding["runtime"], "TF32": False,
                 "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "deterministic_algorithms": True,
                 "hard_deadline_seconds": deadline, "native_finalization_reserve_seconds": 1800,
                 "idle_forward_deadline_seconds": 950,
                 "disk_ceiling_bytes": 2 * GIB, "disk_finalization_reserve_bytes": 512 * 1024**2,
                 "host_disk_floor_bytes": 15 * GIB,
                 "forecast": {"physical_evaluate_case_calls": 408, "oracle_nonphysical_rows": 32,
                              "maximum_generation_forwards": 6732,
                              "maximum_generation_prefix_positions": binding["forecast"]["maximum_processed_prefix_positions"] + added_smoke_positions,
                              "maximum_shape_forward_attempts": 17, "cache_rejection_attempts": 2,
                              "maximum_shape_prefix_positions_attempted": 17 * 2248,
                              "maximum_model_api_attempts": 6732 + 17 + 2,
                              "cost_estimate_status": "Measured qualified GPU cost from run001 available; prospective per-job ceiling fits21600s cumulative allowance; fresh observed maxshape/smoke forecast must fit before DEV"},
                 "review_receipt": "reviews/gpu_retry_dispatch_receipt.json" if deadline == 10800 else "reviews/gpu_dispatch_receipt.json",
                 "native_test": False, "training": False, "optimizer_steps": 0,
                 "reference_cpu_attempt": "runs/native-reference-001/terminal_receipt.json",
                 "reference_cpu_status": "BLOCKED_NO_LOAD_RAM; preserved; not exactCPU reproduction",
                 "retry_lineage": "native-gpu-reference-001 qualified then refused DEV allocation" if deadline == 10800 else None,
                 "infrastructure_retry_number": 1 if deadline == 10800 else 0,
                 "controller_candidate_revision_count": 0,
                 "stochastic_control": "corrected A5 config.seed actual17/29; persistent prefix RNG; moments reset",
                 "primary_endpoint": "inherited binary task score;16 exposed DEV rows; no semantic confirmation",
                 "independent_review_required": True}
    path = PACKAGE / name
    with path.open("x", encoding="utf-8") as f:
        json.dump(admission, f, indent=2, sort_keys=True)
        f.write("\n")
    with (PACKAGE / "REGISTRY.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": admission["id"], "parent": "REF-V1", "type": "separate_device_host_qualification",
                            "status": "FROZEN_PENDING_INDEPENDENT_REVIEW", "admission_file": path.name,
                            "admission_sha256": sha(path), "candidate_revision_counted": False,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    print(json.dumps({"admission_sha256": sha(path), "dispatch_files": len(hashes), "gpu_inventory": inventory}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="gpu_admission_v1.json")
    parser.add_argument("--deadline", type=int, default=7200)
    args = parser.parse_args()
    main(args.name, args.deadline)
