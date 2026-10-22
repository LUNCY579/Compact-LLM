"""Freeze new OPEN corpus/scorer/cap intervention; no model load."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from .run_apparatus import ROOT, PACKAGE, sha
from .gpu_supervisor import gpu_inventory

GIB = 1024**3


def main(name):
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("filename only")
    legacy = json.loads((PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json").read_text())
    packet = PACKAGE / "runs/open-bank-apparatus-002"
    token_path = PACKAGE / "runs/open-tokenizer-binding-001/OPEN_TOKENIZER_BINDING.json"
    token = json.loads(token_path.read_text())
    if token["status"] != "OPEN_TOKENIZER_BOUND_NATIVE_NOT_MEASURED" or len(token["rows"]) != 24:
        raise RuntimeError("complete tokenizer/shape binding missing")
    inv = gpu_inventory()
    paths = []
    for root in (ROOT / "experiments/rineya_taps_adaptive_recurrence_v1",
                 ROOT / "experiments/anchor_routed_middle_recurrence_v1/src",
                 ROOT / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src"):
        paths.extend(root.rglob("*.py"))
    paths.extend(PACKAGE / p for p in ("open_entry.py", "open_supervisor.py", "open_native.py", "open_bank.py",
        "open_binding.py", "gpu_native.py", "control.py", "binding.py", "supervision.py", "native_supervisor.py",
        "run_apparatus.py", "AUTHORITY.txt", "OPEN_NATIVE_REGISTRATION.md", "OPEN_BANK_DESIGN.md",
        "runs/binding-001/MODEL_DATA_BINDING.json", "runs/binding-001/PROMPT_AUDIT.json",
        "runs/open-tokenizer-binding-001/OPEN_TOKENIZER_BINDING.json"))
    paths.extend(packet / p for p in ("manifest.json", "bank.json", "audit.json", "positive_controls.json"))
    paths.extend(Path(x["path"]) for x in legacy["datasets"].values())
    open_positions = 20 * sum(row["max_processed_prefix_positions"] for row in token["rows"])
    old_prompts = json.loads((PACKAGE / "runs/binding-001/PROMPT_AUDIT.json").read_text())
    smoke_positions = 16 * sum(row["maximum_processed_prefix_positions"] for row in old_prompts if row["split"] == "SMOKE")
    admission = {"id": "OPEN-GPU-001", "status": "READY_FOR_OPEN_FP32_DISPATCH",
        "lane": "OPEN_GPU_FP32_INTERVENTION", "source": legacy["source_path"], "interpreter": sys.executable,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "dispatch_hashes": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))},
        "review_receipt": "reviews/open_dispatch_receipt.json", "gpu_uuid": inv["uuid"],
        "gpu_driver_version": inv["driver_version"], "gpu_inventory_at_freeze": inv,
        "device": "cuda:0", "dtype": "float32", "attn_implementation": "eager",
        "checkpoint_files": legacy["checkpoint_files"], "gpu_helpers_sha256": sha(PACKAGE / "gpu_native.py"),
        "open_bank_path": str(packet / "bank.json"), "open_bank_sha256": sha(packet / "bank.json"),
        "open_manifest_sha256": sha(packet / "manifest.json"), "open_audit_sha256": sha(packet / "audit.json"),
        "open_positive_controls_sha256": sha(packet / "positive_controls.json"),
        "open_source_sha256": sha(PACKAGE / "open_bank.py"),
        "open_token_binding_path": str(token_path), "open_token_binding_sha256": sha(token_path),
        "gpu_allocated_ceiling_bytes": 5 * GIB, "gpu_global_free_reserve_bytes": GIB,
        "gpu_initial_free_floor_bytes": 6 * GIB, "worker_host_available_floor_bytes": 6 * GIB,
        "parent_host_available_floor_bytes": 7 * GIB, "worker_rss_ceiling_bytes": 6 * GIB,
        "max_new_tokens": 8, "max_open_input_tokens": 350, "actual_task_shapes": token["shapes"],
        "qualified_maximum_prefix_tokens": 357,
        "TF32": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "deterministic_algorithms": True,
        "hard_deadline_seconds": 14400, "native_finalization_reserve_seconds": 1800,
        "idle_forward_deadline_seconds": 950, "training": False, "native_test": False,
        "controller_candidate_revision_count": 0, "infrastructure_retry_available": False,
        "static_original_A3_mean": 1.1473425965417514,
        "forecast": {"physical_case_evaluations": 608, "OPEN_case_evaluations": 480,
            "SMOKE_case_evaluations": 128, "maximum_generation_forwards": 6912,
            "maximum_generation_prefix_positions": open_positions + smoke_positions,
            "maximum_shape_forward_attempts": 18, "maximum_shape_attempted_prefix_positions": 18 * 357,
            "cache_rejection_attempts": 2, "maximum_model_api_attempts": 6932,
            "oracle_physical_generation": 0, "forecast_factor": 1.5,
            "previous_measured_max_forward_seconds": .883331699998962,
            "raw_all_calls_conservative_seconds": 6932 * .883331699998962 * 1.5,
            "cost_scope": "complete cap plus cold load/qualification; fresh observed forecast before OPEN measurement"}}
    path = PACKAGE / name
    with path.open("x", encoding="utf-8") as f:
        json.dump(admission, f, indent=2, sort_keys=True)
        f.write("\n")
    with (PACKAGE / "REGISTRY.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": admission["id"], "parent": "GPU-REF-V1-002", "type": "new_data_scorer_cap",
            "controller_revision_counted": False, "status": "FROZEN_PENDING_REVIEW_AND_RESOURCES",
            "admission": name, "admission_sha256": sha(path), "timestamp_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    print(json.dumps({"admission_sha256": sha(path), "files": len(admission["dispatch_hashes"]), "forecast": admission["forecast"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="open_admission_v1.json")
    main(p.parse_args().name)
