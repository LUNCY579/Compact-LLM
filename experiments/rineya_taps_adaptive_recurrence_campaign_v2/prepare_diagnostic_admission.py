"""Freeze separate state-gradient diagnostic admission; no model import."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from .run_apparatus import ROOT, PACKAGE, sha
from .gpu_supervisor import gpu_inventory

GIB = 1024**3


def main(name):
    if Path(name).name != name:
        raise ValueError("filename only")
    binding_path = PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json"
    binding = json.loads(binding_path.read_text())
    paths = []
    for root in (ROOT / "experiments/rineya_taps_adaptive_recurrence_v1",
                 ROOT / "experiments/anchor_routed_middle_recurrence_v1/src",
                 ROOT / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src"):
        paths.extend(root.rglob("*.py"))
    paths.extend(PACKAGE / p for p in ("diagnostic_native.py", "diagnostic_entry.py", "diagnostic_supervisor.py",
        "sensitivity.py", "gpu_native.py", "control.py", "binding.py", "supervision.py", "native_supervisor.py",
        "run_apparatus.py", "DIAGNOSTIC_DESIGN.md", "DIAGNOSTIC_NATIVE_REGISTRATION.md", "AUTHORITY.txt",
        "runs/binding-001/MODEL_DATA_BINDING.json", "runs/binding-001/PROMPT_AUDIT.json"))
    paths.extend(Path(x["path"]) for x in binding["datasets"].values())
    inv = gpu_inventory()
    admission = {"id": "H3-STATE-DIAG-001", "status": "READY_FOR_DIAGNOSTIC_FP32_DISPATCH",
        "registered_utc": datetime.now(timezone.utc).isoformat(), "lane": "OFFLINE_STATE_SENSITIVITY_FP32",
        "source": binding["source_path"], "interpreter": sys.executable, "binding_sha256": sha(binding_path),
        "checkpoint_files": binding["checkpoint_files"],
        "dispatch_hashes": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))},
        "review_receipt": "reviews/diagnostic_dispatch_receipt.json", "gpu_uuid": inv["uuid"],
        "gpu_driver_version": inv["driver_version"], "gpu_inventory_at_freeze": inv,
        "device": "cuda:0", "dtype": "float32", "attn_implementation": "eager",
        "maximum_prefix_tokens": 256, "fd_step": .01, "fd_absolute_tolerance": .001,
        "fd_relative_tolerance": .03, "training": False, "native_test": False, "generation_pool_access": False,
        "gpu_allocated_ceiling_bytes": 5 * GIB, "gpu_global_free_reserve_bytes": GIB,
        "gpu_initial_free_floor_bytes": 6 * GIB, "worker_host_available_floor_bytes": 6 * GIB,
        "parent_host_available_floor_bytes": 7 * GIB, "worker_rss_ceiling_bytes": 6 * GIB,
        "hard_deadline_seconds": 1800, "native_finalization_reserve_seconds": 1800,
        "TF32": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "deterministic_algorithms": True,
        "population": {"bound_original_rows": 16, "short": 14, "eligible": 7,
            "inherited_target_unavailable": 7, "shape_excluded": 2, "rule": "all originalDEV;<=256;inherited canonical firsttoken"},
        "forecast": {"full_forward_attempts": 45, "expected_completed": 44, "expected_partial": 1,
            "cache_rejections": 2, "model_api_attempts": 47, "state_adjoints": 9,
            "maximum_attempted_prefix_positions": 47 * 256, "cold_load_hash_allowance_seconds": 200,
            "conservative_forward_seconds": 47 * .8833317 * 1.5, "adjoint_allowance_seconds": 90},
        "infrastructure_retry_available": False, "parameter_gradients_allowed": False,
        "claim_scope": "native local surrogate sensitivity; no semantic efficacy or controller fit"}
    path = PACKAGE / name
    with path.open("x", encoding="utf-8") as f:
        json.dump(admission, f, indent=2, sort_keys=True)
        f.write("\n")
    with (PACKAGE / "REGISTRY.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": admission["id"], "type": "offline_state_gradient_diagnostic",
            "status": "FROZEN_PENDING_REVIEW_AND_RESOURCES", "admission": name, "admission_sha256": sha(path),
            "controller_revision_counted": False, "timestamp_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    print(json.dumps({"admission_sha256": sha(path), "dispatch_files": len(admission["dispatch_hashes"])}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="diagnostic_admission_v1.json")
    main(p.parse_args().name)
