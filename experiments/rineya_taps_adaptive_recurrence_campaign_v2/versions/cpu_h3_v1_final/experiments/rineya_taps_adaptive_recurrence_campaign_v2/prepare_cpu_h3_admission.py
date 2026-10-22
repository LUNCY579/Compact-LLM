"""Freeze the distinct CPU H3 contract without importing Torch or loading weights."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
from .run_apparatus import ROOT, PACKAGE, sha


def main():
    binding_path = PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    paths = []
    for relative in ("experiments/rineya_taps_adaptive_recurrence_v1",
                     "experiments/anchor_routed_middle_recurrence_v1/src",
                     "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src"):
        paths.extend((ROOT / relative).rglob("*.py"))
    paths.extend(PACKAGE / relative for relative in (
        "cpu_h3_native.py", "cpu_h3_entry.py", "cpu_h3_supervisor.py",
        "prepare_cpu_h3_admission.py", "sensitivity.py", "binding.py", "control.py", "supervision.py",
        "run_apparatus.py", "CPU_H3_DESIGN_001.md", "CPU_H3_NATIVE_REGISTRATION_001.md",
        "CPU_H3_FEASIBILITY_001.json", "AUTHORITY.txt",
        "CPU_H3_FEASIBILITY_001.md", "CPU_H3_NATIVE_REGISTRATION_DRAFT_001.md",
        "audits/cpu_H3_authority_feasibility_review_001.json",
        "audits/cpu_H3_math_review_001.json", "reviews/cpu_h3_worker_review_001.json",
        "audits/cpu_H3_math_review_addendum_002.json",
        "reviews/cpu_h3_parent_review_001.json",
        "runs/preflight-cpu-002/preflight.json",
        "runs/binding-001/MODEL_DATA_BINDING.json", "runs/binding-001/PROMPT_AUDIT.json",
        "tests/test_cpu_h3_native.py", "tests/test_cpu_h3_supervision.py",
        "tests/test_sensitivity.py", "tests/test_supervision.py"))
    paths.append(Path(binding["datasets"]["DEV"]["path"]))
    gib = 1024**3
    admission = {
        "id": "CPU-H3-001", "status": "READY_FOR_CPU_H3_FP32_DISPATCH",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "lane": "CPU_H3_STATE_SENSITIVITY_FP32", "device": "cpu", "dtype": "float32",
        "attn_implementation": "eager", "CPU_threads": 4,
        "source": binding["source_path"], "interpreter": sys.executable,
        "binding_sha256": sha(binding_path), "checkpoint_files": binding["checkpoint_files"],
        "loaded_state_sha256": "b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551",
        "dispatch_hashes": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))},
        "review_receipt": "reviews/cpu_h3_dispatch_receipt.json",
        "maximum_prefix_tokens": 256, "fd_step": .01,
        "fd_absolute_tolerance": .001, "fd_relative_tolerance": .03,
        "training": False, "native_test": False, "generation_pool_access": False,
        "parameter_gradients_allowed": False, "optimizer_steps": 0,
        "parent_host_available_floor_bytes": math.ceil(8.4*gib),
        "worker_host_available_floor_bytes": math.ceil(7.4*gib),
        "worker_rss_ceiling_bytes": 6*gib, "host_reserve_bytes": gib,
        "hard_deadline_seconds": 1800, "finalization_reserve_seconds": 180,
        "native_finalization_reserve_seconds": 1800,
        "deterministic_algorithms": True, "GPU_timing_or_gradient_pooling": False,
        "population": {"bound_original_rows": 16, "eligible": 7,
                       "short": 14, "inherited_target_unavailable": 7, "shape_excluded": 2,
                       "rule": "all originalDEV;<=256;inherited canonical firsttoken"},
        "forecast": {"full_forward_attempts": 45, "expected_completed": 44,
                     "expected_partial": 1, "cache_rejections": 2,
                     "model_api_attempts": 47, "state_adjoints": 9,
                     "expected_completed_prefix_positions": 4818,
                     "maximum_attempted_prefix_positions": 12032,
                     "fresh_CPU_maximum_forward_and_adjoint_required_before_task_rows": True,
                     "remaining_forecast": "7*(max256 adjoint-forward+2*max256 value-forward)*1.5",
                     "admit_rule": "elapsed+remaining_forecast+180<=1800"},
        "infrastructure_retry_available": False, "full_bulk_matrix_retry": False,
        "uses_runtime_debug_grant": False, "runtime_debug_grant_consumed": True,
        "controller_revision_counted": False,
        "claim_scope": "exposed DEV first-token local surrogate sensitivity; no semantic efficacy or controller fit",
    }
    path = PACKAGE / "cpu_h3_admission_v1.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(admission, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (PACKAGE / "REGISTRY.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": admission["id"], "type": "CPU_H3_same_population_successor",
                                "status": "FROZEN_PENDING_INDEPENDENT_REVIEW_AND_FRESH_RESOURCES",
                                "admission": path.name, "admission_sha256": sha(path),
                                "controller_revision_counted": False,
                                "timestamp_utc": admission["registered_utc"]})+"\n")
    print(json.dumps({"admission_sha256": sha(path), "dispatch_files": len(admission["dispatch_hashes"])}))


if __name__ == "__main__":
    main()
