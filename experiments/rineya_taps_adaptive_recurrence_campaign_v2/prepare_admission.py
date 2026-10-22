"""Freeze reference admission from completed no-model evidence, before outcomes."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.run_apparatus import ROOT, PACKAGE, sha


def main(name="admission.json"):
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("admission filename only")
    binding_path = PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json"
    binding = json.loads(binding_path.read_text())
    preflight = json.loads((PACKAGE / "runs/preflight-cpu-001/result/preflight.json").read_text())
    if preflight["status"] != "METADATA_READY_NOT_QUALIFIED" or binding["forecast"]["physical_evaluations"] != 392:
        raise RuntimeError("required metadata/count evidence absent")
    if (binding["shapes"]["max_forward_prefix_tokens"] != 2248 or binding["forecast"]["maximum_model_calls"] != 6348):
        raise RuntimeError("shape/count registration mismatch")
    dispatch = []
    for root in (ROOT / "experiments/rineya_taps_adaptive_recurrence_v1",
                 ROOT / "experiments/anchor_routed_middle_recurrence_v1/src",
                 ROOT / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src"):
        dispatch.extend(root.rglob("*.py"))
    dispatch.extend(PACKAGE / name for name in (
        "native_supervisor.py", "native_entry.py", "supervision.py", "run_apparatus.py",
        "PREREGISTRATION.md", "DISPOSITIONS.json", "AUTHORITY.txt",
        "runs/binding-001/MODEL_DATA_BINDING.json", "runs/binding-001/PROMPT_AUDIT.json"))
    dispatch.extend(Path(d["path"]) for d in binding["datasets"].values())
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(dispatch))}
    admission = {"id": "REF-V1-CPU-001", "status": "READY_FOR_REFERENCE_DISPATCH",
                 "independent_approval_required": True,
                 "registered_utc": datetime.now(timezone.utc).isoformat(),
                 "source": binding["source_path"], "interpreter": sys.executable,
                 "dispatch_hashes": hashes,
                 "binding_sha256": sha(binding_path), "source_runtime": binding["runtime"],
                 "checkpoint_files": binding["checkpoint_files"],
                 "model_class_required": "Qwen3AbstainForCausalLM",
                 "expected_loaded_state_sha256": binding["historical_expected_loaded_fp32_state_sha256"],
                 "maximum_shapes": binding["shapes"], "complete_count_forecast": binding["forecast"],
                 "historical_cost_forecast_seconds": 12000,
                 "forecast_basis": "Exact ARMR48-case historical worker1406.4925958seconds; approximately392/48 times historical total, rounded conservatively; current runtime speed unmeasured",
                 "metadata_wording_correction": "binding-001 forecast_status implied native pilot required; unchanged raw packet retained. User-authorized historical reference may use this prospective historical-cost forecast, per independent reviewer scoped interpretation; successor runtime/device may require qualified pilot.",
                 "hard_deadline_seconds": 18000, "native_finalization_reserve_seconds": 1800,
                 "idle_forward_deadline_seconds": 950, "worker_rss_ceiling_bytes": 6 * 1024**3,
                 "host_available_ram_floor_bytes": 7945689498,
                 "host_running_reserve_bytes": 1024**3,
                 "disk_ceiling_bytes": 2 * 1024**3, "disk_finalization_reserve_bytes": 512 * 1024**2,
                 "host_disk_floor_bytes": 15 * 1024**3,
                 "review_receipt": "reviews/reference_dispatch_receipt.json",
                 "reference_scope_limits": ["exposed exploratory DEV", "legacy A5 per-example constant/nominal seed duplication",
                                            "fresh maxshape/forced-native-exception PASS not asserted",
                                            "instrumented timing; no deployment-speed claim", "no independent confirmation"],
                 "training": False, "native_test": False, "gpu_lane": False}
    with (PACKAGE / name).open("x", encoding="utf-8") as f:
        json.dump(admission, f, indent=2, sort_keys=True)
        f.write("\n")
    if not (PACKAGE / "MODEL_BINDING.json").exists():
        with (PACKAGE / "MODEL_BINDING.json").open("x", encoding="utf-8") as f:
            json.dump({"status": "METADATA_BOUND_NOT_NATIVE_QUALIFIED",
                   "metadata_binding": str(binding_path.relative_to(PACKAGE)), "sha256": sha(binding_path),
                   "actual_loaded_state_sha256": None, "actual_loaded_custom_code_sha256": None,
                   "expected_loaded_state_sha256": admission["expected_loaded_state_sha256"],
                   "checkpoint_files": binding["checkpoint_files"], "operator": "wholeL12-L13 raw residual; unchanged first pass;0or1extra",
                   "native_run_binding_required": "native/result/receipt.json plus loaded custom code cache hash"}, f, indent=2)
            f.write("\n")
    with (PACKAGE / "REGISTRY.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "REF-V1-CPU-001", "parent": "REF-V1", "status": "FROZEN_PENDING_INDEPENDENT_REVIEW",
                            "admission_file": name, "admission_sha256": sha(PACKAGE / name), "native_worker_seconds": 0,
                            "controller_candidate_revisions": 0, "timestamp_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    print(json.dumps({"admission_sha256": sha(PACKAGE / name), "dispatch_files": len(hashes),
                      "next": "independent reviewer source-bound receipt before dispatch"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="admission.json")
    main(parser.parse_args().name)
