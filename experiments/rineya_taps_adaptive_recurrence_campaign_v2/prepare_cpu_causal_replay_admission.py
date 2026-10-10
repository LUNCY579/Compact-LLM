"""Freeze the prospective causal replay contract; never import Torch/load weights."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
from .run_apparatus import ROOT, PACKAGE, sha


def main(name):
    if Path(name).name != name:
        raise ValueError("filename only")
    binding_path = PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    inherited = []
    for relative in ("experiments/rineya_taps_adaptive_recurrence_v1",
                     "experiments/anchor_routed_middle_recurrence_v1/src",
                     "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src"):
        inherited.extend((ROOT / relative).rglob("*.py"))
    campaign_names = (
        "AUTHORITY.txt", "cpu_causal_replay_native.py", "cpu_causal_replay_entry.py",
        "cpu_causal_replay_supervisor.py", "prepare_cpu_causal_replay_admission.py",
        "causal_replay_extended_authority.py", "tests/test_causal_replay_extended_authority.py",
        "CAUSAL_REPLAY_EXTENDED_ALLOWANCE_TEMPLATE_001.json",
        "CPU_CAUSAL_REPLAY_EXTENDED_REQUEST_001.md",
        "cpu_h3_native.py", "sensitivity.py", "control.py", "binding.py", "supervision.py",
        "run_apparatus.py", "causal_replay_metadata.py", "bulk_bank.py", "open_bank.py",
        "CPU_CAUSAL_REPLAY_DESIGN_DRAFT_003.md", "CPU_CAUSAL_REPLAY_FEASIBILITY_DRAFT_001.json",
        "CPU_CAUSAL_REPLAY_NATIVE_REGISTRATION_002.md",
        "CPU_CAUSAL_REPLAY_ANALYSIS_PLAN_001.md",
        "cpu_causal_replay_analysis.py", "tests/test_cpu_causal_replay_analysis.py",
        "runs/causal-replay-DEV-export-001/DEVELOPMENT_BANK.json",
        "runs/causal-replay-DEV-export-001/DEVELOPMENT_TOKEN_BINDING.json",
        "runs/causal-replay-DEV-export-001/MANIFEST.json",
        "runs/binding-001/MODEL_DATA_BINDING.json", "runs/binding-001/PROMPT_AUDIT.json",
        "tests/test_cpu_causal_replay_native.py", "tests/test_cpu_causal_replay_supervision.py",
        "tests/test_cpu_causal_replay_parent_refusal.py", "tests/test_sensitivity.py", "tests/test_supervision.py",
        "reviews/cpu_causal_replay_design_review_001.json",
        "reviews/cpu_causal_replay_worker_review_001.json",
        "reviews/cpu_causal_replay_parent_review_001.json",
        "reviews/cpu_causal_replay_analysis_review_001.json",
        "reviews/cpu_causal_replay_extended_worker_review_001.json",
        "reviews/cpu_causal_replay_extended_authority_review_001.json",
        "reviews/cpu_causal_replay_analysis_budget_review_002.json",
    )
    paths = inherited+[PACKAGE / name for name in campaign_names]
    paths.append(Path(binding["datasets"]["DEV"]["path"]))
    gib = 1024**3
    admission = {
        "id": "CPU-CAUSAL-DEV-001", "status": "READY_FOR_CPU_CAUSAL_REPLAY_DISPATCH",
        "lane": "CPU_CAUSAL_REPLAY_FP32", "registered_utc": datetime.now(timezone.utc).isoformat(),
        "source": binding["source_path"], "interpreter": sys.executable,
        "binding_sha256": sha(binding_path), "checkpoint_files": binding["checkpoint_files"],
        "loaded_state_sha256": "b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551",
        "dispatch_hashes": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))},
        "review_receipt": "reviews/cpu_causal_replay_extended_dispatch_receipt.json",
        "requires_additional_human_authorization": True,
        "human_extension_path": str(PACKAGE / "HUMAN_CAUSAL_REPLAY_EXTENDED_ALLOWANCE_001.json"),
        "human_extension_sha256": sha(PACKAGE / "CAUSAL_REPLAY_EXTENDED_ALLOWANCE_TEMPLATE_001.json"),
        "device": "cpu", "dtype": "float32", "attn_implementation": "eager", "CPU_threads": 4,
        "maximum_prefix_tokens": 256, "fd_step": .01, "fd_absolute_tolerance": .001,
        "fd_relative_tolerance": .03, "direction_neutral_tolerance": .001,
        "eos_token_ids": [151643, 151645], "primary": "mean_answer_token_CE",
        "secondary": "accepted_EOS_probability_mass_loss", "predictor": "original_INPUT_only_balance",
        "training": False, "native_test": False, "generation": False, "generation_pool_access": False,
        "parameter_gradients_allowed": False, "controller_fitting": False, "optimizer_steps": 0,
        "parent_host_available_floor_bytes": math.ceil(8.4*gib),
        "worker_host_available_floor_bytes": math.ceil(7.4*gib),
        "worker_rss_ceiling_bytes": 6*gib, "host_reserve_bytes": gib,
        "hard_deadline_seconds": 4500, "finalization_reserve_seconds": 180,
        "native_finalization_reserve_seconds": 1800, "deterministic_algorithms": True,
        "forecast": {"cases": 192, "parents": 96, "answer_targets": 440, "EOS_targets": 192,
                     "task_targets": 632, "task_full_forwards": 1896, "task_state_adjoints": 632,
                     "qualification_API_attempts": 29, "qualification_adjoints": 3,
                     "model_API_attempts": 1925, "full_forward_attempts": 1923,
                     "completed_forwards": 1922, "expected_partial": 1, "cache_rejections": 2,
                     "state_adjoints": 635, "completed_prefix_positions": 252696,
                     "task_prefix_positions": 247932, "partial_positions": None,
                     "fresh_CPU_complete_forecast_before_any_task": True},
        "development_bank": "runs/causal-replay-DEV-export-001/DEVELOPMENT_BANK.json",
        "development_bank_sha256": sha(PACKAGE / "runs/causal-replay-DEV-export-001/DEVELOPMENT_BANK.json"),
        "development_token_binding": "runs/causal-replay-DEV-export-001/DEVELOPMENT_TOKEN_BINDING.json",
        "development_token_binding_sha256": sha(PACKAGE / "runs/causal-replay-DEV-export-001/DEVELOPMENT_TOKEN_BINDING.json"),
        "count_contract": {"cases": 192, "parents": 96, "answer_targets": 440, "EOS_targets": 192,
                           "task_targets": 632, "qualification_API_attempts": 29, "qualification_adjoints": 3,
                           "model_API_attempts": 1925, "full_forward_attempts": 1923,
                           "completed_forwards": 1922, "expected_partial": 1, "cache_rejections": 2,
                           "state_adjoints": 635, "completed_prefix_positions": 252696},
        "metadata_feasibility": "CPU_CAUSAL_REPLAY_FEASIBILITY_DRAFT_001.json",
        "infrastructure_retry_available": False, "uses_runtime_debug_grant": False,
        "full_bulk_matrix_retry": False, "controller_revision_counted": False,
        "claim_scope": "exposed canonical teacher-prefix likelihood mechanism only; no semantic efficacy/fit/GPU pooling",
    }
    path = PACKAGE / name
    with path.open("x", encoding="utf-8") as handle:
        json.dump(admission, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (PACKAGE / "REGISTRY.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": admission["id"], "status": "FROZEN_PENDING_INDEPENDENT_REVIEW_AND_READINESS",
                                "admission": name, "admission_sha256": sha(path),
                                "controller_revision_counted": False, "timestamp_utc": admission["registered_utc"]})+"\n")
    print(json.dumps({"admission_sha256": sha(path), "dispatch_files": len(admission["dispatch_hashes"])}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="cpu_causal_replay_admission_v1.json")
    main(parser.parse_args().name)
