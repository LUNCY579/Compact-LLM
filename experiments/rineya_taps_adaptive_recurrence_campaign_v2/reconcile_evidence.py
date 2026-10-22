"""Reconcile current campaign state with completed immutable native receipts."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from .run_apparatus import PACKAGE, sha


def append(name, value):
    with (PACKAGE / name).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **value}, sort_keys=True) + "\n")


def replace_state(name, value):
    path = PACKAGE / name
    append("STATE_HISTORY.jsonl", {"file": name, "previous_sha256": sha(path),
                                  "previous_value": json.loads(path.read_text(encoding="utf-8-sig"))})
    temporary = path.with_suffix(path.suffix + ".reconcile.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main():
    marker = PACKAGE / "BOOKKEEPING_RECONCILIATION.json"
    with marker.open("x", encoding="utf-8") as handle:
        json.dump({"status": "STARTED"}, handle)
    run = PACKAGE / "runs/native-gpu-reference-002"
    result = json.loads((run / "result/receipt.json").read_text())
    terminal = json.loads((run / "terminal_receipt.json").read_text())
    qualification = json.loads((run / "result/qualification.json").read_text())
    registration = json.loads((run / "result/registration.json").read_text())
    metadata = json.loads((PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json").read_text())
    if result["status"] != "GPU_DEV_COMPLETE_NOT_TESTED" or terminal["status"] != "WORKER_TERMINAL":
        raise RuntimeError("completed native evidence absent")
    binding = {"status": "QUALIFIED_GPU_FP32_REFERENCE_COMPLETE_EXPLORATORY_DEV",
        "metadata_binding": "runs/binding-001/MODEL_DATA_BINDING.json",
        "metadata_sha256": sha(PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json"),
        "checkpoint_files": metadata["checkpoint_files"], "model_class": "Qwen3AbstainForCausalLM",
        "actual_loaded_state_sha256": result["loaded_state_sha256"],
        "actual_loaded_custom_code_sha256": result["loaded_custom_code_sha256"],
        "loaded_custom_code_path": result["loaded_custom_code_path"],
        "expected_loaded_state_sha256": metadata["historical_expected_loaded_fp32_state_sha256"],
        "tokenizer_class": metadata["tokenizer_class"], "tokenizer_length": metadata["tokenizer_length"],
        "vocabulary_size": metadata["model_config"]["vocab_size"], "layer_numbering": "zero-based,28 layers",
        "operator": "unchanged wholeL12-L13 first pass plus0or1extra raw residual; first pass seeds moments",
        "readout": {"abstain_token_id": 51378, "abstain_logit_bias": 5.156249985098839,
                    "application": "every decoded logit position; preserved source behavior"},
        "dtype": "float32", "device": "cuda:0", "attention": "eager", "batch": 1,
        "padding": False, "cache": False, "position_mask_contract": qualification["mask_positions"],
        "moment_reset": "every independent sample and full-prefix call",
        "runtime": registration["runtime"], "maximum_qualified_prefix_tokens": 2248,
        "loaded_weight_and_asset_preservation": result["source_unchanged"], "training": False,
        "qualification_receipt": "runs/native-gpu-reference-002/result/qualification.json",
        "qualification_sha256": sha(run / "result/qualification.json"),
        "complete_receipt": "runs/native-gpu-reference-002/result/receipt.json",
        "complete_receipt_sha256": sha(run / "result/receipt.json"),
        "exact_historical_CPU_reproduction": "BLOCKED_NO_LOAD_RAM; separate GPU successor",
        "independent_confirmation": False, "deployability": "NOT_ASSESSED"}
    replace_state("MODEL_BINDING.json", binding)
    environment = json.loads((PACKAGE / "ENVIRONMENT.json").read_text(encoding="utf-8-sig"))
    environment.update(native_qualified=True, qualified_lane="separate_GPU_FP32_successor",
        native_runtime=registration["runtime"], gpu_inventory=terminal["gpu_initial_inventory"],
        measured_native_peak_allocated_bytes=result["final_resources"]["gpu_peak_allocated_bytes"],
        measured_native_peak_reserved_bytes=result["final_resources"]["gpu_peak_reserved_bytes"],
        measured_native_worker_tree_peak_rss_bytes=terminal["peak_rss_bytes"],
        evidence="runs/native-gpu-reference-002; original host/preflight snapshots preserved")
    replace_state("ENVIRONMENT.json", environment)
    ownership = json.loads((PACKAGE / "OWNERSHIP.json").read_text(encoding="utf-8-sig"))
    ownership["native_worker_authority"] = "/root only; three terminal supervised attempts; no active worker"
    ownership["frozen_sources"] = {"gpu_native.py": "4c1a05c6511ff267d790654475f2a22cb5a3b8fa02857b295aa1ab3ae6c20e6f"}
    ownership["owners"].update({"diagnostic_native.py": "/root/native_engineer",
        "tests/test_diagnostic_native.py": "/root/native_engineer", "open_analysis.py": "/root/native_engineer",
        "tests/test_open_analysis.py": "/root/native_engineer", "audits/completion_audit.json": "/root/evidence_math",
        "audits/completion_audit.md": "/root/evidence_math"})
    replace_state("OWNERSHIP.json", ownership)
    append("REGISTRY.jsonl", {"id": "GPU-REF-V1-002", "parent": "GPU-REF-V1-001",
        "event": "COMPLETED_REFERENCE", "frozen_admission_id": "GPU-REF-V1-001",
        "note": "Explicit unique run node resolves legacy prepare-script admission-id reuse; old freeze unchanged",
        "run": "runs/native-gpu-reference-002", "status": result["status"],
        "receipt_sha256": sha(run / "result/receipt.json"), "native_worker_seconds": terminal["native_worker_seconds"],
        "controller_revision_counted": False, "result": "no positive constant-grid headroom or adaptive advantage"})
    append("REGISTRY.jsonl", {"id": "OPEN-GPU-001", "event": "DISPATCH_REVIEW_APPROVED",
        "status": "READY_PENDING_ACTUAL_RESOURCES", "admission": "open_admission_v2.json",
        "admission_sha256": sha(PACKAGE / "open_admission_v2.json"),
        "review": "reviews/open_dispatch_receipt.json", "native_dispatched": False})
    for version, evidence in ((2, "reviews/control_binding_review_receipt.json"),
                              (3, "reviews/gpu_dispatch_receipt.json"),
                              (4, "reviews/gpu_retry_dispatch_receipt.json")):
        append("RESEARCH_POLICY.jsonl", {"version": version, "event": "REVIEW_DISPOSITION",
            "status": "ACCEPTED_FOR_SCOPED_PROCEDURE", "evidence": evidence,
            "evidence_sha256": sha(PACKAGE / evidence), "empirical_research_allocation_improvement": "NOT_MEASURED"})
    marker.write_text(json.dumps({"status": "COMPLETE", "native_reference_receipt_sha256": sha(run / "result/receipt.json"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
