"""Exact prospective qualification body; specification only, never a native permit."""
from __future__ import annotations
from . import custody, registry, schema


def proposal(packet_receipt):
    if packet_receipt.get("offline_validation_passed") is not True:
        raise ValueError("STOP_QUALIFICATION_PROPOSAL_NEEDS_OFFLINE_PACKET")
    fixed = registry.build_registry()
    if packet_receipt.get("registry_sha256") != fixed["registry_sha256"]:
        raise ValueError("STOP_PROPOSAL_REGISTRY_BINDING")
    source = packet_receipt["source_manifest"]
    development = packet_receipt["development"]
    ids = development["qualification_row_ids"]
    if len(ids) != 8 or len(set(ids)) != 8 or development["reload_readout_row_id"] not in ids:
        raise ValueError("STOP_QUALIFICATION_ROW_MEMBERSHIP")
    body = {"experiment": schema.EXPERIMENT, "proposal_version": 1,
        "status": "PROSPECTIVE_NATIVE_QUALIFICATION_BLOCKED",
        "registry_sha256": fixed["registry_sha256"], "source_manifest_sha256": schema.digest(source),
        "source_manifest": source, "source_checkpoint_sha256": custody.WEIGHT_SHA256,
        "architecture": "Qwen3ForCausalLM", "dtype": "float32", "custom_abstention": False,
        "interpreter": "C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe",
        "required_slot": "OPERATOR_ALLOCATED_EXCLUSIVE_LOCAL_GPU0_SLOT_NOT_YET_SUPPLIED",
        "resources": {"host_available_gib_min": 12, "gpu_available_gib_min": 6,
            "disk_available_gib_min": 20, "host_active_reserve_gib": 1.5,
            "gpu_active_reserve_gib": .5, "cuda_memory_fraction": .78,
            "process_tree_rss_max_gib": 10, "new_output_bytes_max": 6 * 1024 ** 3},
        "attempt": {"model_loads_max": 1, "worker_wall_seconds_max": 540,
            "supervisor_kill_deadline_seconds": 600, "cleanup_wall_seconds_max": 120,
            "load_begun_before_from_pretrained": True, "loaded_attempt_cap": 3,
            "qualification_pool_seconds_max": 3600, "failed_and_unknown_attempts_charged": True,
            "data": "DEVELOPMENT_ONLY_FIXED_ORDER_FIRST_TWO_CANONICAL_WORLDS_PER_REPAIR_FAMILY",
            "development_path": development["path"], "development_sha256": development["sha256"],
            "generation_row_ids_in_order": ids,
            "optimizer_row_ids_in_order": ids + ids,
            "optimizer_policy": "Disposable qualification only; never carry adapter, optimizer or learned state into any trial; all 16 trials restart from pinned Titan source.",
            "generation_rows": 8, "max_output_tokens": 64, "max_input_tokens": 1984,
            "greedy": True, "thinking": False, "attention": "eager", "tf32": False,
            "eos_ids": [151645, 151643], "generation_forward_requests_max": 512,
            "padding_check_rows": 2, "padding_check_steps": 8,
            "padding_check_row_ids_in_order": ids[:2],
            "padding_check_forwards": 32,
            "disposable_optimizer_updates": 2, "optimizer_microbatches": 16,
            "teacher_forced_forward_requests": 16, "backward_requests": 16,
            "reload_readout_rows": 1, "reload_teacher_forced_forward_requests": 2,
            "reload_readout_row_id": development["reload_readout_row_id"],
            "reload_readout_definition": "Identical prompt and target on MERGED_FP32_DENSE_MODEL before serialization and after exact standard-class reload; no LoRA-versus-merge bit-parity assumption",
            "registered_forward_requests_without_checkpoint_recompute": 562,
            "recomputation_forward_accounting": "MEASURE_AND_CHARGE_IF_ENABLED_NO_GUESSED_COUNTS",
            "full_cpu_hf_serialization_check": "REQUIRED_BEFORE_OPTIMIZER_MICROBATCHES",
            "serialize_reload_check": "STANDARD_FP32_DENSE_DISPOSABLE_EXPORT_AND_IDENTICAL_RELOAD_READOUT",
            "training_defaults": registry.TRAINING,
            "qualification_accuracy_is_capability_evidence": False},
        "admission_requires": ["Exact source/runtime/proposal digest and newly reviewed native worker bytes",
            "Genuine user grant for this experiment and attempt naming an exclusive hardware slot",
            "Qualified Windows Job containment plus retained OS hardware handle and central/temp leases",
            "Read-only CRS idle/compatibility check and owner cooperation",
            "Source backup/custody plus fresh full checkpoint hash under the qualified envelope"],
        "qualification_pass_requires": ["No OOM, reserve violation, unbound descendant or orphan",
            "Finite loss/gradients for 16 complete target-bearing microbatches",
            "Standard class, preserved source weights and no offsets/router/quantization",
            "Exact padded/unpadded decode parity and deterministic reloaded artifact readout",
            "A measured source-bound upper-time estimate covering all 16 trials and 22016 evaluation calls plus exports"],
        "stop_policy": "Retain all failures/costs; no dtype fallback, row dropping, sample expansion or permit reuse",
        "native_worker_implementation_in_this_milestone": False,
        "native_execution_authorized": False, "training_sweep_authorized": False,
        "final_evaluation_authorized": False, "promotion_authorized": False}
    body["proposal_sha256"] = schema.digest(body)
    return body


def dispatch_native(*args, **kwargs):
    # The first approved milestone is an offline packet. A future model worker
    # must enter the reviewed source manifest before any human native grant.
    custody.refuse_native("qualification")
