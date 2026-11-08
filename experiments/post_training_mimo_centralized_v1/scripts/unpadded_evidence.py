"""Strict, stdlib-only evidence diagnostics for explicitly selected unpadded CE scopes.

Value fixtures alone never qualify a bundle. A claimed scoped pass must bind the
registered config, current and committed source closure, native checks, ordered
calls/resources/profiles/costs, runtime/model inventories and containment. Only
after those checks are complete are checkpoint bytes streamed for verification.
This establishes local artifact identity and internal consistency, not
cryptographic proof that an untrusted party executed a model. No native executor,
E1 authorization, current-memory qualification or efficacy claim is provided.
The native qualifier and model libraries are never imported or executed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from experiments.post_training_mimo_centralized_v1.scripts import e1_gates as low


CONFIG_RELATIVE = "configs/E0_LAPTOP_STREAM_UNPADDED_CE.json"
CONFIG_SHA256 = "94797dfa80a17cba8e59559854a45f48a0b48a3c5ff901000d6c79b467fe3772"
CONFIG_SOURCE = "experiments/post_training_mimo_centralized_v1/" + CONFIG_RELATIVE
PASS_STATUS = "PASS_E0_UNPADDED_CE_SCOPED"
RESOURCE_PROFILE = "LAPTOP_STREAMED_V1"
RUNTIME_SCOPE = "UNPADDED_SINGLE_SEQUENCE_CE_V1"
LOADER_MODE = "stream_read_2MiB_v1"
LOADER_LABEL = "stock Qwen3 allocated BF16 on CUDA; strict safetensors header and 2MiB bounded read buffer copy"
DEPENDENCIES = ("experiments/post_training_mimo_centralized_v1/scripts/laptop_profile.py",
                "experiments/post_training_mimo_centralized_v1/scripts/stream_checkpoint.py")
NATIVE_HELPERS = ("experiments/maxrl_comp002_pretraining_v1/mechanics.py",
                  "experiments/maxrl_comp002_pretraining_v1/preflight.py",
                  "experiments/opallo_native_collector_v1/contract.py",
                  "experiments/__init__.py", "experiments/opallo_native_collector_v1/__init__.py")
PACKAGES = {**low.PACKAGES, "psutil": "7.2.2"}
EXECUTABLE = "C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe"
# The fixed raw roundtrip is already recorded with this exact tokenizer in
# results/e0_laptop_load_v1/TOKENIZER_RECEIPT.json; it is not a behavior benchmark.
RAW_TOKEN_IDS = [31029, 2404, 25, 220, 22, 488, 220, 20, 284, 220, 16, 17, 13]
TRUE_CHECKS = {"exact_native_load", "eos_pad_mask", "save_reload_resume_bitwise",
               "frozen_base_unchanged", "finite_scoped_backward"}
CHECKS = TRUE_CHECKS | {"zero_adapter_parity", "native_unpadded_causality", "explicit_allones_mask_equivalence"}
OPERATIONS = ("model_construct", "checkpoint_load", "tensor_copy", "forward", "backward",
              "optimizer_step", "checkpoint_save", "checkpoint_reload")
COST_FIELDS = low.COST_FIELDS
UNQUALIFIED = ["padding", "packing", "batched inference", "KV-cache sampler",
               "sampled online/offline likelihood", "evaluation capture", "preservation",
               "E1 data/replay", "held-out capability", "long context beyond1024"]
RESOURCE_LIMITS = {"host_free_gib": 1.25, "worker_free_gib": 1.25, "worker_rss_limit_gib": 1.75,
                   "gpu_free_gib": 4.0, "gpu_worker_free_gib": .5, "disk_free_gib": 4.0,
                   "wall_seconds": 1800, "admission_budget": "system_reserve_plus_remaining_worker_rss"}
METADATA_LIMIT = 2 * 1024**2
EVENT_LIMIT = 1024**2
# Source-derived contract, not a measurement: two FP32 moments per trainable
# element plus 112 FP32 AdamW step scalars in the pinned native implementation.
OPTIMIZER_TENSOR_BYTES = 2 * 1146880 * 4 + 112 * 4
GateError = low.GateError
_require, _same, _number = low._require, low._same, low._number

# A new scope is never inferred from supplied result metadata. The original
# registration remains the public default, including its immutable config pin.
REGISTRATIONS = {
    "unpadded_ce": {
        "config_relative": CONFIG_RELATIVE, "config_sha256": CONFIG_SHA256,
        "pass_status": PASS_STATUS, "runtime_scope": RUNTIME_SCOPE,
        "contexts": (256, 512, 1024), "optimizer_calls": 5, "forward_calls": 72,
        "operation_calls": 396, "forward_positions": 41456, "training_positions": 3840,
        "resource_samples": 78, "journal_records": 875,
        "unqualified": UNQUALIFIED,
        "bundle_name": "e0_laptop_stream_unpadded_ce_v1",
        "consistency_status": "PASS_E0_UNPADDED_SCOPE_EVIDENCE_CONSISTENCY_ONLY",
    },
    "unpadded_ce512": {
        "config_relative": "configs/E0_LAPTOP_STREAM_UNPADDED_CE512.json",
        "config_sha256": "42a32cfa2fe4dc1250a288bacec2a60d150c5498d8dced9bba23b0c5037f5929",
        "pass_status": "PASS_E0_UNPADDED_CE512_SCOPED",
        "runtime_scope": "UNPADDED_SINGLE_SEQUENCE_CE512_V1",
        "contexts": (256, 512), "optimizer_calls": 4, "forward_calls": 71,
        "operation_calls": 393, "forward_positions": 26608, "training_positions": 1792,
        "resource_samples": 75, "journal_records": 865,
        "unqualified": UNQUALIFIED[:-1] + ["long context beyond512"],
        "bundle_name": "e0_laptop_stream_unpadded_ce512_v1",
        "consistency_status": "PASS_E0_UNPADDED_CE512_SCOPE_EVIDENCE_CONSISTENCY_ONLY",
    },
}


def registration(qualification_stage="unpadded_ce"):
    _require(type(qualification_stage) is str and qualification_stage in REGISTRATIONS,
             "UNREGISTERED_QUALIFICATION_STAGE", "select an explicitly registered unpadded CE stage")
    selected = REGISTRATIONS[qualification_stage]
    _require(type(selected["config_sha256"]) is str and re.fullmatch("[0-9a-f]{64}", selected["config_sha256"]),
             "UNFROZEN_QUALIFICATION_STAGE", "the selected prospective config is not byte-pinned")
    return selected


def _read(path, bound):
    _require(Path(path).stat().st_size <= METADATA_LIMIT, "METADATA_SIZE_LIMIT", str(path))
    return low._read(path, bound)


def successful_accounting(*, qualification_stage="unpadded_ce"):
    selected = registration(qualification_stage)
    counts = {"model_construct": 1, "checkpoint_load": 1, "tensor_copy": 310, "forward": selected["forward_calls"],
              "backward": selected["optimizer_calls"], "optimizer_step": selected["optimizer_calls"],
              "checkpoint_save": 1, "checkpoint_reload": 1}
    return {"accounting_version": 2, "max_optimizer_calls": selected["optimizer_calls"], "last_call_id": selected["operation_calls"],
            "operations": {name: {"attempted": counts[name], "returned": counts[name], "failed": 0, "pending": 0}
                           for name in OPERATIONS}, "pending_calls": [],
            "forward_input_positions_attempted": selected["forward_positions"],
            "forward_input_positions_returned": selected["forward_positions"],
            "training_forward_input_positions_attempted": selected["training_positions"],
            "training_forward_input_positions_returned": selected["training_positions"]}


def resource_stages(*, qualification_stage="unpadded_ce"):
    selected = registration(qualification_stage)
    stages = ["preload", "loaded"] + [f"base_parity_{i}" for i in range(32)]
    stages += [f"zero_parity_{i}" for i in range(32)] + ["unpad_causality", "unpad_allones_equivalence"]
    for length in selected["contexts"]:
        stages += [f"before_backward_{length}", f"backward_{length}", f"after_step_{length}"]
    return stages + ["complete"]


def validate_result(result, *, qualification_stage="unpadded_ce"):
    """Values-only validation; this function never returns a qualification grant."""
    selected = registration(qualification_stage)
    low._finite(result)
    _require(type(result) is dict and result.get("status") == selected["pass_status"],
             "UNPADDED_NOT_PASSED", "exact scoped pass required")
    fixed = {"schema_version": 1, "experiment_id": "RINEYA-0P6B-E0", "status": selected["pass_status"],
             "evidence_level": "MEASURED / REPRODUCED", "dirty_at_launch": False, "config_sha256": selected["config_sha256"],
             "pretrained_loads": 1, "optimizer_steps": selected["optimizer_calls"],
             "training_tokens": selected["training_positions"], "generation_tokens": 0,
             "teacher_calls": 0, "e1_authorized": False, "accounting_version": 2,
             "resource_profile": RESOURCE_PROFILE, "qualification_stage": qualification_stage,
             "loader_mode": LOADER_MODE, "runtime_scope": selected["runtime_scope"], "unqualified": selected["unqualified"]}
    variable = {"code_sha", "wall_seconds", "checks", "resources", "resource_profiles", "call_accounting",
                "loader_diagnostics", "load_seconds", "checkpoint_save_seconds", "checkpoint_reload_seconds"}
    _require(set(result) == set(fixed) | variable, "UNPADDED_RESULT_FIELDS", "missing/extra result fields or failure present")
    for name, expected in fixed.items():
        _require(_same(result[name], expected), "UNPADDED_RESULT_CONTRACT", name)
    _require(type(result["code_sha"]) is str and re.fullmatch("[0-9a-f]{40}", result["code_sha"]),
             "SOURCE_IDENTITY", "full producer commit required")
    _require(_number(result["wall_seconds"], minimum=.000001, maximum=1800), "WALL_CEILING", "invalid native duration")
    _require(_same(result["call_accounting"], successful_accounting(qualification_stage=qualification_stage)),
             "CALL_ACCOUNTING_CONTRACT", "registered counters differ")
    checks = result["checks"]
    _require(type(checks) is dict and set(checks) == CHECKS, "UNPADDED_CHECKS", "exact eight scoped checks required")
    for name in TRUE_CHECKS:
        _require(checks[name] is True, "UNPADDED_CHECK_FAILED", name)
    _require(_same(checks["zero_adapter_parity"], {"prompts": 32, "max_abs": 0.0}), "PARITY_CONTRACT", "32 exact cases required")
    _require(_same(checks["native_unpadded_causality"], {"tokens": 80, "perturbed_tail_tokens": 3,
                     "prefix_tokens": 77, "bitwise": True}), "CAUSALITY_CONTRACT", "bitwise unpadded causality required")
    ones = checks["explicit_allones_mask_equivalence"]
    fixed_ones = {"tokens": 79, "finite": True, "qualified": True,
                  "comparison": "explicit_allones_mask_vs_omitted_same_80_tokens"}
    _require(type(ones) is dict and set(ones) == set(fixed_ones) | {"max_abs", "mean_abs"},
             "ALLONES_CONTRACT", "all-ones diagnostic fields differ")
    for name, expected in fixed_ones.items():
        _require(_same(ones[name], expected), "ALLONES_CONTRACT", name)
    _require(_number(ones["max_abs"], maximum=.03125) and _number(ones["mean_abs"], maximum=.01)
             and ones["mean_abs"] <= ones["max_abs"], "ALLONES_CONTRACT", "registered likelihood tolerance failed")
    profiles = result["resource_profiles"]
    _require(type(profiles) is list and len(profiles) == len(selected["contexts"]),
             "RESOURCE_PROFILES", "all selected context profiles required")
    fields = {"context", "loss", "gradient_norm", "seconds", "tokens_per_second", "peak_allocated_bytes", "optimizer_tensor_bytes"}
    for context, profile in zip(selected["contexts"], profiles):
        _require(type(profile) is dict and set(profile) == fields and type(profile["context"]) is int
                 and profile["context"] == context, "RESOURCE_PROFILES", "profile context/fields differ")
        for field in ("loss", "gradient_norm"):
            _require(_number(profile[field]), "RESOURCE_PROFILES", field)
        _require(_number(profile["seconds"], minimum=.000001, maximum=result["wall_seconds"])
                 and _number(profile["tokens_per_second"], minimum=.000001)
                 and math.isclose(profile["tokens_per_second"], context / profile["seconds"], rel_tol=1e-9),
                 "RESOURCE_PROFILES", "profile throughput or duration differs")
        _require(type(profile["peak_allocated_bytes"]) is int and profile["peak_allocated_bytes"] > 0
                 and type(profile["optimizer_tensor_bytes"]) is int and profile["optimizer_tensor_bytes"] == OPTIMIZER_TENSOR_BYTES,
                 "RESOURCE_PROFILES", "profile byte counters differ")
    samples = result["resources"]
    fields = {"stage", "elapsed_seconds", "host_available_gib", "rss_gib", "gpu_free_gib", "gpu_allocated_gib",
              "gpu_reserved_gib", "peak_allocated_gib", "disk_free_gib", "admission_required_gib"}
    _require(type(samples) is list and len(samples) == selected["resource_samples"] and all(type(sample) is dict for sample in samples)
             and [sample.get("stage") for sample in samples] == resource_stages(qualification_stage=qualification_stage),
             "RESOURCE_COVERAGE", "missing/extra/reordered guards")
    previous = -1
    for sample in samples:
        _require(type(sample) is dict and set(sample) == fields and all(_number(sample[name]) for name in fields - {"stage"}),
                 "RESOURCE_SAMPLES", "invalid resource sample fields/values")
        _require(previous <= sample["elapsed_seconds"] <= result["wall_seconds"], "RESOURCE_TIME", sample["stage"])
        previous = sample["elapsed_seconds"]
        admission = 1.25 + max(0.0, 1.75 - sample["rss_gib"])
        _require(_same(sample["admission_required_gib"], admission) and sample["rss_gib"] <= 1.75
                 and sample["host_available_gib"] >= (admission if sample["stage"] == "preload" else 1.25)
                 and sample["gpu_free_gib"] >= (4.0 if sample["stage"] == "preload" else .5)
                 and sample["disk_free_gib"] >= 4.0, "RESOURCE_FLOOR", sample["stage"])
    for field in ("load_seconds", "checkpoint_save_seconds", "checkpoint_reload_seconds"):
        _require(_number(result[field], minimum=.000001, maximum=result["wall_seconds"]), "TIMING_MEASUREMENT", field)


def _expected_events(result, tensor_names, *, qualification_stage="unpadded_ce"):
    selected = registration(qualification_stage)
    _require(type(tensor_names) in (list, tuple) and len(tensor_names) == 310 and len(set(tensor_names)) == 310
             and all(low._nonblank(name) for name in tensor_names), "JOURNAL_TENSOR_INVENTORY", "310 distinct names required")
    events = [{"event": "runtime_import_start"}, {"event": "runtime_import_complete"}]
    samples, call_id = iter(result["resources"]), 0
    def resource(stage):
        sample = next(samples)
        _require(sample["stage"] == stage, "JOURNAL_RESOURCE_ORDER", stage)
        events.append({"event": "resource", **sample})
    def attempt(operation, stage, positions=0, training=False):
        nonlocal call_id
        call_id += 1
        record = {"accounting_version": 2, "call_id": call_id, "operation": operation,
                  "stage": stage, "input_positions": positions, "training": training}
        events.append({"event": "operation_attempt", **record})
        return record
    def returned(record):
        events.append({"event": "operation_returned", **record})
    def call(operation, stage, positions=0, training=False):
        returned(attempt(operation, stage, positions, training))
    resource("preload")
    call("model_construct", "model_construct")
    checkpoint = attempt("checkpoint_load", "checkpoint_load")
    for name in sorted(tensor_names):
        call("tensor_copy", "tensor_copy:" + name)
    returned(checkpoint)
    resource("loaded")
    for prefix in ("base_parity", "zero_parity"):
        for index in range(32):
            stage = f"{prefix}_{index}"
            call("forward", stage, selected["contexts"][index % len(selected["contexts"])])
            resource(stage)
    call("forward", "unpad_reference", 80)
    call("forward", "unpad_causality_perturbed", 80)
    resource("unpad_causality")
    call("forward", "unpad_allones_mask", 80)
    resource("unpad_allones_equivalence")
    for length, profile in zip(selected["contexts"], result["resource_profiles"]):
        stage = f"training_{length}"
        resource(f"before_backward_{length}")
        call("forward", stage, length, True)
        call("backward", stage)
        resource(f"backward_{length}")
        call("optimizer_step", stage)
        events.append({"event": "resource_profile", "profile": profile})
        resource(f"after_step_{length}")
    call("checkpoint_save", "adapter_resume_save")
    call("forward", "resume_uninterrupted", selected["contexts"][-1], True)
    call("backward", "resume_uninterrupted")
    call("optimizer_step", "resume_uninterrupted")
    call("checkpoint_reload", "adapter_resume_reload")
    call("forward", "resume_reloaded", selected["contexts"][-1], True)
    call("backward", "resume_reloaded")
    call("optimizer_step", "resume_reloaded")
    resource("complete")
    _require(call_id == selected["operation_calls"] and next(samples, None) is None
             and len(events) == selected["journal_records"], "JOURNAL_WORKLOAD_CONTRACT", "workload differs")
    return events


def validate_accounting_bundle(result, cost, events, tensor_names, *, qualification_stage="unpadded_ce"):
    selected = registration(qualification_stage)
    validate_result(result, qualification_stage=qualification_stage)
    _require(type(cost) is dict and _same(cost, {name: result[name] for name in COST_FIELDS}), "COST_RECEIPT_DRIFT", "cost/result differ")
    expected = _expected_events(result, tensor_names, qualification_stage=qualification_stage)
    _require(type(events) is list and len(events) == selected["journal_records"],
             "EVENTS_CONTRACT", "complete selected-scope journal required")
    previous = -1
    for index, (event, contract) in enumerate(zip(events, expected)):
        _require(type(event) is dict and set(event) == set(contract) | {"elapsed_seconds"}, "EVENTS_CONTRACT", f"fields at {index}")
        _require(_number(event["elapsed_seconds"]) and previous <= event["elapsed_seconds"] <= result["wall_seconds"],
                 "EVENT_TIME", str(index))
        previous = event["elapsed_seconds"]
        _require(_same({key: value for key, value in event.items() if key != "elapsed_seconds"},
                      {key: value for key, value in contract.items() if key != "elapsed_seconds"}), "EVENTS_CONTRACT", f"record {index}")
        if "elapsed_seconds" in contract:
            _require(_same(event["elapsed_seconds"], contract["elapsed_seconds"]), "EVENTS_CONTRACT", f"sample time {index}")
    # Every native timer starts before intent is journaled and stops after the
    # corresponding Python return. A shorter reported duration contradicts the
    # same ordered trace. This epsilon covers floating subtraction, not a clock
    # resolution allowance or permission to relax measured runtime.
    def elapsed(kind, operation, stage):
        return next(event["elapsed_seconds"] for event in events if event["event"] == kind
                    and event.get("operation") == operation and event.get("stage") == stage)
    minimum_load = (elapsed("operation_returned", "checkpoint_load", "checkpoint_load")
                    - elapsed("operation_attempt", "model_construct", "model_construct"))
    _require(result["load_seconds"] + 1e-9 >= minimum_load, "TIMING_JOURNAL_CONTRADICTION", "load duration shorter than load trace")
    for field, operation, stage in (("checkpoint_save_seconds", "checkpoint_save", "adapter_resume_save"),
                                     ("checkpoint_reload_seconds", "checkpoint_reload", "adapter_resume_reload")):
        minimum = elapsed("operation_returned", operation, stage) - elapsed("operation_attempt", operation, stage)
        _require(result[field] + 1e-9 >= minimum, "TIMING_JOURNAL_CONTRADICTION", field)
    for profile in result["resource_profiles"]:
        stage = f"training_{profile['context']}"
        minimum = (elapsed("operation_returned", "optimizer_step", stage)
                   - elapsed("operation_attempt", "forward", stage))
        _require(profile["seconds"] + 1e-9 >= minimum, "TIMING_JOURNAL_CONTRADICTION", stage)


def _validate_inventory(model, trainable, primary, inventory, result):
    _require(set(model) == {"weight_sha256", "primary_receipt_sha256", "unique_parameters", "tied", "loader", "config",
                            "tensor_sha256", "loader_mode", "loader_diagnostics"}, "MODEL_FIELDS", "model fields differ")
    for field, value in {"weight_sha256": low.WEIGHT_SHA256, "primary_receipt_sha256": low.PRIMARY_SHA256,
                         "unique_parameters": 596049920, "tied": True, "loader": LOADER_LABEL, "loader_mode": LOADER_MODE}.items():
        _require(_same(model[field], value), "MODEL_IDENTITY", field)
    for field in ("architectures", "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                  "intermediate_size", "head_dim", "rope_theta", "rms_norm_eps", "vocab_size", "tie_word_embeddings", "attention_bias",
                  "attention_dropout", "rope_scaling", "use_sliding_window", "max_position_embeddings", "hidden_act"):
        _require(_same(model["config"].get(field), primary["architecture"].get(field)), "MODEL_CONFIG_DRIFT", field)
    hashes = model["tensor_sha256"]
    _require(type(hashes) is dict and set(hashes) == set(inventory["tensors"])
             and all(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) for value in hashes.values()), "TENSOR_INVENTORY", "tensor hashes differ")
    lengths = [entry["data_offsets"][1] - entry["data_offsets"][0] for entry in inventory["tensors"].values()]
    diagnostics = {"loader_mode": LOADER_MODE, "header_bytes": inventory["header_bytes"], "file_bytes": inventory["file_bytes"],
                   "payload_bytes": inventory["payload_bytes"], "serialized_tensors": 310, "read_buffer_bytes": 2097152,
                   "maximum_payload_read_bytes": min(2097152, max(lengths)), "payload_chunk_count": sum((length + 2097151) // 2097152 for length in lengths),
                   "memory_mapped": False, "cpu_model_allocated": False, "blocking_chunk_copies": True}
    _require(_same(model["loader_diagnostics"], diagnostics) and _same(result["loader_diagnostics"], diagnostics),
             "LOADER_DIAGNOSTICS", "bounded-loader receipts differ")
    modules = {f"model.layers.{layer}.self_attn.{projection}_proj" for layer in range(28) for projection in ("q", "v")}
    parameters = {}
    for module in modules:
        parameters[module + ".lora_A"] = {"shape": [8, 1024], "dtype": "torch.float32"}
        parameters[module + ".lora_B"] = {"shape": [2048 if module.endswith("q_proj") else 1024, 8], "dtype": "torch.float32"}
    _require(type(trainable) is dict and set(trainable) == {"modules", "parameters", "count"}
             and type(trainable["modules"]) is list and len(trainable["modules"]) == 56 and set(trainable["modules"]) == modules
             and _same(trainable["parameters"], parameters) and _same(trainable["count"], 1146880), "LORA_SCOPE", "rank-8 q/v inventory differs")


def _bind_binary(path, bound, ceiling, *, nonempty=False):
    path = Path(path).resolve()
    size = path.stat().st_size
    _require(size <= ceiling and (not nonempty or size > 0), "BINARY_SIZE_LIMIT", str(path))
    bound[str(path)] = {"sha256": low._digest(path), "bytes": size}


def _validate_stdout(path, result, *, qualification_stage="unpadded_ce"):
    selected = registration(qualification_stage)
    last = b""
    with Path(path).open("rb") as stream:
        while True:
            line = stream.readline(METADATA_LIMIT + 1)
            if not line:
                break
            _require(len(line) <= METADATA_LIMIT, "STDOUT_CONTRACT", "oversized stdout line")
            if line.strip():
                last = line.strip()
    expected = {"status": selected["pass_status"], "error": None, "wall_seconds": result["wall_seconds"],
                "optimizer_steps": selected["optimizer_calls"]}
    _require(last and _same(low._parse(last), expected), "STDOUT_CONTRACT", "final worker stdout differs from result")


def _verify_assets(cfg, primary, inventory, model, bound):
    model_dir = Path(primary["checkpoint_path"]).resolve()
    _require(Path(cfg["model_dir"]).resolve() == model_dir, "MODEL_PATH", "checkpoint path differs")
    for name, digest in {**cfg["asset_sha256"], "model.safetensors": low.WEIGHT_SHA256}.items():
        path = model_dir / name
        _require(primary["files"][name]["sha256"] == digest and low._digest(path) == digest, "MODEL_ASSET_DRIFT", name)
        bound[str(path)] = {"sha256": digest, "bytes": path.stat().st_size}
    low._check_weight_tensor_bytes(model_dir / "model.safetensors", inventory, model)


def inspect_unpadded_scope(experiment_root, *, e0_directory=None, qualification_stage="unpadded_ce"):
    root = Path(experiment_root).resolve()
    bound = {}
    report = {"status": "BLOCKED", "reasons": [], "bound_files": bound, "native_launch_authorized": False,
              "no_native_executor": True, "e1_authorized": False,
              "claim": "Local artifact identity and internal consistency only; no cryptographic execution proof, current-memory or efficacy claim."}
    try:
        selected = registration(qualification_stage)
        bundle = Path(e0_directory).resolve() if e0_directory is not None else root / "results" / selected["bundle_name"]
        result = _read(bundle / "E0_RESULT.json", bound)
        if result.get("status") != selected["pass_status"]:
            report["reasons"].append({"code": "UNPADDED_NOT_PASSED", "detail": str(result.get("status", "MISSING_STATUS"))[:200]})
            return report
        config_path = root / selected["config_relative"]
        cfg = _read(config_path, bound)
        _require(bound[str(config_path)]["sha256"] == selected["config_sha256"],
                 "UNPADDED_CONFIG_IDENTITY", "registered config bytes changed")
        _require(_same(_read(bundle / "CONFIG.json", bound), cfg), "UNPADDED_CONFIG_DRIFT", "bundle config differs")
        validate_result(result, qualification_stage=qualification_stage)
        supervision = bundle.with_name(bundle.name + "_supervisor")
        _require(not (bundle / "FAILURE_RECEIPT.json").exists() and not (supervision / "FAILURE_RECEIPT.json").exists(),
                 "FAILURE_PRESENT", "pass has failure receipt")
        primary_path = root / "receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json"
        inventory_path = root / "receipts/titan_static_v1/PARAMETER_INVENTORY.json"
        primary, inventory = _read(primary_path, bound), _read(inventory_path, bound)
        _require(bound[str(primary_path)]["sha256"] == low.PRIMARY_SHA256 and bound[str(inventory_path)]["sha256"] == low.INVENTORY_SHA256,
                 "STATIC_IDENTITY", "pinned metadata changed")
        repo = root.parents[1]
        primary_alias = repo / cfg["primary_receipt"].replace("\\", "/")
        alias = _read(primary_alias, bound)
        _require(bound[str(primary_alias.resolve())]["sha256"] == low.PRIMARY_SHA256 and _same(alias, primary),
                 "PRIMARY_RECEIPT_DRIFT", "the actual worker primary-receipt path changed")
        source = _read(bundle / "SOURCE_RECEIPT.json", bound)
        _require(set(source) == {"code_sha", "dirty", "script_sha256", "dependency_sha256"} and source["code_sha"] == result["code_sha"]
                 and source["dirty"] is False and type(source["dependency_sha256"]) is dict
                 and set(source["dependency_sha256"]) == set(DEPENDENCIES), "SOURCE_IDENTITY", "source receipt/closure differs")
        config_source = "experiments/post_training_mimo_centralized_v1/" + selected["config_relative"]
        low._verify_source(repo, result["code_sha"], config_source, selected["config_sha256"], bound)
        low._verify_source(repo, result["code_sha"], low.QUALIFIER, source["script_sha256"], bound)
        for relative, digest in source["dependency_sha256"].items():
            low._verify_source(repo, result["code_sha"], relative, digest, bound)
        for relative in NATIVE_HELPERS:
            low._verify_committed_helper(repo, result["code_sha"], relative, bound)
        runtime = _read(bundle / "RUNTIME_RECEIPT.json", bound)
        _require(set(runtime) == {"python", "executable", "platform", "packages", "gpu", "cuda", "total_vram_bytes",
                                  "attention", "base_dtype", "adapter_dtype"}
                 and _same(runtime.get("packages"), PACKAGES) and runtime.get("attention") == "eager" and runtime.get("base_dtype") == "bfloat16"
                 and runtime.get("adapter_dtype") == "float32" and runtime.get("cuda") == "12.6"
                 and type(runtime.get("python")) is str and runtime["python"].startswith("3.11.15 ")
                 and Path(runtime.get("executable", "")).resolve() == Path(EXECUTABLE).resolve()
                 and type(runtime.get("total_vram_bytes")) is int and runtime["total_vram_bytes"] > 0
                 and low._nonblank(runtime.get("gpu")) and low._nonblank(runtime.get("platform"))
                 and runtime["platform"].startswith("Windows-"), "RUNTIME_IDENTITY", "pinned native runtime differs")
        tokenizer = _read(bundle / "TOKENIZER_RECEIPT.json", bound)
        fixed_tokenizer = {"class_name": "Qwen2TokenizerFast", "template_sha256": low.TEMPLATE_SHA256, "eos_id": 151645,
                           "original_pad_id": 151643, "raw_roundtrip": True, "enable_thinking": False, "add_generation_prompt": True}
        _require(set(tokenizer) == set(fixed_tokenizer) | {"raw_ids"} and all(_same(tokenizer[name], value) for name, value in fixed_tokenizer.items())
                 and _same(tokenizer["raw_ids"], RAW_TOKEN_IDS), "TOKENIZER_IDENTITY", "tokenizer receipt differs")
        model, trainable = _read(bundle / "MODEL_RECEIPT.json", bound), _read(bundle / "TRAINABLE_INVENTORY.json", bound)
        _validate_inventory(model, trainable, primary, inventory, result)
        resources = _read(bundle / "RESOURCE_BASELINE.json", bound)
        _require(_same(resources, {"status": selected["pass_status"], "samples": result["resources"], "profiles": result["resource_profiles"]}),
                 "RESOURCE_RECEIPT_DRIFT", "resource receipt differs")
        cost = _read(bundle / "COST_LEDGER.json", bound)
        event_path = bundle / "EVENTS.jsonl"
        _require(event_path.stat().st_size <= EVENT_LIMIT, "EVENT_SIZE_LIMIT", "journal too large")
        raw = event_path.read_bytes()
        lines = raw.splitlines()
        _require(len(lines) == selected["journal_records"] and all(line.strip() for line in lines),
                 "EVENTS_CONTRACT", "complete journal required")
        bound[str(event_path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        validate_accounting_bundle(result, cost, [low._parse(line) for line in lines], sorted(inventory["tensors"]),
                                   qualification_stage=qualification_stage)
        supervisor = _read(supervision / "SUPERVISOR.json", bound)
        _require(set(supervisor) == {"status", "reason", "exit_code", "native_qualification", "elapsed_seconds", "limit_seconds",
                                    "memory_poll_seconds", "peak_observed_process_tree_rss_bytes"}
                 and supervisor["status"] == "PASS_PROCESS_ONLY" and supervisor["reason"] is None and _same(supervisor["exit_code"], 0)
                 and supervisor["native_qualification"] is False and _same(supervisor["limit_seconds"], 1800)
                 and _same(supervisor["memory_poll_seconds"], .1)
                 and _number(supervisor["elapsed_seconds"], minimum=result["wall_seconds"], maximum=1800)
                 and type(supervisor["peak_observed_process_tree_rss_bytes"]) is int
                 and 0 < supervisor["peak_observed_process_tree_rss_bytes"] <= 1.75 * 2**30,
                 "SUPERVISOR_CONTRACT", "external containment differs")
        launch, go = _read(supervision / "LAUNCH.json", bound), _read(supervision / "GO.json", bound)
        _require(set(launch) == {"command", "config_sha256", "supervisor_sha256", "watchdog_sha256", "observed_gpu_processes", "source_sha",
                                "resource_profile", "qualification_stage", "resource_limits", "loader_mode"}
                 and _same(go, {"config_sha256": selected["config_sha256"]})
                 and launch["config_sha256"] == selected["config_sha256"]
                 and launch["source_sha"] == result["code_sha"] and launch["resource_profile"] == RESOURCE_PROFILE
                 and launch["qualification_stage"] == qualification_stage and launch["loader_mode"] == LOADER_MODE
                 and _same(launch["resource_limits"], RESOURCE_LIMITS), "LAUNCH_IDENTITY", "launch limits/config/scope differ")
        command = launch["command"]
        _require(type(command) is list and len(command) == 10 and command[:4] == [runtime["executable"], "-B", "-m",
                    "experiments.post_training_mimo_centralized_v1.scripts.qualify_native"]
                 and command[4] == "--config" and command[6] == "--output" and command[8] == "--go-path"
                 and Path(command[5]).resolve() == config_path and Path(command[7]).name == bundle.name
                 and Path(command[9]).name == "GO.json"
                 and Path(command[9]).parent == Path(command[7]).with_name(bundle.name + "_supervisor"),
                 "LAUNCH_IDENTITY", "command differs from registered worker")
        observed = launch["observed_gpu_processes"]
        _require(type(observed) is list and all(low._nonblank(line) for line in observed)
                 and not any(term in line.lower() for line in observed for term in ("python", "torch", "llama", "ollama", "vllm", "train", "comfy")),
                 "GPU_COMPETITOR_RECORDED", "competing runtime recorded at launch")
        low._verify_source(repo, result["code_sha"], low.SUPERVISOR, launch["supervisor_sha256"], bound)
        low._verify_source(repo, result["code_sha"], low.WATCHDOG, launch["watchdog_sha256"], bound)
        _require(all(inventory["payload_bytes"] <= profile["peak_allocated_bytes"] <= runtime["total_vram_bytes"]
                     for profile in result["resource_profiles"]), "RESOURCE_PROFILES", "GPU peak contradicts resident base/device memory")
        _require(all(sample["gpu_allocated_gib"] * 2**30 >= inventory["payload_bytes"]
                     for sample in result["resources"] if sample["stage"] != "preload"),
                 "RESOURCE_SAMPLES", "post-load GPU allocation cannot omit resident base weights")
        total_vram_gib = runtime["total_vram_bytes"] / 2**30
        _require(all(sample["gpu_allocated_gib"] <= sample["gpu_reserved_gib"] <= total_vram_gib
                     and sample["gpu_allocated_gib"] <= sample["peak_allocated_gib"] <= total_vram_gib
                     and sample["gpu_free_gib"] <= total_vram_gib for sample in result["resources"]),
                 "RESOURCE_SAMPLES", "allocated/reserved/peak/free memory contradict device capacity")
        _bind_binary(bundle / "adapter_resume.pt", bound, 32 * 1024**2, nonempty=True)
        _bind_binary(supervision / "stdout.log", bound, 16 * 1024**2)
        _bind_binary(supervision / "stderr.log", bound, 16 * 1024**2)
        _require(bound[str((supervision / "stdout.log").resolve())]["bytes"] + bound[str((supervision / "stderr.log").resolve())]["bytes"] <= 16 * 1024**2,
                 "LOG_SIZE_LIMIT", "combined logs exceed watchdog limit")
        _validate_stdout(supervision / "stdout.log", result, qualification_stage=qualification_stage)
        _verify_assets(cfg, primary, inventory, model, bound)
        report.update(status=selected["consistency_status"], runtime_scope=selected["runtime_scope"],
                      qualified_contexts=list(selected["contexts"]), supported_maximum_tokens=selected["contexts"][-1],
                      supported_forward="one unpadded sequence, no cache; validated all-ones mask omitted")
    except GateError as error:
        report["reasons"].append({"code": error.code, "detail": str(error)[:400]})
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError, OverflowError, StopIteration, subprocess.SubprocessError) as error:
        report["reasons"].append({"code": "UNPADDED_ARTIFACT_INVALID_OR_MISSING", "detail": f"{type(error).__name__}: {error}"[:400]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--e0-directory", type=Path)
    parser.add_argument("--qualification-stage", choices=tuple(REGISTRATIONS), default="unpadded_ce",
                        help="Explicit scope selection; the historical unpadded_ce remains the default")
    args = parser.parse_args()
    print(json.dumps(inspect_unpadded_scope(args.experiment_root, e0_directory=args.e0_directory,
                                          qualification_stage=args.qualification_stage), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
