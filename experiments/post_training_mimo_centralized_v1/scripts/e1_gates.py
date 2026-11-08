"""Fail-closed E1 evidence diagnostics; stdlib only, with no native executor.

Contract: a nonpassing E0_RESULT stops E0 inspection immediately. A claimed pass
must bind the pinned E0 config/static Titan receipts, committed and current native
source, all native check values, complete version-2 call/resource/cost records, tokenizer/model/
trainable inventories and process-containment receipts. Only then are current
checkpoint assets streamed for SHA256 verification (never loaded as tensors).
The model receipt's per-tensor digests are checked against safetensors byte spans.

These checks establish local artifact identity and internal evidence consistency,
not cryptographic proof that an untrusted party executed a model. Fixture status
strings or check booleans cannot satisfy the bundle and pinned-byte requirements.
No callback, supplied status or arbitrary receipt can qualify downstream scopes:
source enrollment, fresh language grading and the 2048/4096 runtime extension
remain blocked until their scoped native-evidence validators are implemented.
Every returned report says native_launch_authorized=False/no_native_executor=True.
E0's e1_authorized field is always False, including real successful E0 runs.
New PASS claims require the complete ordered operation journal and matching
attempted/returned counters. Historical nonpassing receipts remain readable
without any version-2 fields. Journal verification here is independently defined;
the native qualifier is never imported or executed by this diagnostic module.

Public entry: preflight(experiment_root, *, e0_directory=None). Errors are stable
codes in gate.reasons; bounded detail strings explain missing/drifted artifacts.
No output files are written, no CHECK/FINAL payloads opened, no prep module imported.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess


WEIGHT_SHA256 = "537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7"
CONFIG_SHA256 = "8fcab3ce006c9a698af86975769a4076edc78cf8005f64ad823d9d47bcbe518d"
PRIMARY_SHA256 = "b6cac2bc479cce98be73a08c4264bbf44e60ca689e6231e40952d8b9e5bc6ba1"
INVENTORY_SHA256 = "8f0e1799cf64ca31afd205e411156b566aa0b14b8a6bfaf9617d617baf5b8124"
TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
PASS_STATUS = "PASS_E0_SCOPED_NO_EFFICACY_CLAIM"
QUALIFIER = "experiments/post_training_mimo_centralized_v1/scripts/qualify_native.py"
SUPERVISOR = "experiments/post_training_mimo_centralized_v1/scripts/supervise_native.py"
WATCHDOG = "experiments/opallo_native_collector_v1/supervisor.py"
NATIVE_HELPERS = ("experiments/maxrl_comp002_pretraining_v1/mechanics.py",
                  "experiments/maxrl_comp002_pretraining_v1/preflight.py",
                  "experiments/opallo_native_collector_v1/contract.py",
                  "experiments/post_training_mimo_centralized_v1/scripts/laptop_profile.py",
                  "experiments/post_training_mimo_centralized_v1/scripts/stream_checkpoint.py")
PACKAGES = {"torch": "2.10.0+cu126", "transformers": "4.57.1",
            "safetensors": "0.7.0", "tokenizers": "0.22.2"}
TRUE_CHECKS = {"exact_native_load", "native_padding_and_causality", "eos_pad_mask",
               "save_reload_resume_bitwise", "frozen_base_unchanged", "finite_scoped_backward"}
COST_FIELDS = ("wall_seconds", "training_tokens", "generation_tokens", "optimizer_steps",
               "pretrained_loads", "teacher_calls", "accounting_version", "call_accounting")
OPERATIONS = ("model_construct", "checkpoint_load", "tensor_copy", "forward", "backward",
              "optimizer_step", "checkpoint_save", "checkpoint_reload")


class GateError(ValueError):
    """A code-bearing diagnostic, never an execution approval request."""
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def _require(condition, code, detail):
    if not condition:
        raise GateError(code, detail)


def _finite(value):
    if type(value) is dict:
        for child in value.values():
            _finite(child)
    elif type(value) is list:
        for child in value:
            _finite(child)
    elif type(value) is float:
        _require(math.isfinite(value), "NONFINITE_JSON", "all numbers must be finite")


def _unique(pairs):
    value = {}
    for key, child in pairs:
        _require(key not in value, "DUPLICATE_JSON_KEY", key)
        value[key] = child
    return value


def _parse(raw):
    def reject(value):
        raise GateError("NONFINITE_JSON", value)
    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=reject)
    _finite(value)
    return value


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path, bound):
    path = Path(path).resolve()
    raw = path.read_bytes()
    value = _parse(raw)
    _require(type(value) is dict, "OBJECT_REQUIRED", str(path))
    bound[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return value


def _number(value, *, minimum=0, maximum=None):
    try:
        return (type(value) in (int, float) and math.isfinite(value) and value >= minimum
                and (maximum is None or value <= maximum))
    except OverflowError:
        return False


def _nonblank(value):
    return type(value) is str and bool(value.strip())


def _same(actual, expected):
    """Prevent True from matching 1 in native count/identity contracts."""
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)


def _successful_accounting(generated_tokens, maximum_optimizer_calls):
    """Registered E0 workload totals; declarations alone are never journal evidence."""
    counts = {"model_construct": 1, "checkpoint_load": 1, "tensor_copy": 310,
              "forward": 75 + generated_tokens, "backward": 5, "optimizer_step": 5,
              "checkpoint_save": 1, "checkpoint_reload": 1}
    positions = 41704 + 81 * generated_tokens + generated_tokens * (generated_tokens - 1) // 2
    return {
        "accounting_version": 2, "max_optimizer_calls": maximum_optimizer_calls,
        "last_call_id": 399 + generated_tokens,
        "operations": {name: {"attempted": counts[name], "returned": counts[name], "failed": 0, "pending": 0}
                       for name in OPERATIONS},
        "pending_calls": [], "forward_input_positions_attempted": positions,
        "forward_input_positions_returned": positions,
        "training_forward_input_positions_attempted": 3840,
        "training_forward_input_positions_returned": 3840,
    }


def validate_result(result, config):
    """Validate E0 values only. This function alone never qualifies an evidence gate."""
    _finite(result)
    _require(type(result) is dict and result.get("status") == PASS_STATUS,
             "E0_NOT_PASSED", "E0 must have its exact registered pass status")
    expected = {"schema_version": 1, "experiment_id": "RINEYA-0P6B-E0",
                "evidence_level": "MEASURED / REPRODUCED", "dirty_at_launch": False,
                "pretrained_loads": 1, "optimizer_steps": 5, "training_tokens": 3840,
                "teacher_calls": 0, "e1_authorized": False, "config_sha256": CONFIG_SHA256,
                "accounting_version": 2}
    for name, value in expected.items():
        _require(_same(result.get(name), value), "E0_RESULT_CONTRACT", name)
    _require(not result.get("error") and not result.get("traceback"), "E0_FAILURE_PRESENT", "pass contains failure")
    _require(type(result.get("code_sha")) is str and re.fullmatch(r"[0-9a-f]{40}", result["code_sha"]),
             "SOURCE_IDENTITY", "full Git source SHA required")
    _require(_number(result.get("wall_seconds"), minimum=0.000001, maximum=config["wall_seconds"]),
             "WALL_CEILING", "invalid E0 wall duration")
    checks = result.get("checks")
    _require(type(checks) is dict and set(checks) == TRUE_CHECKS | {"zero_adapter_parity", "sampled_logprobs"},
             "E0_CHECKS_INCOMPLETE", "exact native mechanical check set required")
    for name in TRUE_CHECKS:
        _require(checks[name] is True, "E0_CHECK_FAILED", name)
    parity = checks["zero_adapter_parity"]
    _require(type(parity) is dict and type(parity.get("prompts")) is int and parity["prompts"] == 32
             and _number(parity.get("max_abs"), maximum=0), "PARITY_CONTRACT", "32 exact zero-adapter comparisons required")
    sampled = checks["sampled_logprobs"]
    _require(type(sampled) is dict, "LIKELIHOOD_CONTRACT", "sampled likelihood measurements missing")
    tokens = sampled.get("tokens")
    _require(type(tokens) is list and 1 <= len(tokens) <= 32
             and all(type(token) is int and 0 <= token < 151936 for token in tokens),
             "LIKELIHOOD_CONTRACT", "invalid sampled token stream")
    _require(type(result.get("generation_tokens")) is int and result["generation_tokens"] == len(tokens),
             "GENERATION_ACCOUNTING", "token stream and counter differ")
    _require(_same(result.get("call_accounting"), _successful_accounting(len(tokens), config["maximum_optimizer_calls"])),
             "CALL_ACCOUNTING_CONTRACT", "complete successful version-2 operation counts and input-position totals required")
    _require(sampled.get("backend") == "full_prefix_no_cache_temperature1_topk0_topp1"
             and _number(sampled.get("max_abs"), maximum=.03125)
             and _number(sampled.get("mean_abs"), maximum=.01)
             and sampled["mean_abs"] <= sampled["max_abs"],
             "LIKELIHOOD_CONTRACT", "backend or numerical likelihood tolerance differs")
    eos = config["eos_ids"]
    _require(not any(token in eos for token in tokens[:-1])
             and sampled.get("stop_reason") == ("eos" if tokens[-1] in eos else "length")
             and (tokens[-1] in eos or len(tokens) == 32),
             "GENERATION_TERMINATION", "EOS/length termination does not match the sampled stream")
    profiles = result.get("resource_profiles")
    _require(type(profiles) is list and all(type(p) is dict and type(p.get("context")) is int for p in profiles)
             and [p.get("context") for p in profiles] == [256, 512, 1024],
             "RESOURCE_PROFILES", "three ordered training-context profiles required")
    for profile in profiles:
        for name in ("loss", "gradient_norm", "seconds", "tokens_per_second", "peak_allocated_bytes", "optimizer_tensor_bytes"):
            _require(_number(profile.get(name), minimum=0.000001), "RESOURCE_PROFILES", name)
        _require(all(type(profile[name]) is int for name in ("peak_allocated_bytes", "optimizer_tensor_bytes"))
                 and profile["seconds"] <= result["wall_seconds"], "RESOURCE_PROFILES", "invalid bytes or duration")
        _require(math.isclose(profile["tokens_per_second"], profile["context"] / profile["seconds"], rel_tol=1e-9),
                 "RESOURCE_PROFILES", "throughput disagrees with measured duration")
    resources = result.get("resources")
    _require(type(resources) is list and bool(resources), "RESOURCE_SAMPLES", "resource measurements missing")
    expected_stages = ["preload", "loaded"] + [f"base_parity_{i}" for i in range(32)]
    expected_stages += [f"zero_parity_{i}" for i in range(32)]
    expected_stages += ["native_padding_2_0", "native_padding_0_2", "native_padding_2_2"]
    expected_stages += ["sample"] * len(tokens)
    for length in config["contexts"]:
        expected_stages += [f"before_backward_{length}", f"backward_{length}", f"after_step_{length}"]
    expected_stages += ["complete"]
    _require([sample.get("stage") for sample in resources] == expected_stages,
             "RESOURCE_COVERAGE", "resource stages are missing, reordered or duplicated")
    elapsed = -1
    for sample in resources:
        stage = sample["stage"]
        for name in ("elapsed_seconds", "host_available_gib", "rss_gib", "gpu_free_gib", "gpu_allocated_gib",
                     "gpu_reserved_gib", "peak_allocated_gib", "disk_free_gib"):
            _require(_number(sample.get(name)), "RESOURCE_SAMPLES", name)
        _require(elapsed <= sample["elapsed_seconds"] <= result["wall_seconds"], "RESOURCE_TIME", stage)
        elapsed = sample["elapsed_seconds"]
        _require(sample["host_available_gib"] >= config["host_free_gib" if stage == "preload" else "worker_free_gib"]
                 and sample["gpu_free_gib"] >= config["gpu_free_gib" if stage == "preload" else "gpu_worker_free_gib"]
                 and sample["disk_free_gib"] >= config["disk_free_gib"], "RESOURCE_FLOOR", stage)
    for name in ("load_seconds", "generation_seconds", "generation_tokens_per_second",
                 "checkpoint_save_seconds", "checkpoint_reload_seconds"):
        _require(_number(result.get(name), minimum=0.000001), "TIMING_MEASUREMENT", name)
        if name != "generation_tokens_per_second":
            _require(result[name] <= result["wall_seconds"], "TIMING_MEASUREMENT", name + " exceeds wall duration")
    _require(math.isclose(result["generation_tokens_per_second"], len(tokens) / result["generation_seconds"], rel_tol=1e-9),
             "TIMING_MEASUREMENT", "generation throughput disagrees with measured duration")


def _expected_native_events(result, tensor_names):
    """Complete registered successful event order, including nested checkpoint copies."""
    _require(type(tensor_names) in (list, tuple) and len(tensor_names) == 310
             and all(_nonblank(name) for name in tensor_names) and len(set(tensor_names)) == 310,
             "JOURNAL_TENSOR_INVENTORY", "exact 310 serialized tensor names required")
    events = [{"event": "runtime_import_start"}, {"event": "runtime_import_complete"}]
    resources = iter(result["resources"])
    call_id = 0

    def resource(stage):
        sample = next(resources)
        _require(sample["stage"] == stage, "JOURNAL_RESOURCE_ORDER", stage)
        events.append({"event": "resource", **sample})

    def attempt(operation, stage, positions=0, training=False):
        nonlocal call_id
        call_id += 1
        call = {"accounting_version": 2, "call_id": call_id, "operation": operation,
                "input_positions": positions, "training": training, "stage": stage}
        events.append({"event": "operation_attempt", **call})
        return call

    def returned(call):
        events.append({"event": "operation_returned", **call})

    def operation(name, stage, positions=0, training=False):
        returned(attempt(name, stage, positions, training))

    resource("preload")
    operation("model_construct", "model_construct")
    checkpoint = attempt("checkpoint_load", "checkpoint_load")
    for name in sorted(tensor_names):
        operation("tensor_copy", "tensor_copy:" + name)
    returned(checkpoint)
    resource("loaded")
    for prefix in ("base_parity", "zero_parity"):
        for index in range(32):
            stage = f"{prefix}_{index}"
            operation("forward", stage, (256, 512, 1024)[index % 3])
            resource(stage)
    operation("forward", "padding_reference", 80)
    for left, right in ((2, 0), (0, 2), (2, 2)):
        stage = f"native_padding_{left}_{right}"
        operation("forward", stage, 80 + left + right)
        resource(stage)
    operation("forward", "causality_perturbed", 80)
    for index in range(result["generation_tokens"]):
        operation("forward", f"sample_{index}", 80 + index)
        resource("sample")
    operation("forward", "sampled_logprobs_offline", 80 + result["generation_tokens"])
    for length, profile in zip((256, 512, 1024), result["resource_profiles"]):
        stage = f"training_{length}"
        resource(f"before_backward_{length}")
        operation("forward", stage, length, True)
        operation("backward", stage)
        resource(f"backward_{length}")
        operation("optimizer_step", stage)
        events.append({"event": "resource_profile", "profile": profile})
        resource(f"after_step_{length}")
    operation("checkpoint_save", "adapter_resume_save")
    for stage in ("resume_uninterrupted", "resume_reloaded"):
        if stage == "resume_reloaded":
            operation("checkpoint_reload", "adapter_resume_reload")
        operation("forward", stage, 1024, True)
        operation("backward", stage)
        operation("optimizer_step", stage)
    resource("complete")
    _require(next(resources, None) is None and call_id == result["call_accounting"]["last_call_id"],
             "JOURNAL_WORKLOAD_CONTRACT", "resource or call inventory differs")
    return events


def validate_accounting_bundle(result, cost, events, config, tensor_names):
    """Check values, exact cost snapshot, and every ordered native journal record.

    A successful journal has no failed or pending operation, duplicate/missing
    attempt, metadata drift, extra call or unregistered event. Its order binds
    forwards to resource guards and completed profiles, rather than accepting an
    event-name whitelist or final counters on their own.
    """
    validate_result(result, config)
    _require(type(cost) is dict and _same(cost, {name: result[name] for name in COST_FIELDS}),
             "COST_RECEIPT_DRIFT", "legacy costs and version-2 accounting snapshot must match result exactly")
    _require(type(events) is list, "EVENTS_CONTRACT", "ordered event list required")
    expected = _expected_native_events(result, tensor_names)
    _require(len(events) == len(expected), "EVENTS_CONTRACT", "incomplete or extra native operation/resource records")
    previous = -1
    for index, (event, contract) in enumerate(zip(events, expected)):
        _require(type(event) is dict and set(event) == set(contract) | {"elapsed_seconds"},
                 "EVENTS_CONTRACT", f"event {index} field inventory")
        _require(_number(event["elapsed_seconds"]) and previous <= event["elapsed_seconds"] <= result["wall_seconds"],
                 "EVENT_TIME", f"event {index} elapsed time")
        previous = event["elapsed_seconds"]
        observed = {key: value for key, value in event.items() if key != "elapsed_seconds"}
        required = {key: value for key, value in contract.items() if key != "elapsed_seconds"}
        _require(_same(observed, required), "EVENTS_CONTRACT", f"event {index} operation, stage, nesting or metadata differs")
        if "elapsed_seconds" in contract:
            _require(_same(event["elapsed_seconds"], contract["elapsed_seconds"]),
                     "EVENTS_CONTRACT", f"event {index} resource sample timestamp differs")


def _verify_source(repo, code_sha, relative, expected_hash, bound):
    _require(type(expected_hash) is str and re.fullmatch(r"[0-9a-f]{64}", expected_hash),
             "SOURCE_IDENTITY", relative)
    historical = subprocess.run(["git", "show", f"{code_sha}:{relative}"], cwd=repo,
                                capture_output=True, check=False, timeout=15)
    _require(historical.returncode == 0 and hashlib.sha256(historical.stdout).hexdigest() == expected_hash,
             "SOURCE_HISTORY_DRIFT", relative)
    current = repo / relative
    _require(_digest(current) == expected_hash, "SOURCE_CURRENT_DRIFT", relative)
    bound[str(current.resolve())] = {"sha256": expected_hash, "bytes": current.stat().st_size}


def _verify_committed_helper(repo, code_sha, relative, bound):
    # E0's clean source commit binds imported helpers although its source receipt
    # predates individual helper hashes. Read the committed blob, never import it.
    historical = subprocess.run(["git", "show", f"{code_sha}:{relative}"], cwd=repo,
                                capture_output=True, check=False, timeout=15)
    _require(historical.returncode == 0, "SOURCE_HISTORY_DRIFT", relative)
    digest = hashlib.sha256(historical.stdout).hexdigest()
    current = repo / relative
    _require(_digest(current) == digest, "SOURCE_CURRENT_DRIFT", relative)
    bound[str(current.resolve())] = {"sha256": digest, "bytes": current.stat().st_size}


def _check_inventory(model, trainable, primary, inventory):
    _require(model.get("weight_sha256") == WEIGHT_SHA256 and model.get("primary_receipt_sha256") == PRIMARY_SHA256
             and type(model.get("unique_parameters")) is int and model["unique_parameters"] == 596049920
             and model.get("tied") is True, "MODEL_IDENTITY", "wrong primary model, count or embedding tie")
    _require(model.get("loader") == "stock Qwen3 allocated BF16 on CUDA; strict streamed safetensors copy",
             "MODEL_LOADER", "unqualified model loader")
    for name in ("architectures", "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
                 "num_key_value_heads", "intermediate_size", "head_dim", "rope_theta", "rms_norm_eps",
                 "vocab_size", "tie_word_embeddings", "attention_bias", "attention_dropout", "rope_scaling",
                 "use_sliding_window", "max_position_embeddings", "hidden_act"):
        _require(_same(model.get("config", {}).get(name), primary["architecture"].get(name)),
                 "MODEL_CONFIG_DRIFT", name)
    tensor_hashes = model.get("tensor_sha256")
    _require(type(tensor_hashes) is dict and set(tensor_hashes) == set(inventory["tensors"])
             and all(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) for value in tensor_hashes.values()),
             "TENSOR_INVENTORY", "every serialized base tensor must be bound")
    modules = {f"model.layers.{i}.self_attn.{projection}_proj" for i in range(28) for projection in ("q", "v")}
    _require(type(trainable.get("modules")) is list and len(trainable["modules"]) == 56
             and set(trainable["modules"]) == modules, "LORA_SCOPE", "rank-8 q/v modules differ")
    parameters = {}
    for module in modules:
        parameters[module + ".lora_A"] = {"shape": [8, 1024], "dtype": "torch.float32"}
        parameters[module + ".lora_B"] = {"shape": [2048 if module.endswith("q_proj") else 1024, 8], "dtype": "torch.float32"}
    _require(_same(trainable.get("parameters"), parameters) and type(trainable.get("count")) is int
             and trainable["count"] == 1146880, "LORA_SCOPE", "adapter shapes/dtypes/count differ")


def _check_weight_tensor_bytes(path, inventory, model):
    """Validate safetensors headers/digests with bounded reads, without model libraries."""
    with path.open("rb") as stream:
        header_prefix = stream.read(8)
        _require(len(header_prefix) == 8, "WEIGHT_HEADER", "missing safetensors header")
        size = struct.unpack("<Q", header_prefix)[0]
        _require(size == inventory["header_bytes"] and size <= 1024 * 1024,
                 "WEIGHT_HEADER", "unregistered safetensors header length")
        header = _parse(stream.read(size))
        _require(set(header) - {"__metadata__"} == set(inventory["tensors"]), "WEIGHT_HEADER", "tensor names differ")
        for name, entry in inventory["tensors"].items():
            _require(_same(header[name], {field: entry[field] for field in ("dtype", "shape", "data_offsets")}),
                     "WEIGHT_HEADER", name)
            left, right = entry["data_offsets"]
            stream.seek(8 + size + left)
            remaining, digest = right - left, hashlib.sha256()
            while remaining:
                block = stream.read(min(4 * 1024 * 1024, remaining))
                _require(bool(block), "WEIGHT_TRUNCATED", name)
                digest.update(block)
                remaining -= len(block)
            _require(digest.hexdigest() == model["tensor_sha256"][name], "LOADED_TENSOR_DIGEST", name)


def inspect_e0(experiment_root, *, e0_directory=None):
    """Read and validate a claimed native bundle; return a blocked diagnostic on failure."""
    root = Path(experiment_root).resolve()
    bundle = Path(e0_directory).resolve() if e0_directory is not None else root / "results/e0_titan_v2"
    bound = {}
    report = {"status": "BLOCKED", "reasons": [], "bound_files": bound,
              "native_launch_authorized": False, "no_native_executor": True}
    try:
        result = _read(bundle / "E0_RESULT.json", bound)
        if result.get("status") != PASS_STATUS:
            report["reasons"].append({"code": "E0_NOT_PASSED", "detail": result.get("status", "MISSING_STATUS")})
            if result.get("error"):
                report["reasons"].append({"code": "E0_RECORDED_FAILURE", "detail": str(result["error"])[:300]})
            return report
        frozen_path = root / "configs/E0_TITAN_BF16.json"
        frozen = _read(frozen_path, bound)
        _require(bound[str(frozen_path)]["sha256"] == CONFIG_SHA256, "E0_CONFIG_IDENTITY", "frozen E0 config bytes changed")
        cfg = _read(bundle / "CONFIG.json", bound)
        _require(_same(cfg, frozen), "E0_CONFIG_DRIFT", "bundle config differs from registered config")
        validate_result(result, cfg)
        for name in ("FAILURE_RECEIPT.json",):
            _require(not (bundle / name).exists(), "E0_FAILURE_PRESENT", name)
        primary_path = root / "receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json"
        primary = _read(primary_path, bound)
        inventory_path = root / "receipts/titan_static_v1/PARAMETER_INVENTORY.json"
        inventory = _read(inventory_path, bound)
        _require(bound[str(primary_path)]["sha256"] == PRIMARY_SHA256
                 and bound[str(inventory_path)]["sha256"] == INVENTORY_SHA256,
                 "STATIC_IDENTITY", "pinned static receipt/inventory drift")
        _require(cfg["weight_sha256"] == primary["weight_sha256"] == WEIGHT_SHA256
                 and cfg["primary_receipt_sha256"] == PRIMARY_SHA256,
                 "MODEL_IDENTITY", "primary checkpoint contract differs")
        source = _read(bundle / "SOURCE_RECEIPT.json", bound)
        _require(source.get("code_sha") == result["code_sha"] and source.get("dirty") is False,
                 "SOURCE_IDENTITY", "native source commit or cleanliness differs")
        repo = root.parents[1]
        _verify_source(repo, result["code_sha"], QUALIFIER, source.get("script_sha256"), bound)
        for relative in NATIVE_HELPERS:
            _verify_committed_helper(repo, result["code_sha"], relative, bound)
        runtime = _read(bundle / "RUNTIME_RECEIPT.json", bound)
        _require(all(runtime.get("packages", {}).get(name) == value for name, value in PACKAGES.items())
                 and runtime.get("attention") == "eager" and runtime.get("base_dtype") == "bfloat16"
                 and runtime.get("adapter_dtype") == "float32" and runtime.get("cuda") == "12.6"
                 and type(runtime.get("total_vram_bytes")) is int and runtime["total_vram_bytes"] > 0
                 and all(_nonblank(runtime.get(name)) for name in ("gpu", "executable", "python", "platform")),
                 "RUNTIME_IDENTITY", "native backend/package/device receipt differs")
        tokenizer = _read(bundle / "TOKENIZER_RECEIPT.json", bound)
        for name, value in {"class_name": "Qwen2TokenizerFast", "template_sha256": TEMPLATE_SHA256,
                            "eos_id": 151645, "original_pad_id": 151643, "raw_roundtrip": True,
                            "enable_thinking": False, "add_generation_prompt": True}.items():
            _require(_same(tokenizer.get(name), value), "TOKENIZER_IDENTITY", name)
        _require(type(tokenizer.get("raw_ids")) is list and bool(tokenizer["raw_ids"])
                 and all(type(t) is int and 0 <= t < 151936 for t in tokenizer["raw_ids"]),
                 "TOKENIZER_IDENTITY", "raw tokenizer fixture IDs missing")
        model = _read(bundle / "MODEL_RECEIPT.json", bound)
        trainable = _read(bundle / "TRAINABLE_INVENTORY.json", bound)
        _check_inventory(model, trainable, primary, inventory)
        resources = _read(bundle / "RESOURCE_BASELINE.json", bound)
        _require(resources.get("status") == PASS_STATUS and _same(resources.get("samples"), result["resources"])
                 and _same(resources.get("profiles"), result["resource_profiles"]),
                 "RESOURCE_RECEIPT_DRIFT", "resource receipt differs from result")
        cost = _read(bundle / "COST_LEDGER.json", bound)
        events_path = bundle / "EVENTS.jsonl"
        event_raw = events_path.read_bytes()
        bound[str(events_path)] = {"sha256": hashlib.sha256(event_raw).hexdigest(), "bytes": len(event_raw)}
        lines = event_raw.splitlines()
        _require(bool(lines) and all(line.strip() for line in lines), "EVENTS_CONTRACT", "empty journal or blank journal record")
        events = [_parse(line) for line in lines]
        validate_accounting_bundle(result, cost, events, cfg, sorted(inventory["tensors"]))
        supervision = bundle.with_name(bundle.name + "_supervisor")
        supervisor = _read(supervision / "SUPERVISOR.json", bound)
        _require(supervisor.get("status") == "PASS_PROCESS_ONLY" and supervisor.get("reason") is None
                 and type(supervisor.get("exit_code")) is int and supervisor["exit_code"] == 0
                 and supervisor.get("native_qualification") is False
                 and _number(supervisor.get("elapsed_seconds"), minimum=result["wall_seconds"], maximum=cfg["wall_seconds"])
                 and supervisor.get("limit_seconds") == cfg["wall_seconds"]
                 and supervisor.get("memory_poll_seconds") == .1
                 and _number(supervisor.get("peak_observed_process_tree_rss_bytes"), minimum=1, maximum=4 * 2**30),
                 "SUPERVISOR_CONTRACT", "native worker did not complete within containment")
        _require(not (supervision / "FAILURE_RECEIPT.json").exists(), "SUPERVISOR_CONTRACT", "supervisor failure receipt present")
        launch = _read(supervision / "LAUNCH.json", bound)
        go = _read(supervision / "GO.json", bound)
        _require(launch.get("config_sha256") == go.get("config_sha256") == CONFIG_SHA256
                 and launch.get("source_sha") == result["code_sha"], "LAUNCH_IDENTITY", "launch config/source mismatch")
        _verify_source(repo, result["code_sha"], SUPERVISOR, launch.get("supervisor_sha256"), bound)
        _verify_source(repo, result["code_sha"], WATCHDOG, launch.get("watchdog_sha256"), bound)
        command = launch.get("command")
        _require(type(command) is list and len(command) == 10 and command[:4] == [runtime["executable"], "-B", "-m",
                 "experiments.post_training_mimo_centralized_v1.scripts.qualify_native"]
                 and command[4] == "--config" and command[6] == "--output" and command[8] == "--go-path"
                 and Path(command[5]).resolve() == frozen_path and Path(command[7]).name == bundle.name
                 and Path(command[9]).name == "GO.json"
                 and Path(command[9]).parent.name == supervision.name,
                 "LAUNCH_IDENTITY", "unexpected native worker command")
        model_dir = Path(primary["checkpoint_path"]).resolve()
        _require(Path(cfg["model_dir"]).resolve() == model_dir, "MODEL_PATH", "checkpoint path differs")
        for name, expected_hash in {**cfg["asset_sha256"], "model.safetensors": WEIGHT_SHA256}.items():
            path = model_dir / name
            _require(primary["files"][name]["sha256"] == expected_hash and _digest(path) == expected_hash,
                     "MODEL_ASSET_DRIFT", name)
            bound[str(path)] = {"sha256": expected_hash, "bytes": path.stat().st_size}
        _check_weight_tensor_bytes(model_dir / "model.safetensors", inventory, model)
        report["status"] = "PASS_E0_EVIDENCE_CONSISTENCY_ONLY"
        report["qualified_contexts"] = [256, 512, 1024]
        report["qualified_sampler"] = "full_prefix_no_cache_temperature1_topk0_topp1"
    except GateError as error:
        report["reasons"].append({"code": error.code, "detail": str(error)[:400]})
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError, OverflowError, subprocess.SubprocessError) as error:
        report["reasons"].append({"code": "E0_ARTIFACT_INVALID_OR_MISSING", "detail": f"{type(error).__name__}: {error}"[:400]})
    return report


def preflight(experiment_root, *, e0_directory=None):
    """Diagnostic gate report. No caller-provided downstream status can grant entry."""
    gates = {"e0_native": inspect_e0(experiment_root, e0_directory=e0_directory)}
    for name, detail in {
        "source_enrollment": "No implemented verifier for complete unchanged-source native enrollment and fixed DEV membership.",
        "fresh_language_grading": "Fresh language DEV/CAL, grader identity/calibration and native scoring remain unqualified.",
        "long_context_runtime": "The separately registered zero-update 2048/4096 runtime extension has no validated native receipt.",
    }.items():
        gates[name] = {"status": "BLOCKED", "reasons": [{"code": "SCOPED_NATIVE_EVIDENCE_VALIDATOR_UNAVAILABLE", "detail": detail}]}
    return {"schema_version": 1, "status": "BLOCKED_E1_DIAGNOSTIC_ONLY", "gates": gates,
            "e1_launch_ready": False, "native_launch_authorized": False, "no_native_executor": True,
            "training_updates": 0, "model_loads": 0, "teacher_calls": 0,
            "held_out_access": {"CHECK": False, "FINAL": False},
            "claim": "Artifact diagnostics only; no model efficacy, current free-memory or native execution qualification claim."}
