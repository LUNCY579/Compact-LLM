"""Independent stdlib evidence checks for the frozen SHORT_GREEDY qualifier.

No producer, tokenizer or native framework is imported or executed. Value-only
validation is not a qualifying native bundle. inspect_capture_scope requires
complete registered results, trace, source/runtime/containment bindings and then
streamed model-asset byte checks. Even consistency PASS grants no native launch,
DEV collection, language grading, E1 training or broad preservation authority.

Text checks bind producer-returned strings to the pinned tokenizer/runtime/code;
they do not independently detokenize IDs. No tokenizer replay occurs here. Failed
or incomplete runs stop before model assets are read, and unattempted fixtures
are never synthesized into response records.
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


CONFIG_RELATIVE = "configs/CAPTURE_GREEDY_SHORT_QUAL_V1.json"
CONFIG_SHA256 = "cab513b8006a8f4212be2e3f8fe1d9862abca96a53ab58ab033165e395a82b27"
PACKAGE = "experiments/post_training_mimo_centralized_v1/"
CORE_CONFIG_SOURCE = PACKAGE + "configs/E0_LAPTOP_STREAM_UNPADDED_CE512.json"
CORE_CONFIG_SHA256 = "42a32cfa2fe4dc1250a288bacec2a60d150c5498d8dced9bba23b0c5037f5929"
CORE_REFERENCE = "receipts/LAPTOP_NATIVE_STAGE_005_RESULTS.json"
CORE_REFERENCE_SHA256 = "396da65c8febbb631de16ef5ad5048d68c5a2f11bc7ddf4b3b7e0f964c65d9f9"
PASS_STATUS = "PASS_FROZEN_GREEDY_SHORT_CAPTURE_SCOPED"
CONSISTENCY_STATUS = "PASS_SHORT_GREEDY_CAPTURE_EVIDENCE_CONSISTENCY_ONLY"
RUNTIME_SCOPE = "FROZEN_GREEDY_PREFIX_CE512_CAPTURE_V1"
RESOURCE_PROFILE = "LAPTOP_CAPTURE_RESIDENT_V1"
LOADER_MODE = "stream_read_2MiB_v1"
LOADER_LABEL = "stock Qwen3 allocated BF16 on CUDA; strict safetensors header and 2MiB bounded read buffer copy"
EXECUTABLE = "C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe"
PACKAGES = {**low.PACKAGES, "psutil": "7.2.2"}
EOS_IDS = (151645, 151643)
VOCAB_SIZE = 151936
PROMPT_LENGTHS = (128, 384)
PROBE_LENGTHS = (256, 512)
OPERATIONS = ("model_construct", "checkpoint_load", "tensor_copy", "forward", "backward",
              "optimizer_step", "checkpoint_save", "checkpoint_reload")
METADATA_LIMIT = 2 * 1024**2
EVENT_LIMIT = 1024**2
LOG_LIMIT = 16 * 1024**2
RESOURCE_LIMITS = {"host_free_gib": 1.25, "worker_free_gib": 1.25, "worker_rss_limit_gib": 2.0,
                   "gpu_free_gib": 4.0, "gpu_worker_free_gib": .5, "disk_free_gib": 4.0,
                   "wall_seconds": 600, "admission_budget": "system_reserve_plus_remaining_worker_rss"}
NEW_SOURCE_FILES = tuple(PACKAGE + "scripts/" + name + ".py" for name in
                         ("greedy_capture", "qualify_capture", "supervise_capture"))
CPU_TEST_SOURCE = PACKAGE + "tests/test_greedy_capture.py"
# Independently frozen after producer review; a self-declared source receipt is
# insufficient to admit a different implementation under the same config.
PRODUCER_PINS = {
    NEW_SOURCE_FILES[0]: {"sha256": "8cc3d0a4e9e4d9d18b5ef9c7f62142f83f0fb84f1779bed389a7c5d99f8c3492", "bytes": 13140},
    NEW_SOURCE_FILES[1]: {"sha256": "53867509693361d92097b4cd7d19aa90cbd91470a45b5d0acd001e3a0cacd3fc", "bytes": 22170},
    NEW_SOURCE_FILES[2]: {"sha256": "c3b479db45335444a5621030a6d9cd1a3d9a12b73c571264725e2f12e5a7a8b8", "bytes": 3299},
    CPU_TEST_SOURCE: {"sha256": "97f5c6936e6a9ebd78f9836a29a0322b741ee897ff27b9460afb77ca2ece3e85", "bytes": 11933},
}
RAW_TOKEN_IDS = [31029, 2404, 25, 220, 22, 488, 220, 20, 284, 220, 16, 17, 13]
UNQUALIFIED = ["DEV/source enrollment", "long context beyond512", "adapter capture", "E1/training",
               "human language grading", "stochastic/probability-based RL", "published-default generation"]
COST_FIELDS = ("wall_seconds", "pretrained_loads", "optimizer_steps", "training_tokens", "generation_tokens",
               "teacher_calls", "accounting_version", "call_accounting")
TRUE_CHECKS = {"exact_native_load", "bare_frozen_eval", "fixture_input_boundaries", "greedy_selected_raw_argmax",
               "eos_stop_accounting", "response_rendering", "frozen_base_unchanged", "source_assets_unchanged"}
ARGMAX_FIELDS = {"selected_raw_logit", "raw_max_logit", "first_maximum_token_id", "raw_argmax_verified"}
FIXTURE_FIELDS = {"fixture_id", "prompt_tokens", "generation_cap", "prompt_ids", "prompt_token_ids_sha256",
                  "generated_ids", "prefix_token_ids_sha256", "raw_argmax_evidence", "online_model_logprobs",
                  "offline_model_logprobs", "behavior_logprobs", "stop_reason", "terminal_eos_id", "visible_ids",
                  "raw_response", "visible_response", "raw_response_sha256", "visible_response_sha256",
                  "generated_token_ids_sha256", "visible_token_ids_sha256", "offline_input_token_ids_sha256",
                  "offline_logits_to_keep", "offline_prediction_indices", "likelihood", "generation_seconds",
                  "generation_tokens_per_second", "generation_timing", "offline_teacher_seconds", "offline_timing"}
GateError = low.GateError
_require, _same, _number = low._require, low._same, low._number


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _keys(value, fields, code):
    _require(type(value) is dict and set(value) == set(fields), code, "missing or extra fields")


def _hash(value, code):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, code, "SHA256 required")


def _ids(value, code, *, count=None, nonempty=True):
    _require(type(value) is list and (bool(value) or not nonempty) and (count is None or len(value) == count)
             and all(type(token) is int and 0 <= token < VOCAB_SIZE for token in value),
             code, "invalid token array")


def _text_identity(text, observed_hash, code):
    _require(type(text) is str and hashlib.sha256(text.encode("utf-8")).hexdigest() == observed_hash,
             code, "producer-returned text/hash mismatch")


def _read(path, bound):
    path = Path(path).absolute()
    _require(path.resolve() == path, "REDIRECTED_EVIDENCE_PATH", str(path))
    _require(path.stat().st_size <= METADATA_LIMIT, "METADATA_SIZE_LIMIT", str(path))
    return low._read(path, bound)


def _bind_binary(path, bound, ceiling):
    path = Path(path).absolute()
    _require(path.resolve() == path, "REDIRECTED_EVIDENCE_PATH", str(path))
    size = path.stat().st_size
    _require(size <= ceiling, "BINARY_SIZE_LIMIT", str(path))
    bound[str(path)] = {"sha256": low._digest(path), "bytes": size}


def successful_accounting(lengths):
    """Actual workload for two completed rollouts; maximums are never substituted."""
    _require(type(lengths) in (list, tuple) and len(lengths) == 2
             and all(type(n) is int and 1 <= n <= 128 for n in lengths),
             "ROLLOUT_LENGTH", "two observed lengths in [1,128] required")
    left, right = lengths
    generated = left + right
    positions = 1280 + 129 * left + 385 * right + left * (left - 1) // 2 + right * (right - 1) // 2
    counts = {"model_construct": 1, "checkpoint_load": 1, "tensor_copy": 310, "forward": 4 + generated,
              "backward": 0, "optimizer_step": 0, "checkpoint_save": 0, "checkpoint_reload": 0}
    return {"accounting_version": 2, "max_optimizer_calls": 0, "last_call_id": 316 + generated,
            "operations": {name: {"attempted": n, "returned": n, "failed": 0, "pending": 0}
                           for name, n in counts.items()}, "pending_calls": [],
            "forward_input_positions_attempted": positions, "forward_input_positions_returned": positions,
            "training_forward_input_positions_attempted": 0, "training_forward_input_positions_returned": 0}


def _likelihood(online, offline, reported):
    _require(type(online) is list and type(offline) is list and len(online) == len(offline) and bool(online)
             and all(type(value) in (int, float) and math.isfinite(value) and value <= 0
                     for value in online + offline), "RAW_MODEL_LOGPROBS", "finite raw model log-probabilities required")
    errors = [abs(left - right) for left, right in zip(online, offline)]
    maximum, mean = max(errors), sum(errors) / len(errors)
    _keys(reported, {"finite", "absolute_errors", "max_abs", "mean_abs", "passed"}, "LIKELIHOOD_FIELDS")
    _require(_same(reported["absolute_errors"], errors) and _same(reported["max_abs"], maximum)
             and _same(reported["mean_abs"], mean), "LIKELIHOOD_RECONSTRUCTION", "reported errors differ from scalar arrays")
    _require(reported["finite"] is True and reported["passed"] is True and maximum <= .03125 and mean <= .01,
             "LIKELIHOOD_TOLERANCE", "frozen raw-model likelihood threshold failed")
    return {"max_abs": maximum, "mean_abs": mean}


def validate_cpu_controls(receipt):
    """Check the eight fixed synthetic branch controls independently of producer code."""
    _keys(receipt, {"status", "cases", "native_EOS_observations", "model_calls", "training_authorized", "E1_authorized",
                    "helper_sha256", "test_source_sha256"},
          "CPU_CONTROL_FIELDS")
    _hash(receipt["helper_sha256"], "CPU_SOURCE_HASH")
    _hash(receipt["test_source_sha256"], "CPU_SOURCE_HASH")
    _require(receipt["status"] == "PASS_CPU_CAPTURE_MECHANICS_ONLY"
             and _same(receipt["native_EOS_observations"], []) and _same(receipt["model_calls"], 0)
             and receipt["training_authorized"] is False and receipt["E1_authorized"] is False,
             "CPU_CONTROL_SCOPE", "synthetic controls cannot claim native observation or authority")
    expected = [("raw_greedy_tie_lowest_id", {"scores": [-2.0, 4.0, 4.0]}, 1)]
    for token, marker in ((151645, "<|im_end|>"), (151643, "<|endoftext|>")):
        expected.append(("terminal_only_render_" + str(token),
                         {"prompt_ids": [token, 10], "generated_ids": [7, 8, 9, token], "cap": 128,
                          "eos_ids": [151645, 151643]},
                         {"stop": "eos", "visible_ids": [7, 8, 9], "raw": " \n<other-special> tail " + marker,
                          "visible": " \n<other-special> tail "}))
    expected += [
        ("prompt_EOS_does_not_stop_new_nonEOS", {"prompt_ids": [151645], "generated_ids": [7], "cap": 128}, None),
        ("EOS_at_cap_has_EOS_priority", {"generated_ids": [7] * 127 + [151643], "cap": 128}, "eos"),
        ("exact_cap_stop_without_EOS", {"generated_ids": [7] * 128, "cap": 128}, "length"),
        ("offline_Nplus1_firstN_shift", {"generated_length": 2, "kept_logits_rows": 3}, [0, 1]),
        ("verified_greedy_behavior_zero_is_not_model_logprob",
         {"token_id": 1, "selected": 4.0, "maximum": 4.0, "first_maximum": 1, "model_logprob": -.5}, 0.0),
    ]
    cases = receipt["cases"]
    _require(type(cases) is list and len(cases) == len(expected), "CPU_CONTROL_COVERAGE", "all eight fixed controls required")
    fields = {"case_id", "designation", "input", "output", "expected", "passed", "input_sha256", "output_sha256"}
    for case, (name, inputs, output) in zip(cases, expected):
        _keys(case, fields, "CPU_CONTROL_CASE_FIELDS")
        _require(case["case_id"] == name and case["designation"] == "SYNTHETIC_MECHANICS_ONLY_NOT_NATIVE_OR_FORCED_MODEL_EOS"
                 and case["passed"] is True and _same(case["input"], inputs)
                 and _same(case["expected"], output) and _same(case["output"], output)
                 and case["input_sha256"] == digest(inputs) and case["output_sha256"] == digest(output),
                 "CPU_CONTROL_CASE", name)


def _fixture(fixture, prompt_length, wall_seconds):
    _keys(fixture, FIXTURE_FIELDS, "FIXTURE_FIELDS")
    _require(fixture["fixture_id"] == f"fixture_{prompt_length}"
             and _same(fixture["prompt_tokens"], prompt_length) and _same(fixture["generation_cap"], 128),
             "FIXTURE_IDENTITY", "registered fixture order/length/cap differs")
    prompt, generated = fixture["prompt_ids"], fixture["generated_ids"]
    _ids(prompt, "PROMPT_IDS", count=prompt_length)
    _ids(generated, "GENERATED_IDS")
    n = len(generated)
    _require(1 <= n <= 128 and prompt_length + n <= 512, "ROLLOUT_LENGTH", "completion exceeds frozen scope")
    _require(fixture["prompt_token_ids_sha256"] == digest(prompt)
             and fixture["generated_token_ids_sha256"] == digest(generated), "TOKEN_HASH", "prompt/generated hash differs")
    _require(not any(token in EOS_IDS for token in generated[:-1]), "EOS_ORDER", "tokens occur after first generated EOS")
    eos = generated[-1] if generated[-1] in EOS_IDS else None
    stop = "eos" if eos is not None else "length"
    _require(stop == "eos" or n == 128, "EARLY_STOP", "non-EOS completion must reach the cap")
    _require(fixture["stop_reason"] == stop and _same(fixture["terminal_eos_id"], eos), "STOP_REASON", "EOS takes precedence at cap")
    visible = generated[:-1] if eos is not None else generated
    _ids(fixture["visible_ids"], "VISIBLE_IDS", nonempty=False)
    _require(_same(fixture["visible_ids"], visible) and fixture["visible_token_ids_sha256"] == digest(visible),
             "TERMINAL_EXTRACTION", "only one terminal EOS ID may be removed")
    _text_identity(fixture["raw_response"], fixture["raw_response_sha256"], "RAW_TEXT_HASH")
    _text_identity(fixture["visible_response"], fixture["visible_response_sha256"], "VISIBLE_TEXT_HASH")
    if eos is None:
        _require(fixture["raw_response"] == fixture["visible_response"], "TERMINAL_EXTRACTION", "identical ID streams have different decoded text")
    if not visible:
        _require(fixture["visible_response"] == "", "TERMINAL_EXTRACTION", "empty visible ID stream must decode empty")
    _require(type(fixture["prefix_token_ids_sha256"]) is list
             and _same(fixture["prefix_token_ids_sha256"], [digest(prompt + generated[:i]) for i in range(n)]),
             "PREFIX_TOKEN_HASH", "greedy forward prefix differs from prompt plus earlier emissions")
    _require(_same(fixture["offline_logits_to_keep"], n + 1)
             and _same(fixture["offline_prediction_indices"], list(range(n)))
             and fixture["offline_input_token_ids_sha256"] == digest(prompt + generated),
             "OFFLINE_CAUSAL_SHIFT", "full prefix / last N+1 / first N alignment differs")
    for name in ("online_model_logprobs", "offline_model_logprobs", "behavior_logprobs", "raw_argmax_evidence"):
        _require(type(fixture[name]) is list and len(fixture[name]) == n, "FIXTURE_VECTOR_LENGTH", name)
    for token, argmax, behavior in zip(generated, fixture["raw_argmax_evidence"], fixture["behavior_logprobs"]):
        _keys(argmax, ARGMAX_FIELDS, "ARGMAX_FIELDS")
        _require(all(type(argmax[name]) in (int, float) and math.isfinite(argmax[name])
                     for name in ("selected_raw_logit", "raw_max_logit")), "ARGMAX_FINITE", "finite raw logits required")
        _require(argmax["selected_raw_logit"] == argmax["raw_max_logit"]
                 and type(argmax["first_maximum_token_id"]) is int and argmax["first_maximum_token_id"] == token
                 and argmax["raw_argmax_verified"] is True, "GREEDY_ARGMAX", "chosen token/raw maximum/first maximum disagree")
        _require(type(behavior) in (int, float) and math.isfinite(behavior) and behavior == 0,
                 "GREEDY_BEHAVIOR_LOGPROB", "point-mass behavior logprob must be zero, separately from raw model logprob")
    _likelihood(fixture["online_model_logprobs"], fixture["offline_model_logprobs"], fixture["likelihood"])
    for field in ("generation_seconds", "offline_teacher_seconds"):
        _require(_number(fixture[field], minimum=.000001, maximum=wall_seconds), "FIXTURE_TIMING", field)
    _require(_number(fixture["generation_tokens_per_second"], minimum=.000001)
             and math.isclose(fixture["generation_tokens_per_second"], n / fixture["generation_seconds"], rel_tol=1e-9)
             and fixture["generation_timing"] == "instrumented generation wall including durable token logging and resource guards"
             and fixture["offline_timing"] == "instrumented teacher-forcing wall including selected model logprobs and resource guard",
             "FIXTURE_TIMING", "instrumented generation wall throughput differs")
    return n


def resource_stages(lengths):
    stages = ["preload", "loaded", "finite_prefix_probe_256", "finite_prefix_probe_512"]
    for prompt, n in zip(PROMPT_LENGTHS, lengths):
        stages.extend(f"greedy_{prompt}_{i}" for i in range(n))
        stages.append(f"teacher_force_{prompt}")
    return stages + ["complete"]


def validate_result(result):
    """Validate synthetic or observed values only; never grant bundle qualification."""
    _require(type(result) is dict and result.get("status") == PASS_STATUS, "CAPTURE_NOT_PASSED", "complete scoped pass required")
    low._finite(result)
    fixed = {"schema_version": 1, "experiment_id": "RINEYA-0P6B-CAPTURE-QUAL-SHORT-V1", "status": PASS_STATUS,
             "runtime_scope": RUNTIME_SCOPE, "resource_profile": RESOURCE_PROFILE, "config_sha256": CONFIG_SHA256,
             "dirty_at_launch": False, "evidence_level": "MEASURED / REPRODUCED", "seed": 20260928,
             "qualified_CE512_reference_sha256": CORE_REFERENCE_SHA256, "pretrained_loads": 1,
             "optimizer_steps": 0, "training_tokens": 0, "teacher_calls": 0, "accounting_version": 2,
             "native_evaluation_bank_access": False, "language_grading_authorized": False,
             "E1_authorized": False, "CHECK_FINAL_access": False, "unqualified": UNQUALIFIED}
    variable = {"code_sha", "generation_tokens", "checks", "resources", "fixtures", "call_accounting",
                "load_seconds", "loader_diagnostics", "actual_counts", "wall_seconds"}
    _keys(result, set(fixed) | variable, "CAPTURE_RESULT_FIELDS")
    for field, value in fixed.items():
        _require(_same(result[field], value), "CAPTURE_RESULT_CONTRACT", field)
    _require(type(result["code_sha"]) is str and re.fullmatch(r"[0-9a-f]{40}", result["code_sha"]) is not None,
             "SOURCE_IDENTITY", "full producer commit required")
    _require(_number(result["wall_seconds"], minimum=.000001, maximum=600)
             and _number(result["load_seconds"], minimum=.000001, maximum=result["wall_seconds"]),
             "TIMING_MEASUREMENT", "invalid worker/load duration")
    fixtures = result["fixtures"]
    _require(type(fixtures) is list and len(fixtures) == 2, "FIXTURE_COVERAGE", "both completed fixed fixtures required")
    lengths = [_fixture(fixture, prompt, result["wall_seconds"]) for fixture, prompt in zip(fixtures, PROMPT_LENGTHS)]
    total = sum(lengths)
    accounting = successful_accounting(lengths)
    _require(_same(result["call_accounting"], accounting) and _same(result["generation_tokens"], total),
             "CALL_ACCOUNTING_CONTRACT", "actual counts, returns and completed records differ")
    expected = {"generated_tokens": total, "forward_calls": 4 + total, "logical_calls": 316 + total,
                "forward_input_positions": accounting["forward_input_positions_returned"],
                "resource_samples": 7 + total, "journal_records": 645 + 4 * total}
    _require(_same(result["actual_counts"], expected), "ACTUAL_COUNT_RECONSTRUCTION", "maximum budgets cannot replace actual work")
    checks = result["checks"]
    _keys(checks, TRUE_CHECKS | {"finite_prefix_probes", "causal_shift_model_logprobs", "native_eos_coverage"}, "CAPTURE_CHECK_FIELDS")
    for name in TRUE_CHECKS:
        _require(checks[name] is True, "CAPTURE_CHECK_FAILED", name)
    _require(_same(checks["finite_prefix_probes"], {"contexts": [256, 512], "logits_to_keep": 1, "finite": True}),
             "PREFIX_PROBE_SCOPE", "probes check last-position vocabulary, not all-position logits")
    errors = [value for fixture in fixtures for value in fixture["likelihood"]["absolute_errors"]]
    _require(_same(checks["causal_shift_model_logprobs"], {"fixtures": 2, "finite": True,
                    "max_abs": max(errors), "mean_abs": sum(errors) / len(errors)}),
             "LIKELIHOOD_AGGREGATE", "combined metrics differ from per-token errors")
    eos = sorted({fixture["terminal_eos_id"] for fixture in fixtures if fixture["terminal_eos_id"] is not None})
    _require(bool(eos) and _same(checks["native_eos_coverage"], {"observed": True, "terminal_eos_ids": eos}),
             "NATIVE_EOS_COVERAGE", "at least one natural native EOS required; CPU controls cannot substitute")
    samples = result["resources"]
    stages = resource_stages(lengths)
    _require(type(samples) is list and len(samples) == len(stages), "RESOURCE_COVERAGE", "missing or extra resource samples")
    fields = {"stage", "elapsed_seconds", "host_available_gib", "rss_gib", "gpu_free_gib", "gpu_allocated_gib",
              "gpu_reserved_gib", "peak_allocated_gib", "disk_free_gib", "admission_required_gib"}
    previous = -1
    for sample, stage in zip(samples, stages):
        _keys(sample, fields, "RESOURCE_FIELDS")
        _require(sample["stage"] == stage and all(_number(sample[field]) for field in fields - {"stage"}),
                 "RESOURCE_VALUES", stage)
        _require(previous <= sample["elapsed_seconds"] <= result["wall_seconds"], "RESOURCE_TIME", stage)
        previous = sample["elapsed_seconds"]
        admission = 1.25 + max(0.0, 2.0 - sample["rss_gib"])
        _require(_same(sample["admission_required_gib"], admission) and sample["rss_gib"] <= 2.0
                 and sample["host_available_gib"] >= (admission if stage == "preload" else 1.25)
                 and sample["gpu_free_gib"] >= (4.0 if stage == "preload" else .5)
                 and sample["disk_free_gib"] >= 4.0, "RESOURCE_FLOOR", stage)
    return lengths


def _expected_events(result, tensor_names):
    _require(type(tensor_names) in (list, tuple) and len(tensor_names) == 310 and len(set(tensor_names)) == 310
             and all(low._nonblank(name) for name in tensor_names), "JOURNAL_TENSOR_INVENTORY", "310 distinct base tensors required")
    events = [{"event": "runtime_import_start"}, {"event": "runtime_import_complete"}]
    samples, call_id = iter(result["resources"]), 0
    def resource(stage):
        sample = next(samples)
        _require(sample["stage"] == stage, "JOURNAL_RESOURCE_ORDER", stage)
        events.append({"event": "resource", **sample})
    def attempt(operation, stage, positions=0):
        nonlocal call_id
        call_id += 1
        record = {"accounting_version": 2, "call_id": call_id, "operation": operation,
                  "stage": stage, "input_positions": positions, "training": False}
        events.append({"event": "operation_attempt", **record})
        return record
    def returned(record):
        events.append({"event": "operation_returned", **record})
    def operation(kind, stage, positions=0):
        returned(attempt(kind, stage, positions))
    resource("preload")
    operation("model_construct", "model_construct")
    checkpoint = attempt("checkpoint_load", "checkpoint_load")
    for name in sorted(tensor_names):
        operation("tensor_copy", "tensor_copy:" + name)
    returned(checkpoint)
    resource("loaded")
    for p in PROBE_LENGTHS:
        operation("forward", f"finite_prefix_probe_{p}", p)
        resource(f"finite_prefix_probe_{p}")
    for fixture in result["fixtures"]:
        p, n = fixture["prompt_tokens"], len(fixture["generated_ids"])
        events.append({"event": "fixture_start", **{field: fixture[field] for field in
                       ("fixture_id", "prompt_tokens", "generation_cap", "prompt_token_ids_sha256")}})
        for ordinal, token in enumerate(fixture["generated_ids"]):
            operation("forward", f"greedy_{p}_{ordinal}", p + ordinal)
            events.append({"event": "generated_token", "fixture_id": fixture["fixture_id"], "prompt_tokens": p,
                           "ordinal": ordinal, "prefix_token_ids_sha256": fixture["prefix_token_ids_sha256"][ordinal],
                           "token_id": token, "model_logprob": fixture["online_model_logprobs"][ordinal],
                           "behavior_logprob": fixture["behavior_logprobs"][ordinal], **fixture["raw_argmax_evidence"][ordinal]})
            resource(f"greedy_{p}_{ordinal}")
        operation("forward", f"teacher_force_{p}", p + n)
        resource(f"teacher_force_{p}")
        events.append({"event": "fixture_complete", "generated_tokens": n, **{field: fixture[field] for field in
                       ("fixture_id", "stop_reason", "terminal_eos_id", "likelihood", "generated_token_ids_sha256",
                        "raw_response_sha256", "visible_response_sha256")}})
    resource("complete")
    _require(next(samples, None) is None and call_id == result["call_accounting"]["last_call_id"]
             and len(events) == result["actual_counts"]["journal_records"], "JOURNAL_WORKLOAD", "registered sequence differs")
    return events


def validate_accounting_bundle(result, cost, events, tensor_names):
    """Independent full journal reconstruction; no native producer logic is called."""
    validate_result(result)
    _require(type(cost) is dict and _same(cost, {name: result[name] for name in COST_FIELDS}),
             "COST_RECEIPT_DRIFT", "cost/result snapshots differ")
    expected = _expected_events(result, tensor_names)
    _require(type(events) is list and len(events) == len(expected), "EVENTS_CONTRACT", "missing or extra journal records")
    previous = -1
    for index, (event, contract) in enumerate(zip(events, expected)):
        _keys(event, set(contract) | {"elapsed_seconds"}, "EVENTS_CONTRACT")
        _require(_number(event["elapsed_seconds"]) and previous <= event["elapsed_seconds"] <= result["wall_seconds"],
                 "EVENT_TIME", str(index))
        previous = event["elapsed_seconds"]
        _require(_same({k: v for k, v in event.items() if k != "elapsed_seconds"},
                       {k: v for k, v in contract.items() if k != "elapsed_seconds"}),
                 "EVENTS_CONTRACT", f"record {index} differs in operation, prefix, token, outcome or order")
        if "elapsed_seconds" in contract:
            _require(_same(event["elapsed_seconds"], contract["elapsed_seconds"]), "EVENTS_CONTRACT", "resource timestamp differs")
    def elapsed(kind, operation=None, stage=None, fixture_id=None):
        return next(event["elapsed_seconds"] for event in events if event["event"] == kind
                    and (operation is None or event.get("operation") == operation)
                    and (stage is None or event.get("stage") == stage)
                    and (fixture_id is None or event.get("fixture_id") == fixture_id))
    minimum = elapsed("operation_returned", "checkpoint_load", "checkpoint_load") - elapsed("operation_attempt", "model_construct", "model_construct")
    maximum = elapsed("resource", stage="loaded") - elapsed("resource", stage="preload")
    _require(minimum - 1e-9 <= result["load_seconds"] <= maximum + 1e-9,
             "TIMING_JOURNAL_CONTRADICTION", "load duration outside trace bounds")
    before_generation = elapsed("resource", stage="finite_prefix_probe_512")
    for fixture in result["fixtures"]:
        p, n = fixture["prompt_tokens"], len(fixture["generated_ids"])
        final_greedy = elapsed("resource", stage=f"greedy_{p}_{n-1}")
        teacher_start = elapsed("operation_attempt", "forward", f"teacher_force_{p}")
        complete = elapsed("fixture_complete", fixture_id=fixture["fixture_id"])
        generated = final_greedy - elapsed("fixture_start", fixture_id=fixture["fixture_id"])
        teacher = elapsed("operation_returned", "forward", f"teacher_force_{p}") - teacher_start
        _require(generated - 1e-9 <= fixture["generation_seconds"] <= teacher_start - before_generation + 1e-9
                 and teacher - 1e-9 <= fixture["offline_teacher_seconds"] <= complete - final_greedy + 1e-9,
                 "TIMING_JOURNAL_CONTRADICTION", fixture["fixture_id"])
        before_generation = complete


def _loader_diagnostics(inventory):
    lengths = [entry["data_offsets"][1] - entry["data_offsets"][0] for entry in inventory["tensors"].values()]
    return {"loader_mode": LOADER_MODE, "header_bytes": inventory["header_bytes"], "file_bytes": inventory["file_bytes"],
            "payload_bytes": inventory["payload_bytes"], "serialized_tensors": 310, "read_buffer_bytes": 2097152,
            "maximum_payload_read_bytes": min(2097152, max(lengths)),
            "payload_chunk_count": sum((length + 2097151) // 2097152 for length in lengths),
            "memory_mapped": False, "cpu_model_allocated": False, "blocking_chunk_copies": True}


def _verify_assets(cfg, primary, inventory, model, bound):
    """Only the final evidence stage reads model assets; no tensor deserialization."""
    model_dir = Path(primary["checkpoint_path"]).resolve()
    _require(Path(cfg["model_dir"]).resolve() == model_dir, "MODEL_PATH", "registered model directory differs")
    for name, expected in {**cfg["asset_sha256"], "model.safetensors": low.WEIGHT_SHA256}.items():
        path = model_dir / name
        _require(path.resolve() == path, "REDIRECTED_MODEL_ASSET", name)
        _require(primary["files"][name]["sha256"] == expected and low._digest(path) == expected,
                 "MODEL_ASSET_DRIFT", name)
        bound[str(path)] = {"sha256": expected, "bytes": path.stat().st_size}
    low._check_weight_tensor_bytes(model_dir / "model.safetensors", inventory, model)


def _validate_model(model, primary, inventory, result):
    fixed = {"weight_sha256": low.WEIGHT_SHA256, "primary_receipt_sha256": low.PRIMARY_SHA256,
             "unique_parameters": 596049920, "serialized_tensors": 310, "tied": True,
             "trainable_parameters": 0, "adapter": None, "loader": LOADER_LABEL, "loader_mode": LOADER_MODE}
    _keys(model, set(fixed) | {"tensor_sha256", "loader_diagnostics", "config"}, "MODEL_FIELDS")
    for field, expected in fixed.items():
        _require(_same(model[field], expected), "MODEL_IDENTITY", field)
    _require(type(model["config"]) is dict, "MODEL_CONFIG", "config object required")
    for field in ("architectures", "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
                  "num_key_value_heads", "intermediate_size", "head_dim", "rope_theta", "rms_norm_eps", "vocab_size",
                  "tie_word_embeddings", "attention_bias", "attention_dropout", "rope_scaling", "use_sliding_window",
                  "max_position_embeddings", "hidden_act"):
        _require(_same(model["config"].get(field), primary["architecture"].get(field)), "MODEL_CONFIG_DRIFT", field)
    hashes = model["tensor_sha256"]
    _require(type(hashes) is dict and set(hashes) == set(inventory["tensors"]) and len(hashes) == 310,
             "TENSOR_INVENTORY", "all 310 serialized tensors required")
    for value in hashes.values():
        _hash(value, "TENSOR_INVENTORY")
    expected = _loader_diagnostics(inventory)
    _require(_same(model["loader_diagnostics"], expected) and _same(result["loader_diagnostics"], expected),
             "LOADER_DIAGNOSTICS", "exact fixed-buffer diagnostics required")


def _validate_runtime(runtime, cfg):
    fixed = {"packages": PACKAGES, "cuda": "12.6", "attention": "eager", "base_dtype": "bfloat16",
             "adapter": None, "use_cache": False, "inference_context": "torch.no_grad()", "torch_threads": 2,
             "allow_tf32": False, "deterministic_algorithms": True}
    _keys(runtime, set(fixed) | {"python", "executable", "platform", "gpu", "total_vram_bytes"}, "RUNTIME_FIELDS")
    for key, value in fixed.items():
        _require(_same(runtime[key], value), "RUNTIME_IDENTITY", key)
    _require(type(runtime["python"]) is str and runtime["python"].startswith("3.11.15 ")
             and type(runtime["executable"]) is str and Path(runtime["executable"]).resolve() == Path(EXECUTABLE).resolve()
             and Path(runtime["executable"]).resolve() == Path(cfg["runtime"]["executable"]).resolve()
             and type(runtime["platform"]) is str and runtime["platform"].startswith("Windows-")
             and low._nonblank(runtime["gpu"]) and type(runtime["total_vram_bytes"]) is int and runtime["total_vram_bytes"] > 0,
             "RUNTIME_IDENTITY", "pinned Python/Windows/device identity differs")


def _validate_tokenizer(tokenizer):
    fixed = {"class_name": "Qwen2TokenizerFast", "template_sha256": low.TEMPLATE_SHA256,
             "eos_id": 151645, "original_pad_id": 151643, "raw_ids": RAW_TOKEN_IDS, "raw_roundtrip": True,
             "enable_thinking": False, "add_generation_prompt": True,
             "skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    _require(_same(tokenizer, fixed), "TOKENIZER_IDENTITY", "registered tokenizer/rendering options differ")


def _validate_source(source, result, cfg, repo, bound):
    _keys(source, {"code_sha", "dirty", "source_files", "installed_runtime_source_files"}, "SOURCE_FIELDS")
    _require(source["code_sha"] == result["code_sha"] and source["dirty"] is False,
             "SOURCE_IDENTITY", "producer commit/cleanliness differs")
    expected = set(cfg["reused_source_files"]) | set(NEW_SOURCE_FILES) | {CPU_TEST_SOURCE, PACKAGE + CONFIG_RELATIVE}
    files = source["source_files"]
    _require(type(files) is dict and set(files) == expected, "SOURCE_CLOSURE", "exact producer/reused/test/config closure required")
    for relative, pin in files.items():
        _keys(pin, {"sha256", "bytes"}, "SOURCE_PIN_FIELDS")
        _hash(pin["sha256"], "SOURCE_PIN_HASH")
        _require(type(pin["bytes"]) is int and 0 < pin["bytes"] <= METADATA_LIMIT, "SOURCE_PIN_SIZE", relative)
        if relative in cfg["reused_source_files"]:
            _require(_same(pin, cfg["reused_source_files"][relative]), "REUSED_SOURCE_DRIFT", relative)
        if relative in PRODUCER_PINS:
            _require(_same(pin, PRODUCER_PINS[relative]), "REGISTERED_PRODUCER_DRIFT", relative)
        if relative == PACKAGE + CONFIG_RELATIVE:
            _require(pin["sha256"] == CONFIG_SHA256, "CONFIG_SOURCE_DRIFT", relative)
        low._verify_source(repo, source["code_sha"], relative, pin["sha256"], bound)
        _require(bound[str((repo / relative).resolve())]["bytes"] == pin["bytes"], "SOURCE_PIN_SIZE", relative)
    _require(_same(source["installed_runtime_source_files"], cfg["installed_runtime_source_files"]),
             "INSTALLED_RUNTIME_CLOSURE", "installed Qwen3/masking source pins differ")
    for absolute, expected_pin in cfg["installed_runtime_source_files"].items():
        path = Path(absolute).absolute()
        _bind_binary(path, bound, METADATA_LIMIT)
        _require(_same(bound[str(path)], expected_pin), "INSTALLED_RUNTIME_SOURCE_DRIFT", absolute)


def _validate_supervisor(supervisor, launch, go, runtime, result, cfg, source, config_path, bundle, repo, bound):
    _keys(supervisor, {"status", "reason", "exit_code", "native_qualification", "elapsed_seconds", "limit_seconds",
                       "memory_poll_seconds", "peak_observed_process_tree_rss_bytes"}, "SUPERVISOR_FIELDS")
    _require(supervisor["status"] == "PASS_PROCESS_ONLY" and supervisor["reason"] is None
             and _same(supervisor["exit_code"], 0) and supervisor["native_qualification"] is False
             and _same(supervisor["limit_seconds"], 600) and _same(supervisor["memory_poll_seconds"], .1)
             and _number(supervisor["elapsed_seconds"], minimum=result["wall_seconds"], maximum=600)
             and type(supervisor["peak_observed_process_tree_rss_bytes"]) is int
             and 0 < supervisor["peak_observed_process_tree_rss_bytes"] <= 2 * 2**30,
             "SUPERVISOR_CONTRACT", "complete contained worker within fixed limits required")
    _keys(launch, {"command", "config_sha256", "source_sha", "supervisor_sha256", "watchdog_sha256",
                   "resource_limits", "observed_gpu_processes"}, "LAUNCH_FIELDS")
    _require(_same(go, {"config_sha256": CONFIG_SHA256}) and launch["config_sha256"] == CONFIG_SHA256
             and launch["source_sha"] == result["code_sha"]
             and _same(launch["resource_limits"], {"resource_profile": RESOURCE_PROFILE, **RESOURCE_LIMITS}),
             "LAUNCH_CONTRACT", "config/source/resource profile differs")
    _require(launch["supervisor_sha256"] == source["source_files"][NEW_SOURCE_FILES[2]]["sha256"]
             and launch["watchdog_sha256"] == cfg["reused_source_files"][low.WATCHDOG]["sha256"],
             "SUPERVISOR_SOURCE", "supervisor/watchdog identity differs")
    low._verify_source(repo, result["code_sha"], NEW_SOURCE_FILES[2], launch["supervisor_sha256"], bound)
    low._verify_source(repo, result["code_sha"], low.WATCHDOG, launch["watchdog_sha256"], bound)
    command = launch["command"]
    _require(type(command) is list and len(command) == 10 and all(type(arg) is str for arg in command)
             and command[:4] == [runtime["executable"], "-B", "-m", "experiments.post_training_mimo_centralized_v1.scripts.qualify_capture"]
             and command[4] == "--config" and command[6] == "--output" and command[8] == "--go-path"
             and Path(command[5]).resolve() == config_path.resolve() and Path(command[7]).name == bundle.name
             and Path(command[9]).name == "GO.json"
             and Path(command[9]).parent == Path(command[7]).with_name(bundle.name + "_supervisor"),
             "LAUNCH_COMMAND", "registered worker arguments differ")
    observed = launch["observed_gpu_processes"]
    _require(type(observed) is list and all(low._nonblank(line) for line in observed)
             and not any(term in line.lower() for line in observed for term in ("python", "torch", "llama", "ollama", "vllm", "train", "comfy")),
             "GPU_COMPETITOR_RECORDED", "competing runtime recorded at launch")


def _resource_device_consistency(result, runtime, inventory, supervisor):
    total = runtime["total_vram_bytes"] / 2**30
    for sample in result["resources"]:
        _require(sample["gpu_allocated_gib"] <= sample["gpu_reserved_gib"] <= total
                 and sample["gpu_allocated_gib"] <= sample["peak_allocated_gib"] <= total
                 and sample["gpu_free_gib"] <= total, "DEVICE_CAPACITY", sample["stage"])
        if sample["stage"] != "preload":
            _require(sample["gpu_allocated_gib"] * 2**30 >= inventory["payload_bytes"],
                     "RESIDENT_BASE_MEMORY", "post-load sample omits resident base weights")
    # The external 100ms sampler can miss an instantaneous internal peak; do not
    # equate its RSS measurement with every worker snapshot or GPU reservation.
    _require(supervisor["peak_observed_process_tree_rss_bytes"] <= 2 * 2**30,
             "SUPERVISOR_RSS", "external peak exceeds prospective cap")


def _validate_stdout(path, result):
    last = b""
    with path.open("rb") as stream:
        while True:
            line = stream.readline(METADATA_LIMIT + 1)
            if not line:
                break
            _require(len(line) <= METADATA_LIMIT, "STDOUT_CONTRACT", "oversized stdout line")
            if line.strip():
                last = line.strip()
    expected = {"status": PASS_STATUS, "error": None, "wall_seconds": result["wall_seconds"],
                "generation_tokens": result["generation_tokens"], "optimizer_steps": 0}
    _require(bool(last) and _same(low._parse(last), expected), "STDOUT_CONTRACT", "last worker stdout differs from result")


def inspect_capture_scope(experiment_root, *, capture_directory=None):
    """Check a complete local bundle, streaming model assets only at the last stage."""
    root = Path(experiment_root).resolve()
    repo = root.parents[1]
    bundle = Path(capture_directory).absolute() if capture_directory is not None else root / "results/capture_greedy_short_qual_v1"
    bound = {}
    report = {"schema_version": 1, "status": "BLOCKED", "reasons": [], "bound_files": bound,
              "native_launch_authorized": False, "native_capture_authorized": False, "DEV_collection_authorized": False,
              "E1_authorized": False, "e1_authorized": False, "language_grading_authorized": False,
              "training_authorized": False, "no_native_executor": True, "independent_detokenization": False,
              "claim": "Local byte/source identity and evidence consistency only; no cryptographic execution proof, independent detokenization, current-memory or efficacy claim."}
    try:
        result = _read(bundle / "CAPTURE_RESULT.json", bound)
        if result.get("status") != PASS_STATUS:
            report["reasons"].append({"code": "CAPTURE_NOT_PASSED", "detail": str(result.get("status", "MISSING_STATUS"))[:200]})
            return report
        lengths = validate_result(result)
        config_path = root / CONFIG_RELATIVE
        cfg = _read(config_path, bound)
        _require(bound[str(config_path)]["sha256"] == CONFIG_SHA256, "CAPTURE_CONFIG_IDENTITY", "prospective config bytes changed")
        _require(_same(_read(bundle / "CONFIG.json", bound), cfg), "CAPTURE_CONFIG_DRIFT", "bundle config differs")
        _require(cfg["qualified_core_config"] == CORE_CONFIG_SOURCE and cfg["qualified_core_config_sha256"] == CORE_CONFIG_SHA256
                 and cfg["qualified_CE512_reference"] == CORE_REFERENCE
                 and cfg["qualified_CE512_reference_sha256"] == CORE_REFERENCE_SHA256,
                 "CORE_REGISTRATION", "qualified CE512 parent differs")
        core_path = repo / CORE_CONFIG_SOURCE
        _read(core_path, bound)
        _require(bound[str(core_path)]["sha256"] == CORE_CONFIG_SHA256, "CORE_CONFIG_IDENTITY", "registered CE512 config changed")
        reference_path = root / CORE_REFERENCE
        reference = _read(reference_path, bound)
        _require(bound[str(reference_path)]["sha256"] == CORE_REFERENCE_SHA256
                 and reference["status"] == "PASS_NATIVE_UNPADDED_CE512_QUALIFICATION_ONLY"
                 and reference["evidence_checker_result"] == "PASS_E0_UNPADDED_CE512_SCOPE_EVIDENCE_CONSISTENCY_ONLY"
                 and reference["runtime_scope"] == "UNPADDED_SINGLE_SEQUENCE_CE512_V1" and reference["E1_authorized"] is False,
                 "CORE_REFERENCE_IDENTITY", "pinned qualified-core summary differs")
        supervision = bundle.with_name(bundle.name + "_supervisor")
        _require(not (bundle / "FAILURE_RECEIPT.json").exists() and not (supervision / "FAILURE_RECEIPT.json").exists(),
                 "FAILURE_PRESENT", "claimed pass includes failure receipt")
        primary_path = root / "receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json"
        inventory_path = root / "receipts/titan_static_v1/PARAMETER_INVENTORY.json"
        primary, inventory = _read(primary_path, bound), _read(inventory_path, bound)
        _require(bound[str(primary_path)]["sha256"] == low.PRIMARY_SHA256
                 and bound[str(inventory_path)]["sha256"] == low.INVENTORY_SHA256,
                 "STATIC_IDENTITY", "primary/static tensor inventory changed")
        alias_path = repo / cfg["primary_receipt"].replace("\\", "/")
        alias = _read(alias_path, bound)
        _require(bound[str(alias_path)]["sha256"] == low.PRIMARY_SHA256 and _same(alias, primary),
                 "PRIMARY_RECEIPT_DRIFT", "actual worker primary path differs")
        source = _read(bundle / "SOURCE_RECEIPT.json", bound)
        _validate_source(source, result, cfg, repo, bound)
        cpu = _read(bundle / "CPU_CONTROL_RECEIPT.json", bound)
        validate_cpu_controls(cpu)
        _require(cpu["helper_sha256"] == source["source_files"][NEW_SOURCE_FILES[0]]["sha256"]
                 and cpu["test_source_sha256"] == source["source_files"][CPU_TEST_SOURCE]["sha256"],
                 "CPU_SOURCE_BINDING", "control receipt helper/test source differs")
        runtime = _read(bundle / "RUNTIME_RECEIPT.json", bound)
        _validate_runtime(runtime, cfg)
        _validate_tokenizer(_read(bundle / "TOKENIZER_RECEIPT.json", bound))
        model = _read(bundle / "MODEL_RECEIPT.json", bound)
        _validate_model(model, primary, inventory, result)
        _require(_same(_read(bundle / "RESOURCE_BASELINE.json", bound), {"status": PASS_STATUS, "samples": result["resources"]}),
                 "RESOURCE_RECEIPT_DRIFT", "resource/result snapshots differ")
        cost = _read(bundle / "COST_LEDGER.json", bound)
        event_path = bundle / "EVENTS.jsonl"
        _bind_binary(event_path, bound, EVENT_LIMIT)
        raw = event_path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == bound[str(event_path)]["sha256"], "EVENT_SNAPSHOT_DRIFT", "journal changed while reading")
        lines = raw.splitlines()
        _require(len(lines) == 645 + 4 * sum(lengths) and all(line.strip() for line in lines),
                 "EVENTS_CONTRACT", "blank/incomplete/extra journal records")
        validate_accounting_bundle(result, cost, [low._parse(line) for line in lines], sorted(inventory["tensors"]))
        supervisor = _read(supervision / "SUPERVISOR.json", bound)
        launch, go = _read(supervision / "LAUNCH.json", bound), _read(supervision / "GO.json", bound)
        _validate_supervisor(supervisor, launch, go, runtime, result, cfg, source, config_path, bundle, repo, bound)
        _resource_device_consistency(result, runtime, inventory, supervisor)
        _bind_binary(supervision / "stdout.log", bound, LOG_LIMIT)
        _bind_binary(supervision / "stderr.log", bound, LOG_LIMIT)
        _require(bound[str(supervision / "stdout.log")]["bytes"] + bound[str(supervision / "stderr.log")]["bytes"] <= LOG_LIMIT,
                 "LOG_SIZE_LIMIT", "combined logs exceed containment limit")
        _validate_stdout(supervision / "stdout.log", result)
        # Every source and metadata byte checked above must still be the same
        # before the final potentially large checkpoint byte reads begin.
        for path, pin in list(bound.items()):
            current = Path(path)
            _require(current.stat().st_size == pin["bytes"] and low._digest(current) == pin["sha256"],
                     "EVIDENCE_SNAPSHOT_DRIFT", path)
        _verify_assets(cfg, primary, inventory, model, bound)
        report.update(status=CONSISTENCY_STATUS, runtime_scope=RUNTIME_SCOPE, observed_rollout_lengths=lengths,
                      observed_generated_tokens=sum(lengths), observed_native_eos_ids=result["checks"]["native_eos_coverage"]["terminal_eos_ids"],
                      observed_native_cap_stop=any(f["stop_reason"] == "length" for f in result["fixtures"]),
                      supported_maximum_tokens=512, registered_generation_cap=128,
                      timing_scope="Instrumented generation and teacher-forcing wall times; no GPU-active-time measurement")
    except GateError as error:
        report["reasons"].append({"code": error.code, "detail": str(error)[:400]})
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError, OverflowError, StopIteration, subprocess.SubprocessError) as error:
        report["reasons"].append({"code": "CAPTURE_ARTIFACT_INVALID_OR_MISSING", "detail": f"{type(error).__name__}: {error}"[:400]})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--capture-directory", type=Path)
    args = parser.parse_args(argv)
    report = inspect_capture_scope(args.experiment_root, capture_directory=args.capture_directory)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == CONSISTENCY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
