"""Offline comparison of supplied export traces; never loads a model or a bank.

All four inputs carry the package's canonical ``sha256`` content digest. These
digests detect changed receipts; they are not digital signatures and cannot prove
that a caller executed a native model. ``TRACE_PARITY_MATCHED`` describes the
supplied records only. Independently reviewed strict-load and DEV scoring receipts
are still necessary for actual loader/scoring qualification.

Schema names are ``gd001_export_parity_policy_v1``,
``gd001_export_parity_expected_v1`` and ``gd001_export_trace_v1``. The expected
record binds the policy digest, exact ordered original snapshot schedule, source,
base, tokenizer, runtime, decoding policy, DEV suite hash and golden probe hashes.
Each trace supplies signed per-checkpoint load and scoring receipts plus full probe
payloads. Each candidate includes every token's target log probability and a bounded
logit vector in the policy's fixed vocabulary order; its total excludes prompt loss.
Golden probes must include a multi-token answer and distractor somewhere in the
suite. Tokens and structural fields compare exactly; only finite numeric endpoints
use the predeclared absolute/relative tolerance. No outcomes set the tolerance.
"""
from __future__ import annotations

import math
import re
from .common import Blocked, digest, require, signed, verify


POLICY_SCHEMA = "gd001_export_parity_policy_v1"
EXPECTED_SCHEMA = "gd001_export_parity_expected_v1"
TRACE_SCHEMA = "gd001_export_trace_v1"
_HEX = re.compile(r"[0-9a-f]{64}")
_LOAD_INFO = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
_IDENTITY = ("trajectory_id", "source_sha256", "base_content_sha256",
             "tokenizer_content_sha256", "expected_base_parameters", "dev_suite_sha256",
             "runtime", "prompt_format", "decoding_policy", "evidence_level")
_CHECKPOINT = ("id", "step", "kind", "checkpoint_content_sha256",
               "snapshot_receipt_sha256", "origin")


def _hash(value, label: str) -> None:
    require(isinstance(value, str) and _HEX.fullmatch(value) is not None,
            f"Missing/invalid {label} content hash")


def _number(value, label: str, nonnegative: bool = False) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"Non-finite/invalid {label}")
    require(not nonnegative or value >= 0, f"Negative {label}")
    return value


def _integer(value, label: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"Invalid {label}")
    return value


def _tokens(value, label: str, limit: int) -> list[int]:
    require(isinstance(value, list) and 0 < len(value) <= limit, f"Missing/over-budget {label}")
    for token in value:
        _integer(token, f"{label} token")
    return value


def _receipt(value, schema: str | None = None) -> dict:
    require(isinstance(value, dict), "Receipt must be an object")
    try:
        verify(value)
    except (TypeError, ValueError) as exc:
        raise Blocked(f"Invalid canonical receipt content/digest: {exc}") from exc
    if schema:
        require(value.get("schema") == schema, f"Unsupported schema; expected {schema}")
    return value


def _close(left: float, right: float, policy: dict, label: str) -> None:
    _number(left, label)
    _number(right, label)
    # Symmetric relative tolerance. No observed value changes registered tolerances.
    bound = policy["abs_tolerance"] + policy["rel_tolerance"] * max(abs(left), abs(right))
    require(math.isfinite(bound), "Registered tolerance produced a non-finite comparison bound")
    require(abs(left - right) <= bound, f"Parity mismatch: {label}")


def _policy(policy: dict, expected: dict) -> None:
    _receipt(policy, POLICY_SCHEMA)
    require(policy.get("registered_before_outcomes") is True,
            "Parity policy must be declared before outcomes")
    require(expected["parity_policy_sha256"] == policy["sha256"],
            "Parity policy changed or loosened after registration")
    _number(policy["abs_tolerance"], "absolute tolerance", True)
    _number(policy["rel_tolerance"], "relative tolerance", True)
    for key in ("max_prompt_tokens", "max_candidate_tokens", "max_generated_tokens",
                "max_probes", "max_logit_probe_tokens"):
        _integer(policy[key], key, 1)
    ids = _tokens(policy["logit_probe_token_ids"], "fixed logit probe vocabulary",
                  policy["max_logit_probe_tokens"])
    require(ids == sorted(set(ids)), "Logit probe vocabulary must be unique and ordered")


def validate_parity_policy(policy: dict) -> dict:
    """Validate an immutable pre-outcome parity policy without executing a model."""
    try:
        _receipt(policy, POLICY_SCHEMA)
        _policy(policy, {"parity_policy_sha256": policy["sha256"]})
        return policy
    except (KeyError, TypeError, OverflowError) as exc:
        raise Blocked(f"Incomplete/invalid export parity policy: {exc}") from exc


def _expected(expected: dict, policy: dict) -> None:
    _receipt(expected, EXPECTED_SCHEMA)
    _policy(policy, expected)
    require(expected.get("registered_before_outcomes") is True,
            "Expected schedule and probes must be declared before outcomes")
    require(isinstance(expected["trajectory_id"], str) and bool(expected["trajectory_id"]),
            "Missing exact trajectory ID")
    for key in ("source_sha256", "base_content_sha256", "tokenizer_content_sha256",
                "dev_suite_sha256"):
        _hash(expected[key], key)
    count = _integer(expected["expected_base_parameters"], "exact base parameter count")
    require(500_000_000 <= count <= 700_000_000, "Exact source must contain 500M-700M base parameters")
    require(expected["evidence_level"] in {"mechanical_fixture", "native_pilot"},
            "Unknown evidence level")
    require(expected.get("split") == "dev", "Only DEV golden probes are admitted")
    runtime = expected["runtime"]
    require(isinstance(runtime, dict), "Missing registered runtime")
    required = {"versions", "python", "device", "dtype", "cpu_threads", "attention",
                "deterministic_algorithms", "local_files_only", "trust_remote_code",
                "quantized", "one_checkpoint_at_a_time"}
    require(required <= set(runtime), "Incomplete runtime identity")
    require(isinstance(runtime["versions"], dict) and bool(runtime["versions"]),
            "Missing exact dependency versions")
    require(all(isinstance(v, str) and v for v in runtime["versions"].values()),
            "Invalid exact dependency versions")
    require(runtime["device"] in {"cpu", "cuda"} and
            runtime["dtype"] in {"float32", "bfloat16"} and
            (runtime["device"] != "cpu" or runtime["dtype"] == "float32"), "Unsupported runtime")
    require(isinstance(runtime["python"], str) and bool(runtime["python"]), "Missing Python version")
    require(_integer(runtime["cpu_threads"], "CPU threads", 1) <= 8, "CPU threads exceed GD guard")
    require(runtime["attention"] == "eager" and
            runtime["deterministic_algorithms"] is True and
            runtime["local_files_only"] is True and runtime["trust_remote_code"] is False and
            runtime["quantized"] is False and runtime["one_checkpoint_at_a_time"] is True,
            "Runtime violates offline strict sequential loading contract")
    require(expected["prompt_format"] in {"plain", "chat"}, "Unsupported prompt format")
    decoding = expected["decoding_policy"]
    require(isinstance(decoding, dict), "Missing decoding policy")
    _number(decoding.get("repetition_penalty"), "repetition penalty")
    require(decoding.get("do_sample") is False and type(decoding.get("num_beams")) is int and
            decoding["num_beams"] == 1 and decoding.get("repetition_penalty") == 1.0 and
            decoding.get("use_cache") is True and decoding.get("enable_thinking") is False and
            decoding.get("truncation") is False and
            decoding.get("candidate_encoding") == "space_prefix_no_special_tokens",
            "Decoding/scoring policy violates GD contract")
    _integer(decoding["eos_token_id"], "EOS token")
    _integer(decoding["pad_token_id"], "padding token")
    limit = _integer(decoding["max_new_tokens"], "new-token budget", 1)
    require(limit <= policy["max_generated_tokens"], "Generation exceeds parity trace bound")

    schedule = expected["checkpoints"]
    require(isinstance(schedule, list) and 3 <= len(schedule) <= 20,
            "Require exact base plus 2-19 original saved snapshots")
    ids, steps, kinds = [], [], set()
    for index, checkpoint in enumerate(schedule):
        require(isinstance(checkpoint, dict) and set(_CHECKPOINT) <= set(checkpoint),
                "Incomplete registered checkpoint identity")
        identifier = checkpoint["id"]
        require(isinstance(identifier, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier),
                "Invalid checkpoint ID")
        ids.append(identifier)
        steps.append(_integer(checkpoint["step"], "checkpoint step"))
        _hash(checkpoint["checkpoint_content_sha256"], "checkpoint")
        if index == 0:
            require(checkpoint["kind"] == "base" and checkpoint["origin"] == "base" and
                    checkpoint["step"] == 0 and checkpoint["snapshot_receipt_sha256"] is None and
                    checkpoint["checkpoint_content_sha256"] == expected["base_content_sha256"],
                    "First checkpoint must be the exact unmodified base")
        else:
            require(checkpoint["kind"] in {"hf", "peft"} and
                    checkpoint["origin"] == "original_training_snapshot",
                    "Require original training saves; qualification updates/derived exports cannot substitute")
            _hash(checkpoint["snapshot_receipt_sha256"], "post-save snapshot receipt")
            kinds.add(checkpoint["kind"])
    require(len(set(ids)) == len(ids) and steps == sorted(set(steps)),
            "Checkpoint schedule must have unique IDs and strictly increasing steps")
    require(len(kinds) == 1, "Mixed HF/PEFT snapshot trajectory")
    require(expected["base_checkpoint_id"] == ids[0] and expected["final_checkpoint_id"] == ids[-1],
            "Base/final references must match the registered schedule")

    probes = expected["probes"]
    require(isinstance(probes, list) and 0 < len(probes) <= policy["max_probes"],
            "Missing/over-budget DEV golden probe inventory")
    probe_ids, multi_token = [], set()
    for probe in probes:
        require(isinstance(probe["id"], str) and bool(probe["id"]), "Missing DEV probe ID")
        probe_ids.append(probe["id"])
        for key in ("prompt_sha256", "rendered_prompt_sha256", "prompt_token_sha256"):
            _hash(probe[key], key)
        require(0 < _integer(probe["prompt_tokens"], "registered prompt token count") <=
                policy["max_prompt_tokens"], "Registered prompt exceeds trace bound")
        require([c["label"] for c in probe["candidates"]] == ["answer", "distractor"],
                "Require both complete answer and distractor candidates")
        for candidate in probe["candidates"]:
            _hash(candidate["text_sha256"], "candidate text")
            _hash(candidate["token_sha256"], "candidate tokens")
            count = _integer(candidate["tokens"], "candidate token count", 1)
            require(count <= min(policy["max_candidate_tokens"], limit),
                    "Registered candidate exceeds trace bound")
            if count > 1:
                multi_token.add(candidate["label"])
    require(len(set(probe_ids)) == len(probe_ids), "Duplicate DEV probe ID")
    require(multi_token == {"answer", "distractor"},
            "Golden probes require multi-token candidates; a one-token screen cannot qualify full scoring")


def _load(load: dict, checkpoint: dict, expected: dict) -> None:
    _receipt(load)
    require(load["status"] == "STRICT_LOAD_REPORTED", "Missing strict-load declaration")
    for key in ("source_sha256", "base_content_sha256", "tokenizer_content_sha256", "evidence_level"):
        require(load[key] == expected[key], f"Strict-load {key} mismatch")
    require(load["checkpoint_content_sha256"] == checkpoint["checkpoint_content_sha256"] and
            load["checkpoint_id"] == checkpoint["id"], "Strict-load checkpoint mismatch")
    require(load["runtime_sha256"] == digest(expected["runtime"]), "Strict-load runtime mismatch")
    require(type(load["actual_base_parameters"]) is int and
            load["actual_base_parameters"] == expected["expected_base_parameters"],
            "Actual loaded base parameter count mismatch")
    require(all(load.get(key) == [] for key in _LOAD_INFO), "Non-strict full-model load")
    require(load.get("eval_mode") is True and load.get("requires_grad") is False,
            "Load must report inference mode without gradients")
    if checkpoint["kind"] == "peft":
        require(load.get("adapter_tensor_keys_equal") is True and
                load.get("adapter_tensor_shapes_equal") is True and
                load.get("adapter_tensors_equal") is True, "Adapter tensors not loaded exactly")
    else:
        require(all(load.get(k) is None for k in ("adapter_tensor_keys_equal",
                "adapter_tensor_shapes_equal", "adapter_tensors_equal")), "Unexpected adapter load")


def _probe(row: dict, expected_probe: dict, expected: dict, policy: dict) -> None:
    require(row["id"] == expected_probe["id"] and row.get("split") == "dev", "DEV probe identity mismatch")
    require(isinstance(row["prompt"], str) and isinstance(row["rendered_prompt"], str),
            "Actual prompt payloads are required")
    for field in ("prompt", "rendered_prompt"):
        require(digest(row[field]) == expected_probe[f"{field}_sha256"], "Prompt payload hash mismatch")
    require(row.get("truncation_applied") is False, "Truncated prompts cannot qualify parity")
    prompt = _tokens(row["prompt_token_ids"], "actual prompt tokens", policy["max_prompt_tokens"])
    require(len(prompt) == expected_probe["prompt_tokens"] and
            row["prompt_token_sha256"] == digest(prompt) == expected_probe["prompt_token_sha256"],
            "Actual prompt tokenization mismatch")
    generated = _tokens(row["generated_token_ids"], "generated tokens", expected["decoding_policy"]["max_new_tokens"])
    eos = expected["decoding_policy"]["eos_token_id"]
    require(type(row["eos_token_id"]) is int and row["eos_token_id"] == eos,
            "Generated EOS policy mismatch")
    require(isinstance(row["generated_text"], str), "Missing generation text")
    if row["stop_reason"] == "eos":
        require(generated[-1] == eos and eos not in generated[:-1], "Invalid EOS stop receipt")
    else:
        require(row["stop_reason"] == "length" and eos not in generated and
                len(generated) == expected["decoding_policy"]["max_new_tokens"], "Invalid length stop receipt")
    require(isinstance(row["candidates"], list) and len(row["candidates"]) == 2,
            "Both complete candidates must be scored")
    for candidate, registered in zip(row["candidates"], expected_probe["candidates"]):
        require(candidate["label"] == registered["label"] and isinstance(candidate["text"], str) and
                digest(candidate["text"]) == registered["text_sha256"], "Candidate text/label mismatch")
        ids = _tokens(candidate["token_ids"], "complete candidate tokens", policy["max_candidate_tokens"])
        require(len(ids) == registered["tokens"] and
                candidate["token_sha256"] == digest(ids) == registered["token_sha256"],
                "Complete candidate tokenization mismatch")
        require(candidate.get("teacher_forced") is True and candidate.get("includes_prompt_loss") is False and
                type(candidate.get("logit_shift")) is int and candidate["logit_shift"] == 1 and
                candidate.get("candidate_encoding") == "space_prefix_no_special_tokens",
                "Full answer-only teacher-forced scoring contract missing")
        vector = candidate["token_scores"]
        require(isinstance(vector, list) and len(vector) == len(ids),
                "Every candidate token needs its own logits/logp vector")
        total = 0.0
        for index, (token, score) in enumerate(zip(ids, vector)):
            require(type(score["position"]) is int and score["position"] == index and
                    type(score["token_id"]) is int and score["token_id"] == token,
                    "Teacher-forced token positions/IDs mismatch")
            logits = score["probe_logits"]
            require(isinstance(logits, list) and len(logits) == len(policy["logit_probe_token_ids"]),
                    "Incomplete bounded logit vector")
            for logit in logits:
                _number(logit, "teacher-forced logit")
            logp = _number(score["target_logp"], "target-token log probability")
            require(logp <= 0, "Positive target-token log probability")
            total += logp
        _close(candidate["answer_logp"], total, policy, "full answer log probability sum")
        require(candidate["answer_logp"] <= 0, "Positive full answer log probability")


def _trace(trace: dict, role: str, expected: dict, policy: dict) -> None:
    _receipt(trace, TRACE_SCHEMA)
    require(trace["role"] == role and trace.get("split") == "dev", "Trace role or DEV split mismatch")
    require(trace["expected_sha256"] == expected["sha256"] and
            trace["parity_policy_sha256"] == policy["sha256"], "Trace registration/policy digest mismatch")
    for key in _IDENTITY:
        require(digest(trace[key]) == digest(expected[key]), f"Trace {key} mismatch")
    require(trace.get("caller_declared_model_execution") is (expected["evidence_level"] == "native_pilot"),
            "Mechanical/native execution declaration mismatch")
    records = trace["checkpoints"]
    require(isinstance(records, list) and len(records) == len(expected["checkpoints"]),
            "Trace omits registered checkpoints")
    for checkpoint, registered in zip(records, expected["checkpoints"]):
        require(all(digest(checkpoint[key]) == digest(registered[key]) for key in _CHECKPOINT),
                "Trace checkpoint schedule/receipt mismatch")
        _load(checkpoint["load_receipt"], registered, expected)
        scoring = _receipt(checkpoint["scoring_receipt"])
        require(scoring["status"] == "DEV_SCORING_REPORTED" and scoring["split"] == "dev" and
                scoring["evidence_level"] == expected["evidence_level"] and
                scoring["checkpoint_id"] == registered["id"] and
                scoring["load_receipt_sha256"] == checkpoint["load_receipt"]["sha256"] and
                scoring["dev_suite_sha256"] == expected["dev_suite_sha256"] and
                scoring["probe_payload_sha256"] == digest(checkpoint["probes"]) and
                scoring["full_candidate_scoring"] is True, "Missing/broken DEV scoring receipt")
        require(isinstance(checkpoint["probes"], list) and
                len(checkpoint["probes"]) == len(expected["probes"]), "Incomplete DEV golden probe inventory")
        for row, registered_probe in zip(checkpoint["probes"], expected["probes"]):
            _probe(row, registered_probe, expected, policy)


def compare_export_parity(producer: dict, gd: dict, expected: dict, policy: dict) -> dict:
    """Compare signed, predeclared DEV trace records without executing model code.

    Raises ``Blocked`` for invalid receipts, incomplete provenance or mismatches.
    Both native-labeled and fixture comparisons retain their declared evidence level;
    successful fixture parity can never satisfy a native expected specification.
    Input digests authenticate content integrity only, not callers or execution.
    """
    try:
        _expected(expected, policy)
        _trace(producer, "producer", expected, policy)
        _trace(gd, "gd", expected, policy)
        compared_tokens = 0
        for left_checkpoint, right_checkpoint in zip(producer["checkpoints"], gd["checkpoints"]):
            for left, right in zip(left_checkpoint["probes"], right_checkpoint["probes"]):
                for key in ("prompt", "rendered_prompt", "prompt_token_ids", "generated_token_ids",
                            "eos_token_id", "stop_reason", "generated_text"):
                    require(left[key] == right[key], f"Parity mismatch: {key}")
                for left_candidate, right_candidate in zip(left["candidates"], right["candidates"]):
                    require(left_candidate["token_ids"] == right_candidate["token_ids"],
                            "Parity mismatch: complete candidate tokens")
                    _close(left_candidate["answer_logp"], right_candidate["answer_logp"],
                           policy, "full answer log probability")
                    for left_score, right_score in zip(left_candidate["token_scores"], right_candidate["token_scores"]):
                        _close(left_score["target_logp"], right_score["target_logp"], policy, "target-token log probability")
                        for left_logit, right_logit in zip(left_score["probe_logits"], right_score["probe_logits"]):
                            _close(left_logit, right_logit, policy, "teacher-forced logit")
                        compared_tokens += 1
        return signed({"schema": "gd001_export_parity_result_v1", "status": "TRACE_PARITY_MATCHED",
                       "evidence_level": expected["evidence_level"], "split": "dev",
                       "trajectory_id": expected["trajectory_id"], "expected_sha256": expected["sha256"],
                       "parity_policy_sha256": policy["sha256"], "producer_trace_sha256": producer["sha256"],
                       "gd_trace_sha256": gd["sha256"], "checkpoint_count": len(expected["checkpoints"]),
                       "probes_per_checkpoint": len(expected["probes"]), "compared_candidate_tokens": compared_tokens,
                       "native_execution_independently_verified": False, "loaded_scored_qualification": "NOT_ESTABLISHED",
                       "requires_separate_strict_load_and_dev_scoring_receipts": True,
                       "registration_timing": "caller-declared; not independently verified",
                       "digest_limit": "Content digests are not digital signatures or execution attestation",
                       "permits_confirmation": False, "permits_model_promotion": False})
    except (KeyError, TypeError, OverflowError) as exc:
        raise Blocked(f"Incomplete/invalid export parity receipt: {exc}") from exc
