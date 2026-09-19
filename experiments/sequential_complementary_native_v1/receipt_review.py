"""Necessary-only student result screening; never a native PASS or a grant.

The controller's terminal/claim/journal verification is still mandatory. This
stdlib-only screen makes it difficult to confuse a fixture, wrong subject,
changed state, or generation run with the proposed first loader/cache lane.
"""
from __future__ import annotations

from .boundary import CLOSED, LIMITS, WEIGHTS, digest, hex_digest, require

SUBJECT = "rineya.core.titan-sparse-experimental.v1.0.0-devfreeze.28L.supa17-rrp015-control-seed17-magprune-k0p25"
LEGACY_CLOSED = ("training_permission", "teacher_comparisons_permission",
                 "confirmation_permission", "quality_permission", "release_permission")
REQUIRED_CHECKS = {"native_prefix_and_eos_masks", "hidden_logits", "short_cache", "causality",
                   "shape2048_last_logits_cache", "fp32", "no_gradients", "zero_update"}


def screen_student_result(sealed: dict, *, binding: dict, runtime: dict) -> dict:
    require(type(sealed) is dict and hex_digest(sealed.get("sha256")), "sealed result required")
    body = {key: value for key, value in sealed.items() if key != "sha256"}
    require(sealed["sha256"] == digest(body), "result bytes changed")
    require(type(binding) is dict and type(runtime) is dict and bool(runtime), "explicit binding/runtime required")
    require(binding.get("schema") == "SCP_NATIVE_BINDING_V1"
            and binding.get("experiment") == "SCP-NQ-001", "foreign expected binding")
    require(body.get("schema") == "SCP_NATIVE_RESULT_V1" and body.get("experiment") == "SCP-NQ-001"
            and body.get("phase") == "student" and body.get("status") == "PASS_QUALIFICATION_ONLY"
            and body.get("evidence_class") == "NATIVE_QUALIFICATION_ONLY", "not a native student candidate")
    require(all(body.get(key) is False and binding.get(key) is False for key in LEGACY_CLOSED),
            "result/binding opens a closed lane")
    require(body.get("binding_sha256") == digest(binding) and body.get("runtime") == runtime,
            "foreign source/host/runtime binding")
    require(hex_digest(body.get("launch_sha256")) and type(body.get("worker_pid")) is int
            and body["worker_pid"] > 0 and body.get("student_run") is None, "invalid first-worker identity")
    loading, checks = body.get("loading"), body.get("checks")
    require(type(loading) is dict and type(checks) is dict, "missing loading/mechanical records")
    require(loading.get("subject_id") == SUBJECT
            and loading.get("source_weight_sha256") == WEIGHTS["source"]
            and loading.get("trained_partial_sha256") == WEIGHTS["trained_partial"]
            and loading.get("subject_weight_sha256") == WEIGHTS["subject"], "wrong three-way asset identity")
    for key, expected in (("full_tensors_verified", 310), ("trained_tensors_reconstructed", 220)):
        require(type(loading.get(key)) is int and loading[key] == expected, "wrong tensor qualification scope")
    for key in ("backward_calls", "optimizer_steps", "currently_trainable_tensors", "answers_scored"):
        require(type(checks.get(key)) is int and checks[key] == 0, "nonzero or missing closed-lane count")
    require(checks.get("zero_update") is True and checks.get("training_shape_qualified") is False
            and type(checks.get("qualified_sequence_tokens")) is int
            and checks["qualified_sequence_tokens"] == 2048, "incorrect mechanical/training scope")
    require(type(checks.get("checks")) is list and set(checks["checks"]) == REQUIRED_CHECKS,
            "incomplete mechanical checks")
    state = checks.get("state_before")
    require(type(state) is dict and set(state) == {"parameters", "buffers"}
            and all(hex_digest(v) for v in state.values()), "materialized state digests required")
    require(state == checks.get("state_after") == loading.get("subject_state"), "subject state changed")
    require(loading.get("storage_tying") is True and type(loading.get("optimizer_steps")) is int
            and loading["optimizer_steps"] == 0, "loading changed tied storage or optimizer state")
    for key in ("generation_requests", "generated_tokens_actual"):
        require(type(body.get(key)) is int and body[key] == 0, "first launch must not generate")
    forwards = body.get("forward_calls_actual_including_failures")
    require(type(forwards) is int and 1 <= forwards <= LIMITS["evaluation_and_qualification_model_calls"],
            "forward count missing or above first-launch ceiling")
    return {"status": "CANDIDATE_REQUIRES_TERMINAL_AND_JOURNAL_VERIFICATION",
            "result_sha256": sealed["sha256"], "native_pass": False,
            "permissions": {key: False for key in CLOSED},
            "unverified": ["supervisor termination and owned-worker claim", "settled hash-chained journal and all charges",
                           "fresh exact-host observation and separate approval", "on-host source and asset revalidation"],
            "interpretation": "Necessary record consistency only; no inference, training, confirmation, or release authority"}
