"""Bounded resource observations after imports; never a model-call executor."""
from __future__ import annotations
import math

LIMITS = {"worker_rss_limit_gib": 2.0, "worker_free_gib": 1.25,
          "gpu_free_gib": 4.0, "gpu_worker_free_gib": 0.5, "disk_free_gib": 4.0}
POLICY = {"seconds": 60, "poll_seconds": 2.0, "consecutive_samples": 2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def admission_measurement(sample):
    required = {"rss_gib", "host_available_gib", "gpu_free_gib", "disk_free_gib"}
    require(type(sample) is dict and required <= set(sample), "resource sample fields")
    require(all(type(sample[name]) in (int, float) and math.isfinite(sample[name]) and sample[name] >= 0
                for name in required), "finite nonnegative resource values required")
    require(sample["rss_gib"] <= 2.0, "WAIT_WORKER_RSS_GUARD")
    require(sample["host_available_gib"] >= 1.25, "WAIT_SYSTEM_RESERVE_GUARD")
    require(sample["gpu_free_gib"] >= 0.5 and sample["disk_free_gib"] >= 4.0, "WAIT_GPU_DISK_GUARD")
    growth = max(0.0, 2.0 - sample["rss_gib"])
    required_host = 1.25 + growth
    return {"required_host_gib": required_host, "remaining_worker_growth_gib": growth,
            "host_admitted": sample["host_available_gib"] >= required_host,
            "gpu_admitted": sample["gpu_free_gib"] >= 4.0,
            "admitted": sample["host_available_gib"] >= required_host and sample["gpu_free_gib"] >= 4.0}


def wait_for_admission(sample, *, clock, pause, record, policy=None):
    """Observe up to60 seconds; no model callbacks or automatic model retries.

    The caller's sampler enforces whole-worker wall/source bounds. Every returned
    resource sample is recorded before later checks. At least two consecutive
    current dynamic-budget observations are needed; timeout never counts as pass.
    Returned admission is only resource evidence, never E1/numerical approval.
    """
    policy = POLICY if policy is None else policy
    require(type(policy) is dict and set(policy) == set(POLICY)
            and all(type(policy[name]) is type(POLICY[name]) and policy[name] == POLICY[name] for name in POLICY),
            "registered post-import wait policy required")
    require(all(callable(value) for value in (sample, clock, pause, record)), "resource observation functions required")
    def now():
        value = clock()
        require(type(value) in (int, float) and math.isfinite(value), "finite monotonic clock required")
        return float(value)
    started = previous = now()
    observations, consecutive = 0, 0
    last = None
    while True:
        current = now()
        require(current >= previous, "admission clock moved backward")
        previous = current
        if current - started >= 60:
            status = "TIMED_OUT_POST_IMPORT_ADMISSION_NO_MODEL"
            break
        require(observations < 31, "post-import observation ceiling")
        last = sample()
        observations += 1
        after = now()
        require(after >= previous, "admission clock moved backward")
        previous = after
        # Persist the actual returned observation even if a later guard fails.
        record({"observation": observations, "elapsed_seconds": after - started, "sample": dict(last)})
        measurement = admission_measurement(last)
        decision_time = now()
        require(decision_time >= previous, "admission clock moved backward")
        previous = decision_time
        if decision_time - started >= 60:
            status = "TIMED_OUT_POST_IMPORT_ADMISSION_NO_MODEL"
            break
        consecutive = consecutive + 1 if measurement["admitted"] else 0
        if consecutive == 2:
            status = "ADMITTED_POST_IMPORT_RESOURCE_ONLY"
            break
        pause(min(2.0, 60 - (decision_time - started)))
    ended = now()
    require(ended >= previous, "admission clock moved backward")
    if ended - started >= 60:
        status = "TIMED_OUT_POST_IMPORT_ADMISSION_NO_MODEL"
    return {"status": status, "elapsed_seconds": ended - started, "observations": observations,
            "consecutive_admitted_samples": consecutive, "last_measurement": admission_measurement(last) if last is not None else None,
            "policy": dict(policy), "model_calls": 0, "native_qualification": False,
            "E1_authorized": False}
