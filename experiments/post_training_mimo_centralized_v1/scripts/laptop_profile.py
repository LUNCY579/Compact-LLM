"""Prospective, exact laptop resource profile; no model-library imports.

This profile changes containment and admission, not the numerical E0 workload.
The short load/forward probe has a distinct status and cannot qualify full E0.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


RESOURCE_PROFILE = "LAPTOP_GPU_RESIDENT_V1"
CONFIG_STATUS = "FROZEN_E0_LAPTOP_ONLY"
ADMISSION_BUDGET = "system_reserve_plus_remaining_worker_rss"
PROBE_PASS_STATUS = "PASS_E0_LOAD_FORWARD_RESOURCE_ONLY"
BACKWARD_PASS_STATUS = "PASS_E0_BACKWARD_RESUME_RESOURCE_ONLY"
FULL_PASS_STATUS = "PASS_E0_SCOPED_NO_EFFICACY_CLAIM"
LEGACY_CONFIG_SHA256 = "8fcab3ce006c9a698af86975769a4076edc78cf8005f64ad823d9d47bcbe518d"
LEGACY_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/E0_TITAN_BF16.json"
PROFILE_FIELDS = frozenset(("resource_profile", "qualification_stage", "worker_rss_limit_gib", "admission_budget"))


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def metadata_present(cfg):
    """Any prospective field forces exact laptop validation, even an invalid name."""
    return type(cfg) is dict and bool(PROFILE_FIELDS.intersection(cfg))


def expected_config(stage):
    """Derive a complete registration from byte-pinned immutable historical E0."""
    _require(type(stage) is str and stage in ("load_forward", "full", "backward_resume"), "unregistered laptop qualification stage")
    raw = LEGACY_CONFIG_PATH.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == LEGACY_CONFIG_SHA256, "historical E0 config drift")
    cfg = deepcopy(json.loads(raw))
    cfg.update(status=CONFIG_STATUS, resource_profile=RESOURCE_PROFILE, qualification_stage=stage,
               host_free_gib=2.0, worker_free_gib=2.0, worker_rss_limit_gib=2.5,
               admission_budget=ADMISSION_BUDGET, wall_seconds=600 if stage == "load_forward" else 1800)
    if stage == "load_forward":
        cfg.update(maximum_optimizer_calls=0, generation_cap=0, parity_prompts=0, contexts=[256],
                   update_semantics="no adapters, backward, optimizer updates, generation or benchmarks; strict load and one frozen 256-token forward only")
    elif stage == "backward_resume":
        cfg.update(generation_cap=0,
                   update_semantics="disposable adapter backward/update/resume resource qualification only; padding and sampled-likelihood qualification remain separate and unresolved")
    return cfg


def validate_config(cfg):
    _require(type(cfg) is dict, "invalid laptop config")
    expected = expected_config(cfg.get("qualification_stage"))
    try:
        actual_json = json.dumps(cfg, sort_keys=True, allow_nan=False, separators=(",", ":"))
        expected_json = json.dumps(expected, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("nonfinite/invalid laptop config") from exc
    _require(actual_json == expected_json, "unregistered laptop config or resource envelope")


def admission_required_gib(cfg, current_rss_gib):
    """Reserve 2 GiB for the system plus the worker's remaining RSS allowance."""
    validate_config(cfg)
    _require(type(current_rss_gib) in (int, float) and math.isfinite(current_rss_gib) and current_rss_gib >= 0,
             "nonfinite/invalid worker RSS")
    return cfg["host_free_gib"] + max(0.0, cfg["worker_rss_limit_gib"] - current_rss_gib)


def resource_limits(cfg):
    validate_config(cfg)
    return {name: cfg[name] for name in ("host_free_gib", "worker_free_gib", "worker_rss_limit_gib",
                                       "gpu_free_gib", "gpu_worker_free_gib", "disk_free_gib", "wall_seconds",
                                       "admission_budget")}
