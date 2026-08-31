"""Semantic validation for the historical production-transfer trial queue."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

TRIAL_QUEUE_SCHEMA_VERSION = "rineya.production-trial-queue.v1"

MECHANISM_CLASSES = {"CHECKPOINT", "RUNTIME", "PRODUCT_POLICY", "SYSTEM"}
ELIGIBILITY_STAGES = {
    "RESEARCH_POSITIVE_TRIAL_ELIGIBLE",
    "ALREADY_INTEGRATED",
    "NOT_TRIAL_ELIGIBLE",
}
TRIAL_STATUSES = {
    "QUEUED",
    "RUNNING",
    "PASS",
    "FAIL",
    "DEFER_MISSING_ARTIFACT",
    "ALREADY_INTEGRATED",
    "EXCLUDED",
}
TRANSFER_RESULTS = {
    "NOT_RUN",
    "TRANSFER_POSITIVE",
    "TRANSFER_FAIL",
    "DEFER_MISSING_ARTIFACT",
    "NOT_APPLICABLE",
}

REQUIRED_FIELDS = {
    "id",
    "mechanism",
    "mechanism_class",
    "eligibility_stage",
    "source_era_phase",
    "source_commit",
    "source_checkpoint",
    "positive_result",
    "control_baseline",
    "preservation_result",
    "artifact_type",
    "bakeability",
    "current_parent_compatibility",
    "expected_integration_cost",
    "known_negative_transfers",
    "trial_configuration",
    "trial_rank",
    "trial_status",
    "transfer_result",
    "reason",
}

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def validate_trial(entry: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(entry))
    if missing:
        return [f"missing fields: {', '.join(missing)}"]

    if entry.get("mechanism_class") not in MECHANISM_CLASSES:
        errors.append(f"unsupported mechanism_class: {entry.get('mechanism_class')!r}")
    if entry.get("eligibility_stage") not in ELIGIBILITY_STAGES:
        errors.append(f"unsupported eligibility_stage: {entry.get('eligibility_stage')!r}")
    if entry.get("trial_status") not in TRIAL_STATUSES:
        errors.append(f"unsupported trial_status: {entry.get('trial_status')!r}")
    if entry.get("transfer_result") not in TRANSFER_RESULTS:
        errors.append(f"unsupported transfer_result: {entry.get('transfer_result')!r}")
    if not isinstance(entry.get("source_commit"), str) or not _COMMIT_RE.fullmatch(
        entry["source_commit"]
    ):
        errors.append("source_commit must be a 40-character lowercase git SHA")
    if not isinstance(entry.get("positive_result"), list) or not entry["positive_result"]:
        errors.append("positive_result must contain at least one evidence artifact")
    if not isinstance(entry.get("known_negative_transfers"), list):
        errors.append("known_negative_transfers must be a list")
    rank = entry.get("trial_rank")
    if rank is not None and (not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0):
        errors.append("trial_rank must be null or a positive integer")

    status = entry.get("trial_status")
    result = entry.get("transfer_result")
    if status == "PASS" and result != "TRANSFER_POSITIVE":
        errors.append("PASS requires transfer_result TRANSFER_POSITIVE")
    if status == "FAIL" and result != "TRANSFER_FAIL":
        errors.append("FAIL requires transfer_result TRANSFER_FAIL")
    if status == "DEFER_MISSING_ARTIFACT" and result != "DEFER_MISSING_ARTIFACT":
        errors.append("DEFER_MISSING_ARTIFACT status requires the matching transfer_result")
    if status == "ALREADY_INTEGRATED" and result != "NOT_APPLICABLE":
        errors.append("ALREADY_INTEGRATED requires transfer_result NOT_APPLICABLE")
    if status in {"QUEUED", "RUNNING"}:
        if entry.get("eligibility_stage") != "RESEARCH_POSITIVE_TRIAL_ELIGIBLE":
            errors.append(f"{status} requires RESEARCH_POSITIVE_TRIAL_ELIGIBLE")
        if result != "NOT_RUN":
            errors.append(f"{status} requires transfer_result NOT_RUN")
    return errors


def validate_trial_queue(queue: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if queue.get("schema_version") != TRIAL_QUEUE_SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {queue.get('schema_version')!r}")
    trials = queue.get("trials")
    if not isinstance(trials, list):
        return errors + ["trials must be a list"]
    if "promotable" in queue:
        errors.append("trial queue must never contain a promotable field")

    ids: set[str] = set()
    ranks: set[int] = set()
    selected = queue.get("selected_trial_id")
    for index, trial in enumerate(trials):
        if not isinstance(trial, Mapping):
            errors.append(f"trials[{index}] must be an object")
            continue
        trial_id = str(trial.get("id", f"index-{index}"))
        if trial_id in ids:
            errors.append(f"duplicate trial id: {trial_id}")
        ids.add(trial_id)
        rank = trial.get("trial_rank")
        if isinstance(rank, int) and not isinstance(rank, bool):
            if rank in ranks:
                errors.append(f"duplicate trial_rank: {rank}")
            ranks.add(rank)
        errors.extend(f"{trial_id}: {error}" for error in validate_trial(trial))
    if selected is not None and selected not in ids:
        errors.append(f"selected_trial_id does not exist: {selected}")
    return errors
