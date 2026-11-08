"""Runtime-only invariants for resumable Phase-C qualification execution.

The helpers here are deliberately model-free.  They make deterministic
``no_update_frozen`` reuse and the persisted unplanned-parameter fingerprint
cache independently testable without loading a multi-billion-parameter model.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .hashing import canonical_json_hash

CellKey = tuple[int, float, str]

NO_UPDATE_ARM = "no_update_frozen"
NO_UPDATE_RECORD_KEYS = frozenset(
    {
        "seed",
        "budget_fraction",
        "arm",
        "k",
        "learning_rate",
        "evaluation",
        "delta_audit",
        "unplanned_parameter_hash_unchanged",
        "unplanned_parameter_hash_before",
        "unplanned_parameter_hash_after",
        "full_parameter_hash_before",
        "full_parameter_hash_after",
        "optimizer_state_bytes",
        "wall_clock_seconds",
        "evaluation_tokens",
        "evaluation_tokens_per_second",
        "gradient_storage_bytes",
        "peak_allocated_vram_bytes",
        "peak_reserved_vram_bytes",
        "base_hash_verified",
        "preregistration_file_sha256",
        "optimizer_config_sha256",
        "learning_rate_grid_sha256",
        "dev_data_signature",
        "encoded_data_signature_sha256",
        "maximum_tokens",
        "access_log_sha256",
        "record_sha256",
    }
)


def record_cell_key(record: dict[str, Any]) -> CellKey:
    """Return the canonical seed/budget/arm identity of a run record."""

    return (
        int(record["seed"]),
        float(record["budget_fraction"]),
        str(record["arm"]),
    )


def ordered_stage_cells(stage: dict[str, Any]) -> tuple[CellKey, ...]:
    """Expand a frozen stage, rejecting duplicate dimension values."""

    seeds = tuple(int(value) for value in stage["seeds"])
    fractions = tuple(float(value) for value in stage["budget_fractions"])
    arms = tuple(str(value) for value in stage["arms"])
    for label, values in (
        ("seed", seeds),
        ("budget fraction", fractions),
        ("arm", arms),
    ):
        if len(values) != len(set(values)):
            raise RuntimeError(f"Frozen stage contains a duplicate {label}")
    return tuple(
        (seed, fraction, arm)
        for seed in seeds
        for fraction in fractions
        for arm in arms
    )


def completed_cell_keys(
    records: list[dict[str, Any]], expected_cells: set[CellKey]
) -> set[CellKey]:
    """Validate and return the exact subset already persisted for resume."""

    completed: set[CellKey] = set()
    for record in records:
        key = record_cell_key(record)
        if key not in expected_cells:
            raise RuntimeError(f"Existing run has an unexpected cell: {key}")
        if key in completed:
            raise RuntimeError(f"Existing run has a duplicate cell: {key}")
        completed.add(key)
    return completed


def _validated_no_update_record(record: dict[str, Any]) -> None:
    keys = set(record)
    if keys != NO_UPDATE_RECORD_KEYS:
        missing = sorted(NO_UPDATE_RECORD_KEYS - keys)
        extra = sorted(keys - NO_UPDATE_RECORD_KEYS)
        raise RuntimeError(
            f"No-update record schema mismatch: missing={missing}, extra={extra}"
        )
    if (
        record["arm"] != NO_UPDATE_ARM
        or int(record["k"]) != 0
        or float(record["learning_rate"]) != 0.0
        or int(record["optimizer_state_bytes"]) != 0
        or record["unplanned_parameter_hash_unchanged"] is not True
        or not isinstance(record["unplanned_parameter_hash_before"], str)
        or len(record["unplanned_parameter_hash_before"]) != 64
        or record["unplanned_parameter_hash_before"]
        != record["unplanned_parameter_hash_after"]
        or not isinstance(record["full_parameter_hash_before"], str)
        or len(record["full_parameter_hash_before"]) != 64
        or record["full_parameter_hash_before"] != record["full_parameter_hash_after"]
        or record["delta_audit"]
        != {"valid": True, "planned_k": 0, "changed_union_size": 0}
    ):
        raise RuntimeError("No-update record violates frozen-baseline invariants")
    declared = record.get("record_sha256")
    payload = dict(record)
    payload.pop("record_sha256", None)
    if not isinstance(declared, str) or declared != canonical_json_hash(payload):
        raise RuntimeError("No-update record self-seal is invalid")


def _no_update_semantic_hash(record: dict[str, Any]) -> str:
    """Hash deterministic evidence while excluding resource/timing measurements."""

    evaluation = deepcopy(record["evaluation"])
    evaluation.pop("elapsed_seconds", None)
    return canonical_json_hash(
        {
            "arm": record["arm"],
            "k": record["k"],
            "learning_rate": record["learning_rate"],
            "evaluation": evaluation,
            "delta_audit": record["delta_audit"],
            "unplanned_parameter_hash_unchanged": record[
                "unplanned_parameter_hash_unchanged"
            ],
            "unplanned_parameter_hash_before": record[
                "unplanned_parameter_hash_before"
            ],
            "unplanned_parameter_hash_after": record["unplanned_parameter_hash_after"],
            "full_parameter_hash_before": record["full_parameter_hash_before"],
            "full_parameter_hash_after": record["full_parameter_hash_after"],
            "optimizer_state_bytes": record["optimizer_state_bytes"],
            "evaluation_tokens": record["evaluation_tokens"],
            "gradient_storage_bytes": record["gradient_storage_bytes"],
            "base_hash_verified": record["base_hash_verified"],
            "preregistration_file_sha256": record["preregistration_file_sha256"],
            "optimizer_config_sha256": record["optimizer_config_sha256"],
            "learning_rate_grid_sha256": record["learning_rate_grid_sha256"],
            "dev_data_signature": record["dev_data_signature"],
            "encoded_data_signature_sha256": record["encoded_data_signature_sha256"],
            "maximum_tokens": record["maximum_tokens"],
            "access_log_sha256": record["access_log_sha256"],
        }
    )


def reusable_no_update_source(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return one sealed deterministic baseline, if resume evidence contains one.

    Multiple pre-existing baseline cells may have different elapsed-time/VRAM
    measurements because older runners evaluated each separately.  Their actual
    loss evidence must nevertheless be identical before any one is reused.
    """

    baselines = [record for record in records if record.get("arm") == NO_UPDATE_ARM]
    if not baselines:
        return None
    for record in baselines:
        _validated_no_update_record(record)
    semantic_hashes = {_no_update_semantic_hash(record) for record in baselines}
    if len(semantic_hashes) != 1:
        raise RuntimeError("Existing no-update evaluations are not deterministic")
    return deepcopy(baselines[0])


def clone_no_update_record(
    source: dict[str, Any], *, seed: int, budget_fraction: float
) -> dict[str, Any]:
    """Materialize a cell from one deterministic baseline without schema drift."""

    _validated_no_update_record(source)
    record = deepcopy(source)
    record["seed"] = int(seed)
    record["budget_fraction"] = float(budget_fraction)
    record.pop("record_sha256")
    record["record_sha256"] = canonical_json_hash(record)
    _validated_no_update_record(record)
    return record


def recover_unplanned_fingerprint(records: list[dict[str, Any]]) -> str | None:
    """Recover the canonical non-target fingerprint from sealed sparse cells.

    Every sparse record must prove that its post-step non-target state equals
    the same canonical fresh-base fingerprint.  This turns prior completed
    cells into a safe resume cache and rejects mixed or weakened evidence.
    """

    fingerprints: set[str] = set()
    for record in records:
        if record.get("arm") == NO_UPDATE_ARM:
            continue
        before = record.get("unplanned_parameter_hash_before")
        after = record.get("unplanned_parameter_hash_after")
        if (
            not isinstance(before, str)
            or len(before) != 64
            or before != after
            or record.get("unplanned_parameter_hash_unchanged") is not True
        ):
            raise RuntimeError(
                "Completed sparse cell lacks exact canonical unplanned evidence"
            )
        fingerprints.add(before)
    if len(fingerprints) > 1:
        raise RuntimeError("Completed sparse cells disagree on the fresh-base hash")
    return next(iter(fingerprints), None)
