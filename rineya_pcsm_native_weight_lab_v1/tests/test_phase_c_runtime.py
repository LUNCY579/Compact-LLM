from __future__ import annotations

from copy import deepcopy

import pytest

from rineya_pcsm.hashing import canonical_json_hash
from rineya_pcsm.phase_c_runtime import (
    NO_UPDATE_RECORD_KEYS,
    clone_no_update_record,
    completed_cell_keys,
    ordered_stage_cells,
    recover_unplanned_fingerprint,
    reusable_no_update_source,
)


def no_update_record(seed: int = 11, fraction: float = 0.001) -> dict[str, object]:
    record: dict[str, object] = {
        "seed": seed,
        "budget_fraction": fraction,
        "arm": "no_update_frozen",
        "k": 0,
        "learning_rate": 0.0,
        "evaluation": {
            "mean_nll": 1.25,
            "supervised_tokens": 3,
            "examples": [
                {
                    "row_id": "example",
                    "tokens": 10,
                    "supervised_tokens": 3,
                    "loss": 1.25,
                }
            ],
            "elapsed_seconds": 2.0,
        },
        "delta_audit": {
            "valid": True,
            "planned_k": 0,
            "changed_union_size": 0,
        },
        "unplanned_parameter_hash_unchanged": True,
        "unplanned_parameter_hash_before": "f" * 64,
        "unplanned_parameter_hash_after": "f" * 64,
        "full_parameter_hash_before": "1" * 64,
        "full_parameter_hash_after": "1" * 64,
        "optimizer_state_bytes": 0,
        "wall_clock_seconds": 3.0,
        "evaluation_tokens": 10,
        "evaluation_tokens_per_second": 5.0,
        "gradient_storage_bytes": 48,
        "peak_allocated_vram_bytes": 100,
        "peak_reserved_vram_bytes": 200,
        "base_hash_verified": True,
        "preregistration_file_sha256": "a" * 64,
        "optimizer_config_sha256": "b" * 64,
        "learning_rate_grid_sha256": "c" * 64,
        "dev_data_signature": "d" * 64,
        "encoded_data_signature_sha256": "0" * 64,
        "maximum_tokens": 40960,
        "access_log_sha256": "e" * 64,
    }
    record["record_sha256"] = canonical_json_hash(record)
    return record


def sparse_record(fingerprint: str, arm: str = "real_peer_pcsm_calibrated") -> dict:
    return {
        "arm": arm,
        "unplanned_parameter_hash_before": fingerprint,
        "unplanned_parameter_hash_after": fingerprint,
        "unplanned_parameter_hash_unchanged": True,
    }


def test_no_update_clone_reuses_evidence_without_schema_drift() -> None:
    source = no_update_record()
    clone = clone_no_update_record(source, seed=31, budget_fraction=0.01)
    assert set(clone) == NO_UPDATE_RECORD_KEYS
    assert clone["seed"] == 31
    assert clone["budget_fraction"] == 0.01
    assert clone["evaluation"] == source["evaluation"]
    assert clone["record_sha256"] == canonical_json_hash(
        {key: value for key, value in clone.items() if key != "record_sha256"}
    )
    assert source["seed"] == 11


def test_resume_reuse_accepts_only_identical_deterministic_evidence() -> None:
    first = no_update_record()
    second = no_update_record(seed=17, fraction=0.01)
    second["wall_clock_seconds"] = 99.0
    second["evaluation"]["elapsed_seconds"] = 98.0  # type: ignore[index]
    second["record_sha256"] = canonical_json_hash(
        {key: value for key, value in second.items() if key != "record_sha256"}
    )
    assert reusable_no_update_source([first, second]) == first

    inconsistent = deepcopy(second)
    inconsistent["evaluation"]["mean_nll"] = 1.5  # type: ignore[index]
    inconsistent["record_sha256"] = canonical_json_hash(
        {key: value for key, value in inconsistent.items() if key != "record_sha256"}
    )
    with pytest.raises(RuntimeError, match="not deterministic"):
        reusable_no_update_source([first, inconsistent])


def test_resume_cells_reject_duplicates_and_out_of_stage_records() -> None:
    stage = {
        "seeds": [11, 17],
        "budget_fractions": [0.001],
        "arms": ["no_update_frozen", "random_sparse_native"],
    }
    ordered = ordered_stage_cells(stage)
    first = {"seed": 11, "budget_fraction": 0.001, "arm": "no_update_frozen"}
    assert completed_cell_keys([first], set(ordered)) == {ordered[0]}
    with pytest.raises(RuntimeError, match="duplicate cell"):
        completed_cell_keys([first, first], set(ordered))
    with pytest.raises(RuntimeError, match="unexpected cell"):
        completed_cell_keys(
            [{"seed": 99, "budget_fraction": 0.001, "arm": "no_update_frozen"}],
            set(ordered),
        )


def test_qualification_shape_has_one_reusable_baseline_for_25_cells() -> None:
    stage = {
        "seeds": [11, 17, 23, 29, 31],
        "budget_fractions": [0.0001, 0.0005, 0.001, 0.005, 0.01],
        "arms": [
            "no_update_frozen",
            "random_sparse_native",
            "gradient_topk_sparse_native",
            "target_only_sparse_native",
            "coordinate_shuffled_peer_pcsm",
            "layer_shuffled_peer_pcsm",
            "real_peer_pcsm_bootstrap",
            "real_peer_pcsm_calibrated",
        ],
    }
    cells = ordered_stage_cells(stage)
    baselines = [cell for cell in cells if cell[2] == "no_update_frozen"]
    assert len(cells) == 200
    assert len(baselines) == 25


def test_persisted_canonical_unplanned_fingerprint_is_fail_closed() -> None:
    fingerprint = "a" * 64
    assert recover_unplanned_fingerprint([]) is None
    assert recover_unplanned_fingerprint([sparse_record(fingerprint)]) == fingerprint
    with pytest.raises(RuntimeError, match="disagree"):
        recover_unplanned_fingerprint(
            [sparse_record(fingerprint), sparse_record("b" * 64, arm="control")]
        )
    weakened = sparse_record(fingerprint)
    weakened["unplanned_parameter_hash_after"] = "c" * 64
    with pytest.raises(RuntimeError, match="lacks exact"):
        recover_unplanned_fingerprint([weakened])
