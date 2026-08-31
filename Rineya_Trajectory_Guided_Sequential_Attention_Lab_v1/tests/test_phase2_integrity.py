from __future__ import annotations

import json
from pathlib import Path

import pytest

import tgsa.phase2_integrity as integrity


def _study() -> dict[str, object]:
    return {
        "schema_version": "tgsa.phase2-test.v1",
        "study_id": "phase2-integrity-test",
        "development_only": True,
        "confirmatory": False,
        "seeds": [1],
        "cells": [
            {"tgsa_arm": "cascade", "mtp_mode": "dormant"},
            {"tgsa_arm": "cascade", "mtp_mode": "trajectory"},
        ],
        "dataset": {
            "vocabulary": {"domain_size": 8},
            "families": {"task": {"min_nodes": 4, "max_nodes": 5}},
            "split_generation_order": ["test", "dev", "train"],
            "splits": {
                "train": {"task": 100},
                "dev": {"task": 10},
                "test": {"task": 10},
            },
        },
        "model": {"embed_dim": 16},
        "mtp": {"max_offset": 2},
        "training": {"steps": 25, "batch_size": 4},
        "evaluation": {"split": "dev"},
        "runtime": {
            "device": "cpu",
            "num_threads": 1,
            "deterministic_algorithms": True,
        },
        "registration": {
            "status": "draft",
            "phase1_competence_report": None,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_marker_is_explicit() -> None:
    study = _study()
    del study["development_only"]
    with pytest.raises(integrity.Phase2IntegrityError, match="development_only"):
        integrity.validate_phase2_contract(study, config_path=Path("config.json"))


def test_development_and_confirmatory_flags_cannot_disagree() -> None:
    study = _study()
    study["development_only"] = False
    with pytest.raises(integrity.Phase2IntegrityError, match="non-confirmatory"):
        integrity.validate_phase2_contract(study, config_path=Path("config.json"))


def test_confirmatory_cannot_be_development_only() -> None:
    study = _study()
    study["confirmatory"] = True
    with pytest.raises(integrity.Phase2IntegrityError, match="cannot be marked"):
        integrity.validate_phase2_contract(study, config_path=Path("config.json"))


def test_split_generation_order_is_fail_closed() -> None:
    study = _study()
    study["dataset"]["split_generation_order"] = ["train", "dev", "test"]  # type: ignore[index]
    with pytest.raises(integrity.Phase2IntegrityError, match="test','dev','train"):
        integrity.validate_phase2_contract(study, config_path=Path("config.json"))


def test_development_go_cannot_be_laundered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "competence.json"
    _write_json(report, {})
    study = _study()
    study["development_only"] = False
    study["confirmatory"] = True
    study["registration"] = {
        "status": "frozen",
        "phase1_competence_report": str(report),
    }
    monkeypatch.setattr(
        integrity,
        "require_go",
        lambda _: {"verdict": "GO", "development_only": True},
    )
    with pytest.raises(integrity.Phase2IntegrityError, match="cannot certify"):
        integrity.validate_phase2_contract(
            study, config_path=tmp_path / "config.json"
        )


def test_external_non_development_go_is_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "competence.json"
    _write_json(report, {"artifact": "immutable"})
    study = _study()
    study["development_only"] = False
    study["confirmatory"] = True
    study["registration"] = {
        "status": "frozen",
        "phase1_competence_report": str(report),
    }
    # The stub must carry the scope it certifies. Within-study qualification is
    # now part of the confirmatory contract: a report that cannot be shown to
    # describe THIS study is refused, however valid its verdict.
    from tgsa.scope import compute_scope

    monkeypatch.setattr(
        integrity,
        "require_go",
        lambda _: {
            "verdict": "GO",
            "development_only": False,
            "schema_version": "test",
            "regime": {"fingerprint": "abc"},
            "scope": compute_scope(study, auxiliary=study["mtp"]),
        },
    )
    contract = integrity.validate_phase2_contract(
        study, config_path=tmp_path / "config.json"
    )
    external = contract["external_phase1_competence"]
    assert external["sha256"] == integrity.sha256_file(report)  # type: ignore[index]
    assert external["development_only"] is False  # type: ignore[index]


def test_device_configuration_changes_the_fingerprint() -> None:
    first = integrity.capture_device_identity(
        {"device": "cpu", "num_threads": 1, "deterministic_algorithms": True}
    )
    second = integrity.capture_device_identity(
        {"device": "cpu", "num_threads": 2, "deterministic_algorithms": True}
    )
    assert integrity.sha256_json(first) != integrity.sha256_json(second)


def test_training_dose_records_data_diversity_and_epochs() -> None:
    dose = integrity.training_dose(_study(), unique_train_examples=100)
    assert dose["sample_presentations"] == 100
    assert dose["unique_train_examples"] == 100
    assert dose["effective_epochs"] == pytest.approx(1.0)


def test_unsupported_distributed_dose_is_rejected() -> None:
    study = _study()
    study["training"]["gradient_accumulation_steps"] = 2  # type: ignore[index]
    with pytest.raises(integrity.Phase2IntegrityError, match="sample-presentation"):
        integrity.validate_phase2_contract(study, config_path=Path("config.json"))


def test_integrity_bundle_verifies_and_detects_tampering(tmp_path: Path) -> None:
    study = _study()
    config = tmp_path / "config.json"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    _write_json(config, study)
    _write_json(
        dataset / "dataset_manifest.json",
        {
            "split_generation_order": ["test", "dev", "train"],
            "unbiased_split": "test",
            "splits": {
                "train": {"examples": 100},
                "dev": {"examples": 10},
                "test": {"examples": 10},
            },
        },
    )
    run_path = output / "runs" / "parallel_ar" / "seed_1" / "run_manifest.json"
    _write_json(
        run_path,
        {
            "schema_version": "test",
            "status": "complete",
            "arm": "parallel_ar",
            "seed": 1,
            "claim_scope": "development",
        },
    )
    _write_json(
        output / "competence_report.json",
        {"schema_version": "test", "verdict": "GO"},
    )
    _write_json(
        output / "matrix_manifest.json",
        {
            "schema_version": "test",
            "status": "complete",
            "study_id": "test",
            "runs": [
                {
                    "arm": "parallel_ar",
                    "seed": 1,
                    "manifest": "runs/parallel_ar/seed_1/run_manifest.json",
                }
            ],
        },
    )

    top = integrity.attach_phase2_integrity(
        config_path=config,
        dataset_dir=dataset,
        output_dir=output,
        implementation="unit-test",
    )
    assert top["development_only"] is True
    assert top["mechanism_interpretation_allowed"] is False
    competence = json.loads(
        (output / "competence_report.json").read_text(encoding="utf-8")
    )
    assert competence["certifies_confirmatory"] is False

    report = integrity.verify_phase2_integrity(
        config_path=config,
        dataset_dir=dataset,
        output_dir=output,
    )
    assert report["passed"] is True, report

    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["seed"] = 2
    _write_json(run_path, run)
    tampered = integrity.verify_phase2_integrity(
        config_path=config,
        dataset_dir=dataset,
        output_dir=output,
    )
    assert tampered["passed"] is False
    assert any("run manifest hash mismatch" in error for error in tampered["errors"])


# --------------------------------------------------------------------------
# Within-study qualification: the report must be ABOUT this study
# --------------------------------------------------------------------------


def _confirmatory_study(tmp_path, **overrides):
    """A minimal confirmatory Phase-2 config plus a matching competence report."""

    import json as _json

    from tgsa.scope import compute_scope

    study = {
        "development_only": False,
        "confirmatory": True,
        "arms": ["cascade", "history_control", "learned_delta"],
        "dataset": {
            "vocabulary": {"domain_size": 32, "max_hops": 8, "max_abs_step": 8},
            "families": {"direct_lookup": {"min_nodes": 5, "max_nodes": 8,
                                           "min_hops": 1, "max_hops": 1}},
            "split_generation_order": ["test", "dev", "train"],
            "splits": {"train": {"direct_lookup": 100}},
        },
        "model": {"embed_dim": 96, "num_heads": 6, "num_layers": 2,
                  "ffn_multiplier": 4.0, "trajectory_rank": 16,
                  "tie_embeddings": True, "bias": False, "dropout": 0.0,
                  "max_sequence_length": 35},
        "training": {"learning_rate": 1e-3, "batch_size": 64,
                     "gradient_accumulation_steps": 1},
        "runtime": {"world_size": 1},
        "analysis": {"primary_families": ["direct_lookup"],
                     "interval_estimator": "crossed_cluster_t",
                     "primary_gate": "lower_bound_above_zero"},
        "registration": {"status": "frozen",
                         "phase1_competence_report": "competence.json"},
    }
    study.update(overrides)

    report_scope = compute_scope(dict(study, **overrides.get("_report_study", {})))
    report = {
        "schema_version": "tgsa.phase1-competence-gate.v3",
        "verdict": "GO",
        "development_only": False,
        "checks": {}, "reasons": [],
        "regime": {"split_generation_order": ["holdout", "train"]},
        "scope": report_scope,
    }
    (tmp_path / "competence.json").write_text(_json.dumps(report), encoding="utf-8")
    config_path = tmp_path / "study.json"
    config_path.write_text(_json.dumps(study), encoding="utf-8")
    return study, config_path


def test_a_confirmatory_study_accepts_a_report_on_its_own_scope(tmp_path) -> None:
    from tgsa.phase2_integrity import validate_phase2_contract

    study, config_path = _confirmatory_study(tmp_path)
    contract = validate_phase2_contract(study, config_path=config_path)
    qualification = contract["external_phase1_competence"]["within_study_qualification"]
    assert set(qualification["matched_fields"]) >= {
        "task_scope_sha256", "model_scope_sha256",
        "objective_scope_sha256", "competence_scope_sha256",
    }


def test_a_report_from_a_different_depth_cannot_certify(tmp_path) -> None:
    """v4 never ran its wider rung. A GO at depth 2 says nothing about depth 4."""

    import json as _json

    from tgsa.phase2_integrity import Phase2IntegrityError, validate_phase2_contract
    from tgsa.scope import compute_scope

    study, config_path = _confirmatory_study(tmp_path)
    shallower = _json.loads(_json.dumps(study))
    shallower["model"]["num_layers"] = 4
    report = _json.loads((tmp_path / "competence.json").read_text(encoding="utf-8"))
    report["scope"] = compute_scope(shallower)
    (tmp_path / "competence.json").write_text(_json.dumps(report), encoding="utf-8")

    with pytest.raises((Phase2IntegrityError, ValueError), match="different science"):
        validate_phase2_contract(study, config_path=config_path)


def test_a_report_from_a_different_band_cannot_certify(tmp_path) -> None:
    """The v4 lesson: a GO is about the families that were actually measured."""

    import json as _json

    from tgsa.phase2_integrity import Phase2IntegrityError, validate_phase2_contract
    from tgsa.scope import compute_scope

    study, config_path = _confirmatory_study(tmp_path)
    other_band = _json.loads(_json.dumps(study))
    other_band["dataset"]["families"]["direct_lookup"]["max_nodes"] = 20
    report = _json.loads((tmp_path / "competence.json").read_text(encoding="utf-8"))
    report["scope"] = compute_scope(other_band)
    (tmp_path / "competence.json").write_text(_json.dumps(report), encoding="utf-8")

    with pytest.raises((Phase2IntegrityError, ValueError), match="different science"):
        validate_phase2_contract(study, config_path=config_path)


def test_a_report_with_no_scope_block_cannot_certify(tmp_path) -> None:
    import json as _json

    from tgsa.phase2_integrity import Phase2IntegrityError, validate_phase2_contract

    study, config_path = _confirmatory_study(tmp_path)
    report = _json.loads((tmp_path / "competence.json").read_text(encoding="utf-8"))
    del report["scope"]
    (tmp_path / "competence.json").write_text(_json.dumps(report), encoding="utf-8")

    with pytest.raises((Phase2IntegrityError, ValueError), match="no `scope` block"):
        validate_phase2_contract(study, config_path=config_path)


def test_development_runs_are_not_scope_gated(tmp_path) -> None:
    """The gate applies to confirmatory studies; the rehearsal must still run."""

    import json as _json

    from tgsa.phase2_integrity import validate_phase2_contract

    study, config_path = _confirmatory_study(tmp_path)
    study["confirmatory"] = False
    study["development_only"] = True
    config_path.write_text(_json.dumps(study), encoding="utf-8")
    contract = validate_phase2_contract(study, config_path=config_path)
    assert contract["external_phase1_competence"]["required"] is False
