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
        "dataset": {
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
    monkeypatch.setattr(
        integrity,
        "require_go",
        lambda _: {
            "verdict": "GO",
            "development_only": False,
            "schema_version": "test",
            "regime": {"fingerprint": "abc"},
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
