"""End-to-end evidence-chain tests against a real (tiny) study bundle.

These are the pre-launch acceptance gates that need artifacts to exist:

* an end-to-end smoke matrix completes and schema-validates;
* a smoke analysis is unconditionally ``smoke_only``;
* stale config / data / source / checkpoint / prediction mutations are each
  rejected by a *negative* test;
* interrupted and uninterrupted training are equivalent under the registered
  determinism contract.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from tgsa.phase1_analysis import analyze_study
from tgsa.phase1_runner import (
    load_study_config,
    run_study_matrix,
    train_one_run,
)
from tgsa.phase1_verify import verify_evidence_chain

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
SMOKE_CONFIG = LAB_ROOT / "configs" / "phase1_identifiability_v3_smoke.json"


@pytest.fixture(scope="module")
def smoke_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One complete smoke study, reused read-only by the chain tests."""

    root = tmp_path_factory.mktemp("tgsa_smoke")
    dataset_dir = root / "dataset"
    output_root = root / "study"
    matrix = run_study_matrix(
        study_config_path=SMOKE_CONFIG,
        dataset_dir=dataset_dir,
        output_root=output_root,
        repo_root=REPO_ROOT,
    )
    assert matrix["status"] == "complete", matrix
    return {"root": root, "dataset_dir": dataset_dir, "output_root": output_root}


def _verify(bundle: dict[str, Path]) -> dict[str, object]:
    return verify_evidence_chain(
        study_config_path=SMOKE_CONFIG,
        output_root=bundle["output_root"],
        dataset_dir=bundle["dataset_dir"],
        repo_root=REPO_ROOT,
    )


def _clone(bundle: dict[str, Path], tmp_path: Path) -> dict[str, Path]:
    dataset_dir = tmp_path / "dataset"
    output_root = tmp_path / "study"
    shutil.copytree(bundle["dataset_dir"], dataset_dir)
    shutil.copytree(bundle["output_root"], output_root)
    return {"root": tmp_path, "dataset_dir": dataset_dir, "output_root": output_root}


def test_end_to_end_smoke_matrix_produces_a_closed_chain(smoke_bundle: dict[str, Path]) -> None:
    report = _verify(smoke_bundle)
    assert report["passed"] is True, report["errors"]
    assert len(report["checks"]) > 50


def test_smoke_analysis_is_unconditionally_non_evidentiary(smoke_bundle: dict[str, Path]) -> None:
    analysis = analyze_study(
        study_config_path=SMOKE_CONFIG,
        output_root=smoke_bundle["output_root"],
        dataset_dir=smoke_bundle["dataset_dir"],
        repo_root=REPO_ROOT,
    )
    # The smoke matrix IS complete relative to its own configuration, which is
    # exactly the situation that previously produced an evidentiary decision.
    assert analysis["is_full_registered_matrix"] is True
    assert analysis["confirmatory_config"] is False
    assert analysis["registered_confirmatory_matrix"] is False
    assert analysis["decision"] == "smoke_only"
    assert analysis["gates"]["paired_initialization_and_artifact_integrity"]["pass"] is True


def test_confirmatory_analysis_refuses_to_run_without_the_dataset(
    smoke_bundle: dict[str, Path], tmp_path: Path
) -> None:
    study = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    study["confirmatory"] = True
    path = tmp_path / "confirmatory.json"
    path.write_text(json.dumps(study), encoding="utf-8")
    with pytest.raises(ValueError, match="requires --dataset-dir"):
        analyze_study(study_config_path=path, output_root=smoke_bundle["output_root"])


# --------------------------------------------------------------------------
# Negative tests: every mutation must break the chain
# --------------------------------------------------------------------------


def test_mutated_prediction_file_is_rejected(smoke_bundle: dict[str, Path], tmp_path: Path) -> None:
    bundle = _clone(smoke_bundle, tmp_path)
    path = bundle["output_root"] / "runs" / "learned_delta" / "seed_7" / "predictions_test__identity.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["correct"] = not rows[0]["correct"]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    report = _verify(bundle)
    assert report["passed"] is False
    assert any("prediction file is missing or its hash changed" in error for error in report["errors"])


def test_mutated_final_checkpoint_is_rejected(smoke_bundle: dict[str, Path], tmp_path: Path) -> None:
    bundle = _clone(smoke_bundle, tmp_path)
    path = bundle["output_root"] / "runs" / "cascade" / "seed_7" / "checkpoint_final.pt"
    path.write_bytes(path.read_bytes() + b"\x00")
    report = _verify(bundle)
    assert report["passed"] is False
    assert any("final checkpoint" in error for error in report["errors"])


def test_mutated_train_log_is_rejected(smoke_bundle: dict[str, Path], tmp_path: Path) -> None:
    bundle = _clone(smoke_bundle, tmp_path)
    path = bundle["output_root"] / "runs" / "history_control" / "seed_7" / "train_log.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"step": 999}\n', encoding="utf-8")
    report = _verify(bundle)
    assert report["passed"] is False
    assert any("train log" in error for error in report["errors"])


def test_mutated_dataset_is_rejected(smoke_bundle: dict[str, Path], tmp_path: Path) -> None:
    bundle = _clone(smoke_bundle, tmp_path)
    manifest_path = bundle["dataset_dir"] / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_seed"] = int(manifest["dataset_seed"]) + 1
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = _verify(bundle)
    assert report["passed"] is False
    assert any("materialized dataset" in error for error in report["errors"])


def test_config_drift_is_rejected(smoke_bundle: dict[str, Path], tmp_path: Path) -> None:
    study = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    study["analysis"]["minimum_history_delta"] = 0.5
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
    report = verify_evidence_chain(
        study_config_path=drifted,
        output_root=smoke_bundle["output_root"],
        dataset_dir=smoke_bundle["dataset_dir"],
        repo_root=REPO_ROOT,
    )
    assert report["passed"] is False
    assert any("different study configuration" in error for error in report["errors"])


def test_missing_registered_cell_is_rejected(smoke_bundle: dict[str, Path], tmp_path: Path) -> None:
    bundle = _clone(smoke_bundle, tmp_path)
    shutil.rmtree(bundle["output_root"] / "runs" / "cascade" / "seed_7")
    report = _verify(bundle)
    assert report["passed"] is False
    assert any("missing" in error for error in report["errors"])


def test_prediction_row_mislabelled_as_another_arm_is_rejected(
    smoke_bundle: dict[str, Path], tmp_path: Path
) -> None:
    """Hash-consistent but semantically wrong evidence must still fail."""

    bundle = _clone(smoke_bundle, tmp_path)
    run_dir = bundle["output_root"] / "runs" / "learned_delta" / "seed_7"
    path = run_dir / "predictions_test__identity.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["arm"] = "cascade"
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    # Re-record the hash so only the *semantic* check can catch this.
    import hashlib

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluations"]["test__identity"]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = _verify(bundle)
    assert report["passed"] is False
    assert any("disagree with the materialized split" in error for error in report["errors"])


def test_expected_confirmatory_flag_is_enforced(smoke_bundle: dict[str, Path]) -> None:
    report = verify_evidence_chain(
        study_config_path=SMOKE_CONFIG,
        output_root=smoke_bundle["output_root"],
        dataset_dir=smoke_bundle["dataset_dir"],
        repo_root=REPO_ROOT,
        require_confirmatory=True,
    )
    assert report["passed"] is False
    assert report["checks"]["confirmatory_expectation"] is False


# --------------------------------------------------------------------------
# Determinism: interrupted training must equal uninterrupted training
# --------------------------------------------------------------------------


def test_interrupted_and_uninterrupted_training_are_equivalent(
    smoke_bundle: dict[str, Path], tmp_path: Path
) -> None:
    study = load_study_config(SMOKE_CONFIG)
    dataset_dir = smoke_bundle["dataset_dir"]

    uninterrupted_root = tmp_path / "uninterrupted"
    reference = train_one_run(
        study=study,
        study_config_path=SMOKE_CONFIG,
        dataset_dir=dataset_dir,
        output_root=uninterrupted_root,
        repo_root=REPO_ROOT,
        arm="learned_delta",
        seed=7,
    )

    # Simulate a crash: train a truncated run, then resume it to the full step
    # count from its checkpoint.
    interrupted_root = tmp_path / "interrupted"
    partial_study = dict(study)
    partial_training = dict(study["training"])
    partial_training["steps"] = 2
    partial_study["training"] = partial_training
    train_one_run(
        study=partial_study,
        study_config_path=SMOKE_CONFIG,
        dataset_dir=dataset_dir,
        output_root=interrupted_root,
        repo_root=REPO_ROOT,
        arm="learned_delta",
        seed=7,
    )
    run_dir = interrupted_root / "runs" / "learned_delta" / "seed_7"
    # A crash leaves the checkpoint but not the completed manifest, and can
    # leave log rows past the checkpoint boundary.
    shutil.copyfile(run_dir / "checkpoint_final.pt", run_dir / "checkpoint_last.pt")
    (run_dir / "run_manifest.json").unlink()
    with (run_dir / "train_log.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"step": 3, "loss": 0.0, "stale": True}) + "\n")

    resumed = train_one_run(
        study=study,
        study_config_path=SMOKE_CONFIG,
        dataset_dir=dataset_dir,
        output_root=interrupted_root,
        repo_root=REPO_ROOT,
        arm="learned_delta",
        seed=7,
    )

    assert resumed["training"]["resumed_from_step"] == 2
    assert resumed["training"]["train_log_repair"]["truncated"] is True
    assert resumed["training"]["completed_steps"] == reference["training"]["completed_steps"]
    assert resumed["final_model_state_sha256"] == reference["final_model_state_sha256"]
    assert resumed["training"]["examples_seen"] == reference["training"]["examples_seen"]
    assert resumed["training"]["tokens_seen"] == reference["training"]["tokens_seen"]

    reference_predictions = (
        uninterrupted_root / "runs" / "learned_delta" / "seed_7" / "predictions_dev__identity.jsonl"
    ).read_text(encoding="utf-8")
    resumed_predictions = (run_dir / "predictions_dev__identity.jsonl").read_text(encoding="utf-8")
    assert resumed_predictions == reference_predictions

    steps = [
        json.loads(line)["step"]
        for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert steps == sorted(set(steps)), "resumed log is not a unique increasing step sequence"


def test_optimizer_state_survives_resume(smoke_bundle: dict[str, Path], tmp_path: Path) -> None:
    study = load_study_config(SMOKE_CONFIG)
    root = tmp_path / "opt"
    manifest = train_one_run(
        study=study,
        study_config_path=SMOKE_CONFIG,
        dataset_dir=smoke_bundle["dataset_dir"],
        output_root=root,
        repo_root=REPO_ROOT,
        arm="cascade",
        seed=7,
    )
    payload = torch.load(
        root / "runs" / "cascade" / "seed_7" / str(manifest["training"]["final_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["completed_steps"] == int(study["training"]["steps"])
    assert payload["optimizer_state"]["state"], "optimizer state was not preserved"
