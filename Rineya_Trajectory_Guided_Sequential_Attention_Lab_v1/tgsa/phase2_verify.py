"""Closed evidence-chain verifier for the TGSA x MTP Phase-2 study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, cast

from .phase1_competence import load_verdict
from .phase2_analysis import evaluate_complementarity_gates
from .phase2_data import sha256_file, validate_phase2_dataset
from .phase2_runner import (
    PHASE2_MATRIX_SCHEMA,
    PHASE2_RUN_SCHEMA,
    PRIMARY_ARMS,
    atomic_json,
    load_phase2_config,
)

VERIFICATION_SCHEMA = "tgsa.mtp-evidence-verification.v1"


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON document must be an object: {path}")
    return value


def verify_phase2_evidence(
    *,
    config_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Re-derive hashes and contracts, then emit the final complementarity gate."""

    errors: list[str] = []
    checks: dict[str, bool] = {}
    study = load_phase2_config(config_path)
    matrix_path = output_dir / "matrix_manifest.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    matrix = _load_object(matrix_path)
    checks["matrix_schema"] = matrix.get("schema_version") == PHASE2_MATRIX_SCHEMA
    if not checks["matrix_schema"]:
        errors.append("matrix manifest has the wrong schema")
    checks["matrix_complete"] = matrix.get("status") == "complete"
    if not checks["matrix_complete"]:
        errors.append("matrix manifest is not complete")
    checks["confirmatory_flag_matches"] = bool(matrix.get("confirmatory")) == bool(
        study["confirmatory"]
    )
    if not checks["confirmatory_flag_matches"]:
        errors.append("matrix confirmatory flag does not match the config")

    config_hash = sha256_file(config_path)
    checks["config_hash"] = matrix.get("config_sha256") == config_hash
    if not checks["config_hash"]:
        errors.append("config hash does not match matrix manifest")

    dataset_manifest_path = dataset_dir / "dataset_manifest.json"
    try:
        validate_phase2_dataset(dataset_manifest_path)
        checks["dataset_valid"] = True
    except Exception as error:  # noqa: BLE001 - verifier aggregates failures
        checks["dataset_valid"] = False
        errors.append(f"dataset validation failed: {type(error).__name__}: {error}")
    dataset_hash = sha256_file(dataset_manifest_path)
    checks["dataset_hash"] = matrix.get("dataset_manifest_sha256") == dataset_hash
    if not checks["dataset_hash"]:
        errors.append("dataset manifest hash does not match matrix manifest")

    expected = {
        (arm, int(seed))
        for arm in PRIMARY_ARMS
        for seed in cast(list[object], study["seeds"])
    }
    run_entries = matrix.get("runs")
    if not isinstance(run_entries, list):
        errors.append("matrix.runs is not a list")
        run_entries = []
    actual: set[tuple[str, int]] = set()
    run_documents: list[dict[str, object]] = []
    for entry in run_entries:
        if not isinstance(entry, Mapping):
            errors.append("matrix contains a non-object run entry")
            continue
        arm = str(entry.get("arm"))
        seed = int(entry.get("seed", -1))
        identity = (arm, seed)
        if identity in actual:
            errors.append(f"duplicate run entry: {identity}")
            continue
        actual.add(identity)
        relative = Path(str(entry.get("manifest")))
        path = output_dir / relative
        if not path.is_file():
            errors.append(f"run manifest is missing: {path}")
            continue
        run = _load_object(path)
        run_documents.append(run)
        if run.get("schema_version") != PHASE2_RUN_SCHEMA:
            errors.append(f"{identity}: wrong run schema")
        if run.get("status") != "complete":
            errors.append(f"{identity}: run is not complete")
        if str(run.get("arm")) != arm or int(run.get("seed", -1)) != seed:
            errors.append(f"{identity}: manifest identity mismatch")
        if run.get("config_sha256") != config_hash:
            errors.append(f"{identity}: config hash mismatch")
        if run.get("dataset_manifest_sha256") != dataset_hash:
            errors.append(f"{identity}: dataset hash mismatch")

        run_dir = path.parent
        training = run.get("training")
        if not isinstance(training, Mapping):
            errors.append(f"{identity}: missing training record")
        else:
            checkpoint = run_dir / str(training.get("final_checkpoint"))
            if not checkpoint.is_file():
                errors.append(f"{identity}: final checkpoint missing")
            elif sha256_file(checkpoint) != training.get("final_checkpoint_sha256"):
                errors.append(f"{identity}: final checkpoint hash mismatch")
            trace = run_dir / str(training.get("trace"))
            if not trace.is_file():
                errors.append(f"{identity}: training trace missing")
            elif sha256_file(trace) != training.get("trace_sha256"):
                errors.append(f"{identity}: training trace hash mismatch")
            if training.get("fixed_final_checkpoint") is not True:
                errors.append(f"{identity}: final checkpoint was not fixed by step count")
            if training.get("early_stopping") is not False:
                errors.append(f"{identity}: early stopping must be false")

        generation = run.get("generation_metrics")
        if not isinstance(generation, Mapping):
            errors.append(f"{identity}: missing generation metrics")
        elif generation.get("exact_greedy_equivalence") is not True:
            errors.append(f"{identity}: verified MTP output diverged from greedy")
        predictions = run_dir / str(run.get("generation_predictions"))
        if not predictions.is_file():
            errors.append(f"{identity}: generation predictions missing")
        elif sha256_file(predictions) != run.get("generation_predictions_sha256"):
            errors.append(f"{identity}: generation prediction hash mismatch")

    checks["primary_matrix_complete"] = expected.issubset(actual)
    if not checks["primary_matrix_complete"]:
        errors.append(f"primary matrix is incomplete: missing {sorted(expected - actual)}")
    checks["run_identity_unique"] = len(actual) == len(run_entries)
    if not checks["run_identity_unique"]:
        errors.append("run identities are not unique")

    init_root = output_dir / "initialization"
    initialization_ok = True
    for seed_value in cast(list[object], study["seeds"]):
        seed = int(seed_value)
        path = init_root / f"seed_{seed}" / "audit.json"
        if not path.is_file():
            errors.append(f"initialization audit missing for seed {seed}")
            initialization_ok = False
            continue
        audit = _load_object(path)
        audit_checks = audit.get("checks")
        if not isinstance(audit_checks, Mapping) or not all(bool(value) for value in audit_checks.values()):
            errors.append(f"initialization audit failed for seed {seed}")
            initialization_ok = False
    checks["paired_initialization"] = initialization_ok

    analysis_path = output_dir / "analysis.json"
    gate_inputs_path = output_dir / "gate_inputs.json"
    competence_path = output_dir / "competence_report.json"
    for label, path in (
        ("analysis", analysis_path),
        ("gate_inputs", gate_inputs_path),
        ("competence", competence_path),
    ):
        checks[f"{label}_present"] = path.is_file()
        if not path.is_file():
            errors.append(f"required artifact is missing: {path}")
    analysis = _load_object(analysis_path) if analysis_path.is_file() else {}
    gate_inputs = _load_object(gate_inputs_path) if gate_inputs_path.is_file() else {}
    competence: dict[str, object] = {}
    if competence_path.is_file():
        try:
            competence = load_verdict(competence_path)
            checks["competence_schema"] = True
        except Exception as error:  # noqa: BLE001
            checks["competence_schema"] = False
            errors.append(f"competence report invalid: {type(error).__name__}: {error}")
    else:
        checks["competence_schema"] = False

    quality = analysis.get("quality_main_accuracy")
    efficiency = analysis.get("efficiency_serial_rounds")
    checks["analysis_contract"] = isinstance(quality, Mapping) and isinstance(efficiency, Mapping)
    if not checks["analysis_contract"]:
        errors.append("analysis is missing primary quality or efficiency effects")

    checks["all_primary_decodes_exact"] = all(
        isinstance(run.get("generation_metrics"), Mapping)
        and cast(Mapping[str, object], run["generation_metrics"]).get(
            "exact_greedy_equivalence"
        )
        is True
        for run in run_documents
        if str(run.get("arm")) in PRIMARY_ARMS
    )
    if not checks["all_primary_decodes_exact"]:
        errors.append("not every primary run passed exact greedy equivalence")

    passed = not errors
    report: dict[str, object] = {
        "schema_version": VERIFICATION_SCHEMA,
        "passed": passed,
        "checks": dict(sorted(checks.items())),
        "errors": errors,
        "verified_run_count": len(run_documents),
        "config_sha256": config_hash,
        "dataset_manifest_sha256": dataset_hash,
        "claim_scope": (
            "artifact integrity and workflow contracts only; passing does not turn a smoke "
            "study into real-model evidence"
        ),
    }
    target_report = report_path or (output_dir / "evidence_verification.json")
    atomic_json(target_report, report)

    if isinstance(quality, Mapping) and isinstance(efficiency, Mapping):
        gate = evaluate_complementarity_gates(
            exact_decode_equivalence=bool(gate_inputs.get("exact_decode_equivalence", False)),
            main_path_initialization_neutral=bool(
                gate_inputs.get("main_path_initialization_neutral", False)
            ),
            evidence_integrity=passed,
            competence_go=competence.get("verdict") == "GO",
            quality_analysis=quality,
            efficiency_analysis=efficiency,
            quality_noninferiority_margin=float(
                gate_inputs.get("quality_noninferiority_margin", 0.0)
            ),
            minimum_serial_round_gain=float(gate_inputs.get("minimum_serial_round_gain", 0.0)),
            require_positive_interaction=bool(
                gate_inputs.get("require_positive_interaction", False)
            ),
        )
        gate["confirmatory_config"] = bool(study["confirmatory"])
        if not bool(study["confirmatory"]):
            gate["decision_scope"] = "SMOKE_ONLY"
            gate["verdict"] = "SMOKE_ONLY"
        atomic_json(output_dir / "gate.json", gate)
        report["gate"] = gate
        atomic_json(target_report, report)
    return report
