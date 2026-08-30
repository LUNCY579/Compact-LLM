"""Fail-closed integrity contract for TGSA × MTP development and confirmatory runs.

The Phase-1 v4 audit established three rules that Phase 2 must inherit:

* development evidence can never be laundered into confirmatory evidence;
* data/order/device/training-dose identity belongs in every cell fingerprint;
* competence is a prerequisite for interpreting a mechanism contrast, not a
  favorable metric produced by the contrast itself.

This module is deliberately model-agnostic.  Both Phase-2 implementations call
it before dispatch and attach its evidence after their native runner completes.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from .phase1_competence import require_go

INTEGRITY_SCHEMA_VERSION = "tgsa.phase2-integrity.v1"
INTEGRITY_VERIFICATION_VERSION = "tgsa.phase2-integrity-verification.v1"
EXPECTED_SPLIT_GENERATION_ORDER: tuple[str, ...] = ("test", "dev", "train")


class Phase2IntegrityError(ValueError):
    """Raised when a Phase-2 evidence contract is incomplete or inconsistent."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase2IntegrityError(f"expected a JSON object at {path}")
    return value


def _resolve_reference(value: str, *, base: Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _normalized_order(value: object) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return [str(item) for item in value]


def _external_competence_value(study: Mapping[str, object]) -> object:
    registration = study.get("registration")
    if isinstance(registration, Mapping):
        candidate = registration.get("phase1_competence_report")
        if candidate not in {None, ""}:
            return candidate
    return study.get("competence_report")


def validate_phase2_contract(
    study: Mapping[str, object],
    *,
    config_path: Path,
) -> dict[str, object]:
    """Validate the marker/split/competence contract before any output is made."""

    if "development_only" not in study:
        raise Phase2IntegrityError(
            "Phase-2 configs must explicitly set development_only=true|false"
        )
    development_only = study["development_only"]
    if not isinstance(development_only, bool):
        raise Phase2IntegrityError("development_only must be an explicit boolean")
    confirmatory = study.get("confirmatory")
    if not isinstance(confirmatory, bool):
        raise Phase2IntegrityError("confirmatory must be an explicit boolean")
    if confirmatory and development_only:
        raise Phase2IntegrityError(
            "a confirmatory Phase-2 config cannot be marked development_only"
        )
    if not confirmatory and not development_only:
        raise Phase2IntegrityError(
            "a non-confirmatory Phase-2 config must be marked development_only"
        )

    dataset = study.get("dataset")
    if not isinstance(dataset, Mapping):
        raise Phase2IntegrityError("dataset must be a JSON object")
    order = _normalized_order(dataset.get("split_generation_order"))
    if order != list(EXPECTED_SPLIT_GENERATION_ORDER):
        raise Phase2IntegrityError(
            "Phase-2 requires split_generation_order=['test','dev','train']; "
            "the test bank must be the unbiased first draw"
        )

    training = study.get("training")
    runtime = study.get("runtime")
    if not isinstance(training, Mapping) or not isinstance(runtime, Mapping):
        raise Phase2IntegrityError("training and runtime must be JSON objects")
    gradient_accumulation = int(training.get("gradient_accumulation_steps", 1))
    world_size = int(runtime.get("world_size", 1))
    if gradient_accumulation != 1 or world_size != 1:
        raise Phase2IntegrityError(
            "the current Phase-2 runners execute one microbatch per optimizer step "
            "on one process; non-unit gradient_accumulation_steps/world_size would "
            "make the recorded sample-presentation dose false"
        )

    registration = study.get("registration")
    external: dict[str, object] = {
        "required": bool(confirmatory),
        "path": None,
        "sha256": None,
        "verdict": "NOT_REQUIRED_FOR_DEVELOPMENT",
        "development_only": None,
    }
    if confirmatory:
        if not isinstance(registration, Mapping):
            raise Phase2IntegrityError(
                "confirmatory Phase-2 configs require a registration object"
            )
        if str(registration.get("status")) != "frozen":
            raise Phase2IntegrityError(
                "confirmatory Phase-2 requires registration.status='frozen'"
            )
        reference = _external_competence_value(study)
        if not isinstance(reference, str) or not reference.strip():
            raise Phase2IntegrityError(
                "confirmatory Phase-2 requires an external frozen Phase-1 "
                "competence report"
            )
        report_path = _resolve_reference(reference, base=Path(config_path).parent)
        # Current v4 require_go rejects development-only evidence by default.
        report = require_go(report_path)
        if bool(report.get("development_only", False)):
            raise Phase2IntegrityError(
                "development-only competence evidence cannot certify Phase 2"
            )
        external = {
            "required": True,
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "verdict": str(report.get("verdict")),
            "schema_version": report.get("schema_version"),
            "development_only": False,
            "regime_fingerprint": (
                cast(Mapping[str, object], report.get("regime", {})).get(
                    "fingerprint"
                )
                if isinstance(report.get("regime"), Mapping)
                else None
            ),
        }
    return {
        "development_only": development_only,
        "confirmatory": confirmatory,
        "split_generation_order": list(EXPECTED_SPLIT_GENERATION_ORDER),
        "external_phase1_competence": external,
    }


def _resolve_device(runtime: Mapping[str, object]) -> torch.device:
    requested = str(runtime.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def capture_device_identity(runtime: Mapping[str, object]) -> dict[str, object]:
    """Return the stable training-device payload used in every cell fingerprint."""

    device = _resolve_device(runtime)
    identity: dict[str, object] = {
        "requested_device": str(runtime.get("device", "auto")),
        "resolved_device": str(device),
        "device_type": device.type,
        "device_index": device.index,
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
        "deterministic_algorithms_requested": bool(
            runtime.get("deterministic_algorithms", True)
        ),
        "num_threads_requested": int(runtime.get("num_threads", 0)),
        "world_size": int(runtime.get("world_size", 1)),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
    }
    if device.type == "cuda":
        index = int(device.index if device.index is not None else 0)
        properties = torch.cuda.get_device_properties(index)
        identity["accelerator"] = {
            "name": properties.name,
            "capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": int(properties.multi_processor_count),
        }
    else:
        identity["accelerator"] = None
    return identity


def validate_dataset_contract(manifest: Mapping[str, object]) -> int:
    order = _normalized_order(manifest.get("split_generation_order"))
    if order != list(EXPECTED_SPLIT_GENERATION_ORDER):
        raise Phase2IntegrityError(
            "materialized dataset does not preserve test/dev/train generation order"
        )
    if manifest.get("unbiased_split") not in {None, "test"}:
        raise Phase2IntegrityError("the materialized unbiased split is not test")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise Phase2IntegrityError("dataset manifest has no splits object")
    train = splits.get("train")
    if not isinstance(train, Mapping):
        raise Phase2IntegrityError("dataset manifest has no train split")
    train_examples = int(train.get("examples", 0))
    if train_examples <= 0:
        raise Phase2IntegrityError("dataset manifest reports no training examples")
    return train_examples


def training_dose(
    study: Mapping[str, object],
    *,
    unique_train_examples: int,
) -> dict[str, object]:
    training = cast(Mapping[str, object], study["training"])
    steps = int(training["steps"])
    batch_size = int(training["batch_size"])
    if steps <= 0 or batch_size <= 0 or unique_train_examples <= 0:
        raise Phase2IntegrityError("training dose inputs must be positive")
    sample_presentations = steps * batch_size
    effective_epochs = sample_presentations / unique_train_examples
    return {
        "optimizer_steps": steps,
        "microbatch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "world_size": 1,
        "unique_train_examples": unique_train_examples,
        "sample_presentations": sample_presentations,
        "effective_epochs": effective_epochs,
        "presentations_per_unique_example": effective_epochs,
        "fixed_compute_axis": (
            "sample presentations and unique examples are recorded separately; "
            "equal presentations do not imply equal data diversity"
        ),
    }


def _json_copy(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def build_integrity_context(
    *,
    study: Mapping[str, object],
    config_path: Path,
    dataset_manifest_path: Path,
    implementation: str,
    device_identity_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    contract = validate_phase2_contract(study, config_path=config_path)
    manifest = _load_json(dataset_manifest_path)
    unique_train_examples = validate_dataset_contract(manifest)
    dose = training_dose(study, unique_train_examples=unique_train_examples)
    device = (
        dict(device_identity_override)
        if device_identity_override is not None
        else capture_device_identity(cast(Mapping[str, object], study["runtime"]))
    )
    config_sha = sha256_file(config_path)
    dataset_sha = sha256_file(dataset_manifest_path)

    regime_payload = {
        "schema_version": "tgsa.phase2-regime-fingerprint.v1",
        "implementation": implementation,
        "study_schema_version": study.get("schema_version"),
        "dataset": _json_copy(study.get("dataset")),
        "model": _json_copy(study.get("model")),
        "mtp": _json_copy(study.get("mtp")),
        "training": _json_copy(study.get("training")),
        "evaluation": _json_copy(study.get("evaluation")),
        "cells": _json_copy(study.get("cells")),
        "tgsa_arm": study.get("tgsa_arm"),
        "include_auxiliary_controls": study.get("include_auxiliary_controls"),
        "split_generation_order": list(EXPECTED_SPLIT_GENERATION_ORDER),
        "training_dose": dose,
    }
    regime_fingerprint = sha256_json(regime_payload)
    device_fingerprint = sha256_json(device)
    external = cast(
        Mapping[str, object], contract["external_phase1_competence"]
    )
    study_payload = {
        "schema_version": "tgsa.phase2-study-fingerprint.v1",
        "implementation": implementation,
        "config_sha256": config_sha,
        "dataset_manifest_sha256": dataset_sha,
        "regime_fingerprint": regime_fingerprint,
        "device_fingerprint": device_fingerprint,
        "seeds": [int(seed) for seed in cast(Sequence[object], study["seeds"])],
        "confirmatory": bool(study["confirmatory"]),
        "development_only": bool(study["development_only"]),
        "external_phase1_competence_sha256": external.get("sha256"),
    }
    return {
        "schema_version": INTEGRITY_SCHEMA_VERSION,
        "implementation": implementation,
        "contract": contract,
        "config_sha256": config_sha,
        "dataset_manifest_sha256": dataset_sha,
        "training_dose": dose,
        "device_identity": device,
        "device_fingerprint": device_fingerprint,
        "regime_payload": regime_payload,
        "regime_fingerprint": regime_fingerprint,
        "study_payload": study_payload,
        "study_fingerprint": sha256_json(study_payload),
    }


def _cell_identity(run: Mapping[str, object]) -> str:
    for key in ("cell_key", "arm"):
        value = run.get(key)
        if isinstance(value, str) and value:
            return value
    cell = run.get("cell")
    if isinstance(cell, Mapping):
        for key in ("key", "name"):
            value = cell.get(key)
            if isinstance(value, str) and value:
                return value
    raise Phase2IntegrityError("run manifest has no cell/arm identity")


def _augment_proxy_semantics(document: MutableMapping[str, object]) -> None:
    semantics = {
        "serial_round_metrics": (
            "algorithmic target-forward amortization proxy; not wall-clock "
            "latency or tokens-per-second"
        ),
        "wall_clock_speed_claim": False,
        "requires_future_cache_aware_benchmark": True,
    }
    document["metric_semantics"] = semantics
    generation = document.get("generation_metrics")
    if isinstance(generation, MutableMapping):
        if "serial_round_speedup_vs_ar" in generation:
            generation["serial_round_amortization_factor_vs_ar"] = generation[
                "serial_round_speedup_vs_ar"
            ]
        if "serial_round_reduction_fraction_vs_ar" in generation:
            generation["serial_round_reduction_proxy_vs_ar"] = generation[
                "serial_round_reduction_fraction_vs_ar"
            ]
        generation["wall_clock_speed_claim"] = False


def attach_phase2_integrity(
    *,
    config_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    implementation: str,
) -> dict[str, object]:
    """Attach fingerprints and claim guards to a completed native Phase-2 run."""

    config_path = Path(config_path).resolve()
    dataset_manifest_path = Path(dataset_dir).resolve() / "dataset_manifest.json"
    output_dir = Path(output_dir).resolve()
    study = _load_json(config_path)
    context = build_integrity_context(
        study=study,
        config_path=config_path,
        dataset_manifest_path=dataset_manifest_path,
        implementation=implementation,
    )
    contract = cast(Mapping[str, object], context["contract"])
    external = cast(
        Mapping[str, object], contract["external_phase1_competence"]
    )
    mechanism_allowed = bool(
        contract["confirmatory"]
        and not contract["development_only"]
        and external.get("verdict") == "GO"
        and not external.get("development_only", False)
    )

    run_paths = sorted(output_dir.rglob("run_manifest.json"))
    if not run_paths:
        raise Phase2IntegrityError("completed Phase-2 output has no run manifests")
    run_hashes: dict[str, str] = {}
    for run_path in run_paths:
        run = _load_json(run_path)
        cell = _cell_identity(run)
        seed = int(run["seed"])
        cell_payload = {
            "schema_version": "tgsa.phase2-cell-fingerprint.v1",
            "study_fingerprint": context["study_fingerprint"],
            "cell": cell,
            "seed": seed,
        }
        run["development_only"] = bool(contract["development_only"])
        run["mechanism_interpretation_allowed"] = mechanism_allowed
        run["training_dose"] = context["training_dose"]
        run["integrity"] = {
            "schema_version": INTEGRITY_SCHEMA_VERSION,
            "config_sha256": context["config_sha256"],
            "dataset_manifest_sha256": context["dataset_manifest_sha256"],
            "regime_fingerprint": context["regime_fingerprint"],
            "device_fingerprint": context["device_fingerprint"],
            "study_fingerprint": context["study_fingerprint"],
            "cell_fingerprint": sha256_json(cell_payload),
        }
        _augment_proxy_semantics(run)
        scope = str(run.get("claim_scope", "")).strip()
        suffix = (
            " Development-only evidence cannot support a TGSA, MTP, "
            "complementarity, or wall-clock speed claim."
            if contract["development_only"]
            else " Mechanism interpretation still depends on the registered "
            "confirmatory estimators and gates."
        )
        run["claim_scope"] = scope + suffix
        _atomic_json(run_path, run)
        relative = run_path.relative_to(output_dir).as_posix()
        run_hashes[relative] = sha256_file(run_path)

    supporting_hashes: dict[str, str] = {}
    for name in (
        "analysis.json",
        "competence_report.json",
        "gate_inputs.json",
        "gate.json",
    ):
        path = output_dir / name
        if not path.is_file():
            continue
        document = _load_json(path)
        document["development_only"] = bool(contract["development_only"])
        document["mechanism_interpretation_allowed"] = mechanism_allowed
        document["training_dose"] = context["training_dose"]
        document["regime_fingerprint"] = context["regime_fingerprint"]
        document["device_fingerprint"] = context["device_fingerprint"]
        if name == "competence_report.json" and contract["development_only"]:
            document["certifies_confirmatory"] = False
            document["provenance"] = "phase2_internal_development"
        if name == "analysis.json":
            document["v4_trend_context"] = {
                "interpretation": (
                    "training exposure and data diversity are regime variables, "
                    "not evidence for or against the TGSA mechanism"
                ),
                "effective_epochs": cast(
                    Mapping[str, object], context["training_dose"]
                )["effective_epochs"],
                "unique_train_examples": cast(
                    Mapping[str, object], context["training_dose"]
                )["unique_train_examples"],
                "sample_presentations": cast(
                    Mapping[str, object], context["training_dose"]
                )["sample_presentations"],
            }
            _augment_proxy_semantics(document)
        _atomic_json(path, document)
        supporting_hashes[name] = sha256_file(path)

    top_level_path: Path | None = None
    for candidate in ("phase2_report.json", "matrix_manifest.json"):
        path = output_dir / candidate
        if path.is_file():
            top_level_path = path
            break
    if top_level_path is None:
        raise Phase2IntegrityError(
            "completed Phase-2 output has no phase2_report.json or matrix_manifest.json"
        )

    integrity_manifest = {
        **context,
        "development_only": bool(contract["development_only"]),
        "confirmatory": bool(contract["confirmatory"]),
        "mechanism_interpretation_allowed": mechanism_allowed,
        "claim_scope": (
            "integrity and prerequisite evidence only; no TGSA/MTP mechanism or "
            "wall-clock result is created by this manifest"
        ),
        "run_manifest_hashes": run_hashes,
        "supporting_artifact_hashes": supporting_hashes,
        "top_level_artifact": top_level_path.name,
    }
    integrity_path = output_dir / "phase2_integrity.json"
    _atomic_json(integrity_path, integrity_manifest)
    integrity_sha = sha256_file(integrity_path)

    top_level = _load_json(top_level_path)
    inline_runs = top_level.get("runs")
    if isinstance(inline_runs, list):
        for item in inline_runs:
            if not isinstance(item, MutableMapping):
                continue
            try:
                cell = _cell_identity(item)
                seed = int(item["seed"])
            except (KeyError, TypeError, ValueError, Phase2IntegrityError):
                # Matrix manifests may contain run references rather than inline
                # run records; their authoritative manifests are hashed above.
                continue
            cell_payload = {
                "schema_version": "tgsa.phase2-cell-fingerprint.v1",
                "study_fingerprint": context["study_fingerprint"],
                "cell": cell,
                "seed": seed,
            }
            item["development_only"] = bool(contract["development_only"])
            item["mechanism_interpretation_allowed"] = mechanism_allowed
            item["training_dose"] = context["training_dose"]
            item["integrity"] = {
                "schema_version": INTEGRITY_SCHEMA_VERSION,
                "config_sha256": context["config_sha256"],
                "dataset_manifest_sha256": context["dataset_manifest_sha256"],
                "regime_fingerprint": context["regime_fingerprint"],
                "device_fingerprint": context["device_fingerprint"],
                "study_fingerprint": context["study_fingerprint"],
                "cell_fingerprint": sha256_json(cell_payload),
            }
            _augment_proxy_semantics(item)
    top_level["development_only"] = bool(contract["development_only"])
    top_level["mechanism_interpretation_allowed"] = mechanism_allowed
    top_level["training_dose"] = context["training_dose"]
    top_level["regime_fingerprint"] = context["regime_fingerprint"]
    top_level["device_fingerprint"] = context["device_fingerprint"]
    top_level["study_fingerprint"] = context["study_fingerprint"]
    top_level["external_phase1_competence"] = external
    top_level["integrity_manifest"] = integrity_path.name
    top_level["integrity_manifest_sha256"] = integrity_sha
    _augment_proxy_semantics(top_level)
    _atomic_json(top_level_path, top_level)
    return top_level


def verify_phase2_integrity(
    *,
    config_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Verify fingerprints without depending on the verifier machine's device."""

    config_path = Path(config_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    integrity_path = output_dir / "phase2_integrity.json"
    errors: list[str] = []
    checks: dict[str, bool] = {}

    try:
        integrity = _load_json(integrity_path)
        study = _load_json(config_path)
        dataset_manifest_path = dataset_dir / "dataset_manifest.json"
        stored_device = integrity.get("device_identity")
        if not isinstance(stored_device, Mapping):
            raise Phase2IntegrityError("integrity manifest has no device identity")
        expected = build_integrity_context(
            study=study,
            config_path=config_path,
            dataset_manifest_path=dataset_manifest_path,
            implementation=str(integrity.get("implementation")),
            device_identity_override=stored_device,
        )

        checks["schema_version"] = (
            integrity.get("schema_version") == INTEGRITY_SCHEMA_VERSION
        )
        checks["config_hash"] = (
            integrity.get("config_sha256") == expected["config_sha256"]
        )
        checks["dataset_hash"] = (
            integrity.get("dataset_manifest_sha256")
            == expected["dataset_manifest_sha256"]
        )
        checks["training_dose"] = (
            integrity.get("training_dose") == expected["training_dose"]
        )
        checks["regime_fingerprint"] = (
            integrity.get("regime_fingerprint")
            == expected["regime_fingerprint"]
        )
        # Device identity is verified internally, not against the machine doing
        # verification; a CPU verifier may inspect a CUDA-produced bundle.
        checks["device_fingerprint"] = (
            isinstance(stored_device, Mapping)
            and sha256_json(stored_device) == integrity.get("device_fingerprint")
        )
        stored_study_payload = integrity.get("study_payload")
        checks["study_fingerprint"] = (
            isinstance(stored_study_payload, Mapping)
            and sha256_json(stored_study_payload)
            == integrity.get("study_fingerprint")
            and integrity.get("study_fingerprint")
            == expected["study_fingerprint"]
        )
        contract = cast(Mapping[str, object], expected["contract"])
        checks["development_marker"] = (
            integrity.get("development_only")
            == contract.get("development_only")
        )
        checks["mechanism_gate"] = (
            integrity.get("mechanism_interpretation_allowed")
            == bool(
                contract.get("confirmatory")
                and not contract.get("development_only")
                and cast(
                    Mapping[str, object],
                    contract["external_phase1_competence"],
                ).get("verdict")
                == "GO"
            )
        )

        top_level_path = output_dir / str(integrity.get("top_level_artifact"))
        top_level = _load_json(top_level_path)
        checks["top_level_integrity_reference"] = (
            top_level.get("integrity_manifest") == integrity_path.name
            and top_level.get("integrity_manifest_sha256")
            == sha256_file(integrity_path)
            and top_level.get("study_fingerprint")
            == integrity.get("study_fingerprint")
        )

        run_hashes = integrity.get("run_manifest_hashes")
        checks["run_hashes"] = isinstance(run_hashes, Mapping)
        if isinstance(run_hashes, Mapping):
            for relative, expected_hash in run_hashes.items():
                path = output_dir / str(relative)
                if not path.is_file() or sha256_file(path) != expected_hash:
                    checks["run_hashes"] = False
                    errors.append(f"run manifest hash mismatch: {relative}")
                    continue
                run = _load_json(path)
                cell = _cell_identity(run)
                seed = int(run["seed"])
                cell_payload = {
                    "schema_version": "tgsa.phase2-cell-fingerprint.v1",
                    "study_fingerprint": integrity["study_fingerprint"],
                    "cell": cell,
                    "seed": seed,
                }
                run_integrity = run.get("integrity")
                if not isinstance(run_integrity, Mapping):
                    checks["run_hashes"] = False
                    errors.append(f"run has no integrity object: {relative}")
                    continue
                if (
                    run_integrity.get("cell_fingerprint")
                    != sha256_json(cell_payload)
                    or run_integrity.get("device_fingerprint")
                    != integrity.get("device_fingerprint")
                    or run_integrity.get("regime_fingerprint")
                    != integrity.get("regime_fingerprint")
                    or run.get("development_only")
                    != integrity.get("development_only")
                ):
                    checks["run_hashes"] = False
                    errors.append(f"run fingerprint mismatch: {relative}")
                semantics = run.get("metric_semantics")
                if not isinstance(semantics, Mapping) or semantics.get(
                    "wall_clock_speed_claim"
                ) is not False:
                    checks["run_hashes"] = False
                    errors.append(f"run lacks proxy-only speed semantics: {relative}")

        supporting = integrity.get("supporting_artifact_hashes")
        checks["supporting_hashes"] = isinstance(supporting, Mapping)
        if isinstance(supporting, Mapping):
            for relative, expected_hash in supporting.items():
                path = output_dir / str(relative)
                if not path.is_file() or sha256_file(path) != expected_hash:
                    checks["supporting_hashes"] = False
                    errors.append(f"supporting artifact hash mismatch: {relative}")

        competence_path = output_dir / "competence_report.json"
        checks["development_competence_not_certifying"] = True
        if integrity.get("development_only") and competence_path.is_file():
            competence = _load_json(competence_path)
            checks["development_competence_not_certifying"] = bool(
                competence.get("development_only") is True
                and competence.get("certifies_confirmatory") is False
            )
    except Exception as error:  # noqa: BLE001 - verifier reports all failures
        errors.append(f"{type(error).__name__}: {error}")

    for name, passed in sorted(checks.items()):
        if not passed and not any(name in error for error in errors):
            errors.append(f"failed integrity check: {name}")
    report = {
        "schema_version": INTEGRITY_VERIFICATION_VERSION,
        "passed": not errors and bool(checks) and all(checks.values()),
        "checks": dict(sorted(checks.items())),
        "errors": errors,
        "claim_scope": "integrity verification only",
    }
    destination = (
        Path(report_path).resolve()
        if report_path is not None
        else output_dir / "phase2_integrity_verification.json"
    )
    _atomic_json(destination, report)
    return report


__all__ = [
    "EXPECTED_SPLIT_GENERATION_ORDER",
    "INTEGRITY_SCHEMA_VERSION",
    "Phase2IntegrityError",
    "attach_phase2_integrity",
    "build_integrity_context",
    "capture_device_identity",
    "sha256_file",
    "sha256_json",
    "training_dose",
    "validate_dataset_contract",
    "validate_phase2_contract",
    "verify_phase2_integrity",
]
