"""Integrity helpers for real-checkpoint TGSA + MTP adapter warm starts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

RETROFIT_INTEGRITY_SCHEMA = "rineya.tgsa-mtp-training-integrity.v1"
RETROFIT_VERIFICATION_SCHEMA = "rineya.tgsa-mtp-training-integrity-verification.v1"


class RetrofitIntegrityError(ValueError):
    pass


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


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RetrofitIntegrityError(f"expected a JSON object at {path}")
    return value


def source_tree_manifest(model_dir: Path) -> dict[str, object]:
    """Hash every regular source-model file, following snapshot symlinks."""

    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    entries: list[dict[str, object]] = []
    total_bytes = 0
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(model_dir).as_posix()
        size = int(path.stat().st_size)
        total_bytes += size
        entries.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise RetrofitIntegrityError(f"source model directory is empty: {model_dir}")
    return {
        "model_dir": str(model_dir),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
        "fingerprint": sha256_json(entries),
    }


def capture_device_identity(device_value: str) -> dict[str, object]:
    device = torch.device(device_value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = int(device.index if device.index is not None else torch.cuda.current_device())
        device = torch.device("cuda", index)
        properties = torch.cuda.get_device_properties(index)
        accelerator: dict[str, object] | None = {
            "name": properties.name,
            "capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": int(properties.multi_processor_count),
        }
    else:
        accelerator = None
    return {
        "resolved_device": str(device),
        "device_type": device.type,
        "device_index": device.index,
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
        "accelerator": accelerator,
    }


def training_dose(
    *,
    rows: int,
    optimizer_steps: int,
    microbatch_size: int,
    gradient_accumulation: int,
) -> dict[str, object]:
    if min(rows, optimizer_steps, microbatch_size, gradient_accumulation) <= 0:
        raise RetrofitIntegrityError("training dose values must be positive")
    presentations = optimizer_steps * microbatch_size * gradient_accumulation
    effective_epochs = presentations / rows
    return {
        "unique_train_rows": rows,
        "optimizer_steps": optimizer_steps,
        "microbatch_size": microbatch_size,
        "gradient_accumulation": gradient_accumulation,
        "sample_presentations": presentations,
        "effective_epochs": effective_epochs,
        "presentations_per_unique_row": effective_epochs,
    }


def attach_training_integrity(
    *,
    output_dir: Path,
    source_manifest: Mapping[str, object],
    device_identity: Mapping[str, object],
    invocation: Mapping[str, object],
) -> dict[str, object]:
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "training_manifest.json"
    manifest = load_json(manifest_path)
    dose = training_dose(
        rows=int(manifest["rows"]),
        optimizer_steps=int(manifest["steps"]),
        microbatch_size=int(manifest["batch_size"]),
        gradient_accumulation=int(manifest["gradient_accumulation"]),
    )
    train_jsonl = Path(str(manifest["train_jsonl"])).resolve()
    train_hash = sha256_file(train_jsonl)
    if train_hash != manifest.get("train_jsonl_sha256"):
        raise RetrofitIntegrityError(
            "training JSONL changed between native training and integrity attachment"
        )
    source_fingerprint = str(source_manifest["fingerprint"])
    device_fingerprint = sha256_json(device_identity)
    regime_payload = {
        "schema_version": "rineya.tgsa-mtp-training-regime.v1",
        "source_model_fingerprint": source_fingerprint,
        "train_jsonl_sha256": train_hash,
        "converted_layers": manifest.get("converted_layers"),
        "stage": manifest.get("stage"),
        "steps": manifest.get("steps"),
        "batch_size": manifest.get("batch_size"),
        "gradient_accumulation": manifest.get("gradient_accumulation"),
        "learning_rate": manifest.get("learning_rate"),
        "seed": manifest.get("seed"),
        "conversion": manifest.get("conversion"),
        "training_dose": dose,
    }
    regime_fingerprint = sha256_json(regime_payload)
    run_payload = {
        "schema_version": "rineya.tgsa-mtp-training-run.v1",
        "regime_fingerprint": regime_fingerprint,
        "device_fingerprint": device_fingerprint,
        "invocation": dict(invocation),
    }
    run_fingerprint = sha256_json(run_payload)

    source_path = output_dir / "source_model_manifest.json"
    atomic_json(source_path, dict(source_manifest))
    artifacts = {
        "adapter_checkpoint": {
            "path": str(manifest["adapter_checkpoint"]),
            "sha256": sha256_file(output_dir / str(manifest["adapter_checkpoint"])),
        },
        "training_trace": {
            "path": str(manifest["training_trace"]),
            "sha256": sha256_file(output_dir / str(manifest["training_trace"])),
        },
        "source_model_manifest": {
            "path": source_path.name,
            "sha256": sha256_file(source_path),
        },
    }
    integrity = {
        "schema_version": RETROFIT_INTEGRITY_SCHEMA,
        "development_only": True,
        "confirmatory": False,
        "mechanism_interpretation_allowed": False,
        "source_model_fingerprint": source_fingerprint,
        "train_jsonl_sha256": train_hash,
        "training_dose": dose,
        "device_identity": dict(device_identity),
        "device_fingerprint": device_fingerprint,
        "regime_payload": regime_payload,
        "regime_fingerprint": regime_fingerprint,
        "run_payload": run_payload,
        "run_fingerprint": run_fingerprint,
        "artifacts": artifacts,
        "claim_scope": (
            "real-checkpoint adapter warm-start integrity only; no held-out "
            "quality, TGSA/MTP complementarity, or wall-clock speed result"
        ),
    }
    integrity_path = output_dir / "training_integrity.json"
    atomic_json(integrity_path, integrity)

    manifest["development_only"] = True
    manifest["confirmatory"] = False
    manifest["mechanism_interpretation_allowed"] = False
    manifest["training_dose"] = dose
    manifest["source_model_fingerprint"] = source_fingerprint
    manifest["device_fingerprint"] = device_fingerprint
    manifest["regime_fingerprint"] = regime_fingerprint
    manifest["run_fingerprint"] = run_fingerprint
    manifest["training_integrity"] = integrity_path.name
    manifest["training_integrity_sha256"] = sha256_file(integrity_path)
    manifest["claim_scope"] = (
        str(manifest.get("claim_scope", "")).strip()
        + " Development-only warm-start evidence cannot certify Phase 2."
    )
    atomic_json(manifest_path, manifest)
    return manifest


def verify_training_integrity(
    *,
    output_dir: Path,
    report_path: Path | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir).resolve()
    errors: list[str] = []
    checks: dict[str, bool] = {}
    try:
        manifest = load_json(output_dir / "training_manifest.json")
        integrity = load_json(output_dir / "training_integrity.json")
        source = load_json(output_dir / "source_model_manifest.json")
        checks["schema"] = integrity.get("schema_version") == RETROFIT_INTEGRITY_SCHEMA
        checks["development_only"] = bool(
            manifest.get("development_only") is True
            and integrity.get("development_only") is True
            and manifest.get("mechanism_interpretation_allowed") is False
        )
        checks["source_fingerprint"] = (
            sha256_json(source.get("files")) == integrity.get("source_model_fingerprint")
            == manifest.get("source_model_fingerprint")
        )
        stored_device = integrity.get("device_identity")
        checks["device_fingerprint"] = bool(
            isinstance(stored_device, Mapping)
            and sha256_json(stored_device) == integrity.get("device_fingerprint")
            == manifest.get("device_fingerprint")
        )
        stored_regime = integrity.get("regime_payload")
        checks["regime_fingerprint"] = bool(
            isinstance(stored_regime, Mapping)
            and sha256_json(stored_regime) == integrity.get("regime_fingerprint")
            == manifest.get("regime_fingerprint")
        )
        stored_run = integrity.get("run_payload")
        checks["run_fingerprint"] = bool(
            isinstance(stored_run, Mapping)
            and sha256_json(stored_run) == integrity.get("run_fingerprint")
            == manifest.get("run_fingerprint")
        )
        expected_dose = training_dose(
            rows=int(manifest["rows"]),
            optimizer_steps=int(manifest["steps"]),
            microbatch_size=int(manifest["batch_size"]),
            gradient_accumulation=int(manifest["gradient_accumulation"]),
        )
        checks["training_dose"] = (
            expected_dose == integrity.get("training_dose")
            == manifest.get("training_dose")
        )
        checks["integrity_reference"] = (
            manifest.get("training_integrity") == "training_integrity.json"
            and manifest.get("training_integrity_sha256")
            == sha256_file(output_dir / "training_integrity.json")
        )
        artifacts = integrity.get("artifacts")
        checks["artifact_hashes"] = isinstance(artifacts, Mapping)
        if isinstance(artifacts, Mapping):
            for name, entry in artifacts.items():
                if not isinstance(entry, Mapping):
                    checks["artifact_hashes"] = False
                    errors.append(f"malformed artifact entry: {name}")
                    continue
                path = output_dir / str(entry["path"])
                if not path.is_file() or sha256_file(path) != entry.get("sha256"):
                    checks["artifact_hashes"] = False
                    errors.append(f"artifact hash mismatch: {name}")
    except Exception as error:  # noqa: BLE001
        errors.append(f"{type(error).__name__}: {error}")

    for name, passed in sorted(checks.items()):
        if not passed and not any(name in item for item in errors):
            errors.append(f"failed integrity check: {name}")
    report = {
        "schema_version": RETROFIT_VERIFICATION_SCHEMA,
        "passed": not errors and bool(checks) and all(checks.values()),
        "checks": dict(sorted(checks.items())),
        "errors": errors,
        "claim_scope": "adapter warm-start integrity verification only",
    }
    destination = (
        Path(report_path).resolve()
        if report_path is not None
        else output_dir / "training_integrity_verification.json"
    )
    atomic_json(destination, report)
    return report


__all__ = [
    "RETROFIT_INTEGRITY_SCHEMA",
    "RetrofitIntegrityError",
    "attach_training_integrity",
    "capture_device_identity",
    "sha256_file",
    "sha256_json",
    "source_tree_manifest",
    "training_dose",
    "verify_training_integrity",
]
