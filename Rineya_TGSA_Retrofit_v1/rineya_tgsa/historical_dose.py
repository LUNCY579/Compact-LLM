"""Exact streaming LoRA-dose reconstruction for the frozen rank-2 trial."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
SOUP_SRC = REPO / "Rineya_Layerwise_Soup_Lab_v1.0" / "src"

DOSE_TRIAL_SCHEMA_VERSION = "rineya.historical-dose-transfer-trial.v1"
DOSE_ARTIFACT_SCHEMA_VERSION = "rineya.historical-dose-artifact.v1"

RUNTIME_ASSETS = (
    ".gitattributes",
    "config.json",
    "generation_config.json",
    "LICENSE",
    "modeling_supa_a11.py",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dose_registration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != DOSE_TRIAL_SCHEMA_VERSION:
        raise ValueError(f"unsupported dose trial schema: {value.get('schema_version')!r}")
    if value.get("registration_status") != "FROZEN_BEFORE_OUTCOME":
        raise ValueError("dose trial is not frozen before outcome")
    return value


def _require_hash(directory: Path, name: str, expected: str) -> dict[str, Any]:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path}: sha256 {actual} != registered {expected}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def _adapter_map(adapter_keys: list[str]) -> dict[str, tuple[str, str]]:
    pairs: dict[str, dict[str, str]] = {}
    prefix = "base_model.model."
    for key in adapter_keys:
        if not key.startswith(prefix):
            raise ValueError(f"unsupported adapter key prefix: {key}")
        shortened = key[len(prefix) :]
        if shortened.endswith(".lora_A.weight"):
            base = shortened.removesuffix(".lora_A.weight") + ".weight"
            pairs.setdefault(base, {})["A"] = key
        elif shortened.endswith(".lora_B.weight"):
            base = shortened.removesuffix(".lora_B.weight") + ".weight"
            pairs.setdefault(base, {})["B"] = key
        else:
            raise ValueError(f"unsupported adapter tensor: {key}")
    incomplete = [base for base, pair in pairs.items() if set(pair) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"incomplete LoRA pairs: {incomplete[:5]}")
    return {base: (pair["A"], pair["B"]) for base, pair in pairs.items()}


def preflight_dose(
    registration_path: Path,
    parent_dir: Path,
    base_dir: Path,
    adapter_dir: Path,
) -> dict[str, Any]:
    registration_path = registration_path.resolve(strict=True)
    parent_dir = parent_dir.resolve(strict=True)
    base_dir = base_dir.resolve(strict=True)
    adapter_dir = adapter_dir.resolve(strict=True)
    registration = load_dose_registration(registration_path)
    parent_spec = registration["parent"]
    base_spec = registration["base"]
    adapter_spec = registration["adapter"]

    parent = {
        "weights": _require_hash(parent_dir, "model.safetensors", parent_spec["weight_sha256"]),
        "config": _require_hash(parent_dir, "config.json", parent_spec["config_sha256"]),
        "tokenizer": _require_hash(parent_dir, "tokenizer.json", parent_spec["tokenizer_sha256"]),
    }
    base = {
        "weights": _require_hash(base_dir, "model.safetensors", base_spec["weight_sha256"]),
        "config": _require_hash(base_dir, "config.json", base_spec["config_sha256"]),
        "tokenizer": _require_hash(base_dir, "tokenizer.json", base_spec["tokenizer_sha256"]),
    }
    adapter = {
        "weights": _require_hash(
            adapter_dir, "adapter_model.safetensors", adapter_spec["weights_sha256"]
        ),
        "config": _require_hash(adapter_dir, "adapter_config.json", adapter_spec["config_sha256"]),
    }
    if base["weights"]["bytes"] != int(base_spec["weight_bytes"]):
        raise ValueError("base weight byte count differs from registration")
    if adapter["weights"]["bytes"] != int(adapter_spec["weights_bytes"]):
        raise ValueError("adapter weight byte count differs from registration")

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if int(adapter_config["r"]) != int(adapter_spec["rank"]):
        raise ValueError("adapter rank drifted")
    if int(adapter_config["lora_alpha"]) != int(adapter_spec["lora_alpha"]):
        raise ValueError("adapter lora_alpha drifted")
    if sorted(adapter_config["target_modules"]) != sorted(adapter_spec["target_modules"]):
        raise ValueError("adapter target_modules drifted")

    with safe_open(parent_dir / "model.safetensors", framework="pt", device="cpu") as parent_file:
        with safe_open(base_dir / "model.safetensors", framework="pt", device="cpu") as base_file:
            with safe_open(
                adapter_dir / "adapter_model.safetensors", framework="pt", device="cpu"
            ) as adapter_file:
                parent_keys = set(parent_file.keys())
                base_keys = set(base_file.keys())
                pairs = _adapter_map(list(adapter_file.keys()))
                missing_base = sorted(parent_keys - base_keys)
                missing_adapter_targets = sorted(set(pairs) - parent_keys)
                shape_mismatches = [
                    key
                    for key in sorted(parent_keys)
                    if parent_file.get_slice(key).get_shape()
                    != base_file.get_slice(key).get_shape()
                ]
    if missing_base or missing_adapter_targets or shape_mismatches:
        raise ValueError(
            "dose source incompatibility: "
            f"missing_base={len(missing_base)}, "
            f"missing_adapter_targets={len(missing_adapter_targets)}, "
            f"shape_mismatches={len(shape_mismatches)}"
        )
    if len(pairs) != int(adapter_spec["expected_lora_slots"]):
        raise ValueError(f"LoRA slot count {len(pairs)} != registered expectation")
    if parent_spec["tokenizer_sha256"] != base_spec["tokenizer_sha256"]:
        raise ValueError("parent/base tokenizer identity differs")

    missing_assets = [name for name in RUNTIME_ASSETS if not (parent_dir / name).is_file()]
    if missing_assets:
        raise FileNotFoundError(f"parent runtime assets missing: {missing_assets}")
    return {
        "schema_version": "rineya.historical-dose-preflight.v1",
        "verified_at_utc": utc_now(),
        "trial_id": registration["trial_id"],
        "registration_sha256": sha256_file(registration_path),
        "parent": parent,
        "base": base,
        "adapter": adapter,
        "parent_tensor_count": len(parent_keys),
        "base_tensor_count": len(base_keys),
        "extra_tied_base_tensors": sorted(base_keys - parent_keys),
        "lora_slots": len(pairs),
        "passed": True,
    }


def _copy_assets(parent_dir: Path, destination: Path) -> None:
    for name in RUNTIME_ASSETS:
        source = parent_dir / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def materialize_dose(
    registration_path: Path,
    parent_dir: Path,
    base_dir: Path,
    adapter_dir: Path,
    destination: Path,
    scale: float,
    preflight_report: Mapping[str, Any],
) -> dict[str, Any]:
    registration = load_dose_registration(registration_path)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.building")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)

    if str(SOUP_SRC) not in sys.path:
        sys.path.insert(0, str(SOUP_SRC))
    from rineya_soup_lab.model.checkpoint import CheckpointReader, StreamingSafetensorsWriter

    writer = StreamingSafetensorsWriter(temporary, int(0.25 * 1024**3))
    alpha = float(registration["adapter"]["lora_alpha"])
    rank = float(registration["adapter"]["rank"])
    multiplier = alpha / rank * float(scale)
    tensor_rows = []
    try:
        _copy_assets(Path(parent_dir), temporary)
        with CheckpointReader(Path(parent_dir)) as parent_reader:
            with CheckpointReader(Path(base_dir)) as base_reader:
                with safe_open(
                    Path(adapter_dir) / "adapter_model.safetensors",
                    framework="pt",
                    device="cpu",
                ) as adapter_file:
                    pairs = _adapter_map(list(adapter_file.keys()))
                    for key in parent_reader.keys:
                        base_tensor = base_reader.get_tensor(key)
                        adapted = key in pairs
                        if adapted:
                            a_key, b_key = pairs[key]
                            a = adapter_file.get_tensor(a_key).float()
                            b = adapter_file.get_tensor(b_key).float()
                            # Match PEFT's merge order exactly: materialize and round
                            # ``B @ A * scaling`` in float32 before adding it to the
                            # float32 base. ``add_(..., alpha=scaling)`` may use a fused
                            # multiply-add and move an FP16 rounding boundary by one ULP.
                            delta = torch.matmul(b, a) * multiplier
                            merged = base_tensor.float() + delta
                        else:
                            merged = base_tensor.float()
                        output = merged.to(torch.float16)
                        writer.add(key, output)
                        tensor_rows.append(
                            {
                                "key": key,
                                "adapted": adapted,
                                "shape": list(output.shape),
                                "base_dtype": str(base_tensor.dtype),
                                "output_dtype": str(output.dtype),
                            }
                        )
        output_index = writer.finalize()
        manifest = {
            "schema_version": DOSE_ARTIFACT_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "trial_id": registration["trial_id"],
            "scale": scale,
            "multiplier_lora_alpha_over_rank_times_scale": multiplier,
            "formula": registration["recipe"]["merge_formula"],
            "registration_sha256": sha256_file(registration_path),
            "source_preflight": dict(preflight_report),
            "output_index": output_index,
            "tensor_count": len(tensor_rows),
            "adapted_tensor_count": sum(row["adapted"] for row in tensor_rows),
        }
        (temporary / "DOSE_ARTIFACT.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def compare_tensors_exact(reference: Path, reconstructed: Path) -> dict[str, Any]:
    if str(SOUP_SRC) not in sys.path:
        sys.path.insert(0, str(SOUP_SRC))
    from rineya_soup_lab.model.checkpoint import CheckpointReader

    mismatches = []
    max_abs = 0.0
    with CheckpointReader(reference) as expected:
        with CheckpointReader(reconstructed) as actual:
            if expected.keys != actual.keys:
                raise ValueError("reconstruction tensor key set differs from parent")
            for key in expected.keys:
                left = expected.get_tensor(key)
                right = actual.get_tensor(key)
                if not torch.equal(left, right):
                    mismatches.append(key)
                    if left.is_floating_point() and right.is_floating_point():
                        max_abs = max(max_abs, float((left.float() - right.float()).abs().max()))
    return {
        "passed": not mismatches,
        "tensor_count": len(expected.keys),
        "bitwise_mismatches": mismatches,
        "max_abs_diff": max_abs,
    }


def build_dose_trial(
    registration_path: Path,
    parent_dir: Path,
    base_dir: Path,
    adapter_dir: Path,
    control_dir: Path,
    candidate_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    registration_path = registration_path.resolve(strict=True)
    registration = load_dose_registration(registration_path)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    report = preflight_dose(registration_path, parent_dir, base_dir, adapter_dir)
    (evidence_dir / "preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    control = materialize_dose(
        registration_path,
        parent_dir,
        base_dir,
        adapter_dir,
        control_dir,
        float(registration["recipe"]["control_scale"]),
        report,
    )
    fidelity = compare_tensors_exact(parent_dir, control_dir)
    (evidence_dir / "control_fidelity.json").write_text(
        json.dumps(fidelity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not fidelity["passed"]:
        raise RuntimeError(
            f"control reconstruction is not bitwise exact: {len(fidelity['bitwise_mismatches'])} tensors"
        )
    candidate = materialize_dose(
        registration_path,
        parent_dir,
        base_dir,
        adapter_dir,
        candidate_dir,
        float(registration["recipe"]["candidate_scale"]),
        report,
    )
    result = {
        "schema_version": "rineya.historical-dose-build.v1",
        "trial_id": registration["trial_id"],
        "registration_sha256": sha256_file(registration_path),
        "control": control,
        "control_fidelity": fidelity,
        "candidate": candidate,
    }
    (evidence_dir / "build_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
