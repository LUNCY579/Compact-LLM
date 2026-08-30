from __future__ import annotations

import json
from pathlib import Path

import pytest

from rineya_tgsa.run_integrity import (
    RetrofitIntegrityError,
    attach_training_integrity,
    capture_device_identity,
    sha256_file,
    sha256_json,
    source_tree_manifest,
    training_dose,
    verify_training_integrity,
)


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_source_tree_fingerprint_changes_with_weights(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _write(model / "config.json", b"{}")
    weight = model / "model.safetensors"
    _write(weight, b"weights-a")
    first = source_tree_manifest(model)
    _write(weight, b"weights-b")
    second = source_tree_manifest(model)
    assert first["fingerprint"] != second["fingerprint"]


def test_training_dose_includes_gradient_accumulation() -> None:
    dose = training_dose(
        rows=100,
        optimizer_steps=10,
        microbatch_size=2,
        gradient_accumulation=5,
    )
    assert dose["sample_presentations"] == 100
    assert dose["effective_epochs"] == pytest.approx(1.0)


def test_training_dose_rejects_nonpositive_values() -> None:
    with pytest.raises(RetrofitIntegrityError, match="positive"):
        training_dose(
            rows=0,
            optimizer_steps=10,
            microbatch_size=2,
            gradient_accumulation=5,
        )


def test_device_configuration_is_fingerprinted() -> None:
    first = capture_device_identity("cpu")
    changed = dict(first)
    changed["resolved_device"] = "cpu:changed"
    assert sha256_json(first) != sha256_json(changed)


def test_training_bundle_verifies_and_detects_tampering(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _write(model / "config.json", b"{}")
    _write(model / "model.safetensors", b"weights")
    source = source_tree_manifest(model)
    train = tmp_path / "train.jsonl"
    _write(train, b'{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}]}\n')
    output = tmp_path / "output"
    checkpoint = output / "tgsa_mtp_adapter.pt"
    trace = output / "training_trace.jsonl"
    _write(checkpoint, b"adapter")
    _write(trace, b'{"step":1}\n')
    _json(
        output / "training_manifest.json",
        {
            "schema_version": "test",
            "status": "complete",
            "train_jsonl": str(train),
            "train_jsonl_sha256": sha256_file(train),
            "rows": 1,
            "steps": 1,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "converted_layers": [0],
            "stage": {"stage": "adapters"},
            "learning_rate": 0.001,
            "seed": 7,
            "conversion": {"test": True},
            "adapter_checkpoint": checkpoint.name,
            "training_trace": trace.name,
            "claim_scope": "development",
        },
    )
    attach_training_integrity(
        output_dir=output,
        source_manifest=source,
        device_identity=capture_device_identity("cpu"),
        invocation={"development_only": True},
    )
    report = verify_training_integrity(output_dir=output)
    assert report["passed"] is True, report
    _write(checkpoint, b"tampered")
    tampered = verify_training_integrity(output_dir=output)
    assert tampered["passed"] is False
    assert any("adapter_checkpoint" in error for error in tampered["errors"])
