"""Admission and output-contract tests for the native collector.

These tests never import torch or transformers and never access checkpoint bytes.
"""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_native
from soup import EXPERIMENT, sha256


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def base_authorization(tmp_path: Path, protocol: Path) -> dict:
    digest = "a" * 64
    return {
        "schema": 1,
        "experiment": EXPERIMENT,
        "authorization_id": "AUTH-001",
        "protocol_sha256": sha256(protocol),
        "native_authorized": True,
        "arm_id": "sparse_opallo_50",
        "split": "CAL",
        "suite_id": "SUITE-001",
        "dataset_sha256": digest,
        "runtime_sha256": digest,
        "scorer_sha256": digest,
        "decoding_sha256": digest,
        "tokenizer_sha256": digest,
        "qualification_receipt_sha256": digest,
        "ancestry_audit_sha256": digest,
        "freeze_sha256": digest,
        "evaluation_dtype": "float32",
        "checkpoint_sha256": {"sparse_opallo_50": digest},
    }


def test_current_protocol_refuses_before_runtime_import(tmp_path):
    root = Path(__file__).resolve().parents[1]
    authorization = tmp_path / "authorization.json"
    protocol = root / "protocol.json"
    value = base_authorization(tmp_path, protocol)
    value["native_authorized"] = False
    write_json(authorization, value)
    model = tmp_path / "model"
    model.mkdir()
    suite = tmp_path / "suite.jsonl"
    suite.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="native_authorized=false"):
        collect_native.validate_admission(
            protocol_path=protocol,
            authorization_path=authorization,
            arm="sparse_opallo_50",
            split="CAL",
            model_dir=model,
            suite_path=suite,
        )


def test_suite_requires_canonical_identity_rows(tmp_path):
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {"experiment": EXPERIMENT, "native_authorized": True})
    suite = tmp_path / "suite.jsonl"
    suite.write_text(json.dumps({
        "id": "row-1",
        "parent_id": "parent-1",
        "domain": "transition",
        "answerable": True,
        "prompt": "rendered prompt",
    }) + "\n", encoding="utf-8")
    authorization = base_authorization(tmp_path, protocol)
    authorization["dataset_sha256"] = sha256(suite)
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(ValueError, match="model.safetensors"):
        collect_native.validate_admission(
            protocol_path=protocol,
            authorization_path=write_json(tmp_path / "authorization.json", authorization),
            arm="sparse_opallo_50",
            split="CAL",
            model_dir=model,
            suite_path=suite,
        )


def test_output_metadata_is_raw_and_not_scored(tmp_path):
    authorization = {
        "authorization_id": "AUTH-001",
        "suite_id": "SUITE-001",
        "checkpoint_sha256": {"sparse_opallo_50": "a" * 64},
        "tokenizer_sha256": "b" * 64,
        "runtime_sha256": "c" * 64,
        "scorer_sha256": "d" * 64,
        "decoding_sha256": "e" * 64,
        "qualification_receipt_sha256": "f" * 64,
        "ancestry_audit_sha256": "0" * 64,
        "freeze_sha256": "1" * 64,
    }
    result = collect_native.render_prediction_run(
        authorization=authorization,
        arm="sparse_opallo_50",
        split="CAL",
        model_dir=tmp_path / "model",
        rows=[{"id": "row-1"}],
        predictions=[{"id": "row-1", "prediction": "answer"}],
        dataset_sha256="2" * 64,
        protocol_sha256="3" * 64,
    )
    assert result["status"] == "RAW_PREDICTIONS_NOT_SCORED"
    assert "correct" not in result["rows"][0]
    assert result["metadata"]["evaluation_dtype"] == "float32"
