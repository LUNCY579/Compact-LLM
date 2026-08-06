from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.phase_9b_memory_reconstruction.check_phase_9b_gate import assess
from scripts.phase_9_student_distillation.phase9_common import validate_knowledge_record
from scripts.phase_9b_memory_reconstruction.phase9b_common import (
    ABSTENTION_TEXT,
    build_answer_messages,
    build_direct_memory_messages,
    build_raw_replay_messages,
    build_reconstruction_messages,
    deterministic_reconstruction,
    parse_reconstruction_output,
    reconstruction_training_rows,
    validate_memory_record,
    validate_reconstruction_record,
    memory_from_phase9_passage,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "phase_9b_memory_reconstruction" / "reviewed.sample.jsonl"


def _rows() -> list[dict]:
    return [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_phase9b_assets_and_runtime_integration_exist() -> None:
    config = json.loads((ROOT / "configs" / "phase_9b_memory_reconstruction.json").read_text(encoding="utf-8"))
    assert config["inference"]["mode"] == "two_pass_reconstruct_then_answer"
    assert set(config["dataset"]["allowed_decisions"]) == {
        "accept", "adapt", "reject", "conflict", "insufficient"
    }
    runner = (ROOT / "scripts" / "phase_9b_memory_reconstruction" / "run_phase_9b.ps1").read_text(encoding="utf-8")
    assert "build_reconstruction_candidates.py" in runner
    assert "generate_teacher_reconstructions.py" in runner
    assert "build_reconstruction_dataset.py" in runner
    assert "evaluate_memory_reconstruction.py" in runner
    assert "-UseLora" in runner

    phase10 = (ROOT / "scripts" / "phase_10_student_architecture" / "run_phase_10.ps1").read_text(encoding="utf-8")
    assert "Phase9BReferenceManifest" in phase10
    assert '"--phase", "9B"' in phase10
    server = (ROOT / "scripts" / "phase_11_production_rag" / "serve_production_rag.py").read_text(encoding="utf-8")
    assert "--memory-reconstruction-mode" in server
    assert "build_reconstruction_messages" in server
    assert "parse_reconstruction_output" in server


def test_phase11_knowledge_passage_upgrades_to_factual_memory() -> None:
    memory = memory_from_phase9_passage(
        {
            "record_id": "doc1",
            "kind": "knowledge",
            "text": "Version 0.9 releases August 17.",
            "source": "release_plan",
            "metadata": {
                "source_state": {"product_version": "0.9"},
                "validity_conditions": {"product_version": "0.9"},
            },
        }
    )
    assert memory["memory_type"] == "factual"
    assert memory["source_state"] == {"product_version": "0.9"}


def test_phase9_knowledge_snapshot_preserves_phase9b_metadata() -> None:
    record = validate_knowledge_record(
        {
            "document_id": "doc1",
            "text": "Version 0.9 releases August 17.",
            "source_state": {"product_version": "0.9"},
            "validity_conditions": {"product_version": "0.9"},
            "memory_type": "factual",
        },
        kind="knowledge",
    )
    assert record["metadata"]["source_state"] == {"product_version": "0.9"}
    assert record["metadata"]["validity_conditions"] == {"product_version": "0.9"}
    assert record["metadata"]["memory_type"] == "factual"


def test_memory_schema_preserves_source_state_and_provenance() -> None:
    memory = validate_memory_record(
        {
            "memory_id": "m1",
            "memory_type": "procedure",
            "text": "Resume only when artifacts exist.",
            "source_state": {"artifacts_exist": True},
            "validity_conditions": {"artifacts_exist": True},
            "source_id": "runbook",
        }
    )
    assert memory["source_state"] == {"artifacts_exist": True}
    assert memory["validity_conditions"] == {"artifacts_exist": True}
    assert memory["source_id"] == "runbook"
    with pytest.raises(ValueError, match="empty text"):
        validate_memory_record({"memory_id": "bad", "text": ""})


def test_deterministic_baseline_accepts_rejects_conflicts_and_abstains() -> None:
    rows = {row["id"]: validate_reconstruction_record(row) for row in _rows()}
    accepted = deterministic_reconstruction(
        current_state=rows["resume_accept"]["current_state"],
        memories=rows["resume_accept"]["memories"],
    )
    rejected = deterministic_reconstruction(
        current_state=rows["resume_reject"]["current_state"],
        memories=rows["resume_reject"]["memories"],
    )
    conflicted = deterministic_reconstruction(
        current_state=rows["release_conflict"]["current_state"],
        memories=rows["release_conflict"]["memories"],
    )
    insufficient = deterministic_reconstruction(current_state={}, memories=[])
    assert accepted["decision"] == "accept"
    assert rejected["decision"] == "reject"
    assert conflicted["decision"] == "conflict"
    assert insufficient["decision"] == "insufficient"


def test_reconstruction_parser_is_strict_and_safely_abstains() -> None:
    valid = parse_reconstruction_output(
        '```json\n{"decision":"accept","source_ids":["m1"],"applicable_facts":["x"],'
        '"invalid_assumptions":[],"guidance":"Use x.","confidence":"high"}\n```'
    )
    assert valid.valid and valid.payload["decision"] == "accept"
    invalid = parse_reconstruction_output("not json")
    assert not invalid.valid
    assert invalid.payload["decision"] == "insufficient"
    messages = build_answer_messages(question="What?", reconstruction=invalid.payload)
    assert ABSTENTION_TEXT in messages[-1]["content"]


def test_raw_replay_omits_source_state_but_reconstruction_uses_it() -> None:
    row = validate_reconstruction_record(_rows()[0])
    raw = json.dumps(build_raw_replay_messages(question=row["question"], memories=row["memories"]), ensure_ascii=False)
    reconstructed = json.dumps(
        build_reconstruction_messages(
            question=row["question"],
            current_state=row["current_state"],
            memories=row["memories"],
            mode=row["mode"],
        ),
        ensure_ascii=False,
    )
    direct = json.dumps(
        build_direct_memory_messages(
            question=row["question"],
            current_state=row["current_state"],
            memories=row["memories"],
            mode=row["mode"],
        ),
        ensure_ascii=False,
    )
    assert "source_state" not in raw
    assert "source_state" in reconstructed
    assert "CURRENT_STATE" in reconstructed
    assert "CURRENT_STATE" in direct


def test_reviewed_rows_materialize_three_training_views() -> None:
    for raw in _rows():
        row = validate_reconstruction_record(raw)
        training = reconstruction_training_rows(row)
        assert {item["task_type"] for item in training} == {
            "memory_reconstruction", "answer_from_reconstruction", "direct_memory_distillation"
        }
        assert all(item["messages"][-1]["role"] == "assistant" for item in training)


def test_sample_dataset_builder_smoke_path(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "phase_9b_memory_reconstruction" / "build_reconstruction_dataset.py"),
            "--reviewed-input",
            str(SAMPLE),
            "--output-dir",
            str(output),
            "--smoke-test",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["smoke_test"] is True
    assert manifest["accepted_rows"] == len(_rows())
    assert (output / "train_combined.jsonl").exists()
    assert (output / "test_direct.jsonl").exists()
    assert (output / "preferences_train.jsonl").exists()


def test_phase9b_gate_uses_proportional_benefit_and_cost() -> None:
    summary = {
        "benchmark": "test.jsonl",
        "research_ready": True,
        "smoke_test": False,
        "evaluation_valid": True,
        "baseline_model": "baseline_student",
        "baseline_adapter": None,
        "candidate_model": "student",
        "candidate_adapter": "adapter",
        "peak_allocated_vram_bytes": {"baseline": 100, "candidate": 105},
        "strategies": {
            "raw_replay": {
                "case_count": 10,
                "functional_success_rate": 0.5,
                "mean_total_generation_seconds": 1.0,
                "mean_total_prompt_tokens": 100,
                "mean_total_generated_tokens": 10,
            },
            "prompted_reconstruction": {
                "case_count": 10,
                "reconstruction_case_count": 10,
                "functional_success_rate": 0.6,
                "reconstruction_decision_accuracy": 0.6,
                "source_id_recall": 0.6,
                "mean_total_generation_seconds": 1.1,
                "mean_total_prompt_tokens": 105,
                "mean_total_generated_tokens": 11,
            },
            "trained_reconstruction": {
                "case_count": 10,
                "reconstruction_case_count": 10,
                "functional_success_rate": 0.8,
                "reconstruction_decision_accuracy": 0.9,
                "reconstruction_parse_valid_rate": 1.0,
                "source_id_recall": 0.9,
                "mean_total_generation_seconds": 1.1,
                "mean_total_prompt_tokens": 105,
                "mean_total_generated_tokens": 11,
            },
        },
        "pairwise": {
            "raw_replay": {"negative_transfer_rate": 0.3},
            "trained_reconstruction": {
                "negative_transfer_rate": 0.1,
                "raw_failure_rescue_rate": 0.3,
                "raw_success_harm_rate": 0.0,
            },
        },
    }
    config = json.loads((ROOT / "configs" / "phase_9b_memory_reconstruction.json").read_text(encoding="utf-8"))
    result = assess(summary, config, {"trained_reconstruction": {"model": "student", "adapter": "adapter"}})
    assert result["strategies"]["prompted_reconstruction"]["artifact"] == {
        "model": "baseline_student", "adapter": None
    }
    packet = result["strategies"]["trained_reconstruction"]
    assert packet["combined_reconstruction_benefit"] > 0
    assert packet["combined_runtime_cost"] > 0
    assert packet["opportunity_margin"] > 0
    assert packet["eligible"] is True
