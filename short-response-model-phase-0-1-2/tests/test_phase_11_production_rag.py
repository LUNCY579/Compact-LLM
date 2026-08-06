from __future__ import annotations

import json
from pathlib import Path

from scripts.phase_11_production_rag.phase11_common import (
    AnswerabilityFeatures,
    answerability_gate,
    calibrate_answerability,
    production_grounding_check,
    route_request,
)


def test_phase11_routing_separates_high_risk_complex_and_social() -> None:
    assert route_request("Good morning!").mode == "social_roleplay"
    assert route_request("What dosage should I take?").mode == "high_risk"
    assert route_request("Please implement a complete detailed compiler").mode == "complex"
    assert route_request("When is the release?").mode == "grounded_factual"


def test_answerability_calibration_prefers_false_answer_safety() -> None:
    rows = [
        {
            "id": "a",
            "answerable": True,
            "features": {
                "best_retrieval_score": 0.9,
                "question_coverage": 0.9,
                "evidence_count_score": 1.0,
                "conflict_detected": False,
            },
        },
        {
            "id": "b",
            "answerable": False,
            "features": {
                "best_retrieval_score": 0.7,
                "question_coverage": 0.2,
                "evidence_count_score": 1.0,
                "conflict_detected": False,
            },
        },
    ]
    best, packets = calibrate_answerability(
        rows,
        weight_candidates=[
            {"best_retrieval_score": 1.0, "question_coverage": 0.0, "evidence_count_score": 0.0},
            {"best_retrieval_score": 0.5, "question_coverage": 0.5, "evidence_count_score": 0.0},
        ],
        false_answer_cost=4.0,
        false_abstention_cost=1.0,
    )
    assert packets
    assert best["false_positive"] == 0


def test_answerability_gate_detects_curated_conflict() -> None:
    calibration = {
        "weights": {"best_retrieval_score": 0.5, "question_coverage": 0.5, "evidence_count_score": 0.0},
        "threshold": 0.3,
    }
    passages = [
        {
            "record_id": "a",
            "text": "The release is in 2026.",
            "normalized_score": 0.8,
            "claim_key": "release_year",
            "claim_value": "2026",
        },
        {
            "record_id": "b",
            "text": "The release is in 2027.",
            "normalized_score": 0.7,
            "claim_key": "release_year",
            "claim_value": "2027",
        },
    ]
    decision = answerability_gate("When is the release?", passages, calibration)
    assert not decision.should_answer
    assert decision.conflict_detected


def test_production_grounding_rejects_unsupported_entity_and_number() -> None:
    result = production_grounding_check(
        "Nookumi launches in 2027 with Alice.",
        [{"text": "Nookumi launches in 2026 with Sera."}],
    )
    assert not result.supported
    assert any("2027" in failure for failure in result.hard_failures)
    assert any("alice" in failure for failure in result.hard_failures)


def test_phase11_config_and_runner_are_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/phase_11_production_rag.json").read_text(encoding="utf-8"))
    assert config["answerability"]["threshold"] is None
    assert config["evaluation"]["unsupported_confident_answer_cap"] is None
    runner = (root / "scripts/phase_11_production_rag/run_phase_11.ps1").read_text(encoding="utf-8")
    for script in (
        "build_knowledge_snapshot.py",
        "calibrate_answerability.py",
        "quantize_student.py",
        "benchmark_production_runtime.py",
        "check_phase_11_gate.py",
        "package_phase_11_release.py",
    ):
        assert script in runner
    assert (root / "scripts/phase_11_production_rag/serve_production_rag.py").exists()


def test_answerability_calibration_treats_conflict_as_unanswerable_zero_score() -> None:
    rows = [
        {
            "id": "conflict",
            "answerable": False,
            "features": {
                "best_retrieval_score": 1.0,
                "question_coverage": 1.0,
                "evidence_count_score": 1.0,
                "conflict_detected": True,
            },
        },
        {
            "id": "supported",
            "answerable": True,
            "features": {
                "best_retrieval_score": 0.8,
                "question_coverage": 0.8,
                "evidence_count_score": 1.0,
                "conflict_detected": False,
            },
        },
    ]
    best, packets = calibrate_answerability(
        rows,
        weight_candidates=[
            {"best_retrieval_score": 0.5, "question_coverage": 0.5, "evidence_count_score": 0.0}
        ],
        false_answer_cost=3.0,
        false_abstention_cost=1.0,
    )
    assert packets
    assert best["false_positive"] == 0


def test_phase11_runner_passes_context_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/phase_11_production_rag/run_phase_11.ps1").read_text(encoding="utf-8")
    assert "MaximumEvidenceTokens" in runner
    assert "--maximum-evidence-tokens" in runner
