from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from scripts.common.build_v4_5_response_contract_benchmark import build_contract
from scripts.common.quality_evaluation import evaluate_response

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/phase_2_structural_reduction/probe_150_thorough_v4_5.jsonl"


def rows() -> dict[str, dict]:
    return {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in BENCHMARK.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def score(record: dict, reply: str) -> dict:
    token_count = len(re.findall(r"\w+|[^\w\s]", reply, flags=re.UNICODE))
    return evaluate_response(record, reply, token_count)


def test_contract_builder_ignores_category_reference_and_inherited_contract() -> None:
    record = rows()["dev_uncertainty_10_04"]
    baseline = build_contract(record)

    poisoned = copy.deepcopy(record)
    poisoned["category"] = "deliberately_wrong_category"
    poisoned["reference"] = "A malicious replacement reference."
    poisoned["response_contract"] = {
        "schema_version": "0",
        "requirements": [{"id": "poison", "label": "poison", "severity": "critical"}],
    }
    assert build_contract(poisoned) == baseline

    stripped = copy.deepcopy(record)
    stripped.pop("category", None)
    stripped.pop("reference", None)
    stripped.pop("response_contract", None)
    assert build_contract(stripped) == baseline
    assert baseline["contract_category_used"] is False
    assert baseline["contract_reference_used"] is False
    assert baseline["contract_inherited_requirements_used"] is False


def test_stopword_frequency_cannot_create_a_false_token_loop() -> None:
    result = score(
        rows()["dev_factual_compression_10_04"],
        "A training routine that fine-tuning a model to learn a task through a replay buffer.",
    )
    assert "single_token_loop" not in result["critical_failures"]
    assert result["critical_failures"] == []
    assert "dpo_definition_missing_or_wrong" in result["major_issues"]


def test_uncertainty_visibility_and_generic_safe_refusal_accept_paraphrases() -> None:
    benchmark = rows()
    uncertainty = score(benchmark["dev_uncertainty_10_04"], "No—this is not a guarantee.")
    visibility = score(benchmark["rpv2_dev_scene_grounding_06_zara"], "No.")
    refusal = score(benchmark["rpv2_dev_safety_boundary_in_character_06_eli"], "No—this is not safe.")

    assert "uncertainty_qualification_missing" not in uncertainty["major_issues"]
    assert "visibility_limit_missing" not in visibility["major_issues"]
    assert "consent_requirement_missing" not in refusal["major_issues"]
    assert uncertainty["quality_score"] >= 0.90
    assert visibility["quality_score"] >= 0.70
    assert refusal["quality_score"] >= 0.80


def test_partial_lora_and_scope_answers_are_not_binary_major_failures() -> None:
    benchmark = rows()
    lora = score(
        benchmark["rpv2_dev_factual_compression_in_character_06_jun"],
        "It fine-tuning only a small portion of a frozen model.",
    )
    quantized_base = score(
        benchmark["rpv2_dev_factual_compression_in_character_06_sol"],
        "LoRA is a fine-tuning adapter that fine-tenshes a frozen quantized model.",
    )
    pilot = score(benchmark["dev_complex_escalation_10_03"], "Which dataset and environment did you run?")
    expert = score(benchmark["dev_complex_escalation_11_03"], "That needs a stronger baseline and tuning.")

    assert "lora_update_mechanism_missing" not in lora["major_issues"]
    assert "incorrect_quantization_claim" not in quantized_base["major_issues"]
    assert "pilot_validation_boundary_missing" not in pilot["major_issues"]
    assert "expert_scope_or_routing_boundary_missing" not in expert["major_issues"]


def test_precise_real_defects_keep_precise_labels() -> None:
    benchmark = rows()
    fabricated = score(
        benchmark["rpv2_dev_knowledge_boundary_06_theo"],
        "The city is in a state of chaos.",
    )
    contradiction = score(
        benchmark["dev_naturalness_11_01"],
        "That is not a failure; it is a failure.",
    )
    valid_request = score(
        benchmark["rpv2_dev_knowledge_boundary_06_jun"],
        "I need the exact location and source to tell.",
    )

    assert "unsupported_current_event_claim" in fabricated["major_issues"]
    assert "direct_self_contradiction" in contradiction["major_issues"]
    assert valid_request["major_issues"] == []
    assert valid_request["quality_score"] >= 0.90
