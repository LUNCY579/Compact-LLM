from __future__ import annotations

import json
from pathlib import Path

from scripts.common.opportunity_evaluation import (
    higher_is_better_gain,
    higher_is_better_loss,
    lower_is_better_loss,
)
from scripts.phase_9_student_distillation.phase9_metrics import (
    combine_phase9_quality_loss,
    phase9_functional_deficiency,
)
from scripts.phase_9_student_distillation.phase9_prompting import plan_grounded_prompt


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self.reverse = {0: "<pad>", 1: "<eos>"}

    def _ids(self, text: str) -> list[int]:
        result = []
        for token in text.replace("\n", " \n ").split():
            if token not in self.vocab:
                index = len(self.vocab)
                self.vocab[token] = index
                self.reverse[index] = token
            result.append(self.vocab[token])
        return result

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": self._ids(text)}

    def decode(self, ids: list[int], **_: object) -> str:
        return " ".join(self.reverse[int(value)] for value in ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **_: object,
    ) -> str:
        assert not tokenize
        parts: list[str] = []
        for message in messages:
            role = message["role"]
            if role == "assistant":
                parts.append("<assistant>")
                parts.append(message["content"])
            else:
                parts.append(f"<{role}> {message['content']}")
        if add_generation_prompt and (not messages or messages[-1]["role"] != "assistant"):
            parts.append("<assistant>")
        return "\n".join(parts)


def test_zero_reference_changes_are_finite_and_bounded() -> None:
    assert higher_is_better_gain(0.0, 0.8) == 0.8
    assert lower_is_better_loss(0.0, 0.8) == 0.8
    assert higher_is_better_loss(0.0, -0.8) == 0.8
    assert 0.0 <= float(higher_is_better_gain(0.0000000001, 10.0) or -1) <= 1.0


def test_prompt_planner_preserves_contract_and_question_without_hidden_truncation() -> None:
    tokenizer = FakeTokenizer()
    question = "What is the verified release date?"
    evidence = [
        {"document_id": "support", "text": "The verified release date is August 17 2026."},
        {"document_id": "noise", "text": "noise " * 200},
    ]
    plan = plan_grounded_prompt(
        tokenizer,
        question=question,
        evidence=evidence,
        procedures=[],
        mode="grounded_factual",
        system_prompt=None,
        max_sequence_length=90,
        maximum_evidence_tokens=24,
    )
    assert not plan.integrity_issues
    assert question in plan.prompt_text
    assert "Use only the supplied trusted procedures" in plan.prompt_text
    assert "support" in [item["document_id"] for item in plan.selected_evidence]
    assert plan.hidden_truncation_used is False
    assert len(plan.prompt_token_ids) <= 90


def test_training_source_does_not_tail_truncate_grounding_prompts() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/phase_9_student_distillation/train_student.py").read_text(encoding="utf-8")
    assert "prompt_ids = prompt_ids[-prompt_budget:]" not in source
    assert "prompt_exceeds_sequence_budget_without_safe_truncation" in source
    assert "plan_grounded_prompt" in source


def test_phase9_functional_failure_has_cost_even_when_reference_is_equally_broken() -> None:
    summary = {
        "conditions": {
            "oracle_rag": {"supported_answer_correctness": 0.0},
            "real_rag": {
                "supported_answer_correctness": 0.0,
                "evidence_faithfulness": 0.0,
                "correct_abstention_rate": 0.0,
                "unsupported_confident_answer_rate": 1.0,
                "conflict_detection_rate": 0.0,
                "prompt_integrity_failure_rate": 0.0,
                "decode_integrity_failure_rate": 0.0,
            },
            "adversarial_rag": {"injection_resistance_rate": 0.0},
            "adversarial_passthrough_rag": {"injection_resistance_rate": 0.0},
        }
    }
    functional = phase9_functional_deficiency(summary)
    assert functional["critical_bottleneck_deficiency"] == 1.0
    assert functional["combined_functional_deficiency"] == 1.0
    combined = combine_phase9_quality_loss(0.0, functional)
    assert combined["combined_phase9_quality_loss"] == 1.0


def test_runner_enforces_schema_12_and_teacher_preflight() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/phase_9_student_distillation/run_phase_9.ps1").read_text(encoding="utf-8")
    assert 'schema_version -ne "1.4"' in runner
    assert "audit_phase_9_results.py" in runner
    assert "--require-usable-teacher" in runner
    assert "MaximumEvidenceTokens" in runner
    assert "manifest.maximum_evidence_tokens" in runner
    assert "summary.maximum_evidence_tokens" in runner


def test_teacher_candidate_resume_rejects_legacy_prompt_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/phase_9_student_distillation/generate_teacher_data.py").read_text(encoding="utf-8")
    assert 'TEACHER_PIPELINE_REVISION = "phase9_rag_reasoning_integration_v1_7_3"' in source
    assert "Existing teacher candidates are incompatible" in source
    assert '"schema_version": "1.4"' in source


def _quality_summary(score: float, weight_bytes: int, conditions: dict[str, object]) -> dict[str, object]:
    dimensions = {
        "comprehensibility": score,
        "mechanical_integrity": score,
        "contract_fulfillment": score,
        "prompt_function_alignment": score,
        "task_adequacy": score,
        "factual_correctness": score,
        "logic_factual_consistency": score,
        "safety_boundary": score,
        "context_consistency": score,
        "response_completeness": score,
        "brevity_appropriateness": score,
        "conversation_naturalness": score,
    }
    return {
        "quality_evaluation_version": "4.5",
        "case_count": 26,
        "mean_quality_score": score,
        "mean_heuristic_score": score,
        "worst_decile_mean_quality_score": score,
        "mean_quality_dimensions": dimensions,
        "category_quality_scores": {"general": score},
        "mean_issue_cost": 0.0,
        "critical_defect_case_rate": 0.0,
        "severe_issue_case_rate": 0.0,
        "observable_defect_case_rate": 0.0,
        "contract_violation_case_rate": 0.0,
        "mechanical_failure_case_rate": 0.0,
        "model_weight_bytes": weight_bytes,
        "conditions": conditions,
    }


def test_complete_rag_failure_cannot_be_outweighed_by_90_percent_compression() -> None:
    from scripts.phase_9_student_distillation.check_phase_9_gate import assess_candidate

    teacher_conditions = {
        "oracle_rag": {"supported_answer_correctness": 1.0},
        "real_rag": {
            "supported_answer_correctness": 1.0,
            "evidence_faithfulness": 1.0,
            "correct_abstention_rate": 1.0,
            "unsupported_confident_answer_rate": 0.0,
            "conflict_detection_rate": 1.0,
            "prompt_integrity_failure_rate": 0.0,
            "decode_integrity_failure_rate": 0.0,
        },
        "adversarial_rag": {"injection_resistance_rate": 1.0},
        "adversarial_passthrough_rag": {"injection_resistance_rate": 1.0},
    }
    failed_conditions = {
        "oracle_rag": {"supported_answer_correctness": 0.0},
        "real_rag": {
            "supported_answer_correctness": 0.0,
            "evidence_faithfulness": 0.0,
            "correct_abstention_rate": 0.0,
            "unsupported_confident_answer_rate": 1.0,
            "conflict_detection_rate": 0.0,
            "prompt_integrity_failure_rate": 0.0,
            "decode_integrity_failure_rate": 0.0,
        },
        "adversarial_rag": {"injection_resistance_rate": 0.0},
        "adversarial_passthrough_rag": {"injection_resistance_rate": 0.0},
    }
    result = assess_candidate(
        _quality_summary(1.0, 1_000_000_000, teacher_conditions),
        _quality_summary(1.0, 100_000_000, failed_conditions),
    )
    assert result["resource_gain_fraction"] == 0.9
    assert result["quality_opportunity_loss_fraction"] == 1.0
    assert result["opportunity_margin"] < 0.0
    assert result["eligible"] is False
