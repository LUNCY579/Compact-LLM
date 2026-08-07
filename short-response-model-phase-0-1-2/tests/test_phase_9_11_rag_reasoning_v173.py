from __future__ import annotations

import json
from pathlib import Path

from scripts.phase_9_student_distillation.build_pilot_dataset import (
    has_complete_intended_splits,
    normalize_training_row,
    split_by_intended,
)
from scripts.phase_9_student_distillation.phase9_common import detect_structured_conflict
from scripts.phase_9_student_distillation.phase9_metrics import (
    decision_class,
    is_safe_conflict_response,
)
from scripts.phase_9_student_distillation.validate_rag_reasoning_corpus import validate
from scripts.phase_9b_memory_reconstruction.phase9b_common import deterministic_reconstruction

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "phase_9_student_distillation" / "rag_reasoning_v2"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_integrated_corpus_manifest_and_hashes_are_valid() -> None:
    packet = validate(CORPUS)
    assert packet["status"] == "PASS", packet["problems"]
    assert packet["reviewed_rows"] == 530
    assert packet["split_counts"] == {"train": 392, "dev": 67, "test": 71}
    assert packet["knowledge_documents"] == 464


def test_procedure_answerability_ontology_and_two_repairs() -> None:
    rows = read_jsonl(CORPUS / "teacher_candidates.reviewed.jsonl")
    assert all(row.get("decision") != "procedure" for row in rows)
    by_id = {row["id"]: row for row in rows}
    assert by_id["train_procedure_semver_family_04_conflicting_inputs"]["decision"] == "answer"
    assert by_id["train_procedure_semver_family_04_conflicting_inputs"]["approved_answer"] == "Version 2.10.0."
    assert by_id["train_procedure_iso_date_family_04_conflicting_inputs"]["decision"] == "answer"
    assert by_id["train_procedure_iso_date_family_04_conflicting_inputs"]["approved_answer"] == "2032-11-02."


def test_all_530_reviewed_rows_are_phase9_builder_compatible() -> None:
    raw = read_jsonl(CORPUS / "teacher_candidates.reviewed.jsonl")
    normalized = [normalize_training_row(row, allow_pending=False) for row in raw]
    assert len(normalized) == 530
    assert has_complete_intended_splits(normalized)
    splits = split_by_intended(normalized)
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 392, "dev": 67, "test": 71}


def test_conflict_can_safely_abstain_without_forced_word_conflict() -> None:
    answer = "I don't know based on the available information."
    assert is_safe_conflict_response(answer)
    rows = read_jsonl(CORPUS / "teacher_candidates.reviewed.jsonl")
    conflict = next(row for row in rows if row.get("category") == "conflict")
    assert decision_class(conflict) == "conflict"
    normalized = normalize_training_row(conflict, allow_pending=False)
    assert normalized["decision"] == "abstain"
    assert normalized["reference"] == answer


def test_structured_conflict_respects_trust_precedence() -> None:
    stale_pair = [
        {"claim_key": "firmware", "claim_value": "1.8", "trust_level": "superseded"},
        {"claim_key": "firmware", "claim_value": "2.1", "trust_level": "standard"},
    ]
    equal_authority = [
        {"claim_key": "firmware", "claim_value": "1.8", "trust_level": "standard"},
        {"claim_key": "firmware", "claim_value": "2.1", "trust_level": "standard"},
    ]
    injection_pair = [
        {"claim_key": "price", "claim_value": "10", "trust_level": "standard"},
        {"claim_key": "price", "claim_value": "999", "trust_level": "untrusted"},
    ]
    assert not detect_structured_conflict(stale_pair)
    assert not detect_structured_conflict(injection_pair)
    assert detect_structured_conflict(equal_authority)


def test_stale_memory_seed_reconstructs_by_adapting_not_conflicting() -> None:
    rows = read_jsonl(CORPUS / "prompts.all.jsonl")
    stale = next(row for row in rows if row.get("category") == "stale_evidence")
    target = deterministic_reconstruction(
        current_state=stale.get("current_state") or {},
        memories=stale["evidence"],
    )
    assert target["decision"] == "adapt"
    assert len(target["source_ids"]) == 1
    assert any("superseded_for_current_scope" in item for item in target["invalid_assumptions"])


def test_preferences_and_evaluator_assets_are_first_class() -> None:
    manifest = json.loads((CORPUS / "manifests/dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["preference_train_pairs"] == 1176
    assert manifest["counts"]["preference_dev_pairs"] == 201
    assert manifest["counts"]["evaluator_calibration_cases"] == 1836
    assert manifest["counts"]["heldout_test_candidate_cases"] == 284


def test_phase9_runner_uses_dev_for_selection_and_test_only_after_freeze() -> None:
    runner = (ROOT / "scripts/phase_9_student_distillation/run_phase_9_grounded_recovery.ps1").read_text(encoding="utf-8")
    assert '"--split-policy", "intended"' in runner
    assert '"--holdout", $DevFile' in runner and '"--holdout", $TestFile' in runner
    assert '$DevFile $TeacherOracle' in runner
    assert '$DevFile $WarmOracle' in runner
    assert '$DevFile $EvaluationRoot' in runner
    freeze = runner.index("[Stage E] Selecting the final refinement on DEV only")
    first_test = runner.index("[Sealed Test]")
    assert freeze < first_test



def test_general_phase9_runner_is_dev_first_and_seals_test_after_selection() -> None:
    runner = (ROOT / "scripts/phase_9_student_distillation/run_phase_9.ps1").read_text(encoding="utf-8")
    assert 'if ([string]::IsNullOrWhiteSpace($Benchmark)) { $Benchmark = $DevFile }' in runner
    assert '"--benchmark", $DevFile' in runner
    assert '[DEV selection] Building the proportional Phase 9 development assessment' in runner
    dev_selection = runner.index('[DEV selection] Building the proportional Phase 9 development assessment')
    sealed = runner.index('[Sealed TEST] Evaluating the teacher reference')
    assert dev_selection < sealed
    assert 'SmokeTest does not touch the sealed test split' in runner
    test_uses = [line for line in runner.splitlines() if '$TestFile' in line]
    assert any('--holdout' in line for line in test_uses)
    assert any('[Sealed TEST]' in line for line in test_uses)
    assert not any('--benchmark", $TestFile' in line for line in test_uses)


def test_phase9b_runner_preserves_authored_split_and_shared_corpus_defaults() -> None:
    runner = (ROOT / "scripts/phase_9b_memory_reconstruction/run_phase_9b.ps1").read_text(encoding="utf-8")
    assert r'rag_reasoning_v2\prompts.all.jsonl' in runner
    assert r'rag_reasoning_v2\knowledge.jsonl' in runner
    assert '"--split-policy", "intended"' in runner

def test_phase10_freezes_on_dev_before_sealed_test() -> None:
    runner = (ROOT / "scripts/phase_10_student_architecture/run_phase_10.ps1").read_text(encoding="utf-8")
    assert '"--purpose", "final"' in runner
    freeze = runner.index("Freezing the final Phase 10 candidate on DEV")
    sealed = runner.index("Sealed TEST reference evaluation")
    assert freeze < sealed
    assert '$DevFile $RawEval' in runner


def test_phase11_uses_supported_decision_class_for_answerability() -> None:
    source = (ROOT / "scripts/phase_11_production_rag/calibrate_answerability.py").read_text(encoding="utf-8")
    assert 'answerable = decision_class(row) == "supported"' in source
    benchmark = (ROOT / "scripts/phase_11_production_rag/benchmark_production_runtime.py").read_text(encoding="utf-8")
    assert "functional_answer_correct" in benchmark
    assert "conflict_safe_handling_rate" in benchmark
    assert "stale_precedence_rate" in benchmark
