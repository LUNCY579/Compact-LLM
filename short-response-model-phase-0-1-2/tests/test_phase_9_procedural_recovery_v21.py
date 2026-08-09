from __future__ import annotations

import json
from pathlib import Path

from scripts.phase_9_student_distillation.diagnose_procedural_transfer_gap import diagnose
from scripts.phase_9_student_distillation.phase9_common import select_retrieved_passages
from scripts.phase_9_student_distillation.validate_rag_reasoning_corpus import validate

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/phase_9_student_distillation/procedural_recovery_v21"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_query_aware_selector_removes_wrong_sibling_records() -> None:
    question = "In Birch latency audit 5154, record 2, which relay is faster?"
    passages = [
        {"record_id": "r1", "rank": 1, "normalized_score": 0.92, "text": "Birch latency audit 5154, record 1. Relay Cedar 14 ms. Relay Birch 9 ms."},
        {"record_id": "r2", "rank": 2, "normalized_score": 0.88, "text": "Birch latency audit 5154, record 2. Relay Cedar 20 ms. Relay Birch 12 ms."},
        {"record_id": "r3", "rank": 3, "normalized_score": 0.84, "text": "Birch latency audit 5154, record 3. Relay Cedar 7 ms. Relay Birch 16 ms."},
        {"record_id": "other", "rank": 4, "normalized_score": 0.80, "text": "Birch latency audit 9000, record 2. Relay Cedar 1 ms. Relay Birch 2 ms."},
    ]
    selected = select_retrieved_passages(question, passages, selected_top_k=3, kind="knowledge", policy="query_aware_v21")
    assert [row["record_id"] for row in selected] == ["r2"]


def test_query_aware_selector_preserves_multiple_requested_records() -> None:
    question = "Compare audit 77, record 1 and record 2."
    passages = [
        {"record_id": "r1", "rank": 1, "normalized_score": 0.9, "text": "Audit 77, record 1. Value 10."},
        {"record_id": "r2", "rank": 2, "normalized_score": 0.8, "text": "Audit 77, record 2. Value 20."},
        {"record_id": "r3", "rank": 3, "normalized_score": 0.7, "text": "Audit 77, record 3. Value 30."},
    ]
    selected = select_retrieved_passages(question, passages, selected_top_k=3, kind="knowledge", policy="query_aware_v21")
    assert {row["record_id"] for row in selected} == {"r1", "r2"}


def test_legacy_selector_is_unchanged() -> None:
    passages = [{"record_id": str(i), "rank": i, "normalized_score": 1 / i, "text": f"record {i}"} for i in range(1, 5)]
    selected = select_retrieved_passages("record 4", passages, selected_top_k=3, kind="knowledge", policy="legacy")
    assert [row["record_id"] for row in selected] == ["1", "2", "3"]


def test_query_aware_procedure_selection_deduplicates_equivalent_rules() -> None:
    passages = [
        {"record_id": "p1", "rank": 1, "normalized_score": 0.9, "text": "Choose the smaller latency."},
        {"record_id": "p2", "rank": 2, "normalized_score": 0.8, "text": "Choose the smaller latency."},
        {"record_id": "p3", "rank": 3, "normalized_score": 0.7, "text": "Choose the larger capacity."},
    ]
    selected = select_retrieved_passages("which is faster", passages, selected_top_k=3, kind="procedure", policy="query_aware_v21")
    assert [row["record_id"] for row in selected] == ["p1", "p3"]


def test_v21_procedural_corpus_is_large_valid_and_collision_heavy() -> None:
    packet = validate(CORPUS)
    assert packet["status"] == "PASS", packet["problems"]
    assert packet["reviewed_rows"] == 6120
    assert packet["split_counts"] == {"train": 4648, "dev": 736, "test": 736}
    manifest = json.loads((CORPUS / "manifests/dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["recovery_views"]["recovery_mix_rows"] == 5810
    assert len(manifest["design"]["procedure_diversity"]) == 14
    train = read_jsonl(CORPUS / "reviewed_splits/train.reviewed.jsonl")
    collision = [row for row in train if "retrieval_collision" in row.get("scenario_flags", [])]
    assert len(collision) > 4000
    assert all(len(row["evidence"]) >= 3 for row in collision)
    assert all(row["support_document_ids"][0] in row["retrieved_document_ids"] for row in collision)


def test_v21_training_mix_uses_train_only_preservation_rows() -> None:
    mix = read_jsonl(CORPUS / "curriculum_views/train_recovery_mix.reviewed.jsonl")
    preservation = [row for row in mix if str(row.get("id", "")).startswith("preserve_v21::")]
    assert len(preservation) == 1162
    assert all(row.get("intended_split") == "train" for row in preservation)
    assert all(str(row.get("mode") or "").casefold() != "procedure" for row in preservation)


def test_transfer_gap_diagnosis_identifies_collision_binding_failure() -> None:
    base = {
        "id": "p1", "mode": "procedure", "case_class": "supported", "reference": "Relay Birch.",
        "support_document_ids": ["target"],
        "procedures": [{"text": "Choose the smaller latency."}],
    }
    rows = [
        {**base, "condition": "oracle_rag", "answer": "Relay Birch.", "supported_answer_correct": True,
         "evidence": [{"document_id": "target", "text": "Audit 7, record 1. Relay Birch 9 ms. Relay Cedar 14 ms."}]},
        {**base, "condition": "real_rag", "answer": "Relay Cedar.", "supported_answer_correct": False,
         "support_retrieved": True, "support_in_prompt": True,
         "evidence": [
             {"document_id": "target", "text": "Audit 7, record 1. Relay Birch 9 ms. Relay Cedar 14 ms."},
             {"document_id": "sibling", "text": "Audit 7, record 2. Relay Birch 19 ms. Relay Cedar 10 ms."},
         ]},
    ]
    packet = diagnose(rows)
    assert packet["oracle_correctness"] == 1.0
    assert packet["real_correctness"] == 0.0
    assert packet["classification_counts"]["retrieval_collision_or_evidence_binding"] == 1


def test_v21_runner_and_production_config_use_query_aware_policy() -> None:
    runner = (ROOT / "scripts/phase_9_student_distillation/run_phase_9_procedural_recovery_v21.ps1").read_text(encoding="utf-8")
    assert "query_aware_v21" in runner
    assert "train_answerable_hard.reviewed.jsonl" in runner
    assert "train_recovery_mix.reviewed.jsonl" in runner

    general_runner = (ROOT / "scripts/phase_9_student_distillation/run_phase_9.ps1").read_text(encoding="utf-8")
    recovery_runner = (ROOT / "scripts/phase_9_student_distillation/run_phase_9_grounded_recovery.ps1").read_text(encoding="utf-8")
    assert 'RetrievalSelectionPolicy = "query_aware_v21"' in general_runner
    assert 'RetrievalSelectionPolicy = "query_aware_v21"' in recovery_runner
    assert '"--retrieval-selection-policy", $RetrievalSelectionPolicy' in general_runner
    assert '"--retrieval-selection-policy", $RetrievalSelectionPolicy' in recovery_runner

    phase9_cfg = json.loads((ROOT / "configs/phase_9_student_distillation.json").read_text(encoding="utf-8"))
    assert phase9_cfg["rag"]["retrieval_selection_policy"] == "query_aware_v21"
    cfg = json.loads((ROOT / "configs/phase_11_production_rag.json").read_text(encoding="utf-8"))
    assert cfg["retrieval"]["selection_policy"] == "query_aware_v21"
