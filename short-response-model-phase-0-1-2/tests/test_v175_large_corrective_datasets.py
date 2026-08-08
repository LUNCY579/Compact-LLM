from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.phase_9_student_distillation.validate_rag_reasoning_corpus import validate as validate_rag
from scripts.phase_9b_memory_reconstruction.validate_memory_reconstruction_corpus import validate as validate_recon
from scripts.phase_9_student_distillation.rebalance_behavior_curriculum import PROFILES, integer_quotas

ROOT = Path(__file__).resolve().parents[1]
RAG = ROOT / "data/phase_9_student_distillation/rag_reasoning_v3_large"
RECON = ROOT / "data/phase_9b_memory_reconstruction/reconstruction_corpus_v2_large"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v175_rag_large_corpus_is_valid_and_answer_heavy() -> None:
    packet = validate_rag(RAG)
    assert packet["status"] == "PASS", packet["problems"]
    assert packet["reviewed_rows"] == 5900
    assert packet["split_counts"] == {"train": 4410, "dev": 752, "test": 738}
    assert packet["knowledge_documents"] == 6000
    assert packet["procedures"] == 200

    rows = _read_jsonl(RAG / "prompts.all.jsonl")
    modes = Counter(row["mode"] for row in rows)
    decisions = Counter(row["decision"] for row in rows)
    assert modes["procedure"] == 1600
    assert decisions["answer"] == 4200
    assert all(row["review_status"] == "auto_verified" for row in rows)


def test_v175_recovery_profile_dedicates_80_percent_to_supported_and_procedure() -> None:
    quotas = integer_quotas(1000, PROFILES["support_procedure_recovery"])
    assert quotas["supported"] == 550
    assert quotas["procedure"] == 250
    assert quotas["supported"] + quotas["procedure"] == 800


def test_v175_rag_has_dedicated_corrective_views() -> None:
    recovery = _read_jsonl(RAG / "curriculum_views/support_procedure_recovery.train.reviewed.jsonl")
    procedures = _read_jsonl(RAG / "curriculum_views/procedure_bootstrap.train.reviewed.jsonl")
    assert len(recovery) == 3543
    assert len(procedures) == 900
    assert all(row["mode"] == "procedure" and row["decision"] == "answer" for row in procedures)
    assert sum(row["decision"] == "answer" for row in recovery) > 3000


def test_v175_memory_reconstruction_large_corpus_is_valid() -> None:
    packet = validate_recon(RECON)
    assert packet["status"] == "PASS", packet["problems"]
    assert packet["memory_records"] == 1500
    assert packet["reviewed_rows"] == 3600
    assert packet["split_counts"] == {"train": 2700, "dev": 456, "test": 444}
    assert packet["decision_counts"] == {
        "accept": 900,
        "adapt": 1200,
        "conflict": 300,
        "insufficient": 600,
        "reject": 600,
    }
    assert packet["counterfactual_pairs"] == 600
    assert packet["evaluator_candidate_rows"] == 14400


def test_v175_memory_rows_have_current_and_source_state() -> None:
    rows = _read_jsonl(RECON / "reviewed_memory_reconstruction.jsonl")
    assert all(row.get("current_state") for row in rows)
    assert all(row["review_status"] == "auto_verified" for row in rows)
    assert all(memory.get("source_state") for row in rows for memory in row["memories"])
    assert all(isinstance(memory.get("validity_conditions"), dict) for row in rows for memory in row["memories"])
    modes = Counter(row["mode"] for row in rows)
    assert modes["procedure"] > modes["grounded_factual"] > 0


def test_v175_runners_default_to_large_corpora() -> None:
    p9 = (ROOT / "scripts/phase_9_student_distillation/run_phase_9_grounded_recovery.ps1").read_text(encoding="utf-8")
    p9b = (ROOT / "scripts/phase_9b_memory_reconstruction/run_phase_9b.ps1").read_text(encoding="utf-8")
    p10 = (ROOT / "scripts/phase_10_student_architecture/run_phase_10.ps1").read_text(encoding="utf-8")
    p11 = (ROOT / "scripts/phase_11_production_rag/run_phase_11.ps1").read_text(encoding="utf-8")
    assert r"rag_reasoning_v3_large" in p9
    assert 'BehaviorProfile = "support_procedure_recovery"' in p9
    assert r"reconstruction_corpus_v2_large" in p9b
    assert r"rag_reasoning_v3_large" in p9b
    assert r"rag_reasoning_v3_large" in p10 and r"reconstruction_corpus_v2_large" in p10
    assert r"rag_reasoning_v3_large" in p11 and r"reconstruction_corpus_v2_large" in p11
