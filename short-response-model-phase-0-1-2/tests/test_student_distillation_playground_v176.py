from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND = ROOT / "Student_Distillation_Playground_v1.0"
SCRIPT = PLAYGROUND / "student_playground.py"

spec = importlib.util.spec_from_file_location("student_playground_v176", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _rows(name: str):
    path = PLAYGROUND / "datasets" / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_showcase_counts_and_unique_ids() -> None:
    logic = _rows("logic_games.jsonl")
    evidence = _rows("evidence_reasoning.jsonl")
    memory = _rows("memory_reconstruction.jsonl")
    all_rows = _rows("showcase_suite.jsonl")
    assert (len(logic), len(evidence), len(memory), len(all_rows)) == (10, 12, 12, 34)
    ids = [row["id"] for row in all_rows]
    assert len(ids) == len(set(ids))
    assert {row["task_type"] for row in all_rows} == {"logic", "evidence", "memory"}


def test_showcase_ids_do_not_overlap_official_phase_9_or_9b() -> None:
    showcase_ids = {row["id"] for row in _rows("showcase_suite.jsonl")}
    official_ids: set[str] = set()
    phase9 = ROOT / "data" / "phase_9_student_distillation" / "rag_reasoning_v2" / "prompts.all.jsonl"
    phase9b = ROOT / "data" / "phase_9b_memory_reconstruction" / "reconstruction_corpus_v1" / "reviewed_memory_reconstruction.jsonl"
    for path in (phase9, phase9b):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                official_ids.add(json.loads(line)["id"])
    assert showcase_ids.isdisjoint(official_ids)


def test_score_final_answer_handles_answer_abstain_and_forbidden_content() -> None:
    answer_case = {
        "expected": {"decision": "answer", "answer": "Dock 4", "required_phrases": ["Dock 4"]}
    }
    assert module.score_final_answer(answer_case, "Dock 4.")["correct"] is True

    abstain_case = {
        "expected": {"decision": "abstain", "answer": module.ABSTENTION_TEXT}
    }
    assert module.score_final_answer(abstain_case, module.ABSTENTION_TEXT)["correct"] is True

    forbidden_case = {
        "expected": {
            "decision": "answer",
            "answer": "4.1",
            "required_phrases": ["4.1"],
            "forbidden_phrases": ["3.8"],
        }
    }
    assert module.score_final_answer(forbidden_case, "Current version is 4.1.")["correct"] is True
    assert module.score_final_answer(forbidden_case, "It might be 3.8 or 4.1.")["correct"] is False


def test_memory_cases_have_stateful_reconstruction_targets() -> None:
    rows = _rows("memory_reconstruction.jsonl")
    decisions = set()
    for row in rows:
        assert isinstance(row.get("current_state"), dict)
        assert isinstance(row.get("memories"), list)
        target = row["expected"].get("reconstruction")
        assert isinstance(target, dict)
        decisions.add(target["decision"])
        for memory in row["memories"]:
            assert "source_state" in memory
            assert "validity_conditions" in memory
            assert "trust_level" in memory
    assert {"accept", "adapt", "reject", "conflict", "insufficient"}.issubset(decisions)


def test_summary_is_explicitly_non_gating() -> None:
    results = [
        {
            "task_type": "logic",
            "difficulty": "easy",
            "correct": True,
            "expected": {"decision": "answer"},
            "timing": {"total_case_seconds": 0.1, "generation_seconds": 0.08, "tokens_per_second": 50.0},
            "peak_vram_bytes": None,
            "diagnostics": {},
        },
        {
            "task_type": "evidence",
            "difficulty": "medium",
            "correct": False,
            "expected": {"decision": "abstain"},
            "timing": {"total_case_seconds": 0.2, "generation_seconds": 0.18, "tokens_per_second": 40.0},
            "peak_vram_bytes": None,
            "diagnostics": {"grounding_supported": False},
        },
    ]
    summary = module._summarize(results)
    assert summary["official_gate_evidence"] is False
    assert summary["case_count"] == 2
    assert summary["accuracy"] == 0.5


def test_wrapper_and_docs_emphasize_separation() -> None:
    readme = (PLAYGROUND / "README.md").read_text(encoding="utf-8")
    wrapper = (PLAYGROUND / "run_playground.ps1").read_text(encoding="utf-8")
    assert "not official phase-gate evidence" in readme
    assert "before Phase 10" in readme
    assert "--memory-mode" in wrapper
