from __future__ import annotations

import json
from pathlib import Path

from scripts.common.rescore_quality_artifacts import rescore


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_rescore_replaces_stale_v3_fields_without_generation(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.jsonl"
    write_jsonl(
        benchmark,
        [
            {
                "id": "case",
                "category": "contextual_follow_up",
                "messages": [{"role": "user", "content": "I reduced the context window."}],
                "reference": "How did latency change?",
                "response_contract": {
                    "schema_version": "4.4",
                    "open_ended": False,
                    "min_tokens": 1,
                    "max_tokens": 20,
                    "question_required": True,
                    "question_missing_label": "required_follow_up_question_missing",
                    "requirements": [],
                    "prohibitions": [],
                },
            }
        ],
    )
    generation = tmp_path / "generation"
    results = generation / "results.jsonl"
    write_jsonl(
        results,
        [
            {
                "id": "case",
                "category": "contextual_follow_up",
                "reply": "You gives. you gives. You can you is the duplicate.",
                "generated_tokens": 10,
                "visible_generated_tokens": 10,
                "elapsed_seconds": 0.2,
                "quality_evaluation_version": "3.1",
                "quality_score": 0.963,
                "semantic_coherence_score": 1.0,
                "prompt_relevance_score": 1.0,
                "intent_fulfillment_score": 1.0,
                "semantic_integrity_diagnostics": {"stale": True},
                "major_issues": ["semantically_unresponsive_output"],
            }
        ],
    )
    (generation / "summary.json").write_text(
        json.dumps({"quality_evaluation_version": "3.1", "case_count": 1}),
        encoding="utf-8",
    )

    output = tmp_path / "rescored"
    result_out, summary_out = rescore(
        benchmark,
        results,
        output,
        Path("configs/quality_evaluation.json").resolve(),
    )
    row = json.loads(result_out.read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads(summary_out.read_text(encoding="utf-8"))

    assert row["quality_evaluation_version"] == "4.4"
    assert row["quality_score"] == 0.0
    assert "semantic_integrity_diagnostics" not in row
    assert "prompt_relevance_score" not in row
    assert "intent_fulfillment_score" not in row
    assert "semantically_unresponsive_output" not in row["major_issues"]
    assert "broken_clause_sequence" in row["critical_failures"]
    assert row["elapsed_seconds"] == 0.2
    assert summary["quality_evaluation_version"] == "4.4"
    assert summary["critical_defect_case_rate"] == 1.0
    assert summary["mechanical_failure_case_rate"] == 1.0
    assert summary["rescored_from_quality_evaluation_version"] == "3.1"
