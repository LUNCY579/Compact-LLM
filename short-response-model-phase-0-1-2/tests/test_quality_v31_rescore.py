from __future__ import annotations

import json
from pathlib import Path

from scripts.common.rescore_quality_artifacts import rescore


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_rescore_replaces_stale_quality_fields_without_generation(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.jsonl"
    write_jsonl(
        benchmark,
        [
            {
                "id": "case",
                "category": "contextual_follow_up",
                "messages": [
                    {"role": "user", "content": "I reduced the context window."},
                    {"role": "assistant", "content": "How did latency change?"},
                ],
                "reference": "How did latency change?",
                "expected": {"min_tokens": 1, "max_tokens": 20},
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
                "quality_evaluation_version": "3.0",
                "quality_score": 0.963,
                "semantic_coherence_score": 1.0,
                "prompt_relevance_score": 1.0,
                "intent_fulfillment_score": 1.0,
                "semantic_integrity_diagnostics": {"stale": True},
            }
        ],
    )
    (generation / "summary.json").write_text(
        json.dumps({"quality_evaluation_version": "3.0", "case_count": 1}),
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

    assert row["quality_evaluation_version"] == "3.1"
    assert row["quality_score"] == 0.0
    assert row["semantic_integrity_diagnostics"] != {"stale": True}
    assert "unrecoverable_semantic_word_salad" in row["critical_failures"]
    assert row["elapsed_seconds"] == 0.2
    assert summary["quality_evaluation_version"] == "3.1"
    assert summary["semantic_collapse_case_rate"] == 1.0
    assert summary["rescored_from_quality_evaluation_version"] == "3.0"
