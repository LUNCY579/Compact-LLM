from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/phase_2_layer_selection/rescore_phase_2b_artifacts.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_generation(path: Path, reply: str) -> None:
    write_jsonl(
        path / "results.jsonl",
        [{
            "id": "case", "category": "follow_up", "reply": reply,
            "generated_tokens": max(1, len(reply.split())),
            "visible_generated_tokens": max(1, len(reply.split())),
            "elapsed_seconds": 0.1,
            "quality_evaluation_version": "3.1",
            "quality_score": 0.4,
            "major_issues": ["semantically_unresponsive_output"],
        }],
    )
    (path / "summary.json").write_text(
        json.dumps({"quality_evaluation_version": "3.1", "case_count": 1}),
        encoding="utf-8",
    )


def test_phase2b_in_place_migration_reuses_replies_and_archives_decisions(tmp_path: Path) -> None:
    benchmark = tmp_path / "probe_v4.jsonl"
    write_jsonl(
        benchmark,
        [{
            "id": "case", "category": "follow_up",
            "messages": [{"role": "user", "content": "I reduced the context window."}],
            "reference": "Did it improve?",
            "response_contract": {
                "schema_version": "4.3", "open_ended": False,
                "min_tokens": 1, "max_tokens": 20,
                "question_required": True,
                "question_missing_label": "required_follow_up_question_missing",
                "requirements": [], "prohibitions": [],
            },
        }],
    )
    run_root = tmp_path / "run"
    reference_generation = run_root / "reference/evidence/generation"
    candidate_evidence = run_root / "evidence/stage_01/raw_drop_4"
    candidate_generation = candidate_evidence / "generation"
    write_generation(reference_generation, "Did it improve?")
    write_generation(candidate_generation, "Did it improve?")

    (run_root / "model_catalog.json").write_text(json.dumps({
        "run_configuration": {
            "max_new_tokens": 32, "recovery_steps": 50, "recovery_seed": 11,
            "qlora": True, "gradient_checkpointing": True,
        }
    }), encoding="utf-8")
    (run_root / "candidate_registry.json").write_text(json.dumps({
        "candidates": [{
            "stage": 1, "variant_id": "raw_drop_4", "variant_kind": "raw deletion",
            "evidence_dir": str(candidate_evidence), "parent_model": str(tmp_path / "parent"),
            "dropped_original_layer_id": 4, "retained_original_layer_ids": [0, 1, 2, 3, 5],
        }]
    }), encoding="utf-8")
    frontier = run_root / "frontier/stage_01"
    frontier.mkdir(parents=True)
    (frontier / "stage_result.json").write_text("{}", encoding="utf-8")

    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--run-root", str(run_root),
            "--benchmark", str(benchmark),
            "--in-place", "--prepare-resume",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    reference_row = json.loads((reference_generation / "results.jsonl").read_text().splitlines()[0])
    candidate_row = json.loads((candidate_generation / "results.jsonl").read_text().splitlines()[0])
    assert reference_row["reply"] == "Did it improve?"
    assert candidate_row["reply"] == "Did it improve?"
    assert reference_row["quality_evaluation_version"] == "4.3"
    assert candidate_row["quality_evaluation_version"] == "4.3"
    assert candidate_row["major_issues"] == []
    assert (candidate_evidence / "output_deviation.json").exists()
    assert (candidate_evidence / "evidence_provenance.json").exists()
    assert not (run_root / "model_catalog.json").exists()
    assert not (run_root / "candidate_registry.json").exists()
    manifest = json.loads((run_root / "PHASE2B_QUALITY_V4_3_MIGRATION.json").read_text())
    assert manifest["benchmark_contract_audit"] == {"case_count": 1, "contracted_case_count": 1}
    assert manifest["rescored_count"] == 2


def test_phase2b_migration_rejects_legacy_uncontracted_benchmark(tmp_path: Path) -> None:
    benchmark = tmp_path / "legacy.jsonl"
    write_jsonl(benchmark, [{"id": "case", "messages": [{"role": "user", "content": "Hi"}]}])
    run_root = tmp_path / "run"
    write_generation(run_root / "reference/evidence/generation", "Hello.")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-root", str(run_root), "--benchmark", str(benchmark)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "explicit Evaluator-v4.3 benchmark" in (completed.stderr + completed.stdout)
