from __future__ import annotations

from pathlib import Path

from scripts.phase_9_student_distillation.select_grounded_sft_candidate import (
    SELECTION_POLICY_REVISION,
    build_selection_packet,
)


def make_summary(
    score: float,
    *,
    decode_failures: int = 0,
    prompt_failures: int = 0,
    missing_procedure_coverage: bool = False,
    model: str = "model",
    adapter: str = "adapter",
) -> dict:
    # Set every weighted capability component to the same value so the aggregate
    # capability score is exactly `score` while preserving all behavior buckets.
    procedure_cases = 0 if missing_procedure_coverage else 2
    procedure_correct = 0 if missing_procedure_coverage else max(1, round(2 * score))
    successes = max(1, round(2 * score)) if score > 0 else 0
    return {
        "model": model,
        "adapter": adapter,
        "evaluation_valid": True,
        "conditions": {
            "oracle_rag": {
                "case_count": 12,
                "supported_case_count": 2,
                "supported_correct_count": successes,
                "supported_answer_correctness": score,
                "evidence_faithfulness": score,
                "unsupported_case_count": 2,
                "correct_abstention_count": successes,
                "correct_abstention_rate": score,
                "unsupported_confident_answer_rate": 1.0 - score,
                "conflict_case_count": 2,
                "conflict_detected_count": successes,
                "conflict_detection_rate": score,
                "procedure_case_count": procedure_cases,
                "procedure_correct_count": procedure_correct,
                "procedure_correctness": score if procedure_cases else None,
                "social_case_count": 2,
                "social_usable_count": successes,
                "social_usable_rate": score,
                "prompt_integrity_failure_count": prompt_failures,
                "decode_integrity_failure_count": decode_failures,
                "median_completion_tokens": 8,
                "mean_generation_seconds": 0.25,
            },
            "adversarial_passthrough_rag": {
                "case_count": 2,
                "adversarial_applicable_count": 2,
                "adversarial_success_count": successes,
                "injection_resistance_rate": score,
                "prompt_integrity_failure_count": prompt_failures,
                "decode_integrity_failure_count": decode_failures,
            },
        },
    }


def select(initialization: dict, *candidates: tuple[str, dict]) -> dict:
    rows = [
        (name, Path(f"/{name}/summary.json"), summary)
        for name, summary in candidates
    ]
    return build_selection_packet(
        initialization,
        rows,
        initialization_summary="/initialization/summary.json",
    )


def test_dirty_stage_b_decode_integrity_is_warning_and_perfect_candidate_is_selected() -> None:
    initialization = make_summary(0.25, decode_failures=2)
    weak = make_summary(0.66, model="warm", adapter="weak")
    strong = make_summary(0.85, model="warm", adapter="strong")
    perfect = make_summary(1.0, model="warm", adapter="perfect")

    packet = select(
        initialization,
        ("lr5e5_s600", weak),
        ("lr5e5_s1200", strong),
        ("lr1e4_s1200", perfect),
    )

    assert packet["selection_policy_revision"] == SELECTION_POLICY_REVISION
    assert packet["status"] == "SFT_CANDIDATE_SELECTED"
    assert packet["selected"]["name"] == "lr1e4_s1200"
    assert packet["initialization_blocking_issues"] == []
    assert packet["initialization_integrity_is_selection_blocking"] is False
    assert "oracle_rag:decode_integrity_failure" in packet["initialization_integrity_warnings"]
    assert all("initialization_reference_invalid" not in row["issues"] for row in packet["candidates"])


def test_dirty_stage_b_prompt_integrity_is_also_nonblocking() -> None:
    packet = select(
        make_summary(0.25, prompt_failures=1),
        ("winner", make_summary(1.0)),
    )
    assert packet["status"] == "SFT_CANDIDATE_SELECTED"
    assert packet["selected"]["name"] == "winner"
    assert any("prompt_integrity_failure" in issue for issue in packet["initialization_integrity_warnings"])


def test_candidate_own_decode_integrity_remains_fatal() -> None:
    packet = select(
        make_summary(0.25, decode_failures=2),
        ("broken_candidate", make_summary(1.0, decode_failures=1)),
        ("clean_candidate", make_summary(0.85)),
    )
    broken = next(row for row in packet["candidates"] if row["name"] == "broken_candidate")
    assert broken["valid"] is False
    assert any("decode_integrity_failure" in issue for issue in broken["issues"])
    assert packet["selected"]["name"] == "clean_candidate"


def test_initialization_coverage_failure_remains_a_reference_blocker() -> None:
    packet = select(
        make_summary(0.25, missing_procedure_coverage=True),
        ("otherwise_perfect", make_summary(1.0)),
    )
    assert "oracle_rag:zero_procedure_cases" in packet["initialization_blocking_issues"]
    assert packet["status"] == "NO_VALID_QUALITY_FRONTIER"
    assert packet["selected"] is None
    assert "initialization_reference_invalid" in packet["candidates"][0]["issues"]


def test_reuse_existing_refreshes_derived_stage_d_and_stage_e_selections() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/phase_9_student_distillation/run_phase_9_grounded_recovery.ps1").read_text(
        encoding="utf-8"
    )
    assert "Recomputing the derived SFT selection from existing candidate summaries" in runner
    assert "Refreshing the non-degenerate Oracle frontier" in runner
    assert "Recomputing the derived DEV selection under the current selector policy" in runner
    assert "Refreshing the final refinement selection on DEV only" in runner


def test_reselect_existing_utility_repairs_stale_selection_without_training(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    run_root = tmp_path / "run"
    init_dir = run_root / "evaluation_warm_initialization_oracle"
    init_dir.mkdir(parents=True)
    (init_dir / "summary.json").write_text(
        json.dumps(make_summary(0.25, decode_failures=2)), encoding="utf-8"
    )
    for name, score in (("lr5e5_s600", 0.66), ("lr5e5_s1200", 0.85), ("lr1e4_s1200", 1.0)):
        path = run_root / "warm_sft_sweep" / name / "evaluation_oracle"
        path.mkdir(parents=True)
        (path / "summary.json").write_text(json.dumps(make_summary(score)), encoding="utf-8")

    sweep = run_root / "warm_sft_sweep"
    (sweep / "selection.json").write_text(
        json.dumps({"status": "NO_VALID_QUALITY_FRONTIER", "selected": None}), encoding="utf-8"
    )
    (sweep / "SELECTION_REPORT.md").write_text("stale", encoding="utf-8")

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/phase_9_student_distillation/reselect_existing_grounded_sft.py"),
            "--run-root",
            str(run_root),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    repaired = json.loads((sweep / "selection.json").read_text(encoding="utf-8"))
    assert repaired["status"] == "SFT_CANDIDATE_SELECTED"
    assert repaired["selected"]["name"] == "lr1e4_s1200"
    assert "phase9_sft_selection_gate_v1_7_7" in (sweep / "SELECTION_REPORT.md").read_text(encoding="utf-8")
