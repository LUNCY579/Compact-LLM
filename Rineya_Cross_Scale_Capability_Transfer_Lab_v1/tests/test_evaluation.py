from __future__ import annotations

import json

import pytest

from rineya_cross_scale.evaluation.banks import FAMILIES, build_banks, generate_bank
from rineya_cross_scale.evaluation.scoring import (
    aggregate_predictions,
    score_response,
)
from rineya_cross_scale.leakage import LeakageError, guard_non_sealed_path


def test_scoring_modes() -> None:
    assert score_response("  Electron. ", {"type": "exact", "answers": ["electron"]})[
        "correct"
    ]
    assert score_response("40 plus 2 equals 42.", {"type": "numeric", "value": 42})[
        "correct"
    ]
    assert score_response(
        "Answer: YES.", {"type": "choice", "allowed": ["YES", "NO"], "value": "YES"}
    )["correct"]
    assert score_response(
        '{"ok": true}', {"type": "json_exact", "value": {"ok": True}}
    )["correct"]
    assert score_response(
        "The plant may wilt after a long dry period.",
        {"type": "keywords", "phrase": "it wilts"},
    )["correct"]
    assert score_response(
        "Warm greetings to Maya",
        {
            "type": "constraints",
            "minimum_words": 3,
            "maximum_words": 7,
            "contains": ["Maya"],
            "forbids": ["!"],
        },
    )["correct"]


def test_bank_has_all_families_and_clustered_paraphrases() -> None:
    rows = generate_bank("mergedev", 314159, 4)
    assert {row["family"] for row in rows} == set(FAMILIES)
    paraphrases = [row for row in rows if row["family"] == "paraphrase_robustness"]
    scenarios = {}
    for row in paraphrases:
        scenarios.setdefault(row["scenario_id"], []).append(row)
    assert len(scenarios) == 4
    assert all(len(group) == 3 for group in scenarios.values())


def test_build_banks_has_zero_prompt_overlap(tmp_path) -> None:
    output = tmp_path / "public"
    sealed = tmp_path / "private" / "sealed.jsonl"
    manifest = build_banks(
        output,
        sealed,
        specifications={
            "calibration": (271828, 2),
            "mergedev": (314159, 2),
            "dev_full": (161803, 3),
            "sealed": (141421, 3),
        },
    )
    assert not any(manifest["prompt_hash_overlap"].values())
    assert sealed.is_file()
    assert json.loads((output / "bank_manifest.json").read_text())["sealed_answers_not_printed"]


def test_production_bank_specification_has_zero_prompt_overlap(tmp_path) -> None:
    manifest = build_banks(
        tmp_path / "public",
        tmp_path / "private" / "sealed.jsonl",
        specifications={
            "calibration": (271828, 4),
            "mergedev": (314159, 4),
            "dev_full": (161803, 12),
            "sealed": (141421, 16),
        },
    )
    assert not any(manifest["prompt_hash_overlap"].values())


def test_leakage_guard_rejects_invisible_limit_test2(tmp_path) -> None:
    forbidden = tmp_path / "Rineya_Invisible_Limit_Lab_v1" / "data" / "test2_answer_key.jsonl"
    with pytest.raises(LeakageError, match="TEST2"):
        guard_non_sealed_path(forbidden)


def test_aggregate_preserves_family_and_scenario_levels() -> None:
    rows = [
        {
            "family": "a",
            "scenario_id": "s1",
            "score": {"score": 1.0},
            "latency_seconds": 1.0,
            "generated_tokens": 2,
        },
        {
            "family": "b",
            "scenario_id": "s1",
            "score": {"score": 0.0},
            "latency_seconds": 1.0,
            "generated_tokens": 2,
        },
    ]
    summary = aggregate_predictions(rows)
    assert summary["overall_score"] == 0.5
    assert summary["scenario_macro_score"] == 0.5
    assert summary["worst_family_score"] == 0.0
