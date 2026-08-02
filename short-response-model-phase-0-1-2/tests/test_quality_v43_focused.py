from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.common.quality_evaluation import evaluate_response, load_quality_config

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/phase_2_structural_reduction/probe_150_thorough_v4_3.jsonl"


def _rows() -> dict[str, dict]:
    return {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in BENCHMARK.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _score(record: dict, reply: str) -> dict:
    tokens = max(1, len(reply.split()))
    return evaluate_response(record, reply, tokens, visible_model_tokens=tokens, config=load_quality_config())


def test_category_and_reference_do_not_change_prompt_adequacy() -> None:
    record = _rows()["dev_naturalness_10_02"]
    reply = "That is so small it is useless."
    baseline = _score(record, reply)
    variant = copy.deepcopy(record)
    variant["category"] = "deliberately_wrong"
    variant["reference"] = "Completely unrelated reference text."
    changed = _score(variant, reply)
    for field in (
        "quality_score", "quality_dimensions", "quality_defects",
        "prompt_function_alignment_score", "prompt_adequacy_dimension_scores",
    ):
        assert changed[field] == baseline[field]


def test_false_high_and_false_low_regressions() -> None:
    rows = _rows()
    bad_seed = _score(rows["dev_empathy_11_01"], "Use the same seed every time.")
    bad_adapter = _score(rows["dev_naturalness_10_02"], "That is so small it is useless.")
    good_flops = _score(rows["dev_safety_boundary_11_00"], "That is not sufficient; it does not guarantee.")
    good_sealed = _score(rows["dev_context_consistency_10_00"], "The sealed set must stay untouched.")
    partial_visibility = _score(rows["rpv2_dev_scene_grounding_06_zara"], "Not clearly.")

    assert bad_seed["quality_score"] < 0.55
    assert bad_adapter["quality_score"] < 0.55
    assert "flops_only_speedup_claim_not_rejected" not in good_flops["major_issues"]
    assert "flops_only_speedup_claim_not_rejected" not in good_flops["minor_issues"]
    assert "sealed_set_isolation_fact_missing" not in good_sealed["major_issues"]
    assert "visibility_limit_missing" not in partial_visibility["major_issues"]


def test_prompt_subject_reversal_is_lower_than_direct_banter() -> None:
    record = _rows()["rpv2_dev_playful_banter_06_noa"]
    direct = _score(record, "You look heroic enough.")
    reversed_subject = _score(record, "I look so admirable.")
    assert direct["quality_score"] > reversed_subject["quality_score"] + 0.15
