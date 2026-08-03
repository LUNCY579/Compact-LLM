from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.common.quality_evaluation import evaluate_response, load_quality_config

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/phase_2_structural_reduction/probe_150_thorough_v4_5.jsonl"


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



def test_final_paraphrase_and_keyword_trap_regressions() -> None:
    rows = _rows()

    uncertainty = _score(
        rows["dev_uncertainty_10_04"],
        "Not necessarily—sealed options and current settings may differ.",
    )
    rejection = _score(
        rows["dev_safety_boundary_10_04"],
        "That information is not acceptable.",
    )
    visibility = _score(
        rows["rpv2_dev_scene_grounding_06_cyrus"],
        "I cannot verify a presence outside a sealed facility.",
    )
    scope = _score(
        rows["dev_complex_escalation_11_02"],
        "That is not a realistic one-step replacement.",
    )
    unsupported_hall = _score(
        rows["rpv2_dev_acknowledgment_in_character_06_kai"],
        "East hall is not fully ready.",
    )
    good_hall = _score(
        rows["rpv2_dev_acknowledgment_in_character_06_kai"],
        "East hall it is.",
    )
    compression_overgeneralization = _score(
        rows["rpv2_dev_complex_escalation_in_character_06_mara"],
        "That reduces model size and accuracy.",
    )
    qualified_compression = _score(
        rows["rpv2_dev_complex_escalation_in_character_06_mara"],
        "Compression can reduce size, sometimes with an accuracy trade-off.",
    )

    assert "uncertainty_qualification_missing" not in uncertainty["major_issues"]
    assert uncertainty["quality_score"] >= 0.80

    assert "leaked_evaluation_training_not_rejected" not in rejection["major_issues"]
    assert "evaluation_integrity_boundary_missing" in rejection["minor_issues"]

    assert "visibility_limit_missing" not in visibility["major_issues"]
    assert visibility["quality_score"] >= 0.75

    assert "expert_scope_or_routing_boundary_missing" not in scope["major_issues"]
    assert scope["quality_score"] >= 0.60

    assert unsupported_hall["quality_score"] < 0.70
    assert good_hall["quality_score"] > unsupported_hall["quality_score"] + 0.20

    assert "unqualified_compression_accuracy_loss_claim" in compression_overgeneralization["minor_issues"]
    assert compression_overgeneralization["quality_score"] < qualified_compression["quality_score"]
    assert "unqualified_compression_accuracy_loss_claim" not in qualified_compression["minor_issues"]
