from __future__ import annotations

from rineya_cross_scale.promotion import evaluate_two_control
from rineya_cross_scale.v2 import (
    expand_alpha_plans,
    first_wave_plans,
    load_frozen_grids,
    one_layer_plans,
    rank_two_control,
    successive_halving_names,
)


def summary(overall: float, family: float, *, dataset: str = "x"):
    return {
        "dataset_sha256": dataset,
        "items": 2,
        "overall_score": overall,
        "scenario_macro_score": overall,
        "worst_family_score": family,
        "families": {"f": {"score": family}},
        "invalid_generation_rate": 0.0,
        "completion_rate": 1.0,
        "format_adherence": {"overall": 1.0},
    }


def test_frozen_grids_match_amendment() -> None:
    grids = load_frozen_grids()
    assert grids["alphas"] == [0.1, 0.15, 0.2]
    assert grids["first_wave_alpha"] == 0.15
    assert grids["cosine_thresholds"] == [0.3, 0.5, 0.7]
    assert grids["ties_keep_fractions"] == [0.2, 0.5, 0.8]
    assert grids["dare_seeds"] == [17, 23, 42]
    assert grids["mergedev_fast_must_remain_frozen"] is True


def test_two_control_requires_both_beats() -> None:
    supa = summary(0.70, 0.70)
    consensus = summary(0.72, 0.70)
    beats_supa_only = evaluate_two_control(supa, consensus, summary(0.71, 0.71))
    assert not beats_supa_only["eligible"]
    beats_both = evaluate_two_control(supa, consensus, summary(0.80, 0.80))
    assert beats_both["eligible"]
    regression = evaluate_two_control(supa, consensus, summary(0.90, 0.50))
    assert not regression["eligible"]


def test_first_wave_is_alpha_015_only() -> None:
    plans = first_wave_plans()
    assert plans
    assert all(abs(plan.coefficient - 0.15) < 1e-12 for plan in plans)
    families = {plan.merge_family for plan in plans}
    assert families == {
        "CONSENSUS_SIGN",
        "CONSENSUS_COSINE",
        "TIES",
        "DARE",
        "DARETIES",
    }


def test_expand_alpha_respects_selected_families() -> None:
    plans = expand_alpha_plans({"TIES"})
    assert plans
    assert all(plan.merge_family == "TIES" for plan in plans)
    assert {plan.coefficient for plan in plans} == {0.1, 0.2}


def test_one_layer_count_is_frozen() -> None:
    plans = one_layer_plans()
    assert len(plans) == 56
    assert all(plan.parameters["layers"][0] in range(28) for plan in plans)


def test_successive_halving_prefers_eligible() -> None:
    supa = summary(0.5, 0.5)
    consensus = summary(0.55, 0.5)
    ranked = rank_two_control(
        supa,
        consensus,
        {
            "weak": summary(0.52, 0.52),
            "strong": summary(0.80, 0.80),
            "mid": summary(0.70, 0.70),
        },
    )
    assert successive_halving_names(ranked, 2) == ["strong", "mid"]
