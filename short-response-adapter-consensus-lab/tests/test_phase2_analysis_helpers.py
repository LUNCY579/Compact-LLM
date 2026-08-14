from __future__ import annotations

import pytest

from experiments.phase2_mechanism.analyze_selected_free_run import first_divergence, holm_adjust
from experiments.phase2_mechanism.analyze_robust_soft_controls import (
    clipped_candidate_probabilities,
    hard_winner as robust_hard_winner,
)
from experiments.phase2_mechanism.fit_adaptive_teacher_forced import hard_winner
from experiments.phase2_mechanism.make_paper_figures import (
    calibration_robustness_figure,
    coalition_heatmap_figure,
    committee_scaling_figure,
    quality_compute_figure,
    rescue_harm_figure,
    teacher_free_figure,
)
from experiments.phase2_mechanism.make_paper_tables import (
    load_optional_complete,
    selected_committee_loo_table,
)
from experiments.phase2_mechanism.make_final_synthesis_facts import (
    HYPOTHESIS_EVIDENCE_MAP,
    QUESTION_EVIDENCE_MAP,
    adaptive_section,
    calibrated_section,
    committee_section,
    cross_section,
    selected_section,
)
from experiments.phase2_mechanism.run_remaining_pipeline import commands
from experiments.phase2_mechanism.verify_final_synthesis import (
    REQUIRED_FINAL_REFERENCES,
    SYNTHESIS_DOCUMENTS,
    interim_marker_hits,
)

import numpy as np


def test_first_divergence_handles_prefix_and_identity() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [9, 2, 3]) == 0
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_holm_adjustment_is_order_safe_and_monotone() -> None:
    raw = [0.04, 0.001, 0.02]
    adjusted = holm_adjust(raw)
    assert adjusted == pytest.approx([0.04, 0.003, 0.04])
    assert all(value >= source for value, source in zip(adjusted, raw, strict=True))


def test_cached_hard_winner_uses_confidence_tie_break() -> None:
    token = {
        "candidate_token_ids": [10, 20],
        "candidate_logits_by_expert": [
            [3.0, 2.0],
            [1.0, 4.0],
        ],
        "expert_metrics": [
            {"top1_token_id": 10},
            {"top1_token_id": 20},
        ],
    }
    winner, agreement = hard_winner(token, [0, 1])
    assert winner == 20
    assert agreement == 0.5


def test_confidence_clipping_caps_top1_and_renormalizes() -> None:
    probabilities = np.asarray([[0.8, 0.15, 0.05], [0.4, 0.35, 0.25]])
    candidate_ids = np.asarray([10, 20, 30])
    top1_ids = np.asarray([10, 10])
    top1_probabilities = np.asarray([0.8, 0.4])

    clipped, normalizers = clipped_candidate_probabilities(
        probabilities, candidate_ids, top1_ids, top1_probabilities, 0.5
    )

    assert normalizers == pytest.approx([0.7, 1.0])
    assert clipped.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert clipped[0] == pytest.approx([0.5 / 0.7, 0.15 / 0.7, 0.05 / 0.7])
    assert clipped[1] == pytest.approx(probabilities[1])


def test_validation_weights_can_change_hard_vote_without_changing_proposals() -> None:
    candidate_ids = np.asarray([10, 20])
    top1_ids = np.asarray([10, 20, 20])
    raw_log_probabilities = np.log(
        np.asarray([[0.9, 0.1], [0.4, 0.6], [0.4, 0.6]], dtype=float)
    )

    assert robust_hard_winner(candidate_ids, top1_ids, raw_log_probabilities) == 1
    assert (
        robust_hard_winner(
            candidate_ids,
            top1_ids,
            raw_log_probabilities,
            np.asarray([10.0, 0.1, 0.1]),
        )
        == 0
    )


def test_committee_figure_separates_free_running_estimand() -> None:
    methods = ("hard_plurality", "probability_mean", "calibrated_probability_mean")
    teacher = {
        "committee_scaling": [
            {
                "method": method,
                "k": k,
                "mean_example_accuracy": 0.1 + 0.01 * k,
                "best_example_accuracy": 0.12 + 0.01 * k,
            }
            for method in methods
            for k in range(1, 8)
        ]
    }
    diagnostics = {
        "balanced_committee_scaling": [
            {
                "method": "hard_plurality",
                "k": k,
                "mean_balanced_answer_class_accuracy": 0.2 + 0.01 * k,
                "best_balanced_answer_class_accuracy": 0.3 + 0.01 * k,
            }
            for k in range(1, 8)
        ]
    }
    selected = {
        "selected_best_at_k_scaling": [
            {
                "k": k,
                "accuracy": 0.15 + 0.01 * k,
                "balanced_answer_class_accuracy": 0.25 + 0.01 * k,
            }
            for k in range(1, 8)
        ]
    }

    figure = committee_scaling_figure(teacher, diagnostics, selected)
    try:
        assert len(figure.axes) == 3
        assert figure.axes[0].get_ylabel() == "All-gold-token example accuracy (%)"
        assert figure.axes[2].get_ylabel() == "Free-running sequence accuracy (%)"
        assert len(figure.axes[2].lines) == 2
    finally:
        figure.clear()


def test_calibration_figure_preserves_four_robustness_panels() -> None:
    experts = ("track_a", "track_b", "general", "grounding", "procedure", "abstention", "social")
    temperatures = (0.25, 0.5, 1.0, 2.0, 4.0)
    teacher = {
        "calibration": {
            "experts": {
                expert: {
                    "top1_accuracy": 0.6,
                    "mean_confidence": 0.7,
                    "temperature": 1.0,
                    "mean_nll_by_temperature": {
                        str(temperature): 1.0 + abs(temperature - 1.0)
                        for temperature in temperatures
                    },
                }
                for expert in experts
            }
        },
        "committee_results": [
            {"mask": 127, "method": "hard_plurality", "example_accuracy": 0.25}
        ],
        "temperature_interventions": {
            "shared_temperature": {
                str(temperature): {"example_accuracy": 0.2 + temperature / 100.0}
                for temperature in temperatures
            },
            "one_expert_perturbed": {
                expert: {
                    str(temperature): {"example_accuracy": 0.2 + temperature / 100.0}
                    for temperature in temperatures
                }
                for expert in experts
            },
        },
    }

    figure = calibration_robustness_figure(teacher)
    try:
        assert len(figure.axes) == 4
        assert [axis.get_title()[0] for axis in figure.axes] == ["A", "B", "C", "D"]
        assert figure.axes[0].get_xlabel() == "Gold-token top-1 accuracy (%)"
        assert figure.axes[2].get_ylabel() == "All-gold-token example accuracy (%)"
        assert figure.axes[3].get_ylabel() == "All-gold-token example accuracy (%)"
    finally:
        figure.clear()


def test_coalition_figure_includes_expert_and_mask_views() -> None:
    existing = {
        "coalitions": {
            "coalition_summary": [{"mask": "1111111"}, {"mask": "0000001"}],
            "coalition_detail": [
                {
                    "task_category": "family_a",
                    "mask": "1111111",
                    "members": [
                        "track_a",
                        "track_b",
                        "general",
                        "grounding",
                        "procedure",
                        "abstention",
                        "social",
                    ],
                    "token_count": 7,
                },
                {
                    "task_category": "family_b",
                    "mask": "0000001",
                    "members": ["social"],
                    "token_count": 3,
                },
            ],
        }
    }

    figure = coalition_heatmap_figure(existing)
    try:
        titles = {axis.get_title() for axis in figure.axes}
        assert "A. Expert support for the winning token" in titles
        assert "B. Ten most common coalition masks" in titles
        assert any(axis.get_ylabel() == "Expert" for axis in figure.axes)
    finally:
        figure.clear()


def test_rescue_harm_figure_preserves_all_four_paired_states() -> None:
    existing = {
        "rescue_harm": [
            {
                "dimension": "global",
                "comparison": comparison,
                "n": 100,
                "both_correct": 20,
                "rescue": 15,
                "harm": 5,
                "both_wrong": 60,
            }
            for comparison in ("hard_vs_track_a", "hard_vs_probability")
        ]
    }

    figure = rescue_harm_figure(existing)
    try:
        assert len(figure.axes) == 2
        for axis in figure.axes:
            labels = " ".join(text.get_text() for text in axis.texts)
            assert all(
                state in labels for state in ("Both correct", "Rescue", "Harm", "Both wrong")
            )
    finally:
        figure.clear()


def test_quality_compute_figure_includes_forward_and_latency_frontiers() -> None:
    keys = ("backbone", "track_a", "abstention", "probability_k7", "hard_k7")
    existing = {
        "behavior": {
            key: {
                "accuracy": 0.1 + 0.01 * index,
                "mean_forward_passes": 1.0 + index,
                "mean_latency_seconds": 0.1 + 0.1 * index,
            }
            for index, key in enumerate(keys)
        }
    }

    figure = quality_compute_figure(existing)
    try:
        assert len(figure.axes) == 2
        assert figure.axes[0].get_xlabel() == "Mean expert-token forward proxy"
        assert figure.axes[1].get_xlabel() == "Mean latency per example (s)"
    finally:
        figure.clear()


def test_teacher_free_figure_keeps_same_split_estimands_explicit() -> None:
    methods = (
        "hard_plurality",
        "probability_mean",
        "calibrated_probability_mean",
        "reciprocal_rank_top5",
    )
    free_run = {
        "teacher_forced_vs_free_running": [
            {
                "method": method,
                "teacher_example_advantage_vs_hard": 0.01 * index,
                "free_sequence_advantage_vs_hard": -0.01 * index,
            }
            for index, method in enumerate(methods)
        ]
    }

    figure = teacher_free_figure(free_run)
    try:
        axis = figure.axes[0]
        assert axis.get_ylabel() == "Difference from hard plurality (percentage points)"
        legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
        assert legend_labels == [
            "Gold-prefix all-token example gap",
            "Free-running exact gap",
        ]
        assert len(axis.patches) == 8
    finally:
        figure.clear()


def test_optional_table_source_requires_strict_completion_status(tmp_path) -> None:
    source = tmp_path / "analysis.json"
    assert load_optional_complete(source, "COMPLETE") is None

    source.write_text('{"status": "PARTIAL"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing incomplete optional table source"):
        load_optional_complete(source, "COMPLETE")

    source.write_text('{"status": "COMPLETE", "value": 7}\n', encoding="utf-8")
    assert load_optional_complete(source, "COMPLETE")["value"] == 7


def test_selected_committee_table_keeps_answer_class_deltas() -> None:
    analysis = {
        "leave_one_out_family": {
            "comparisons": [
                {
                    "removed_expert": "grounding",
                    "difference_full_minus_loo": -0.01,
                }
            ],
            "descriptive_strata": [
                {
                    "removed_expert": "grounding",
                    "dimension": "answer_class",
                    "value": "answer",
                    "difference_full_minus_loo": -0.02,
                },
                {
                    "removed_expert": "grounding",
                    "dimension": "answer_class",
                    "value": "abstain",
                    "difference_full_minus_loo": 0.03,
                },
            ],
        }
    }

    rows = selected_committee_loo_table(analysis)
    assert rows[0]["answerable_delta"] == pytest.approx(-0.02)
    assert rows[0]["abstention_delta"] == pytest.approx(0.03)


def test_final_synthesis_marker_scan_distinguishes_workflow_placeholders() -> None:
    documents = {
        "complete.md": "A causal mechanism remains unresolved.",
        "interim.md": "[FROZEN SAME-SPLIT DEV COMPARISON REQUIRED]",
    }
    hits = interim_marker_hits(documents)
    assert hits == [
        {
            "path": "interim.md",
            "marker": "[FROZEN SAME-SPLIT DEV COMPARISON REQUIRED]",
        }
    ]


def test_final_synthesis_binds_figure_and_table_plans() -> None:
    assert "paper/FIGURE_PLAN.md" in SYNTHESIS_DOCUMENTS
    assert "paper/TABLE_PLAN.md" in SYNTHESIS_DOCUMENTS
    assert "paper/QUALITATIVE_ANALYSIS.md" in SYNTHESIS_DOCUMENTS
    assert "paper/EXPERIMENTAL_PROTOCOL.md" in SYNTHESIS_DOCUMENTS


def test_final_synthesis_requires_every_core_narrative_surface() -> None:
    assert set(REQUIRED_FINAL_REFERENCES) == {
        "reports/PHASE2_MECHANISM_REPORT.md",
        "paper/PAPER_ANALYSIS.md",
        "paper/RESULTS_ANALYSIS.md",
        "paper/EVIDENCE_LEDGER.md",
        "paper/CLAIMS_AND_SUPPORT.md",
        "paper/REQUIREMENT_TRACEABILITY.md",
        "paper/QUALITATIVE_ANALYSIS.md",
        "paper/EXPERIMENTAL_PROTOCOL.md",
        "paper/LIMITATIONS_AND_ETHICS.md",
    }
    assert all(
        "final_synthesis_facts.json" in references
        for references in REQUIRED_FINAL_REFERENCES.values()
    )


def test_remaining_pipeline_generates_fact_packet_before_paper_artifacts() -> None:
    names = [name for name, _ in commands()]
    fact_index = names.index("generate_final_synthesis_fact_packet")
    assert names[fact_index - 1] == "analyze_cross_v212"
    assert names[fact_index + 1 : fact_index + 3] == [
        "regenerate_paper_figures",
        "regenerate_paper_tables",
    ]


def test_final_fact_packet_routes_all_long_run_dependent_questions() -> None:
    assert set(QUESTION_EVIDENCE_MAP) == {
        "6",
        "7",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
    }
    assert set(HYPOTHESIS_EVIDENCE_MAP) == {
        "H1_calibration_robust_consensus",
        "H4_ordinal_bottleneck_rank_sufficiency",
        "H6_committee_size_threshold",
        "H7_trajectory_correction",
        "H8_abstention_eos_composition",
        "H10_specialization_matters",
    }
    assert all(paths for paths in QUESTION_EVIDENCE_MAP.values())
    assert all(paths for paths in HYPOTHESIS_EVIDENCE_MAP.values())


def test_final_fact_packet_formatters_match_analysis_contracts() -> None:
    summary = {
        "method": "hard_plurality",
        "condition": "hard_plurality",
        "correct": 1,
        "n": 2,
        "accuracy": 0.5,
        "answerable_accuracy": 0.5,
        "abstention_accuracy": 0.5,
        "mean_latency_seconds": 1.0,
        "mean_forward_passes": 7.0,
        "mean_forwards_per_generated_token": 7.0,
        "adaptive_trigger_token_rate": 0.0,
    }
    selected = {
        "method_summaries": [summary],
        "paired_family": {
            "comparisons": [
                {
                    "other_method": "probability_mean",
                    "difference_other_minus_hard": 0.0,
                    "bootstrap_ci95_low": -0.1,
                    "bootstrap_ci95_high": 0.1,
                    "other_only_correct": 1,
                    "hard_only_correct": 1,
                    "holm_adjusted_mcnemar_p": 1.0,
                }
            ]
        },
        "teacher_forced_vs_free_running": [
            {
                "method": "hard_plurality",
                "teacher_forced_token_accuracy": 0.8,
                "teacher_forced_all_token_example_accuracy": 0.3,
                "free_running_exact_accuracy": 0.5,
                "free_sequence_advantage_vs_hard": 0.0,
            }
        ],
    }
    adaptive = {
        "summaries": [summary],
        "paired_accuracy": {
            "difference_adaptive_minus_hard": 0.0,
            "paired_bootstrap_ci95_low": -0.1,
            "paired_bootstrap_ci95_high": 0.1,
            "mcnemar_exact_p": 1.0,
        },
        "efficiency": {
            "forward_pass_reduction_vs_hard": 0.5,
            "latency_ratio_adaptive_over_hard": 0.7,
        },
    }
    committee = {
        "selected_best_at_k_scaling": [
            {
                "k": 1,
                "committee": "track_a",
                "accuracy": 0.5,
                "balanced_answer_class_accuracy": 0.5,
                "mean_latency_seconds": 0.2,
                "mean_forward_passes": 1.0,
                "accuracy_per_mean_forward_pass": 0.5,
            }
        ],
        "leave_one_out_family": {
            "comparisons": [
                {
                    "removed_expert": "track_a",
                    "difference_full_minus_loo": 0.0,
                    "bootstrap_ci95_low": -0.1,
                    "bootstrap_ci95_high": 0.1,
                    "full_only_correct": 1,
                    "loo_only_correct": 1,
                    "holm_adjusted_mcnemar_p": 1.0,
                }
            ]
        },
    }
    calibrated = {
        "condition_summaries": [summary],
        "paired_family": {
            "comparisons": [
                {
                    "comparison": "calibrated_vs_hard",
                    "difference_calibrated_minus_baseline": 0.0,
                    "bootstrap_ci95_low": -0.1,
                    "bootstrap_ci95_high": 0.1,
                    "calibrated_only_correct": 1,
                    "baseline_only_correct": 1,
                    "holm_adjusted_mcnemar_p": 1.0,
                }
            ]
        },
    }
    cross = {
        "condition_summaries": [summary],
        "paired_family": {
            "comparisons": [
                {
                    "baseline": "track_a",
                    "difference_hard_minus_baseline": 0.0,
                    "bootstrap_ci95_low": -0.1,
                    "bootstrap_ci95_high": 0.1,
                    "rescue": 1,
                    "harm": 1,
                    "holm_adjusted_mcnemar_p": 1.0,
                }
            ]
        },
    }

    rendered = (
        selected_section(selected),
        adaptive_section(adaptive),
        committee_section(committee),
        calibrated_section(calibrated),
        cross_section(cross),
    )
    assert all(any(line.startswith("## ") for line in section) for section in rendered)
    assert all(any("|" in line for line in section) for section in rendered)
