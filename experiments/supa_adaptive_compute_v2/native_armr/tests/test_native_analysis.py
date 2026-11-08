"""Synthetic outcomes test decisions; no test fixture is model evidence."""
import copy

import pytest

from native_armr.src.analysis import analyze_complete_bank, audit_rows, linked_robustness
from native_armr.src.operators import REGISTERED_ARMS
from native_armr.src.protocol import DEFAULT_CONFIG, load_protocol
from shared.datasets import generate_rows, diagnostic_rows
from shared.scoring import canonical_answer, score_response
from armr.metrics import output_flags


def evidence(cases, scores):
    result = {}
    for arm in REGISTERED_ARMS:
        result[arm] = []
        for i, case in enumerate(cases):
            good = scores(arm, i, case)
            response = canonical_answer(case) if good else "incorrect fixture answer"
            tokens = [11 if good else 12, 2]
            score = score_response(response, case["scorer"])
            result[arm].append({**{k: case[k] for k in ("example_id", "scenario_id", "family", "track", "split", "variant", "template_id", "template_unseen_in_engineering")},
                                "arm": arm, "response": response, "token_ids": tokens,
                                "termination": "eos", "eos_emitted": True, "truncated": False,
                                "score": score["score"], "correct": score["correct"],
                                "safety_stopped": False, "gold_nll": 1. if good else 2.,
                                "wall_seconds": 1. if arm == "A0" else 1.2,
                                "generation_dense_matmul_flops_estimate": 100 if arm == "A0" else 110,
                                "peak_rss_bytes": 1024, "trajectory_min_rms_ratio": 1.,
                                "trajectory_max_rms_ratio": 1., **output_flags(response, tokens)})
    return result


@pytest.fixture
def config():
    value = load_protocol(DEFAULT_CONFIG)
    # Decision unit tests reduce Monte Carlo cost only. Public execution uses
    # the unchanged frozen 10,000-replicate configuration.
    value["analysis"]["bootstrap_replicates"] = 200
    return value


@pytest.fixture
def bank():
    return generate_rows("native_armr", "confirmatory_dev")


def test_complete_positive_result_requires_baseline_and_matched_control(bank, config):
    rows = evidence(bank, lambda arm, i, c: i % (8 if arm == "A3" else 4) != 0)
    report = analyze_complete_bank(bank, rows, config, "confirmatory_dev")
    assert report["decision"] == "GO_NATIVE_RECURRENCE"
    assert report["hierarchy"]["A3_vs_AM"]["tested"]
    assert report["comparisons"]["A3_vs_A0"]["overall"]["paired_wins"] == 128
    assert report["comparisons"]["A3_vs_A0"]["overall"]["paired_losses"] == 0
    assert report["preservation"]["families"] == config["analysis"]["preservation_families"]
    assert report["preservation"]["n"] == 384
    assert report["preservation"]["pass"]
    assert set(report["secondary_holm"]) == {"A1_vs_A0", "A16_vs_A0"}
    assert all(row["comparison_family_size"] == 2 for row in report["secondary_holm"].values())
    assert set(report["comparisons"]["A3_vs_A0"]["families"]) == {c["family"] for c in bank}
    assert report["cost"]["A3"]["mean_wall_seconds"] > report["cost"]["A0"]["mean_wall_seconds"]
    assert report["quality_cost_frontier"]["frontier"] == ["A0", "A3"]
    assert report["quality_cost_frontier"]["arms"]["A3"]["added_parameters"] == 0
    assert report["quality_cost_frontier"]["arms"]["AM"]["added_parameters"] == 31457280


def test_compute_control_tie_cannot_support_recurrence_specific_go(bank, config):
    rows = evidence(bank, lambda arm, i, c: i % (8 if arm in {"A3", "AM"} else 4) != 0)
    report = analyze_complete_bank(bank, rows, config, "confirmatory_dev")
    assert report["decision"] == "EXTRA_COMPUTE_SIGNAL_RECURRENCE_UNRESOLVED"
    assert not report["efficacy_claim_licensed"]
    assert not report["hierarchy"]["A3_vs_AM"]["pass"]


def test_hierarchy_closes_and_secondary_winner_does_not_replace_a3(bank, config):
    rows = evidence(bank, lambda arm, i, c: True if arm == "A16" else i % 4 != 0)
    report = analyze_complete_bank(bank, rows, config, "confirmatory_dev")
    assert report["decision"] == "NO_GO_OR_INCONCLUSIVE"
    assert report["secondary_holm"]["A16_vs_A0"]["holm_p"] < .05
    assert not report["hierarchy"]["A3_vs_AM"]["tested"]
    assert report["hierarchy"]["A3_vs_AM"]["p_value_status"] == "descriptive_only_hierarchy_closed"


def test_pooled_preservation_regression_blocks_otherwise_positive_result(bank, config):
    preservation = set(config["analysis"]["preservation_families"])
    def scores(arm, i, case):
        if arm != "A3":
            return i % 2 != 0
        if case["family"] in preservation:
            return i % 2 != 0 and i % 16 != 1
        return True
    report = analyze_complete_bank(bank, evidence(bank, scores), config, "confirmatory_dev")
    assert report["hierarchy"]["A3_vs_A0"]["pass"]
    assert report["hierarchy"]["A3_vs_AM"]["pass"]
    assert not report["preservation"]["pass"]
    assert report["decision"] == "NO_GO_OR_INCONCLUSIVE"


def test_extra_invalid_output_blocks_positive_accuracy(bank, config):
    rows = evidence(bank, lambda arm, i, c: i % (8 if arm == "A3" else 4) != 0)
    row = rows["A3"][0]  # Already incorrect, so replacing its text leaves scores fixed.
    row.update(response="", **output_flags("", row["token_ids"]))
    report = analyze_complete_bank(bank, rows, config, "confirmatory_dev")
    assert report["hierarchy"]["A3_vs_AM"]["pass"]
    assert not report["primary_gates"]["validity"]
    assert report["decision"] == "NO_GO_OR_INCONCLUSIVE"


def test_linked_robustness_exposes_lost_advantage_and_checks_source(bank, config):
    diagnostics = diagnostic_rows(bank)
    originals = evidence(bank, lambda arm, i, c: arm == "A3")
    variants = evidence(diagnostics, lambda arm, i, c: False)
    report = linked_robustness(bank, originals, diagnostics, variants, config)
    assert "longer_chain" in report["variants"]
    assert "changed_relation_direction" in report["variants"]
    for value in report["variants"].values():
        effect = value["arms"]["A3"]
        assert effect["original_advantage_vs_A0"] == 1.
        assert effect["variant_advantage_vs_A0"] == 0.
        assert effect["change_in_advantage_scenario_bootstrap_95_ci"] == [-1., -1.]
    diagnostics[0]["source_prompt_sha256"] = "falsified source"
    with pytest.raises(ValueError, match="frozen original"):
        linked_robustness(bank, originals, diagnostics, variants, config)


@pytest.mark.parametrize("role", ["architecture_dev", "confirmatory_diagnostics"])
def test_small_or_diagnostic_roles_never_license_efficacy(config, role):
    cases = generate_rows("native_armr", "architecture_dev") if role == "architecture_dev" else diagnostic_rows(generate_rows("native_armr", "confirmatory_dev"))
    rows = evidence(cases, lambda arm, i, c: arm == "A3")
    report = analyze_complete_bank(cases, rows, config, role)
    assert not report["efficacy_claim_licensed"]
    assert report["decision"] in {"MECHANICS_PASS_NO_EFFICACY_CLAIM", "DIAGNOSTIC_ONLY_NO_EFFICACY_CLAIM"}


@pytest.mark.parametrize("mutation", ["partial_bank", "missing_a0", "missing_am", "partial_arm", "score", "validity", "role", "metadata", "safety", "token_cap", "cost", "rms"])
def test_independent_audit_rejects_incomplete_or_falsified_evidence(bank, config, mutation):
    rows = evidence(bank, lambda arm, i, c: True)
    cases = copy.deepcopy(bank)
    if mutation == "partial_bank":
        cases.pop()
    elif mutation == "missing_a0":
        del rows["A0"]
    elif mutation == "missing_am":
        del rows["AM"]
    elif mutation == "partial_arm":
        rows["A3"].pop()
    elif mutation == "score":
        rows["A3"][0]["score"] = 0.
    elif mutation == "validity":
        rows["A3"][0]["invalid"] = True
    elif mutation == "role":
        cases[0]["split"] = "calibration"
    elif mutation == "metadata":
        rows["A3"][0]["family"] = "fictional_family"
    elif mutation == "safety":
        rows["A3"][0]["safety_stopped"] = True
    elif mutation == "token_cap":
        rows["A3"][0]["token_ids"] = list(range(48))
    elif mutation == "cost":
        rows["A3"][0]["wall_seconds"] = float("nan")
    else:
        rows["A3"][0]["trajectory_max_rms_ratio"] = 4.1
    with pytest.raises(ValueError):
        audit_rows(cases, rows, config, "confirmatory_dev")
