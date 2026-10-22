from copy import deepcopy
import json

import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.analysis import (
    GRID, SEEDS, expected_cells, grid_headroom, known_sum, read_journal, summarize, validate_lineage, validate_traces,
)


def synthetic_matrix(n_dev=2):
    metadata = [{"case_id": f"case-{i}", "split": "DEV", "family": f"family-{i}",
                 "independent_cluster": "shared-parent", "audit": {"constant_answer_scores": {"UNKNOWN": 0}, "prompt_copy_score": 0}}
                for i in range(n_dev)]
    metadata.append({"case_id": "smoke", "split": "SMOKE", "family": "smoke",
                     "independent_cluster": "smoke-parent", "audit": {"constant_answer_scores": {"UNKNOWN": 0}, "prompt_copy_score": 0}})
    rows = []
    # Best global .9 wins case0; .8 wins case1. Oracle1, fixed.5, A3zero.
    for split, seed, case_id, arm, extra in sorted(expected_cells(metadata)):
        actual_arm = "A2" if arm.startswith("A2:") else arm
        alpha = float(arm.split(":")[1]) if actual_arm == "A2" else None
        score = int(case_id == "smoke" or (actual_arm == "A2" and
                    ((alpha == .8 and case_id == "case-1") or (alpha >= .9 and case_id == "case-0"))))
        rows.append({"split": split, "seed": seed, "case_id": case_id, "arm": actual_arm,
                     "extra_loops": extra, "fixed_alpha": alpha, "physical_run": True,
                     "score": score, "token_ids": [score, 99], "response": "UNKNOWN" if not score else "YES",
                     "format_score": 1, "invalid": False, "termination": "eos", "safety_stopped": False,
                     "gold_nll": None, "gold_answer": None,
                     "wall_seconds": 2., "controller_seconds": .1, "controller_logging_seconds": .01,
                     "full_model_calls_including_gold_nll": 2, "processed_token_positions": 7,
                     "total_token_positions_including_gold_nll": 7, "generation_dense_matmul_flops_estimate": 100})
    for seed in SEEDS:
        for case_id in (f"case-{i}" for i in range(n_dev)):
            fixed = [row for row in rows if row["split"] == "DEV" and row["seed"] == seed
                     and row["case_id"] == case_id and row["arm"] == "A2"]
            selected = max(fixed, key=lambda row: (row["score"], -row["fixed_alpha"]))
            rows.append({**selected, "arm": "A4_example", "physical_run": False,
                         "wall_seconds": 9999., "full_model_calls_including_gold_nll": 9999})
    native = {"status": "DEV_COMPLETE_NOT_TESTED", "source_unchanged": True,
              "checkpoint_files_unchanged": True, "trainable_parameters": 0, "training": False}
    worker = {"status": "WORKER_TERMINAL", "exit_code": 0, "native_worker_seconds": 125.,
              "supervisor_wall_seconds": 126., "peak_rss_bytes": 1000}
    return rows, metadata, native, worker


def report(rows=None, metadata=None, native=None, worker=None, **kwargs):
    default_rows, default_metadata, default_native, default_worker = synthetic_matrix()
    used_rows = rows if rows is not None else default_rows
    return summarize(used_rows, synthetic_traces(used_rows),
                     metadata if metadata is not None else default_metadata, {"alpha": .8},
                     native if native is not None else default_native,
                     worker if worker is not None else default_worker, **kwargs)


def synthetic_traces(rows):
    return [{"case_id": row["case_id"], "split": row["split"], "seed": row["seed"],
             "arm": row["arm"], "extra_loops": row["extra_loops"], "forward_index": index,
             "iteration": iteration, "alpha": row["fixed_alpha"] if row["arm"] == "A2" and iteration else 1.}
            for row in rows if row["physical_run"] and row["arm"] != "qualification_legacy"
            for index in (1, 2) for iteration in range(1 + row["extra_loops"])]


def test_complete_negative_capture_keeps_grid_oracle_and_reuse_cost_separate(tmp_path):
    rows, metadata, native, worker = synthetic_matrix()
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    parsed, errors = read_journal(path)
    result = report(parsed, metadata, native, worker, read_errors=errors)
    assert result["complete"] and result["physical_rows"] == 49
    assert result["reused_oracle_rows"] == 4
    assert result["constant_oracle"]["headroom"] == .5 and result["constant_oracle"]["capture"] == -1
    assert result["costs"]["physical_row_sums"]["wall_seconds"]["complete_value"] == 98
    assert result["costs"]["physical_row_sums"]["full_model_calls_including_gold_nll"]["complete_value"] == 98
    assert result["costs"]["reused_oracle_additional_generation_cost"] == 0
    assert result["frozen_fixed_comparator"]["alpha"] == .8
    assert not result["efficacy_claim_supported"]


def test_missing_arm_suppresses_all_primary_comparisons_and_capture():
    rows, metadata, native, worker = synthetic_matrix()
    rows = [row for row in rows if not (row["arm"] == "A5" and row["case_id"] == "case-0" and row["seed"] == 29)]
    result = report(rows, metadata, native, worker)
    assert not result["complete"] and result["pairs"] == {}
    assert result["constant_oracle"]["headroom"] is None and result["constant_oracle"]["capture"] is None
    assert all(row["accuracy"] is None for row in result["tables"])
    assert len(result["missing_cells"]) == 1


def test_deterministic_repeats_average_within_case_and_share_parent_cluster():
    result = report()
    assert result["dev_items"] == 2 and result["provisional_cluster_count"] == 1
    assert result["cluster_sizes"] == {"shared-parent": 2}
    assert not result["seeds_are_independent_examples"]
    pair = result["pairs"]["frozen_fixed"]
    assert len(pair["per_case"]) == 2 and pair["score_delta"] == -.5
    assert pair["provisional_cluster_deltas"] == {"shared-parent": -.5}
    assert pair["losses"] == ["case-1"] and pair["confirmatory_p_value"] is None


def test_capture_above_one_is_not_clipped_and_zero_or_incomplete_is_null():
    assert grid_headroom(1., .25, .5, True)["capture"] == 3
    assert grid_headroom(.1, .25, .5, True)["capture"] == pytest.approx(-.6)
    assert grid_headroom(.9, .5, .5, True)["capture"] is None
    assert grid_headroom(None, None, None, False)["capture"] is None


def test_complete_raw_matrix_can_exceed_its_finite_constant_grid_oracle():
    rows, metadata, native, worker = synthetic_matrix(n_dev=4)
    for row in rows:
        if row["arm"] == "A3" and row["split"] == "DEV":
            row.update(score=1, response="YES", token_ids=[1, 99])
    result = report(rows, metadata, native, worker)
    assert result["complete"] and result["constant_oracle"]["headroom"] == .25
    assert result["constant_oracle"]["capture"] == 3


def test_mutation_receipt_and_unsafe_oracle_duplicates_refuse_complete():
    rows, metadata, native, worker = synthetic_matrix()
    native["source_unchanged"] = False
    assert not report(rows, metadata, native, worker)["complete"]
    native["source_unchanged"] = True
    rows.append(deepcopy(rows[-1]))
    result = report(rows, metadata, native, worker)
    assert not result["complete"] and not result["constant_oracle"]["reported_rows_verified"]


def test_truncated_jsonl_retains_explicit_failure_and_missing_costs_stay_null(tmp_path):
    path = tmp_path / "outcomes.jsonl"
    rows, metadata, native, worker = synthetic_matrix()
    rows[0]["controller_seconds"] = None
    path.write_text("\n".join(json.dumps(row) for row in rows) + '\n{"unfinished":')
    parsed, errors = read_journal(path)
    result = report(parsed, metadata, native, worker, read_errors=errors)
    assert len(errors) == 1 and not result["complete"]
    measured = result["costs"]["physical_row_sums"]["controller_seconds"]
    assert measured["complete_value"] is None and measured["missing_rows"] == 1
    assert result["costs"]["cumulative_native_worker_seconds"] is None


def test_budget_resources_and_abstention_are_not_repeated_examples():
    events = [{"event": "native_completed", "run": "first", "native_worker_seconds": 125},
              {"event": "native_completed", "run": "failed", "native_worker_seconds": 20}]
    result = report(resources=[{"rss_bytes": 1100}, {"rss_bytes": 1200}], budget_events=events)
    assert result["costs"]["cumulative_native_worker_seconds"] == 145
    assert result["costs"]["remaining_native_seconds"] == 21455
    assert result["costs"]["peak_sampled_worker_tree_rss_bytes"] == 1200
    adaptive = next(row for row in result["tables"] if row["policy"] == "A3")
    assert adaptive["literal_UNKNOWN_rate"] == 1 and adaptive["semantic_abstention"] == "NOT_ASSESSED"
    assert result["own_bank_floors"]["DEV"]["items"] == 2


def test_empty_trajectory_and_deterministic_repeat_divergence_refuse_complete():
    rows, metadata, native, worker = synthetic_matrix()
    assert validate_traces([row for row in rows if row["physical_run"]], [])
    row = next(row for row in rows if row["arm"] == "A3" and row["seed"] == 29 and row["case_id"] == "case-0")
    row["token_ids"] = [7, 99]
    result = report(rows, metadata, native, worker)
    assert not result["complete"] and result["deterministic_repeat_divergence"]


def test_missing_or_mismatched_review_lineage_and_pool_fail_closed():
    assert validate_lineage({}, "admission-digest", {}, {}, {}, "binding-digest", "protocol-digest",
                            {}, {}, {}, [], {}, {})
    # Pool identity is checked from extra-update DEV rows, with bounds clipping.
    binding = {"runtime": {}, "datasets": {"DEV": {"sha256": "dev"}, "SMOKE": {"sha256": "smoke"}}}
    selection = {"config": {}, "dataset_sha256": "dev", "native_TEST_authorized": False, "pool": [1.]}
    errors = validate_lineage({}, "x", {}, {}, binding, "b", "p", {}, {}, selection,
                              [{"arm": "A3", "split": "DEV", "iteration": 1, "alpha": 1.200000047}], {}, {})
    assert "frozen random pool differs from generation-only DEV A3 journal" in errors


def test_absent_measurements_are_null_while_measured_zero_and_worker_cost_remain_real():
    summary = known_sum([], "wall_seconds")
    assert summary["complete_value"] is None and summary["known_subtotal"] is None
    assert summary["measured_rows"] == summary["rows"] == 0
    assert known_sum([{"wall_seconds": 0.}], "wall_seconds")["complete_value"] == 0
    result = report(rows=[], worker={"native_worker_seconds": 18.734})
    assert result["costs"]["physical_row_sums"]["wall_seconds"]["complete_value"] is None
    assert result["costs"]["worker_wall_seconds"] == 18.734
