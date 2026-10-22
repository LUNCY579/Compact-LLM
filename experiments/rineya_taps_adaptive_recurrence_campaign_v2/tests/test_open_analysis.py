from copy import deepcopy
import struct

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import open_bank
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.open_analysis import (
    GRID, SEEDS, aggregate, expected_open_cells, payload_digest, raw_diagnostics,
)


def fixture():
    bank = open_bank.generate("analysis-fixture", 17)
    smoke = [{"case_id": f"smoke-{i}", "input_tokens": 3} for i in range(8)]
    token_rows = [{"case_id": row["id"], "input_tokens": 3} for row in bank]
    by_id = {row["id"]: row for row in bank}
    rows = []
    # Match actual dispatch order: complete SMOKE, DEV, then validation.
    rank = {"SMOKE": 0, "DEVELOPMENT": 1, "OPEN_VALIDATION": 2}
    for split, seed, case_id, arm, extra in sorted(expected_open_cells(bank, smoke), key=lambda cell: (rank[cell[0]], cell[1:])):
        actual_arm = "A2" if arm.startswith("A2:") else arm
        alpha = float(arm.split(":")[1]) if actual_arm == "A2" else None
        if alpha not in GRID and actual_arm == "A2":
            alpha = next(value for value in GRID if f"{value:g}" == arm.split(":")[1])
        raw = by_id.get(case_id)
        response = raw["gold"] if raw else "YES"
        scored = open_bank.score_response(response, raw) if raw else {"correct": True, "format_valid": True}
        row = {"split": split, "seed": seed, "configured_seed": seed, "case_id": case_id,
               "arm": actual_arm, "fixed_alpha": alpha, "extra_loops": extra, "physical_run": True,
               "response": response, "token_ids": [1], "score": int(scored["correct"]), "format_score": int(scored["format_valid"]),
               "safety_stopped": False, "gold_nll": None, "gold_answer": None,
               "wall_seconds": 2., "full_model_calls_including_gold_nll": 1, "processed_token_positions": 3,
               "initial_block_applications": 1, "extra_layer_applications": extra * 2}
        if raw:
            row.update(parent_id=raw["parent_id"], generator_family=raw["generator_family"], **raw_diagnostics(response, raw))
        rows.append(row)
    physical = list(rows)
    traces = [{"case_id": row["case_id"], "split": row["split"], "seed": row["seed"], "configured_seed": row["seed"],
               "arm": row["arm"], "extra_loops": row["extra_loops"], "forward_index": 1, "iteration": iteration,
               "alpha": struct.unpack("f", struct.pack("f", row["fixed_alpha"]))[0] if row["arm"] == "A2" and iteration else 1.,
               **({"random_draw_index": iteration} if row["arm"] == "A5" else {})}
              for row in physical if row["arm"] != "qualification_legacy" for iteration in range(row["extra_loops"]+1)]
    freeze = {"status": "FROZEN_OPEN_DEVELOPMENT", "alpha": .8, "pool": [1.] * 32,
              "selection_case_ids": [row["id"] for row in bank if row["split"] == "DEVELOPMENT"],
              "OPEN_VALIDATION_native_exposure_at_freeze": False}
    freeze["freeze_sha256"] = payload_digest(freeze)
    for raw in bank:
        for seed in SEEDS:
            selected = next(row for row in physical if row["case_id"] == raw["id"] and row["seed"] == seed and row["arm"] == "A2" and row["fixed_alpha"] == .8)
            rows.append({**selected, "arm": "A4_example", "physical_run": False, "wall_seconds": 9999.})
            if raw["split"] == "OPEN_VALIDATION": rows.append({**selected, "arm": "A2_frozen", "physical_run": False, "wall_seconds": 9999.})
    forwards = [{"event": event, "case_id": row["case_id"], "split": row["split"], "seed": row["seed"],
                 "arm": row["arm"], "extra_loops": row["extra_loops"], "full_model_call": 1,
                 "prefix_tokens": 3, "forward_seconds": .1 if event == "FORWARD_COMPLETED" else None}
                for row in physical for event in ("FORWARD_STARTED", "FORWARD_COMPLETED")]
    for index in range(18):
        forwards.append({"event": "MAXIMUM_FORWARD_STARTED", "prefix_tokens": 357, "attempt": index+1})
        forwards.append({"event": "MAXIMUM_FORWARD_FAILED" if index == 6 else "MAXIMUM_FORWARD_COMPLETED",
                         "prefix_tokens": 357, "attempt": index+1, "error": "EXPECTED_OPEN_FORCED_EXCEPTION" if index == 6 else None})
    for use_cache, present in ((True, False), (False, True)):
        forwards.append({"event": "CACHE_REJECTION_STARTED", "use_cache": use_cache, "cache_object_present": present})
        forwards.append({"event": "CACHE_REJECTED", "model_compute_started": False, "wrapper_forward_count": 0})
    native = {"status": "OPEN_MATRIX_COMPLETE_NOT_CONFIRMATORY", "source_unchanged": True,
              "checkpoint_files_unchanged": True, "trainable_parameters": 0, "training": False}
    worker = {"status": "WORKER_TERMINAL", "exit_code": 0, "native_worker_seconds": 50., "peak_rss_bytes": 1000}
    qualification = {"status": "PASS_OPEN_NEW_CAP_SAME_OPERATOR", "maximum_prefix_tokens": 357,
                     "noextra_unit_originalA3_identity": True, "forced_exception_restoration": True,
                     "cache_rejections": 2, "source_unchanged": True,
                     "before_state_sha256": "state", "after_state_sha256": "state"}
    a5 = {"status": "PASS_FROZEN_OPEN_DEV_POOL", "actual_seed_binding": True}
    return [rows, traces, bank, smoke, token_rows, freeze, native, worker, qualification, a5, forwards]


def test_complete_open608_six_scales_freeze_and_nonphysical_costs():
    report = aggregate(*fixture())
    assert report["complete"] and report["physical_rows"] == report["expected_physical_rows"] == 608
    assert report["oracle_reused_rows"] == 48 and report["frozen_alias_rows"] == 16
    assert report["costs"]["physical_row_measurements"]["wall_seconds"]["complete_value"] == 1216
    assert report["costs"]["nonphysical_alias_oracle_extra_generation"] == 0
    assert report["splits"]["OPEN_VALIDATION"]["DEV_frozen_comparator"]["alpha"] == .8
    assert report["splits"]["OPEN_VALIDATION"]["constant_oracle"]["capture"] is None


def test_missing_or_duplicate_open_cell_suppresses_all_comparisons():
    args = fixture(); args[0].pop(0)
    report = aggregate(*args)
    assert not report["complete"]
    assert all(not split["pairs"] and all(row["accuracy"] is None for row in split["tables"]) for split in report["splits"].values())
    args = fixture(); args[0].append(deepcopy(args[0][0]))
    assert not aggregate(*args)["complete"]


def test_validation_freeze_mutation_and_pool_leak_fail_closed():
    args = fixture(); args[5]["alpha"] = .9
    assert not aggregate(*args)["complete"]
    args = fixture(); args[5]["OPEN_VALIDATION_native_exposure_at_freeze"] = True
    assert not aggregate(*args)["complete"]


def test_own_bank_floors_and_equal_categories_are_recomputed():
    report = aggregate(*fixture())
    split = report["splits"]["DEVELOPMENT"]
    assert len(split["generator_clusters"]) == 4 and sum(split["generator_clusters"].values()) == 16
    assert set(split["category_weights"].values()) == {.25}
    assert split["own_bank_blind_floor_counts"]["copy_full_prompt"] == 0
    assert not report["independent_confirmation"] and not report["broad_intelligence_claim"]


def test_strict_regrade_and_deterministic_divergence_are_not_hidden_by_aggregate():
    args = fixture()
    row = next(row for row in args[0] if row["split"] == "DEVELOPMENT" and row["arm"] == "A3" and row["seed"] == 29)
    row["response"] += " explanation"
    row["token_ids"] = [2]
    report = aggregate(*args)
    assert not report["complete"] and report["deterministic_divergence"]


def test_a5_stochastic_repeats_average_within_parent_and_do_not_create_new_units():
    args = fixture()
    row = next(row for row in args[0] if row["split"] == "DEVELOPMENT" and row["arm"] == "A5" and row["seed"] == 29)
    raw = next(raw for raw in args[2] if raw["id"] == row["case_id"])
    row.update(response="", score=0, format_score=0, token_ids=[2], **raw_diagnostics("", raw))
    report = aggregate(*args)
    assert report["complete"] and not report["deterministic_divergence"]
    stochastic = next(row for row in report["splits"]["DEVELOPMENT"]["tables"] if row["policy"] == "A5")
    assert stochastic["accuracy"] == 1 - .5/16
    assert report["splits"]["DEVELOPMENT"]["parents"] == 16


def test_rounded_static_alpha_unit_trace_smoke_score_caps_and_initial_counts_refuse():
    args = fixture()
    row = next(row for row in args[0] if row["physical_run"] and row["arm"] == "A2" and row["fixed_alpha"] == 1.1473425965417514)
    row["fixed_alpha"] = 1.14734
    assert not aggregate(*args)["complete"]
    args = fixture(); event = next(event for event in args[1] if event["arm"] == "A1" and event["iteration"] == 1)
    event["alpha"] = .9
    assert not aggregate(*args)["complete"]
    args = fixture(); row = next(row for row in args[0] if row["physical_run"] and row["split"] == "SMOKE")
    row["score"] = .5
    assert not aggregate(*args)["complete"]
    args = fixture(); row = next(row for row in args[0] if row["physical_run"] and row["split"] == "SMOKE")
    row["format_score"] = .5
    assert not aggregate(*args)["complete"]
    args = fixture(); row = next(row for row in args[0] if row["physical_run"] and row["split"] == "DEVELOPMENT")
    row["token_ids"] = [1]*9
    assert not aggregate(*args)["complete"]
    args = fixture(); args[0][0]["initial_block_applications"] = 0
    assert not aggregate(*args)["complete"]


def test_native_and_parent_resource_ceiling_reserve_failures_suppress_primary():
    args = fixture(); args[6]["final_resources_on_stop"] = {"rss_bytes": 7*1024**3}
    assert not aggregate(*args)["complete"]
    args = fixture()
    result = aggregate(*args, resources=[{"host_available_bytes": 512*1024**2}])
    assert not result["complete"] and not result["splits"]["DEVELOPMENT"]["pairs"]
