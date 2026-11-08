from copy import deepcopy
import json

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.gpu_analysis import (
    aggregate, forward_audit, gpu_expected_cells, validate_gpu_traces,
)
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_analysis import synthetic_matrix


def gpu_fixture(n_dev=2):
    rows, metadata, _, worker = synthetic_matrix(n_dev)
    for item in metadata:
        item["input_tokens"] = 3
    existing = {tuple((r["split"], r["seed"], r["case_id"],
                       f"A2:{r['fixed_alpha']:g}" if r["arm"] == "A2" else r["arm"], r["extra_loops"])) for r in rows if r["physical_run"]}
    smoke = next(row for row in rows if row["case_id"] == "smoke")
    for split, seed, case_id, arm, extra in sorted(gpu_expected_cells(metadata) - existing):
        rows.append({**smoke, "split": split, "seed": seed, "case_id": case_id, "arm": arm,
                     "extra_loops": extra, "fixed_alpha": None, "physical_run": True})
    for row in rows:
        row["configured_seed"] = row["seed"]
        row["initial_block_applications"] = 2
        row["extra_layer_applications"] = 4 * row["extra_loops"]
    physical = [row for row in rows if row["physical_run"]]
    traces = [{"case_id": row["case_id"], "split": row["split"], "seed": row["seed"], "configured_seed": row["seed"],
               "arm": row["arm"], "extra_loops": row["extra_loops"], "forward_index": index,
               "iteration": iteration, "alpha": row["fixed_alpha"] if row["arm"] == "A2" and iteration else 1.,
               **({"random_draw_index": index if iteration else index - 1} if row["arm"] == "A5" else {})}
              for row in physical if row["arm"] != "qualification_legacy"
              for index in (1, 2) for iteration in range(1 + row["extra_loops"])]
    resources = {"gpu_peak_allocated_bytes": 3 * 1024**3, "gpu_peak_reserved_bytes": 4 * 1024**3,
                 "gpu_free_bytes": 2 * 1024**3, "rss_bytes": 1000}
    forwards = [{"event": event, "completed": event == "FORWARD_COMPLETED", "case_id": row["case_id"],
                 "split": row["split"], "seed": row["seed"], "arm": row["arm"], "extra_loops": row["extra_loops"],
                 "full_model_call": index, "forward_seconds": .1 if event == "FORWARD_COMPLETED" else None,
                 "prefix_tokens": 3 + index - 1,
                 "resources": resources if event == "FORWARD_COMPLETED" else None}
                for row in physical for index in (1, 2) for event in ("FORWARD_STARTED", "FORWARD_COMPLETED")]
    for index in range(1, 18):
        forwards.append({"event": "MAXIMUM_FORWARD_STARTED", "attempt": index, "completed": False, "prefix_tokens": 2248})
        forwards.append({"event": "MAXIMUM_FORWARD_FAILED" if index == 7 else "MAXIMUM_FORWARD_COMPLETED",
                         "attempt": index, "completed": index != 7, "forward_seconds": .2, "resources": resources,
                         "prefix_tokens": 2248,
                         "partial_compute": "UNKNOWN" if index == 7 else None,
                         "error": "EXPECTED_FORCED_RESTORATION_EXCEPTION" if index == 7 else None})
    for use_cache, present in ((True, False), (False, True)):
        forwards.append({"event": "CACHE_REJECTION_STARTED", "use_cache": use_cache, "cache_object_present": present})
        forwards.append({"event": "CACHE_REJECTED", "model_compute_started": False, "wrapper_forward_count": 0,
                         "forward_seconds": .001})
    native = {"status": "GPU_DEV_COMPLETE_NOT_TESTED", "source_unchanged": True,
              "checkpoint_files_unchanged": True, "trainable_parameters": 0, "training": False,
              "final_resources": resources}
    qualification = {"status": "QUALIFIED_GPU_FP32_REFERENCE_OPERATOR", "original_new_A3_identity": "PASS",
                     "L0_L2_one_shot": "PASS_EVERY_COMPLETED_FORWARD", "scaled_arms_smoke": "PASS",
                     "checkpoint_files_unchanged": True, "before_state_sha256": "state", "after_state_sha256": "state",
                     "mask_positions": {"positions": "CUDA arange checked", "replay_conditioning": "same objects checked"},
                     "maximum_shape": [{"maximum_prefix_tokens": 2248, "identity": "PASS",
                                         "forced_exception_restoration": "PASS", "cache_rejection": "PASS"}]}
    a5 = {"status": "PASS_FROZEN_POOL_SMOKE_AND_MAXIMUM_SHAPE", "actual_seed_binding": True}
    selection = {"alpha": .8, "pool": [1.]}
    return rows, traces, metadata, selection, native, worker, qualification, a5, forwards


def test_gpu_complete_counts_cluster_averaging_negative_capture_and_oracle_cost():
    result = aggregate(*gpu_fixture())
    assert result["complete"] and result["physical_rows"] == 51 and result["reused_oracle_rows"] == 4
    assert result["constant_oracle"]["capture"] == -1
    assert result["provisional_clusters"] == {"shared-parent": 2}
    assert result["costs"]["physical_row_measurements"]["wall_seconds"]["complete_value"] == 102
    assert result["costs"]["oracle_additional_generation_cost"] == 0
    attempts = result["costs"]["forward_attempt_journal"]
    assert attempts["maximum_attempts"] == 17 and attempts["maximum_completed"] == 16
    assert attempts["partial_failure_compute_positions"] is None


def test_gpu_a5_seed_output_variance_is_allowed_but_a3_divergence_refuses():
    fixture = list(gpu_fixture())
    row = next(row for row in fixture[0] if row["arm"] == "A5" and row["seed"] == 29 and row["case_id"] == "case-0")
    row.update(score=1, token_ids=[1, 99], response="YES")
    result = aggregate(*fixture)
    assert result["complete"] and not result["deterministic_repeat_divergence"]
    row = next(row for row in fixture[0] if row["arm"] == "A3" and row["seed"] == 29 and row["case_id"] == "case-0")
    row["token_ids"] = [7, 99]
    result = aggregate(*fixture)
    assert not result["complete"] and result["pairs"] == {}


def test_missing_gpu_arm_and_frozen_a5_gate_suppress_primary_comparisons():
    fixture = list(gpu_fixture())
    fixture[0] = [row for row in fixture[0] if not (row["arm"] == "A5" and row["seed"] == 29 and row["case_id"] == "case-0")]
    result = aggregate(*fixture)
    assert not result["complete"] and result["constant_oracle"]["headroom"] is None
    assert all(row["accuracy"] is None for row in result["tables"])
    fixture = list(gpu_fixture()); fixture[7] = {}
    assert not aggregate(*fixture)["complete"]


def test_maximum_failed_attempt_cache_variants_and_rng_draws_are_required():
    fixture = list(gpu_fixture())
    fixture[8] = [event for event in fixture[8] if event["event"] != "MAXIMUM_FORWARD_FAILED"]
    assert not aggregate(*fixture)["complete"]
    fixture = list(gpu_fixture())
    event = next(event for event in fixture[1] if event["arm"] == "A5" and event["iteration"] == 1)
    event["random_draw_index"] = 999
    assert not aggregate(*fixture)["complete"]


def test_gpu_capture_above_one_remains_unclipped_and_missing_costs_null():
    fixture = list(gpu_fixture(n_dev=4))
    for row in fixture[0]:
        if row["arm"] == "A3" and row["split"] == "DEV":
            row.update(score=1, token_ids=[1, 99], response="YES")
    result = aggregate(*fixture)
    assert result["complete"] and result["constant_oracle"]["capture"] == 3
    empty = aggregate([], [], fixture[2], {}, {}, {"native_worker_seconds": 18.734}, {}, {}, [])
    assert empty["costs"]["physical_row_measurements"]["wall_seconds"]["complete_value"] is None
    assert empty["costs"]["worker_wall_seconds"] == 18.734


def test_recorded_gpu_ceiling_failure_suppresses_precomputed_tables_and_pairs():
    fixture = list(gpu_fixture())
    fixture[4]["final_resources"] = {"gpu_peak_allocated_bytes": 6 * 1024**3}
    result = aggregate(*fixture)
    assert not result["complete"] and result["pairs"] == {}
    assert result["constant_oracle"]["capture"] is None and all(row["accuracy"] is None for row in result["tables"])


def test_partial_loader_snapshot_and_prefix_counter_mismatch_are_visible():
    fixture = list(gpu_fixture())
    fixture[4]["after_load_resources"] = {"gpu_peak_allocated_bytes": 4 * 1024**3, "gpu_peak_reserved_bytes": 5 * 1024**3}
    fixture[4]["final_resources_on_stop"] = {"gpu_peak_allocated_bytes": 4 * 1024**3, "gpu_peak_reserved_bytes": 5 * 1024**3}
    result = aggregate(*fixture)
    assert result["costs"]["peak_gpu_allocated_bytes"] == 4 * 1024**3
    assert result["costs"]["peak_gpu_reserved_bytes"] == 5 * 1024**3
    fixture[8][0]["prefix_tokens"] = 999
    result = aggregate(*fixture)
    assert not result["complete"] and any("prefix length" in message for message in result["errors"])
