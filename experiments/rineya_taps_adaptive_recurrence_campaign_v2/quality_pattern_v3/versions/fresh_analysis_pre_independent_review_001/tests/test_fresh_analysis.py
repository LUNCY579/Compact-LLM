import ast
import copy
import inspect
import json
import random

import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import fresh_analysis as a


def labels_and_cells():
    science, panel, cells = [], [], {}
    for fam in range(6):
        for mech in range(2):
            for parent in range(18):
                for variant in (0, 1):
                    cid = f"fixture{fam}:{mech}:{parent}:{variant}"
                    science.append({"case_id": cid, "parent_id": f"p{fam}:{mech}:{parent}", "family": str(fam),
                                    "mechanism": str(mech), "variant_id": variant, "gold": "1", "prompt": "fixture"})
                    for path in a.PATHS:
                        response = "1" if path in (a.GUARD, "PRIMARY_REPEAT", a.CONFIDENT) or variant == 0 else "0"
                        cells[cid, path] = {"response": response, "token_ids": [1, 151645], "termination": "eos",
                                           "eos_emitted": True, "truncated": False, "recognized": False}
    for i in range(24):
        cid = f"panel{i}"; panel.append({"case_id": cid, "gold": "COPY", "prompt": "copy"})
        for path in a.PATHS: cells[cid, path] = {"response": "COPY", "token_ids": [1, 151645]}
    return science, panel, cells


def score(response, label):
    return {"correct": response == label["gold"], "format_valid": response in ("1", "0", "UNKNOWN"),
            "canonical_output": response, "false_UNKNOWN": response == "UNKNOWN" and label["gold"] != "UNKNOWN",
            "unsupported_canonical_answer": response in ("0", "1") and label["gold"] == "UNKNOWN"}


def panel_score(response, gold): return {"correct": response.strip() == gold}


def test_parent_not_item_weighting_and_constant_conditional_bootstrap():
    labels = [{"case_id": str(i), "family": "f", "mechanism": "m", "parent_id": "many" if i < 3 else "one"} for i in range(4)]
    means = a.parent_means(labels, {str(i): float(i < 3) for i in range(4)})
    assert a.stratified_mean(means) == .5  # case-weighted value would.75
    parents = {(str(f), str(m), str(p)): .125 for f in range(6) for m in range(2) for p in range(18)}
    boot = a.paired_bootstrap(parents)
    assert boot["point"] == boot["lower"] == boot["upper"] == .125
    assert boot["draws"] == 20000 and boot["seed"] == 730217 and boot["endpoint_indices"] == [499, 19499]


def test_primary_gain_one_correct_case_loss_disqualifies_even_with_winning_secondary():
    science, panel, cells = labels_and_cells()
    cid = science[0]["case_id"]; cells[cid, a.GUARD]["response"] = "0"
    result = a.analyze_scored(science, panel, cells, score, panel_score)
    assert result["primary"]["delta"] > .45 and result["primary"]["conditional95_interval"]["lower"] > 0
    assert not result["primary"]["eligible_restricted_quality_result"]
    assert result["primary_case_contrasts"]["UNIT"]["losses"] == [cid]
    assert result["accuracy"][a.CONFIDENT] == 1 and result["primary"]["policy"] == a.GUARD
    assert not result["secondary_policy_promotion"] and not result["repeats_independent_units"]


def test_new_format_and_false_unknown_harms_are_not_aggregate_cancelled():
    science, panel, cells = labels_and_cells()
    first, second = science[0]["case_id"], science[1]["case_id"]
    cells[first, a.GUARD]["response"] = "bad"; cells[second, "UNIT"]["response"] = "bad"
    cells[first, "UNIT"]["response"] = "0"  # correctness loss irrelevant; format harm still gates
    result = a.analyze_scored(science, panel, cells, score, panel_score)
    assert result["primary_case_contrasts"]["UNIT"]["format_valid_count_delta"] == 0
    assert result["primary_case_contrasts"]["UNIT"]["new_case_endpoint_harms"]["format_invalid"] == [first]
    assert not result["primary"]["eligible_restricted_quality_result"]


def test_panel_loss_and_missing_or_duplicate_label_cells_fail_closed():
    science, panel, cells = labels_and_cells(); cells[panel[0]["case_id"], a.GUARD]["response"] = "copy"
    result = a.analyze_scored(science, panel, cells, score, panel_score)
    assert result["primary"]["copy_panel_loss_cases"]["UNIT"] == [panel[0]["case_id"]]
    assert not result["primary"]["eligible_restricted_quality_result"]
    with pytest.raises(ValueError): a.analyze_scored(science[:-1], panel, cells, score, panel_score)
    with pytest.raises(ValueError): a.analyze_scored(science + [science[0]], panel, cells, score, panel_score)


def raw_fixture():
    resources = {"gpu_peak_allocated_bytes": 2 * a.GIB, "gpu_free_bytes": 2 * a.GIB,
                 "rss_bytes": a.GIB, "host_available_bytes": 2 * a.GIB}
    population, outputs, events = [], [], []; api = 0
    def call(prefix, k, stage, phase, cid=None, path=None, failed=False, backend="fixture", error=None):
        nonlocal api
        api += 1
        base = {"API_attempt": api, "local_API_attempt": api, "backend": backend, "phase": phase,
                "prefix_tokens": len(prefix), "prefix_ids_sha256": a.digest(prefix), "K": k, "stage": stage,
                "depth": 28 + 2 * k, "case_id": cid, "path": path, "parent_elapsed_seconds": api / 100}
        events.append({**base, "event": "FORWARD_STARTED"})
        terminal = {**base, "event": "FORWARD_FAILED" if failed else "FORWARD_COMPLETED", "resources": resources}
        if failed: terminal.update(error=error, partial_compute="UNKNOWN")
        events.append(terminal); return api
    for i, length in enumerate((60, 37, 34, 32, 30, 28, 37, 37)):
        for mode in range(7): call([1] * length, int(mode in (3, 4, 5)), "GENERATION", "QUALIFICATION_PREFIX", cid="smoke" + str(i), backend="prefix")
    for mode in range(7): call([1] * 357, int(mode in (3, 4, 5)), "GENERATION", "QUALIFICATION_PREFIX", backend="prefix")
    call([1] * 357, 1, "GENERATION", "QUALIFICATION_PREFIX", backend="prefix", failed=True, error="RuntimeError: EXPECTED_FRESH_PREFIX_EXCEPTION")
    call([1] * 357, 0, "GENERATION", "QUALIFICATION_PREFIX", backend="prefix")
    for cache, present in ((True, False), (False, True)):
        api += 1; base = {"API_attempt": api, "local_API_attempt": api, "backend": "fixture", "phase": "QUALIFICATION",
            "use_cache": cache, "cache_object_present": present, "parent_elapsed_seconds": api / 100}
        events.append({**base, "event": "CACHE_REJECTION_STARTED"}); events.append({**base, "event": "CACHE_REJECTED", "model_compute_started": False})
    for i in range(8):
        for mode in range(5): call([1] * 30, int(mode % 2), "GENERATION", "QUALIFICATION_GENERATION", cid="smoke" + str(i))
    for mode in range(9):
        call([1] * 357, int(mode % 2), "GENERATION", "QUALIFICATION_MAX_DRIVER", failed=mode == 6,
             error="RuntimeError: EXPECTED_FRESH_DRIVER_EXCEPTION" if mode == 6 else None)
    qual_end = api
    rngs = {seed: random.Random(seed) for seed in (17, 29)}; route = {seed: [] for seed in rngs}
    for i in range(456):
        cid = f"raw{i}"; cohort = "SCIENTIFIC" if i < 432 else "PRESERVATION"
        population.append({"case_id": cid, "cohort": cohort, "raw_prompt": "fixture", "input_token_ids": [1, 2], "prefix_ids_sha256": a.digest([1, 2])})
        for seed, rng in rngs.items(): route[seed].append(int(rng.random() < 160 / 192))
        for path in a.PATHS:
            k = 0 if path in ("K0", a.UNCERTAIN) else route[int(path[-2:])] [-1] if path.startswith("RANDOM") else 1
            pilot = path in (a.CONFIDENT, a.UNCERTAIN); ids = []
            if pilot: ids.append(call([1, 2], 0, "PILOT", "TASK", cid, path))
            if not pilot or k: ids.append(call([1, 2], k, "GENERATION", "TASK", cid, path))
            cert = "CONFIDENCE_RULE" if pilot else "UNRECOGNIZED" if path in (a.GUARD, "PRIMARY_REPEAT") else "HEADER_CUE" if path == "HEADER_CUE" else "MATCHED_RANDOM_K" if path.startswith("RANDOM") else "FIXED_K_CONTROL"
            outputs.append({"case_id": cid, "cohort": cohort, "path": path, "response": "1", "token_ids": [151645],
                "termination": "eos", "K": k, "alpha": 1., "generation_cap": 8, "state_adjoints": 0, "gold_calls": 0,
                "prefix_ids_sha256": a.digest([1, 2]), "physical_run": True, "safety_stopped": False, "eos_emitted": True,
                "truncated": False, "pilot_used": pilot, "logical_first_reused": pilot and k == 0,
                "pilot_probability": .75 if pilot else None, "pilot_logits_sha256": "same" if pilot else None,
                "selected_first_logits_sha256": "selected", "recognized": False, "certificate": cert,
                "physical_API_attempts": ids, "full_model_calls": len(ids), "extra_layer_applications": 2 * k,
                "completed_token_positions": 2 * len(ids)})
    done = [x for x in events if x["event"] == "FORWARD_COMPLETED"]
    count = {"model_API_attempts": api, "completed_forwards": len(done), "full_forward_attempts": api - 2,
             "failed_forwards": 2, "cache_rejections": 2, "completed_prefix_positions": sum(x["prefix_tokens"] for x in done)}
    receipt = {"status": "FRESH_K01_RAW_MATRIX_COMPLETE_NOT_ANALYZED", "complete": True, "before_state_sha256": a.STATE,
               "after_state_sha256": a.STATE, "source_unchanged": True, "parameter_gradients": "NONE", "counts": count}
    qual = {"status": "PASS_FRESH_K01_NATIVE_MECHANICS", "max_driver_API_attempts": 9, "minimal_probability_native_parity": True,
            "forced_exception_restoration": True, "before_state_sha256": a.STATE, "after_state_sha256": a.STATE,
            "source_unchanged": True, "parameter_gradients": "NONE",
            "prefix_counts": {"model_API_attempts": 67, "full_forward_attempts": 65, "completed_forwards": 64,
                "expected_partial": 1, "cache_rejections": 2, "completed_prefix_positions": 4921, "state_adjoints": 0},
            "prefix_parity": [{"case_id": str(i), "hashes": ["a", "a", "a", "b", "b", "b", "a"]} for i in range(9)],
            "full_generation_parity": [{"case_id": "smoke" + str(i), "K0_tokens": [151645], "unit_tokens": [151645],
                                         "score_status": "NOT_COLLECTED_NOT_SCORED"} for i in range(8)],
            "max_driver_proof": {"forced_exception_restoration": True}, "actual_input_policy_mapping": [{} for _ in range(8)]}
    forecast = {"admitted": True, "task_API_cap": 33744, "task_rows": 4104, "forward_factor": 1.5,
                "internal_reserve_seconds": 1800, "parent_deadline_seconds": 15600, "remaining_forward_seconds": 10.,
                "remaining_nonforward_seconds": 10., "elapsed_seconds": (qual_end + .5) / 100}
    routes = {"rate": 160 / 192, "seed17": route[17], "seed29": route[29], "order_ids_sha256": a.digest([x["case_id"] for x in population])}
    return population, outputs, events, receipt, qual, forecast, routes


def test_full_raw_matrix_gate_then_duplicate_missing_and_resource_failure_refuse():
    data = raw_fixture(); errors, _, _ = a.raw_gate(*data)
    assert errors == []
    bad = list(data); bad[1] = data[1] + [data[1][0]]
    assert a.raw_gate(*bad)[0]
    bad = list(data); bad[1] = data[1][1:]
    assert a.raw_gate(*bad)[0]
    bad = copy.deepcopy(data); bad[2][1]["resources"]["rss_bytes"] = 7 * a.GIB
    assert a.raw_gate(*bad)[0]


def test_failed_native_gate_never_opens_labels_and_nulls_primary(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "V3", tmp_path)
    def failed(_): raise RuntimeError("terminal failed")
    def forbidden(_): raise AssertionError("fresh labels opened beforeproof")
    monkeypatch.setattr(a, "load_native", failed); monkeypatch.setattr(a, "read_offline_labels", forbidden)
    report = a.run("not accessed", tmp_path / "analyses/out")
    assert report["status"] == "INCOMPLETE_NO_EFFICACY" and report["primary"] is None and not report["fresh_labels_read"]


def test_reader_imports_are_stdlib_and_no_native_worker_or_torch():
    tree = ast.parse(inspect.getsource(a))
    names = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {x.name for n in ast.walk(tree) if isinstance(n, ast.Import) for x in n.names}
    assert not {"torch", "transformers", "fresh_native", "opposed_gate_generation", "gpu_native"}.intersection(names)
