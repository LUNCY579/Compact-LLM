import ast
import copy
import inspect
import json
import random
from datetime import datetime, timedelta, timezone

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
        base["timestamp_utc"] = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=api / 100)).isoformat()
        events.append({**base, "event": "FORWARD_STARTED"})
        terminal = {**base, "event": "FORWARD_FAILED" if failed else "FORWARD_COMPLETED", "resources": resources, "elapsed_seconds": .001}
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
        base["timestamp_utc"] = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=api / 100)).isoformat()
        events.append({**base, "event": "CACHE_REJECTION_STARTED"}); events.append({**base, "event": "CACHE_REJECTED", "model_compute_started": False, "forward_seconds": .001})
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
                "completed_token_positions": 2 * len(ids), "wall_seconds": .1, "physical_forward_seconds": .01,
                "probability_seconds": .001 if pilot else 0., "prefill_seconds": .05, "decode_seconds": .05})
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
                "internal_reserve_seconds": 1800, "parent_deadline_seconds": 15600,
                "max_observed_warm_forward_seconds": .001, "nonforward_per_row_bound_seconds": .01,
                "remaining_forward_seconds": 33744 * 1.5 * .001, "remaining_nonforward_seconds": 4104 * .01,
                "remaining_forecast_seconds": 33744 * 1.5 * .001 + 4104 * .01,
                "elapsed_seconds": (qual_end + .5) / 100}
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


def test_nonfinite_negative_or_disordered_native_cost_time_resource_refuses():
    for where, key, value in (("event", "elapsed_seconds", float("nan")), ("event", "parent_elapsed_seconds", -1.),
                              ("resource", "rss_bytes", float("nan")), ("row", "wall_seconds", float("inf"))):
        fixture = copy.deepcopy(raw_fixture())
        if where == "event": fixture[2][1][key] = value
        elif where == "resource": fixture[2][1]["resources"][key] = value
        else: fixture[1][0][key] = value
        assert a.raw_gate(*fixture)[0]
    fixture = raw_fixture(); fixture[2][1]["timestamp_utc"] = "2025-01-01T00:00:00+00:00"
    assert a.raw_gate(*fixture)[0]


@pytest.mark.parametrize("field", ["remaining_forward_seconds", "remaining_nonforward_seconds", "remaining_forecast_seconds",
                                   "elapsed_seconds", "max_observed_warm_forward_seconds", "nonforward_per_row_bound_seconds"])
def test_negative_forecast_component_cannot_discount_full_run(field):
    fixture = raw_fixture(); fixture[5][field] = -1.
    assert "complete all-control forecast proof missing" in a.raw_gate(*fixture)[0]


def test_forecast_sum_factor_and_reserve_reconcile_exact_registered_scope():
    for field, value in (("remaining_forecast_seconds", 1.), ("forward_factor", 1.), ("internal_reserve_seconds", 0),
                         ("max_observed_warm_forward_seconds", 0.), ("nonforward_per_row_bound_seconds", float("nan"))):
        fixture = raw_fixture(); fixture[5][field] = value
        assert "complete all-control forecast proof missing" in a.raw_gate(*fixture)[0]


def test_reader_exact_v2_admission_and_review_path_reject_prior_or_generic(tmp_path, monkeypatch):
    assert a.ADMISSION.name == "fresh_admission_v2.json"
    monkeypatch.setattr(a, "V3", tmp_path)
    (tmp_path / "reviews").mkdir()
    expected = tmp_path / "reviews/fresh_dispatch_review_002.json"; expected.write_text("{}", encoding="utf-8")
    assert a.review_receipt_path({"review_receipt": "reviews/fresh_dispatch_review_002.json"}) == expected
    for path in ("reviews/fresh_dispatch_review_001.json", "review.json", "../outside.json"):
        with pytest.raises(PermissionError): a.review_receipt_path({"review_receipt": path})


def test_pinned_export_population_join_rejects_detached_tokens_prompts_order_and_missing_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "ROOT", tmp_path); monkeypatch.setattr(a, "DATA", tmp_path / "data")
    directory = a.DATA / "model_inputs"; directory.mkdir(parents=True)
    admission = {"source": "source", "dispatch_hashes": {}}; hashes = {}; expected = []
    for key, cohort, size in (("scientific", "SCIENTIFIC", 432), ("preservation", "PRESERVATION", 24)):
        raw = []; tokens = []
        for i in range(size):
            text = "fixture🙂 " + str(i); h = __import__("hashlib").sha256(text.encode()).hexdigest()
            row = {"case_id": key + str(i), "raw_prompt": text, "raw_prompt_sha256": h}
            token = {**row, "rendered_prompt": text, "rendered_prompt_sha256": h, "input_token_ids": [1, 2], "input_tokens": 2}
            raw.append(row); tokens.append(token)
            expected.append({**token, "cohort": cohort, "prefix_ids_sha256": a.digest([1, 2])})
        for field, name, payload in ((key + "_bank", key.upper() + "_PROMPTS.json", raw),
            (key + "_token_binding", key.upper() + "_TOKEN_BINDING.json",
             {"status": "FROZEN_FRESH_PROMPT_ONLY_TOKEN_BINDING", "canonical_source": "source", "model_loads": 0, "rows": tokens})):
            path = directory / name; path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            hashes[name] = a.sha(path); admission[field + "_path"] = str(path); admission[field + "_sha256"] = hashes[name]
            admission["dispatch_hashes"][path.relative_to(a.ROOT).as_posix()] = hashes[name]
    manifest = {"status": "FROZEN_FRESH_OPEN_MODEL_EXPORT", "scientific_cases": 432, "preservation_cases": 24,
                "maximum_input_tokens": 350, "maximum_forward_prefix_tokens": 357, "labels_excluded": True,
                "tokenizer_source": "source", "model_loads": 0, "files": hashes}
    path = directory / "MODEL_EXPORT_MANIFEST.json"; path.write_text(json.dumps(manifest), encoding="utf-8")
    admission.update(prompt_export_manifest_path=str(path), prompt_export_manifest_sha256=a.sha(path))
    admission["dispatch_hashes"][path.relative_to(a.ROOT).as_posix()] = a.sha(path)
    assert a.validate_native_population(admission, expected) == expected
    for altered in (list(reversed(expected)), [{**expected[0], "input_token_ids": [1, 3]}] + expected[1:],
                    [{**expected[0], "raw_prompt": "detached"}] + expected[1:]):
        with pytest.raises(RuntimeError, match="detached"):
            a.validate_native_population(admission, altered)
    admission["scientific_bank_sha256"] = "0" * 64
    with pytest.raises(PermissionError): a.validate_native_population(admission, expected)
