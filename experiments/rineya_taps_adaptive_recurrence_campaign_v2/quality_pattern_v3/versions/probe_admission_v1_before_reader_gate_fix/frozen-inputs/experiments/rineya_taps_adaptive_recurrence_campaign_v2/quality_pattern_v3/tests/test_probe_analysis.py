"""Standard-library fixtures; no Torch or models, including in the test process."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import probe_analysis as reader


def stat(iteration=0):
    old, new = (1., 2.) if iteration == 0 else (2., 3.)
    return {"old_l2": old, "new_l2": new, "delta_l2": 1., "old_rms": old, "new_rms": new,
        "normal_rms": 2., "delta_rms": 1., "state_dot": old * new,
        "state_cosine_denominator": old * new, "state_cosine": 1.,
        "delta_over_old_l2_denominator": old, "delta_over_old_l2": 1. / old,
        "zero_old_norm": False, "zero_new_norm": False, "zero_delta_norm": False,
        "denominator_policy": reader.DENOM_POLICY, "finite": True,
        "previous_delta_l2": None if iteration == 0 else 1.,
        "delta_dot_previous": None if iteration == 0 else 1.,
        "delta_cosine_denominator": None if iteration == 0 else 1.,
        "delta_cosine_previous": None if iteration == 0 else 1.,
        "delta_over_previous_l2": None if iteration == 0 else 1.}


def fixture():
    bank, tokens, old_bank, old = [], [], [], []
    for i in range(192):
        cid, prompt = str(i), "fixture input " + str(i)
        row = {"case_id": cid, "split": "DEVELOPMENT", "raw_prompt": prompt,
               "raw_prompt_sha256": reader.text_sha(prompt)}
        ids = [1] * (129 if i < 32 else 128)
        bank.append(row)
        tokens.append({**row, "rendered_prompt": prompt, "rendered_prompt_sha256": reader.text_sha(prompt),
                       "input_token_ids": ids, "input_tokens": len(ids)})
        old_bank.append({"id": cid, "split": "DEVELOPMENT", "parent_id": "p" + str(i // 2), "category": "f" + str(i % 6)})
        aa, uu = ((1, 1), (1, 0), (0, 1), (0, 0))[i % 4]
        for arm, score in (("A0", aa), ("A1", uu)):
            old.append({"case_id": cid, "split": "DEVELOPMENT", "physical_run": True, "arm": arm, "seed": 17, "score": score})
    _, population = reader.validate_export(bank, tokens)
    events, features, readouts, proofs = [], [], [], []
    api, clock = 0, 0.
    resources = {"gpu_allocated_bytes": 2 * 1024**3, "gpu_free_bytes": 3 * 1024**3,
                 "host_available_bytes": 7 * 1024**3, "rss_bytes": 1024**3}
    def forward(context, length, prefix_hash, failed=False):
        nonlocal api, clock
        api += 1; clock += .01
        base = {**context, "API_attempt": api, "prefix_tokens": length, "prefix_ids_sha256": prefix_hash,
                "native_layers": [12, 13], "extra_loops": int(context["mode"].endswith("UNIT"))}
        events.append({**base, "event": "FORWARD_STARTED", "completed": False, "parent_elapsed_seconds": clock})
        clock += .01
        if failed:
            events.append({**base, "event": "FORWARD_FAILED", "completed": False, "parent_elapsed_seconds": clock,
                "forward_seconds": .01, "expected_forced_exception": True, "partial_compute": "UNKNOWN",
                "error": "RuntimeError: EXPECTED_PASSIVE_FORCED_EXCEPTION"})
        else:
            h = "b" * 64 if context["mode"].endswith("UNIT") else "a" * 64
            events.append({**base, "event": "FORWARD_COMPLETED", "completed": True, "parent_elapsed_seconds": clock,
                "forward_seconds": .01, "resources": resources, "post_forward_guard_error": None, "logits_sha256": h})
        return base
    def feature(context, length, task):
        for iteration in ([0, 1] if context["mode"] == "OBSERVED_UNIT" else [0]):
            features.append({**context, "schema": reader.SCHEMA, "sample_id": context["case_id"],
                "forward_index": 1, "iteration": iteration, "prefix_tokens": length, "last_token_index": length - 1,
                "stage": "ORDINARY_PRE_EXTRA" if iteration == 0 else "UNIT_PROPOSAL_POST_EXTRA",
                "available_before_extra": iteration == 0, "extra_block_already_computed": iteration == 1,
                "prefix_kind": "SYNTHETIC_MAX357_NO_TARGET" if context["split"] == "MECHANICAL" else "ORIGINAL_INPUT_ONLY",
                "preceding_gold_tokens": 0, "target_independent": True, "teacher_conditioned": False,
                "alpha": 1., "statistics": stat(iteration), "task_measurement": task})
    for i in range(9):
        cid = "smoke" + str(i) if i < 8 else "SYNTHETIC_MAX357_NO_TARGET"
        length = 36 if i < 7 else 43 if i == 7 else 357
        context = {"case_id": cid, "split": "SMOKE" if i < 8 else "MECHANICAL", "phase": "QUALIFICATION"}
        hashes = []
        for mode in reader.QUAL_MODES:
            forward({**context, "mode": mode}, length, "c" * 64)
            hashes.append(events[-1]["logits_sha256"])
            if mode.startswith("OBSERVED"):
                feature({**context, "mode": mode}, length, False)
        proofs.append({"context": context, "modes": list(reader.QUAL_MODES), "logits_sha256": hashes,
                       "observer_seconds": [0.] * 7})
    context = {"case_id": "SYNTHETIC_MAX357_NO_TARGET", "split": "MECHANICAL", "phase": "QUALIFICATION"}
    forward({**context, "mode": "OBSERVED_UNIT", "expected_forced_exception": True}, 357, "c" * 64, True)
    forward({**context, "mode": "RESTORED_INTACT", "after_forced_exception": True}, 357, "c" * 64)
    for use, present in ((True, False), (False, True)):
        api += 1; clock += .01
        base = {"API_attempt": api, "use_cache": use, "cache_object_present": present, "task_measurement": False}
        events.append({**base, "event": "CACHE_REJECTION_STARTED", "parent_elapsed_seconds": clock})
        clock += .01
        events.append({**base, "event": "CACHE_REJECTED", "parent_elapsed_seconds": clock,
                       "completed": False, "model_compute_started": False})
    forecast_elapsed = clock + .005
    for row in population:
        for mode in ("OBSERVED_K0", "OBSERVED_UNIT"):
            context = {k: row[k] for k in ("case_id", "split", "prefix_ids_sha256", "prompt_sha256", "rendered_prompt_sha256")}
            context.update(phase="TASK", mode=mode)
            forward(context, row["prefix_tokens"], row["prefix_ids_sha256"])
            feature(context, row["prefix_tokens"], True)
            readouts.append({**context, "schema": reader.SCHEMA, "vocabulary_rows": 151936, "UNKNOWN_token_id": 51378,
                "UNKNOWN_logit": 2., "rival_logit": 3., "rival_token_id": 100, "UNKNOWN_minus_rival_margin": -1.,
                "rival_set": reader.RIVAL_SET, "top1_token_id": 100, "top1_probability": math.exp(-1),
                "top2_token_ids": [100, 51378], "top2_logits": [3., 2.], "top2_margin": 1., "entropy_nats": 1.,
                "logsumexp": 4., "logits_changed": False, "target_independent": True, "finite": True,
                "prefix_kind": "ORIGINAL_INPUT_ONLY", "preceding_gold_tokens": 0, "teacher_conditioned": False,
                "observer_seconds": .001, "readout_observer_seconds": .001, "extra_model_calls_for_statistics": 0})
    hashes = {str(i): "d" * 64 for i in range(10)}
    dispatch = {"fixture.py": "e" * 64}
    admission_sha = "f" * 64
    admission = {"status": "READY_FOR_PASSIVE_PROBE_FP32_DISPATCH", "study_id": reader.STUDY_ID, "lane": reader.LANE,
        "count_contract": reader.CONTRACT, "training": False, "native_test": False, "checkpoint_files": hashes,
        "dispatch_hashes": dispatch, "dev_bank_sha256": "g", "dev_token_binding_sha256": "h"}
    runtime = {"tf32": False, "CPU_threads": 4, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    registration = {"lane": reader.LANE, "study_id": reader.STUDY_ID, "schema": reader.SCHEMA,
        "count_contract": reader.CONTRACT, "native_layers": [12, 13], "extra_loops": [0, 1], "alpha": 1., "seed": 17,
        "prefix_kind": "ORIGINAL_INPUT_ONLY", "generation": False, "teacher_conditioned": False, "training": False,
        "state_adjoints": 0, "readout_changed": False, "controller_routing": False, "hard_deadline_seconds": 900,
        "body_deadline_seconds": 780, "internal_reserve_seconds": 120, "global_finalization_reserve_seconds": 1800,
        "admission_sha256": admission_sha, "data_sha256": "g", "token_binding_sha256": "h", "runtime": runtime}
    preservation = {"before_state_sha256": reader.STATE, "after_state_sha256": reader.STATE,
        "checkpoint_files_before": hashes, "checkpoint_files_after": hashes, "dispatch_hashes": dispatch,
        "parameter_gradients": "NONE", "checkpoint_files_unchanged": True, "source_unchanged": True}
    qualification = {**preservation, "status": "PASS_PASSIVE_INPUT_FEATURE_MECHANICS", "counts": reader.QUAL_COUNTS,
        "maximum_prefix_tokens": 357, "cache_rejections": 2, "parity_proofs": proofs,
        "forced_exception_restoration": True, "L0_L2_one_shot": True, "head_and_operator_unchanged": True,
        "observer_output_identity": True}
    receipt = {**preservation, "status": "PASSIVE_INPUT_FEATURE_PROBE_COMPLETE_NOT_EFFICACY", "complete": True,
        "counts": reader.COUNTS, "cases": 192, "task_readouts": 384, "task_feature_events": 576,
        "loaded_custom_code_sha256": reader.CUSTOM, "trainable_parameters": 0, "native_model_loads": 1,
        "model_load_attempts": 1, "training": False, "native_test": False, "state_adjoints": 0,
        "efficacy_claim": False, "runtime": runtime, "parent_elapsed_seconds": clock + .01}
    terminal = {"status": "WORKER_TERMINAL", "exit_code": 0, "study_id": reader.STUDY_ID, "lane": reader.LANE,
        "admission_sha256": admission_sha, "native_worker_seconds": clock + .1, "run": "fixture", "worker_pid": 1,
        "peak_rss_bytes": 1024**3}
    ledger = [{"event": "native_started", "study_id": reader.STUDY_ID, "run": "fixture"},
              {"event": "native_completed", **terminal}]
    return {"admission": admission, "admission_sha256": admission_sha, "registration": registration,
        "qualification": qualification, "receipt": receipt, "terminal": terminal, "ledger": ledger,
        "forecast": {"admitted": True, "remaining_calls": 384, "forecast_factor": 1.5, "internal_reserve_seconds": 120,
            "parent_deadline_seconds": 900, "max_observed_warm_forward_seconds": .01,
            "remaining_forecast_seconds": 384 * .01 * 1.5, "elapsed_seconds": forecast_elapsed},
        "export_bank": bank, "export_tokens": tokens, "population": population, "forwards": events,
        "features": features, "readouts": readouts, "old_bank": old_bank, "old_outcomes": old,
        "resources": [{"rss_bytes": 1024**3, "host_available_bytes": 7 * 1024**3}], "binding_errors": []}


class ProbeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.bundle = fixture()

    def assert_refused(self, bundle, fragment):
        report = reader.analyze(bundle)
        self.assertIsNone(report["primary"])
        self.assertTrue(any(fragment in x for x in report["errors"]), report["errors"])

    def test_complete_in_memory_matrix_and_all_four_groups(self):
        report = reader.analyze(self.bundle)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["counts"]["completed_prefix_positions"], 54137)
        self.assertEqual({g: x["cases"] for g, x in report["primary"]["groups"].items()},
                         dict.fromkeys(("both_correct", "A0_only", "unit_only", "both_wrong"), 48))
        self.assertEqual(len(report["primary"]["parent_rows"]), 96)
        self.assertEqual(len(report["primary"]["family"]), 6)
        self.assertTrue(report["primary"]["A0_only_vs_unit_only_range_overlap"]["ordinary.delta_l2"]["ranges_overlap"])
        self.assertIsNone(report["generation_or_quality_effect"])

    def test_incomplete_task_has_null_primary(self):
        self.bundle["readouts"].pop()
        self.assert_refused(self.bundle, "384 readout")

    def test_missing_or_malformed_packet_never_emits_partial_primary(self):
        self.assertIsNone(reader.analyze({})["primary"])
        self.bundle["readouts"][0] = None
        self.assert_refused(self.bundle, "malformed")

    def test_cache_compute_and_unexpected_failure_fail_closed(self):
        cache = next(x for x in self.bundle["forwards"] if x["event"] == "CACHE_REJECTED")
        cache["model_compute_started"] = True
        self.assert_refused(self.bundle, "before compute")
        broken = fixture()
        next(x for x in broken["forwards"] if x["event"] == "FORWARD_FAILED")["expected_forced_exception"] = False
        self.assert_refused(broken, "unexpected failure")

    def test_labels_nested_metadata_and_teacher_contamination_refuse(self):
        for kind in ("label", "teacher", "metadata"):
            broken = copy.deepcopy(self.bundle)
            row = next(x for x in broken["features"] if x["task_measurement"])
            if kind == "label": row["gold"] = "UNKNOWN"
            elif kind == "teacher": row["preceding_gold_tokens"] = 1
            else: row["statistics"]["metadata"] = {"source_status": "CONFLICT"}
            self.assert_refused(broken, "feature")

    def test_stage_and_common_ordinary_input_identity_refuse(self):
        broken = copy.deepcopy(self.bundle)
        row = next(x for x in broken["features"] if x["task_measurement"] and x["iteration"] == 1)
        row["available_before_extra"] = True
        self.assert_refused(broken, "availability")
        row = next(x for x in self.bundle["features"] if x["task_measurement"] and x["mode"] == "OBSERVED_UNIT" and x["iteration"] == 0)
        row["statistics"]["normal_rms"] += 1
        self.assert_refused(self.bundle, "ordinary same-prefix")

    def test_known_raw_unknown_margin_and_top2_operands(self):
        self.assertEqual(reader.validate_readout(self.bundle["readouts"][0]), [])
        self.bundle["readouts"][0]["UNKNOWN_minus_rival_margin"] = -.5
        self.assert_refused(self.bundle, "raw FP32 operands")
        row = copy.deepcopy(self.bundle["readouts"][1]); row["rival_set"] = "DIGITS_ONLY"
        self.assertTrue(reader.validate_readout(row))

    def test_raw_ratio_zero_null_policy_and_nonfinite(self):
        s = stat(); s.update(old_l2=0., old_rms=0., state_dot=0., state_cosine_denominator=0.,
                            state_cosine=None, delta_over_old_l2_denominator=0., delta_over_old_l2=None, zero_old_norm=True)
        self.assertEqual(reader.validate_statistics(s, 0), [])
        s["delta_over_old_l2"] = 0.
        self.assertTrue(reader.validate_statistics(s, 0))
        self.bundle["readouts"][0]["entropy_nats"] = math.nan
        self.assert_refused(self.bundle, "finite")

    def test_false_parity_forecast_state_and_study_ledger_refuse(self):
        for kind, fragment in (("parity", "parity"), ("forecast", "warm journal"), ("state", "stateb23"), ("ledger", "ledger")):
            broken = copy.deepcopy(self.bundle)
            if kind == "parity": broken["qualification"]["parity_proofs"][0]["logits_sha256"][2] = "z" * 64
            elif kind == "forecast":
                broken["forecast"]["max_observed_warm_forward_seconds"] = .02
                broken["forecast"]["remaining_forecast_seconds"] = 384 * .02 * 1.5
            elif kind == "state": broken["receipt"]["after_state_sha256"] = "0" * 64
            else: broken["ledger"][0]["study_id"] = "OTHER"
            self.assert_refused(broken, fragment)

    def test_actual_export_membership_without_fabricated_labels(self):
        folder = reader.STUDY / "data/dev_prefix_probe_001"
        for name, expected in reader.EXPORT_PINS.items():
            self.assertEqual(reader.sha(folder / name), expected)
        bank = json.loads((folder / "DEVELOPMENT_BANK.json").read_text(encoding="utf-8"))
        tokens = json.loads((folder / "DEVELOPMENT_TOKEN_BINDING.json").read_text(encoding="utf-8"))["rows"]
        errors, population = reader.validate_export(bank, tokens)
        self.assertEqual(errors, [])
        self.assertEqual(sum(row["prefix_tokens"] for row in population), 24608)
        bank[0]["category"] = "forbidden"
        self.assertTrue(reader.validate_export(bank, tokens)[0])

    def test_exact_reader_plan_membership_no_manifest_fallback(self):
        admission = {"dispatch_hashes": {"reader.py": "a", "plan.md": "b"},
                     "analysis_source_sha256": "a", "analysis_plan_sha256": "b"}
        manifest = {"sha256": admission["dispatch_hashes"]}
        self.assertEqual(reader.analysis_membership(admission, manifest, admission["dispatch_hashes"], "reader.py", "plan.md"), [])
        changed = copy.deepcopy(admission); changed["dispatch_hashes"].pop("plan.md")
        self.assertTrue(reader.analysis_membership(changed, manifest, manifest["sha256"], "reader.py", "plan.md"))

    def test_actual_admission_membership_once_frozen(self):
        path = reader.STUDY / "probe_admission_v1.json"
        if not path.exists():
            self.skipTest("No frozen admission yet: actual export checked; this test must rerun after freeze before dispatch.")
        admission = json.loads(path.read_text(encoding="utf-8"))
        current = {k: reader.sha(reader.ROOT / k) for k in admission["dispatch_hashes"]}
        self.assertEqual(reader.analysis_membership(admission, {"sha256": admission["dispatch_hashes"]}, current,
            Path(reader.__file__).resolve().relative_to(reader.ROOT).as_posix(),
            (reader.STUDY / "PROBE_ANALYSIS_PLAN_001.md").relative_to(reader.ROOT).as_posix()), [])

    def test_module_and_tests_import_no_torch(self):
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
