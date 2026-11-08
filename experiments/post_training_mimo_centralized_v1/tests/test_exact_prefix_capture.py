"""Independent assembly, likelihood, scope and failure-cost regressions; no model."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import exact_prefix_capture as control
from experiments.post_training_mimo_centralized_v1.scripts import greedy_capture as old
from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting, verify_call_journal
from experiments.post_training_mimo_centralized_v1.scripts.qualify_exact_prefix import exclusive_lease, validate_go_binding, validate_returned_run, verify_worker_owner


class ExactPrefixControls(unittest.TestCase):
    def setUp(self):
        self.cfg = control.load_config(control.PACKAGE / control.CONFIG_RELATIVE)
        self.events = []
        self.calls = []

    def event(self, kind, **fields):
        self.events.append({"event": kind, **fields})

    def fake_forward(self, ids, target, stage):
        # Only these independently enumerated trace prefixes are valid inputs.
        table = {
            tuple(self.cfg["prompt_ids"]): (16, -.5026993751525879, "a" * 64),
            tuple(self.cfg["prompt_ids"] + [16]): (22, -.41050243377685547, "b" * 64),
            tuple(self.cfg["prompt_ids"] + [16, 22]): (151645, -.0005460678366944194, "c" * 64),
        }
        token, probability, digest = table[tuple(ids)]
        self.calls.append((stage, list(ids), target))
        return {"input_ids": list(ids), "input_ids_sha256": old.token_hash(ids), "target_id": token if target is None else target,
                "model_logprob": probability, "selected_raw_logit": 7., "raw_max_logit": 7.,
                "first_maximum_token_id": token, "raw_logits_sha256": digest}

    def test_independent_reverse_replay_excludes_current_and_future_targets(self):
        trace = control.capture_trace(self.cfg, self.fake_forward, self.event, {})
        self.assertEqual([x[0] for x in self.calls], ["online_0", "online_1", "online_2", "offline_2", "offline_1", "offline_0"])
        self.assertEqual([x[2] for x in self.calls], [None, None, None, 151645, 22, 16])
        self.assertEqual([len(x[1]) for x in self.calls], [128, 129, 130, 130, 129, 128])
        self.assertEqual(sum(len(x[1]) for x in self.calls), 774)
        self.assertEqual(trace["generated_ids"], [16, 22, 151645])
        self.assertEqual(trace["behavior_logprobs"], [0., 0., 0.])
        self.assertEqual(trace["likelihood"]["max_abs"], 0.)
        self.assertEqual(trace["offline_model_logprobs"], trace["online_model_logprobs"])
        self.assertEqual(trace["raw_logits_bitwise_equal"], [True, True, True])

    def test_prefix_assembly_returns_separate_saved_arrays(self):
        prompt, targets = [1, 2, 151645], [16, 22, 151645]
        ids, target = control.prefix_inputs(prompt, targets, 1)
        self.assertEqual((ids, target), ([1, 2, 151645, 16], 22))
        ids[0] = 9
        self.assertEqual(prompt, [1, 2, 151645])
        self.assertEqual(targets, [16, 22, 151645])

    def test_target_ordinals_reject_boolean_and_outside_trace(self):
        for ordinal in (True, -1, 3, 1.):
            with self.subTest(ordinal=ordinal), self.assertRaises(RuntimeError):
                control.prefix_inputs([1], [16, 22, 151645], ordinal)

    def test_target_after_first_eos_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "after terminal EOS"):
            control.prefix_inputs([1], [151645, 22, 151645], 0)

    def test_wrong_prompt_with_plausible_length_is_rejected(self):
        self.cfg["prompt_ids"][0] = 17
        with self.assertRaisesRegex(RuntimeError, "prompt identity"):
            control.capture_trace(self.cfg, self.fake_forward, self.event, {})
        self.assertFalse(self.calls)

    def test_future_suffix_or_changed_target_cannot_enter_offline_plan(self):
        self.cfg["expected_generated_ids"] = [16, 23, 151645]
        with self.assertRaisesRegex(RuntimeError, "target identity"):
            control.capture_trace(self.cfg, self.fake_forward, self.event, {})
        self.assertFalse(self.calls)

    def test_early_natural_eos_retains_partial_online_trace(self):
        def early(ids, target, stage):
            value = self.fake_forward(ids, target, stage)
            value.update(target_id=151645, first_maximum_token_id=151645)
            return value
        trace = {}
        with self.assertRaisesRegex(RuntimeError, "raw choice differs"):
            control.capture_trace(self.cfg, early, self.event, trace)
        self.assertEqual(trace["generated_ids"], [151645])
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(trace["offline"])

    def test_offline_different_scores_are_measured_instead_of_online_reuse(self):
        def divergent(ids, target, stage):
            value = self.fake_forward(ids, target, stage)
            if stage == "offline_0":
                value["model_logprob"] -= .04
            return value
        trace = {}
        with self.assertRaisesRegex(RuntimeError, "model likelihood qualification failed"):
            control.capture_trace(self.cfg, divergent, self.event, trace)
        self.assertEqual(len(self.calls), 6)
        self.assertGreater(trace["likelihood"]["max_abs"], .03125)
        self.assertFalse(trace["likelihood"]["passed"])

    def test_mean_tolerance_is_not_replaced_by_maximum_gate(self):
        def shifted(ids, target, stage):
            value = self.fake_forward(ids, target, stage)
            if stage in ("offline_0", "offline_1"):
                value["model_logprob"] -= .02
            return value
        trace = {}
        with self.assertRaisesRegex(RuntimeError, "model likelihood qualification failed"):
            control.capture_trace(self.cfg, shifted, self.event, trace)
        self.assertLess(trace["likelihood"]["max_abs"], .03125)
        self.assertGreater(trace["likelihood"]["mean_abs"], .01)

    def test_nonfinite_offline_likelihood_fails_and_remains_in_partial_trace(self):
        def nonfinite(ids, target, stage):
            value = self.fake_forward(ids, target, stage)
            if stage == "offline_2":
                value["model_logprob"] = float("nan")
            return value
        trace = {}
        with self.assertRaisesRegex(RuntimeError, "model likelihood qualification failed"):
            control.capture_trace(self.cfg, nonfinite, self.event, trace)
        self.assertFalse(trace["likelihood"]["finite"])
        self.assertIsNone(old.safe_numbers(trace)["offline"][0]["model_logprob"])

    def test_wrong_offline_target_is_not_accepted(self):
        def wrong(ids, target, stage):
            value = self.fake_forward(ids, target, stage)
            if stage == "offline_2":
                value["target_id"] = 151643
            return value
        trace = {}
        with self.assertRaisesRegex(RuntimeError, "offline prefix/target identity"):
            control.capture_trace(self.cfg, wrong, self.event, trace)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(trace["offline"][0]["target_id"], 151643)

    def test_wrong_offline_prefix_with_recomputed_hash_is_rejected(self):
        def wrong(ids, target, stage):
            value = self.fake_forward(ids, target, stage)
            if stage == "offline_2":
                value["input_ids"][-1] = 23
                value["input_ids_sha256"] = old.token_hash(value["input_ids"])
            return value
        with self.assertRaisesRegex(RuntimeError, "offline prefix/target identity"):
            control.capture_trace(self.cfg, wrong, self.event, {})

    def test_unchanged_tolerances_and_no_authority_expansion(self):
        for mutation in ({"scoring": {**self.cfg["scoring"], "logprob_max_atol": .05}},
                         {"scoring": {**self.cfg["scoring"], "logprob_mean_atol": .02}},
                         {"scoring": {**self.cfg["scoring"], "behavior_logprob_comparison": True}},
                         {"E1_authorized": True}, {"general_capture_qualified": True}, {"generation_cap": 4}):
            cfg = copy.deepcopy(self.cfg)
            cfg.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                control.validate_plan(cfg)
        self.assertNotEqual(control.PASS_STATUS, old.PASS_STATUS)

    def test_inconsistent_historical_reproduction_cannot_pass_exact_match_replay(self):
        self.cfg["historical_online_model_logprobs"][0] -= .04
        trace = {}
        with self.assertRaisesRegex(RuntimeError, "model likelihood qualification failed"):
            control.capture_trace(self.cfg, self.fake_forward, self.event, trace)
        self.assertTrue(trace["likelihood"]["passed"])
        self.assertFalse(trace["historical_reproduction"]["passed"])

    def test_registered_native_prerequisites_preserve_failed_bulk(self):
        root = control.PACKAGE.parents[1]
        _, failed = control.prerequisite_inputs(root, self.cfg)
        self.assertFalse(failed["passed"])
        self.assertGreater(failed["max_abs"], .03125)
        self.assertGreater(failed["mean_abs"], .01)

    def test_attempted_forward_limit_blocks_before_seventh_callback(self):
        accounting = CallAccounting(self.event, maximum_optimizer_calls=0)
        calls = old.BudgetedCalls(accounting, self.cfg)
        count = []
        for i in range(6):
            calls.call("forward", lambda: count.append(1), stage="fixture_" + str(i), input_positions=128)
        with self.assertRaisesRegex(RuntimeError, "FORWARD_CEILING"):
            calls.call("forward", lambda: count.append(1), stage="extra", input_positions=1)
        self.assertEqual(len(count), 6)
        self.assertEqual(accounting.snapshot()["operations"]["forward"]["attempted"], 6)

    def test_position_limit_blocks_before_over_budget_callback(self):
        accounting = CallAccounting(self.event, maximum_optimizer_calls=0)
        calls = old.BudgetedCalls(accounting, self.cfg)
        calls.call("forward", lambda: None, stage="one", input_positions=774)
        with self.assertRaisesRegex(RuntimeError, "POSITION_CEILING"):
            calls.call("forward", lambda: self.fail("callback must not run"), stage="extra", input_positions=1)
        self.assertEqual(accounting.snapshot()["forward_input_positions_attempted"], 774)

    def test_no_training_callback_is_permitted(self):
        accounting = CallAccounting(self.event, maximum_optimizer_calls=0)
        calls = old.BudgetedCalls(accounting, self.cfg)
        with self.assertRaisesRegex(RuntimeError, "no training"):
            calls.call("backward", lambda: self.fail("must not run"), stage="forbidden")
        self.assertEqual(accounting.snapshot()["last_call_id"], 0)

    def test_failed_native_call_current_costs_are_identical_in_all_receipts(self):
        accounting = CallAccounting(self.event, maximum_optimizer_calls=0)
        accounting.call("forward", lambda: None, stage="returned", input_positions=128)
        def fail():
            raise RuntimeError("injected kernel failure")
        with self.assertRaisesRegex(RuntimeError, "injected kernel"):
            accounting.call("forward", fail, stage="failed", input_positions=129)
        snapshot = accounting.snapshot()
        verify_call_journal(self.events, snapshot, maximum_optimizer_calls=0)
        result = {"status": "BLOCKED_NATIVE_EXACT_PREFIX_FIXED_TRACE", "pretrained_loads": 1, "optimizer_steps": 0,
                  "training_positions": 0, "teacher_calls": 0, "trace": {"generated_ids": [16]}}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            control.persist_outcome(output, result, accounting, 1.5)
            failure = json.loads((output / "FAILURE_RECEIPT.json").read_bytes())
            terminal = json.loads((output / "CAPTURE_RESULT.json").read_bytes())
            ledger = json.loads((output / "COST_LEDGER.json").read_bytes())
            self.assertEqual(failure, terminal)
            self.assertEqual(failure["call_accounting"], ledger["call_accounting"])
            self.assertEqual(failure["call_accounting"], snapshot)
            self.assertEqual(snapshot["operations"]["forward"], {"attempted": 2, "returned": 1, "failed": 1, "pending": 0})
            self.assertEqual(snapshot["forward_input_positions_attempted"], 257)
            self.assertEqual(snapshot["forward_input_positions_returned"], 128)

    def test_pending_calls_survive_terminal_failure_snapshot(self):
        accounting = CallAccounting(self.event, maximum_optimizer_calls=0)
        accounting.begin("forward", stage="hard_kill_intent", input_positions=130)
        with tempfile.TemporaryDirectory() as temp:
            result = control.persist_outcome(Path(temp), {"status": "BLOCKED", "pretrained_loads": 1, "optimizer_steps": 0,
                "training_positions": 0, "teacher_calls": 0, "trace": {}}, accounting, 2.)
            self.assertEqual(result["call_accounting"]["operations"]["forward"]["pending"], 1)
            self.assertEqual(result["call_accounting"]["forward_input_positions_returned"], 0)

    def test_lease_release_cannot_remove_peer_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gpu.lock"
            with self.assertRaisesRegex(RuntimeError, "ownership changed"):
                with exclusive_lease(path, {"token": "mine"}):
                    path.write_bytes(b'{"token":"peer"}\n')
            self.assertEqual(json.loads(path.read_bytes())["token"], "peer")

    def test_existing_lease_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gpu.lock"
            path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                with exclusive_lease(path, {"token": "mine"}):
                    self.fail("lease should not be acquired")
            self.assertEqual(path.read_bytes(), b"existing")

    def test_static_go_is_rejected_before_any_platform_or_framework_work(self):
        with self.assertRaisesRegex(RuntimeError, "complete current supervisor GO"):
            verify_worker_owner({"config_sha256": control.CONFIG_SHA256}, {"code_sha": "a" * 40}, Path("unused"), Path("unused"))

    def test_go_binds_exact_snapshot_and_process_job_attempt_identities(self):
        source = {"code_sha": "a" * 40, "source_files": {"fixed.py": {"sha256": "f" * 64}}}
        go = {"config_sha256": control.CONFIG_SHA256, "source_sha": source["code_sha"],
              "source_snapshot_sha256": hashlib.sha256(control.canonical(source)).hexdigest(),
              "supervisor_pid": 123, "worker_root_pid": 456, "supervisor_created": 1., "worker_root_created": 2.,
              "job_handle": 789, "lease_token": "a" * 32}
        validate_go_binding(go, source)
        for key, value in (("source_sha", "b" * 40), ("source_snapshot_sha256", "b" * 64), ("job_handle", 0),
                           ("worker_root_pid", True), ("supervisor_created", float("nan")), ("lease_token", "stale")):
            bad = dict(go)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_go_binding(bad, source)
        changed = copy.deepcopy(source)
        changed["source_files"]["fixed.py"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "source/config binding"):
            validate_go_binding(go, changed)

    def test_parent_rejects_different_worker_source_even_with_pass_status(self):
        with self.assertRaisesRegex(RuntimeError, "source/config/status mismatch"):
            validate_returned_run({"status": control.PASS_STATUS, "code_sha": "b" * 40, "config_sha256": control.CONFIG_SHA256},
                                  Path("unused"), self.cfg, {"code_sha": "a" * 40}, Path("unused"))

    def test_parent_validates_returned_source_receipt_and_final_recheck(self):
        source = {"code_sha": "a" * 40, "source_files": {}}
        completed = {"status": control.PASS_STATUS, "code_sha": source["code_sha"], "config_sha256": control.CONFIG_SHA256}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            old.write(output / "SOURCE_RECEIPT.json", source)
            old.write(output / "CONFIG.json", self.cfg)
            with patch.object(old, "recheck_sources") as recheck:
                validate_returned_run(completed, output, self.cfg, source, Path("root"))
                recheck.assert_called_once_with(Path("root"), source)
            returned = {"code_sha": "b" * 40, "source_files": {}}
            (output / "SOURCE_RECEIPT.json").write_bytes(control.canonical(returned))
            with self.assertRaisesRegex(RuntimeError, "receipts differ"):
                validate_returned_run(completed, output, self.cfg, source, Path("root"))

    def test_hard_stop_recovery_keeps_pending_intent_and_partial_return_uncertainty(self):
        events = []
        accounting = CallAccounting(lambda kind, **fields: events.append({"event": kind, **fields}), maximum_optimizer_calls=0)
        accounting.call("forward", lambda: None, stage="online_0", input_positions=128)
        accounting.begin("forward", stage="online_1", input_positions=129)
        with tempfile.TemporaryDirectory() as temp:
            output, supervision = Path(temp) / "worker", Path(temp) / "supervisor"
            output.mkdir()
            supervision.mkdir()
            raw = b"".join(control.canonical(event) + b"\n" for event in events) + b'{"event":"operation_returned"'
            (output / "EVENTS.jsonl").write_bytes(raw)
            old.write(supervision / "SUPERVISOR.json", {"status": "BLOCKED", "reason": "WALL_TIME_LIMIT", "elapsed_seconds": 600.1})
            result = control.recover_worker_costs(output, supervision, self.cfg, {"code_sha": "a" * 40}, "watchdog stop")
            self.assertEqual(result["call_accounting"], accounting.snapshot())
            self.assertEqual(result["call_accounting"]["operations"]["forward"], {"attempted": 2, "returned": 1, "failed": 0, "pending": 1})
            self.assertTrue(result["generation_count_uncertain"])
            self.assertEqual(result["generation_tokens_attempted_upper_bound"], 2)
            self.assertEqual(result["incomplete_tail_bytes"], len(b'{"event":"operation_returned"'))
            failure = json.loads((output / "FAILURE_RECEIPT.json").read_bytes())
            terminal = json.loads((output / "CAPTURE_RESULT.json").read_bytes())
            ledger = json.loads((output / "COST_LEDGER.json").read_bytes())
            self.assertEqual(failure, terminal)
            self.assertEqual(ledger["call_accounting"], failure["call_accounting"])
            self.assertEqual((output / "EVENTS.jsonl").read_bytes(), raw)

    def test_recovery_preserves_worker_receipts_and_records_corrupt_journal_as_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            output, supervision = Path(temp) / "worker", Path(temp) / "supervisor"
            output.mkdir()
            supervision.mkdir()
            (output / "EVENTS.jsonl").write_bytes(b'{invalid JSON}\n')
            original = b'{"status":"ORIGINAL_PARTIAL"}\n'
            (output / "CAPTURE_RESULT.json").write_bytes(original)
            result = control.recover_worker_costs(output, supervision, self.cfg, {"code_sha": "a" * 40}, "external stop")
            self.assertFalse(result["journal_verified"])
            self.assertIsNone(result["call_accounting"])
            self.assertIsNone(result["pretrained_loads"])
            self.assertIsNone(result["generation_tokens"])
            self.assertTrue(result["generation_count_uncertain"])
            self.assertEqual((output / "CAPTURE_RESULT.json").read_bytes(), original)
            recovered = json.loads((supervision / "worker_recovery/CAPTURE_RESULT.json").read_bytes())
            self.assertEqual(recovered["call_accounting"], result["call_accounting"])

    def test_new_modules_import_without_model_frameworks(self):
        import sys
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)


if __name__ == "__main__":
    unittest.main()
