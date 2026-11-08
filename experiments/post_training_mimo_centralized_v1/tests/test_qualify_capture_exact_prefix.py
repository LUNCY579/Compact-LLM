"""Exact-prefix negative controls; stdlib only, no framework or native jobs."""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import qualify_capture_exact_prefix as q
from experiments.post_training_mimo_centralized_v1.scripts import greedy_capture as old
from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting


class Clock:
    def __init__(self):
        self.value = 0.
    def now(self):
        return self.value
    def pause(self, duration):
        self.value += duration


class ExactPrefixControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = q.load_config(q.ROOT / q.CONFIG_RELATIVE)

    def witness(self, **changes):
        return {"ordinal": 0, "target_id": 16, "prefix_token_ids_sha256": "a" * 64, "raw_logits_sha256": "b" * 64,
                "logits_shape": [1, 1, 151936], "logits_dtype": "bfloat16", "finite": True, "model_logprob": -.5, **changes}

    def test_config_exact_bytes_and_prospective_gates_cannot_drift(self):
        raw = (q.ROOT / q.CONFIG_RELATIVE).read_bytes()
        self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (29319, q.CONFIG_SHA256))
        for mode in ("tolerance", "gpu", "authority", "cap", "fixture"):
            cfg = deepcopy(self.cfg)
            if mode == "tolerance": cfg["teacher_forcing"]["logprob_max_atol"] = .03125
            if mode == "gpu": cfg["gpu_free_gib"] = 2.
            if mode == "authority": cfg["DEV_collection_authorized"] = True
            if mode == "cap": cfg["ceilings"]["forward_calls"] += 1
            if mode == "fixture": cfg["frozen_fixture_ids"]["128"]["ids"][0] += 1
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                q.validate_config(cfg)
        with self.assertRaises(RuntimeError):
            q.load_config(q.ROOT / "elsewhere.json")

    def test_tokenizer_reconstructs_all_frozen_lengths_and_refuses_different_ids(self):
        template = self.cfg["frozen_fixture_ids"]["128"]["ids"][-31:]
        options = []
        def apply(messages, **kwargs):
            options.append(kwargs)
            return template[:]
        codec = SimpleNamespace(apply_chat_template=apply, encode=lambda text, **kwargs: [20628, 2266])
        built = q.build_fixture_ids(codec, self.cfg)
        self.assertEqual(set(built), {128, 256, 384, 512})
        self.assertEqual(options, [{"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}])
        for length, ids in built.items():
            self.assertEqual(ids, self.cfg["frozen_fixture_ids"][str(length)]["ids"])
        codec.encode = lambda text, **kwargs: [20629, 2266]
        with self.assertRaisesRegex(RuntimeError, "tokenizer-built fixture differs"):
            q.build_fixture_ids(codec, self.cfg)

    def test_each_replay_input_excludes_current_target_and_includes_prior_generated_ids(self):
        prompt = self.cfg["frozen_fixture_ids"]["128"]["ids"][:]
        generated = [16, 22, 151645]
        for ordinal, token in enumerate(generated):
            target = q.prefix_for_target(prompt, generated, ordinal)
            self.assertEqual(target["ids"], prompt + generated[:ordinal])
            self.assertEqual(target["target_id"], token)
            self.assertEqual(target["input_positions"], 128 + ordinal)
            self.assertEqual(target["prefix_token_ids_sha256"], q.token_hash(prompt + generated[:ordinal]))
        self.assertEqual(generated, [16, 22, 151645])
        self.assertEqual(prompt, self.cfg["frozen_fixture_ids"]["128"]["ids"])
        self.assertNotEqual(q.token_hash(prompt + generated), q.prefix_for_target(prompt, generated, 2)["prefix_token_ids_sha256"])

    def test_both_eos_ids_and_cap_only_include_every_target(self):
        prompt = self.cfg["frozen_fixture_ids"]["384"]["ids"]
        for eos in q.EOS_IDS:
            generated = [7] * 127 + [eos]
            target = q.prefix_for_target(prompt, generated, 127)
            self.assertEqual((target["input_positions"], target["target_id"]), (511, eos))
            self.assertEqual(old.termination_after_emit(prompt, generated, 128, q.EOS_IDS), "eos")
        self.assertEqual(q.prefix_for_target(prompt, [7] * 128, 127)["target_id"], 7)
        self.assertEqual(old.termination_after_emit(prompt, [7] * 128, 128, q.EOS_IDS), "length")
        self.assertIsNone(old.termination_after_emit([151645], [7], 128, q.EOS_IDS))

    def test_invalid_partial_or_shifted_rollouts_are_refused(self):
        prompt = self.cfg["frozen_fixture_ids"]["128"]["ids"]
        for generated, ordinal in (([], 0), ([7], 0), ([151645, 7], 0), ([7] * 129, 0), ([151645], True), ([151645], 1)):
            with self.subTest(generated_length=len(generated), ordinal=ordinal), self.assertRaises(RuntimeError):
                q.prefix_for_target(prompt, generated, ordinal)
        with self.assertRaises(RuntimeError):
            q.prefix_for_target(prompt[:-1], [151645], 0)

    def test_actual_counts_match_direct_sum_including_eos_and_maximum(self):
        for first, second in ((1, 1), (3, 7), (128, 128)):
            positions = 768 + 2 * (sum(128 + ordinal for ordinal in range(first)) + sum(384 + ordinal for ordinal in range(second)))
            self.assertEqual(q.expected_counts(first, second), {"generated_tokens": first + second,
                "forward_calls": 2 + 2 * (first + second), "logical_calls": 314 + 2 * (first + second), "forward_input_positions": positions})
        self.assertEqual(q.expected_counts(128, 128)["forward_input_positions"], 164352)
        for count in (0, 129, True):
            with self.assertRaises(RuntimeError): q.expected_counts(count, 1)

    def test_exact_logit_hash_target_prefix_and_probability_are_joint_gates(self):
        online = self.witness()
        self.assertTrue(q.prefix_comparison(online, self.witness())["passed"])
        for changes in ({"raw_logits_sha256": "c" * 64}, {"prefix_token_ids_sha256": "c" * 64}, {"target_id": 22},
                        {"ordinal": 1}, {"logits_shape": [1, 2, 151936]}, {"logits_dtype": "float32"},
                        {"finite": False}, {"model_logprob": -.5002}, {"model_logprob": float("nan")}, {"model_logprob": .1}):
            with self.subTest(changes=changes):
                observed = q.prefix_comparison(online, self.witness(**changes))
                self.assertFalse(observed["passed"])
                q.canonical(q.safe_numbers(observed))
        for changes in ({"target_id": True}, {"ordinal": True}, {"model_logprob": True}, {"raw_logits_sha256": "bad"}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                q.prefix_comparison(online, self.witness(**changes))

    def test_mean_limit_does_not_hide_a_single_selected_probability_failure(self):
        measured = old.likelihood_measurement([-1.] * 3, [-1., -1., -1.0002], 1e-4, 1e-4)
        self.assertLess(measured["mean_abs"], 1e-4)
        self.assertGreater(measured["max_abs"], 1e-4)
        self.assertFalse(measured["passed"])

    def test_raw_greedy_tie_behavior_and_terminal_only_rendering(self):
        self.assertEqual(old.greedy_index([2., 4., 4.]), 1)
        self.assertEqual(old.validate_choice(1, 4., 4., 1), 0.)
        with self.assertRaises(RuntimeError): old.validate_choice(2, 4., 4., 1)
        codec = SimpleNamespace(decode=lambda ids, **kwargs: "".join({7: " \n", 8: "<other-special>", 151645: "<end>"}[token] for token in ids))
        rendered = old.render_response(codec, [7, 8, 151645], 128, q.EOS_IDS)
        self.assertEqual(rendered["raw_response"], " \n<other-special><end>")
        self.assertEqual(rendered["visible_response"], " \n<other-special>")

    def complete_accounting(self):
        counts = q.expected_counts(3, 3)
        ledger = CallAccounting(lambda *args, **kwargs: None, maximum_optimizer_calls=0)
        snapshot = ledger.snapshot()
        for operation, count in (("model_construct", 1), ("checkpoint_load", 1), ("tensor_copy", 310), ("forward", counts["forward_calls"])):
            snapshot["operations"][operation] = {"attempted": count, "returned": count, "failed": 0, "pending": 0}
        snapshot["last_call_id"] = counts["logical_calls"]
        snapshot["forward_input_positions_attempted"] = snapshot["forward_input_positions_returned"] = counts["forward_input_positions"]
        phases = q.new_phase_accounting()
        half = (counts["forward_input_positions"] - 768) // 2
        for phase, count, positions in (("finite", 2, 768), ("online", 6, half), ("replay", 6, half)):
            phases[phase] = {"operations": {"attempted": count, "returned": count, "failed": 0, "pending": 0},
                             "input_positions_attempted": positions, "input_positions_returned": positions}
        return snapshot, phases

    def test_complete_accounting_rejects_missing_extra_failed_pending_and_phase_dose(self):
        snapshot, phases = self.complete_accounting()
        self.assertEqual(q.verify_complete_accounting(snapshot, phases, 3, 3), q.expected_counts(3, 3))
        for mode in ("failed", "pending", "extra", "copy", "phase", "boolean"):
            snapshot, phases = self.complete_accounting()
            if mode == "failed": snapshot["operations"]["forward"]["failed"] = 1
            if mode == "pending": snapshot["pending_calls"] = [{"call_id": 1}]
            if mode == "extra": snapshot["operations"]["optimizer_step"]["attempted"] = 1
            if mode == "copy": snapshot["operations"]["tensor_copy"]["returned"] = 309
            if mode == "phase": phases["replay"]["input_positions_returned"] -= 1
            if mode == "boolean": snapshot["max_optimizer_calls"] = False
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                q.verify_complete_accounting(snapshot, phases, 3, 3)

    def test_failed_marker_io_preserves_original_physical_failure_and_partial_cost(self):
        original = RuntimeError("synthetic physical forward failed")
        phases, result = q.new_phase_accounting(), {}
        def append(path, record):
            if record["event"] == "operation_failed": raise OSError("failed marker unavailable")
        with patch.object(q, "append_event", side_effect=append):
            ledger = CallAccounting(q.event_sink("unused", 0, result, phases, clock=lambda: 1), maximum_optimizer_calls=0)
            def fail(): raise original
            with self.assertRaises(RuntimeError) as observed:
                ledger.call("forward", fail, stage="online_128_0", input_positions=128)
        self.assertIs(observed.exception, original)
        self.assertEqual(ledger.snapshot()["operations"]["forward"], {"attempted": 1, "returned": 0, "failed": 1, "pending": 0})
        self.assertEqual(phases["online"]["input_positions_attempted"], 128)
        self.assertEqual(phases["online"]["input_positions_returned"], 0)
        self.assertEqual(result["publication_errors"][0]["error_type"], "OSError")

    def test_attempt_and_return_marker_io_do_not_erase_phase_costs_or_execute_extra_work(self):
        for kind in ("operation_attempt", "operation_returned"):
            phases, result, calls = q.new_phase_accounting(), {}, []
            original = OSError("synthetic journal unavailable")
            def append(path, record):
                if record["event"] == kind: raise original
            with patch.object(q, "append_event", side_effect=append):
                ledger = CallAccounting(q.event_sink("unused", 0, result, phases, clock=lambda: 1), maximum_optimizer_calls=0)
                with self.assertRaises(OSError) as observed:
                    ledger.call("forward", lambda: calls.append(1), stage="replay_128_0", input_positions=128)
            self.assertIs(observed.exception, original)
            returned = int(kind == "operation_returned")
            self.assertEqual(calls, [1] if returned else [])
            self.assertEqual(ledger.snapshot()["operations"]["forward"], {"attempted": 1, "returned": returned, "failed": 0, "pending": 1 - returned})
            self.assertEqual(phases["replay"]["operations"], ledger.snapshot()["operations"]["forward"])
            self.assertEqual(phases["replay"]["input_positions_returned"], 128 * returned)

    def test_model_return_is_retained_when_later_logits_extraction_fails(self):
        original = RuntimeError("synthetic missing output logits")
        ledger = CallAccounting(lambda *args, **kwargs: None, maximum_optimizer_calls=0)
        options = []
        class Output:
            @property
            def logits(self):
                raise original
        def model(tensor, **kwargs):
            options.append(kwargs)
            return Output()
        with self.assertRaises(RuntimeError) as observed:
            q.single_prefix_forward(ledger, model, object(), stage="online_128_0", input_positions=128)
        self.assertIs(observed.exception, original)
        self.assertEqual(options, [{"use_cache": False, "logits_to_keep": 1}])
        self.assertEqual(ledger.snapshot()["operations"]["forward"], {"attempted": 1, "returned": 1, "failed": 0, "pending": 0})
        self.assertEqual(ledger.snapshot()["forward_input_positions_returned"], 128)

    def test_preworker_gate_requires_two_current_ram_gpu_and_disk_observations(self):
        clock = Clock()
        samples = iter([{"host_available_gib": 7., "gpu_free_gib": value, "disk_free_gib": 10.} for value in (4., 2.3, 4., 4.)])
        observed = []
        decision = q.preworker_wait(source_check=lambda: None, sample=lambda: next(samples), clock=clock.now, pause=clock.pause, record=observed.append)
        self.assertEqual((decision["status"], decision["observations"], clock.value), ("ADMITTED_PREWORKER_RESOURCE_ONLY", 4, 6.))
        self.assertIs(decision["E1_authorized"], False)
        for fields in ({"host_available_gib": 3.3}, {"gpu_free_gib": 3.99}, {"disk_free_gib": 3.99}):
            self.assertFalse(q.preworker_measurement({"host_available_gib": 7., "gpu_free_gib": 5., "disk_free_gib": 10., **fields})["admitted"])

    def test_slow_journal_source_and_low_gpu_never_count_as_admission(self):
        for mode in ("gpu", "source", "journal"):
            clock = Clock()
            def source():
                if mode == "source": clock.value += 31.
            def record(value):
                if mode == "journal" and value["observation"] == 2: clock.value = 31.
            result = q.preworker_wait(source_check=source, sample=lambda: {"host_available_gib": 7., "gpu_free_gib": 2.3 if mode == "gpu" else 4.,
                "disk_free_gib": 10.}, clock=clock.now, pause=clock.pause, record=record)
            self.assertEqual(result["status"], "TIMED_OUT_PREWORKER_NO_WORKER")
            self.assertEqual((result["model_calls"], result["framework_imports"]), (0, 0))

    def test_low_gpu_controller_never_imports_framework_or_launches_worker(self):
        clock = Clock()
        fake_psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=7 * 2**30), disk_usage=lambda path: SimpleNamespace(free=20 * 2**30))
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "worker"
            args = SimpleNamespace(config=q.ROOT / q.CONFIG_RELATIVE, output=None)
            with patch.object(q, "output_path", return_value=target), patch.object(q, "source_snapshot", return_value={}), \
                 patch.object(q, "validate_prerequisites"), patch.object(q, "recheck_sources"), patch.object(q, "gpu_free_gib", return_value=2.3), \
                 patch.object(q.time, "monotonic", side_effect=clock.now), patch.object(q.time, "sleep", side_effect=clock.pause), \
                 patch.dict(sys.modules, {"psutil": fake_psutil}):
                self.assertEqual(q.supervise(args, self.cfg), 2)
            receipt = json.loads((target.with_name("worker_controller") / "ADMISSION_RESULT.json").read_bytes())
            self.assertEqual(receipt["status"], "TIMED_OUT_PREWORKER_NO_WORKER")
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name("worker_supervisor").exists())
        self.assertFalse({"torch", "transformers", "safetensors", "tokenizers"}.intersection(sys.modules))

    def test_final_fresh_gpu_drop_refuses_dispatch_after_two_good_samples(self):
        clock = Clock()
        fake_psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=7 * 2**30), disk_usage=lambda path: SimpleNamespace(free=20 * 2**30))
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "worker"
            args = SimpleNamespace(config=q.ROOT / q.CONFIG_RELATIVE, output=None)
            gpu_samples = iter([4.5, 4.5, 4.5, 3.9])
            def check_output(command, **kwargs):
                if command[0] == "git": return temporary
                if command[0] == "nvidia-smi": return ""
                self.fail("unexpected external command")
            with patch.object(q, "output_path", return_value=target), patch.object(q, "source_snapshot", return_value={}), \
                 patch.object(q, "validate_prerequisites"), patch.object(q, "recheck_sources"), \
                 patch.object(q, "gpu_free_gib", side_effect=lambda: next(gpu_samples)), \
                 patch.object(q.subprocess, "check_output", side_effect=check_output), \
                 patch.object(q.time, "monotonic", side_effect=clock.now), patch.object(q.time, "sleep", side_effect=clock.pause), \
                 patch.dict(sys.modules, {"psutil": fake_psutil}), \
                 patch("experiments.opallo_native_collector_v1.supervisor.run_bounded") as dispatch, patch("builtins.print"):
                self.assertEqual(q.supervise(args, self.cfg), 2)
                dispatch.assert_not_called()
            receipt = json.loads((target.with_name("worker_controller") / "FAILURE_RECEIPT.json").read_bytes())
            self.assertIs(receipt["worker_launch_attempted"], False)
            self.assertIn("final prelaunch", receipt["error"])
            self.assertFalse(target.exists())

    def test_new_call_ceilings_refuse_overruns_before_physical_dispatch(self):
        for mode in ("logical", "forward", "positions"):
            ledger = CallAccounting(lambda *args, **kwargs: None, maximum_optimizer_calls=0)
            if mode == "logical": ledger.last_call_id = 826
            if mode == "forward": ledger.operations["forward"]["attempted"] = 514
            if mode == "positions": ledger.positions["forward_input_positions_attempted"] = 164352
            calls, observed = old.BudgetedCalls(ledger, self.cfg), []
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                calls.call("forward", lambda: observed.append(1), stage="online_128_0", input_positions=128)
            self.assertEqual(observed, [])

    def test_terminal_io_failure_preserves_original_error_and_final_partial_ledger(self):
        ledger = CallAccounting(lambda *args, **kwargs: None, maximum_optimizer_calls=0)
        ledger.begin("forward", stage="online_128_0", input_positions=128)
        result = {"status": q.FAIL_STATUS, "error_type": "RuntimeError", "error": "original model error", "resources": [],
            "wall_seconds": 1., "pretrained_loads": 0, "optimizer_steps": 0, "training_tokens": 0, "generation_tokens": 0,
            "teacher_calls": 0, "accounting_version": 2, "call_accounting": ledger.snapshot(), "phase_forward_accounting": q.new_phase_accounting()}
        published = {}
        def write(path, value):
            if path.name == "RESOURCE_BASELINE.json": raise OSError("secondary publication failure")
            published[path.name] = deepcopy(value)
        with patch.object(q, "write", side_effect=write):
            q.terminal_publish(Path("unused"), result)
        self.assertEqual(published["CAPTURE_RESULT.json"]["error"], "original model error")
        self.assertEqual(published["FAILURE_RECEIPT.json"]["error_type"], "RuntimeError")
        self.assertEqual(published["COST_LEDGER.json"]["call_accounting"]["operations"]["forward"]["pending"], 1)

    def test_scoped_cpu_receipt_and_source_omit_bulk_sampling_or_mutation(self):
        receipt = q.run_cpu_controls(self.cfg)
        self.assertEqual(receipt["model_calls"], 0)
        self.assertEqual(receipt["native_EOS_observations"], [])
        for field in ("DEV_collection_authorized", "E1_authorized", "CHECK_FINAL_access", "native_evaluation_bank_access", "goal_complete"):
            self.assertIs(self.cfg[field], False)
        source = Path(q.__file__).read_text()
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr in ("generate", "multinomial", "backward", "train", "step", "kill", "terminate") for node in calls))
        self.assertIn("model(tensor, use_cache=False, logits_to_keep=1)", source)
        self.assertNotIn("logits_to_keep=length + 1", source)
        self.assertLess(source.index("fixtures = build_fixture_ids(tokenizer, cfg)"), source.index('calls.call("model_construct"'))
        self.assertFalse({"torch", "transformers", "safetensors", "tokenizers"}.intersection(sys.modules))


if __name__ == "__main__":
    unittest.main()
