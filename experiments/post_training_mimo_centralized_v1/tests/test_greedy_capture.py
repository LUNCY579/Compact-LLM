"""Synthetic short-capture controls. No framework imports, native jobs or weights."""
from copy import deepcopy
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.post_training_mimo_centralized_v1.scripts import greedy_capture as control


class FakeCodec:
    def __init__(self): self.calls = []
    def decode(self, ids, **kwargs):
        self.calls.append((ids[:], kwargs))
        return "".join({7: " \n", 8: "<other-special>", 151645: "<end>", 151643: "<textend>"}.get(token, str(token)) for token in ids)


class FakeAccounting:
    def __init__(self):
        self.state = {"last_call_id": 0, "forward_input_positions_attempted": 0,
                      "operations": {"forward": {"attempted": 0}}}
    def snapshot(self): return deepcopy(self.state)
    def call(self, operation, function, **kwargs):
        self.state["last_call_id"] += 1
        if operation == "forward":
            self.state["operations"]["forward"]["attempted"] += 1
            self.state["forward_input_positions_attempted"] += kwargs["input_positions"]
        return function()


class GreedyCaptureControls(unittest.TestCase):
    def setUp(self): self.cfg = control.registered_config()

    def test_registered_config_exactbytes_and_no_user_override(self):
        self.assertEqual(control.file_identity(control.PACKAGE / control.CONFIG_RELATIVE)["sha256"], control.CONFIG_SHA256)
        control.validate_config(self.cfg)
        for field, value in (("maximum_optimizer_calls", 1), ("worker_rss_limit_gib", 3.0), ("gpu_worker_free_gib", .1),
                             ("native_evaluation_bank_access", True), ("E1_authorized", True)):
            changed = deepcopy(self.cfg); changed[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError): control.validate_config(changed)
        for field, value in (("do_sample", True), ("top_k", 50), ("skip_special_tokens", True),
                             ("clean_up_tokenization_spaces", True), ("logits_processors", ["filter"]), ("use_cache", True)):
            changed = deepcopy(self.cfg); changed["decoding"][field] = value
            with self.subTest(decoding=field), self.assertRaises(RuntimeError): control.validate_config(changed)

    def test_raw_finite_argmax_keeps_lowest_tie_id(self):
        self.assertEqual(control.greedy_index([-2, 4, 4]), 1)
        self.assertEqual(control.greedy_index([0, 0, 0]), 0)
        for scores in ([], [1, float("nan")], [float("inf"), 1], [True, 1]):
            with self.subTest(scores=scores), self.assertRaises(RuntimeError): control.greedy_index(scores)

    def test_behavior_zero_requires_verified_raw_argmax_and_is_not_model_NLL(self):
        self.assertEqual(control.validate_choice(1, 4, 4, 1), 0.0)
        self.assertNotEqual(control.validate_choice(1, 4, 4, 1), -.5)
        for args in ((2, 4, 4, 1), (1, 3, 4, 1), (1, float("nan"), 4, 1), (True, 4, 4, 1)):
            with self.subTest(args=args), self.assertRaises(RuntimeError): control.validate_choice(*args)

    def test_only_new_emitted_EOS_stops_and_both_EOS_ids_work(self):
        for eos in (151645, 151643):
            self.assertIsNone(control.termination_after_emit([eos], [7], 128, [151645, 151643]))
            self.assertEqual(control.termination_after_emit([7], [eos], 128, [151645, 151643]), "eos")
            self.assertEqual(control.termination_after_emit([7], [7] * 127 + [eos], 128, [151645, 151643]), "eos")
        self.assertEqual(control.termination_after_emit([151645], [7] * 128, 128, [151645, 151643]), "length")
        with self.assertRaises(RuntimeError): control.termination_after_emit([7], [151645, 7], 128, [151645, 151643])
        with self.assertRaises(RuntimeError): control.termination_after_emit([7], [7] * 129, 128, [151645, 151643])

    def test_terminal_only_removal_preserves_whitespace_and_other_specials(self):
        for eos in (151645, 151643):
            codec = FakeCodec(); result = control.render_response(codec, [7, 8, eos], 128, [151645, 151643])
            self.assertEqual(result["visible_ids"], [7, 8])
            self.assertEqual(result["visible_response"], " \n<other-special>")
            self.assertEqual(result["raw_response_sha256"], hashlib.sha256(result["raw_response"].encode()).hexdigest())
            self.assertEqual(codec.calls[0][1], {"skip_special_tokens": False, "clean_up_tokenization_spaces": False})
            self.assertEqual(codec.calls[1][1], codec.calls[0][1])
        result = control.render_response(FakeCodec(), [7, 8], 2, [151645, 151643])
        self.assertEqual(result["raw_response"], result["visible_response"])
        self.assertEqual(result["visible_ids"], [7, 8])
        with self.assertRaises(RuntimeError): control.render_response(FakeCodec(), [7], 128, [151645, 151643])

    def test_offline_keep_Nplus1_firstN_is_exact_causal_shift(self):
        self.assertEqual(control.offline_indices(1, 2), [0])
        self.assertEqual(control.offline_indices(128, 129), list(range(128)))
        for args in ((2, 2), (2, 4), (0, 1), (129, 130), (True, 2)):
            with self.subTest(args=args), self.assertRaises(RuntimeError): control.offline_indices(*args)

    def test_numeric_diagnostics_retain_threshold_failures_and_nonfinite_flags(self):
        result = control.likelihood_measurement([-1.0, -2.0], [-1.001, -2.002])
        self.assertTrue(result["passed"])
        self.assertEqual(result["absolute_errors"], [abs(-1.0 + 1.001), abs(-2.0 + 2.002)])
        failed = control.likelihood_measurement([-1.0], [-1.04])
        self.assertFalse(failed["passed"]); self.assertGreater(failed["max_abs"], .03125)
        failed = control.likelihood_measurement([-1.0], [float("nan")])
        self.assertFalse(failed["finite"]); self.assertIsNone(failed["absolute_errors"])
        self.assertEqual(control.safe_numbers({"values": [float("nan"), float("inf"), -1.0]}), {"values": [None, None, -1.0]})
        with self.assertRaises(RuntimeError): control.likelihood_measurement([-1.0], [])
        with self.assertRaises(RuntimeError): control.likelihood_measurement([True], [-1.0])

    def test_actual_counts_include_EOS_and_maximum_envelopes(self):
        counts = control.expected_counts(128, 128)
        self.assertEqual(counts, {"generated_tokens": 256, "forward_calls": 260, "logical_calls": 572,
            "forward_input_positions": 83328, "resource_samples": 263, "journal_records": 1669})
        shortest = control.expected_counts(1, 1)
        self.assertEqual(shortest["forward_input_positions"], 1794)
        self.assertEqual(shortest["logical_calls"], 318)
        for first in (0, 129, True):
            with self.assertRaises(RuntimeError): control.expected_counts(first, 1)

    def test_source_prefix_hash_is_content_sensitive_and_retains_EOS(self):
        self.assertNotEqual(control.token_hash([10, 20]), control.token_hash([10, 21]))
        self.assertNotEqual(control.token_hash([10, 151645]), control.token_hash([10]))
        for ids in ([True], [151936], [-1]):
            with self.assertRaises(RuntimeError): control.token_hash(ids)

    def test_call_budget_stops_before_dispatched_mutations_or_overruns(self):
        accounting = FakeAccounting(); calls = control.BudgetedCalls(accounting, self.cfg); observed = []
        for operation in ("optimizer_step", "backward", "checkpoint_save", "checkpoint_reload"):
            with self.subTest(operation=operation), self.assertRaises(RuntimeError):
                calls.call(operation, lambda: observed.append("bad"), stage="blocked")
        with self.assertRaises(RuntimeError): calls.call("forward", lambda: observed.append("bad"), stage="blocked", input_positions=1, training=True)
        accounting.state["last_call_id"] = 572
        with self.assertRaises(RuntimeError): calls.call("forward", lambda: observed.append("bad"), stage="blocked", input_positions=1)
        accounting.state["last_call_id"] = 0; accounting.state["operations"]["forward"]["attempted"] = 260
        with self.assertRaises(RuntimeError): calls.call("forward", lambda: observed.append("bad"), stage="blocked", input_positions=1)
        accounting.state["operations"]["forward"]["attempted"] = 0; accounting.state["forward_input_positions_attempted"] = 83328
        with self.assertRaises(RuntimeError): calls.call("forward", lambda: observed.append("bad"), stage="blocked", input_positions=1)
        self.assertEqual(observed, [])

    def test_computed_CPU_controls_are_synthetic_source_bound_cases(self):
        report = control.run_cpu_controls()
        self.assertEqual(report["status"], "PASS_CPU_CAPTURE_MECHANICS_ONLY")
        self.assertEqual(len(report["cases"]), 8)
        self.assertEqual(report["native_EOS_observations"], [])
        for case in report["cases"]:
            self.assertTrue(case["passed"])
            self.assertEqual(case["output"], case["expected"])
            self.assertEqual(case["input_sha256"], hashlib.sha256(control.canonical(case["input"])).hexdigest())
        self.assertFalse(report["E1_authorized"])

    def test_capture_reserve_is_distinct_and_dynamic(self):
        self.assertEqual(control.admission_required(self.cfg, 0), 3.25)
        self.assertEqual(control.admission_required(self.cfg, 1), 2.25)
        self.assertEqual(control.admission_required(self.cfg, 2), 1.25)
        for rss in (-1, True, float("nan")):
            with self.assertRaises(RuntimeError): control.admission_required(self.cfg, rss)

    def test_native_worker_source_omits_sampling_and_keeps_offline_exception(self):
        path = control.PACKAGE / "scripts/qualify_capture.py"
        tree = ast.parse(path.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr in ("generate", "multinomial", "backward", "train", "step") for node in calls))
        source = path.read_text()
        self.assertIn('model(tensor, use_cache=False, logits_to_keep=keep)', source)
        self.assertIn('forward(full, f"teacher_force_{prompt_length}", length + 1)', source)
        self.assertIn('logits[0, :length].float().log_softmax(-1)', source)
        self.assertIn('control.require(bool(terminal), "native natural EOS coverage missing")', source)
        self.assertIn('record["generation_tokens_per_second"]', source)
        self.assertIn('record["offline_teacher_seconds"]', source)

    def test_source_recheck_detects_HEAD_and_dependency_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); file = root / "source.py"; file.write_bytes(b"original")
            snap = {"code_sha": "a" * 40, "source_files": {"source.py": control.file_identity(file)}, "installed_runtime_source_files": {}}
            with patch.object(control.subprocess, "check_output", return_value="b" * 40), self.assertRaises(RuntimeError): control.recheck_sources(root, snap)
            file.write_bytes(b"changed")
            with patch.object(control.subprocess, "check_output", return_value="a" * 40), self.assertRaises(RuntimeError): control.recheck_sources(root, snap)

    def test_failure_receipt_retains_returned_and_failed_call_accounting(self):
        from experiments.post_training_mimo_centralized_v1.scripts import qualify_capture, qualify_native
        constructor = qualify_native.CallAccounting; journals = []
        def make_accounting(*args, **kwargs):
            journal = constructor(*args, **kwargs); journals.append(journal)
            return journal
        def fail_cpu_controls():
            journal = journals[0]
            journal.call("forward", lambda: None, stage="synthetic_returned", input_positions=7)
            def fail(): raise RuntimeError("synthetic CPU call failed")
            journal.call("forward", fail, stage="synthetic_failed", input_positions=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); go = root / "GO.json"; output = root / "output"
            go.write_bytes(control.canonical({"config_sha256": control.CONFIG_SHA256}))
            args = SimpleNamespace(config=control.PACKAGE / control.CONFIG_RELATIVE, go_path=go, output=output)
            with patch.object(control, "source_snapshot", return_value={"code_sha": "a" * 40, "dirty": False}), \
                 patch.object(qualify_native, "CallAccounting", side_effect=make_accounting), \
                 patch.object(control, "run_cpu_controls", side_effect=fail_cpu_controls):
                self.assertEqual(qualify_capture.run(args), 2)
            failure = control.parse((output / "FAILURE_RECEIPT.json").read_bytes())
            result = control.parse((output / "CAPTURE_RESULT.json").read_bytes())
            ledger = control.parse((output / "COST_LEDGER.json").read_bytes())
            expected = journals[0].snapshot()
            self.assertEqual(result["status"], "BLOCKED_CAPTURE_QUALIFICATION")
            self.assertEqual(result["error"], "synthetic CPU call failed")
            for receipt in (failure, result, ledger):
                self.assertEqual(receipt["call_accounting"], expected)
            self.assertEqual(expected["operations"]["forward"], {"attempted": 2, "returned": 1, "failed": 1, "pending": 0})
            self.assertEqual(expected["last_call_id"], 2)
            self.assertEqual(expected["forward_input_positions_attempted"], 10)
            self.assertEqual(expected["forward_input_positions_returned"], 7)
            events = [json.loads(line) for line in (output / "EVENTS.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(qualify_native.verify_call_journal(events, expected, maximum_optimizer_calls=0), expected)
            self.assertFalse(any(record["event"] == "runtime_import_start" for record in events))

    def test_all_new_modules_import_without_model_frameworks(self):
        code = "import sys; import experiments.post_training_mimo_centralized_v1.scripts.greedy_capture; import experiments.post_training_mimo_centralized_v1.scripts.qualify_capture; import experiments.post_training_mimo_centralized_v1.scripts.supervise_capture; assert not any(n == 'torch' or n.startswith('torch.') or n == 'transformers' or n.startswith('transformers.') for n in sys.modules)"
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == "__main__": unittest.main()
