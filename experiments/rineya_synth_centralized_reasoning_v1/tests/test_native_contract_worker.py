"""Framework-free contract and capture-boundary controls; no native launch."""
import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import native_contract as contract
from native_worker import Journal, capture


class Scalar:
    def __init__(self, value): self.value = value
    def item(self): return self.value


class Row:
    def __init__(self, position, length, drift=0):
        self.token = 1 if position == 2 else 151645
        self.prob = -0.5 - (drift if length > 3 and position == 2 else 0)
    def argmax(self): return Scalar(self.token)
    def __getitem__(self, token): return Scalar(self.prob)


class Output:
    dtype = "float32"
    def __init__(self, length, drift): self.length, self.drift = length, drift
    def __getitem__(self, key):
        position = self.length - 1 if key[1] == -1 else key[1]
        return Row(position, self.length, self.drift)


class FakeTorch:
    float32 = "float32"
    long = "long"
    no_grad = staticmethod(contextlib.nullcontext)
    tensor = staticmethod(lambda values, **kwargs: values)
    ones_like = staticmethod(lambda ids: [[1] * len(ids[0])])
    isfinite = staticmethod(lambda output: SimpleNamespace(all=lambda: True))
    log_softmax = staticmethod(lambda row, **kwargs: row)


class ContractTests(unittest.TestCase):
    def test_guard_constants_preserve_proposed_thresholds(self):
        proposal = contract.read_json(PACKAGE / "NATIVE_QUALIFICATION_PROPOSAL.json")
        for key in contract.TOLERANCES:
            self.assertEqual(proposal["fixed_tolerances"][key], contract.TOLERANCES[key])
        for key in proposal["resources"]:
            if key in contract.RESOURCE_BOUNDS:
                self.assertEqual(proposal["resources"][key], contract.RESOURCE_BOUNDS[key])
        self.assertIs(False, contract.AUTHORIZATION["production_training"])
        self.assertIs(False, contract.AUTHORIZATION["protected_bank_access"])

    def test_json_duplicate_nonfinite_and_size_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file.json"
            for value in ('{"x":1,"x":2}', '{"x":NaN}', '[] trailing', 'x'*33):
                path.write_text(value, encoding="utf-8")
                with self.subTest(value=value), self.assertRaises(ValueError):
                    contract.read_json(path, 32)

    def test_preflight_rejects_external_config_before_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scope.json"
            path.write_text("{}")
            with self.assertRaises(ValueError): contract.preflight(path)

    def test_runtime_reader_does_not_import_frameworks(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertEqual("2.6.0+cu124", contract.RUNTIME["torch"])


class JournalTests(unittest.TestCase):
    def test_returns_and_failures_preserve_attempts_and_pending_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory))
            self.assertEqual(7, journal.call("forward", lambda: 7))
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                journal.call("forward", lambda: (_ for _ in ()).throw(RuntimeError("fixture")))
            snapshot = journal.snapshot()["operations"]["forward"]
            self.assertEqual({"attempted": 2, "returned": 1, "failed": 1, "pending": 0}, snapshot)
            journal.stream.close()
            events = contract.read_jsonl(Path(directory) / "events.jsonl")
            self.assertEqual(["attempt", "returned", "attempt", "failed"], [e["event"] for e in events])

    def test_failed_pre_guard_performs_no_function_or_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory))
            journal.guard = lambda stage: (_ for _ in ()).throw(ValueError("guard"))
            called = []
            with self.assertRaises(ValueError): journal.call("forward", lambda: called.append(True))
            self.assertEqual([], called)
            self.assertEqual({}, journal.snapshot()["operations"])
            journal.stream.close()


class CaptureTests(unittest.TestCase):
    def fixture(self, drift=0):
        operations, inputs = [], []
        def model(**kwargs):
            inputs.append(list(kwargs["input_ids"][0]))
            return SimpleNamespace(logits=Output(len(inputs[-1]), drift))
        def call(name, function):
            operations.append(name)
            return function()
        config = {"dose": {"capture_max_generated_tokens": 32}, "tolerances": contract.TOLERANCES}
        return model, call, config, operations, inputs

    def test_greedy_repeats_use_only_generated_tokens_and_verify_bulk(self):
        model, call, config, operations, inputs = self.fixture()
        result = capture(FakeTorch, model, [7, 8, 9], call, lambda stage: None, config)
        self.assertEqual([7, 8, 9, 1], inputs[1])
        self.assertEqual([7, 8, 9], inputs[2])
        self.assertEqual(4, operations.count("capture_forward"))
        self.assertEqual(1, operations.count("capture_bulk_forward"))
        self.assertEqual(0, result["selected_logprob_max_error"])
        self.assertFalse(result["capability_correctness_scored"])

    def test_bulk_likelihood_failure_cannot_be_marked_qualified(self):
        model, call, config, _, _ = self.fixture(drift=.1)
        evidence = {}
        with self.assertRaisesRegex(ValueError, "numerical gate"):
            capture(FakeTorch, model, [7, 8, 9], call, lambda stage: None, config, evidence)
        self.assertEqual(2, len(evidence["traces"]))
        self.assertAlmostEqual(.1, evidence["selected_logprob_max_error"])
        self.assertGreater(evidence["selected_logprob_mean_error"], .01)

    def test_whole_window_exhaustion_stops_without_truncating(self):
        model, call, config, operations, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "window exhausted"):
            capture(FakeTorch, model, [1]*513, call, lambda stage: None, config)
        self.assertEqual([], operations)


if __name__ == "__main__": unittest.main()
