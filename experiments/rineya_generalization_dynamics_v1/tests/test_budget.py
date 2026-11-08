"""Budget mechanics and real PyTorch hooks; no native model or bank access."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rineya_generalization_dynamics_v1.budget import NativeBudget
from experiments.rineya_generalization_dynamics_v1.common import Blocked, read, signed, verify
from experiments.rineya_generalization_dynamics_v1.pipeline import DEFAULT_NATIVE, DEFAULT_POLICY, validate_config


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.clock = 100.0
        self.patch = patch("experiments.rineya_generalization_dynamics_v1.budget.time.monotonic",
                           side_effect=lambda: self.clock)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.manifest = signed({"runtime": {**DEFAULT_NATIVE, "max_native_wall_seconds": 10,
                                           "max_native_forward_calls": 4}})

    def budget(self, split="dev"):
        return NativeBudget(self.run, split, self.manifest)

    def finished(self, attempt=1, split="dev"):
        return verify(read(self.run / "native_budget" / split / f"{attempt:06d}.finished.json"))

    def test_aggregate_wall_expiry_is_recorded_and_cannot_reset(self):
        with self.assertRaisesRegex(Blocked, "wall budget"):
            with self.budget() as budget:
                self.clock += 10
                budget.check_time()
        receipt = self.finished()
        self.assertEqual(receipt["status"], "BLOCKED")
        self.assertEqual(receipt["total_wall_seconds"], 10)
        with self.assertRaisesRegex(Blocked, "wall budget"):
            self.budget()

    def test_aggregate_forwards_exhausted_before_an_extra_call(self):
        with self.assertRaisesRegex(Blocked, "forward budget"):
            with self.budget() as budget:
                for _ in range(5):
                    budget.before_forward()
        self.assertEqual(self.finished()["forward_calls"], 4)
        with self.assertRaisesRegex(Blocked, "forward budget"):
            self.budget()

    def test_resume_counts_prior_time_and_forwards(self):
        with self.budget() as first:
            first.before_forward()
            self.clock += 3
        first_bytes = (self.run / "native_budget/dev/000001.finished.json").read_bytes()
        self.clock += 1000  # Idle time between attempts is not native runtime.
        with self.budget() as second:
            second.before_forward()
            self.clock += 2
        receipt = self.finished(2)
        self.assertEqual(receipt["total_forward_calls"], 2)
        self.assertEqual(receipt["total_wall_seconds"], 5)
        self.assertEqual((self.run / "native_budget/dev/000001.finished.json").read_bytes(), first_bytes)

    def test_interrupted_started_receipt_blocks_resume(self):
        budget = self.budget()
        budget.__enter__()
        budget.before_forward()
        with self.assertRaisesRegex(Blocked, "Interrupted"):
            self.budget()
        self.assertFalse((self.run / "native_budget/dev/000001.finished.json").exists())

    def test_failure_receipt_preserved_and_resume_uses_remaining_budget(self):
        with self.assertRaises(RuntimeError):
            with self.budget() as first:
                first.before_forward()
                self.clock += 2
                raise RuntimeError("synthetic model failure")
        self.assertEqual(self.finished()["status"], "FAILED")
        self.assertIn("synthetic model failure", self.finished()["reason"])
        with self.budget() as second:
            self.assertEqual(second.prior_forward_calls, 1)
            self.assertEqual(second.prior_wall_seconds, 2)

    def test_split_budgets_are_independent(self):
        with self.budget() as dev:
            dev.before_forward()
            self.clock += 3
        with self.budget("confirm") as confirmation:
            self.assertEqual(confirmation.prior_forward_calls, 0)
            self.assertEqual(confirmation.prior_wall_seconds, 0)
        # This creates accounting only, never a confirmation suite or predictions.
        self.assertFalse((self.run / "confirmation_suite.json").exists())

    def test_binding_changes_are_rejected(self):
        with self.budget():
            pass
        changed = signed({"runtime": {**self.manifest["runtime"], "max_native_forward_calls": 100}})
        with self.assertRaisesRegex(Blocked, "binding"):
            NativeBudget(self.run, "dev", changed)

    def test_receipt_tampering_blocks_resume(self):
        with self.budget():
            pass
        path = self.run / "native_budget/dev/000001.finished.json"
        value = read(path)
        value["forward_calls"] = 0 if value["forward_calls"] else 1
        import json
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(Blocked, "digest"):
            self.budget()

    def test_budget_config_requires_positive_finite_limits(self):
        for key, invalid in (("max_native_wall_seconds", 0), ("max_native_wall_seconds", float("inf")),
                             ("max_native_wall_seconds", True), ("max_native_forward_calls", 0),
                             ("max_native_forward_calls", 1.5), ("max_native_forward_calls", True)):
            with self.subTest(key=key, invalid=invalid), self.assertRaises(Blocked):
                validate_config(DEFAULT_POLICY, {**DEFAULT_NATIVE, key: invalid})

    def test_completed_inspection_allows_exact_forward_limit(self):
        with self.budget() as budget:
            for _ in range(4):
                budget.before_forward()
        receipt = NativeBudget.inspect_complete(self.run, "dev", self.manifest)
        self.assertEqual(receipt["total_forward_calls"], 4)
        self.assertEqual(len(list((self.run / "native_budget/dev").iterdir())), 2)

    def test_completed_inspection_rejects_no_ledger_or_failed_attempt(self):
        with self.assertRaisesRegex(Blocked, "No completed"):
            NativeBudget.inspect_complete(self.run, "dev", self.manifest)
        with self.assertRaises(RuntimeError):
            with self.budget():
                raise RuntimeError("synthetic failure")
        with self.assertRaisesRegex(Blocked, "did not complete"):
            NativeBudget.inspect_complete(self.run, "dev", self.manifest)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional real torch budget-hook tests")
class TorchBudgetTests(unittest.TestCase):
    setUp = BudgetTests.setUp
    budget = BudgetTests.budget
    finished = BudgetTests.finished
    def tiny(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actual_calls = 0
            def forward(self, input_ids=None, **kwargs):
                self.actual_calls += 1
                return input_ids
            def generate(self, input_ids=None, **kwargs):
                self(input_ids=input_ids)
                self(input_ids=input_ids)
                return input_ids
        return Tiny()

    def test_real_hook_counts_generation_and_likelihood_across_checkpoints(self):
        with self.budget() as budget:
            first = self.tiny()
            hook = first.register_forward_pre_hook(budget.before_forward)
            first.generate(input_ids="stub")
            first(input_ids="stub")
            hook.remove()  # Same accounting object survives checkpoint unload.
            second = self.tiny()
            hook = second.register_forward_pre_hook(budget.before_forward)
            second(input_ids="stub")
            with self.assertRaisesRegex(Blocked, "forward budget"):
                second(input_ids="stub")
            hook.remove()
            self.assertEqual(first.actual_calls, 3)
            self.assertEqual(second.actual_calls, 1)
        self.assertEqual(self.finished()["forward_calls"], 4)

    def test_peft_style_generate_delegation_counts_underlying_model(self):
        # Actual PEFT/HF compatibility remains unqualified; this verifies delegation.
        base = self.tiny()
        class Wrapper:
            def get_base_model(self):
                return base
            def generate(self, **kwargs):
                return base.generate(**kwargs)
            def __call__(self, **kwargs):
                return base(**kwargs)
        wrapper = Wrapper()
        with self.budget() as budget:
            hook = wrapper.get_base_model().register_forward_pre_hook(budget.before_forward)
            wrapper.generate(input_ids="stub")
            wrapper(input_ids="stub")
            hook.remove()
        self.assertEqual(self.finished()["forward_calls"], 3)

    def test_wall_expiry_prevents_starting_real_forward(self):
        model = self.tiny()
        with self.assertRaisesRegex(Blocked, "wall budget"):
            with self.budget() as budget:
                hook = model.register_forward_pre_hook(budget.before_forward)
                try:
                    self.clock += 10
                    model(input_ids="stub")
                finally:
                    hook.remove()
        self.assertEqual(model.actual_calls, 0)
        self.assertEqual(self.finished()["forward_calls"], 0)

    def pipeline_fixture(self, limit=100):
        from experiments.rineya_generalization_dynamics_v1.common import code_hash, digest, write_new
        from experiments.rineya_generalization_dynamics_v1.diagnostics import build_suite, suite_hash
        checkpoints = [{"id": name, "step": step, "kind": "hf", "asset": {"content_sha256": digest(name)}}
                       for name, step in (("reference", 0), ("middle", 100), ("final", 200))]
        rows = build_suite("dev", 17, 2)
        self.manifest = signed({"runtime": {**DEFAULT_NATIVE, "max_native_wall_seconds": 10,
                                           "max_native_forward_calls": limit},
                               "registry": {"checkpoints": checkpoints}, "evidence_kind": "native_pilot",
                               "schema": 1, "experiment": "GD-001", "code_sha256": code_hash(),
                               "policy": DEFAULT_POLICY, "dev_suite_sha256": suite_hash(rows)})
        write_new(self.run / "manifest.json", self.manifest)
        write_new(self.run / "development_suite.json", rows)
        owner = self
        class StubEvaluator:
            fail_once = False
            loads = []
            live = 0
            maximum_live = 0
            close_delay = 0
            def __init__(self, registry, config, budget):
                self.budget = budget
                self.hook = None
            def load(self, checkpoint):
                self.budget.check_time()
                self.checkpoint = checkpoint
                self.model = owner.tiny()
                self.hook = self.model.register_forward_pre_hook(self.budget.before_forward)
                self.loads.append(checkpoint["id"])
                type(self).live += 1
                type(self).maximum_live = max(type(self).maximum_live, type(self).live)
                return {"backend": "tensor-stub-NOT-native-compatibility"}
            def predict(self, row):
                from experiments.rineya_generalization_dynamics_v1.pipeline import fixture_prediction
                self.model(input_ids="stub")
                if self.checkpoint["id"] == "middle" and type(self).fail_once:
                    type(self).fail_once = False
                    raise RuntimeError("synthetic failure after one checkpoint")
                return fixture_prediction(row, self.checkpoint)
            def resources(self):
                return {"measured_native_resources": False}
            def close(self):
                if self.hook is not None:
                    self.hook.remove()
                    self.hook = None
                    type(self).live -= 1
                    owner.clock += type(self).close_delay
                self.model = None
        return checkpoints, rows, StubEvaluator

    def evaluate_stub(self, checkpoints, rows, stub, *, final_failure=False, verification_effect=None):
        from contextlib import nullcontext
        from experiments.rineya_generalization_dynamics_v1 import native, pipeline
        if final_failure:
            verification_effect = [None] * (len(checkpoints) + 1) + [Blocked("Registered asset changed")]
        with patch.object(native, "NativeEvaluator", stub), patch.object(native, "device_lock", return_value=nullcontext()), \
             patch.object(pipeline, "verify_registry", side_effect=verification_effect):
            return pipeline.evaluate(self.run, self.manifest, rows, "dev", checkpoints, True)

    def test_pipeline_resume_reuses_receipt_and_counts_failed_work(self):
        checkpoints, rows, stub = self.pipeline_fixture()
        stub.fail_once = True
        with self.assertRaises(RuntimeError):
            self.evaluate_stub(checkpoints, rows, stub)
        first_receipt = (self.run / "dev/reference.json").read_bytes()
        self.assertEqual(self.finished()["forward_calls"], 25)
        results = self.evaluate_stub(checkpoints, rows, stub)
        self.assertEqual(self.finished(2)["total_forward_calls"], 73)
        self.assertEqual(len(results), 3)
        self.assertEqual((self.run / "dev/reference.json").read_bytes(), first_receipt)
        self.assertEqual(stub.loads, ["reference", "middle", "middle", "final"])
        self.assertEqual(stub.maximum_live, 1)
        self.assertEqual(stub.live, 0)
        self.assertFalse(any("confirm" in p.name for p in self.run.iterdir()))

    def test_pipeline_forward_limit_spans_checkpoints_and_blocks_reset(self):
        checkpoints, rows, stub = self.pipeline_fixture(limit=25)
        with self.assertRaisesRegex(Blocked, "forward budget"):
            self.evaluate_stub(checkpoints, rows, stub)
        self.assertEqual(self.finished()["total_forward_calls"], 25)
        self.assertTrue((self.run / "dev/reference.json").exists())
        self.assertFalse((self.run / "dev/middle.json").exists())
        with self.assertRaisesRegex(Blocked, "forward budget"):
            self.evaluate_stub(checkpoints, rows, stub)
        self.assertEqual(stub.loads, ["reference", "middle"])
        self.assertEqual(stub.maximum_live, 1)

    def test_unload_deadline_prevents_publishing_checkpoint_receipt(self):
        checkpoints, rows, stub = self.pipeline_fixture()
        stub.close_delay = 10
        with self.assertRaisesRegex(Blocked, "wall budget"):
            self.evaluate_stub(checkpoints, rows, stub)
        self.assertFalse((self.run / "dev/reference.json").exists())
        self.assertEqual(self.finished()["status"], "BLOCKED")
        self.assertEqual(stub.live, 0)

    def test_failed_final_sweep_cannot_select_existing_checkpoint_receipts(self):
        from experiments.rineya_generalization_dynamics_v1.pipeline import select
        checkpoints, rows, stub = self.pipeline_fixture()
        with self.assertRaisesRegex(Blocked, "Registered asset changed"):
            self.evaluate_stub(checkpoints, rows, stub, final_failure=True)
        self.assertTrue(all((self.run / "dev" / (c["id"] + ".json")).exists() for c in checkpoints))
        old_results = {c["id"]: (self.run / "dev" / (c["id"] + ".json")).read_bytes() for c in checkpoints}
        invalidation_path = self.run / "native_registry_invalidations/dev/000001.json"
        invalidation = verify(read(invalidation_path))
        self.assertEqual(invalidation["stage"], "final-sweep")
        self.assertEqual(invalidation["started_sha256"], self.finished()["started_sha256"])
        with self.assertRaisesRegex(Blocked, "permanently invalidated"):
            select(self.run)
        self.assertFalse((self.run / "selection.json").exists())
        # The mock verification now passes, simulating restoration of original
        # assets. That cannot establish which weights the old attempt scored.
        with self.assertRaisesRegex(Blocked, "permanently invalidated"):
            self.evaluate_stub(checkpoints, rows, stub)
        self.assertFalse((self.run / "native_budget/dev/000002.STARTED.json").exists())
        self.assertEqual(old_results, {c["id"]: (self.run / "dev" / (c["id"] + ".json")).read_bytes() for c in checkpoints})

    def test_interrupted_after_final_result_cannot_select(self):
        from experiments.rineya_generalization_dynamics_v1.pipeline import select
        checkpoints, rows, stub = self.pipeline_fixture()
        self.evaluate_stub(checkpoints, rows, stub)
        # Simulate an owned worker dying before closing its accounting receipt.
        (self.run / "native_budget/dev/000001.finished.json").unlink()
        with self.assertRaisesRegex(Blocked, "Interrupted"):
            select(self.run)
        self.assertFalse((self.run / "selection.json").exists())

    def test_select_accepts_successfully_exhausted_forward_budget(self):
        from experiments.rineya_generalization_dynamics_v1.pipeline import select
        checkpoints, rows, stub = self.pipeline_fixture(limit=72)
        self.evaluate_stub(checkpoints, rows, stub)
        selection = select(self.run)
        self.assertEqual(selection["native_budget_receipt_sha256"], self.finished()["sha256"])

    def test_completed_sweep_replay_at_exact_cap_spends_no_budget(self):
        checkpoints, rows, stub = self.pipeline_fixture(limit=72)
        first = self.evaluate_stub(checkpoints, rows, stub)
        accounting = {p.name: p.read_bytes() for p in (self.run / "native_budget/dev").iterdir()}
        results = {c["id"]: (self.run / "dev" / (c["id"] + ".json")).read_bytes() for c in checkpoints}
        second = self.evaluate_stub(checkpoints, rows, stub)
        self.assertEqual(first, second)
        self.assertEqual(stub.loads, ["reference", "middle", "final"])
        self.assertEqual(accounting, {p.name: p.read_bytes() for p in (self.run / "native_budget/dev").iterdir()})
        self.assertEqual(results, {c["id"]: (self.run / "dev" / (c["id"] + ".json")).read_bytes() for c in checkpoints})

    def test_completed_sweep_replay_near_deadline_preserves_complete_receipt(self):
        checkpoints, rows, stub = self.pipeline_fixture()
        stub.close_delay = 3.3
        self.evaluate_stub(checkpoints, rows, stub)
        accounting = {p.name: p.read_bytes() for p in (self.run / "native_budget/dev").iterdir()}
        def slow_read_only_verification(registry):
            self.clock += 2
        self.evaluate_stub(checkpoints, rows, stub, verification_effect=slow_read_only_verification)
        self.assertEqual(accounting, {p.name: p.read_bytes() for p in (self.run / "native_budget/dev").iterdir()})
        self.assertEqual(self.finished()["status"], "COMPLETE")
        self.assertLess(self.finished()["total_wall_seconds"], 10)
        self.assertEqual(stub.loads, ["reference", "middle", "final"])

    def test_checkpoint_registry_failure_prevents_receipt_and_resume(self):
        checkpoints, rows, stub = self.pipeline_fixture()
        with self.assertRaisesRegex(Blocked, "Registered asset changed"):
            self.evaluate_stub(checkpoints, rows, stub,
                               verification_effect=[None, Blocked("Registered asset changed")])
        self.assertFalse((self.run / "dev/reference.json").exists())
        invalidation = verify(read(self.run / "native_registry_invalidations/dev/000001.json"))
        self.assertEqual(invalidation["stage"], "checkpoint:reference")
        with self.assertRaisesRegex(Blocked, "permanently invalidated"):
            self.evaluate_stub(checkpoints, rows, stub)
        self.assertEqual(stub.loads, ["reference"])

    def test_completed_replay_registry_failure_permanently_bars_selection(self):
        from experiments.rineya_generalization_dynamics_v1.pipeline import select
        checkpoints, rows, stub = self.pipeline_fixture()
        self.evaluate_stub(checkpoints, rows, stub)
        accounting = {p.name: p.read_bytes() for p in (self.run / "native_budget/dev").iterdir()}
        with self.assertRaisesRegex(Blocked, "Registered asset changed"):
            self.evaluate_stub(checkpoints, rows, stub,
                               verification_effect=Blocked("Registered asset changed"))
        self.assertEqual(accounting, {p.name: p.read_bytes() for p in (self.run / "native_budget/dev").iterdir()})
        invalidation = verify(read(self.run / "native_registry_invalidations/dev/000001.json"))
        self.assertEqual(invalidation["stage"], "completed-sweep-reuse")
        with self.assertRaisesRegex(Blocked, "permanently invalidated"):
            select(self.run)


if __name__ == "__main__":
    unittest.main()
