"""Real scalar quadratic/Adam/RNG CPU controls; no framework/model/GPU."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import study_engine as engine
import study_schedule as schedule


class QuadraticAdam:
    execution_scope = "CPU_COMPONENT_CONTROL"

    def __init__(self, directory):
        self.directory = Path(directory)
        self.rng = random.Random(101)
        self.weight = self.rng.random()
        self.gradient = self.m = self.v = 0.0
        self.step = 0
        self.order = []
        self.calls = 0
        self.fail_operation = None

    def touch(self, name):
        self.calls += 1
        if self.fail_operation == name:
            raise RuntimeError("injected " + name)

    def reset_gradients(self):
        self.touch("reset")
        self.gradient = 0.0

    def forward(self, row):
        self.touch("forward")
        self.order.append(row["row_id"])
        target = sum(row["completion_ids"][:-1]) / max(1, len(row["completion_ids"]) - 1) / 100000
        return self.weight + self.rng.random() * .01 - target

    def completion_loss(self, output, row):
        self.touch("loss")
        self.last_derivative = output
        return output * output / 2

    def backward(self, loss, *, scale):
        self.touch("backward")
        self.gradient += self.last_derivative * scale

    def finite_clip_gradients(self, *, max_norm):
        self.touch("clip")
        if not math.isfinite(self.gradient):
            raise ValueError("nonfinite gradient")
        norm = abs(self.gradient)
        self.gradient = max(-max_norm, min(max_norm, self.gradient))
        return norm

    def optimizer_step(self, *, lr):
        self.touch("step")
        self.step += 1
        self.m = .9 * self.m + .1 * self.gradient
        self.v = .999 * self.v + .001 * self.gradient ** 2
        self.weight -= lr * (self.m / (1 - .9 ** self.step)) / (math.sqrt(self.v / (1 - .999 ** self.step)) + 1e-8)
        self.gradient = 0.0  # completed group has no pending accumulated gradient

    def save_checkpoint(self):
        self.touch("save")
        state = self.rng.getstate()
        return {"parameters": {"weight": self.weight}, "optimizer": {"step": self.step, "m": self.m, "v": self.v},
                "rng": {"python": [state[0], list(state[1]), state[2]], "order": list(self.order)},
                "gradients": {"empty": self.gradient == 0.0, "value": self.gradient}}

    def validate_state(self, state, completed):
        return (state["optimizer"]["step"] == completed and len(state["rng"]["order"]) == completed * 20
                and all(math.isfinite(x) for x in (state["parameters"]["weight"], state["optimizer"]["m"], state["optimizer"]["v"]))
                and state["optimizer"]["v"] >= 0 and state["gradients"] == {"empty": True, "value": 0.0})

    def publish_checkpoint(self, envelope):
        self.touch("publish")
        path = self.directory / ("boundary_" + str(envelope["cursor"]["completed_updates"]) + ".json")
        raw = schedule.canonical(envelope).encode()
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return {"path": str(path), "file_sha256": hashlib.sha256(raw).hexdigest(), "envelope_sha256": envelope["sha256"], "durable": True}

    def verify_publication(self, publication, *, envelope_sha256):
        raw = Path(publication["path"]).read_bytes()
        body = json.loads(raw)
        return (hashlib.sha256(raw).hexdigest() == publication["file_sha256"]
                and body["sha256"] == publication["envelope_sha256"] == envelope_sha256
                and body["sha256"] == schedule.digest({k: v for k, v in body.items() if k != "sha256"}))

    def restore_checkpoint(self, state):
        self.touch("restore")
        self.weight = state["parameters"]["weight"]
        self.step, self.m, self.v = state["optimizer"]["step"], state["optimizer"]["m"], state["optimizer"]["v"]
        r = state["rng"]["python"]
        self.rng.setstate((r[0], tuple(r[1]), r[2]))
        self.order = list(state["rng"]["order"])
        self.gradient = state["gradients"]["value"]


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration = schedule.read_json(PACKAGE / "STUDY_DEV_REGISTRATION_DRAFT.json")
        cls.pools = {arm: {row["row_id"]: row for row in schedule.read_jsonl(schedule.BUNDLE / ("token_pool_" + arm + ".jsonl"))}
                     for arm in ("M1", "M2")}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def backend(self, name):
        path = Path(self.temp.name) / name
        path.mkdir()
        return QuadraticAdam(path)

    def execute(self, arm, backend, ledger=None, **kwargs):
        ledger = ledger or engine.CallLedger(lambda event: None)
        return engine.run(self.registration, arm, self.pools[arm].__getitem__, backend, ledger, **kwargs)

    def test27_versus_three9_chunks_bitwise_parameters_moments_RNG_order_dose_both_arms(self):
        for arm in ("M1", "M2"):
            full = self.backend(arm + "full")
            result = self.execute(arm, full)
            self.assertEqual("COMPLETE_CPU_COMPONENT_CONTROL", result["status"])
            part = self.backend(arm + "first")
            chunk = self.execute(arm, part, stop_after_update=9)
            for boundary in (18, 27):
                checkpoint = chunk["checkpoints"][-1]
                part = self.backend(arm + str(boundary))
                ledger = engine.CallLedger(lambda event: None, checkpoint["terminal_costs"])
                chunk = self.execute(arm, part, ledger, resume=checkpoint,
                                     latest_terminal_costs=checkpoint["terminal_costs"], stop_after_update=boundary)
                self.assertNotEqual("FAILED_CPU_COMPONENT_CONTROL", chunk["status"])
            self.assertEqual(full.save_checkpoint(), part.save_checkpoint())
            self.assertEqual(full.order, part.order)
            self.assertEqual(540, len(full.order))
            self.assertEqual(result["costs"]["positions"], chunk["costs"]["positions"])
            self.assertEqual(27, chunk["costs"]["operations"]["optimizer_step"]["returned"])
            self.assertEqual(540, chunk["costs"]["operations"]["backward"]["returned"])
            self.assertEqual(2, chunk["costs"]["operations"]["checkpoint_reload"]["returned"])
            self.assertFalse(chunk["native_launch_authorized"])

    def test_durable_jsonl_sink_and_postsave_envelope_cost_binding(self):
        sink = engine.JsonlSink(Path(self.temp.name) / "events.jsonl")
        try:
            result = self.execute("M1", self.backend("durable"), engine.CallLedger(sink), stop_after_update=9)
        finally:
            sink.close()
        cp = result["checkpoints"][-1]
        self.assertEqual(cp["envelope"]["save_call_id"], cp["envelope"]["posted_save_costs"]["last_call_id"])
        events = [json.loads(line) for line in (Path(self.temp.name) / "events.jsonl").read_text().splitlines()]
        self.assertEqual("returned", events[-1]["event"])
        self.assertEqual("checkpoint_promote", events[-1]["operation"])
        marker = next(event for event in events if event["event"] == "checkpoint_promoted")
        self.assertEqual(cp["envelope"]["posted_save_costs"], marker["costs_before_promotion"])

    def test_publisher_failure_retains_save_and_training_costs_refuses_promotion(self):
        backend = self.backend("publisher_failure")
        backend.fail_operation = "publish"
        result = self.execute("M1", backend, stop_after_update=9)
        self.assertEqual("FAILED_CPU_COMPONENT_CONTROL", result["status"])
        self.assertEqual([], result["checkpoints"])
        self.assertEqual(9, result["costs"]["operations"]["optimizer_step"]["returned"])
        self.assertEqual(0, result["costs"]["operations"]["checkpoint_promote"]["attempted"])
        self.assertEqual(1, result["costs"]["operations"]["checkpoint_save"]["returned"])
        self.assertEqual(1, result["costs"]["operations"]["checkpoint_publish"]["failed"])

    def test_failed_attempt_sink_keeps_pending_without_invoking_operation(self):
        def fail(event):
            if event["event"] == "attempt":
                raise OSError("durable sink unavailable")
        backend = self.backend("attempt_failure")
        result = self.execute("M1", backend, engine.CallLedger(fail), stop_after_update=9)
        self.assertEqual(0, backend.calls)
        self.assertEqual(1, result["costs"]["operations"]["gradient_reset"]["pending"])
        self.assertEqual(1, len(result["costs"]["pending"]))

    def test_failed_return_sink_does_not_rewrite_returned_optimizer_as_failed(self):
        def fail(event):
            if event["event"] == "returned" and event["operation"] == "optimizer_step":
                raise OSError("returned sink failure")
        result = self.execute("M1", self.backend("return_failure"), engine.CallLedger(fail), stop_after_update=9)
        counts = result["costs"]["operations"]["optimizer_step"]
        self.assertEqual({"attempted": 1, "returned": 1, "failed": 0, "pending": 0}, counts)
        self.assertEqual([], result["checkpoints"])

    def test_corrupt_source_cursor_moment_partial_state_rejected_before_restore(self):
        first = self.execute("M1", self.backend("checkpoint"), stop_after_update=9)
        original = first["checkpoints"][-1]
        for kind in ("source", "cursor", "moment", "partial"):
            cp = deepcopy(original)
            if kind == "source":
                cp["envelope"]["recipe"]["engine_source_sha256"] = "0" * 64
            elif kind == "cursor":
                cp["envelope"]["cursor"]["consumed_occurrences"] -= 1
            elif kind == "moment":
                cp["envelope"]["backend_state"]["optimizer"]["m"] += 1
            else:
                cp["envelope"]["backend_state"]["gradients"]["empty"] = False
            cp["envelope"]["sha256"] = schedule.digest({k: v for k, v in cp["envelope"].items() if k != "sha256"})
            cp["publication"]["envelope_sha256"] = cp["envelope"]["sha256"]
            backend = self.backend("corrupt_" + kind)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.execute("M1", backend, engine.CallLedger(lambda e: None, cp["terminal_costs"]),
                             resume=cp, latest_terminal_costs=cp["terminal_costs"], stop_after_update=18)
            self.assertEqual(0, backend.calls)

    def test_failed_checkpoint_promotion_marker_exposes_no_resume_boundary(self):
        def fail(event):
            if event["event"] == "checkpoint_promoted":
                raise OSError("promotion marker unavailable")
        result = self.execute("M1", self.backend("promotion_failed"), engine.CallLedger(fail), stop_after_update=9)
        self.assertEqual("FAILED_CPU_COMPONENT_CONTROL", result["status"])
        self.assertEqual([], result["checkpoints"])
        self.assertEqual(1, result["costs"]["operations"]["checkpoint_publish"]["returned"])
        self.assertEqual(9, result["costs"]["operations"]["optimizer_step"]["returned"])
        self.assertEqual(1, result["costs"]["operations"]["checkpoint_promote"]["failed"])

    def test_later_failed_work_cannot_roll_back_and_erase_costs(self):
        first = self.execute("M1", self.backend("rollback_first"), stop_after_update=9)
        cp = first["checkpoints"][-1]
        backend = self.backend("rollback_later")
        backend.fail_operation = "forward"
        ledger = engine.CallLedger(lambda e: None, cp["terminal_costs"])
        failed = self.execute("M1", backend, ledger, resume=cp, latest_terminal_costs=cp["terminal_costs"], stop_after_update=18)
        self.assertEqual(1, failed["costs"]["operations"]["forward"]["failed"])
        new = self.backend("rollback_refused")
        with self.assertRaisesRegex(ValueError, "later costs/rollback refused"):
            self.execute("M1", new, engine.CallLedger(lambda e: None, failed["costs"]), resume=cp,
                         latest_terminal_costs=failed["costs"], stop_after_update=18)
        self.assertEqual(0, new.calls)

    def test_wrong_scope_qualification_STATE1_and_modified_reader_refused_before_ops(self):
        backend = self.backend("wrong_scope")
        backend.execution_scope = "UNQUALIFIED_NATIVE_COMPONENT"
        with self.assertRaisesRegex(ValueError, "CPU_COMPONENT_CONTROL"):
            self.execute("M1", backend, stop_after_update=9)
        self.assertEqual(0, backend.calls)
        backend = self.backend("STATE1")
        with self.assertRaisesRegex(ValueError, "qualification STATE1 forbidden"):
            self.execute("M1", backend, resume={"kind": "STATE1.pt"}, latest_terminal_costs={}, stop_after_update=18)
        row = deepcopy(next(iter(self.pools["M1"].values())))
        row["labels"][0] = row["input_ids"][0]
        with self.assertRaises(ValueError):
            engine.run(self.registration, "M1", lambda identifier: row, backend, engine.CallLedger(lambda e: None), stop_after_update=9)
        self.assertEqual(0, backend.calls)


if __name__ == "__main__":
    unittest.main()
