"""Registered-prefix/admission/accounting CPU controls; no tensor physics."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest

from experiments.post_training_mimo_centralized_v1.scripts import qualify_native_backend as q


class Clock:
    def __init__(self): self.value = 0.
    def now(self): return self.value
    def pause(self, value): self.value += value


class QualificationControls(unittest.TestCase):
    def setUp(self):
        self.cfg = q.load_config(q.ROOT / q.CONFIG_RELATIVE)

    def test_frozen_registration_and62_forward_dose(self):
        self.assertEqual(hashlib.sha256((q.ROOT / q.CONFIG_RELATIVE).read_bytes()).hexdigest(), q.CONFIG_SHA256)
        dose = q.validate_prefix(self.cfg)
        self.assertEqual((dose["total_forwards"], dose["total_input_positions"], dose["total_supervised_forward_positions"]), (62, 8326, 736))
        self.assertEqual((dose["native_input_positions"], dose["native_supervised_positions"]), (5248, 463))
        self.assertEqual((dose["reference_input_positions"], dose["reference_supervised_positions"]), (2876, 251))

    def test_foreign_config_and_mutated_prefix_refused(self):
        with self.assertRaises(ValueError): q.load_config(q.ROOT / "elsewhere.json")
        for mode in ("reorder", "duplicate", "boolean", "component", "dose"):
            cfg = deepcopy(self.cfg)
            if mode == "reorder": cfg["prefix_rows"][0], cfg["prefix_rows"][1] = cfg["prefix_rows"][1], cfg["prefix_rows"][0]
            if mode == "duplicate": cfg["prefix_rows"][1]["row_key"] = cfg["prefix_rows"][0]["row_key"]
            if mode == "boolean": cfg["prefix_rows"][0]["optimizer_step"] = True
            if mode == "component": cfg["prefix_rows"][0]["component"] = "language_replay"
            if mode == "dose": cfg["prepared_dose"]["total_input_positions"] -= 1
            with self.assertRaises(ValueError): q.validate_prefix(cfg)

    def toy_row(self):
        labels = [-100, 11, 151645]
        expected = {"row_key": "acquisition:fixture", "component": "acquisition", "nonpadding_tokens": 3,
                    "supervised_tokens": 2, "input_token_ids_sha256": "a" * 64, "source_row_sha256": "b" * 64,
                    "labels_sha256": hashlib.sha256(q.canonical(labels)).hexdigest()}
        row = {"row_id": "fixture", "nonpadding_tokens": 3, "completion_tokens": 2,
               "input_token_ids_sha256": "a" * 64, "source_row_sha256": "b" * 64, "labels": labels}
        return row, expected

    def test_labels_hash_is_computed_and_component_supplied_as_copy(self):
        row, expected = self.toy_row()
        output = q.validate_row(row, expected)
        self.assertEqual(output["component"], "acquisition")
        self.assertNotIn("component", row)
        self.assertNotIn("labels_sha256", row)
        row["labels"][0] = 10
        with self.assertRaises(ValueError): q.validate_row(row, expected)

    def test_source_count_and_supplied_component_drift_refused(self):
        for field, value in (("row_id", "wrong"), ("nonpadding_tokens", 4), ("completion_tokens", 1),
                             ("source_row_sha256", "c" * 64), ("component", "language_replay")):
            row, expected = self.toy_row(); row[field] = value
            with self.assertRaises(ValueError): q.validate_row(row, expected)

    def journal(self, emit=None, caps=None):
        return q.ReferenceJournal(emit or (lambda *args, **kwargs: None), caps or {"reference_forward": 20, "reference_autograd_grad": 20})

    def test_reference_attempted_and_returned_position_accounting(self):
        journal = self.journal()
        self.assertEqual(journal.call("reference_forward", lambda: 42, sequence=20, input_positions=101, supervised_positions=11), 42)
        self.assertEqual(journal.snapshot()["positions"], {"input_positions_attempted": 101, "input_positions_returned": 101,
                         "supervised_positions_attempted": 11, "supervised_positions_returned": 11})

    def test_reference_failed_work_preserves_attempted_positions(self):
        journal = self.journal()
        def fail(): raise RuntimeError("synthetic gradient/forward failure")
        with self.assertRaises(RuntimeError): journal.call("reference_forward", fail, sequence=20, input_positions=101, supervised_positions=11)
        self.assertEqual(journal.snapshot()["operations"]["reference_forward"], {"attempted": 1, "returned": 0, "failed": 1, "pending": 0})
        self.assertEqual(journal.snapshot()["positions"]["input_positions_attempted"], 101)
        self.assertEqual(journal.snapshot()["positions"]["input_positions_returned"], 0)

    def test_reference_return_event_failure_keeps_actual_return(self):
        def emit(kind, **fields):
            if kind == "reference_returned": raise OSError("journal unavailable")
        journal = self.journal(emit)
        with self.assertRaises(OSError): journal.call("reference_forward", lambda: None, sequence=20, input_positions=101, supervised_positions=11)
        self.assertEqual(journal.snapshot()["operations"]["reference_forward"]["returned"], 1)
        self.assertEqual(journal.snapshot()["positions"]["supervised_positions_returned"], 11)
        self.assertEqual(journal.snapshot()["operations"]["reference_forward"]["failed"], 0)

    def test_reference_attempt_event_failure_prevents_execution(self):
        calls = []
        def emit(*args, **kwargs): raise OSError("journal unavailable")
        journal = self.journal(emit)
        with self.assertRaises(OSError): journal.call("reference_forward", lambda: calls.append(1), sequence=20, input_positions=101, supervised_positions=11)
        self.assertEqual(calls, [])
        self.assertEqual(journal.snapshot()["operations"]["reference_forward"]["pending"], 1)

    def test_reference_call_and_position_caps_do_not_reset(self):
        journal = self.journal(caps={"reference_forward": 2})
        for sequence in (20, 21): journal.call("reference_forward", lambda: None, sequence=sequence, input_positions=100, supervised_positions=10)
        with self.assertRaises(ValueError): journal.call("reference_forward", lambda: None, sequence=22, input_positions=100, supervised_positions=10)
        journal = self.journal()
        for sequence in range(20, 25): journal.call("reference_forward", lambda: None, sequence=sequence, input_positions=512, supervised_positions=1)
        with self.assertRaises(ValueError): journal.call("reference_forward", lambda: None, sequence=25, input_positions=512, supervised_positions=1)

    def test_bad_reference_shapes_counts_and_booleans_rejected(self):
        for sequence, inputs, targets in ((True, 10, 2), (19, 10, 2), (40, 10, 2), (20, True, 1), (20, 513, 1), (20, 10, 0), (20, 10, 10)):
            with self.assertRaises(ValueError): self.journal().call("reference_forward", lambda: None, sequence=sequence,
                input_positions=inputs, supervised_positions=targets)

    def test_preworker_two_consecutive_current_samples_required(self):
        clock = Clock(); samples = iter([3.4, 3.0, 3.4, 3.4]); recorded = []
        result = q.preworker_wait(source_check=lambda: None, available=lambda: next(samples), clock=clock.now, pause=clock.pause, record=recorded.append)
        self.assertEqual(result["status"], "ADMITTED_PREWORKER_RESOURCE_ONLY")
        self.assertEqual(result["observations"], 4)
        self.assertEqual(clock.value, 6.)
        self.assertIs(result["E1_authorized"], False)

    def test_preworker_timeout_and_slow_source_or_journal_cannot_pass(self):
        for mode in ("low", "source", "journal"):
            clock = Clock(); records = []
            def source():
                if mode == "source": clock.value += 31.
            def record(value):
                records.append(value)
                if mode == "journal" and len(records) == 2: clock.value = 31.
            result = q.preworker_wait(source_check=source, available=lambda: 1. if mode == "low" else 3.4,
                clock=clock.now, pause=clock.pause, record=record)
            self.assertEqual(result["status"], "TIMED_OUT_NO_WORKER")
            self.assertEqual(result["model_calls"], 0)

    def test_preworker_journal_error_refuses_admission(self):
        clock = Clock()
        def record(value): raise OSError("disk unavailable")
        with self.assertRaises(OSError): q.preworker_wait(source_check=lambda: None, available=lambda: 3.4,
            clock=clock.now, pause=clock.pause, record=record)

    def test_nonfinite_diagnostic_values_preserved_as_null_with_flags(self):
        value = q.safe_numbers({"gradient": {"max_abs": float("nan"), "mean_abs": float("inf"), "finite": False, "passed": False}})
        self.assertIsNone(value["gradient"]["max_abs"])
        self.assertIs(value["gradient"]["finite"], False)
        q.canonical(value)

    def test_cpu_controls_import_no_framework_and_scientific_grants_closed(self):
        self.assertFalse(set(("torch", "transformers", "tokenizers", "safetensors")).intersection(sys.modules))
        for field in ("native_math_qualified", "native_resume_qualified", "E1_authorized", "CHECK_FINAL_access", "goal_complete"):
            self.assertIs(self.cfg[field], False)

    def complete_counters(self):
        def operations(values): return {"operations": {name: {"attempted": value, "returned": value, "failed": 0, "pending": 0} for name, value in values.items()}}
        def positions(inputs, targets): return {"input_positions_attempted": inputs, "input_positions_returned": inputs,
                                               "supervised_positions_attempted": targets, "supervised_positions_returned": targets}
        outer = operations({"forward": 40, "backward": 40, "gradient_reset": 2, "gradient_clip": 2, "optimizer_step": 2,
                            "checkpoint_save": 2, "checkpoint_reload": 1}); outer["positions"] = positions(5248, 463)
        primitive = operations({"optimizer_construct": 1, "forward": 40, "backward": 40, "gradient_reset": 3,
                                "gradient_clip": 2, "optimizer_step": 2, "checkpoint_save": 2, "checkpoint_load": 1})
        loader = operations({"model_construct": 1, "checkpoint_load": 1, "tensor_copy": 310, "forward": 2})
        loader.update(forward_input_positions_attempted=202, forward_input_positions_returned=202)
        reference = operations({"reference_forward": 20, "reference_autograd_grad": 20, "reference_optimizer_construct": 1,
                                "reference_gradient_clip": 1, "reference_optimizer_step": 1}); reference["positions"] = positions(2876, 251)
        faults = operations({name: 1 for name in ("restore_fault_adapter", "restore_fault_moment", "restore_fault_python_seed",
                                               "restore_fault_torch_cpu_seed", "restore_fault_cuda_seed")})
        return [outer, primitive, loader, reference, faults, positions(5248, 463), {"attempted": 22, "returned": 22}]

    def test_complete_mechanical_scopes_and_extra_restore_reset(self):
        result = q.verify_accounting(*self.complete_counters())
        self.assertEqual(result, {"forwards": 62, "input_positions": 8326, "supervised_positions": 736})

    def test_incomplete_bool_extra_call_or_position_counts_refuse_pass(self):
        for mode in ("primitive_reset", "reference", "fault", "boolean", "extra", "position"):
            counters = self.complete_counters()
            if mode == "primitive_reset": counters[1]["operations"]["gradient_reset"]["returned"] = 2
            if mode == "reference": counters[3]["operations"]["reference_forward"]["returned"] = 19
            if mode == "fault": counters[4]["operations"]["restore_fault_cuda_seed"]["failed"] = 1
            if mode == "boolean": counters[1]["operations"]["optimizer_construct"]["returned"] = True
            if mode == "extra": counters[0]["operations"]["other"] = {"attempted": 1, "returned": 1, "failed": 0, "pending": 0}
            if mode == "position": counters[5]["input_positions_returned"] -= 1
            with self.assertRaises(ValueError): q.verify_accounting(*counters)


if __name__ == "__main__": unittest.main()
