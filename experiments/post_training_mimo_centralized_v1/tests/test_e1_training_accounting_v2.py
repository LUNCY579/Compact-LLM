"""Versioned journal failure controls; no framework or native qualification."""
import sys
import unittest

from experiments.post_training_mimo_centralized_v1.scripts import e1_training_accounting as legacy
from experiments.post_training_mimo_centralized_v1.scripts import e1_training_accounting_v2 as v2


class VersionedAccountingControls(unittest.TestCase):
    def test_nested_attempt_publication_failure_preserves_error_and_pending_spans(self):
        original = OSError("synthetic inner journal unavailable")
        events, copies = [], []

        def emit(kind, **fields):
            if kind == "operation_attempt" and fields["operation"] == "tensor_copy":
                raise original
            events.append((kind, fields))

        ledger = v2.E1CallAccountingV2(emit)
        with self.assertRaises(OSError) as observed:
            ledger.call("checkpoint_load", lambda: ledger.call(
                "tensor_copy", lambda: copies.append(1), stage="copy"), stage="load")
        self.assertIs(observed.exception, original)
        self.assertEqual(copies, [])
        self.assertEqual([kind for kind, _ in events], ["operation_attempt"])
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot["last_call_id"], 2)
        self.assertEqual([call["call_id"] for call in snapshot["pending_calls"]], [1, 2])
        for operation in ("checkpoint_load", "tensor_copy"):
            self.assertEqual(snapshot["operations"][operation], {
                "attempted": 1, "returned": 0, "failed": 0, "pending": 1})
        with self.assertRaisesRegex(ValueError, "pending calls require explicit reconciliation"):
            v2.E1CallAccountingV2.from_snapshot(lambda *args, **kwargs: None, snapshot)

    def test_ordinary_operation_failure_remains_failed_and_preserves_position_costs(self):
        original = RuntimeError("synthetic forward failure")
        events = []
        ledger = v2.E1CallAccountingV2(lambda kind, **fields: events.append((kind, fields)))

        def fail():
            raise original

        with self.assertRaises(RuntimeError) as observed:
            ledger.call("forward", fail, stage="row", input_positions=7, supervised_positions=3)
        self.assertIs(observed.exception, original)
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot["operations"]["forward"], {
            "attempted": 1, "returned": 0, "failed": 1, "pending": 0})
        self.assertEqual(snapshot["pending_calls"], [])
        self.assertEqual(snapshot["positions"], {
            "input_positions_attempted": 7, "input_positions_returned": 0,
            "supervised_positions_attempted": 3, "supervised_positions_returned": 0})
        self.assertEqual([kind for kind, _ in events], ["operation_attempt", "operation_failed"])
        restored = v2.E1CallAccountingV2.from_snapshot(lambda *args, **kwargs: None, snapshot)
        self.assertIs(type(restored), v2.E1CallAccountingV2)
        self.assertEqual(restored.snapshot(), snapshot)

    def test_return_publication_failure_keeps_actual_return_and_position_costs(self):
        original = OSError("synthetic returned marker unavailable")
        calls = []

        def emit(kind, **fields):
            if kind == "operation_returned":
                raise original

        ledger = v2.E1CallAccountingV2(emit)
        with self.assertRaises(OSError) as observed:
            ledger.call("forward", lambda: calls.append(1), stage="row",
                        input_positions=7, supervised_positions=3)
        self.assertIs(observed.exception, original)
        self.assertEqual(calls, [1])
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot["operations"]["forward"], {
            "attempted": 1, "returned": 1, "failed": 0, "pending": 0})
        self.assertEqual(snapshot["pending_calls"], [])
        self.assertEqual(snapshot["positions"], {
            "input_positions_attempted": 7, "input_positions_returned": 7,
            "supervised_positions_attempted": 3, "supervised_positions_returned": 3})

    def test_source_version_has_compatible_snapshot_and_no_framework_or_authority(self):
        emit = lambda *args, **kwargs: None
        ledger = v2.E1CallAccountingV2(emit)
        self.assertEqual(ledger.snapshot(), legacy.E1CallAccounting(emit).snapshot())
        self.assertEqual(ledger.source_version, 2)
        self.assertEqual(ledger.source_scope, v2.SOURCE_SCOPE)
        self.assertIs(ledger.snapshot()["native_launch_authorized"], False)
        self.assertIs(ledger.snapshot()["E1_authorized"], False)
        self.assertFalse({"torch", "transformers", "safetensors", "tokenizers"}.intersection(sys.modules))


if __name__ == "__main__":
    unittest.main()
