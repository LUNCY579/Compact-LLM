"""CPU arithmetic and registration refusal fixtures, no model imports."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geo_frozen
import parentq_rows
import parentq_validation as v

b = geo_frozen.load("benchmark")


class ValidationTests(unittest.TestCase):
    def test_operative_key_ignores_untouched_accounts_tick_and_names(self):
        p = {"before": {"A": 19, "B": 7}, "event": {"kind": "scale", "account": "A", "other": "B", "amount": 3, "status": "posted", "tick": 1}}
        q = {"before": {"X": 19, "Y": -90}, "event": {"kind": "scale", "account": "X", "other": "Y", "amount": 3, "status": "posted", "tick": 22}}
        self.assertEqual(v.operative_number_key(b, "ledger_replay", p), v.operative_number_key(b, "ledger_replay", q))
        q["before"]["X"] = 20
        self.assertNotEqual(v.operative_number_key(b, "ledger_replay", p), v.operative_number_key(b, "ledger_replay", q))

    def test_detects_reviewed_collision_without_mutating_bank(self):
        rows = parentq_rows.build_bank(b)["T"]
        before = [r.to_dict() for r in rows]
        screen_parent = [b._transition_h("screen", i + 1, family, payload)
                         for i, (family, payload) in enumerate(b._SECTION_H["screen"]["t"])]
        report = v.operative_collisions(b, rows, screen_parent, [])
        self.assertIn("h-screen-T2-ledger_replay", report["hits"]["h-parentq-T14-ledger_replay"])
        self.assertFalse(report["collision_free"])
        self.assertEqual(before, [r.to_dict() for r in rows])

    def test_readiness_pending_refuses_and_empty_decisions_cannot_pass(self):
        here = Path(__file__).parent
        self.assertFalse(v.readiness(here)[1])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            payload = {"status": "READY_FOR_REVIEWED_PROPOSAL", "native_execution_authorized": False,
                       "independent_review": {"status": "PASS"}, "decisions": {}}
            (path / "READINESS_PARENTQ.json").write_text(json.dumps(payload))
            self.assertFalse(v.readiness(path)[1])
            payload["decisions"] = {"estimand_and_calibration": {"status": "RESOLVED"}}
            (path / "READINESS_PARENTQ.json").write_text(json.dumps(payload))
            self.assertFalse(v.readiness(path)[1])

    def test_both_exposed_hashes_are_required_before_payload_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in v.EXPOSED_T_PINS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"rows": []}')
            with self.assertRaisesRegex(ValueError, "STOP_EXPOSED_T_BANK_DIGEST_MISMATCH"):
                list(v.pinned_exposed_payloads(root, b, root / "delta", "0" * 64))

    def test_freeze_stops_on_pending_design_before_any_output_or_tests(self):
        import build_parentq
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "uncreated"
            with self.assertRaisesRegex(SystemExit, "STOP_UNFROZEN_DESIGN_DECISIONS_PENDING"):
                build_parentq.freeze(root)
            self.assertFalse(root.exists())

    def test_readiness_definitions_match_native_safety(self):
        import parentq_safety
        self.assertEqual(v.REQUIRED_DECISIONS, parentq_safety.REQUIRED_DECISIONS)


if __name__ == "__main__":
    unittest.main()
