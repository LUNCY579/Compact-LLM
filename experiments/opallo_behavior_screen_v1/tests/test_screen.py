"""Synthetic tests only; no model outputs or scientific efficacy evidence."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "screen.py"
spec = importlib.util.spec_from_file_location("opallo_behavior_screen", PATH)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def capture(split="calibration", formatted=False):
    rows = []
    for i, key in enumerate(s.fixture(split)):
        text = key["answer"] if key["answerable"] else "The record does not say."
        if key["answerable"] and not formatted and i % 2 == 0:
            text = "Answer: " + text
        rows.append({"id": key["id"], "status": "ok", "terminated": True,
                     "prediction": text, "input_tokens": 100, "generated_tokens": 12,
                     "unknown_probability": 0.001 if key["answerable"] else 0.2})
    return {"protocol": s.PROTOCOL, "kind": "synthetic_fixture", "split": split,
            "formatted": formatted, "prompts_sha256": s.digest(s.prompts(split, formatted)),
            "identity": {"weight_sha256": s.WEIGHT_SHA256, "runtime_sha256": "a" * 64,
                         "unknown_single_token": True, "unknown_token_id": 917}, "rows": rows}


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.bc, self.fc = capture(), capture(formatted=True)
        self.b, self.f = capture("check"), capture("check", True)

    def test_counts(self):
        for split, n in (("calibration", 64), ("check", 128)):
            rows = s.fixture(split)
            self.assertEqual(len(rows), n)
            self.assertEqual(sum(r["answerable"] for r in rows), n * 3 // 4)
            self.assertEqual(len({r["scenario_id"] for r in rows}), n // 4)

    def test_split_disjointness(self):
        cal, check = s.fixture("calibration"), s.fixture("check")
        for field in ("id", "scenario_id"):
            self.assertFalse({r[field] for r in cal} & {r[field] for r in check})
        self.assertFalse({r["answer"] for r in cal if r["answerable"]} &
                         {r["answer"] for r in check if r["answerable"]})

    def test_fixture_oracle(self):
        for split in ("calibration", "check"):
            for row in s.fixture(split):
                record = row["messages"][1]["content"].split("\n\n")[0]
                if row["answerable"]:
                    self.assertIn(f'{row["task_type"]}: {row["answer"]}', record)
                else:
                    self.assertNotIn(f'{row["task_type"]}:', record)
                    self.assertEqual(row["answer"], "UNKNOWN")

    def test_prompt_only(self):
        for split in ("calibration", "check"):
            for row in s.prompts(split, False):
                self.assertEqual(set(row), {"id", "messages"})

    def test_single_intervention(self):
        for b, f in zip(s.prompts("check", False), s.prompts("check", True)):
            self.assertEqual(b["messages"][1], f["messages"][1])
            self.assertEqual(f["messages"][0]["content"], b["messages"][0]["content"] + "\n" + s.FORMAT)

    def test_deterministic(self):
        self.assertEqual(s.digest(s.fixture("check")), s.digest(s.fixture("check")))

    def test_invalid_split(self):
        with self.assertRaises(ValueError):
            s.fixture("test")

    def test_legacy_exact_boundaries(self):
        self.assertTrue(s.legacy_exact("Value.\nextra", "value"))
        self.assertFalse(s.legacy_exact("The value.", "value"))
        self.assertFalse(s.legacy_exact("Answer: value", "value"))
        self.assertFalse(s.legacy_exact("not value", "value"))
        self.assertFalse(s.legacy_exact("\nvalue", "value"))

    def test_unknown_is_not_substring(self):
        self.assertTrue(s.literal_unknown(" UNKNOWN \n"))
        for text in ("unknown", "UNKNOWN.", "UNKNOWN\nMaybe Tuesday", "not UNKNOWN", "unknownish", "not determinable"):
            self.assertFalse(s.literal_unknown(text))

    def test_literal_format_separate(self):
        rows = [r["answer"] + "." for r in s.fixture("check")]
        m = s.metrics("check", rows)
        self.assertEqual(m["answerable_legacy_exact"], 96)
        self.assertEqual(m["answerable_full_literal_exact"], 0)
        self.assertEqual(m["literal_unknown_on_unanswerable"], 0)

    def test_capture_validation(self):
        self.assertEqual(len(s.validate_capture(self.bc, "calibration", False)), 64)

    def test_capture_top_level_corruption(self):
        mutations = [("protocol", "wrong"), ("kind", ""), ("split", "check"),
                     ("formatted", 0), ("prompts_sha256", ""), ("rows", [])]
        for key, value in mutations:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.bc)
                bad[key] = value
                with self.assertRaises(ValueError):
                    s.validate_capture(bad, "calibration", False)

    def test_capture_identity_corruption(self):
        for key, value in (("weight_sha256", "0" * 64), ("runtime_sha256", "x" * 64),
                           ("unknown_single_token", False), ("unknown_token_id", -1),
                           ("unknown_token_id", True)):
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(self.bc)
                bad["identity"][key] = value
                with self.assertRaises(ValueError):
                    s.validate_capture(bad, "calibration", False)

    def test_missing_duplicate_extra_reordered(self):
        for mode in ("missing", "duplicate", "extra", "reverse", "nonobject"):
            with self.subTest(mode=mode):
                bad = copy.deepcopy(self.bc)
                if mode == "missing":
                    bad["rows"].pop()
                elif mode == "duplicate":
                    bad["rows"][1] = bad["rows"][0]
                elif mode == "extra":
                    bad["rows"].append(bad["rows"][0])
                elif mode == "nonobject":
                    bad["rows"][0] = None
                else:
                    bad["rows"].reverse()
                with self.assertRaises(ValueError):
                    s.validate_capture(bad, "calibration", False)

    def test_row_corruption(self):
        for key, value in (("status", "skipped"), ("terminated", False), ("terminated", 1),
                           ("prediction", None), ("generated_tokens", -1), ("generated_tokens", 25),
                           ("generated_tokens", True), ("input_tokens", 4073), ("input_tokens", 0),
                           ("unknown_probability", math.nan), ("unknown_probability", math.inf),
                           ("unknown_probability", -0.1), ("unknown_probability", 1.1),
                           ("unknown_probability", True)):
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(self.bc)
                bad["rows"][0][key] = value
                with self.assertRaises(ValueError):
                    s.validate_capture(bad, "calibration", False)

    def test_threshold_selection(self):
        spec = s.calibrate(self.bc, self.fc)
        self.assertEqual(spec["gate_threshold"], 0.1)

    def test_boundary_inclusive(self):
        row = {"prediction": "value", "unknown_probability": 0.1}
        self.assertEqual(s.gated([row], 0.1), ["UNKNOWN"])
        self.assertEqual(s.gated([row], 0.3), ["value"])
        self.assertEqual(s.gated([row], None), ["value"])

    def test_unregistered_thresholds(self):
        for tau in (True, False, -1, 0.123, math.nan, math.inf):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                s.gated([], tau)

    def test_off_wins_tie(self):
        for row, key in zip(self.bc["rows"], s.fixture("calibration")):
            row["prediction"] = key["answer"]
        self.assertIsNone(s.calibrate(self.bc, self.fc)["gate_threshold"])

    def test_cannot_buy_calibration_abstention(self):
        for row in self.bc["rows"]:
            row["unknown_probability"] = 0.2
        self.assertIsNone(s.calibrate(self.bc, self.fc)["gate_threshold"])

    def test_synthetic_success_never_native(self):
        result = s.replay(s.calibrate(self.bc, self.fc), self.bc, self.fc, self.b, self.f)
        self.assertTrue(result["screen_criteria_met"])
        self.assertEqual(result["status"], "SYNTHETIC_ONLY")
        self.assertTrue(all(result[k] is False for k in s.CLOSED))
        self.assertEqual(set(result["metrics"]), {"B", "F", "G", "FG"})

    def test_native_replay_never_attests_execution(self):
        for c in (self.bc, self.fc, self.b, self.f):
            c["kind"] = "native_capture"
        result = s.replay(s.calibrate(self.bc, self.fc), self.bc, self.fc, self.b, self.f)
        self.assertEqual(result["status"], "OFFLINE_REPLAY_ONLY")
        self.assertFalse(result["licenses_release"])

    def test_tampered_spec_rejected(self):
        spec = s.calibrate(self.bc, self.fc)
        spec["gate_threshold"] = 0.03
        with self.assertRaises(ValueError):
            s.replay(spec, self.bc, self.fc, self.b, self.f)

    def test_calibration_source_change_rejected(self):
        spec = s.calibrate(self.bc, self.fc)
        self.bc["rows"][0]["prediction"] += " "
        with self.assertRaises(ValueError):
            s.replay(spec, self.bc, self.fc, self.b, self.f)

    def test_cross_identity_rejected(self):
        self.f["identity"]["runtime_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            s.replay(s.calibrate(self.bc, self.fc), self.bc, self.fc, self.b, self.f)

    def test_calibration_identity_rejected(self):
        self.fc["identity"]["runtime_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            s.calibrate(self.bc, self.fc)

    def test_false_unknown_veto(self):
        self.f["rows"][0]["unknown_probability"] = 0.9
        result = s.replay(s.calibrate(self.bc, self.fc), self.bc, self.fc, self.b, self.f)
        self.assertFalse(result["screen_criteria_met"])
        self.assertFalse(result["conditions"]["zero_new_false_unknown"])

    def test_preservation_veto(self):
        self.f["rows"][1]["prediction"] = "incorrect"
        result = s.replay(s.calibrate(self.bc, self.fc), self.bc, self.fc, self.b, self.f)
        self.assertFalse(result["screen_criteria_met"])
        self.assertFalse(result["conditions"]["zero_lost_baseline_correct_answerable"])

    def test_no_gain_does_not_pass(self):
        for dst, src in zip(self.f["rows"], self.b["rows"]):
            dst["prediction"] = src["prediction"]
        result = s.replay(s.calibrate(self.bc, self.fc), self.bc, self.fc, self.b, self.f)
        self.assertFalse(result["screen_criteria_met"])

    def test_coverage_checks(self):
        with self.assertRaises(ValueError):
            s.metrics("check", [])
        with self.assertRaises(ValueError):
            s.losses("check", [], [])

    def test_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            s.write_new(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                s.write_new(path, {"a": 2})
            self.assertEqual(s.read(str(path)), {"a": 1})

    def test_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaises(ValueError):
                s.read(str(path))


if __name__ == "__main__":
    unittest.main()
