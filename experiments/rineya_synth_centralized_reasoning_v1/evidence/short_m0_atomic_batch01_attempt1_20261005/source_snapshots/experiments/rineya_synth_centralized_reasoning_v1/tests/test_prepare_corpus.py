"""Focused CPU preparation boundary tests; all bundle writes are mocked.

No tokenizer engine, matching optimizer, GPU, model, or native run is invoked.
No recursive test discovery or filesystem cleanup is performed by these tests.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import mock_open, patch

PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

import prepare_corpus as preparation
from rs_contract import digest, render


def fixture_pool():
    spec = {"initial": [1, 3, 7, 15, 31, 63], "selection": {"k": 2, "target": 34}}
    rows = []
    parents = {f"p{i:03d}": {} for i in range(64)}
    for i in range(20):
        for arm in ("M1", "M2"):
            for ci in range(4):
                for template in range(4):
                    row = {"parent_id": f"p{i:03d}", "arm": arm, "condition_index": ci,
                           "template": template, "split": "train", "messages": render(spec, template),
                           "target": '{"answer":[1,4]}'}
                    row["row_id"] = digest(row)
                    rows.append(row)
    plan = {"version": "synth-balanced-corpus-1", "parents": parents, "sha256": "p" * 64}
    policy = json.loads((PACKAGE / "MATCHING_POLICY.json").read_text(encoding="utf-8"))
    decisions = json.loads((PACKAGE / "PILOT_DECISIONS.json").read_text(encoding="utf-8"))
    enrollment = {"parents": list(parents)[:20], "policy_sha256": digest(policy),
                  "record_ids": {arm: [r["row_id"] for r in rows if r["arm"] == arm] for arm in ("M1", "M2")}}
    return plan, enrollment, rows, policy, decisions


class FakeTokenizer:
    def __init__(self, events, fail_after=None):
        self.events, self.fail_after, self.encoded = events, fail_after, 0

    def verify_binding(self):
        self.events.append("binding")
        return {"pure_cpu_test_binding": True}

    def encode_example(self, messages, target):
        if self.fail_after is not None and self.encoded == self.fail_after:
            raise ValueError("deliberate CPU tokenizer fixture failure")
        self.encoded += 1
        return {"prompt_ids": [1], "completion_ids": [2, 151645], "input_ids": [1, 2, 151645],
                "labels": [-100, 2, 151645], "attention_mask": [1, 1, 1],
                "prompt_tokens": 1, "supervised_tokens": 2, "total_tokens": 3,
                "prompt_text_sha256": "1" * 64, "full_template_sha256": "2" * 64}


class PreparationBoundaryTests(unittest.TestCase):
    def test_only_one_new_direct_data_child_is_accepted(self):
        root = (PACKAGE / "data").resolve()
        leaf = "__uncreated_prepare_unit_fixture__"
        self.assertEqual(root / leaf, preparation.resolve_output(leaf))
        self.assertEqual(root / leaf, preparation.resolve_output(Path("data") / leaf))
        self.assertEqual(root / leaf, preparation.resolve_output(root / leaf))
        for path in (PACKAGE / "outside", Path("..") / leaf, root / leaf / "nested", root):
            with self.subTest(path=path), self.assertRaises(ValueError):
                preparation.resolve_output(path)
        with patch.object(Path, "exists", return_value=True), self.assertRaises(ValueError):
            preparation.resolve_output(leaf)
        with patch.object(Path, "is_symlink", return_value=True), self.assertRaises(ValueError):
            preparation.resolve_output(leaf)

    def test_writers_use_exclusive_creation_and_never_overwrite(self):
        mocked = mock_open()
        with patch.object(Path, "open", mocked):
            preparation.write_json_once(Path("new.json"), {"value": 1})
        self.assertEqual("x", mocked.call_args.args[0])
        self.assertEqual({"value": 1}, json.loads(mocked().write.call_args.args[0]))
        with patch.object(Path, "open", side_effect=FileExistsError("existing artifact")), self.assertRaises(FileExistsError):
            preparation.write_json_once(Path("existing.json"), {})
        mocked = mock_open()
        with patch.object(Path, "open", mocked):
            preparation.write_jsonl_once(Path("new.jsonl"), [{"value": 1}, {"value": 2}])
        self.assertEqual("x", mocked.call_args.args[0])
        self.assertEqual(2, mocked().write.call_count)

    def test_policy_defaults_and_closed_execution_permissions_are_required(self):
        _, _, _, policy, decisions = fixture_pool()
        preparation.validate_policy(policy)
        preparation.validate_decisions(decisions)
        for key, bad in (("source_seed", 1), ("candidate_parents", True),
                         ("training_admitted", True), ("quadruple_repeat_max", 7)):
            changed = copy.deepcopy(policy)
            changed[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                preparation.validate_policy(changed)
        decisions["permissions"]["native_inference"] = True
        with self.assertRaises(ValueError):
            preparation.validate_decisions(decisions)

    def test_framework_and_cooperative_resource_guards_fail_closed(self):
        with patch.dict(sys.modules, {"torch.test_fixture": object()}), self.assertRaises(ValueError):
            preparation.no_frameworks()
        with patch.object(preparation, "peak_working_set_bytes", return_value=preparation.CPU_PEAK_LIMIT_BYTES + 1):
            watch = preparation.ResourceWatch()
            with self.assertRaises(ValueError):
                watch.check("CPU fixture memory check")
            self.assertFalse(watch.record()["cpu_budget_satisfied"])
        with patch.object(preparation, "peak_working_set_bytes", return_value=1024), \
             patch.object(preparation.time, "perf_counter", side_effect=[0, 601]):
            watch = preparation.ResourceWatch()
            with self.assertRaises(ValueError):
                watch.check("CPU fixture wall check")


class MockedPreparationTests(unittest.TestCase):
    def run_fixture(self, *, matched=False, fail_after=None, drift=False, source_policy=1):
        plan, enrollment, rows, policy, decisions = fixture_pool()
        artifacts, events = {}, []
        bundle = PACKAGE / "data" / "__mocked_preparation_no_directory_created__"
        tokenizer = FakeTokenizer(events, fail_after=fail_after)
        raw_sources = {"MATCHING_POLICY.json": json.dumps(policy).encode("utf-8"),
                       "PILOT_DECISIONS.json": json.dumps(decisions).encode("utf-8")}
        if source_policy == 2:
            plan["version"] = "synth-balanced-corpus-2"
            raw_sources.update({name: (PACKAGE / name).read_bytes()
                                for name in ("SOURCE_POLICY_002.json", "CORPUS_AMENDMENT_002.md")})
        result = {"status": "MATCHED_CPU_SCHEDULE" if matched else "BLOCKED_MATCHING",
                  "reason": "independent CPU fixture outcome", "training_admitted": False}
        if matched:
            result.update({"sha256": "s" * 64,
                           "schedules": {arm: enrollment["record_ids"][arm] for arm in ("M1", "M2")}})

        def save(path, value):
            key = path.name
            if key in artifacts:
                raise FileExistsError(key)
            artifacts[key] = copy.deepcopy(value)

        def enroll(value):
            events.append("enrollment")
            return enrollment

        def load(*args, **kwargs):
            events.append("tokenizer_load")
            self.assertIn("enrollment.json", artifacts)
            return tokenizer

        with ExitStack() as stack:
            for name, value in (("create_bundle", lambda value: bundle), ("write_json_once", save),
                                ("write_jsonl_once", save), ("write_bytes_once", save),
                                ("snapshot_sources", lambda b, h, source_policy: h),
                                ("build_balanced_plan", lambda **kwargs: plan), ("freeze_enrollment", enroll),
                                ("compile_balanced_records", lambda p: rows), ("validate_balanced_records", lambda r, p: None),
                                ("load_local_tokenizer", load), ("match_schedule", lambda *args: result),
                                ("validate_schedule", lambda *args: events.append("schedule_validated")),
                                ("git_head", lambda: "CPU_FIXTURE_HEAD"), ("bundle_file_hashes", lambda b: {}),
                                ("peak_working_set_bytes", lambda: 32 * 1024 * 1024)):
                stack.enter_context(patch.object(preparation, name, side_effect=value))
            pins = {name: hashlib.sha256(raw).hexdigest() for name, raw in raw_sources.items()}
            hashes = [{**pins, "prepare_corpus.py": "a"}, {**pins, "prepare_corpus.py": "b" if drift else "a"}]
            stack.enter_context(patch.object(preparation, "source_hashes", side_effect=hashes))
            stack.enter_context(patch.object(Path, "read_bytes", lambda path: raw_sources[path.name]))
            report = preparation.prepare_bundle("mocked_unused_leaf", source_policy=source_policy)
        return report, artifacts, events

    def test_blocked_outcome_preserves_all_candidates_and_closed_flags(self):
        report, artifacts, events = self.run_fixture()
        self.assertEqual("BLOCKED_MATCHING", report["status"])
        self.assertEqual(2, report["exit_code"])
        self.assertLess(events.index("enrollment"), events.index("tokenizer_load"))
        self.assertEqual(640, len(artifacts["candidate_records.jsonl"]))
        for arm in ("M1", "M2"):
            self.assertEqual(320, len(artifacts[f"student_pool_{arm}.jsonl"]))
            self.assertTrue(all(set(row) == {"messages", "target"}
                                for row in artifacts[f"student_pool_{arm}.jsonl"]))
            self.assertEqual(320, len(artifacts[f"token_pool_{arm}.jsonl"]))
            self.assertTrue(all(set(row) == {"row_id", *preparation.ARRAY_FIELDS}
                                for row in artifacts[f"token_pool_{arm}.jsonl"]))
        self.assertNotIn("training_occurrences.json", artifacts)
        receipt = artifacts["preparation_receipt.json"]
        self.assertEqual(640, receipt["counts"]["token_arrays_materialized"])
        self.assertTrue(receipt["tokenizer_binding_verified_unchanged"])
        self.assertFalse(receipt["training_admitted"])
        self.assertFalse(receipt["native_qualification"])

    def test_partial_tokenizer_failure_retains_completed_arrays_and_reason(self):
        report, artifacts, _ = self.run_fixture(fail_after=2)
        self.assertEqual("CPU_PREPARATION_FAILED", report["status"])
        self.assertEqual(2, report["exit_code"])
        self.assertEqual(2, artifacts["preparation_receipt.json"]["counts"]["token_arrays_materialized"])
        self.assertEqual(2, len(artifacts["token_pool_M1.jsonl"]))
        self.assertEqual(0, len(artifacts["token_pool_M2.jsonl"]))
        self.assertIn("deliberate CPU tokenizer fixture failure", report["reason"])
        self.assertIn("schedule_result.json", artifacts)

    def test_matched_occurrences_reference_arrays_and_still_do_not_admit_training(self):
        report, artifacts, events = self.run_fixture(matched=True)
        self.assertEqual(0, report["exit_code"])
        self.assertIn("schedule_validated", events)
        occurrences = artifacts["training_occurrences.json"]
        self.assertEqual({"M1", "M2"}, set(occurrences["row_ids"]))
        self.assertFalse(occurrences["training_admitted"])
        self.assertNotIn("input_ids", occurrences)

    def test_source_drift_blocks_preparation_without_erasing_matching_outcome(self):
        report, artifacts, _ = self.run_fixture(drift=True)
        self.assertEqual("CPU_PREPARATION_FAILED", report["status"])
        self.assertEqual("BLOCKED_MATCHING", artifacts["schedule_result.json"]["status"])
        self.assertIn("production source", report["reason"])

    def test_source_policy_two_registration_is_preserved_before_tokenization(self):
        report, artifacts, events = self.run_fixture(source_policy=2)
        self.assertEqual("BLOCKED_MATCHING", report["status"])
        self.assertIn("source_policy_002.json", artifacts)
        self.assertIn("corpus_amendment_002.md", artifacts)
        receipt = artifacts["preparation_receipt.json"]
        self.assertEqual(2, receipt["source_policy"])
        self.assertEqual("synth-balanced-corpus-2", receipt["plan_version"])
        self.assertIn("SOURCE_POLICY_002.json", receipt["code_snapshot_hashes"])


if __name__ == "__main__":
    unittest.main()
