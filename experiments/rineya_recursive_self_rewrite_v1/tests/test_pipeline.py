import copy
import tempfile
import unittest
from pathlib import Path

from experiments.rineya_recursive_self_rewrite_v1.core import *
from experiments.rineya_recursive_self_rewrite_v1.fixtures import catalog, fixture_backend, ByteTokenizer
from experiments.rineya_recursive_self_rewrite_v1.pipeline import *
from experiments.rineya_recursive_self_rewrite_v1.data import compile_dataset, encode_row
from experiments.rineya_recursive_self_rewrite_v1.artifacts import write_run, verify_run
from experiments.rineya_recursive_self_rewrite_v1.runtime import execute


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tasks, self.ancestry = catalog(1, offset=1)
        self.task, self.backend = self.tasks[0], fixture_backend()
        self.found = discovery(self.tasks, self.backend, Limits(), attempts=1)
        self.source = next(r for r in self.found["records"] if r["arm"] == "portfolio" and r["harness"] == "H1")

    def rewrite(self, **kwargs):
        return reconstruct(self.task, self.source, self.backend, self.backend, self.backend, Limits(), **kwargs)

    def dataset(self, rewrite=None, arm="recursive", cap=4):
        return compile_dataset(self.tasks, self.ancestry, self.found["records"],
                               [rewrite or self.rewrite()], arm=arm, parent_cap=cap,
                               source_ids=[self.source["id"]])

    def test_matched_schedules(self):
        report = discovery_report(self.found)
        self.assertEqual(report["h0_resampling"]["attempts"], report["portfolio"]["attempts"])
        self.assertEqual(self.found["ledgers"][0]["report"]["limits"], self.found["ledgers"][1]["report"]["limits"])

    def test_failure_denominators_complete(self):
        found = discovery(self.tasks, self.backend, Limits(calls=1), attempts=2)
        self.assertEqual(len(found["records"]), 12)
        self.assertGreater(discovery_report(found)["portfolio"]["failures"]["BUDGET"], 0)

    def test_missing_attempt_rejected(self):
        self.found["records"].pop()
        with self.assertRaises(ValueError):
            discovery_report(self.found)

    def test_changed_schedule_rejected(self):
        self.found["records"][0]["harness"] = "H2"
        with self.assertRaises(ValueError):
            discovery_report(self.found)

    def test_fresh_rewrite_provenance(self):
        rewritten = self.rewrite()
        validate_rewrite(rewritten, self.source, self.task)
        self.assertEqual(rewritten["track"], "self")
        self.assertEqual(len({r["sandbox"] for r in rewritten["records"]} | {self.source["sandbox"]}), 5)

    def test_teacher_assistance_labeled(self):
        teacher = Backend("helper-checkpoint", "fixture", self.backend.generate)
        rewritten = reconstruct(self.task, self.source, self.backend, teacher, self.backend, Limits())
        self.assertEqual(rewritten["track"], "teacher_assisted")

    def test_executor_identity_must_match(self):
        other = Backend("other-student", "fixture", self.backend.generate)
        with self.assertRaises(ValueError):
            reconstruct(self.task, self.source, other, other, other, Limits())

    def test_recursive_schema_repair(self):
        def plan(prompt, seed, maximum):
            value = loads(prompt)
            if value["previous"] is None:
                return Reply('{"goal":"evidence_backed_completion","steps":["answer 42"]}')
            return self.backend.generate(prompt, seed, maximum)
        planner = Backend("repair-planner", "fixture", plan)
        one = reconstruct(self.task, self.source, self.backend, planner, self.backend, Limits(), arm="one_pass", candidates=1)
        recursive = reconstruct(self.task, self.source, self.backend, planner, self.backend, Limits(), candidates=1)
        self.assertFalse(one["records"])
        self.assertEqual(len(recursive["drafts"]), 2)
        self.assertTrue(recursive["records"])

    def test_rejection_budget_is_bounded(self):
        def reject(prompt, *args):
            return Reply(canonical(dict(request_hash=loads(prompt)["request_hash"], decision="REJECT", reason="incomplete")))
        critic = Backend("reject", "fixture", reject)
        rewritten = reconstruct(self.task, self.source, self.backend, self.backend, critic, Limits(), candidates=2, revisions=2)
        self.assertEqual(len(rewritten["drafts"]), 6)
        self.assertFalse(rewritten["records"])

    def test_stale_critic_rejected(self):
        critic = Backend("stale", "fixture", lambda *args: Reply(canonical(dict(request_hash="stale", decision="ACCEPT", reason="accepted"))))
        rewritten = reconstruct(self.task, self.source, self.backend, self.backend, critic, Limits())
        self.assertFalse(rewritten["records"])
        self.assertTrue(all(d["feedback"] == "error" for d in rewritten["drafts"]))

    def test_critic_cannot_see_source_or_gold(self):
        rewritten = self.rewrite()
        for call in rewritten["calls"]:
            if call["role"] == "critic":
                value = loads(call["prompt"])
                self.assertNotIn("source_public", value)
                self.assertNotIn("expected", value)
                self.assertNotIn("records", value)

    def test_unapproved_runbook_rejected(self):
        rewritten = self.rewrite()
        rewritten["drafts"][0]["accepted"] = False
        rewritten["drafts"][1]["accepted"] = False
        with self.assertRaises(ValueError):
            validate_rewrite(rewritten, self.source, self.task)

    def test_forged_critic_approval_rejected(self):
        rewritten = self.rewrite()
        rewritten["drafts"][0]["critique"]["request_hash"] = "wrong"
        with self.assertRaises(ValueError):
            validate_rewrite(rewritten, self.source, self.task)

    def test_reused_sandbox_rejected(self):
        rewritten = self.rewrite()
        rewritten["records"][0]["sandbox"] = self.source["sandbox"]
        with self.assertRaises(ValueError):
            validate_rewrite(rewritten, self.source, self.task)

    def test_rewrites_of_rewrites_rejected(self):
        rewritten = self.rewrite()
        with self.assertRaises(ValueError):
            reconstruct(self.task, rewritten["records"][0], self.backend, self.backend, self.backend, Limits())

    def test_guidance_removed_from_sft_inputs(self):
        data = self.dataset()
        self.assertTrue(data["rows"])
        self.assertNotIn("PRIVATE_TRAINING_GUIDANCE", canonical(data["rows"]))
        self.assertNotIn("runbook", canonical(data["rows"]))

    def test_direct_keeps_source_context(self):
        data = self.dataset(arm="direct")
        self.assertIn("PRIVATE_TRAINING_GUIDANCE", data["rows"][0]["prompt"])
        self.assertEqual(data["track"], "source_context")

    def test_parent_cap(self):
        data = self.dataset(cap=1)
        self.assertEqual(data["parent_counts"], {self.task.parent: 1})
        self.assertEqual(data["rejected"]["parent_cap"], 3)

    def test_duplicate_source_rejected(self):
        with self.assertRaises(ValueError):
            compile_dataset(self.tasks, self.ancestry, self.found["records"] * 2, [], arm="direct")

    def test_mixed_assistance_rejected(self):
        first = self.rewrite()
        teacher = Backend("teacher", "fixture", self.backend.generate)
        second = reconstruct(self.task, self.source, self.backend, teacher, self.backend, Limits())
        with self.assertRaises(ValueError):
            compile_dataset(self.tasks, self.ancestry, self.found["records"], [first, second], arm="recursive", parent_cap=100)

    def test_exact_prompt_mask_and_one_eos(self):
        row = self.dataset()["rows"][0]
        encoded = encode_row(row, ByteTokenizer(), 10000)
        n = len(encoded["prompt_ids"])
        self.assertTrue(all(label == -100 for label in encoded["labels"][:n]))
        self.assertEqual(encoded["input_ids"][-1], 1)
        self.assertEqual(encoded["labels"].count(1), 1)
        self.assertEqual(encoded["prompt_ids"], ByteTokenizer().encode(row["prompt"]))

    def test_tokenization_overflow_rejected(self):
        with self.assertRaises(ValueError):
            encode_row(self.dataset()["rows"][0], ByteTokenizer(), 1)

    def test_eos_contaminated_completion_rejected(self):
        class BadTokenizer(ByteTokenizer):
            def encode(self, text, add_special_tokens=False):
                return [1, 3]
        with self.assertRaises(ValueError):
            encode_row(self.dataset()["rows"][0], BadTokenizer(), 100)

    def test_bundle_replay_roundtrip(self):
        rewritten = self.rewrite()
        dataset = self.dataset(rewritten)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "new"
            write_run(path, self.tasks, self.ancestry, self.found, [rewritten], [dataset])
            self.assertTrue(verify_run(path)["verified"])
            with self.assertRaises(FileExistsError):
                write_run(path, self.tasks, self.ancestry, self.found, [rewritten], [dataset])

    def test_corrupt_bundle_rejected(self):
        rewritten = self.rewrite()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "new"
            write_run(path, self.tasks, self.ancestry, self.found, [rewritten], [self.dataset(rewritten)])
            (path / "rewrites.private.json").write_text("[]")
            with self.assertRaises(ValueError):
                verify_run(path)

    def test_extra_bundle_file_rejected(self):
        rewritten = self.rewrite()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "new"
            write_run(path, self.tasks, self.ancestry, self.found, [rewritten], [self.dataset(rewritten)])
            (path / "unexpected").touch()
            with self.assertRaises(ValueError):
                verify_run(path)

    def test_corrected_track_uses_rhd_review_binding(self):
        def review(prompt, seed, maximum):
            value = loads(prompt)
            return Reply(canonical(dict(request_hash=value["request_hash"], decision="PASS")))
        reviewer = Backend("reviewer", "fixture", review)
        rewritten = self.rewrite(arm="corrected", reviewer=reviewer)
        validate_rewrite(rewritten, self.source, self.task)
        data = self.dataset(rewritten, arm="corrected")
        self.assertEqual(data["track"], "teacher_assisted")
        self.assertNotIn('"review"', canonical(data["rows"]))

    def test_actual_costs_are_reported_separately_from_caps(self):
        report = discovery_report(self.found)
        self.assertGreater(report["portfolio"]["actual_usage"]["calls"], 0)
        self.assertIsNone(report["portfolio"]["actual_usage"]["tokens"])

    def test_stale_local_review_is_rejected(self):
        reviewer = Backend("stale-reviewer", "fixture", lambda *args: Reply(canonical(
            dict(request_hash="wrong-session", decision="PASS"))))
        rewritten = self.rewrite(arm="corrected", reviewer=reviewer)
        self.assertTrue(all(r["status"] == "INVALID" for r in rewritten["records"]))
        with self.assertRaises(ValueError):
            self.dataset(rewritten, arm="corrected")

    def test_replacement_and_private_notes_do_not_enter_sft(self):
        original = self.backend
        def student(prompt, seed, maximum):
            reply = original.generate(prompt, seed, maximum)
            value = loads(reply.text)
            if '"runbook":' in prompt and value.get("kind") == "answer":
                return Reply(Action("answer", "UNSUPPORTED_GUESS").encode())
            return reply
        executor = Backend(original.identity, "fixture", student)
        def review(prompt, seed, maximum):
            value = loads(prompt)
            request, proposal = value["review"], loads(value["review"]["proposal"])
            result = dict(request_hash=value["request_hash"], decision="PASS", private_note="PRIVATE_REVIEW_NOTE")
            if proposal["value"] == "UNSUPPORTED_GUESS":
                tools = [loads(m["content"]) for m in request["view"]["messages"] if m["role"] == "tool"]
                result.update(decision="REPLACE", replacement=Action("answer", tools[-1]["result"]).encode(), reason="repair")
            return Reply(canonical(result))
        reviewer = Backend("correcting-reviewer", "fixture", review)
        rewritten = reconstruct(self.task, self.source, executor, original, original, Limits(), arm="corrected", reviewer=reviewer)
        validate_rewrite(rewritten, self.source, self.task)
        data = self.dataset(rewritten, arm="corrected")
        self.assertIn("UNSUPPORTED_GUESS", canonical(rewritten))
        self.assertNotIn("UNSUPPORTED_GUESS", canonical(data["rows"]))
        self.assertNotIn("PRIVATE_REVIEW_NOTE", canonical(data["rows"]))
