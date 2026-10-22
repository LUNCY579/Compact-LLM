import copy
import unittest
from dataclasses import replace

from experiments.rineya_recursive_self_rewrite_v1.core import *
from experiments.rineya_recursive_self_rewrite_v1.fixtures import catalog, fixture_backend
from experiments.rineya_recursive_self_rewrite_v1.runtime import execute, audit, supported, harness_hint
from experiments.rineya_harness_distillation_v1.contracts import Message


def rehash(record):
    record["record_hash"] = digest({k: v for k, v in record.items() if k != "record_hash"})
    return record


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tasks, self.ancestry = catalog(3)
        self.backend = fixture_backend()
        self.task = self.tasks[1]

    def run_task(self, **kwargs):
        return execute(self.task, self.backend, Ledger(Limits()), **kwargs)

    def test_pinned_upstream(self):
        self.assertEqual(len(source_hash()), 64)

    def test_catalog_ancestry_required(self):
        with self.assertRaises(ValueError):
            enroll(self.tasks, {})

    def test_cross_split_ancestor_rejected(self):
        tasks = [self.tasks[0], replace(self.task, split="calibration")]
        with self.assertRaises(ValueError):
            enroll(tasks, {t.key: ["shared-original"] for t in tasks})

    def test_cross_split_prompt_rejected(self):
        task = replace(self.task, key="duplicate", parent="other", split="calibration")
        with self.assertRaises(ValueError):
            validate_catalog([self.task, task])

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            validate_catalog([self.task, self.task])

    def test_strict_action_parsing(self):
        bad = ['{"kind":"answer","kind":"retrieve","value":"x"}',
               '{"kind":"shell","value":"ls"}', '{"kind":"answer","value":NaN}',
               '{"kind":"answer","value":"x","extra":1}']
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                Action.parse(raw)

    def test_runbook_roundtrip(self):
        book = Runbook(tuple(STEPS))
        self.assertEqual(Runbook.parse(book.encode()), book)

    def test_literal_runbook_rejected(self):
        for raw in ['{"goal":"evidence_backed_completion","steps":["answer 42"]}',
                    '{"goal":"evidence_backed_completion","steps":["answer_observed"],"answer":42}',
                    '{"goal":"solve","steps":["answer_observed"]}']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                Runbook.parse(raw)

    def test_duplicate_runbook_operation(self):
        with self.assertRaises(ValueError):
            Runbook(("answer_observed", "answer_observed"))

    def test_bad_limit_types(self):
        for kwargs in ({"calls": True}, {"steps": 0}, {"tokens": -1}, {"wall_seconds": float("nan")},
                       {"response_chars": 4096}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Limits(**kwargs)

    def test_native_requires_counter(self):
        with self.assertRaises(ValueError):
            Backend("native", "native", lambda *args: Reply("x"))

    def test_native_requires_explicit_token_cap(self):
        backend = Backend("native", "native", lambda *args: Reply("x", 1, 1), lambda _: 1)
        with self.assertRaises(ValueError):
            Ledger(Limits()).call(backend, "executor", "x", 1)

    def test_token_budget_reservation(self):
        calls = []
        backend = Backend("native", "native", lambda *args: calls.append(1), lambda _: 10)
        with self.assertRaises(BudgetExceeded):
            Ledger(Limits(tokens=20, new_tokens=16)).call(backend, "executor", "x", 1)
        self.assertFalse(calls)

    def test_native_exact_token_accounting(self):
        backend = Backend("native", "native", lambda *args: Reply("x", 10, 2), lambda _: 10)
        ledger = Ledger(Limits(tokens=100, new_tokens=10))
        ledger.call(backend, "executor", "x", 1)
        self.assertEqual(ledger.report()["tokens"], 12)

    def test_failed_native_call_reserves_cost(self):
        def fail(*args):
            raise RuntimeError("private message")
        backend = Backend("native", "native", fail, lambda _: 10)
        ledger = Ledger(Limits(tokens=100, new_tokens=10))
        with self.assertRaises(RuntimeError):
            ledger.call(backend, "executor", "x", 1)
        self.assertEqual(ledger.used["tokens"], 20)
        self.assertNotIn("private message", canonical(ledger.events))

    def test_context_never_truncated(self):
        ledger = Ledger(Limits(context_chars=1))
        with self.assertRaises(BudgetExceeded):
            ledger.call(self.backend, "executor", "long prompt", 1)
        self.assertEqual(ledger.used["calls"], 0)

    def test_successful_replay(self):
        record = self.run_task()
        self.assertTrue(audit(record, self.task)["eligible"])
        self.assertIsNone(record["usage"]["tokens"])

    def test_fresh_environment_and_sandbox(self):
        a, b = self.run_task(), self.run_task()
        self.assertNotEqual(a["sandbox"], b["sandbox"])
        self.assertEqual(a["steps"][0]["before"], b["steps"][0]["before"])

    def test_observation_tampering_rejected_after_rehash(self):
        row = self.run_task()
        row["steps"][0]["observation"] = "{}"
        with self.assertRaises(ValueError):
            audit(rehash(row), self.task)

    def test_forged_success_rejected(self):
        row = self.run_task()
        row["success"] = False
        with self.assertRaises(ValueError):
            audit(rehash(row), self.task)

    def test_prompt_tampering_rejected(self):
        row = self.run_task()
        row["steps"][0]["student_prompt"] += "hidden gold"
        with self.assertRaises(ValueError):
            audit(rehash(row), self.task)

    def test_task_binding(self):
        with self.assertRaises(ValueError):
            audit(self.run_task(), replace(self.task, expected="wrong"))

    def test_code_drift_rejected(self):
        row = self.run_task()
        row["code_hash"] = "stale"
        with self.assertRaises(ValueError):
            audit(rehash(row), self.task)

    def test_correct_guess_not_training_eligible(self):
        backend = Backend("guess", "fixture", lambda *args: Reply(Action("answer", self.task.expected).encode()))
        row = execute(self.task, backend, Ledger(Limits()))
        self.assertTrue(row["success"])
        self.assertFalse(audit(row, self.task)["eligible"])

    def test_hidden_record_key_unsupported(self):
        view = Environment(self.task).view()
        self.assertFalse(supported(Action("retrieve", "secret_key"), view))

    def test_substring_record_key_unsupported(self):
        view = Environment(self.task).view()
        self.assertFalse(supported(Action("retrieve", "record"), view))

    def test_future_arithmetic_unsupported(self):
        self.assertFalse(supported(Action("check", self.task.records[0][1]), Environment(self.task).view()))

    def test_abstention_requires_missing_evidence(self):
        self.assertFalse(supported(Action("abstain", "unknown"), Environment(self.task).view()))

    def test_sealed_cannot_execute(self):
        with self.assertRaises(ValueError):
            execute(replace(self.task, split="sealed"), self.backend, Ledger(Limits()))

    def test_retention_train_cannot_collect(self):
        with self.assertRaises(ValueError):
            execute(replace(self.task, split="retention_train"), self.backend, Ledger(Limits()))

    def test_evaluation_has_no_guidance(self):
        task = replace(self.task, split="calibration")
        with self.assertRaises(ValueError):
            execute(task, self.backend, Ledger(Limits()), arm="evaluation", harness="H1")

    def test_truncation_is_failure(self):
        row = execute(self.task, self.backend, Ledger(Limits(steps=1)))
        self.assertEqual(row["status"], "TRUNCATED")
        self.assertFalse(audit(row, self.task)["eligible"])

    def test_tool_budget_stops_before_execution(self):
        row = execute(self.task, self.backend, Ledger(Limits(tools=1)))
        self.assertEqual(row["status"], "BUDGET")
        self.assertEqual(len(row["steps"]), 1)

    def test_invalid_actions_count(self):
        bad = Backend("bad", "fixture", lambda *args: Reply("not-json"))
        row = execute(self.task, bad, Ledger(Limits()))
        self.assertEqual(row["status"], "INVALID")
        self.assertEqual(row["usage"]["calls"], 1)

    def test_callback_errors_count(self):
        def fail(*args):
            raise RuntimeError("secret detail")
        row = execute(self.task, Backend("bad", "fixture", fail), Ledger(Limits()))
        self.assertEqual(row["status"], "ERROR")
        self.assertEqual(row["usage"]["calls"], 1)
        self.assertNotIn("secret detail", canonical(row))

    def test_oversized_response_rejected(self):
        bad = Backend("bad", "fixture", lambda *args: Reply("x" * 2049))
        row = execute(self.task, bad, Ledger(Limits()))
        self.assertEqual(row["status"], "INVALID")

    def test_hints_do_not_read_gold(self):
        original = Environment(self.task).view()
        changed = Environment(replace(self.task, expected="secret", records=())).view()
        for harness in ("H0", "H1", "H2"):
            self.assertEqual(harness_hint(harness, original), harness_hint(harness, changed))

    def test_all_fixture_skills_execute(self):
        for task in self.tasks:
            with self.subTest(skill=task.skill):
                self.assertTrue(audit(execute(task, self.backend, Ledger(Limits())), task)["eligible"])
