"""Offline unit, adversarial, replay, and end-to-end protocol qualification."""
from __future__ import annotations

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from experiments.rineya_harness_distillation_v1.contracts import (
    EMPTY, HARNESS_HASH, PROCEDURAL, Action, Limits, Reference, Review, ReviewRequest,
    Task, canonical, digest, loads, render, sampling_seed, validate_catalog)
from experiments.rineya_harness_distillation_v1.collection import Ledger, collect
from experiments.rineya_harness_distillation_v1.datasets import compile_dataset, replay, tokenize_sft
from experiments.rineya_harness_distillation_v1.environment import Environment, arithmetic
from experiments.rineya_harness_distillation_v1.evaluation import evaluate, paired_report
from experiments.rineya_harness_distillation_v1.fixtures import FixtureStudent, RuleReviewer, ScriptedExpert, make_tasks
from experiments.rineya_harness_distillation_v1.prlc_bridge import PRLCStudent, to_prlc_demonstrations
from experiments.rineya_harness_distillation_v1.storage import read_jsonl, verify_bundle, write_bundle
from experiments.rineya_harness_distillation_v1.cli import DEFAULT_CONFIG, demo, main, plan


class FixedStudent:
    identity = "test-fixed"
    def __init__(self, raw): self.raw = raw
    def propose(self, view, seed): return self.raw


class ByteTokenizer:
    eos_token_id = 0
    chat_template = None
    def encode(self, text, add_special_tokens=False):
        return ([999] if add_special_tokens else []) + [ord(c) + 1 for c in text]


def traced(task=None, *, mode="review_k", limits=None, student=None, reviewer=None):
    task, limits = task or make_tasks(1)[0], limits or Limits()
    return collect(task, student or FixtureStudent(), mode=mode, limits=limits,
                   ledger=Ledger(limits), reviewer=(reviewer or RuleReviewer()) if mode.startswith("review_") else None,
                   reference=PROCEDURAL if mode == "review_k" else EMPTY)


def resign(record):
    record.pop("trace_hash", None)
    record["trace_hash"] = digest(record)
    return record


class ContractTests(unittest.TestCase):
    def test_round_trip_task(self):
        task = make_tasks(2)[1]
        self.assertEqual(task, Task.from_dict(loads(canonical(asdict(task)))))

    def test_public_view_excludes_private_fields(self):
        task = make_tasks(2)[1]
        view = render(Environment(task).view())
        self.assertNotIn(task.expected, view)
        self.assertNotIn("expected", view)
        self.assertNotIn("records", view)
        self.assertNotIn("revision", view)

    def test_hidden_answer_cannot_change_initial_view(self):
        task = make_tasks(1)[0]
        self.assertEqual(Environment(task).view(), Environment(replace(task, expected="SECRET")).view())

    def test_reference_is_immutable(self):
        with self.assertRaises((AttributeError, TypeError)):
            PROCEDURAL.rules[0] = "new"

    def test_catalog_parent_overlap(self):
        task = make_tasks(1)[0]
        with self.assertRaises(ValueError):
            validate_catalog([task, replace(task, key="other", prompt="different", split="calibration")])

    def test_catalog_prompt_overlap(self):
        task = make_tasks(1)[0]
        with self.assertRaises(ValueError):
            validate_catalog([task, replace(task, key="other", parent="other", split="retention_eval", prompt=" COMPUTE 1 + 7. ")])

    def test_duplicate_key(self):
        with self.assertRaises(ValueError): validate_catalog(make_tasks(1) * 2)

    def test_group_variants_same_split(self):
        task = make_tasks(1)[0]
        self.assertEqual(len(validate_catalog([task, replace(task, key="variant", prompt="variant")])), 2)

    def test_duplicate_record_keys(self):
        with self.assertRaises(ValueError): replace(make_tasks(1)[0], records=(("a", "b"), ("a", "c")))

    def test_pass_cannot_rewrite(self):
        request = ReviewRequest(Environment(make_tasks(1)[0]).view(), Action("answer", "8").encode(), EMPTY, "s", 0)
        with self.assertRaises(ValueError): Review(request.fingerprint, "PASS", request.proposal).accept(request)

    def test_stale_review(self):
        request = ReviewRequest(Environment(make_tasks(1)[0]).view(), Action("answer", "8").encode(), EMPTY, "s", 0)
        with self.assertRaises(ValueError): Review("bad", "PASS").accept(request)

    def test_replacement_noop(self):
        request = ReviewRequest(Environment(make_tasks(1)[0]).view(), Action("answer", "8").encode(), EMPTY, "s", 0)
        with self.assertRaises(ValueError): Review(request.fingerprint, "REPLACE", request.proposal).accept(request)


BAD_ACTIONS = ["not-json", "{}", "[]", "null", '{"kind":"bash","value":"rm -rf /"}',
               '{"kind":"answer","value":3}', '{"kind":"answer","value":""}',
               '{"kind":"answer","value":"x","private_reasoning":"secret"}',
               '{"kind":"answer","kind":"check","value":"1+1"}',
               '{"kind":"answer","value":NaN}', '{"kind":[],"value":"x"}',
               '{"kind":"answer","value":"x"}\nPASS', Action("answer", "x").encode() * 100]
for i, raw in enumerate(BAD_ACTIONS):
    def test(self, raw=raw):
        with self.assertRaises((ValueError, TypeError)): Action.parse(raw)
    setattr(ContractTests, f"test_reject_action_{i:02d}", test)
for name, value in [("max_steps", 0), ("max_calls", True), ("max_response_chars", 2049),
                    ("max_context_chars", -1), ("max_wall_seconds", float("nan")),
                    ("max_wall_seconds", 0)]:
    def test(self, name=name, value=value):
        with self.assertRaises(ValueError): Limits(**{name: value})
    setattr(ContractTests, f"test_bad_limit_{name}_{str(value).replace('-', 'neg')}", test)


class EnvironmentTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(arithmetic("-2 * 3"), "-6")
        self.assertEqual(arithmetic("9 - -1"), "10")
        self.assertEqual(arithmetic("2+3"), "5")

    def test_no_python_execution(self):
        with self.assertRaises(ValueError): arithmetic("__import__('os').system('whoami')")
        with self.assertRaises(ValueError): arithmetic("2 ** 999999")
        with self.assertRaises(ValueError): arithmetic("1000000 + 1")

    def test_tool_error_is_observation(self):
        env = Environment(make_tasks(1)[0])
        self.assertFalse(loads(env.execute(Action("check", "oops").encode()))["ok"])
        self.assertFalse(env.success())

    def test_retrieval_is_explicit(self):
        task = make_tasks(2)[1]
        env = Environment(task)
        self.assertNotIn(task.expected, render(env.view()))
        env.execute(Action("retrieve", task.records[0][0]).encode())
        self.assertIn(task.expected, render(env.view()))

    def test_unknown_key_does_not_dump_store(self):
        env = Environment(make_tasks(2)[1])
        result = loads(env.execute(Action("retrieve", "../secrets").encode()))
        self.assertFalse(result["found"])
        self.assertIsNone(result["result"])

    def test_terminal_freeze(self):
        env = Environment(make_tasks(1)[0])
        env.execute(Action("answer", "8").encode())
        self.assertTrue(env.success())
        with self.assertRaises(ValueError): env.execute(Action("check", "1+1").encode())

    def test_abstention_grader(self):
        unknown = Environment(make_tasks(3)[2])
        unknown.execute(Action("abstain", "unknown").encode())
        self.assertTrue(unknown.success())
        known = Environment(make_tasks(1)[0])
        known.execute(Action("abstain", "unknown").encode())
        self.assertFalse(known.success())


class CollectionTests(unittest.TestCase):
    def test_attempt_sampling_seeds_are_independent(self):
        self.assertNotEqual(sampling_seed("t", 0, 17, 0), sampling_seed("t", 1, 17, 0))
        self.assertEqual(sampling_seed("t", 0, 17, 0), sampling_seed("t", 0, 17, 0))

    def test_collector_actually_uses_attempt_seed(self):
        class Capture(FixtureStudent):
            def propose(self, view, seed):
                self.seed = seed
                return super().propose(view, seed)
        student, task, limits = Capture(), make_tasks(1)[0], Limits()
        collect(task, student, mode="student_only", limits=limits, ledger=Ledger(limits), attempt=1, seed=17)
        self.assertEqual(student.seed, sampling_seed(task.key, 1, 17, 0))

    def test_reviewed_success_all_skills(self):
        for task in make_tasks(3):
            record = traced(task)
            self.assertTrue(record["success"])
            self.assertEqual(record["status"], "SUCCESS")
            self.assertEqual([s["review"]["decision"] for s in record["steps"]], ["REPLACE", "PASS"])

    def test_student_trajectory_contains_only_accepted_actions(self):
        record = traced()
        rows, _, _ = compile_dataset([record], make_tasks(1))
        self.assertNotIn("unverified", rows[1]["prompt"])
        self.assertNotIn("PRIVATE_REVIEW_SENTINEL", canonical(rows))
        self.assertIn("PRIVATE_REVIEW_SENTINEL", canonical(record))

    def test_pass_preserves_whitespace(self):
        raw = '{ "kind": "answer", "value": "8" }'
        record = traced(mode="review_empty", student=FixedStudent(raw))
        self.assertEqual(record["steps"][0]["accepted"], raw)

    def test_empty_reference_control(self):
        record = traced(mode="review_empty")
        self.assertFalse(record["success"])
        self.assertEqual(record["steps"][0]["review"]["decision"], "PASS")

    def test_teacher_protocol(self):
        self.assertTrue(traced(mode="teacher", student=ScriptedExpert())["success"])

    def test_invalid_pass_is_not_executed(self):
        record = traced(mode="review_empty", student=FixedStudent("broken"))
        self.assertEqual(record["status"], "INVALID")
        self.assertEqual(record["steps"], [])
        self.assertEqual(record["cost"]["reviewer_calls"], 1)

    def test_bad_proposal_can_be_repaired(self):
        record = traced(student=FixedStudent("broken"))
        self.assertTrue(record["success"])

    def test_call_failure_charged_and_redacted(self):
        class Broken(FixtureStudent):
            def propose(self, view, seed): raise RuntimeError("SECRET_KEY")
        record = traced(student=Broken())
        self.assertEqual(record["status"], "ERROR")
        self.assertEqual(record["cost"]["student_calls"], 1)
        self.assertNotIn("SECRET_KEY", canonical(record))

    def test_cap_exhaustion(self):
        record = traced(limits=Limits(max_calls=1))
        self.assertEqual(record["status"], "BUDGET")
        self.assertEqual(record["cost"]["student_calls"], 1)
        self.assertEqual(record["cost"]["reviewer_calls"], 0)

    def test_wall_limit_checked_before_call(self):
        now = [0.0]
        limits = Limits(max_wall_seconds=1)
        ledger = Ledger(limits, clock=lambda: now[0])
        now[0] = 2.0
        record = collect(make_tasks(1)[0], FixtureStudent(), mode="student_only", limits=limits, ledger=ledger)
        self.assertEqual(record["status"], "BUDGET")
        self.assertEqual(record["cost"]["student_calls"], 0)

    def test_context_limit_has_no_silent_truncation(self):
        record = traced(limits=Limits(max_context_chars=1))
        self.assertEqual(record["status"], "TRUNCATED")
        self.assertEqual(record["steps"], [])

    def test_step_limit_has_no_false_success(self):
        record = traced(limits=Limits(max_steps=1))
        self.assertEqual(record["status"], "TRUNCATED")
        self.assertFalse(record["success"])
        self.assertEqual(compile_dataset([record], make_tasks(1))[0], [])

    def test_oversized_response(self):
        self.assertEqual(traced(student=FixedStudent("x"*2049))["status"], "INVALID")

    def test_shared_ledger_does_not_reset(self):
        limits, task = Limits(max_calls=1), make_tasks(1)[0]
        ledger = Ledger(limits)
        collect(task, FixtureStudent(), mode="student_only", limits=limits, ledger=ledger)
        record = collect(task, FixtureStudent(), mode="student_only", limits=limits, ledger=ledger, attempt=1)
        self.assertEqual(record["status"], "BUDGET")

    def test_frozen_reference_and_arm(self):
        with self.assertRaises(ValueError):
            collect(make_tasks(1)[0], FixtureStudent(), mode="student_only", limits=Limits(),
                    ledger=Ledger(Limits()), reference=PROCEDURAL)


for split in ("calibration", "retention_train", "retention_eval", "sealed"):
    def test(self, split=split):
        with self.assertRaises(ValueError): traced(replace(make_tasks(1)[0], split=split))
    setattr(CollectionTests, "test_collection_rejects_" + split, test)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.task = make_tasks(1)[0]
        self.record = traced(self.task)

    def test_replay_targets_per_action(self):
        rows, pairs, success = replay(self.record, self.task)
        self.assertTrue(success)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(rows[0]["parent"], self.task.parent)
        self.assertFalse(pairs[0]["preference_eligible"])
        self.assertNotIn("private_note", canonical(rows))

    def test_filtering_and_interventions(self):
        rows, _, report = compile_dataset([self.record], [self.task])
        self.assertEqual(report["retained"], 1)
        self.assertEqual(report["intervention_rate_executed_reviews"], .5)
        self.assertEqual(report["sft_rows"], len(rows))

    def test_no_review_metric_is_null(self):
        report = compile_dataset([traced(mode="student_only")], [self.task])[2]
        self.assertIsNone(report["intervention_rate_executed_reviews"])

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError): compile_dataset([self.record]*2, [self.task])

    def test_arm_mixing_rejected(self):
        with self.assertRaises(ValueError): compile_dataset([self.record, traced(mode="student_only")], [self.task])

    def test_gold_drift_rejected(self):
        with self.assertRaises(ValueError): replay(self.record, replace(self.task, expected="9"))

    def test_integrity_hash_rejected(self):
        record = copy.deepcopy(self.record)
        record["success"] = False
        with self.assertRaises(ValueError): replay(record, self.task)

    def test_semantic_replay_after_rehash(self):
        record = copy.deepcopy(self.record)
        record["steps"][0]["observation"] = '{"kind":"check","ok":true,"result":"999"}'
        with self.assertRaises(ValueError): replay(resign(record), self.task)

    def test_success_flag_after_rehash(self):
        record = copy.deepcopy(self.record)
        record["success"] = False
        with self.assertRaises(ValueError): replay(resign(record), self.task)

    def test_source_drift_after_rehash(self):
        record = copy.deepcopy(self.record)
        record["source_hash"] = "new"
        with self.assertRaises(ValueError): replay(resign(record), self.task)

    def test_post_terminal_after_rehash(self):
        record = copy.deepcopy(self.record)
        record["steps"].append(record["steps"][-1])
        with self.assertRaises(ValueError): replay(resign(record), self.task)

    def test_unknown_reference_after_rehash(self):
        record = copy.deepcopy(self.record)
        record["reference_hash"] = "other"
        with self.assertRaises(ValueError): replay(resign(record), self.task)

    def test_stale_review_after_rehash(self):
        record = copy.deepcopy(self.record)
        record["steps"][0]["review"]["request_hash"] = "stale"
        with self.assertRaises(ValueError): replay(resign(record), self.task)

    def test_evaluation_trace_cannot_export(self):
        task = replace(self.task, split="calibration")
        record = collect(task, ScriptedExpert(), mode="student_only", limits=Limits(), ledger=Ledger(Limits()), evaluation=True)
        with self.assertRaises(ValueError): compile_dataset([record], [task])

    def test_token_labels(self):
        row = replay(self.record, self.task)[0][1]
        tokenized = tokenize_sft(row, ByteTokenizer(), max_prompt_tokens=5000, max_context_tokens=6000)
        n = tokenized["prompt_tokens"]
        self.assertEqual(tokenized["labels"][:n], [-100]*n)
        self.assertEqual(tokenized["labels"][n:], tokenized["input_ids"][n:])
        self.assertEqual(tokenized["labels"][-1], 0)
        self.assertEqual(tokenized["supervised_tokens"], len(row["completion"]) + 1)

    def test_prompt_and_context_overflow(self):
        row = replay(self.record, self.task)[0][0]
        with self.assertRaises(ValueError): tokenize_sft(row, ByteTokenizer(), max_prompt_tokens=1, max_context_tokens=6000)
        with self.assertRaises(ValueError): tokenize_sft(row, ByteTokenizer(), max_prompt_tokens=5000, max_context_tokens=1)

    def test_embedded_eos_rejected(self):
        class Bad(ByteTokenizer):
            def encode(self, text, add_special_tokens=False): return [1, 0, 2]
        with self.assertRaises(ValueError): tokenize_sft(replay(self.record, self.task)[0][0], Bad(), max_prompt_tokens=9, max_context_tokens=99)

    def test_chat_template_branch(self):
        class Chat(ByteTokenizer):
            chat_template = "fixture"
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.messages = messages
                return [4, 5, 6]
        tokenizer = Chat()
        row = replay(self.record, self.task)[0][0]
        encoded = tokenize_sft(row, tokenizer, max_prompt_tokens=9, max_context_tokens=9000)
        self.assertEqual(encoded["prompt_tokens"], 3)
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": row["prompt"]}])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tasks = make_tasks(6, split="calibration", offset=500)
        self.panel = digest(sorted((t.key, t.fingerprint) for t in self.tasks))
        self.base = evaluate(self.tasks, FixtureStudent(), Limits(), panel_hash=self.panel)
        self.expert = evaluate(self.tasks, ScriptedExpert(), Limits(), panel_hash=self.panel)

    def test_review_free_report(self):
        self.assertTrue(all(r["reviewer_calls"] == 0 for r in self.expert))
        report = paired_report(self.base, self.expert)
        self.assertEqual(report["parent_mean_delta"], 1)
        self.assertFalse(report["confirmatory"])

    def test_deterministic_bootstrap(self):
        self.assertEqual(paired_report(self.base, self.expert), paired_report(self.base, self.expert))

    def test_sealed_forbidden(self):
        with self.assertRaises(ValueError): evaluate(make_tasks(3, split="sealed"), FixtureStudent(), Limits(), panel_hash="x")

    def test_panel_drift(self):
        with self.assertRaises(ValueError): evaluate(self.tasks, FixtureStudent(), Limits(), panel_hash="x")

    def test_missing_coverage(self):
        with self.assertRaises(ValueError): paired_report(self.base, self.expert[:-1])

    def test_duplicate_eval(self):
        with self.assertRaises(ValueError): paired_report(self.base, self.expert + self.expert[:1])

    def test_different_budgets(self):
        candidate = copy.deepcopy(self.expert)
        for row in candidate: row["limits_hash"] = "changed"
        with self.assertRaises(ValueError): paired_report(self.base, candidate)

    def test_mixed_panels(self):
        candidate = copy.deepcopy(self.expert)
        candidate[0]["panel_hash"] = "changed"
        with self.assertRaises(ValueError): paired_report(self.base, candidate)

    def test_review_contamination(self):
        candidate = copy.deepcopy(self.expert)
        candidate[0]["reviewer_calls"] = 1
        with self.assertRaises(ValueError): paired_report(self.base, candidate)

    def test_retention_sources_not_hidden(self):
        base, candidate = copy.deepcopy(self.base), copy.deepcopy(self.expert)
        for rows in (base, candidate):
            rows[0]["source"] = "fragile_retention"
        base[0]["success"], candidate[0]["success"] = True, False
        result = paired_report(base, candidate)
        self.assertEqual(result["sources"]["fragile_retention"]["delta"], -1)


class BridgeTests(unittest.TestCase):
    def test_transport_uses_actual_parent_task(self):
        from experiments.progressive_rl_curriculum_v1.schema import Task as ParentTask
        rows = compile_dataset([traced()], make_tasks(1))[0]
        pairs = to_prlc_demonstrations(rows)
        self.assertTrue(all(isinstance(task, ParentTask) for task, _ in pairs))
        self.assertEqual(pairs[0][0].prompt, rows[0]["prompt"])
        self.assertEqual(pairs[0][0].answer, rows[0]["completion"])
        self.assertNotIn("PRIVATE_REVIEW_SENTINEL", pairs[0][0].prompt)

    def test_native_call_not_authorized_by_default(self):
        with self.assertRaises(ValueError): PRLCStudent(object(), max_new_tokens=32)

    def test_wrapper_uses_actual_parent_prompt(self):
        from experiments.progressive_rl_curriculum_v1.schema import Prompt, Episode
        class Backend:
            version, identity = 0, {"kind": "fixture"}
            def collect(self, prompt, count, cap, seed):
                self.prompt = prompt
                return [Episode(prompt.key, Action("answer", "8").encode(), (1,), (2, 0), (-.2, -.3), 0)]
        backend = Backend()
        student = PRLCStudent(backend, max_new_tokens=32, allow_model_calls=True)
        view = Environment(make_tasks(1)[0]).view()
        self.assertEqual(Action.parse(student.propose(view, 1)).value, "8")
        self.assertIsInstance(backend.prompt, Prompt)
        self.assertEqual(backend.prompt.text, render(view))
        backend.version = 1
        with self.assertRaises(ValueError): student.propose(view, 1)

    def test_wrapper_rejects_incomplete(self):
        backend = SimpleNamespace(version=0, identity={"kind": "fixture"})
        backend.collect = lambda p, *args: [SimpleNamespace(key=p.key, policy_version=0, terminated=False)]
        with self.assertRaises(ValueError):
            PRLCStudent(backend, max_new_tokens=32, allow_model_calls=True).propose(Environment(make_tasks(1)[0]).view(), 0)

    def test_transport_rejects_duplicate_rows(self):
        rows = compile_dataset([traced()], make_tasks(1))[0]
        with self.assertRaises(ValueError): to_prlc_demonstrations(rows + rows)


class StorageAndCliTests(unittest.TestCase):
    def test_bundle_integrity_no_clobber(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            write_bundle(out, {"data.jsonl": [{"x": 1}]}, evidence="fixture")
            verify_bundle(out)
            with self.assertRaises(FileExistsError): write_bundle(out, {"data.jsonl": []}, evidence="fixture")
            (out / "data.jsonl").write_text("corrupt")
            with self.assertRaises(ValueError): verify_bundle(out)

    def test_safe_names(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): write_bundle(Path(d)/"out", {"../secret": []}, evidence="fixture")

    def test_jsonl_strict_and_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"x.jsonl"
            for raw in ('\n', '[]\n', '{"x":NaN}\n', '{"x":1,"x":2}\n'):
                p.write_text(raw)
                with self.assertRaises(ValueError): read_jsonl(p)
            p.write_text('{"x":1}\n')
            with self.assertRaises(ValueError): read_jsonl(p, max_bytes=1)
            self.assertEqual(read_jsonl(p), [{"x": 1}])

    def test_plan_is_not_authorization(self):
        result = plan(DEFAULT_CONFIG)
        self.assertEqual(result["native_status"], "BLOCKED_NOT_QUALIFIED")
        self.assertFalse(result["plan"]["native_training_authorized"])

    def test_demo_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/"demo"
            report = demo(out, count=6, attempts=2)
            self.assertEqual(report["collection"]["review_k"]["sft_rows"], 24)
            self.assertEqual(report["optimizer_steps"], 0)
            verify_bundle(out)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["compile", "--catalog", str(out/"catalog.jsonl"),
                    "--trajectories", str(out/"trajectories.private.jsonl"), "--mode", "review_k",
                    "--out", str(Path(d)/"compiled")]), 0)
            verify_bundle(Path(d)/"compiled")

    def test_native_train_command_absent(self):
        from contextlib import redirect_stderr
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): main(["train"])


if __name__ == "__main__":
    unittest.main()
