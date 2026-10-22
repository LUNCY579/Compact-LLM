from __future__ import annotations

from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.rineya_verified_resampling_v1 import contracts as C
from experiments.rineya_verified_resampling_v1.datasets import (prepare_fixture, prepare_centralized,
                                                               freeze_study, load_study)
from experiments.rineya_verified_resampling_v1.evaluation import evaluate, paired_interval
from experiments.rineya_verified_resampling_v1.runner import FixtureBackend, Journal, collect, slots, validate_bank
from experiments.rineya_verified_resampling_v1.selection import normalized, pass_at_k, public_candidate, select


class Workspace(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = C.read_json(C.PACKAGE / "configs/titan_pilot.json")
        self.config["bootstrap_replicates"] = 100
        C.write_new(self.root / "config.json", self.config)
        prepare_fixture(self.root / "data", 16)
        freeze_study(self.root / "data", self.root / "config.json", self.root / "study")
        self.study, self.public = load_study(self.root / "study")

    def tearDown(self):
        self.temp.cleanup()

    def run_fixture(self, scenario="minority", out="run", **kw):
        return collect(self.study, self.public, FixtureBackend(scenario), self.root / out, **kw)

    def report(self, out="report"):
        return evaluate(self.root / "study", self.root / "run", self.root / "data/oracle.private.jsonl", self.root / out)


class ContractTests(unittest.TestCase):
    def test_json_duplicate_rejected(self):
        with self.assertRaises(C.Blocked): C.loads('{"x":1,"x":2}')

    def test_nonfinite_rejected(self):
        for text in ['{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}']:
            with self.subTest(text=text), self.assertRaises(C.Blocked): C.loads(text)

    def test_answer_empty_values(self):
        self.assertEqual(C.parse_answer('{"answer":""}'), '{"answer":""}')
        self.assertEqual(C.parse_answer('{"answer":[]}'), '{"answer":[]}')

    def test_answer_canonical_whitespace(self):
        self.assertEqual(C.parse_answer(' { "answer": [ 1, -2 ] }\n'), '{"answer":[1,-2]}')

    def test_answer_types(self):
        for text in ['{"answer":true}', '{"answer":1.0}', '{"answer":null}', '{"answer":{}}', '{"answer":[false]}']:
            with self.subTest(text=text): self.assertIsNone(C.parse_answer(text))

    def test_answer_extra_fields_and_rationale(self):
        for text in ['{"answer":1,"reason":"x"}', 'Reason: {"answer":1}', '{"answer":1}\n{"answer":1}']:
            self.assertIsNone(C.parse_answer(text))

    def test_oversize_answer(self):
        self.assertIsNone(C.parse_answer('{"answer":"'+'x'*5000+'"}'))

    def test_seed_replay_and_separation(self):
        a = C.seed_for(1, "abc", "sample", 1)
        self.assertEqual(a, C.seed_for(1, "abc", "sample", 1))
        self.assertNotEqual(a, C.seed_for(1, "abc", "sample", 2))
        self.assertNotEqual(a, C.seed_for(1, "abc", "greedy", 1))
        self.assertNotEqual(a, C.seed_for(1, "def", "sample", 1))

    def test_public_rejects_gold(self):
        row = {"id": "a"*64, "messages": [{"role": "user", "content": "hi"}], "answer": 1}
        with self.assertRaises(C.Blocked): C.validate_public([row])

    def test_public_duplicate_prompt(self):
        rows = [{"id": x*64, "messages": [{"role": "user", "content": "hi"}]} for x in "ab"]
        with self.assertRaises(C.Blocked): C.validate_public(rows)

    def test_config_unknown_keys(self):
        config = C.read_json(C.PACKAGE / "configs/titan_pilot.json")
        config["allow_hidden_answers"] = True
        with self.assertRaises(C.Blocked): C.validate_config(config)

    def test_config_temperature_and_context(self):
        config = C.read_json(C.PACKAGE / "configs/titan_pilot.json")
        for t in [0, -1, True, float("nan"), 3]:
            with self.subTest(t=t), self.assertRaises(C.Blocked): C.validate_config({**config, "temperature": t})
        with self.assertRaises(C.Blocked): C.validate_config({**config, "max_context_tokens": 512})

    def test_config_budgets(self):
        config = C.read_json(C.PACKAGE / "configs/titan_pilot.json")
        for budgets in [[], [1, 16, 8], [1, 8], [True, 16], [1, 1, 16]]:
            with self.subTest(budgets=budgets), self.assertRaises(C.Blocked):
                C.validate_config({**config, "budgets": budgets})

    def test_write_once(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            C.write_new(p, {"a": 1})
            with self.assertRaises(FileExistsError): C.write_new(p, {"a": 2})
            self.assertEqual(C.read_json(p), {"a": 1})


class SelectorTests(unittest.TestCase):
    @staticmethod
    def c(slot, answer, stop="eos", state="returned"):
        return {"slot": slot, "text": json.dumps({"answer": answer}), "stop": stop, "state": state}

    def test_oracle_minority_is_not_vote(self):
        candidates = [self.c(0, 7), self.c(1, 9), self.c(2, 9)]
        self.assertEqual(select(candidates, "vote", 1), 1)

    def test_tie_is_earliest(self):
        self.assertEqual(select([self.c(8, 3), self.c(9, 4)], "vote", 1), 8)

    def test_valid_empty_value_not_dropped(self):
        self.assertEqual(select([self.c(1, ""), self.c(2, "")], "vote", 1), 1)
        self.assertEqual(select([self.c(3, [])], "vote", 1), 3)

    def test_no_eos_is_not_eligible(self):
        self.assertIsNone(select([self.c(1, 8, "length")], "vote", 1))

    def test_failed_slot_stays_in_random_pool(self):
        c = [self.c(0, 7, state="failed")]
        self.assertEqual(select(c, "random", 1), 0)
        self.assertIsNone(select(c, "vote", 1))

    def test_selector_rejects_gold_field(self):
        c = self.c(0, 7)
        c["correct"] = True
        for policy in ["vote", "first", "random"]:
            with self.assertRaises(C.Blocked): select([c], policy, 1)

    def test_projection_drops_all_private_fields(self):
        c = self.c(0, 7)
        c.update(correct=True, expected="7", family="hidden")
        self.assertEqual(set(public_candidate(c)), {"slot", "text", "stop", "state"})

    def test_random_replay(self):
        c = [self.c(i, i) for i in range(16)]
        self.assertEqual(select(c, "random", 42), select(c, "random", 42))

    def test_duplicate_slots_rejected(self):
        with self.assertRaises(C.Blocked): select([self.c(0, 1), self.c(0, 2)], "vote", 1)

    def test_pass_estimator_matches_enumeration(self):
        for n in range(1, 8):
            for c in range(n+1):
                labels = [1]*c + [0]*(n-c)
                for k in range(1, n+1):
                    truth = sum(any(labels[i] for i in group) for group in itertools.combinations(range(n), k))/math.comb(n, k)
                    self.assertAlmostEqual(pass_at_k(n, c, k), truth)

    def test_pass_invalid_counts(self):
        for values in [(0, 0, 1), (4, -1, 1), (4, 5, 1), (4, 2, 0), (4, 2, 5), (4, True, 1)]:
            with self.assertRaises(C.Blocked): pass_at_k(*values)

    def test_pass_endpoints(self):
        self.assertEqual(pass_at_k(16, 0, 16), 0)
        self.assertEqual(pass_at_k(16, 1, 16), 1)
        self.assertEqual(pass_at_k(16, 16, 1), 1)


class DatasetTests(Workspace):
    def test_no_oracle_in_study(self):
        self.assertFalse((self.root / "study/oracle.private.jsonl").exists())
        for row in self.public: self.assertEqual(set(row), {"id", "messages"})

    def test_population_count(self):
        self.assertEqual(self.study["slots"], 16*17)

    def test_changed_public_is_blocked(self):
        with (self.root / "study/public.jsonl").open("ab") as f: f.write(b"\n")
        with self.assertRaises(C.Blocked): load_study(self.root / "study")

    def test_config_frozen_by_digest(self):
        p = self.root / "study/study.json"
        s = C.read_json(p)
        s["config"]["temperature"] = 0.9
        p.write_bytes(C.encoded(s))
        with self.assertRaises(C.Blocked): load_study(self.root / "study")

    def test_insufficient_token_budget(self):
        C.write_new(self.root / "small.json", {**self.config, "max_generated_tokens": 2})
        with self.assertRaises(C.Blocked): freeze_study(self.root / "data", self.root / "small.json", self.root / "small")

    def test_protected_roles_rejected_before_import(self):
        for role in ["TRAIN", "CHECK", "FINAL", "sealed"]:
            with self.assertRaises(C.Blocked): prepare_centralized(self.root / role, role=role)

    def test_dataset_write_once(self):
        with self.assertRaises(FileExistsError): prepare_fixture(self.root / "data", 16)


class RunnerTests(Workspace):
    def test_complete_fixture(self):
        r = self.run_fixture()
        self.assertEqual((r["status"], r["returned"], r["failed"]), ("COMPLETE", 272, 0))

    def test_resume_matches_uninterrupted_bytes(self):
        self.run_fixture(out="full")
        self.run_fixture(out="part", max_attempts=19)
        self.run_fixture(out="part", resume=True)
        self.assertEqual((self.root / "full/events.jsonl").read_bytes(), (self.root / "part/events.jsonl").read_bytes())

    def test_resume_requires_explicit_flag(self):
        self.run_fixture(max_attempts=1)
        with self.assertRaises(C.Blocked): self.run_fixture()

    def test_backend_drift_blocks_resume(self):
        self.run_fixture(max_attempts=1)
        with self.assertRaises(C.Blocked): self.run_fixture("flat", resume=True)

    def test_journal_tamper(self):
        self.run_fixture(max_attempts=1)
        p = self.root / "run/events.jsonl"
        p.write_bytes(p.read_bytes().replace(b'"raw_lp":-0.7', b'"raw_lp":-0.8', 1))
        with self.assertRaises(C.Blocked): Journal(p, self.config["max_output_bytes"])

    def test_torn_journal_rejected(self):
        self.run_fixture(max_attempts=1)
        p = self.root / "run/events.jsonl"
        p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaises(C.Blocked): Journal(p, self.config["max_output_bytes"])

    def test_failed_attempt_not_redrawn(self):
        class Fail(FixtureBackend):
            def generate(self, task, seed, greedy, config, emit):
                emit(12, -1, -1)
                raise RuntimeError("intentional fixture failure")
        result = collect(self.study, self.public, Fail(), self.root / "run", max_attempts=2)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["durably_recorded_tokens"], 1)
        self.run_fixture(resume=True)
        journal = Journal(self.root / "run/events.jsonl", self.config["max_output_bytes"])
        self.assertEqual(journal.records[0]["state"], "failed")
        self.assertEqual(len(journal.records), 272)

    def test_pending_requires_explicit_abandonment(self):
        self.run_fixture(max_attempts=1)
        j = Journal(self.root / "run/events.jsonl", self.config["max_output_bytes"])
        j.append("begin", slots(self.study, self.public)[1])
        j.append("token", {"slot": 1, "id": 7, "raw_lp": -1, "policy_lp": -1})
        with self.assertRaises(C.Blocked): self.run_fixture(resume=True)
        self.run_fixture(resume=True, abandon_interrupted=True)
        j = Journal(self.root / "run/events.jsonl", self.config["max_output_bytes"])
        self.assertEqual(j.records[1]["state"], "failed")
        self.assertEqual(j.records[1]["token_ids"], [7])

    def test_bad_seed_cannot_replay(self):
        self.run_fixture(max_attempts=1)
        j = Journal(self.root / "run/events.jsonl", self.config["max_output_bytes"])
        j.records[0]["seed"] += 1
        with self.assertRaises(C.Blocked): validate_bank(j, slots(self.study, self.public), self.config)

    def test_token_budget_stops_faulty_backend(self):
        class TooMany(FixtureBackend):
            def generate(self, task, seed, greedy, config, emit):
                for _ in range(129): emit(1, -1, -1)
        result = collect(self.study, self.public, TooMany(), self.root / "run")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["durably_recorded_tokens"], 128)

    def test_duplicate_terminal_rejected(self):
        self.run_fixture(max_attempts=1)
        j = Journal(self.root / "run/events.jsonl", self.config["max_output_bytes"])
        with self.assertRaises(C.Blocked): j.append("failed", {"slot": 0, "reason": "x", "uncertain_work": True})

    def test_disk_budget_rejects_before_write(self):
        p = self.root / "x.jsonl"
        j = Journal(p, 4)
        with self.assertRaises(C.Blocked): j.append("begin", slots(self.study, self.public)[0])
        self.assertFalse(p.exists())

    def test_invalid_logprob_rejected(self):
        j = Journal(self.root / "x.jsonl", 2**20)
        j.append("begin", slots(self.study, self.public)[0])
        with self.assertRaises(C.Blocked): j.append("token", {"slot": 0, "id": 1, "raw_lp": 1, "policy_lp": -1})


class EvaluationTests(Workspace):
    def test_coverage_is_not_selection(self):
        self.run_fixture()
        r = self.report()
        self.assertEqual(r["evidence"], "SYNTHETIC_ONLY")
        self.assertGreater(r["curves"][-1]["prefix_coverage_lower"], r["curves"][-1]["vote_accuracy"])
        self.assertEqual(r["status"], "COMPLETE_BANK")

    def test_all_wrong_has_null_recovery(self):
        self.run_fixture("flat")
        r = self.report()
        self.assertIsNone(r["curves"][-1]["vote_conditional_recovery"])
        self.assertEqual(r["inference"], "NO_HEADROOM_OBSERVED_WITHIN_BUDGET")

    def test_all_correct_has_zero_gap(self):
        self.run_fixture("all_correct")
        r = self.report()
        for c in r["curves"]:
            self.assertEqual(c["vote_accuracy"], 1)
            self.assertEqual(c["matched_selection_gap"], 0)
            self.assertEqual(c["random_subset_pass_estimate"], 1)

    def test_useful_vote_control(self):
        self.run_fixture("vote_gain")
        r = self.report()
        self.assertGreater(r["curves"][-1]["vote_accuracy"], r["curves"][-1]["first_sample_accuracy"])

    def test_incomplete_not_promoted(self):
        self.run_fixture(max_attempts=4)
        r = self.report()
        self.assertEqual(r["status"], "INCOMPLETE_BANK")
        self.assertEqual(r["inference"], "INCOMPLETE_DO_NOT_DECIDE")
        self.assertIsNone(r["curves"][-1]["random_subset_pass_estimate"])
        self.assertEqual(r["curves"][-1]["prefix_coverage_upper"], 1)

    def test_oracle_mismatch_rejected(self):
        self.run_fixture()
        oracle = self.root / "data/oracle.private.jsonl"
        oracle.write_bytes(oracle.read_bytes()+b"\n")
        with self.assertRaises(C.Blocked): self.report()

    def test_report_files_present(self):
        self.run_fixture()
        self.report()
        self.assertEqual({p.name for p in (self.root / "report").iterdir()},
                         {"report.json", "REPORT.md", "curves.csv", "cases.private.jsonl"})

    def test_bootstrap_cluster_bounds(self):
        rows = [{"parent": "a", "t": 1, "b": 0}]*3 + [{"parent": "b", "t": 0, "b": 0}]*3
        self.assertEqual(paired_interval(rows, "t", "b", 1000, 1), [0, 1])
        self.assertIsNone(paired_interval(rows, "t", "b", 0, 1))

    def test_report_does_not_include_gold_text(self):
        self.run_fixture()
        r = self.report()
        self.assertNotIn("expected", json.dumps(r))


class NativeEntryTests(Workspace):
    def test_missing_permission_blocks_before_native_import(self):
        from experiments.rineya_verified_resampling_v1.driver import native_launch
        with self.assertRaises(C.Blocked):
            native_launch("qualify", self.root / "study", self.root / "no-model", self.root / "native")
        self.assertFalse((self.root / "native").exists())

    def test_nonwindows_native_is_blocked(self):
        from experiments.rineya_verified_resampling_v1.driver import native_launch
        with patch("experiments.rineya_verified_resampling_v1.driver.os.name", "posix"):
            with self.assertRaises(C.Blocked):
                native_launch("qualify", self.root / "study", self.root / "no-model", self.root / "native", allow_model_load=True)

    def test_no_optimizer_code(self):
        for name in ("native.py", "torch_sampler.py", "driver.py", "runner.py"):
            text = (C.PACKAGE / name).read_text()
            for forbidden in ("torch.optim", ".backward(", "eval(response", "exec(response"):
                self.assertNotIn(forbidden, text)

    def test_cli_demo_subprocess(self):
        result = subprocess.run([sys.executable, "-B", "-m", "experiments.rineya_verified_resampling_v1",
                                 "demo", "--out", str(self.root / "cli"), "--parents", "4"], cwd=C.ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["evidence"], "SYNTHETIC_ONLY")


@unittest.skipUnless((C.ROOT / C.BRIDGE).exists(), "inherited generator is available in full GitHub checkout, not this isolated local workspace")
class CentralizedIntegrationTests(unittest.TestCase):
    def test_actual_generator_and_independent_verifier(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "data"
            manifest = prepare_centralized(out, per_family=2)
            self.assertEqual(manifest["rows"], 8)
            self.assertEqual(set(manifest["family_counts"].values()), {2})
            self.assertEqual(manifest["kind"], "centralized_bridge")
            self.assertEqual(manifest["CHECK"], "CLOSED")
            private = C.read_jsonl(out / "oracle.private.jsonl")
            self.assertTrue(all(r["role"] == "DEV" for r in private))

    def test_calibration_parents_disjoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prepare_centralized(root / "dev", per_family=2)
            prepare_centralized(root / "cal", per_family=2, role="TRAIN_SELECTION")
            a = {r["parent"] for r in C.read_jsonl(root / "dev/oracle.private.jsonl")}
            b = {r["parent"] for r in C.read_jsonl(root / "cal/oracle.private.jsonl")}
            self.assertFalse(a & b)


if __name__ == "__main__":
    unittest.main()
