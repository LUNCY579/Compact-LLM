"""Mechanical and adversarial tests; fixtures are never native efficacy evidence."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rineya_generalization_dynamics_v1 import cli, diagnostics as d, metrics as m, pipeline as p, registry as r
from experiments.rineya_generalization_dynamics_v1.common import Blocked, digest, exclusive, read, signed, verify, write_new
from experiments.rineya_generalization_dynamics_v1.integration import record_snapshot


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.rows = d.build_suite("dev", 17, 8)

    def test_reproducible(self):
        self.assertEqual(self.rows, d.build_suite("dev", 17, 8))
        self.assertNotEqual(self.rows, d.build_suite("dev", 19, 8))

    def test_size_and_pairs(self):
        self.assertEqual(len(self.rows), 96)
        self.assertEqual(len({x["parent_id"] for x in self.rows}), 48)
        d.validate_suite(self.rows, "dev")

    def test_confirmation_is_disjoint(self):
        d.assert_disjoint(self.rows, d.build_suite("confirm", 101, 8))

    def test_overlap_detected(self):
        with self.assertRaises(Blocked):
            d.assert_disjoint(self.rows, self.rows)

    def test_linked_composition(self):
        by_case = {}
        for row in self.rows:
            if row["skill_case_id"] and row["condition"] == "clean":
                by_case.setdefault(row["skill_case_id"], {})[row["family"]] = row
        self.assertEqual(len(by_case), 8)
        for case in by_case.values():
            self.assertEqual(case["transform_S"]["answer"], case["compose_RS"]["answer"])
            self.assertIn("Input: " + case["retrieve_R"]["answer"], case["transform_S"]["prompt"])
            self.assertEqual(len({row["cluster_id"] for row in case.values()}), 1)

    def test_prompt_pair_answers_unchanged(self):
        for a, b in zip(self.rows[::2], self.rows[1::2]):
            self.assertEqual(a["answer"], b["answer"])
            self.assertNotEqual(a["prompt"], b["prompt"])
            self.assertNotEqual(a["answer"], a["distractor"])

    def test_invalid_split_count_seed(self):
        for args in (("test", 1, 8), ("dev", -1, 8), ("dev", True, 8), ("dev", 1, 1), ("dev", 1, 1001)):
            with self.subTest(args=args), self.assertRaises(Blocked):
                d.build_suite(*args)

    def test_invalid_pair_or_split(self):
        for mutation in (lambda x: x.pop(), lambda x: x[0].update(split="confirm"), lambda x: x[0].update(answer=x[0]["distractor"])):
            rows = copy.deepcopy(self.rows)
            mutation(rows)
            with self.assertRaises(Blocked):
                d.validate_suite(rows, "dev")


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.rows = d.build_suite("dev", 17, 8)
        self.good = [{"id": row["id"], "text": row["answer"], "gold_logp": -.1,
                      "distractor_logp": -2., "gold_tokens": 1, "distractor_tokens": 1, "stop_reason": "eos"} for row in self.rows]
        self.bad = [{**x, "text": row["distractor"], "gold_logp": -2., "distractor_logp": -.1} for row, x in zip(self.rows, self.good)]

    def test_integer_parser(self):
        for text, expected in (("  +012\n", "12"), ("-4", "-4"), ("12.", None), ("Answer: 12", None), ("12 13", None), ("<think>12</think>", None), ("1e2", None), ("", None), ("1" * 30, None)):
            self.assertEqual(m.integer_answer(text), expected)

    def test_perfect_generated_score(self):
        summary = m.summarize(self.rows, self.good)
        self.assertEqual(summary["robust_macro"], 1)
        self.assertEqual(summary["composition_given_components"]["misleading_RS_given_R_and_S"], 1)

    def test_length_truncation_is_not_success(self):
        pred = copy.deepcopy(self.good)
        pred[0]["stop_reason"] = "length"
        self.assertFalse(m.grade(self.rows, pred)[self.rows[0]["id"]]["correct"])

    def test_logp_success_is_not_generation_success(self):
        pred = [{**x, "text": "I do not know"} for x in self.good]
        summary = m.summarize(self.rows, pred)
        self.assertEqual(summary["robust_macro"], 0)
        self.assertEqual(summary["families"]["successive"]["candidate_preference"], 1)

    def test_zero_clean_support_is_undefined(self):
        summary = m.summarize(self.rows, self.bad)
        self.assertIsNone(summary["families"]["successive"]["conditional_robustness"])
        self.assertIsNone(summary["retention"]["preserved_fraction"])
        self.assertIsNone(summary["composition_given_components"]["clean_RS_given_R_and_S"])

    def test_fixed_reference_support_does_not_shrink(self):
        summary = m.summarize(self.rows, self.bad, self.good)
        self.assertEqual(summary["families"]["successive"]["reference_clean_denominator"], 8)
        self.assertEqual(summary["families"]["successive"]["own_clean_denominator"], 0)
        self.assertEqual(summary["retention"]["damages"], 16)

    def test_missing_duplicate_foreign_prediction(self):
        for predictions in (self.good[:-1], self.good[:-1] + [self.good[0]], [{**x, "id": "foreign"} for x in self.good]):
            with self.assertRaises(Blocked):
                m.grade(self.rows, predictions)

    def test_nonfinite_and_positive_logp_rejected(self):
        for value in (float("nan"), float("inf"), .1, "-0.1", True):
            pred = copy.deepcopy(self.good)
            pred[0]["gold_logp"] = value
            with self.assertRaises(Blocked):
                m.grade(self.rows, pred)

    def test_multitoken_statistics(self):
        pred = [{**x, "gold_tokens": 2, "gold_logp": -2., "distractor_logp": -1.5} for x in self.good]
        result = m.summarize(self.rows, pred)["families"]["successive"]
        self.assertEqual(result["mean_logp_margin"], -.5)
        self.assertEqual(result["mean_token_margin"], .5)

    def test_paired_clusters_and_null_interval(self):
        result = m.paired_comparison(self.rows, self.good, self.good, 100)
        self.assertEqual(result["macro_difference"], 0)
        self.assertEqual(result["independent_clusters"], 24)
        self.assertEqual(result["stratified_cluster_bootstrap_95_interval"], [0, 0])

    def test_paired_direction_and_reversal(self):
        result = m.paired_comparison(self.rows, self.good, self.bad, 100)
        self.assertEqual(result["macro_difference"], 1)
        self.assertEqual(result["paired_wins"], 40)
        t = m.trajectory(self.rows, [self.good, self.bad, self.good])
        self.assertEqual(t["rows_with_three_checkpoint_reversal"], 96)
        self.assertEqual(t["adjacent_transitions"][0]["damages"], 96)

    def test_no_retention_gate_without_support(self):
        summary = m.summarize(self.rows, self.bad)
        allowed, reasons = m.eligible(summary, summary, p.DEFAULT_POLICY)
        self.assertFalse(allowed)
        self.assertIn("insufficient_reference_retention_support", reasons)


class IOTests(unittest.TestCase):
    def test_digest_is_canonical_and_checked(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        value = signed({"a": 1})
        verify(value)
        value["a"] = 2
        with self.assertRaises(Blocked):
            verify(value)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.json"
            write_new(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                write_new(path, {"a": 2})
            self.assertEqual(read(path), {"a": 1})

    def test_lock_excludes_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lock"
            with exclusive(path):
                with self.assertRaises(Blocked):
                    with exclusive(path):
                        pass
            self.assertFalse(path.exists())

    def test_nonfinite_json_is_not_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text('{"x": NaN}')
            with self.assertRaises(Blocked):
                read(path)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("base", "one", "two"):
            folder = self.root / name
            folder.mkdir()
            (folder / "config.json").write_text('{"model_type":"qwen3", "hidden_size":1024}')
            (folder / "model.safetensors").write_bytes(b"mechanical inventory fixture, NOT real weights")
        tok = self.root / "tokenizer"
        tok.mkdir()
        (tok / "tokenizer_config.json").write_text('{}')
        (tok / "tokenizer.json").write_text('{}')
        training = self.root / "train.jsonl"
        training.write_text('{"prompt":"unrelated training prompt", "parent_id":"training-case"}\n')
        self.spec = {"trajectory_id": "fixture", "expected_base_parameters": 596049920,
                     "base_model": str(self.root / "base"), "tokenizer": str(tok), "training_manifest": str(training),
                     "checkpoints": [{"id": "ref", "step": 0, "kind": "base"},
                                     {"id": "one", "step": 10, "kind": "hf", "path": str(self.root / "one")},
                                     {"id": "two", "step": 20, "kind": "hf", "path": str(self.root / "two")}]}

    def test_freeze_and_verify(self):
        reg = r.freeze_registry(self.spec)
        r.verify_registry(reg)
        self.assertIn("caller-declared", reg["lineage_status"])

    def test_mutated_weights_rejected(self):
        reg = r.freeze_registry(self.spec)
        (self.root / "one/model.safetensors").write_bytes(b"changed")
        with self.assertRaises(Blocked):
            r.verify_registry(reg)

    def test_missing_weights_no_fallback(self):
        (self.root / "base/model.safetensors").unlink()
        with self.assertRaises(Blocked):
            r.freeze_registry(self.spec)

    def test_lfs_pointer_rejected(self):
        (self.root / "base/model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc")
        with self.assertRaises(Blocked):
            r.freeze_registry(self.spec)

    def test_symlinked_asset_rejected(self):
        path = self.root / "one/model.safetensors"
        path.unlink()
        path.symlink_to(self.root / "base/model.safetensors")
        with self.assertRaises(Blocked):
            r.freeze_registry(self.spec)

    def test_shard_index_cannot_escape_export(self):
        (self.root / "base/model.safetensors.index.json").write_text('{"weight_map":{"x":"../escape.safetensors"}}')
        with self.assertRaises(Blocked):
            r.freeze_registry(self.spec)

    def test_remote_code_rejected(self):
        (self.root / "base/config.json").write_text('{"auto_map":{}}')
        with self.assertRaises(Blocked):
            r.freeze_registry(self.spec)

    def test_exact_parameter_class_and_steps(self):
        for mutate in (lambda s: s.update(expected_base_parameters=344000000),
                       lambda s: s["checkpoints"][1].update(step=20),
                       lambda s: s["checkpoints"][1].update(id="../bad"),
                       lambda s: s["checkpoints"][1].update(kind="base")):
            spec = copy.deepcopy(self.spec)
            mutate(spec)
            with self.assertRaises(Blocked):
                r.freeze_registry(spec)

    def test_architecture_mismatch(self):
        (self.root / "one/config.json").write_text('{"model_type":"qwen3", "hidden_size":2048}')
        with self.assertRaises(Blocked):
            r.freeze_registry(self.spec)

    def test_training_overlap(self):
        row = d.build_suite("dev", 17, 8)[0]
        (self.root / "train.jsonl").write_text(json.dumps({"prompt": row["prompt"]}) + "\n")
        reg = r.freeze_registry(self.spec)
        with self.assertRaises(Blocked):
            r.audit_training_overlap([row], reg)

    def test_parent_overlap(self):
        row = d.build_suite("dev", 17, 8)[0]
        (self.root / "train.jsonl").write_text(json.dumps({"prompt": "another prompt", "parent_id": row["parent_id"]}) + "\n")
        with self.assertRaises(Blocked):
            r.audit_training_overlap([row], r.freeze_registry(self.spec))

    def test_training_mutation_detected(self):
        reg = r.freeze_registry(self.spec)
        (self.root / "train.jsonl").write_text('{"prompt":"changed"}\n')
        with self.assertRaises(Blocked):
            r.verify_registry(reg)

    def test_export_hook_is_immutable(self):
        record_snapshot(str(self.root / "receipts"), str(self.root / "one"), 10, "hf")
        with self.assertRaises(FileExistsError):
            record_snapshot(str(self.root / "receipts"), str(self.root / "one"), 10, "hf")

    def test_native_inference_requires_explicit_flag(self):
        run = self.root / "run"
        p.initialize(run, self.spec, per_family=8)
        with self.assertRaises(Blocked):
            p.evaluate_development(run)
        self.assertFalse((run / "dev").exists())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        p.initialize(self.run, mechanical=True, per_family=8)

    def lock(self):
        p.evaluate_development(self.run)
        return p.select(self.run)

    def test_full_development_and_confirmation(self):
        selection = self.lock()
        self.assertEqual(selection["chosen"]["generalization"], "intermediate")
        self.assertEqual(selection["chosen"]["ordinary"], "final")
        report = p.confirm(self.run, 101, allow_confirmation=True)
        self.assertEqual(report["evidence_kind"], "mechanical_fixture")
        self.assertFalse(report["permits_model_promotion"])
        self.assertGreater(report["comparisons"]["generalization_minus_final"]["macro_difference"], 0)

    def test_no_confirmation_material_before_lock(self):
        p.evaluate_development(self.run)
        self.assertFalse(any("confirm" in f.name for f in self.run.iterdir()))
        with self.assertRaises(FileNotFoundError):
            p.confirm(self.run, 101, allow_confirmation=True)
        self.assertFalse((self.run / "confirmation.STARTED.json").exists())

    def test_selection_needs_every_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            p.select(self.run)

    def test_development_resume_does_not_rewrite(self):
        p.evaluate_development(self.run)
        path = self.run / "dev/reference.json"
        before = path.read_bytes()
        p.evaluate_development(self.run)
        self.assertEqual(path.read_bytes(), before)

    def test_development_cannot_reopen(self):
        self.lock()
        with self.assertRaises(Blocked):
            p.evaluate_development(self.run)
        with self.assertRaises(Blocked):
            p.select(self.run)

    def test_explicit_confirmation_authorization(self):
        self.lock()
        with self.assertRaises(Blocked):
            p.confirm(self.run, 101)
        self.assertFalse((self.run / "confirmation.STARTED.json").exists())

    def test_seed_must_be_independent(self):
        self.lock()
        with self.assertRaises(Blocked):
            p.confirm(self.run, 17, allow_confirmation=True)
        self.assertFalse((self.run / "confirmation.STARTED.json").exists())

    def test_confirmation_cannot_repeat(self):
        self.lock()
        p.confirm(self.run, 101, allow_confirmation=True)
        with self.assertRaises(Blocked):
            p.confirm(self.run, 102, allow_confirmation=True)

    def test_failure_burns_confirmation_before_bank_generation(self):
        self.lock()
        with patch.object(p, "build_suite", side_effect=RuntimeError("synthetic interruption")):
            with self.assertRaises(RuntimeError):
                p.confirm(self.run, 101, allow_confirmation=True)
        self.assertTrue((self.run / "confirmation.STARTED.json").exists())
        with self.assertRaises(Blocked):
            p.confirm(self.run, 102, allow_confirmation=True)

    def test_code_change_blocks_run(self):
        with patch.object(p, "code_hash", return_value="modified"):
            with self.assertRaises(Blocked):
                p.evaluate_development(self.run)

    def test_development_data_tampering_detected(self):
        path = self.run / "development_suite.json"
        rows = read(path)
        rows[0]["prompt"] += " modified"
        path.write_text(json.dumps(rows))
        with self.assertRaises(Blocked):
            p.evaluate_development(self.run)

    def test_result_digest_tampering_detected(self):
        p.evaluate_development(self.run)
        path = self.run / "dev/final.json"
        result = read(path)
        result["predictions"][0]["text"] = "changed"
        path.write_text(json.dumps(result))
        with self.assertRaises(Blocked):
            p.select(self.run)

    def test_no_eligible_checkpoint_blocks_confirmation(self):
        # Policy frozen before any evaluation, not changed after seeing results.
        other = self.run.parent / "blocked"
        policy = {**p.DEFAULT_POLICY, "min_acquisition_delta": 1.0}
        p.initialize(other, mechanical=True, per_family=8, policy=policy)
        p.evaluate_development(other)
        selection = p.select(other)
        self.assertEqual(selection["status"], "BLOCKED_NO_ELIGIBLE_CHECKPOINT")
        with self.assertRaises(Blocked):
            p.confirm(other, 101, allow_confirmation=True)

    def test_changed_receipt_after_lock_rejected(self):
        self.lock()
        path = self.run / "dev/final.json"
        result = read(path)
        result.pop("sha256")
        result["predictions"][0]["text"] = "99999"
        path.write_text(json.dumps(signed(result)))
        with self.assertRaises(Blocked):
            p.confirm(self.run, 101, allow_confirmation=True)

    def test_runtime_mixture_rejected(self):
        p.evaluate_development(self.run)
        path = self.run / "dev/final.json"
        result = read(path)
        result.pop("sha256")
        result["runtime"]["backend"] = "other-backend"
        path.write_text(json.dumps(signed(result)))
        with self.assertRaises(Blocked):
            p.select(self.run)

    def test_fixture_cannot_masquerade_as_native(self):
        p.evaluate_development(self.run)
        path = self.run / "dev/final.json"
        result = read(path)
        result.pop("sha256")
        result["evidence_kind"] = "native_pilot"
        path.write_text(json.dumps(signed(result)))
        with self.assertRaises(Blocked):
            p.select(self.run)

    def test_resource_guards_not_weakened(self):
        for key in ("min_free_ram_gib", "min_free_vram_gib", "min_free_disk_gib"):
            with self.assertRaises(Blocked):
                p.validate_config(p.DEFAULT_POLICY, {**p.DEFAULT_NATIVE, key: 0})

    def test_unknown_policy_rejected(self):
        with self.assertRaises(Blocked):
            p.validate_config({**p.DEFAULT_POLICY, "secret_weight": 3}, p.DEFAULT_NATIVE)

    def test_cli_reports_errors(self):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            status = cli.main(["confirm", "--run", str(self.run), "--seed", "101"])
        self.assertEqual(status, 2)
        self.assertIn("BLOCKED", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
