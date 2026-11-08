"""CPU mechanics only. Scripted logits are NOT pretrained/native Opallo evidence."""
from __future__ import annotations

import array
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.opallo_native_collector_v1 import contract as c
from experiments.opallo_native_collector_v1 import native as n
from experiments.opallo_native_collector_v1 import session as s
from experiments.opallo_native_collector_v1 import supervisor as v
from experiments.opallo_native_collector_v1 import cli

import torch
torch.set_num_threads(1)


class Tokenizer:
    all_special_ids = [151645, 151643]
    chat_template = "synthetic-test-only"
    def encode(self, text, **kwargs):
        return [3] if text == "UNKNOWN" else [1, 2]
    def decode(self, ids, **kwargs):
        return "".join({3: "UNKNOWN", 4: " value", 151645: "<|im_end|>", 151643: "<|endoftext|>"}.get(i, "?") for i in ids)
    def apply_chat_template(self, messages, tokenize, **kwargs):
        return [1, 2] if tokenize else "synthetic:" + messages[1]["content"]


class Scripted:
    """No weights, trained parameters, Transformers or model class involved."""
    vocabulary = 151646
    unknown = 3
    eos = [151645, 151643]
    tokenizer = Tokenizer()
    def __init__(self, tokens=(3, 151645), bad=None):
        self.tokens, self.bad, self.calls = tokens, bad, []
    def synchronize(self):
        pass
    def forward(self, ids, past):
        assert not torch.is_grad_enabled()
        step = 0 if past is None else past
        self.calls.append((tuple(ids), past))
        logits = torch.full((self.vocabulary,), -10.0, dtype=torch.float32)
        logits[self.tokens[min(step, len(self.tokens) - 1)]] = 5.0
        if self.bad is not None:
            logits[0] = self.bad
        return logits, step + 1


def identity():
    return {"vocabulary_size": Scripted.vocabulary, "unknown_token_id": 3,
            "unknown_single_token": True, "unknown_encoded_ids": [3], "unknown_decoded": "UNKNOWN",
            "eos_token_ids": [151645, 151643], "runtime_sha256": "a" * 64,
            "recipe_sha256": c.digest({"runtime_sha256": "a" * 64}),
            "weight_sha256": c.pins()["assets"]["model.safetensors"]["sha256"]}


def recipe():
    runtime = {"packages": {"transformers": "4.51.3"}}
    return {"collector": c.COLLECTOR, "schema": 1, "effective": c.EFFECTIVE,
            "device": "cpu", "compute_dtype": "float32", "threads": 2,
            "official_eos_ids": c.pins()["official_eos_ids"], "runtime": runtime,
            "runtime_sha256": c.digest(runtime), "source_sha256": c.source_manifest(),
            "limits": {"per_capture_seconds": 900, "screen_seconds": 3600, "generation_max": 400,
                       "min_free_host_gib": 3.0, "max_process_tree_rss_gib": 8.0,
                       "min_free_cuda_gib_before_load": 3.0}, **c.CLOSED}


class ContractTests(unittest.TestCase):
    def test_frozen_source_and_collector_registration(self):
        self.assertEqual(c.verify_source(), c.source_manifest())
    def test_frozen_fixture_and_prompt_hashes(self):
        frozen = c.screen()
        for split in ("calibration", "check"):
            self.assertEqual(frozen.digest(frozen.fixture(split)), c.pins()["fixture_canonical_sha256"][split])
            for formatted in (False, True):
                name = split + ("_formatted" if formatted else "_baseline")
                self.assertEqual(frozen.digest(frozen.prompts(split, formatted)), c.pins()["prompt_canonical_sha256"][name])
    def test_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.json"
            c.write_new(p, {"x": 1})
            with self.assertRaises(FileExistsError): c.write_new(p, {"x": 2})
            self.assertEqual(c.read(p), {"x": 1})
    def test_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x"; p.write_text('{"x":1,"x":2}')
            with self.assertRaises(c.Blocked): c.read(p)
    def test_nonfinite_json(self):
        for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "x"; p.write_text(text)
                with self.assertRaises((c.Blocked, ValueError)): c.read(p)
    def test_json_object_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x"; p.write_text('[]')
            with self.assertRaises(c.Blocked): c.read(p)
    def test_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in ("../escape", "/absolute", "C:\\Windows", ".git/config", "a\\b", ""):
                with self.subTest(value=value), self.assertRaises(c.Blocked): c.safe_child(Path(tmp), value)
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p / "target").write_text("x"); (p / "link").symlink_to(p / "target")
            with self.assertRaises(c.Blocked): c.safe_child(p, "link")
    def test_recipe_valid_structure(self): c.validate_recipe(recipe())
    def test_recipe_sampling_rejected(self):
        value = recipe(); value["effective"] = dict(c.EFFECTIVE, do_sample=True)
        with self.assertRaises(c.Blocked): c.validate_recipe(value)
    def test_recipe_partial_eos_rejected(self):
        value = recipe(); value["official_eos_ids"] = [151645]
        with self.assertRaises(c.Blocked): c.validate_recipe(value)
    def test_recipe_runtime_drift(self):
        value = recipe(); value["runtime"]["packages"]["transformers"] = "4.57.1"
        with self.assertRaises(c.Blocked): c.validate_recipe(value)
    def test_recipe_no_cpu_bf16_fallback(self):
        value = recipe(); value["compute_dtype"] = "bfloat16"
        with self.assertRaises(c.Blocked): c.validate_recipe(value)
    def test_recipe_budget_cannot_expand(self):
        value = recipe(); value["limits"]["generation_max"] = 408
        with self.assertRaises(c.Blocked): c.validate_recipe(value)
    def test_recipe_training_cannot_open(self):
        value = recipe(); value["licenses_training"] = True
        with self.assertRaises(c.Blocked): c.validate_recipe(value)
    def test_recipe_thread_bool_rejected(self):
        value = recipe(); value["threads"] = True
        with self.assertRaises(c.Blocked): c.validate_recipe(value)
    def test_asset_extra_override_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p / "adapter_config.json").write_text('{}')
            with self.assertRaises(c.Blocked): c.asset_manifest(p)
    def test_missing_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(c.Blocked): c.asset_manifest(Path(tmp))
    def test_wrong_asset_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p / "config.json").write_text('{}')
            with self.assertRaisesRegex(c.Blocked, 'size mismatch'): c.asset_manifest(p)
    def test_pending_human_review_rejected(self):
        with self.assertRaises(c.Blocked): c.validate_human({"status": "PENDING"})
    def test_incomplete_human_review_rejected(self):
        with self.assertRaises(c.Blocked): c.validate_human({"status": "APPROVED", "rows": []})
    def test_full_human_review_one_unresolved_rejected(self):
        rows = c.screen().fixture("calibration") + c.screen().fixture("check")
        review = {"status": "APPROVED", "rows": [dict(r, decision="APPROVED", complete_value_correct=True,
                    field_presence_correct=True, unambiguous_contract=True) for r in rows]}
        c.validate_human(review)
        review["rows"][-1]["unambiguous_contract"] = False
        with self.assertRaises(c.Blocked): c.validate_human(review)
    def test_union_missing_categories(self):
        with self.assertRaises(c.Blocked):
            c.validate_audit({"status": "PASS_REVIEWED_UNION", "unresolved_findings": 0, "exclusions": [], "sources": []}, ROOT)
    def test_union_exclusions_require_new_registration(self):
        with self.assertRaises(c.Blocked):
            c.validate_audit({"status": "PASS_REVIEWED_UNION", "unresolved_findings": 0, "exclusions": ["row"]}, ROOT)
    def test_templates_are_pending_not_approvals(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pending"; cli.templates(p)
            human = c.read(p / "human_review.pending.json")
            self.assertEqual(len(human["rows"]), 192)
            with self.assertRaises(c.Blocked): c.validate_human(human)
            self.assertIsNone(c.read(p / "authorization.pending.json")["host_quiescent"])
    def test_inventory_never_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = cli.inventory(Path(tmp) / "inventory.json", None)
            self.assertEqual(value["status"], "BLOCKED_BEFORE_NATIVE")
            self.assertEqual(value["native_generations"], 0)
            self.assertFalse(value["native_collector_qualified"])
            self.assertTrue(all(value[k] is False for k in c.CLOSED))


class SignalTests(unittest.TestCase):
    def check_signal(self, logits, unknown=1):
        signal, blob = n.signal_from_logits(logits, unknown, logits.numel())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "signal.f64").write_bytes(blob); signal["path"] = "signal.f64"
            n.validate_signal(signal, root, unknown, logits.numel())
        return signal
    def test_full_vocabulary_not_top_k(self):
        value = self.check_signal(torch.zeros(20, dtype=torch.float64))
        self.assertAlmostEqual(value["unknown_probability"], 1 / 20, places=15)
    def test_extreme_logits_stable(self):
        value = self.check_signal(torch.tensor([1000., 999., -1000.]))
        self.assertAlmostEqual(value["unknown_probability"], 1 / (1 + math.e), places=14)
    def test_extreme_negative_log_probability_underflow_is_recorded(self):
        value = self.check_signal(torch.tensor([1000., -1000.]))
        self.assertEqual(value["unknown_probability"], 0.0)
        self.assertEqual(value["unknown_log_probability"], -2000.0)
    def test_nonfinite_logits_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad), self.assertRaises(c.Blocked): n.signal_from_logits(torch.tensor([bad, 1.]), 1, 2)
    def test_partial_vocab_rejected(self):
        with self.assertRaises(c.Blocked): n.signal_from_logits(torch.ones(4), 1, 5)
    def test_matrix_logits_rejected(self):
        with self.assertRaises(c.Blocked): n.signal_from_logits(torch.ones(1, 4), 1, 4)
    def test_bool_unknown_rejected(self):
        with self.assertRaises(c.Blocked): n.signal_from_logits(torch.ones(4), True, 4)
    def test_first_argmax_tie_uses_first_id(self):
        value = self.check_signal(torch.ones(4))
        self.assertEqual(value["first_argmax_id"], 0)
    def test_raw_logits_not_modified(self):
        logits = torch.randn(16); before = logits.clone()
        self.check_signal(logits)
        self.assertTrue(torch.equal(logits, before))
    def test_raw_signal_tampering_rejected(self):
        signal, blob = n.signal_from_logits(torch.zeros(4), 1, 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / "s"; path.write_bytes(blob); signal["path"] = "s"
            path.write_bytes(blob[:-1] + bytes([blob[-1] ^ 1]))
            with self.assertRaises(c.Blocked): n.validate_signal(signal, root, 1, 4)
    def test_probability_tampering_rejected(self):
        signal, blob = n.signal_from_logits(torch.zeros(4), 1, 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "s").write_bytes(blob); signal["path"] = "s"
            signal["unknown_probability"] += 0.1
            with self.assertRaises(c.Blocked): n.validate_signal(signal, root, 1, 4)
    def test_log_probability_tampering_rejected(self):
        signal, blob = n.signal_from_logits(torch.zeros(4), 1, 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "s").write_bytes(blob); signal["path"] = "s"
            signal["unknown_log_probability"] += 0.1
            with self.assertRaises(c.Blocked): n.validate_signal(signal, root, 1, 4)
    def test_multitoken_unknown_rejected(self):
        tokenizer = Tokenizer(); tokenizer.encode = lambda *a, **kw: [3, 4]
        with self.assertRaises(c.Blocked): n.unknown_token(tokenizer, 151646)
    def test_nonliteral_unknown_decode_rejected(self):
        tokenizer = Tokenizer(); tokenizer.decode = lambda *a, **kw: "UNKNOWN."
        with self.assertRaises(c.Blocked): n.unknown_token(tokenizer, 151646)
    def test_valid_unknown_not_hardcoded(self):
        self.assertEqual(n.unknown_token(Tokenizer(), 151646), 3)
    def test_special_unknown_rejected(self):
        tokenizer = Tokenizer(); tokenizer.all_special_ids = [3]
        with self.assertRaises(c.Blocked): n.unknown_token(tokenizer, 151646)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.request = c.screen().prompts("calibration", False)[0]
    def tearDown(self): self.tmp.cleanup()
    def capture(self, tokens=(3, 151645), confidence=True, name="s"):
        engine = Scripted(tokens)
        row = n.capture_one(engine, self.request, self.root, name, confidence)
        return engine, row
    def validate(self, row, confidence=True): n.validate_row(row, self.request, identity(), self.root, confidence)
    def test_greedy_cache_and_eos_accounting(self):
        engine, row = self.capture(); self.validate(row)
        self.assertEqual(engine.calls, [((1, 2), None), ((3,), 1)])
        self.assertEqual(row["generated_ids"], [3, 151645]); self.assertEqual(row["generation_steps"], 2)
        self.assertEqual(row["generated_tokens"], 1); self.assertEqual(row["prediction"], "UNKNOWN")
    def test_second_official_eos(self):
        _, row = self.capture((3, 151643)); self.validate(row)
        self.assertEqual(row["eos_token_id"], 151643)
    def test_eos_on_first_step(self):
        _, row = self.capture((151643,)); self.validate(row)
        self.assertEqual(row["generated_tokens"], 0); self.assertEqual(row["generation_steps"], 1)
    def test_eos_on_last_permitted_step(self):
        _, row = self.capture((4,) * 23 + (151645,)); self.validate(row)
        self.assertEqual(row["generation_steps"], 24)
    def test_no_eos_is_blocked_not_scored(self):
        engine, row = self.capture((4,))
        self.assertEqual(len(engine.calls), 24); self.assertFalse(row["terminated"])
        with self.assertRaises(c.Blocked): self.validate(row)
    def test_boolean_eos_cannot_replace_real_token(self):
        _, row = self.capture(); row["generated_ids"][-1] = 4
        with self.assertRaises(c.Blocked): self.validate(row)
    def test_post_eos_tokens_rejected(self):
        _, row = self.capture(); row["generated_ids"] = [151643, 3, 151645]
        with self.assertRaises(c.Blocked): self.validate(row)
    def test_eos_content_accounting_tamper(self):
        _, row = self.capture(); row["generated_tokens"] = 2
        with self.assertRaises(c.Blocked): self.validate(row)
    def test_boolean_count_rejected(self):
        _, row = self.capture(); row["generated_tokens"] = True
        with self.assertRaises(c.Blocked): self.validate(row)
    def test_signal_row_mismatch(self):
        _, row = self.capture(); row["unknown_probability"] = 0.5
        with self.assertRaises(c.Blocked): self.validate(row)
    def test_prompt_id_tamper_rejected(self):
        _, row = self.capture(); row["id"] = "other"
        with self.assertRaises(c.Blocked): self.validate(row)
    def test_prediction_keeps_outer_whitespace(self):
        _, row = self.capture((4, 151645)); self.validate(row)
        self.assertEqual(row["prediction"], " value")
    def test_confidence_off_has_no_signal(self):
        _, row = self.capture(confidence=False); self.validate(row, False)
        self.assertIsNone(row["signal"]); self.assertIsNone(row["unknown_probability"])
    def test_confidence_off_cannot_fabricate_probability(self):
        _, row = self.capture(confidence=False); row["unknown_probability"] = 0.1
        with self.assertRaises(c.Blocked): self.validate(row, False)
    def test_generation_starts_with_fresh_cache(self):
        engine = Scripted()
        n.capture_one(engine, self.request, self.root, "one", False)
        n.capture_one(engine, self.request, self.root, "two", False)
        self.assertEqual([p for _, p in engine.calls], [None, 1, None, 1])
    def test_bad_native_logits_block_before_signal(self):
        with self.assertRaises(c.Blocked):
            n.capture_one(Scripted(bad=float("nan")), self.request, self.root, "s", True)
    def test_input_overflow_no_truncation(self):
        tokenizer = Tokenizer()
        tokenizer.encode = lambda *a, **kw: [1] * 4073
        tokenizer.apply_chat_template = lambda *a, **kw: [1] * 4073 if kw["tokenize"] else "text"
        with self.assertRaises(c.Blocked): n.render_input(tokenizer, self.request, 151646)
    def test_empty_input_rejected(self):
        tokenizer = Tokenizer()
        tokenizer.encode = lambda *a, **kw: []
        tokenizer.apply_chat_template = lambda *a, **kw: [] if kw["tokenize"] else "text"
        with self.assertRaises(c.Blocked): n.render_input(tokenizer, self.request, 151646)
    def test_two_tokenization_routes_must_match(self):
        tokenizer = Tokenizer()
        tokenizer.encode = lambda *a, **kw: [2, 1]
        with self.assertRaises(c.Blocked): n.render_input(tokenizer, self.request, 151646)
    def test_gold_fields_not_allowed(self):
        with self.assertRaises(c.Blocked): n.render_input(Tokenizer(), dict(self.request, answer="secret"), 151646)
    def test_parity_all_eight_rows(self):
        on, off, repeat = [], [], []
        for i in range(8):
            for values, mode, label in ((on, True, "on"), (off, False, "off"), (repeat, True, "repeat")):
                _, row = self.capture(confidence=mode, name=f"{label}{i}"); values.append(row)
        n.parity(on, off, repeat)
        repeat[7]["prediction"] += " "
        with self.assertRaises(c.Blocked): n.parity(on, off, repeat)
    def test_partial_parity_rejected(self):
        with self.assertRaises(c.Blocked): n.parity([], [], [])


class GitAndWatchdogTests(unittest.TestCase):
    def test_git_receipt_checks_exact_committed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            c.git(root, "config", "user.name", "Mechanical Test"); c.git(root, "config", "user.email", "test@example.invalid")
            (root / "receipt.json").write_text('{"fixture":true}\n')
            c.git(root, "add", "receipt.json"); c.git(root, "commit", "-qm", "fixture receipt")
            head = c.git(root, "rev-parse", "HEAD").decode().strip()
            c.committed_file(root, head, "receipt.json")
            (root / "receipt.json").write_text('{"fixture":false}\n')
            with self.assertRaises(c.Blocked): c.committed_file(root, head, "receipt.json")
    def test_non_commit_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "x").write_text("x")
            blob = c.git(root, "hash-object", "-w", "x").decode().strip()
            with self.assertRaises(c.Blocked): c.committed_file(root, blob, "x")
    def test_worker_success_is_not_native_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = v.run_bounded([sys.executable, "-c", "print('mechanical child only')"], root, root / "job",
                                    seconds=5, min_free_gib=0, max_rss_gib=1)
            self.assertEqual(receipt["status"], "PASS_PROCESS_ONLY"); self.assertFalse(receipt["native_qualification"])
    def test_watchdog_kills_hanging_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); start = time.monotonic()
            with self.assertRaises(c.Blocked):
                v.run_bounded([sys.executable, "-c", "import time; time.sleep(60)"], root, root / "job",
                              seconds=0.3, min_free_gib=0, max_rss_gib=1)
            self.assertLess(time.monotonic() - start, 5)
            self.assertEqual(c.read(root / "job" / "SUPERVISOR.json")["reason"], "WALL_TIME_LIMIT")
    def test_watchdog_nonzero_exit_preserves_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(c.Blocked):
                v.run_bounded([sys.executable, "-c", "raise SystemExit(7)"], root, root / "job",
                              seconds=5, min_free_gib=0, max_rss_gib=1)
            self.assertEqual(c.read(root / "job" / "SUPERVISOR.json")["exit_code"], 7)
    def test_host_memory_failure_prevents_spawn(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(v, "memory_snapshot", return_value=(0, 0)), patch.object(v.subprocess, "Popen") as launch:
            with self.assertRaises(c.Blocked):
                v.run_bounded([sys.executable, "-c", "pass"], Path(tmp), Path(tmp) / "job",
                              seconds=5, min_free_gib=3, max_rss_gib=1)
            launch.assert_not_called()
    def test_rss_watchdog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(c.Blocked):
                v.run_bounded([sys.executable, "-c", "import time; time.sleep(60)"], root, root / "job",
                              seconds=5, min_free_gib=0, max_rss_gib=0.00001)
            self.assertEqual(c.read(root / "job" / "SUPERVISOR.json")["reason"], "PROCESS_TREE_RSS_LIMIT")
    def test_wrong_worker_parent_rejected(self):
        with self.assertRaises(c.Blocked): v.contain_worker(-1)
    def test_phase_generation_budget_exact(self):
        self.assertEqual({k: sum(p[4] - p[3] for p in plans) for k, plans in s.PLANS.items()}, s.COUNTS)
        self.assertEqual(sum(s.COUNTS.values()), 400)
        self.assertEqual(s.requests_for(s.PLANS["calibrate"][0])[0], c.screen().prompts("calibration", False)[8])
    def test_check_cannot_open_before_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((c.Blocked, FileNotFoundError)): s.ready(Path(tmp), "check", {})
    def test_reserved_phase_cannot_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "qualify").mkdir()
            with self.assertRaises(c.Blocked): s.ready(root, "qualify", {})
    def test_whole_screen_clock_not_reset(self):
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c.write_new(root / "CLOCK.json", {"wall": time.time()-3601, "monotonic": time.monotonic()-3601, "boot_time": psutil.boot_time()})
            with self.assertRaises(c.Blocked): s.remaining(root)
    def test_clock_rollback_rejected(self):
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c.write_new(root / "CLOCK.json", {"wall": time.time()+100, "monotonic": time.monotonic(), "boot_time": psutil.boot_time()})
            with self.assertRaises(c.Blocked): s.remaining(root)
    def test_reboot_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c.write_new(root / "CLOCK.json", {"wall": time.time(), "monotonic": time.monotonic(), "boot_time": 0})
            with self.assertRaises(c.Blocked): s.remaining(root)
    def test_native_loader_ast_no_generate_or_training(self):
        import ast
        tree = ast.parse((c.ROOT / "native.py").read_text())
        forbidden = {"generate", "backward", "step", "save_pretrained", "push_to_hub"}
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(x.func, ast.Attribute) and x.func.attr in forbidden for x in calls))
        loads = [x for x in calls if isinstance(x.func, ast.Attribute) and x.func.attr == "from_pretrained"]
        for node in loads:
            kw = {x.arg: x.value for x in node.keywords}
            self.assertIs(kw["local_files_only"].value, True)
            self.assertIs(kw["trust_remote_code"].value, False)
            self.assertNotIn("device_map", kw)
    def test_worker_ast_does_not_access_fixture_keys_or_scorer(self):
        import ast, inspect
        tree = ast.parse(inspect.getsource(cli.worker))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute): self.assertNotIn(node.attr, {"fixture", "calibrate", "replay", "metrics"})


if __name__ == "__main__": unittest.main()
