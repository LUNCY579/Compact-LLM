"""CPU-only token/mask boundary tests; real local integration is opt-in."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

import cpu_tokenizer as tokens
from rs_contract import render

SPEC = {"initial": [1, 2, 4, 8, 16, 32], "selection": {"k": 2, "target": 3}}
TARGET = '{"answer":[0,1]}'


class FakeEngine:
    """Deterministic reversible lexical fixture, never Titan token evidence."""
    def __init__(self):
        self.ids = {"<|im_end|>": tokens.EOS_ID, "<|im_start|>": 151644,
                    "<think>": 151667, "</think>": 151668}
        self.reverse = {value: key for key, value in self.ids.items()}
        self.fault = None

    def encode(self, text, add_special_tokens):
        if add_special_tokens is not False:
            raise AssertionError("automatic special tokens")
        pieces = re.findall(r"<\|im_end\|>|<\|im_start\|>|<think>|</think>|\n|[^\s<>]+[ \t]*|.", text)
        result = []
        for piece in pieces:
            if piece not in self.ids:
                value = 1000 + len(self.ids)
                self.ids[piece] = value
                self.reverse[value] = piece
            result.append(self.ids[piece])
        full = text.endswith("<|im_end|>\n") and 'assistant\n' in text
        if self.fault == "prefix" and full:
            result[0] += 1
        elif self.fault == "completion" and full:
            result[-2] += 1
        elif self.fault == "tail" and full:
            result.append(123)
        elif self.fault == "target_eos" and text == TARGET:
            result.append(tokens.EOS_ID)
        elif self.fault == "boolean_id" and text == TARGET:
            result[0] = True
        elif self.fault == "empty" and text == TARGET:
            result = []
        return SimpleNamespace(ids=result)

    def decode(self, values, skip_special_tokens):
        if skip_special_tokens is not False:
            raise AssertionError("EOS hidden in decode")
        if self.fault == "decode":
            return "wrong"
        return "".join(self.reverse[value] for value in values)


class FakeTemplate:
    def __init__(self):
        self.fault = None

    def render(self, *, messages, tools, add_generation_prompt, enable_thinking):
        if tools is not None or enable_thinking is not False:
            raise AssertionError("wrong template settings")
        result = "".join(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                         for message in messages if message["role"] != "assistant")
        result += tokens.ASSISTANT_PREFIX
        if not add_generation_prompt:
            result += messages[-1]["content"] + "<|im_end|>\n"
        if self.fault == "prefix":
            result = result.replace("<think>\n\n</think>\n\n", "")
        elif self.fault == "full" and not add_generation_prompt:
            result += "extra"
        return result


class CpuBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.template = FakeTemplate()
        self.tokenizer = tokens.CpuTokenizer(_engine=self.engine, _template=self.template, _binding={})

    def test_mask_counts_eos_and_template_tail(self):
        row = self.tokenizer.encode_example(render(SPEC), TARGET)
        p = row["prompt_tokens"]
        self.assertEqual(row["input_ids"], row["prompt_ids"] + row["completion_ids"])
        self.assertEqual(row["labels"], [-100] * p + row["completion_ids"])
        self.assertEqual(row["attention_mask"], [1] * row["total_tokens"])
        self.assertEqual(row["supervised_tokens"], len(row["completion_ids"]))
        self.assertEqual(row["total_tokens"], p + row["supervised_tokens"])
        self.assertEqual(row["input_ids"][-1], tokens.EOS_ID)
        self.assertIn(tokens.EOS_ID, row["prompt_ids"])
        self.assertTrue(all(x == -100 for x in row["labels"][:p]))
        self.assertEqual(row["labels"][-1], tokens.EOS_ID)
        self.assertEqual(set(row), {"prompt_ids", "completion_ids", "input_ids", "labels", "attention_mask",
                                   "prompt_tokens", "supervised_tokens", "total_tokens",
                                   "prompt_text_sha256", "full_template_sha256"})

    def test_whole_example_cap_is_inclusive_no_truncation(self):
        row = self.tokenizer.encode_example(render(SPEC), TARGET)
        self.assertEqual(row, self.tokenizer.encode_example(render(SPEC), TARGET, row["total_tokens"]))
        with self.assertRaisesRegex(ValueError, "truncation forbidden"):
            self.tokenizer.encode_example(render(SPEC), TARGET, row["total_tokens"] - 1)

    def test_invalid_caps_rejected(self):
        for cap in (True, False, 0, -1, 513, 512.0, "512", None):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.tokenizer.encode_example(render(SPEC), TARGET, cap)

    def test_incorrect_noncanonical_or_nonanswer_targets_rejected(self):
        for target in ('{"answer":[0,2]}', '{"answer":"UNKNOWN"}', '{"answer": [0,1]}',
                       '{"answer":[1,0]}', '{"answer":[0,1],"reason":"gold"}', True, None):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.tokenizer.encode_example(render(SPEC), target)

    def test_prompt_rules_roles_and_metadata_rejected(self):
        original = render(SPEC)
        for kind in ("rules", "metadata", "assistant", "role"):
            messages = copy.deepcopy(original)
            if kind == "rules":
                messages[0]["content"] += " gold"
            elif kind == "metadata":
                messages[1]["parent_id"] = "hidden"
            elif kind == "assistant":
                messages.append({"role": "assistant", "content": TARGET})
            else:
                messages[1]["role"] = "system"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.tokenizer.encode_example(messages, TARGET)

    def test_rs_training_arrays_rejected(self):
        spec = copy.deepcopy(SPEC)
        spec["lookup"] = {"links": {"a": 0, "b": 1}, "requested": ["a", "b"]}
        with self.assertRaisesRegex(ValueError, "only atomic"):
            self.tokenizer.encode_example(render(spec), TARGET)

    def test_template_prefix_and_full_render_drift_rejected(self):
        for fault in ("prefix", "full"):
            self.template.fault = fault
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.tokenizer.encode_example(render(SPEC), TARGET)

    def test_joint_token_boundary_decode_and_token_type_failures_rejected(self):
        for fault in ("prefix", "completion", "tail", "target_eos", "decode", "boolean_id", "empty"):
            self.engine.fault = fault
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.tokenizer.encode_example(render(SPEC), TARGET)

    def test_binding_returns_detached_values(self):
        self.tokenizer._binding = {"assets": {"example": {"sha256": "pin"}}}
        detached = self.tokenizer.binding
        detached["assets"]["example"]["sha256"] = "changed"
        self.assertEqual(self.tokenizer.binding["assets"]["example"]["sha256"], "pin")

    def test_framework_process_fails_closed(self):
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
            with self.assertRaisesRegex(ValueError, "no model frameworks"):
                self.tokenizer.encode_example(render(SPEC), TARGET)

    def test_binding_detects_asset_and_decision_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            asset = path / "tokenizer.json"
            source = path / "decision.json"
            asset.write_bytes(b"asset")
            source.write_bytes(b"decision")
            binding = {"model_dir": directory,
                       "assets": {"tokenizer.json": {"sha256": hashlib.sha256(b"asset").hexdigest(), "bytes": 5}},
                       "binding_sources": {str(source): hashlib.sha256(b"decision").hexdigest()},
                       "dependency_versions": {}}
            self.tokenizer._binding = binding
            self.assertEqual(binding, self.tokenizer.verify_binding())
            asset.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "asset drift"):
                self.tokenizer.verify_binding()
            asset.write_bytes(b"asset")
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.tokenizer.verify_binding()

    def test_dependency_version_drift_rejected(self):
        self.tokenizer._binding = {"model_dir": "unused", "assets": {}, "binding_sources": {},
                                   "dependency_versions": {"tokenizers": "pinned"}}
        with patch.object(tokens.importlib.metadata, "version", return_value="different"):
            with self.assertRaisesRegex(ValueError, "dependency drift"):
                self.tokenizer.verify_binding()

    def test_missing_or_wrong_decision_pins_fail_before_optional_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.json"
            for text in ('{}', '{"student":true}', '{"student":{"tokenizer_sha256":"wrong"}}',
                         '{"student":{},"student":{}}'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    tokens.load_local_tokenizer(path, model_dir=Path(directory))


@unittest.skipUnless(os.environ.get("SYNTH_CRS_LOCAL_TOKENIZER_DIR"),
                     "set SYNTH_CRS_LOCAL_TOKENIZER_DIR for pinned local CPU integration")
class LocalTokenizerIntegrationTests(unittest.TestCase):
    def test_exact_local_template_and_all_384_fixtures(self):
        from task_compiler import compile_records, fixture_catalog
        tokenizer = tokens.load_local_tokenizer(model_dir=os.environ["SYNTH_CRS_LOCAL_TOKENIZER_DIR"])
        before = tokenizer.verify_binding()
        rows = compile_records(fixture_catalog())
        self.assertEqual(384, len(rows))
        for row in rows:
            encoded = tokenizer.encode_example(row["messages"], row["candidate"])
            self.assertLessEqual(encoded["total_tokens"], 512)
            self.assertEqual(encoded["completion_ids"][-1], tokens.EOS_ID)
            self.assertEqual(encoded["labels"][-1], tokens.EOS_ID)
        self.assertEqual(before, tokenizer.verify_binding())
        self.assertEqual([], before["framework_modules_imported"])
        self.assertFalse(before["training_admitted"])
        self.assertFalse(before["native_qualification"])


if __name__ == "__main__":
    unittest.main()
