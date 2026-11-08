"""Tokenizer plumbing tests use a tiny local fake, never a model or network."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import bridge_data as b
from experiments.post_training_mimo_centralized_v1.scripts import tokenize_bridge as t
from experiments.post_training_mimo_centralized_v1.tests.test_bridge_data import parent, registration


class FakeTokenizer:
    eos_token_id = t.EOS_ID
    eos_token = "<|im_end|>"
    pad_token_id = 151643
    chat_template = "synthetic-fixture-template"

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("unexpected extra special tokens")
        result = []
        while text:
            if text.startswith(self.eos_token):
                result.append(self.eos_token_id)
                text = text[len(self.eos_token):]
            else:
                result.append(ord(text[0]) + 1000)
                text = text[1:]
        return result

    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            raise AssertionError("unexpected special-token removal")
        return "".join(self.eos_token if x == self.eos_token_id else chr(x - 1000) for x in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        if enable_thinking:
            raise AssertionError("thinking must be disabled")
        prefix = ("<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        if len(messages) == 1:
            if not add_generation_prompt:
                raise AssertionError("missing generation prefix")
            text = prefix
        else:
            if add_generation_prompt:
                raise AssertionError("unexpected second assistant prefix")
            text = prefix + messages[1]["content"] + "<|im_end|>\n"
        return self.encode(text) if tokenize else text


def fixtures(root):
    reg = registration()
    reg["context_contract"] = {"max_total_tokens": 1024, "max_target_tokens": 128,
                               "truncation": "forbidden; reject before manifest freeze"}
    reg_path = root / "registration.json"
    reg_path.write_bytes(b.json_bytes(reg))
    data_dir = root / "source"
    b.build_dataset(reg_path, data_dir, "a" * 40)
    model_dir = root / "tokenizer"
    model_dir.mkdir()
    for name in t.TOKENIZER_FILES:
        (model_dir / name).write_bytes(b.json_bytes({"chat_template": FakeTokenizer.chat_template})
                                      if name == "tokenizer_config.json" else name.encode())
    tokenizer_receipt = root / "TOKENIZER_RECEIPT.json"
    tokenizer_receipt.write_bytes(b.json_bytes({
        "checkpoint_path": str(model_dir),
        "tokenizer_files": {name: t.file_identity(model_dir / name) for name in t.TOKENIZER_FILES},
        "chat_template_sha256": b.sha256(FakeTokenizer.chat_template.encode()),
    }))
    return data_dir, tokenizer_receipt


class MaskTests(unittest.TestCase):
    def test_prompt_eos_masked_completion_eos_supervised_padding_eos_masked(self):
        result = t.assemble_example([7, t.EOS_ID], [8, t.EOS_ID], pad_to=6)
        self.assertEqual(result["input_ids"], [7, t.EOS_ID, 8, t.EOS_ID, t.EOS_ID, t.EOS_ID])
        self.assertEqual(result["labels"], [-100, -100, 8, t.EOS_ID, -100, -100])
        self.assertEqual(result["completion_mask"], [0, 0, 1, 1, 0, 0])
        self.assertEqual(result["attention_mask"], [1, 1, 1, 1, 0, 0])
        self.assertEqual(result["nonpadding_tokens"], 4)

    def test_exact_boundaries_and_oversize_refusal(self):
        result = t.assemble_example([1] * 896, [2] * 127 + [t.EOS_ID])
        self.assertEqual(result["nonpadding_tokens"], 1024)
        for prompt, completion in (([1] * 897, [2] * 127 + [t.EOS_ID]), ([1], [2] * 128 + [t.EOS_ID])):
            with self.assertRaises(ValueError):
                t.assemble_example(prompt, completion)

    def test_invalid_ids_and_missing_eos(self):
        for prompt, completion, pad_to in (([], [t.EOS_ID], None), ([True], [t.EOS_ID], None),
                                            ([1], [2], None), ([1], [], None),
                                            ([1], [t.EOS_ID], True), ([1], [t.EOS_ID], 1),
                                            ([1], [t.EOS_ID], 1025)):
            with self.assertRaises(ValueError):
                t.assemble_example(prompt, completion, pad_to)


class RowTests(unittest.TestCase):
    def test_all_families_tokenize_with_exact_hashes_and_prefix(self):
        tokenizer = FakeTokenizer()
        for family in b.FAMILIES:
            row = b.make_row(parent(family), "a" * 40)
            result = t.tokenize_row(row, tokenizer)
            prefix = result["input_ids"][:result["prompt_tokens"]]
            self.assertEqual(result["tokenized_prompt_sha256"], b.sha256(b.json_bytes(prefix)))
            self.assertEqual(result["labels"][-1], t.EOS_ID)
            self.assertEqual(sum(result["completion_mask"]), result["completion_tokens"])
            self.assertFalse(result["training_eligibility"])
            self.assertTrue(result["training_role_eligible"])

    def test_sealed_role_fails_before_tokenizer_call(self):
        row = b.make_row(parent(b.FAMILIES[0]), "a" * 40)
        row["role"] = "CHECK"
        with self.assertRaises(ValueError):
            t.tokenize_row(row, object())

    def test_source_prompt_target_parent_corruption_rejected(self):
        row = b.make_row(parent(b.FAMILIES[0]), "a" * 40)
        for field, value in (("prompt", "altered"), ("target", '{"answer":9999}'),
                             ("parent_id", "wrong"), ("canonical_parent_sha256", "0" * 64)):
            mutation = copy.deepcopy(row)
            mutation[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                t.tokenize_row(mutation, FakeTokenizer())

    def test_full_conversation_template_drift_rejected(self):
        class DriftTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
                text = super().apply_chat_template(messages, False, add_generation_prompt, enable_thinking)
                if len(messages) == 2:
                    text += "unexpected"
                return self.encode(text) if tokenize else text
        with self.assertRaisesRegex(ValueError, "exact prompt"):
            t.tokenize_row(b.make_row(parent(b.FAMILIES[0]), "a" * 40), DriftTokenizer())

    def test_thinking_prefix_drift_rejected(self):
        class ThinkingTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
                text = super().apply_chat_template(messages, False, add_generation_prompt, enable_thinking)
                text = text.replace("<think>\n\n</think>\n\n", "<think>\n")
                return self.encode(text) if tokenize else text
        with self.assertRaisesRegex(ValueError, "generation prefix"):
            t.tokenize_row(b.make_row(parent(b.FAMILIES[0]), "a" * 40), ThinkingTokenizer())

    def test_token_prefix_boundary_drift_rejected(self):
        class BoundaryTokenizer(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                ids = super().encode(text, add_special_tokens)
                if text.endswith("<|im_end|>\n") and text.startswith("<|im_start|>"):
                    ids[0] += 1
                return ids
        with self.assertRaisesRegex(ValueError, "prompt token prefix"):
            t.tokenize_row(b.make_row(parent(b.FAMILIES[0]), "a" * 40), BoundaryTokenizer())

    def test_wrong_eos_contract_rejected(self):
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = 0
        with self.assertRaisesRegex(ValueError, "EOS ID"):
            t.tokenize_row(b.make_row(parent(b.FAMILIES[0]), "a" * 40), tokenizer)


class DatasetTests(unittest.TestCase):
    def test_wrong_runtime_rejected_before_transformers_import(self):
        with patch.object(t, "sys", SimpleNamespace(modules={})), \
                patch.object(t.importlib.metadata, "version", return_value="5.14.1"):
            with self.assertRaisesRegex(ValueError, "wrong runtime"):
                t.load_tokenizer(Path("unused"))

    def test_hash_bound_write_once_output_no_sealed_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, tokenizer_receipt = fixtures(root)
            original_read = Path.read_bytes
            def refuse_sealed(path):
                if path.name == "SEALED_RESERVATIONS.jsonl":
                    raise AssertionError("sealed file must never be opened")
                return original_read(path)
            source_hashes = {role: t.file_identity(source / (role + ".jsonl")) for role in t.OPEN_ROLES}
            with patch.object(t, "load_tokenizer", return_value=FakeTokenizer()), patch.object(Path, "read_bytes", refuse_sealed):
                result = t.tokenize_dataset(source, tokenizer_receipt, root / "out")
                self.assertFalse(result["sealed_access"]["opened"])
                self.assertEqual(result["source_row_counts"], {role: 4 for role in t.OPEN_ROLES})
                self.assertEqual(result["distributions"]["all_open"]["nonpadding_tokens"]["count"], 12)
                self.assertFalse(result["e1_launch_ready"])
                self.assertFalse(result["training_eligibility"])
                self.assertEqual(result["environment"]["USE_TORCH"], "0")
                with self.assertRaises(ValueError):
                    t.tokenize_dataset(source, tokenizer_receipt, root / "out")
            for role in t.OPEN_ROLES:
                self.assertEqual(source_hashes[role], t.file_identity(source / (role + ".jsonl")))
            for name, identity in result["files"].items():
                self.assertEqual(identity, t.file_identity(root / "out" / name))

    def test_source_file_tampering_fails_before_loading_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, tokenizer_receipt = fixtures(root)
            with (source / "DEV.jsonl").open("ab") as handle:
                handle.write(b"\n")
            with patch.object(t, "load_tokenizer", side_effect=AssertionError("loaded before source checks")):
                with self.assertRaisesRegex(ValueError, "source file drift"):
                    t.tokenize_dataset(source, tokenizer_receipt, root / "out")
            self.assertFalse((root / "out").exists())

    def test_tokenizer_file_drift_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, tokenizer_receipt = fixtures(root)
            (root / "tokenizer" / "vocab.json").write_bytes(b"changed")
            with patch.object(t, "load_tokenizer", side_effect=AssertionError("loaded drifted tokenizer")):
                with self.assertRaisesRegex(ValueError, "tokenizer file drift"):
                    t.tokenize_dataset(source, tokenizer_receipt, root / "out")

    def test_runtime_template_drift_creates_no_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, tokenizer_receipt = fixtures(root)
            tokenizer = FakeTokenizer()
            tokenizer.chat_template = "drifted"
            with patch.object(t, "load_tokenizer", return_value=tokenizer), self.assertRaises(ValueError):
                t.tokenize_dataset(source, tokenizer_receipt, root / "out")
            self.assertFalse((root / "out").exists())

    def test_measured_distribution_uses_nearest_rank(self):
        result = t.distribution(list(range(1, 101)))
        self.assertEqual((result["sum"], result["p50"], result["p95"], result["p99"]), (5050, 50, 95, 99))
        self.assertIsNone(t.distribution([])["max"])


if __name__ == "__main__":
    unittest.main()
