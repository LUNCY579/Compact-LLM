"""Synthetic tokenizer checks; no production payload, native model or network."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import retention_data as r
from experiments.post_training_mimo_centralized_v1.scripts import tokenize_retention as t
from experiments.post_training_mimo_centralized_v1.tests.test_retention_data import registration
from experiments.post_training_mimo_centralized_v1.tests.test_tokenize_bridge import FakeTokenizer


def compact_spec(domain, seed, index):
    if domain == "instruction_following":
        return {"domain": domain, "template": r.FORMATS[index % 4], "directives": [
            {"priority": "base", "slot": "recipient", "text": "A"},
            {"priority": "base", "slot": "message", "text": "B"},
            {"priority": "base", "slot": "note", "text": "N" + str(index)},
            {"priority": "final", "slot": "message", "text": "C"},
        ]}
    return {"domain": domain, "records": [
        {"entity": "E" + str(index), "field": "f", "value": "A"},
        {"entity": "F" + str(index), "field": "g", "value": "B"},
        {"entity": "Z" + str(index), "field": domain, "value": "C"}],
        "queries": [{"entity": "E" + str(index), "field": "f"},
                    {"entity": "F" + str(index), "field": "g"}]}


def compact_parent(domain="retrieval", role="TRAIN"):
    for index in range(10000):
        parent = r.allocate_parent(compact_spec(domain, 913, index), 913, index)
        if parent.role == role:
            return parent
    raise AssertionError("missing synthetic role")


def fixtures(root):
    reg = registration()
    reg.update(context_ceiling=1024, completion_ceiling_including_eos=128)
    reg_path = root / "registration.json"
    reg_path.write_bytes(r.json_bytes(reg))
    source = root / "source"
    with patch.object(r, "generate_spec", side_effect=compact_spec):
        r.build_dataset(reg_path, source, "a" * 40)
    model = root / "tokenizer"
    model.mkdir()
    for name in t.TOKENIZER_FILES:
        value = {"chat_template": FakeTokenizer.chat_template} if name == "tokenizer_config.json" else {"synthetic": name}
        (model / name).write_bytes(r.json_bytes(value))
    token_receipt = root / "TOKENIZER_RECEIPT.json"
    token_receipt.write_bytes(r.json_bytes({
        "checkpoint_path": str(model), "tokenizer_files": {name: t.file_identity(model / name) for name in t.TOKENIZER_FILES},
        "chat_template_sha256": r.sha256(FakeTokenizer.chat_template.encode()),
    }))
    return source, token_receipt


def rewrite_role(source, role, rows):
    path = source / (role + ".jsonl")
    path.write_bytes(b"".join(r.json_bytes(row) + b"\n" for row in rows))
    receipt_path = source / "DATA_RECEIPT.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["files"][path.name] = t.file_identity(path)
    receipt["open_row_counts"][role] = len(rows)
    receipt_path.write_bytes(r.json_bytes(receipt))


class RetentionRowTests(unittest.TestCase):
    def test_twins_keep_parent_distinct_rows_and_completion_eos(self):
        parent = compact_parent("answerability_abstention")
        rows = [r.make_row(parent, "a" * 40, variant) for variant in ("answerable", "missing_evidence")]
        actual = [t.tokenize_row(row, FakeTokenizer()) for row in rows]
        self.assertEqual(actual[0]["parent_id"], actual[1]["parent_id"])
        self.assertNotEqual(actual[0]["row_id"], actual[1]["row_id"])
        self.assertNotEqual(actual[0]["tokenized_prompt_sha256"], actual[1]["tokenized_prompt_sha256"])
        for source, row in zip(rows, actual):
            self.assertEqual(row["row_id"], source["row_id"])
            self.assertIsInstance(row["variant_id"], str)
            self.assertEqual(row["labels"][-1], 151645)
            self.assertEqual(row["completion_mask"][-1], 1)
            self.assertEqual(row["labels"][:row["prompt_tokens"]], [-100] * row["prompt_tokens"])
            self.assertFalse(row["training_eligibility"])
            self.assertIsNone(row["source_correct_membership"])

    def test_source_identity_and_string_variant_are_verified(self):
        good = r.make_row(compact_parent(), "a" * 40)
        for key, value in (("variant_id", 0), ("row_id", "changed"), ("parent_id", "changed"),
                           ("split_sha256", "0" * 64), ("family", "answerability_abstention"),
                           ("role", "CHECK"), ("canonical_parent_sha256", "0" * 64)):
            row = copy.deepcopy(good)
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                t.tokenize_row(row, FakeTokenizer())

    def test_source_answer_and_prompt_cannot_be_rebound_by_hash_only(self):
        good = r.make_row(compact_parent(), "a" * 40)
        for key, text in (("target", '{"answer":null}'), ("prompt", "Unrelated prompt")):
            row = copy.deepcopy(good)
            row[key] = text
            row[key + "_sha256"] = r.sha256(text.encode())
            if key == "prompt":
                row["normalized_prompt_sha256"] = r.sha256(" ".join(text.split()).encode())
            with self.subTest(key=key), self.assertRaises(ValueError):
                t.tokenize_row(row, FakeTokenizer())

    def test_completion_boundary_and_full_template_are_checked(self):
        row = r.make_row(compact_parent(), "a" * 40)
        class ChangedFull(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
                value = super().apply_chat_template(messages, tokenize, add_generation_prompt, enable_thinking)
                if len(messages) > 1:
                    return value + ([1000] if tokenize else "x")
                return value
        with self.assertRaisesRegex(ValueError, "assistant template"):
            t.tokenize_row(row, ChangedFull())

    def test_oversize_is_rejected_by_unchanged_helper(self):
        row = r.make_row(compact_parent(), "a" * 40)
        class LargePrompt(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
                value = super().apply_chat_template(messages, False, add_generation_prompt, enable_thinking)
                value = "x" * 1024 + value
                return self.encode(value) if tokenize else value
        with self.assertRaisesRegex(ValueError, "1024.*truncation forbidden"):
            t.tokenize_row(row, LargePrompt())
        self.assertIs(t.assemble_example, t.bridge_tokens.assemble_example)
        self.assertIs(t.load_tokenizer, t.bridge_tokens.load_tokenizer)
        self.assertIs(t.distribution, t.bridge_tokens.distribution)
        self.assertIs(t.TEMPLATE_KWARGS, t.bridge_tokens.TEMPLATE_KWARGS)
        with self.assertRaisesRegex(ValueError, "128.*truncation forbidden"):
            t.assemble_example([1], [2] * 128 + [151645])


class RetentionDatasetTests(unittest.TestCase):
    def test_open_roles_only_never_read_sealed_payload_preserve_twins_write_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, token_receipt = fixtures(root)
            forbidden = {source / name for name in ("CHECK.jsonl", "FINAL.jsonl", "SEALED_RESERVATIONS.jsonl")}
            # Content is deliberately invalid after the source receipt was bound.
            # The tokenizer may copy the recorded hash, but may not read/hash it.
            for path in forbidden:
                path.write_text("MUST NEVER BE READ", encoding="utf-8")
            original = Path.open
            def guarded(path, *args, **kwargs):
                if path in forbidden:
                    raise AssertionError("sealed payload accessed")
                return original(path, *args, **kwargs)
            output = root / "out"
            with patch.object(Path, "open", guarded), patch.object(t, "load_tokenizer", return_value=FakeTokenizer()):
                receipt = t.tokenize_dataset(source, token_receipt, output)
            self.assertEqual(receipt["source_role_parent_counts"], {"TRAIN": 12, "DEV": 12})
            self.assertEqual(receipt["source_row_counts"], {"TRAIN": 16, "DEV": 16})
            self.assertEqual(receipt["twins"]["parent_counts"], {"TRAIN": 4, "DEV": 4})
            self.assertEqual(receipt["dependency_files"]["tokenize_bridge.py"], t.file_identity(Path(t.bridge_tokens.__file__)))
            self.assertFalse(receipt["sealed_access"]["reservation_payload_opened"])
            self.assertFalse(receipt["sealed_access"]["independently_hashed"])
            self.assertEqual(set(receipt["files"]), {"TRAIN.tokenized.jsonl", "DEV.tokenized.jsonl"})
            for role in t.OPEN_ROLES:
                rows = [json.loads(line) for line in (output / (role + ".tokenized.jsonl")).read_text().splitlines()]
                self.assertEqual(len(rows), 16)
                self.assertEqual(receipt["distributions"]["by_role"][role]["nonpadding_tokens"]["sum"],
                                 sum(row["nonpadding_tokens"] for row in rows))
                self.assertEqual(receipt["distributions"]["by_role_domain"][role]["answerability_abstention"]["completion_tokens"]["count"], 8)
            for key in ("training_eligibility", "e1_launch_ready", "retention_dataset_qualified", "native_qualification_inferred",
                        "language_preservation_qualified", "long_context_preservation_qualified", "source_correct_enrollment_qualified"):
                self.assertFalse(receipt[key])
            with self.assertRaisesRegex(ValueError, "write-once"):
                t.tokenize_dataset(source, token_receipt, output)

    def test_duplicate_twin_cannot_replace_missing_variant(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, token_receipt = fixtures(root)
            rows = [json.loads(line) for line in (source / "TRAIN.jsonl").read_text().splitlines()]
            missing_index = next(i for i, row in enumerate(rows) if row["variant_id"] == "missing_evidence")
            twin = next(row for row in rows if row["parent_id"] == rows[missing_index]["parent_id"] and row["variant_id"] == "answerable")
            rows[missing_index] = copy.deepcopy(twin)
            rewrite_role(source, "TRAIN", rows)
            with patch.object(t, "load_tokenizer", side_effect=AssertionError("must fail before tokenizer")), \
                 self.assertRaisesRegex(ValueError, "duplicate retention row ID"):
                t.tokenize_dataset(source, token_receipt, root / "out")
            self.assertFalse((root / "out").exists())

    def test_missing_twin_is_rejected_not_silently_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, token_receipt = fixtures(root)
            rows = [json.loads(line) for line in (source / "TRAIN.jsonl").read_text().splitlines()]
            rows.pop(next(i for i, row in enumerate(rows) if row["variant_id"] == "missing_evidence"))
            rewrite_role(source, "TRAIN", rows)
            with patch.object(t, "load_tokenizer", side_effect=AssertionError("must fail before tokenizer")), \
                 self.assertRaisesRegex(ValueError, "domain row count"):
                t.tokenize_dataset(source, token_receipt, root / "out")
            self.assertFalse((root / "out").exists())

    def test_source_file_and_generator_hash_drift_fail_before_tokenizer(self):
        for kind in ("file", "generator"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, token_receipt = fixtures(root)
                if kind == "file":
                    with (source / "TRAIN.jsonl").open("ab") as handle:
                        handle.write(b"\n")
                else:
                    path = source / "DATA_RECEIPT.json"
                    receipt = json.loads(path.read_bytes())
                    receipt["generator_file_sha256"] = "0" * 64
                    path.write_bytes(r.json_bytes(receipt))
                with patch.object(t, "load_tokenizer", side_effect=AssertionError("must fail before tokenizer")), \
                     self.assertRaisesRegex(ValueError, "drift"):
                    t.tokenize_dataset(source, token_receipt, root / "out")
                self.assertFalse((root / "out").exists())

    def test_tokenizer_asset_and_runtime_template_drift(self):
        for kind in ("asset", "runtime"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, token_receipt = fixtures(root)
                tokenizer = FakeTokenizer()
                if kind == "asset":
                    (root / "tokenizer" / "vocab.json").write_text("drift", encoding="utf-8")
                else:
                    tokenizer.chat_template = "different-template"
                with patch.object(t, "load_tokenizer", return_value=tokenizer), self.assertRaisesRegex(ValueError, "drift"):
                    t.tokenize_dataset(source, token_receipt, root / "out")
                self.assertFalse((root / "out").exists())

    def test_empty_open_population_cannot_claim_token_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reg = registration()
            reg.update(context_ceiling=1024, completion_ceiling_including_eos=128)
            for role in t.OPEN_ROLES:
                reg["counts"][role] = {domain: 0 for domain in r.DOMAINS}
            reg_path = root / "registration.json"
            reg_path.write_bytes(r.json_bytes(reg))
            with patch.object(r, "generate_spec", side_effect=compact_spec):
                r.build_dataset(reg_path, root / "source", "a" * 40)
            token_receipt = root / "TOKENIZER_RECEIPT.json"
            token_receipt.write_text("{}", encoding="utf-8")
            with patch.object(t, "load_tokenizer", side_effect=AssertionError("must fail before tokenizer")), \
                 self.assertRaisesRegex(ValueError, "no open retention parents"):
                t.tokenize_dataset(root / "source", token_receipt, root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
