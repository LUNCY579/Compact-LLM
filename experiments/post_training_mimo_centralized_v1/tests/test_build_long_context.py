"""Synthetic tokenizer mechanics only: no native weights, network, or GPU."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import build_long_context as b
from experiments.post_training_mimo_centralized_v1.scripts import long_context_tasks as tasks


class SyntheticTokenizer:
    """Deterministic word/punctuation offsets with explicit native special IDs."""
    eos_token_id = 151645
    eos_token = "<|im_end|>"
    pad_token_id = 151643
    chat_template = "synthetic-source-interface-fixture-v1"
    all_special_ids = [151643, 151644, 151645]
    pattern = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|[ \t]?[A-Za-z]+|[0-9]|\n|[^\nA-Za-z0-9]")

    def __init__(self):
        self.vocab = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645}
        self.inverse = {value: key for key, value in self.vocab.items()}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if add_special_tokens:
            raise AssertionError("extra special tokens forbidden")
        matches = list(self.pattern.finditer(text))
        if "".join(match.group() for match in matches) != text:
            raise AssertionError("fixture tokenizer dropped characters")
        ids = []
        for match in matches:
            token = match.group()
            if token not in self.vocab:
                index = len(self.vocab) + 100
                self.vocab[token] = index
                self.inverse[index] = token
            ids.append(self.vocab[token])
        value = {"input_ids": ids}
        if return_offsets_mapping:
            value["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        return value

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            raise AssertionError("special-token removal forbidden")
        return "".join(self.inverse[index] for index in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        if enable_thinking is not False or add_generation_prompt is not True or len(messages) != 1:
            raise AssertionError("template contract changed")
        text = ("<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return self.encode(text) if tokenize else text


def registration():
    return {"schema_version": 1, "status": "FROZEN_FAST_LANE_LONG_CONTEXT_ONLY", "selection_seed": 20260928,
            "counts": {"DEV": 32, "CHECK": 64, "FINAL": 0}, "prompt_lengths": [2024, 4072],
            "generation_reserve_tokens": 24, "max_target_tokens": 16,
            "families": list(tasks.FAMILIES), "positions": list(tasks.POSITIONS),
            "fit_policy": dict(b.FIT_POLICY), "template_kwargs": dict(b.TEMPLATE_KWARGS),
            "primary_student_weight_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
            "chat_template_sha256": "c" * 64,
            "wall_seconds": 1800,
            "authorization": {"dataset_artifact_generation": True, "native_execution": False, "training": False,
                              "check_rendering": False, "final_rendering": False},
            "model_outcome_based_selection": False}


def fixtures(root):
    reg = root / "registration.json"
    reg.write_bytes(b.json_bytes(registration()))
    model_dir = root / "tokenizer"
    model_dir.mkdir()
    for name in b.TOKENIZER_FILES:
        (model_dir / name).write_bytes(b.json_bytes({"chat_template": SyntheticTokenizer.chat_template})
                                      if name == "tokenizer_config.json" else name.encode())
    receipt = root / "TOKENIZER_RECEIPT.json"
    receipt.write_bytes(b.json_bytes({"checkpoint_path": str(model_dir),
                                    "weight_sha256": "a" * 64,
                                    "tokenizer_files": {name: b.file_identity(model_dir / name) for name in b.TOKENIZER_FILES},
                                    "chat_template_sha256": b.sha256(SyntheticTokenizer.chat_template.encode())}))
    value = registration()
    value.update(primary_student_weight_sha256="a" * 64,
                 tokenizer_sha256=b.file_identity(model_dir / "tokenizer.json")["sha256"],
                 chat_template_sha256=b.sha256(SyntheticTokenizer.chat_template.encode()))
    reg.write_bytes(b.json_bytes(value))
    return reg, receipt


class LongContextRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.allocation = tasks.allocate_bank()

    def test_every_family_position_exact_final_length_and_all_support_spans(self):
        tokenizer = SyntheticTokenizer()
        fitter = b.NoteFitter(tokenizer)
        seen = set()
        for parent in self.allocation["dev"]:
            key = parent.family, parent.placement
            if key in seen:
                continue
            seen.add(key)
            rows = b.render_parent(parent, tokenizer, fitter)
            self.assertEqual(len(rows), 8)
            b.validate_parent_rows(rows)
            for row in rows:
                self.assertEqual(len(row["input_ids"]), row["prompt_tokens"])
                self.assertIn(row["prompt_tokens"], (2024, 4072))
                self.assertEqual(row["total_sequence_envelope"], row["prompt_tokens"] + 24)
                self.assertLessEqual(row["target_tokens"], 16)
                self.assertEqual(row["visible_text_oracle"], row["target"])
                self.assertEqual(row["target"], tasks.reference_answer(parent, row["answerable"]))
                for span in row["support_spans"]:
                    lo, hi = span["interval"]
                    self.assertGreaterEqual(span["body_token_start"], lo * row["body_tokens"])
                    self.assertLessEqual(span["body_token_end"], hi * row["body_tokens"])
                    self.assertEqual(row["rendered_prompt"][span["whole_prompt_char_start"]:span["whole_prompt_char_end"]], span["text"])
                    self.assertEqual(row["input_ids"][span["whole_prompt_token_start"]:span["whole_prompt_token_end"]], span["token_ids"])
                present = sum(span["required_fact_present"] for span in row["support_spans"])
                self.assertEqual(present, 3 if row["answerable"] else 2)
                self.assertEqual(row["policy_input_only"], ["prompt", "input_ids"])
                self.assertFalse(row["training_eligibility"])
        self.assertEqual(len(seen), 8)

    def test_query_decoys_and_filler_do_not_leak_target_but_remaining_candidates_are_allowed(self):
        tokenizer = SyntheticTokenizer()
        parent = next(p for p in self.allocation["dev"] if p.missing_index != 2)
        rows = b.render_parent(parent, tokenizer)
        target = tasks.reference_answer(parent)
        for row in rows:
            self.assertNotIn(target, row["query"])
            self.assertNotIn(target, tasks.SYSTEM_PROMPT)
            if not row["answerable"]:
                self.assertEqual(row["target"], "UNKNOWN")
                self.assertEqual(tasks.solve_visible(row["query"], row["context_body"]), "UNKNOWN")
        self.assertTrue(any(target in r["context_body"] for r in rows if not r["answerable"]))

    def test_offset_tampering_rehashed_row_still_fails(self):
        tokenizer = SyntheticTokenizer()
        row = b.render_parent(self.allocation["dev"][0], tokenizer)[0]
        for key in ("whole_prompt_token_start", "body_token_start", "whole_prompt_char_start"):
            changed = copy.deepcopy(row)
            changed["support_spans"][0][key] += 1
            changed["row_sha256"] = b.sha256(b.json_bytes({k: v for k, v in changed.items() if k != "row_sha256"}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "span|substring"):
                b.validate_row(changed, tokenizer)

    def test_native_offsets_are_required_not_character_estimates(self):
        class BadOffsets(SyntheticTokenizer):
            def __call__(self, *args, **kwargs):
                result = super().__call__(*args, **kwargs)
                if "offset_mapping" in result:
                    result["offset_mapping"] = [(0, 0)] * len(result["input_ids"])
                return result
        with self.assertRaisesRegex(ValueError, "span"):
            b.render_parent(self.allocation["dev"][0], BadOffsets())

    def test_rehashed_target_cap_and_support_receipt_tampering_fail(self):
        tokenizer = SyntheticTokenizer()
        row = b.render_parent(self.allocation["dev"][0], tokenizer)[0]
        mutations = (("target_token_ids", [55]), ("target_tokens", 17), ("target_token_ids_sha256", "0" * 64),
                     ("generation_reserve_tokens", 23), ("total_sequence_envelope", 4096), ("support_spans", []),
                     ("evidence_position", "distributed"))
        for key, value in mutations:
            changed = copy.deepcopy(row)
            changed[key] = value
            changed["row_sha256"] = b.sha256(b.json_bytes({k: v for k, v in changed.items() if k != "row_sha256"}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                b.validate_row(changed, tokenizer)

    def test_padding_specials_and_sealed_parent_rendering_are_rejected(self):
        for text in ("NOTE <|im_end|>.", "NOTE <|im_start|>.", "NOTE <|endoftext|>."):
            with self.subTest(text=text), patch.object(tasks, "FILLER_TEXTS", (text,)), self.assertRaisesRegex(ValueError, "irrelevant prose tokens"):
                b.NoteFitter(SyntheticTokenizer())
        for role in ("CHECK", "FINAL", "TRAIN"):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "only DEV"):
                b.render_parent(replace(self.allocation["dev"][0], role=role), SyntheticTokenizer())

    def test_bounded_fitter_and_oversize_geometry_fail_without_truncation(self):
        tokenizer = SyntheticTokenizer()
        fitter = b.NoteFitter(tokenizer)
        self.assertEqual(fitter.fill(0), "")
        for count in (30, 199, 1000, 4072):
            self.assertEqual(len(tokenizer.encode(fitter.fill(count))), count)
        with self.assertRaisesRegex(ValueError, "cannot fill"):
            fitter.fill(4073)
        with self.assertRaisesRegex(ValueError, "interval"):
            b._layout(fitter, 1800, "early", [100, 100, 100], 0)

    def test_incomplete_factorial_and_changed_other_facts_fail(self):
        rows = b.render_parent(self.allocation["dev"][0], SyntheticTokenizer())
        with self.assertRaisesRegex(ValueError, "eight"):
            b.validate_parent_rows(rows[:-1])
        changed = copy.deepcopy(rows)
        changed[1]["support_spans"][0]["body_token_end"] += 1
        with self.assertRaisesRegex(ValueError, "facts"):
            b.validate_parent_rows(changed)
        for kind in ("support", "target"):
            changed = copy.deepcopy(rows)
            for row in changed:
                if row["prompt_tokens"] == 4072:
                    if kind == "support":
                        row["support_spans"][0]["text"] += " changed"
                    elif row["answerable"]:
                        row["target"] = "SZZZZZ"
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "across lengths"):
                b.validate_parent_rows(changed)

    def test_pinned_loader_and_exact_template_interface_are_reused(self):
        self.assertIs(b.load_tokenizer, b.bridge_tokens.load_tokenizer)
        self.assertIs(b.TEMPLATE_KWARGS, b.bridge_tokens.TEMPLATE_KWARGS)
        self.assertEqual(b.TEMPLATE_KWARGS, {"enable_thinking": False, "add_generation_prompt": True})
        class WrongTemplate(SyntheticTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                result = super().apply_chat_template(*args, **kwargs)
                return result + ("x" if isinstance(result, str) else [5])
        with self.assertRaisesRegex(ValueError, "prefix"):
            b.render_parent(self.allocation["dev"][0], WrongTemplate())


class LongContextProductionGateTests(unittest.TestCase):
    def setUp(self):
        isolated_sys = SimpleNamespace(modules={}, version=sys.version, executable=sys.executable)
        guard = patch.object(b, "sys", isolated_sys)
        guard.start()
        self.addCleanup(guard.stop)

    def test_no_torch_guard_precedes_all_file_reads(self):
        b.sys.modules["torch"] = object()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(b, "read_json", side_effect=AssertionError("read too early")), self.assertRaisesRegex(ValueError, "already contain torch"):
                b.build_dataset(root / "reg", root / "receipt", root / "out", "a" * 40)
            self.assertFalse((root / "out").exists())

    def test_proposal_uncommitted_source_and_drift_fail_before_tokenizer(self):
        for failure in ("proposal", "uncommitted", "tokenizer"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                reg, receipt = fixtures(root)
                if failure == "proposal":
                    value = registration()
                    value["status"] = "PROPOSED_UNQUALIFIED"
                    reg.write_bytes(b.json_bytes(value))
                if failure == "tokenizer":
                    (root / "tokenizer" / "vocab.json").write_text("drift")
                with patch.object(b, "_committed_source", side_effect=ValueError("uncommitted") if failure == "uncommitted" else None,
                                  return_value={"test_source": "bound"}), \
                     patch.object(b, "load_tokenizer", side_effect=AssertionError("tokenizer loaded before gates")), self.assertRaises(ValueError):
                    b.build_dataset(reg, receipt, root / "out", "a" * 40)
                self.assertFalse((root / "out").exists())

    def test_self_consistent_substituted_tokenizer_receipt_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reg, receipt = fixtures(root)
            (root / "tokenizer" / "tokenizer.json").write_text("substituted tokenizer")
            value = json.loads(receipt.read_bytes())
            value["tokenizer_files"]["tokenizer.json"] = b.file_identity(root / "tokenizer" / "tokenizer.json")
            receipt.write_bytes(b.json_bytes(value))
            with patch.object(b, "_committed_source", return_value={"source": "bound"}), \
                 patch.object(b, "load_tokenizer", side_effect=AssertionError("must fail before load")), \
                 self.assertRaisesRegex(ValueError, "registered tokenizer"):
                b.build_dataset(reg, receipt, root / "out", "a" * 40)

    def test_wall_cap_is_frozen_and_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reg, receipt = fixtures(root)
            with patch.object(b, "_committed_source", return_value={"source": "bound"}), \
                 patch.object(b, "load_tokenizer", return_value=SyntheticTokenizer()), \
                 patch.object(b.time, "perf_counter", side_effect=[0.0, 1801.0, 1801.1]), \
                 self.assertRaisesRegex(ValueError, "wall ceiling"):
                b.build_dataset(reg, receipt, root / "out", "a" * 40)
            failure = json.loads((root / "out" / "FAILED_CONSTRUCTION.json").read_bytes())
            self.assertEqual(failure["completed_rows"], 0)
            bad = registration()
            bad["wall_seconds"] += 1
            with self.assertRaisesRegex(ValueError, "wall ceiling"):
                b.validate_registration(bad)

    def test_failed_parent_records_once_no_skip_or_bank_and_blocks_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reg, receipt = fixtures(root)
            output = root / "out"
            with patch.object(b, "_committed_source", return_value={"source": "bound"}), \
                 patch.object(b, "load_tokenizer", return_value=SyntheticTokenizer()), \
                 patch.object(b, "render_parent", side_effect=ValueError("bounded fit failure")) as render, self.assertRaisesRegex(ValueError, "bounded fit"):
                b.build_dataset(reg, receipt, output, "a" * 40)
            self.assertEqual(render.call_count, 1)
            failure = json.loads((output / "FAILED_CONSTRUCTION.json").read_bytes())
            self.assertFalse(failure["automatic_retry"])
            self.assertFalse(failure["parent_replacement"])
            self.assertEqual(failure["completed_rows"], 0)
            self.assertFalse((output / "DATA_RECEIPT.json").exists())
            self.assertFalse((output / "DEV.jsonl").exists())
            with self.assertRaisesRegex(ValueError, "write-once"):
                b.build_dataset(reg, receipt, output, "a" * 40)

    def test_full_open_bank_and_hash_only_reservations_without_sealed_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reg, receipt = fixtures(root)
            output = root / "out"
            original = Path.open
            def guard(path, *args, **kwargs):
                if path.name in ("CHECK.jsonl", "FINAL.jsonl"):
                    raise AssertionError("sealed payload opened")
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", guard), patch.object(b, "_committed_source", return_value={"source": "bound"}), \
                 patch.object(b, "load_tokenizer", return_value=SyntheticTokenizer()), \
                 patch.object(b.importlib.metadata, "version", return_value="synthetic-test-only"):
                result = b.build_dataset(reg, receipt, output, "a" * 40)
            self.assertEqual(result["parent_counts"], {"DEV": 32, "CHECK": 64, "FINAL": 0})
            self.assertEqual(result["open_row_counts"], {"DEV": 256, "CHECK": 0, "FINAL": 0})
            self.assertFalse(result["sealed_access"]["payloads_opened"])
            self.assertEqual(result["cost"]["model_calls"], 0)
            self.assertFalse(result["long_context_preservation_qualified"])
            rows = [json.loads(line) for line in (output / "DEV.jsonl").read_bytes().splitlines()]
            self.assertEqual(len(rows), 256)
            self.assertEqual(len({row["parent_id"] for row in rows}), 32)
            reservations = [json.loads(line) for line in (output / "SEALED_RESERVATIONS.jsonl").read_bytes().splitlines()]
            self.assertEqual(len(reservations), 64)
            for reserved in reservations:
                self.assertNotIn("prompt", reserved)
                self.assertNotIn("target", reserved)
                self.assertNotIn("canonical_json", reserved)


if __name__ == "__main__":
    unittest.main()
