"""Staged standard-library controls; fake tokenizer only, no root source edits."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streaming_inputs as adapter


class Tokenizer:
    def __init__(self): self.binding, self.calls, self.drift, self.encoding_fault = {"fixture": "CPU_FAKE_ONLY"}, [], False, None
    def verify_binding(self): return {"fixture": "CHANGED"} if self.drift else deepcopy(self.binding)
    def encode_prompt(self, messages):
        self.calls.append(deepcopy(messages))
        count = int(messages[0]["content"].split(":")[1])
        result = [5] * count
        return self.encoding_fault(result) if self.encoding_fault else result


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=adapter.WORK)
        self.root = Path(self.directory.name)
        self.source, self.output = self.root / "source", self.root / "output"
        self.source.mkdir()
        self.tokenizer = Tokenizer()
        self.rows = []
        for index, length in enumerate((992, 2016, 4064)):
            ids = [5] * length
            self.rows.append({"row_id": adapter.digest([index, length]), "messages": [{"role": "user", "content": "visible:%d" % length}],
                              "input_ids": ids, "attention_mask": [1] * length, "prompt_tokens": length, "prompt_token_sha256": adapter.digest(ids)})
        self.receipt = {"constructed_rows": 3, "planned_rows": 3, "blocked_variant_rows": 0, "blockers": [],
                        "all_worlds_retained": True, "generation_reserve_tokens": 32, "tokenizer_binding": self.tokenizer.binding}
        for flag in ("native_long_context_profile_qualified", "source_correct_membership_enrolled", "statistical_preservation_qualified",
                     "confirmation_eligible", "native_resource_qualification_inferred"): self.receipt[flag] = False
        self.patch_rows = patch.object(adapter, "SOURCE_ROWS", 3)
        self.patch_work = patch.object(adapter, "WORK", self.root)
        self.patch_rows.start(); self.patch_work.start()
        (self.source / "catalog.json").write_text('{"CPU_public_metadata_only":true}\n')
        self.sign()

    def tearDown(self):
        self.patch_work.stop(); self.patch_rows.stop()
        self.directory.cleanup()

    def sign(self, raw=None):
        (self.source / "policy_inputs.jsonl").write_bytes(raw if raw is not None else b"".join((adapter.canonical(row) + "\n").encode("ascii") for row in self.rows))
        self.receipt["policy_file_sha256"] = adapter.file_binding(self.source / "policy_inputs.jsonl")["sha256"]
        (self.source / "receipt.json").write_text(adapter.canonical(self.receipt) + "\n", encoding="ascii")
        self.pins = {name: adapter.file_binding(self.source / name) for name in adapter.SOURCE_PINS}

    def validate(self):
        return adapter.validate_packet(self.source, tokenizer=self.tokenizer, _fixture_pins=self.pins)

    def test_exact_envelopes_output_denominator_order_and_no_oracle_fields(self):
        admission, tokenizer = self.validate()
        self.assertFalse(self.output.exists())
        receipt = adapter.materialize(admission, tokenizer, self.output)
        with (self.output / "prompt_arrays.jsonl").open("rb") as stream:
            outputs = [json.loads(line) for line in stream]
        self.assertEqual([992, 2016, 4064], [row["prompt_tokens"] for row in outputs])
        self.assertEqual([1024, 2048, 4096], [row["total_sequence_envelope"] for row in outputs])
        self.assertEqual([row["row_id"] for row in self.rows], [row["row_id"] for row in outputs])
        self.assertEqual(3, receipt["rows"])
        self.assertEqual("CPU_FIXTURE", receipt["admission_kind"])
        self.assertEqual(0, receipt["rows_dropped"])
        self.assertTrue(receipt["whole_packet_validated_before_output"])
        self.assertFalse(receipt["oracle_inputs_read"])
        self.assertFalse(receipt["native_execution_authorized"])
        self.assertFalse(receipt["human_language_qualification_established"])
        self.assertEqual(adapter.file_binding(self.output / "prompt_arrays.jsonl"), receipt["arrays"])
        self.assertTrue(all("messages" not in row and "target" not in row and "oracle" not in row for row in outputs))

    def test_unknown_private_fields_in_last_row_fail_before_output(self):
        self.rows[-1]["oracle"] = {"answer": "PRIVATE"}
        self.sign()
        with self.assertRaisesRegex(ValueError, "allowlist"): self.validate()
        self.assertFalse(self.output.exists())

    def test_last_prompt_overflow_cannot_truncate_drop_or_start_processing(self):
        self.rows[-1].update(prompt_tokens=4065, input_ids=[5] * 4065, attention_mask=[1] * 4065,
                            prompt_token_sha256=adapter.digest([5] * 4065))
        self.sign()
        with self.assertRaisesRegex(ValueError, "envelope"): self.validate()
        self.assertFalse(self.output.exists())

    def test_masks_padding_boolean_ids_and_prompt_only_message_boundary(self):
        mutations = (lambda row: row["attention_mask"].__setitem__(0, 0),
                     lambda row: row["attention_mask"].__setitem__(0, True),
                     lambda row: row["input_ids"].__setitem__(0, True),
                     lambda row: row["input_ids"].__setitem__(0, 151643),
                     lambda row: row["messages"].append({"role": "assistant", "content": "target"}))
        for change in mutations:
            original = deepcopy(self.rows)
            change(self.rows[-1]); self.sign()
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate()
            self.rows = original

    def test_malformed_utf8_nonfinite_duplicate_keys_and_blank_or_partial_lines(self):
        raws = (b'bad\n', b'\xff\n', b'{"x":NaN}\n', b'{"x":Infinity}\n', b'{"x":1e999}\n',
                b'{"row_id":1,"row_id":2}\n', b'\n', adapter.canonical(self.rows[0]).encode("ascii"))
        for raw in raws:
            self.sign(raw)
            with self.subTest(raw=raw[:30]), self.assertRaises(ValueError): self.validate()
            self.assertFalse(self.output.exists())

    def test_line_packet_and_metadata_byte_limits_are_enforced_before_parsing(self):
        self.sign(b"x" * (adapter.MAX_LINE + 1) + b"\n")
        with self.assertRaisesRegex(ValueError, "line bound"): self.validate()
        with patch.object(adapter, "MAX_PACKET", 10):
            with self.assertRaises(ValueError): list(adapter.lines(self.source / "policy_inputs.jsonl"))
        with patch.object(adapter, "MAX_METADATA", 10):
            with self.assertRaisesRegex(ValueError, "metadata byte"): adapter.metadata(self.source / "receipt.json")

    def test_wrong_file_or_token_checksum_and_duplicate_row_id(self):
        self.pins["policy_inputs.jsonl"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"): self.validate()
        self.rows[-1]["prompt_token_sha256"] = "0" * 64; self.sign()
        with self.assertRaisesRegex(ValueError, "token checksum"): self.validate()
        self.rows[-1]["prompt_token_sha256"] = adapter.digest(self.rows[-1]["input_ids"])
        self.rows[-1]["row_id"] = self.rows[0]["row_id"]; self.sign()
        with self.assertRaisesRegex(ValueError, "duplicate source"): self.validate()

    def test_token_binding_and_tokenizer_drift_before_or_during_validation(self):
        self.tokenizer.encoding_fault = lambda ids: [6] + ids[1:]
        with self.assertRaisesRegex(ValueError, "token binding"): self.validate()
        self.tokenizer.encoding_fault = None
        self.tokenizer.drift = True
        with self.assertRaisesRegex(ValueError, "receipt binding"): self.validate()
        self.tokenizer.drift = False
        original = self.tokenizer.encode_prompt
        def mutate(messages):
            result = original(messages)
            self.tokenizer.drift = True
            return result
        self.tokenizer.encode_prompt = mutate
        with self.assertRaisesRegex(ValueError, "during whole-packet"): self.validate()

    def test_source_order_or_bytes_drift_after_admission_rejected_before_output(self):
        admission, tokenizer = self.validate()
        self.rows.reverse(); self.sign()
        with self.assertRaisesRegex(ValueError, "checksum"): adapter.materialize(admission, tokenizer, self.output)
        self.assertFalse(self.output.exists())

    def test_drift_during_output_preserves_failure_and_no_admission_receipt(self):
        admission, tokenizer = self.validate()
        calls = [0]
        def guard():
            calls[0] += 1
            if calls[0] == 2:
                self.rows[-1]["row_id"] = adapter.digest("CHANGED")
                self.sign()
        with self.assertRaisesRegex(ValueError, "order/content drift|checksum"):
            adapter.materialize(admission, tokenizer, self.output, resource_guard=guard)
        self.assertTrue((self.output / "failure.json").is_file())
        self.assertFalse((self.output / "receipt.json").exists())
        failure = adapter.metadata(self.output / "failure.json")
        self.assertFalse(failure["native_execution_authorized"])
        # A bounded buffer may retain earlier valid bytes after file mutation.
        # The final source checksum must still reject success publication.
        self.assertLessEqual(failure["rows_written"], 3)
        final_admission, final_tokenizer = self.validate()
        final_output = self.root / "final_guard_failure"
        final_calls = [0]
        def final_guard():
            final_calls[0] += 1
            if final_calls[0] == 4: raise ValueError("final resource guard")
        with self.assertRaisesRegex(ValueError, "final resource guard"):
            adapter.materialize(final_admission, final_tokenizer, final_output, resource_guard=final_guard)
        self.assertFalse((final_output / "receipt.json").exists())
        self.assertEqual(3, adapter.metadata(final_output / "failure.json")["rows_written"])

    def test_path_traversal_symlinks_reparse_points_and_external_output_fail(self):
        for name in ("../policy_inputs.jsonl", "/absolute", "nested//file", "C:/outside", "..", "sub:stream", "..\\file"):
            with self.subTest(name=name), self.assertRaises(ValueError): adapter.safe_path(self.source, name)
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "symlink"): adapter.checked_root(self.source)
            with self.assertRaisesRegex(ValueError, "symlink"): adapter.checked_root(self.root / "dangling", exists=False)
        with patch.object(Path, "is_symlink", return_value=False), \
                patch.object(Path, "lstat", return_value=SimpleNamespace(st_file_attributes=0x400)):
            with self.assertRaisesRegex(ValueError, "reparse"): adapter.checked_root(self.source)
        admission, tokenizer = self.validate()
        with self.assertRaisesRegex(ValueError, "direct staging"): adapter.materialize(admission, tokenizer, self.source / "output")

    def test_qualified_flags_missing_denominator_and_changed_inventory_rejected(self):
        self.receipt["confirmation_eligible"] = True; self.sign()
        with self.assertRaisesRegex(ValueError, "must remain closed"): self.validate()
        self.receipt["confirmation_eligible"] = False
        self.receipt["constructed_rows"] = 2; self.sign()
        with self.assertRaisesRegex(ValueError, "denominator"): self.validate()
        self.receipt["constructed_rows"] = 3
        self.rows.pop(); self.sign()
        with self.assertRaisesRegex(ValueError, "inventory"): self.validate()

    def test_import_and_entire_fixture_process_have_no_framework_or_solver(self):
        for name in ("torch", "transformers", "rs_verifier", "long_context_guards", "long_context_tasks"):
            self.assertNotIn(name, sys.modules)
        self.validate()
        adapter.no_frameworks()


if __name__ == "__main__": unittest.main()
