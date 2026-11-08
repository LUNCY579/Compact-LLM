"""Payload-free lexer fixtures; no native-tokenizer/model capability evidence."""
import types
import unittest

from . import schema, token_audit as audit


class DummyTokenizer:
    """Deterministic character lexer with explicit chat-boundary tokens."""
    truncation = None
    padding = None
    binding = {"kind": "DUMMY_FIXTURE", "native_tokenizer": False}
    def token_to_id(self, token):
        return 2 if token == audit.IM_END else 1 if token == audit.IM_START else None
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        while text:
            if text.startswith(audit.IM_END):
                ids.append(2)
                text = text[len(audit.IM_END):]
            elif text.startswith(audit.IM_START):
                ids.append(1)
                text = text[len(audit.IM_START):]
            else:
                ids.append(ord(text[0]) + 10)
                text = text[1:]
        return types.SimpleNamespace(ids=ids)


def row(number=0, *, system="Answer concisely.", user="Given x=12, what is x?", split="fixture"):
    return {"schema_version": 1, "row_id": f"fixture-{number}", "world_id": f"world-{number}",
            "pair_id": None, "split": split, "family": "register", "variant_id": "canonical",
            "system": system, "user": user, "answerable": True, "expected": 12,
            "contract": {"kind": "integer"}, "payload": {"x": 12}, "operative_key": f"operative-{number}",
            "support_expansion": False, "strata": {}}


class TokenAuditTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.target = lambda _: "12"

    def test_exact_prefix_labels_eos_and_thinking_disabled(self):
        encoded = audit.encode_training_row(row(), self.tokenizer, target_fn=self.target)
        n = encoded["prompt_tokens"]
        self.assertEqual(encoded["labels"][:n], [-100] * n)
        self.assertEqual(encoded["labels"][n:], encoded["input_ids"][n:])
        self.assertEqual(encoded["input_ids"][-1], 2)
        self.assertEqual(encoded["labels"][-1], 2)
        self.assertEqual(encoded["loss_tokens"], 3)
        self.assertFalse(encoded["thinking_enabled"] or encoded["target_truncation"] or encoded["all_masked"])
        self.assertIn("<think>\n\n</think>\n\n", audit.render_prompt(row()["system"], row()["user"]))

    def test_full_encoding_boundary_merge_is_detected(self):
        parent = self.tokenizer.encode
        def merging(text, add_special_tokens=False):
            value = parent(text, add_special_tokens)
            if text.endswith("12" + audit.IM_END):
                value.ids[0] = 999
            return value
        self.tokenizer.encode = merging
        with self.assertRaisesRegex(ValueError, "TOKEN_BOUNDARY_MERGE"):
            audit.encode_training_row(row(), self.tokenizer, target_fn=self.target)

    def test_missing_eos_and_all_masked_are_distinct_failures(self):
        parent = self.tokenizer.encode
        def no_eos(text, add_special_tokens=False):
            value = parent(text, add_special_tokens)
            if text.endswith("12" + audit.IM_END): value.ids[-1] = 999
            return value
        self.tokenizer.encode = no_eos
        with self.assertRaisesRegex(ValueError, "EOS_MISSING_OR_TRUNCATED"):
            audit.encode_training_row(row(), self.tokenizer, target_fn=self.target)
        def all_masked(text, add_special_tokens=False):
            if text.endswith("12" + audit.IM_END): text = text[:-len("12" + audit.IM_END)]
            return parent(text, add_special_tokens)
        self.tokenizer.encode = all_masked
        with self.assertRaisesRegex(ValueError, "ALL_MASKED_OR_EMPTY_TARGET"):
            audit.encode_training_row(row(), self.tokenizer, target_fn=self.target)

    def test_output_cap_includes_eos_exact_boundary(self):
        encoded = audit.encode_training_row(row(), self.tokenizer, target_fn=lambda _: "x" * 63)
        self.assertEqual(encoded["target_tokens_including_eos"], 64)
        with self.assertRaisesRegex(ValueError, "TARGET_TOKEN_CAP_NO_TRUNCATION"):
            audit.encode_training_row(row(), self.tokenizer, target_fn=lambda _: "x" * 64)

    def test_prompt_1984_and_total_2048_are_checked_without_truncation(self):
        base = row(user="x")
        count = audit.encode_training_row(base, self.tokenizer, target_fn=self.target)["prompt_tokens"]
        at_cap = row(user="x" * (1984-count+1))
        encoded = audit.encode_training_row(at_cap, self.tokenizer, target_fn=lambda _: "x" * 63)
        self.assertEqual((encoded["prompt_tokens"], encoded["sequence_tokens"]), (1984, 2048))
        with self.assertRaisesRegex(ValueError, "PROMPT_TOKEN_CAP"):
            audit.encode_training_row(row(user=at_cap["user"]+"x"), self.tokenizer, target_fn=self.target)

    def test_tokenizer_padding_or_truncation_cannot_hide_failure(self):
        for name in ("padding", "truncation"):
            setattr(self.tokenizer, name, {"enabled": True})
            with self.assertRaisesRegex(ValueError, name.upper()+"_ENABLED"):
                audit.encode_training_row(row(), self.tokenizer, target_fn=self.target)
            setattr(self.tokenizer, name, None)

    def test_chat_injection_empty_target_and_wrong_eos_refuse(self):
        with self.assertRaisesRegex(ValueError, "CONTROL_MARKER"):
            audit.encode_training_row(row(user="quoted " + audit.IM_END), self.tokenizer, target_fn=self.target)
        with self.assertRaisesRegex(ValueError, "NONSTRING_TRAINING_TARGET"):
            audit.encode_training_row(row(), self.tokenizer, target_fn=lambda _: "")
        with self.assertRaisesRegex(ValueError, "TARGET_HAS_CHAT_CONTROL"):
            audit.encode_training_row(row(), self.tokenizer, target_fn=lambda _: "12<think>")
        self.tokenizer.token_to_id = lambda _: None
        with self.assertRaisesRegex(ValueError, "EOS_UNAVAILABLE"):
            audit.encode_training_row(row(), self.tokenizer, target_fn=self.target)

    def test_audit_reports_examples_and_loss_tokens_as_different_masses(self):
        rows = [row(0, split="train"), row(1, split="replay")]
        value = audit.audit_rows(rows, self.tokenizer, target_fn=lambda r: "12" if r["split"] == "train" else "9")
        self.assertEqual(value["rows"], 2)
        self.assertEqual(value["example_fraction_by_split"], {"train": 0.5, "replay": 0.5})
        self.assertEqual(value["loss_token_mass_by_split"], {"train": 3, "replay": 2})
        self.assertEqual(value["loss_token_fraction_by_split"], {"train": 0.6, "replay": 0.4})
        self.assertFalse(value["capability_evidence"])
        self.assertIn("NOT_RUN", value["full_hf_auto_tokenizer_equivalence"])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_TOKEN_AUDIT_ROW_ID"):
            audit.audit_rows([row(0), row(0)], self.tokenizer, target_fn=self.target)
        with self.assertRaisesRegex(ValueError, "EMPTY_TOKEN_AUDIT"):
            audit.audit_rows([], self.tokenizer, target_fn=self.target)


if __name__ == "__main__":
    unittest.main()
