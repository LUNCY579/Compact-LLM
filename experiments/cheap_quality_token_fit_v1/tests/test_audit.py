"""Synthetic importless mechanics; no real tokenizer/model/dependency load."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cheap_token_fit_audit", STUDY / "audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class FakeTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 151645

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            if text.startswith(self.eos_token):
                result.append(self.eos_token_id)
                text = text[len(self.eos_token):]
            else:
                result.append(ord(text[0]) + 100)
                text = text[1:]
        return result

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.eos_token if value == self.eos_token_id else chr(value - 100) for value in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        if add_generation_prompt:
            prompt = messages
            target = None
        else:
            prompt = messages[:-1]
            target = messages[-1]["content"]
        text = "".join("<|im_start|>" + item["role"] + "\n" + item["content"] + self.eos_token + "\n" for item in prompt)
        text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if target is not None:
            text += target + self.eos_token + "\n"
        return self.encode(text) if tokenize else text


class TokenFitMechanics(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.messages = [{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": "Compute 2+3."}]
        self.target = '{"answer":5}'

    def test_exact_response_supervision_masks_native_tail_and_prompt(self):
        measured = audit.native_measure(self.messages, self.target, self.tokenizer)
        start, end = measured["answer_start"], measured["answer_end"]
        self.assertEqual(measured["labels"][:start], [-100] * start)
        self.assertEqual(measured["labels"][start:end + 1], measured["ids"][start:end + 1])
        self.assertEqual(measured["labels"][end + 1:], [-100])
        self.assertEqual(measured["supervised_tokens"], len(self.target) + 1)
        self.assertEqual(measured["final_content_tokens"], len(self.target))
        self.assertEqual(measured["response_eos_count"], 1)
        self.assertEqual(sum(measured["completion_mask"]), measured["supervised_tokens"])

    def test_profile_rejects_reserve_without_truncating(self):
        measured = {"processed_tokens": 480, "prompt_tokens": 450, "supervised_tokens": 29}
        decision = audit.profile_decision(measured, 512, 128, 128)
        self.assertFalse(decision["retained"])
        self.assertEqual(decision["rejection_reasons"], ["PROMPT_PLUS_COMPLETION_RESERVE_EXCEEDS_PROFILE"])
        self.assertEqual(measured["processed_tokens"], 480)
        self.assertTrue(audit.profile_decision(measured, 1024, 128, 128)["retained"])

    def test_profiles_retain_all_distinct_rejection_reasons(self):
        decision = audit.profile_decision({"processed_tokens": 700, "prompt_tokens": 530, "supervised_tokens": 169}, 512, 128, 128)
        self.assertEqual(len(decision["rejection_reasons"]), 3)

    def test_independent_masks_match_pinned_serializer_and_explicit_overlength(self):
        path = audit.REPO / "experiments/reasoning_posttraining_v1/serialization.py"
        serializer = audit.pure_module("test_token_fit_serializer", path.read_bytes(), path, {"__future__", "re"})
        measured = audit.native_measure(self.messages, self.target, self.tokenizer)
        self.assertEqual(audit.compare_serializer(serializer, self.messages, self.target, self.tokenizer, measured, 512, 128),
                         "EXACT_NATIVE_IDS_PREFIX_EOS_MASK_PARITY")
        self.assertEqual(audit.compare_serializer(serializer, self.messages, self.target, self.tokenizer, measured, 140, 128),
                         "EXPLICIT_OVERLENGTH_REPRODUCED")

    def test_native_prefix_corruption_invalidates_measurement(self):
        class ChangedPrefix(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
                result = super().apply_chat_template(messages, tokenize, add_generation_prompt, enable_thinking)
                if tokenize and not add_generation_prompt:
                    result[0] += 1
                return result
        with self.assertRaisesRegex(ValueError, "native full tokens differ"):
            audit.native_measure(self.messages, self.target, ChangedPrefix())

    def test_train_guard_precedes_all_sealed_render_solver_calls(self):
        calls = []
        bridge = types.SimpleNamespace(make_row=lambda *args: calls.append(args))
        for role in ("CHECK", "FINAL", "DEV", "TRAIN_SELECTION"):
            with self.assertRaisesRegex(ValueError, "only TRAIN"):
                audit.train_row(bridge, types.SimpleNamespace(role=role), "a" * 40, 0)
        self.assertEqual(calls, [])

    def test_direct_modules_reject_numerical_model_imports(self):
        with self.assertRaisesRegex(ValueError, "undeclared import"):
            audit.pure_module("invalid_token_fit_source", b"import torch\n", Path("synthetic.py"), {"re"})
        self.assertFalse(audit.FORBIDDEN_FRAMEWORKS & set(sys.modules))

    def test_duplicate_json_and_nonfinite_values_fail_closed(self):
        for value in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError):
                audit.read_json(value)


if __name__ == "__main__":
    unittest.main()
