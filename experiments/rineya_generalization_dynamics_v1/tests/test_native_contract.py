"""Real PyTorch tensor arithmetic plus a stub generation API; no model downloads."""
import importlib.util
import math
import sys
import types
import unittest
from unittest.mock import patch

from experiments.rineya_generalization_dynamics_v1.common import Blocked
from experiments.rineya_generalization_dynamics_v1.native import NativeEvaluator, continuation_logp
from experiments.rineya_generalization_dynamics_v1.pipeline import DEFAULT_NATIVE

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "optional PyTorch tensor-contract lane; NOT a native model test")
class NativeContractTests(unittest.TestCase):
    def setUp(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
                self.config = types.SimpleNamespace(max_position_embeddings=128)
                self.stop = True
            def forward(self, input_ids, attention_mask, use_cache=False):
                logits = torch.log(torch.tensor([.1, .2, .3, .4])).repeat(1, input_ids.shape[1], 1)
                return types.SimpleNamespace(logits=logits)
            def generate(self, input_ids, attention_mask, generation_config):
                suffix = [2, 3, 0] if self.stop else [2, 3]
                return torch.cat((input_ids, torch.tensor([suffix])), dim=1)
        self.model = Tiny().eval()
        class Tokenizer:
            eos_token_id, pad_token_id = 0, None
            chat_template = "fixture-template"
            def encode(self, text, **kwargs):
                return [2, 3] if text == " 12" else [3] if text == " 9" else [0, 1]
            def decode(self, ids, **kwargs):
                if ids != [2, 3]:
                    raise AssertionError("Prompt or EOS leaked into decoded completion")
                return "12"
            def apply_chat_template(self, messages, **kwargs):
                return [0, 1]
        self.tok = Tokenizer()
        self.evaluator = NativeEvaluator({}, dict(DEFAULT_NATIVE))
        self.evaluator.model = self.model
        self.evaluator.tokenizer = self.tok
        self.row = {"id": "fixture", "prompt": "test prompt", "answer": "12", "distractor": "9"}
        self.module = types.ModuleType("transformers")
        self.module.GenerationConfig = lambda **kw: types.SimpleNamespace(**kw)

    def test_multitoken_shift_and_prompt_mask(self):
        actual = continuation_logp(self.model, [0, 1, 0, 1], [2, 3], "cpu")
        self.assertAlmostEqual(actual, math.log(.3) + math.log(.4), places=6)

    def test_one_token_scoring(self):
        self.assertAlmostEqual(continuation_logp(self.model, [0, 1], [3], "cpu"), math.log(.4), places=6)

    def test_empty_sequence_rejected(self):
        for prompt, answer in (([], [1]), ([1], [])):
            with self.assertRaises(Blocked):
                continuation_logp(self.model, prompt, answer, "cpu")

    def test_native_predict_slices_prompt_and_scores_full_answers(self):
        with patch.dict(sys.modules, {"transformers": self.module}):
            result = self.evaluator.predict(self.row)
        self.assertEqual(result["text"], "12")
        self.assertEqual(result["gold_tokens"], 2)
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(result["generated_token_ids"], [2, 3, 0])
        self.assertAlmostEqual(result["gold_logp"], math.log(.3 * .4), places=6)

    def test_length_stop_is_recorded(self):
        self.model.stop = False
        with patch.dict(sys.modules, {"transformers": self.module}):
            self.assertEqual(self.evaluator.predict(self.row)["stop_reason"], "length")

    def test_no_silent_prompt_truncation(self):
        self.evaluator.config["max_input_tokens"] = 1
        with patch.dict(sys.modules, {"transformers": self.module}), self.assertRaises(Blocked):
            self.evaluator.predict(self.row)

    def test_chat_mode_explicit(self):
        self.evaluator.config["prompt_format"] = "chat"
        with patch.dict(sys.modules, {"transformers": self.module}):
            self.assertEqual(self.evaluator.predict(self.row)["text"], "12")

    def test_missing_chat_template_blocks(self):
        self.evaluator.config["prompt_format"] = "chat"
        self.tok.chat_template = None
        with patch.dict(sys.modules, {"transformers": self.module}), self.assertRaises(Blocked):
            self.evaluator.predict(self.row)


if __name__ == "__main__":
    unittest.main()
