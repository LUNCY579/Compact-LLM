"""Actual CPU torch operations on a tiny causal fixture; no pretrained weights."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from experiments.rineya_verified_resampling_v1.contracts import Blocked
from experiments.rineya_verified_resampling_v1.torch_sampler import TorchSampler


if torch is not None:
    class TinyCausal(torch.nn.Module):
        def __init__(self, corrupt_cache=False, force_eos=False, nan=False):
            super().__init__()
            self.corrupt_cache, self.force_eos, self.nan = corrupt_cache, force_eos, nan
            self.observed_grad = []

        def forward(self, ids, *, use_cache=False, logits_to_keep=1, past_key_values=None, attention_mask=None):
            self.observed_grad.append(torch.is_grad_enabled())
            offset = 0 if past_key_values is None else past_key_values
            values = ids.cumsum(-1)+offset
            last = values[:, -1:] % 13
            logits = -0.3*(torch.arange(32).view(1, 1, 32)-last.unsqueeze(-1)).abs().float()
            if self.force_eos:
                logits.fill_(-100)
                logits[:, :, 31] = 100
            if self.nan:
                logits[:, :, 0] = float("nan")
            cache = values[:, -1:] + (1 if self.corrupt_cache else 0)
            return SimpleNamespace(logits=logits, past_key_values=cache if use_cache else None)


@unittest.skipIf(torch is None, "optional CPU torch is not installed")
class TorchSamplerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = TinyCausal()
        self.sampler = TorchSampler(self.model, torch, "cpu")
        self.args = dict(seed=123, temperature=0.7, max_tokens=16, greedy=False, cache=True, eos=(), emit=lambda *_: None)

    def test_seeded_sampling_repeats(self):
        a = self.sampler.sample([1, 2, 3], **self.args)
        b = self.sampler.sample([1, 2, 3], **self.args)
        self.assertEqual(a["token_ids"], b["token_ids"])
        self.assertEqual(a["policy_logprobs"], b["policy_logprobs"])

    def test_independent_seed_changes_trace(self):
        a = self.sampler.sample([1, 2, 3], **self.args)
        b = self.sampler.sample([1, 2, 3], **{**self.args, "seed": 124})
        self.assertNotEqual(a["token_ids"], b["token_ids"])

    def test_full_prefix_and_cache_agree(self):
        a = self.sampler.sample([1, 2, 3], **self.args)
        b = self.sampler.sample([1, 2, 3], **{**self.args, "cache": False})
        self.assertEqual(a["token_ids"], b["token_ids"])
        self.assertEqual(a["raw_logprobs"], b["raw_logprobs"])
        self.assertLess(a["forward_positions"], b["forward_positions"])

    def test_eos_is_recorded_once(self):
        sampler = TorchSampler(TinyCausal(force_eos=True), torch, "cpu")
        tokens = []
        result = sampler.sample([1], **{**self.args, "eos": (31,), "emit": lambda token, *_: tokens.append(token)})
        self.assertEqual(tokens, [31])
        self.assertEqual(result["stop"], "eos")

    def test_cap_is_not_eos(self):
        result = self.sampler.sample([1], **self.args)
        self.assertEqual(len(result["token_ids"]), 16)
        self.assertEqual(result["stop"], "length")

    def test_inference_mode_has_no_gradients(self):
        self.sampler.sample([1, 2, 3], **self.args)
        self.assertFalse(any(self.model.observed_grad))

    def test_greedy_policy_logprob_zero(self):
        result = self.sampler.sample([1], **{**self.args, "greedy": True})
        self.assertEqual(result["policy_logprobs"], [0.0]*16)
        self.assertTrue(all(v <= 0 for v in result["raw_logprobs"]))

    def test_nan_logits_fail_closed(self):
        sampler = TorchSampler(TinyCausal(nan=True), torch, "cpu")
        with self.assertRaises(Blocked): sampler.sample([1], **self.args)

    def test_sampler_qualification(self):
        result = self.sampler.qualify([1, 2, 3], {"seed": 123, "temperature": 0.7, "max_new_tokens": 16, "cache": True}, ())
        self.assertTrue(result["seeded_repeat_exact"])
        self.assertEqual(result["raw_logprob_max_error"], 0)

    def test_corrupt_cache_fails_qualification(self):
        sampler = TorchSampler(TinyCausal(corrupt_cache=True), torch, "cpu")
        with self.assertRaises(Blocked):
            sampler.qualify([1, 2, 3], {"seed": 123, "temperature": 0.7, "max_new_tokens": 16, "cache": True}, ())

    def test_generator_does_not_use_global_rng(self):
        state = torch.get_rng_state().clone()
        self.sampler.sample([1, 2], **self.args)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
