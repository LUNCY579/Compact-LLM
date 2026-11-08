"""Real CPU autograd/optimizer tests on a tiny random model, not Rineya evidence."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

TORCH = importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from experiments.progressive_rl_curriculum_v1.backends import TorchBackend
    from experiments.progressive_rl_curriculum_v1.cli import demo_inputs
    from experiments.progressive_rl_curriculum_v1.objectives import policy_loss
    from experiments.progressive_rl_curriculum_v1.schema import Episode, Prompt

    class Tokenizer:
        eos_token_id = 0
        pad_token_id = 0
        chat_template = None

        def encode(self, text, add_special_tokens=True):
            return ([1] if add_special_tokens else []) + [ord(c) % 4 + 1 for c in text]

        def decode(self, ids, skip_special_tokens=True):
            return ''.join(' abcd'[i] for i in ids if i != 0)

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 16)
            self.head = torch.nn.Linear(16, 5)
            self.dropout = torch.nn.Dropout(.5)
            self.lora_B = torch.nn.Parameter(torch.zeros(5))
            with torch.no_grad():
                self.head.bias[0] = 2
            self.last_kwargs = None

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            return SimpleNamespace(logits=self.head(self.dropout(self.embedding(input_ids))) + self.lora_B)

        def generate(self, input_ids, attention_mask, **kwargs):
            self.last_kwargs = kwargs
            result = input_ids
            scores = []
            for _ in range(kwargs['max_new_tokens']):
                logits = self(result).logits[:, -1]
                scores.append(logits)
                probabilities = logits.softmax(-1)
                token = torch.multinomial(probabilities, 1)
                result = torch.cat((result, token), 1)
                if token.item() == kwargs['eos_token_id']:
                    break
            return SimpleNamespace(sequences=result, scores=scores)


@unittest.skipUnless(TORCH, 'optional torch dependency not installed')
class ObjectiveTests(unittest.TestCase):
    def test_positive_and_negative_advantage_clipping(self):
        current = torch.log(torch.tensor([[.6], [.01]], requires_grad=True))
        old = torch.log(torch.tensor([[.2], [.2]]))
        loss = policy_loss(current, old, current.detach(), torch.tensor([1., -1.]),
                           torch.ones_like(current, dtype=torch.bool), beta=0)
        self.assertAlmostEqual(loss.item(), (-1.2 + .8) / 2, places=5)

    def test_masked_nan_sanitized_before_nonlinearity(self):
        current = torch.tensor([[-1., float('nan')]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        loss = policy_loss(current, current.detach(), current.detach(), torch.tensor([1.]), mask, beta=0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertEqual(current.grad[0, 1], 0)

    def test_behavior_and_reference_detached(self):
        current = torch.tensor([[-1.]], requires_grad=True)
        old = torch.tensor([[-1.]], requires_grad=True)
        ref = torch.tensor([[-1.1]], requires_grad=True)
        policy_loss(current, old, ref, torch.tensor([1.]), torch.tensor([[True]])).backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(ref.grad)
        self.assertIsNotNone(current.grad)

    def test_constant_reward_can_still_have_kl_gradient(self):
        current = torch.tensor([[-1.]], requires_grad=True)
        loss = policy_loss(current, current.detach(), torch.tensor([[-2.]]), torch.tensor([0.]), torch.tensor([[True]]))
        loss.backward()
        self.assertGreater(loss.item(), 0)
        self.assertNotEqual(current.grad.item(), 0)

    def test_empty_masks_rejected(self):
        with self.assertRaises(ValueError):
            policy_loss(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1),
                        torch.tensor([1.]), torch.tensor([[False]]))

    def test_invalid_logprobs_rejected(self):
        with self.assertRaises(ValueError):
            policy_loss(torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1, 1),
                        torch.tensor([1.]), torch.tensor([[True]]))

    def test_large_ratio_rejected(self):
        with self.assertRaises(FloatingPointError):
            policy_loss(torch.tensor([[-1.]]), torch.tensor([[-30.]]), torch.tensor([[-1.]]),
                        torch.tensor([1.]), torch.tensor([[True]]))

    def test_low_precision_loss_rejected(self):
        with self.assertRaises(ValueError):
            policy_loss(torch.zeros(1, 1, dtype=torch.bfloat16), torch.zeros(1, 1), torch.zeros(1, 1),
                        torch.tensor([1.]), torch.tensor([[True]]))


@unittest.skipUnless(TORCH, 'optional torch dependency not installed')
class BackendTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        plan, self.catalog = demo_inputs()
        self.plan = replace(plan, learning_rate=.03, trainable_patterns=('lora_B',))
        self.backend = TorchBackend(TinyLM(), Tokenizer(), self.plan, device='cpu', identity={'kind': 'unit_torch'})

    def groups(self):
        prompt = (1, 2)
        episodes = []
        for response in ((1, 0), (2, 0), (3, 0), (4, 0)):
            with torch.no_grad():
                old = tuple(self.backend._logps(self.backend.model, prompt, response, 'cpu').tolist())
            episodes.append(Episode('fixture', 'x', prompt, response, old, self.backend.version))
        return [(episodes, [1., 0., 0., 0.])]

    def test_actual_optimizer_changes_only_declared_parameters(self):
        before = {n: p.detach().clone() for n, p in self.backend.model.named_parameters()}
        self.backend.update(self.groups(), [])
        after = dict(self.backend.model.named_parameters())
        self.assertFalse(torch.equal(before['lora_B'], after['lora_B']))
        self.assertTrue(all(torch.equal(before[n], p) for n, p in after.items() if n != 'lora_B'))
        self.assertEqual(self.backend.version, 1)

    def test_reference_stays_frozen(self):
        before = {n: p.detach().clone() for n, p in self.backend.reference.named_parameters()}
        self.backend.update(self.groups(), [])
        for name, parameter in self.backend.reference.named_parameters():
            self.assertTrue(torch.equal(before[name], parameter))
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)

    def test_stale_policy_is_rejected(self):
        groups = self.groups()
        self.backend.version += 1
        with self.assertRaises(ValueError):
            self.backend.update(groups, [])

    def test_old_logprob_mismatch_is_rejected(self):
        groups = self.groups()
        groups[0][0][0] = replace(groups[0][0][0], old_logps=(-10., -10.))
        with self.assertRaises(ValueError):
            self.backend.update(groups, [])

    def test_collect_aligns_tokens_and_logprobs(self):
        episodes = self.backend.collect(Prompt('fixture', '1+1'), 4, 8, 5)
        for episode in episodes:
            self.assertEqual(len(episode.response_ids), len(episode.old_logps))
            self.assertEqual(episode.terminated, episode.response_ids[-1] == 0)
        kwargs = self.backend.model.last_kwargs
        self.assertEqual((kwargs['temperature'], kwargs['top_p'], kwargs['top_k']), (1., 1., 0))

    def test_collection_reproducible(self):
        args = (Prompt('fixture', '1+1'), 4, 8, 5)
        self.assertEqual(self.backend.collect(*args), self.backend.collect(*args))

    def test_prompt_overflow_never_silently_truncated(self):
        with self.assertRaises(ValueError):
            self.backend.collect(Prompt('fixture', 'x' * 300), 4, 8, 5)

    def test_sft_includes_eos_and_updates(self):
        task = self.catalog.tasks[0]
        prompt = tuple(self.backend._prompt_ids(task.public()))
        response = tuple(self.backend.tokenizer.encode(task.answer, add_special_tokens=False)) + (0,)
        expected = -self.backend._logps(self.backend.model, prompt, response, 'cpu').mean()
        self.assertTrue(torch.allclose(expected, self.backend._ce(task, task.answer)))
        before = self.backend.model.lora_B.detach().clone()
        self.backend.sft([(task, task.answer)])
        self.assertFalse(torch.equal(before, self.backend.model.lora_B))

    def test_optimizer_checkpoint_roundtrip_reproduces_next_update(self):
        self.backend.update(self.groups(), [])
        with tempfile.TemporaryDirectory() as tmp:
            self.backend.save(Path(tmp))
            self.backend.update(self.groups(), [])
            expected = self.backend.model.lora_B.detach().clone()
            self.backend.load(Path(tmp))
            self.backend.update(self.groups(), [])
            self.assertTrue(torch.equal(expected, self.backend.model.lora_B))

    def test_unmatched_trainable_selection_rejected(self):
        with self.assertRaises(ValueError):
            TorchBackend(TinyLM(), Tokenizer(), replace(self.plan, trainable_patterns=('missing',)),
                         device='cpu', identity={})

    def test_full_weight_training_not_silently_enabled(self):
        with self.assertRaises(ValueError):
            TorchBackend(TinyLM(), Tokenizer(), replace(self.plan, trainable_patterns=('.*',)),
                         device='cpu', identity={})


if __name__ == '__main__':
    unittest.main()
