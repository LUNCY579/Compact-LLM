import copy
import importlib.util
import unittest
from types import SimpleNamespace

from experiments.rineya_recursive_self_rewrite_v1.core import *
from experiments.rineya_recursive_self_rewrite_v1.fixtures import catalog, fixture_backend, ByteTokenizer
from experiments.rineya_recursive_self_rewrite_v1.pipeline import discovery, reconstruct
from experiments.rineya_recursive_self_rewrite_v1.data import compile_dataset, match_controls
from experiments.rineya_recursive_self_rewrite_v1.evaluation import *
from experiments.rineya_recursive_self_rewrite_v1.training import train_sft, hf_backend


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        a, _ = catalog(3, split="calibration", offset=10)
        b, _ = catalog(3, split="retention_eval", offset=20)
        self.tasks = a + b
        self.panel = evaluate_panel(self.tasks, {"base": fixture_backend(), "candidate": fixture_backend()}, Limits())

    def test_guidance_free_paired_comparison(self):
        report = compare_panel(self.panel, self.tasks, baseline="base", candidate="candidate")
        self.assertEqual(report["groups"]["calibration"]["mean"], 0)
        self.assertEqual(report["groups"]["retention_eval"]["damages"], 0)
        self.assertIn("NOT_AUTHORIZED", report["promotion"])

    def test_per_skill_retention_breakdown(self):
        report = compare_panel(self.panel, self.tasks, baseline="base", candidate="candidate")
        self.assertEqual(len(report["per_source_skill"]), 3)

    def test_missing_eval_row_rejected(self):
        self.panel["records"]["candidate"].pop()
        with self.assertRaises(ValueError):
            compare_panel(self.panel, self.tasks, baseline="base", candidate="candidate")

    def test_duplicate_eval_row_rejected(self):
        self.panel["records"]["candidate"][0] = self.panel["records"]["candidate"][1]
        with self.assertRaises(ValueError):
            compare_panel(self.panel, self.tasks, baseline="base", candidate="candidate")

    def test_panel_hash_drift(self):
        self.panel["panel_hash"] = "changed"
        with self.assertRaises(ValueError):
            compare_panel(self.panel, self.tasks, baseline="base", candidate="candidate")

    def test_pass_at_k(self):
        self.assertAlmostEqual(pass_at_k(3, 1, 1), 1 / 3)
        self.assertEqual(pass_at_k(3, 1, 3), 1)
        self.assertEqual(pass_at_k(3, 0, 3), 0)

    def test_pass_at_k_bounds(self):
        for args in ((2, 3, 1), (2, 1, 3), (0, 0, 0), (3, True, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                pass_at_k(*args)

    def test_cluster_bootstrap_repeatable(self):
        a, b = self.panel["records"]["base"], self.panel["records"]["candidate"]
        self.assertEqual(paired_effect(a, b), paired_effect(a, b))
        self.assertEqual(paired_effect(a, b)["parents"], 6)

    def test_forbidden_training_panel(self):
        tasks, _ = catalog(1)
        with self.assertRaises(ValueError):
            evaluate_panel(tasks, {"base": fixture_backend()}, Limits())

    def test_training_authorization_required(self):
        with self.assertRaises(ValueError):
            train_sft(None, None, {}, initial_checkpoint="x", tokenizer_id="x", trainable_names=["x"])

    def test_native_calls_authorization_required(self):
        with self.assertRaises(ValueError):
            hf_backend(None, None, checkpoint_id="x")


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch not installed")
class TorchTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    def setUp(self):
        torch = self.torch
        class TinyCausalFixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(260, 8)
                self.output = torch.nn.Linear(8, 260)
            def forward(self, input_ids, attention_mask, use_cache=False):
                return SimpleNamespace(logits=self.output(self.embedding(input_ids)))
        torch.manual_seed(0)
        self.model = TinyCausalFixture()
        tasks, ancestry = catalog(1)
        backend = fixture_backend()
        found = discovery(tasks, backend, Limits(), attempts=1)
        self.data = compile_dataset(tasks, ancestry, found["records"], [], arm="direct", parent_cap=1)
        self.kwargs = dict(authorized=True, initial_checkpoint="tiny-fixture-not-rineya",
                           tokenizer_id="byte-fixture", trainable_names=["output.weight", "output.bias"],
                           optimizer_steps=2, tokens_per_step=7, max_context=10000)

    def test_real_optimizer_exact_tokens_and_scope(self):
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        receipt = train_sft(self.model, ByteTokenizer(), self.data, **self.kwargs)
        self.assertEqual(receipt["supervised_tokens"], 14)
        self.assertEqual(sum(receipt["row_token_exposures"]), 14)
        self.assertTrue(self.torch.equal(before["embedding.weight"], self.model.embedding.weight))
        self.assertFalse(self.torch.equal(before["output.weight"], self.model.output.weight))
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_rng_and_mode_restored(self):
        self.model.eval()
        state = self.torch.random.get_rng_state().clone()
        train_sft(self.model, ByteTokenizer(), self.data, **self.kwargs)
        self.assertTrue(self.torch.equal(state, self.torch.random.get_rng_state()))
        self.assertFalse(self.model.training)

    def test_unknown_trainable_scope_rejected(self):
        kwargs = dict(self.kwargs, trainable_names=["nonexistent"])
        with self.assertRaises(ValueError):
            train_sft(self.model, ByteTokenizer(), self.data, **kwargs)

    def test_tampered_training_rows_rejected(self):
        self.data["rows"][0]["completion"] += "tampered"
        with self.assertRaises(ValueError):
            train_sft(self.model, ByteTokenizer(), self.data, **self.kwargs)

    def test_matching_controls_and_mismatch_rejection(self):
        receipt = train_sft(self.model, ByteTokenizer(), self.data, **self.kwargs)
        self.assertTrue(match_controls([self.data, self.data], [receipt, receipt])["matched"])
        altered = dict(receipt, supervised_tokens=15)
        with self.assertRaises(ValueError):
            match_controls([self.data, self.data], [receipt, altered])

    def test_unimplemented_replay_not_claimed(self):
        kwargs = dict(self.kwargs, retention_policy="claimed-replay")
        with self.assertRaises(ValueError):
            train_sft(self.model, ByteTokenizer(), self.data, **kwargs)

    def test_native_wrapper_token_boundary(self):
        model, torch = self.model, self.torch
        class Tokenizer(ByteTokenizer):
            def decode(self, ids, skip_special_tokens=True):
                return bytes(t - 2 for t in ids if t != 1).decode()
        tokenizer = Tokenizer()
        def generate(input_ids, **kwargs):
            tail = tokenizer.encode('{"kind":"answer","value":"1"}') + [1]
            return torch.cat([input_ids, torch.tensor([tail])], dim=1)
        model.generate = generate
        backend = hf_backend(model, tokenizer, authorized=True, checkpoint_id="fixture-checkpoint", max_context=1000)
        reply = backend.generate("test", 37, 64)
        self.assertEqual(reply.input_tokens, 4)
        self.assertEqual(reply.output_tokens, len(tokenizer.encode(reply.text)) + 1)
        self.assertTrue(model.training)
