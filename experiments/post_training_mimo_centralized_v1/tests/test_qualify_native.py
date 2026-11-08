"""CPU-only numerical controls for E0 helpers; no checkpoint or CUDA access."""
from __future__ import annotations

import io
import json
from pathlib import Path
import unittest

import torch
from torch import nn

from experiments.maxrl_comp002_pretraining_v1.mechanics import completion_labels
from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import (
    completion_loss, install_lora, tensor_digest, validate_config,
)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 8, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.q_proj(x)[..., :4] + self.v_proj(x) + self.k_proj(x)


class TinyModel(nn.Module):
    """All 56 projection sites execute, using only a few thousand parameters."""
    def __init__(self, layers=28, dtype=torch.float32):
        super().__init__()
        self.embed = nn.Embedding(13, 4)
        self.layers = nn.ModuleList(TinyAttention() for _ in range(layers))
        self.head = nn.Linear(4, 13, bias=False)
        self.to(dtype=dtype).requires_grad_(False)

    def forward(self, ids):
        x = self.embed(ids)
        return self.head(sum(layer(x) for layer in self.layers) / len(self.layers))


class NativeHelperControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(101)
        self.ids = torch.tensor([[1, 2, 3, 4, 5], [2, 1, 4, 5, 6]])
        self.labels = self.ids.clone()
        self.labels[:, :3] = -100

    def test_zero_adapter_bitwise_parity_fp32_and_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = TinyModel(dtype=dtype)
                expected = model(self.ids).detach().clone()
                names = install_lora(model, torch)
                self.assertEqual(len(names), 56)
                self.assertTrue(torch.equal(model(self.ids).detach(), expected))
                trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
                self.assertEqual(len(trainable), 112)
                self.assertTrue(all(n.endswith((".lora_A", ".lora_B")) for n in trainable))
                self.assertTrue(all(p.dtype == torch.float32 for p in trainable.values()))

    def test_only_adapter_gradients_and_base_immutable_after_update(self):
        model = TinyModel(dtype=torch.bfloat16)
        install_lora(model, torch)
        frozen = {n: tensor_digest(torch, p) for n, p in model.named_parameters() if not p.requires_grad}
        allowed = {n: p for n, p in model.named_parameters() if p.requires_grad}
        before = {n: p.detach().clone() for n, p in allowed.items()}
        optimizer = torch.optim.AdamW(allowed.values(), lr=1e-3, weight_decay=0)
        completion_loss(torch, model(self.ids), self.ids, self.labels).backward()
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in allowed.values()))
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for n, p in allowed.items() if n.endswith(".lora_A")))
        self.assertTrue(any(torch.count_nonzero(p.grad) > 0 for n, p in allowed.items() if n.endswith(".lora_B")))
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[n], p) for n, p in allowed.items()))
        self.assertEqual(frozen, {n: tensor_digest(torch, p) for n, p in model.named_parameters() if not p.requires_grad})

    def test_mask_and_single_causal_shift_against_explicit_token_probabilities(self):
        logits = torch.arange(2 * 5 * 13, dtype=torch.float32).reshape(2, 5, 13).remainder(17) / 3
        actual = completion_loss(torch, logits, self.ids, self.labels)
        terms = []
        for row in range(2):
            for position in (3, 4):
                scores = logits[row, position - 1].double()
                terms.append(torch.logsumexp(scores, 0) - scores[self.ids[row, position]])
        expected = torch.stack(terms).mean()
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)
        changed = logits.clone()
        changed[:, :2] = 500  # Neither prompt-only nor final-position logits contribute.
        changed[:, -1] = -500
        self.assertTrue(torch.equal(actual, completion_loss(torch, changed, self.ids, self.labels)))

    def test_eos_equals_pad_uses_positions_and_includes_terminal_eos(self):
        ids = torch.tensor([[0, 2, 3, 4, 0, 0], [2, 3, 4, 0, 0, 0]])
        attention = torch.tensor([[False, True, True, True, True, False],
                                  [True, True, True, True, False, False]])
        labels = completion_labels(ids, attention, [2, 2], [2, 2], [0], ["eos", "eos"], 8)
        self.assertEqual(labels.tolist(), [[-100, -100, -100, 4, 0, -100],
                                          [-100, -100, 4, 0, -100, -100]])
        logits = torch.zeros((2, 6, 13))
        self.assertAlmostEqual(completion_loss(torch, logits, ids, labels).item(), torch.log(torch.tensor(13.)).item())

    def test_empty_completion_first_token_and_label_mismatch_rejected(self):
        logits = torch.zeros((2, 5, 13))
        bad_labels = [torch.full_like(self.ids, -100), self.labels.clone(), self.labels.clone()]
        bad_labels[1][0, 0] = self.ids[0, 0]
        bad_labels[2][0, -1] = 9
        for labels in bad_labels:
            with self.subTest(labels=labels.tolist()), self.assertRaises(RuntimeError):
                completion_loss(torch, logits, self.ids, labels)

    def test_empty_row_must_not_silently_change_batch_weighting(self):
        labels = self.labels.clone()
        labels[0] = -100
        with self.assertRaises(RuntimeError):
            completion_loss(torch, torch.zeros((2, 5, 13)), self.ids, labels)

    def test_each_sequence_has_equal_weight_when_lengths_differ(self):
        labels = self.labels.clone()
        labels[0, 3] = -100
        logits = torch.zeros((2, 5, 13))
        logits[0, 3, self.ids[0, 4]] = 5
        logits[1, 2, self.ids[1, 3]] = -2
        logits[1, 3, self.ids[1, 4]] = -1
        row_losses = []
        for row in range(2):
            terms = []
            for position in range(1, 5):
                if labels[row, position] != -100:
                    score = logits[row, position - 1].double()
                    terms.append(torch.logsumexp(score, 0) - score[self.ids[row, position]])
            row_losses.append(torch.stack(terms).mean())
        expected = torch.stack(row_losses).mean()
        actual = completion_loss(torch, logits, self.ids, labels)
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)

    def test_loss_shape_dtype_and_vocabulary_errors_rejected(self):
        logits = torch.zeros((2, 5, 13))
        bad_ids = self.ids.clone()
        bad_ids[0, 1] = 13
        cases = [(logits[0], self.ids, self.labels),
                 (logits, self.ids.float(), self.labels),
                 (logits, self.ids, self.labels.float()),
                 (logits, bad_ids, self.labels),
                 (logits[:, :4], self.ids, self.labels)]
        for scores, ids, labels in cases:
            with self.subTest(shapes=(scores.shape, ids.shape, labels.shape)), self.assertRaises(RuntimeError):
                completion_loss(torch, scores, ids, labels)

    def test_nonfinite_supervised_logits_rejected(self):
        logits = torch.zeros((2, 5, 13))
        logits[0, 2, 3] = float("nan")
        with self.assertRaises(RuntimeError):
            completion_loss(torch, logits, self.ids, self.labels)

    def test_wrong_projection_topology_rejected_before_mutation(self):
        model = TinyModel(layers=1)
        original = dict(model.named_modules())
        with self.assertRaises(RuntimeError):
            install_lora(model, torch)
        self.assertEqual(set(original), set(dict(model.named_modules())))
        self.assertIs(model.layers[0].q_proj, original["layers.0.q_proj"])

    def test_frozen_config_rejects_weakened_or_nonfinite_envelope(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "E0_TITAN_BF16.json"
        cfg = json.loads(path.read_text())
        validate_config(cfg)
        cases = [("host_free_gib", 7.3), ("worker_free_gib", 5.9),
                 ("gpu_free_gib", 3.9), ("gpu_worker_free_gib", 0.49),
                 ("disk_free_gib", 3.9), ("wall_seconds", 1801),
                 ("host_free_gib", float("inf")), ("host_free_gib", float("nan")),
                 ("gpu_worker_free_gib", -1), ("worker_free_gib", True),
                 ("maximum_optimizer_calls", 6), ("attention", "sdpa"),
                 ("base_dtype", "float32"), ("eos_ids", [151645]),
                 ("contexts", [128]), ("weight_sha256", "0" * 64)]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises((RuntimeError, ValueError)):
                validate_config({**cfg, key: value})

    def test_reversible_cpu_optimizer_resume_is_bitwise(self):
        model = TinyModel()
        install_lora(model, torch)
        allowed = {n: p for n, p in model.named_parameters() if p.requires_grad}
        initial = {n: p.detach().clone() for n, p in allowed.items()}
        optimizer = torch.optim.AdamW(allowed.values(), lr=1e-3, weight_decay=0)
        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = completion_loss(torch, model(self.ids), self.ids, self.labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(allowed.values()), 1.0, error_if_nonfinite=True)
            optimizer.step()
            return loss.item(), {n: p.detach().clone() for n, p in allowed.items()}
        step()
        checkpoint = io.BytesIO()
        torch.save({"parameters": {n: p.detach().clone() for n, p in allowed.items()},
                    "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}, checkpoint)
        expected_loss, expected = step()
        checkpoint.seek(0)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for n, p in allowed.items():
                p.copy_(saved["parameters"][n])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        actual_loss, actual = step()
        self.assertEqual(expected_loss, actual_loss)
        self.assertTrue(all(torch.equal(expected[n], actual[n]) for n in expected))
        with torch.no_grad():
            for n, p in allowed.items():
                p.copy_(initial[n])
        self.assertTrue(all(torch.equal(initial[n], p) for n, p in allowed.items()))


if __name__ == "__main__":
    unittest.main()
