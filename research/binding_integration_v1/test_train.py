"""Tiny CPU integration/state tests; never open production model tensor files."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random
import sys
import tempfile
import unittest

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))
from research.binding_integration_v1 import train as t

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file
from experiments.model_quality_gpu_v1.optimizer import CpuAdamW


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.21, -0.14], [-0.09, 0.32]], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([0.03, -0.04], dtype=torch.float32))
        self.frozen = nn.Parameter(torch.tensor(1.25), requires_grad=False)

    def forward(self, value):
        return F.linear(value, self.weight, self.bias) * self.frozen


def named(model):
    return sorted((name, value) for name, value in model.named_parameters() if value.requires_grad)


def capture_rng():
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state().tolist(), "torch_cuda": []}


def restore_rng(value):
    random.setstate(value["python"])
    torch.set_rng_state(torch.tensor(value["torch_cpu"], dtype=torch.uint8))


def loss_sum(model, row):
    return F.cross_entropy(model(row["x"]), row["y"], reduction="sum")


def original_native_step(model, optimizer, rows, group, denominator, rate, design):
    """Independent transcription of the frozen native accumulation/update order."""
    model.train(); optimizer.zero_grad()
    for index in group:
        loss = loss_sum(model, rows[index])
        (loss / denominator).backward()
    torch.nn.utils.clip_grad_norm_([value for _, value in optimizer.named], design.clip_norm, error_if_nonfinite=True)
    optimizer.step(rate); optimizer.zero_grad()


class IntegrationNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        random.seed(17); torch.manual_seed(17)
        self.design = t.Design(processed_budget=100)
        self.rows = [
            {"tag": "native0", "x": torch.tensor([[1., 2.], [-1., .5]]), "y": torch.tensor([0, 1])},
            {"tag": "native1", "x": torch.tensor([[.2, -.3], [.8, .1], [-.4, -.5]]), "y": torch.tensor([1, 0, 1])},
            {"tag": "native2", "x": torch.tensor([[.7, .9], [1.1, -.6]]), "y": torch.tensor([0, 1])},
            {"tag": "binding0", "x": torch.tensor([[-2., .5], [.3, -1.2]]), "y": torch.tensor([0, 1])},
            {"tag": "binding1", "x": torch.tensor([[.4, 1.7]]), "y": torch.tensor([1])},
        ]
        self.native = [[0, 1], [2]]; self.added = [[3], [4]]
        self.denominators = [5, 2]; self.rates = [2.672470144728172e-06, 5.2924359644125685e-06]

    def test_zero_addend_recovers_parameters_and_both_adam_moments_bitwise(self):
        reference, candidate = TinyNet(), TinyNet()
        old = CpuAdamW(named(reference), self.design); new = CpuAdamW(named(candidate), self.design)
        seen = []
        def guarded_loss(model, row):
            if row["tag"].startswith("binding"):
                torch.rand(1); random.random()
                self.fail("disabled binding consumed a forward or RNG")
            seen.append(row["tag"])
            return loss_sum(model, row)
        before_rng = capture_rng()
        for index in range(2):
            original_native_step(reference, old, self.rows, self.native[index], self.denominators[index], self.rates[index], self.design)
            result = t._apply_update(candidate, self.rows, self.native[index], self.added[index], self.denominators[index],
                                     self.rates[index], new, self.design, guarded_loss, binding_enabled=False)
            self.assertEqual(result["binding_forward_calls"], 0)
        self.assertEqual(seen, ["native0", "native1", "native2"])
        self.assertEqual(t.canonical(before_rng), t.canonical(capture_rng()))
        self.assertEqual(old.step_count, new.step_count)
        for (name, left), (_, right) in zip(old.named, new.named):
            self.assertTrue(t._same_bits(left, right), name)
            self.assertTrue(all(t._same_bits(a, b) for a, b in zip(old.state[name], new.state[name])), name)

    def test_active_sum_uses_native_denominator_and_matches_joint_gradient(self):
        integrated, reference = TinyNet(), TinyNet()
        actual_opt = CpuAdamW(named(integrated), self.design); reference_opt = CpuAdamW(named(reference), self.design)
        selected = self.native[0] + self.added[0]
        reference.train(); reference_opt.zero_grad()
        joint = sum(loss_sum(reference, self.rows[index]) for index in selected) / self.denominators[0]
        joint.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for _, p in reference_opt.named))
        torch.nn.utils.clip_grad_norm_([p for _, p in reference_opt.named], self.design.clip_norm, error_if_nonfinite=True)
        reference_opt.step(self.rates[0]); reference_opt.zero_grad()
        observed = t._apply_update(integrated, self.rows, self.native[0], self.added[0], self.denominators[0],
                                   self.rates[0], actual_opt, self.design, loss_sum, binding_enabled=True)
        self.assertAlmostEqual(observed["objective_ce"], float(joint.detach()), places=6)
        self.assertEqual((observed["native_forward_calls"], observed["binding_forward_calls"]), (2, 1))
        self.assertGreater(observed["binding_objective_native_normalized"], 0)
        for (name, left), (_, right) in zip(actual_opt.named, reference_opt.named):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
            for actual, expected in zip(actual_opt.state[name], reference_opt.state[name]):
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertIsNone(integrated.frozen.grad)

    def disposable(self, model, *, fail=False):
        def step(optimizer, index):
            torch.rand(7); random.random()
            if fail and index == 1:
                raise RuntimeError("synthetic prefix failure")
            return t._apply_update(model, self.rows, self.native[index], self.added[index], self.denominators[index],
                                   self.rates[index], optimizer, self.design, loss_sum, binding_enabled=False)
        def compare(optimizer):
            self.assertEqual(optimizer.step_count, 2)
            return {"live_optimizer_step_count": optimizer.step_count, "live_state_differs_from_initial": True}
        return t._disposable_prefix(model, named(model), lambda: CpuAdamW(named(model), self.design), step, compare, capture_rng, restore_rng)

    def test_successful_prefix_restores_initial_state_rng_and_model_mode(self):
        model = TinyNet(); model.eval(); initial = copy.deepcopy(model.state_dict()); rng = capture_rng()
        result = self.disposable(model)
        self.assertIsNone(result["failure"])
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertTrue(result["trainable_state_restored_exactly"] and result["rng_restored_exactly"])
        self.assertTrue(result["qualification_optimizer_discarded"] and result["initial_model_training_mode_restored"])
        self.assertFalse(model.training)
        for name, value in model.state_dict().items():
            self.assertTrue(t._same_bits(value, initial[name]), name)
        self.assertEqual(t.canonical(capture_rng()), t.canonical(rng))
        fresh = CpuAdamW(named(model), self.design)
        self.assertEqual(fresh.step_count, 0)
        self.assertTrue(all(not bool(value.any()) for state in fresh.state.values() for value in state))

    def test_failed_prefix_also_restores_initial_values_rng_and_clears_gradients(self):
        model = TinyNet(); initial = copy.deepcopy(model.state_dict()); rng = capture_rng()
        result = self.disposable(model, fail=True)
        self.assertEqual(result["failure"]["type"], "RuntimeError")
        self.assertEqual(result["optimizer_updates"], 1)
        self.assertTrue(result["trainable_state_restored_exactly"] and result["rng_restored_exactly"])
        self.assertEqual(t.canonical(capture_rng()), t.canonical(rng))
        self.assertTrue(all(t._same_bits(value, initial[name]) for name, value in model.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_reference_comparison_checks_moments_not_just_parameters(self):
        model = TinyNet(); optimizer = CpuAdamW(named(model), self.design)
        for index in range(2):
            original_native_step(model, optimizer, self.rows, self.native[index], self.denominators[index], self.rates[index], self.design)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp).resolve(); entries = []
            for index, (name, parameter) in enumerate(optimizer.named):
                first, second = optimizer.state[name]
                state = {"parameter": parameter.detach().cpu().clone(), "exp_avg": first.clone(), "exp_avg_sq": second.clone()}
                if index == 0:
                    state["exp_avg"].reshape(-1)[0] += 0.01
                path = folder / f"tensor-{index:04d}.safetensors"; save_file(state, str(path))
                entries.append({"name": name, "file": path.name, "shape": list(parameter.shape), "sha256": t.file_hash(path)})
            checks = t._compare_reference_state(optimizer, {"entries": entries}, folder)
        self.assertTrue(all(value["parameter"]["equal"] and value["exp_avg_sq"]["equal"] for value in checks))
        self.assertFalse(checks[0]["exp_avg"]["equal"])
        self.assertNotEqual(checks[0]["exp_avg"]["actual_sha256"], checks[0]["exp_avg"]["expected_sha256"])

    def test_bitwise_contract_distinguishes_signed_zero(self):
        positive, negative = torch.tensor([0.0]), torch.tensor([-0.0])
        self.assertTrue(torch.equal(positive, negative))
        self.assertFalse(t._same_bits(positive, negative))
        self.assertNotEqual(t._tensor_sha("x", "parameter", positive), t._tensor_sha("x", "parameter", negative))


class ContractTests(unittest.TestCase):
    def test_binding_subsequence_floor_partition_preserves_native_groups(self):
        prior = [
            {"source": "native_dual899", "source_index": 0, "mixture_index": 0},
            {"source": "binding_v2", "source_index": 2, "mixture_index": 1},
            {"source": "native_dual899", "source_index": 1, "mixture_index": 2},
            {"source": "binding_v2", "source_index": 0, "mixture_index": 3},
            {"source": "native_dual899", "source_index": 2, "mixture_index": 4},
            {"source": "binding_v2", "source_index": 1, "mixture_index": 5},
            {"source": "native_dual899", "source_index": 3, "mixture_index": 6},
        ]
        actual = t.expected_membership([[0, 1], [2, 3]], prior)
        self.assertEqual(actual, [("native_dual899", 0, 0, 0), ("native_dual899", 1, 2, 0), ("binding_v2", 2, 1, 0),
                                  ("native_dual899", 2, 4, 1), ("native_dual899", 3, 6, 1), ("binding_v2", 0, 3, 1), ("binding_v2", 1, 5, 1)])

    def test_qualification_charge_and_checkpoint_ceiling_fit_unchanged_caps(self):
        self.assertEqual(577417 + 8796 + t.ZERO_PREFIX_TOKENS, t.EXPECTED_CHARGED_TOKENS)
        self.assertEqual(48 + t.ZERO_PREFIX_ROWS, t.EXPECTED_QUALIFICATION_CALLS)
        self.assertLessEqual(t.EXPECTED_CHARGED_TOKENS, t.LIMITS["processed_training_tokens_all_attempts"])
        self.assertLessEqual(t.EXPECTED_QUALIFICATION_CALLS, t.LIMITS["evaluation_and_qualification_model_calls"])
        self.assertLessEqual(2 * t.CHECKPOINT_BYTES, t.LIMITS["new_disk_bytes"])

    def test_qualified_adapter_implementation_is_an_exact_copy(self):
        original = WORKTREE / "research/binding_mixture_v1/adapters.py"
        current = Path(t.__file__).parent / "adapters.py"
        self.assertEqual(hashlib.sha256(original.read_bytes()).digest(), hashlib.sha256(current.read_bytes()).digest())


if __name__ == "__main__":
    unittest.main()
