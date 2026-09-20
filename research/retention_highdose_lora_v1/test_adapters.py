"""Tiny CPU mechanics only. No source models, CUDA, downloads, or scientific claims."""
from dataclasses import FrozenInstanceError
import hashlib
import json
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from . import adapters


class ScaleNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * self.weight


class TinyBlock(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.input_layernorm = ScaleNorm(width)
        self.post_attention_layernorm = ScaleNorm(width)
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, nn.Linear(width, width, bias=False))
        self.self_attn.q_norm = ScaleNorm(width)
        self.self_attn.k_norm = ScaleNorm(width)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(width, 2 * width, bias=False)
        self.mlp.up_proj = nn.Linear(width, 2 * width, bias=False)
        self.mlp.down_proj = nn.Linear(2 * width, width, bias=False)

    def forward(self, x):
        z = self.input_layernorm(x)
        a = self.self_attn
        x = x + a.o_proj(a.q_norm(a.q_proj(z)) + a.k_norm(a.k_proj(z)) + a.v_proj(z))
        z = self.post_attention_layernorm(x)
        return x + self.mlp.down_proj(F.silu(self.mlp.gate_proj(z)) * self.mlp.up_proj(z))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock() for _ in range(3)])
        self.outside = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return self.outside(x)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = TinyModel()
        self.x = torch.randn(2, 5, 4)

    def inject(self):
        return adapters._inject_scope(self.model, layer_start=1, layer_stop=3, rank=2, alpha=4.0)

    def test_zero_b_is_exact_and_original_weight_objects_survive(self):
        before = self.model(self.x).detach()
        original = dict(self.model.named_parameters())
        named, spec = self.inject()
        torch.testing.assert_close(self.model(self.x), before, rtol=0, atol=0)
        self.assertEqual(len(named), 36)
        self.assertEqual(tuple(name for name, _ in named), tuple(sorted(spec.trainable_names)))
        self.assertEqual(len(spec.projections), 14)
        self.assertEqual(len(spec.norm_names), 8)
        self.assertEqual(sum(p.numel() for _, p in named), spec.trainable_parameters)
        actual = dict(self.model.named_parameters())
        for name, parameter in original.items():
            self.assertIs(actual[name], parameter)
        for projection in spec.projections:
            self.assertFalse(actual[projection.name + ".weight"].requires_grad)
        with self.assertRaises(FrozenInstanceError):
            spec.rank = 3

    def test_zero_b_gradient_contract_and_cpu_adam_update(self):
        named, spec = self.inject()
        frozen = adapters.frozen_base_digest(self.model, spec)
        self.model(self.x).square().sum().backward()
        report = adapters.gradient_summary(self.model, spec)
        self.assertEqual(report["finite_tensors"], 36)
        self.assertEqual(report["active_adapter_A"], 0)
        self.assertEqual(report["active_adapter_B"], 14)
        self.assertEqual(report["active_norms"], 8)
        self.assertTrue(report["zero_B_initialization"])
        # Exercise the existing optimizer interface, not a parallel LoRA optimizer.
        from experiments.model_quality_gpu_v1.optimizer import CpuAdamW
        from experiments.model_quality_gpu_v1.config import Design
        optimizer = CpuAdamW(named, Design())
        optimizer.step(1e-4)
        optimizer.zero_grad()
        self.assertEqual(adapters.frozen_base_digest(self.model, spec), frozen)
        self.assertTrue(any(bool(p.count_nonzero()) for name, p in named if name.endswith(".lora_B")))

    def test_nonzero_adapter_dense_merge_matches_forward(self):
        _, spec = self.inject()
        with torch.no_grad():
            for projection in spec.projections:
                self.model.get_submodule(projection.name).lora_B.normal_(0, 0.1)
        dense = dict(adapters.iter_dense_updates(self.model, spec))
        self.assertEqual(len(dense), 22)
        parameters = dict(self.model.named_parameters())
        for projection in spec.projections:
            module = self.model.get_submodule(projection.name)
            x = torch.randn(3, projection.in_features)
            torch.testing.assert_close(module(x), F.linear(x, dense[projection.name + ".weight"]),
                                       rtol=1e-5, atol=1e-6)
            self.assertIsNot(dense[projection.name + ".weight"], module.weight)
        for name in spec.norm_names:
            torch.testing.assert_close(dense[name], parameters[name], rtol=0, atol=0)
            self.assertNotEqual(dense[name].data_ptr(), parameters[name].data_ptr())

    def test_temporary_cpu_dense_model_hashes_forward_and_restore(self):
        _, spec = self.inject()
        with torch.no_grad():
            for projection in spec.projections:
                self.model.get_submodule(projection.name).lora_B.normal_(0, 0.1)
            for name in spec.norm_names:
                dict(self.model.named_parameters())[name].add_(0.02)
        modules = {p.name: self.model.get_submodule(p.name) for p in spec.projections}
        values = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        frozen = adapters.frozen_base_digest(self.model, spec)
        before = self.model(self.x).detach()
        rng = torch.get_rng_state().clone()
        expected = dict(adapters.iter_dense_updates(self.model, spec))
        with adapters.temporary_dense_merge(self.model, spec) as hashes:
            self.assertEqual(set(hashes), set(expected))
            for name, value in expected.items():
                self.assertEqual(value.device.type, "cpu")
                metadata = json.dumps([name, list(value.shape), "torch.float32"],
                                      ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                independent = hashlib.sha256(metadata + value.numpy().tobytes()).hexdigest()
                self.assertEqual(hashes[name], independent)
            torch.testing.assert_close(self.model(self.x), before, rtol=1e-5, atol=1e-6)
            self.assertTrue(all(type(self.model.get_submodule(p.name)) is nn.Linear for p in spec.projections))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, module in modules.items():
            self.assertIs(self.model.get_submodule(name), module)
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter, values[name], rtol=0, atol=0)
        self.assertEqual(adapters.frozen_base_digest(self.model, spec), frozen)
        with self.assertRaisesRegex(RuntimeError, "qualification failed"):
            with adapters.temporary_dense_model(self.model, spec):
                raise RuntimeError("qualification failed")
        for name, module in modules.items():
            self.assertIs(self.model.get_submodule(name), module)
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter, values[name], rtol=0, atol=0)

    def test_partial_dense_merge_failure_restores_prior_modules(self):
        _, spec = self.inject()
        originals = {p.name: self.model.get_submodule(p.name) for p in spec.projections}
        values = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        merge = adapters.canonical_cpu_merge
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated partial merge failure")
            return merge(*args, **kwargs)

        with patch.object(adapters, "canonical_cpu_merge", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "partial merge failure"):
                with adapters.temporary_dense_merge(self.model, spec):
                    self.fail("failed merge must not yield")
        for name, module in originals.items():
            self.assertIs(self.model.get_submodule(name), module)
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter, values[name], rtol=0, atol=0)

    def test_canonical_merge_refuses_precision_and_nonfinite_inputs(self):
        weight, A, B = torch.ones(4, 4), torch.ones(2, 4), torch.ones(4, 2)
        expected = weight + (B @ A) * 2
        merged = adapters.canonical_cpu_merge(weight, A, B)
        torch.testing.assert_close(merged, expected, rtol=0, atol=0)
        self.assertEqual(merged.device.type, "cpu")
        self.assertNotEqual(merged.data_ptr(), weight.data_ptr())
        with self.assertRaises(ValueError):
            adapters.canonical_cpu_merge(weight.bfloat16(), A, B)
        B[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            adapters.canonical_cpu_merge(weight, A, B)

    def test_frozen_digest_covers_projection_and_ignores_only_adapters_and_norms(self):
        _, spec = self.inject()
        initial = adapters.frozen_base_digest(self.model, spec)
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            params[spec.norm_names[0]].add_(0.25)
            params[spec.adapter_names[0]].add_(0.5)
        self.assertEqual(adapters.frozen_base_digest(self.model, spec), initial)
        with torch.no_grad():
            params[spec.projections[0].name + ".weight"].add_(0.125)
        self.assertNotEqual(adapters.frozen_base_digest(self.model, spec), initial)

    def test_wrong_geometry_rejected_without_partial_injection(self):
        before = {name: p.requires_grad for name, p in self.model.named_parameters()}
        with self.assertRaises(ValueError):
            adapters.inject_lora(self.model)
        self.assertFalse(any(isinstance(m, adapters.LoRALinear) for m in self.model.modules()))
        self.assertEqual(before, {name: p.requires_grad for name, p in self.model.named_parameters()})
        self.model.model.layers[2].mlp.down_proj = nn.Linear(8, 4, bias=True)
        with self.assertRaises(ValueError):
            self.inject()
        self.assertFalse(any(isinstance(m, adapters.LoRALinear) for m in self.model.modules()))

    def test_gradient_or_trainable_tampering_is_refused(self):
        named, spec = self.inject()
        self.model(self.x).square().sum().backward()
        named[0][1].grad.fill_(float("nan"))
        with self.assertRaises(ValueError):
            adapters.gradient_summary(self.model, spec)
        self.model.zero_grad(set_to_none=True)
        self.model.outside.weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            adapters.named_trainables(self.model, spec)


if __name__ == "__main__":
    unittest.main()
