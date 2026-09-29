"""Synthetic CPU mechanics only. No model is constructed or any CUDA API invoked."""
from __future__ import annotations

import importlib.util
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from experiments.rineya_evidence_function_equilibrium_v1.fresh_validation import native_adapter as n


class ClosedEntryTests(unittest.TestCase):
    def test_import_is_lazy_and_launch_cannot_be_flag_unlocked(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys; from experiments.rineya_evidence_function_equilibrium_v1.fresh_validation "
            "import native_adapter; assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for protocol in ({"native_launch_enabled": False}, {"native_launch_enabled": True}):
            with self.assertRaises(n.Blocked):
                n.load_candidate({}, "independent", protocol=protocol)

    def test_missing_asset_fails_before_runtime_or_model_load(self):
        with tempfile.TemporaryDirectory() as root, patch.object(n, "runtime_identity") as runtime:
            with self.assertRaisesRegex(n.Blocked, "missing exact source"):
                n.asset_binding(Path(root), Path(root) / "T", Path(root) / "H")
            runtime.assert_not_called()

    def test_inventory_refuses_one_missing_extra_or_wrong_draw_key(self):
        n.check_inventory(n.SOURCE_KEYS, n.ENDPOINT_KEYS, n.ENDPOINT_KEYS)
        for keys in (set(n.ENDPOINT_KEYS) - {next(iter(n.ENDPOINT_KEYS))},
                     set(n.ENDPOINT_KEYS) | {"model.layers.7.mlp.up_proj.weight"}):
            with self.assertRaises(n.Blocked):
                n.check_inventory(n.SOURCE_KEYS, keys, n.ENDPOINT_KEYS)
        with self.assertRaises(n.Blocked):
            n.check_inventory(set(n.SOURCE_KEYS) | {"lm_head.weight"}, n.ENDPOINT_KEYS, n.ENDPOINT_KEYS)

    def test_rehashed_binding_cannot_change_arithmetic_runtime_or_endpoint(self):
        body = {"schema": 1, "experiment": "EFE-FUNC-001-FRESH-STAGE1",
            "source_dir": "synthetic-path-never-opened", "source_assets": dict(n.SOURCE_ASSETS),
            "endpoints": {s: {"path": "synthetic-path-never-opened", "sha256": h}
                          for s, h in n.ENDPOINT_SHA.items()}, "runtime": dict(n.RUNTIME_PINS),
            "arms": {k: list(v) for k, v in n.FROZEN_ARMS.items()}, "rank": 16,
            "svd_cpu_threads": n.SVD_THREADS, "arithmetic": n.ARITHMETIC,
            "original_untouched_storage": "BF16; exact upcast in all-FP32 loaded function",
            "qualification_status": "NOT_RUN", "native_launch_enabled": False}
        n.validate_binding({**body, "binding_sha256": n.digest(body)})
        for key, replacement in (("arithmetic", "FP64 subtraction"), ("rank", 8),
                                  ("native_launch_enabled", True), ("qualification_status", "PASS")):
            changed = copy.deepcopy(body)
            changed[key] = replacement
            with self.assertRaises(n.Blocked):
                n.validate_binding({**changed, "binding_sha256": n.digest(changed)})
        for key, nested, replacement in (("runtime", "torch", "2.6.0"),
                                         ("endpoints", "T", {"path": "other", "sha256": "0" * 64})):
            changed = copy.deepcopy(body)
            changed[key][nested] = replacement
            with self.assertRaises(n.Blocked):
                n.validate_binding({**changed, "binding_sha256": n.digest(changed)})

    def test_frozen_arms_and_zero_optional_budget(self):
        self.assertEqual(n.FROZEN_ARMS["independent"], (1.0, 0.25))
        self.assertEqual(n.FROZEN_ARMS["H025"], (0.0, 0.25))
        self.assertEqual(n.FROZEN_ARMS["H"], (0.0, 1.0))
        with self.assertRaises(TypeError):
            n.FROZEN_ARMS["independent"] = (1.0, 1.0)
        self.assertEqual(n.RAW_SUM_BUDGET + n.SHARED_SCALE_BUDGET, 0)
        with self.assertRaises(n.Blocked):
            n.load_candidate({}, "raw_sum", protocol={"native_launch_enabled": True})


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable: synthetic tensor mechanics skipped")
class SyntheticTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)  # tiny CPU fixtures; no native SVD identity claim

    def test_rank16_known_diagonal_and_full_vector(self):
        t = self.torch
        values = t.arange(24, 0, -1, dtype=t.float64)
        projected = n.p16(t.diag(values))
        expected = t.diag(t.cat([values[:16], t.zeros(8, dtype=t.float64)]))
        self.assertTrue(t.equal(projected, expected))
        vector = t.tensor([0.1, -0.2], dtype=t.float32)
        self.assertTrue(t.equal(n.p16(vector), vector.double()))

    def test_original_subtracts_fp32_before_fp64(self):
        t = self.torch
        source = t.ones(1, dtype=t.bfloat16)
        endpoint = t.tensor([1e-9], dtype=t.float32)
        result = n.compose_tensor(source, endpoint, source.float(), "T")
        double_subtraction = (source.double() + (endpoint.double() - source.double())).float()
        self.assertEqual(float(result[0]), 0.0)
        self.assertGreater(float(double_subtraction[0]), 0.0)

    def test_projected_components_sum_before_single_fp32_round(self):
        t = self.torch
        g = t.Generator().manual_seed(782)
        source = t.ones((24, 24), dtype=t.bfloat16)
        a = source.float() + t.randn(24, 24, generator=g) * 1e-5
        b = source.float() + t.randn(24, 24, generator=g) * 1e-5
        dt, dh = n.p16(a - source.float()), n.p16(b - source.float())
        expected = (source.double() + (0.25 * dh + dt)).float()
        prematurely_rounded = ((source.double() + dt).float().double() + 0.25 * dh).float()
        self.assertTrue(t.equal(n.compose_tensor(source, a, b, "independent"), expected))
        self.assertFalse(t.equal(expected, prematurely_rounded))

    def test_wrong_precision_shape_nonfinite_and_free_coefficient_rejected(self):
        t = self.torch
        source = t.ones(3, dtype=t.bfloat16)
        endpoint = source.float()
        for bad in (endpoint.bfloat16(), endpoint[:2], t.full((3,), float("nan"))):
            with self.assertRaises(ValueError):
                n.compose_tensor(source, bad, endpoint, "independent")
        with self.assertRaises(ValueError):
            n.compose_tensor(source, endpoint, endpoint, "t1h0.5")
        self.assertTrue(t.equal(n.compose_tensor(source, None, None, "source"), source.float()))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable: fake decoding mechanics skipped")
class SyntheticGenerationTests(unittest.TestCase):
    def fake_model(self, next_tokens):
        import torch
        class Scripted:
            def __init__(self):
                self.calls = []
            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                logits = torch.full((1, 1, 151646), -10.0)
                logits[0, 0, next_tokens[len(self.calls)-1]] = 10.0
                return SimpleNamespace(logits=logits, past_key_values=("fake", len(self.calls)))
        return Scripted()

    def test_eos_full_kv_and_limit_forward_counts(self):
        model = self.fake_model([73, n.EOS_IDS[1]])
        stages = []
        result = n.greedy_decode(model, [9, 8], max_new_tokens=4, max_context=6,
                                 guard=stages.append)
        self.assertEqual(result["token_ids"], [73, n.EOS_IDS[1]])
        self.assertEqual(result["body_token_ids"], [73])
        self.assertTrue(result["terminated"])
        self.assertEqual(result["forward_calls"], 2)
        self.assertEqual(stages, ["prefill", "decode"])
        self.assertIsNone(model.calls[0]["past_key_values"])
        self.assertEqual(model.calls[1]["past_key_values"], ("fake", 1))
        self.assertEqual(model.calls[1]["input_ids"].tolist(), [[73]])
        limited = self.fake_model([40, 41])
        result = n.greedy_decode(limited, [1], max_new_tokens=2, max_context=3,
                                 guard=lambda stage: None)
        self.assertFalse(result["terminated"])
        self.assertEqual(result["forward_calls"], 2)  # no unnecessary third forward
        self.assertEqual(result["stop_reason"], "max_new_tokens")
        self.assertEqual(result["qualification_status"], "NOT_RUN")

    def test_overflow_and_guard_failure_precede_any_forward(self):
        model = self.fake_model([1])
        result = n.greedy_decode(model, [1, 2], max_new_tokens=2, max_context=3,
                                 guard=lambda stage: None)
        self.assertEqual(result["status"], "skipped_context_overflow")
        self.assertEqual(model.calls, [])
        def refuse(stage):
            raise n.Blocked("synthetic resource refusal")
        with self.assertRaises(n.Blocked):
            n.greedy_decode(model, [1], max_new_tokens=1, max_context=2, guard=refuse)
        self.assertEqual(model.calls, [])
        with self.assertRaises(n.Blocked):
            n.greedy_decode(model, [1], max_new_tokens=1, max_context=2,
                            guard=lambda stage: None, device="cuda")


if __name__ == "__main__":
    unittest.main()
