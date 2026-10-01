"""Pure admission tests always; tiny CPU graph tests require explicit clearance.

Set COMP_RS_RUN_NATIVE_CPU_TESTS=1 only after the operator clears RAM/resources. No GPU,
pretrained loader, dataset read or native study is used by these fixtures.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import update as u


def recipe(**changes):
    value = {"schema": u.SCHEMA, "synthetic_cpu": True, "approved": False, "training_authorized": False,
             "B": 2, "G": 8, "boundary_id": 0, "generation": 0, "symbolic_snapshot": "b" * 64,
             "start_policy_id": "a" * 64, "trainable_names": ["weight"], "eos_ids": [2], "pad_id": 0,
             "max_new_tokens": 4, "objective": {"method": "maxrl", "epsilon": 1e-6},
             "ratio_clipping": None, "gradient_clipping": None, "anchor_KL": {"enabled": False, "coefficient": 0},
             "gradient_checkpointing": {"use_reentrant": False},
             "optimizer": {"class": "torch.optim.AdamW", "lr": 1e-6, "betas": [0.9, 0.999], "eps": 1e-8,
                           "weight_decay": 0.0, "foreach": False, "capturable": False}}
    value.update(changes)
    return value


class PureTests(unittest.TestCase):
    def test_streamed_tensor_header_is_exact_registered_compact_convention(self):
        expected=hashlib.sha256(b'{"dtype":"torch.float32","shape":[2,3]}\n').hexdigest()
        self.assertEqual(u._tensor_header_hash("torch.float32",[2,3]).hexdigest(),expected)
        # Independent source binding: the arm/runtime conventions really use
        # separators(',', ':'); a whitespace-sensitive digest is intentional.
        self.assertNotEqual(expected,hashlib.sha256((json.dumps({"dtype":"torch.float32","shape":[2,3]},
                           sort_keys=True)+"\n").encode()).hexdigest())
    def test_no_framework_import_is_needed_for_recipe_and_labels(self):
        u.validate_recipe(recipe())
        u.validate_trajectory([3, 2], "eos", [-1.0, -2.0], [2], 4)
        u.validate_trajectory([3, 4, 5, 6], "length", [-1.0] * 4, [2], 4)

    def test_unapproved_native_refused_before_model_access(self):
        with self.assertRaisesRegex(ValueError, "authorization"):
            u.boundary_update(object(), object(), [], [], recipe(synthetic_cpu=False), lambda *a, **k: None)

    def test_no_hidden_optimizer_clipping_or_KL_defaults(self):
        for field in ("optimizer", "ratio_clipping", "gradient_clipping", "anchor_KL", "gradient_checkpointing"):
            value = recipe()
            del value[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                u.validate_recipe(value)
        for value in (recipe(ratio_clipping={"epsilon": .2}), recipe(gradient_clipping={"norm": 1}),
                      recipe(anchor_KL={"enabled": True, "coefficient": .01})):
            with self.assertRaises(ValueError):
                u.validate_recipe(value)

    def test_stop_contract_refuses_fake_or_internal_EOS(self):
        for tokens, reason in (([3, 2, 4], "eos"), ([3], "eos"), ([3, 4], "length"), ([3, 4, 5, 2], "length")):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                u.validate_trajectory(tokens, reason, [-1.0] * len(tokens), [2], 4)

    def test_uniform_and_fixed_denominator_algebra(self):
        self.assertEqual(u.group_advantages([1] * 8, recipe()["objective"]), [0.0] * 8)
        a = u.group_advantages([1] + [0] * 7, recipe()["objective"])
        self.assertAlmostEqual(sum(a), 0.0)
        sums = [-2.0] + [-1.0] * 7
        group = -sum(x * y for x, y in zip(a, sums)) / 8
        boundary = -sum(x * y for x, y in zip([0.0] * 8 + a, [-3.0] * 8 + sums)) / 16
        self.assertAlmostEqual(boundary * 2, group)

    def test_grpo_is_explicit_CPU_only(self):
        u.validate_recipe(recipe(objective={"method": "grpo", "epsilon": 1e-6, "ddof": 1}))
        with self.assertRaisesRegex(ValueError, "CPU only"):
            u.validate_recipe(recipe(synthetic_cpu=False, approved=True, training_authorized=True,
                eos_ids=[151645, 151643], pad_id=151643, max_new_tokens=128,
                objective={"method": "grpo", "epsilon": 1e-6, "ddof": 1}))

    def test_observer_span_has_no_duplicate_kind_and_accounts_failures(self):
        events = []
        def observer(kind, phase, **meta):
            events.append((kind, phase, meta))
        with u._span(observer, "checkpoint", {"generation": 2}) as record:
            self.assertEqual(record["generation"], 2)
        self.assertEqual([event[1] for event in events], ["started", "returned"])
        self.assertEqual(events[0][2]["operation_id"], events[1][2]["operation_id"])
        with self.assertRaisesRegex(ValueError, "example"):
            with u._span(observer, "backward", {}):
                raise ValueError("example")
        self.assertEqual(events[-1][1], "failed")

    def test_failed_receipt_preserves_physical_work_error(self):
        physical = ValueError('physical backward failure')
        def observer(kind, phase, **metadata):
            if phase == 'failed': raise RuntimeError('failed receipt write')
        with self.assertRaises(ValueError) as captured:
            with u._span(observer, 'backward', {}): raise physical
        self.assertIs(captured.exception, physical)
        self.assertEqual(physical.accounting_failure, {'type': 'RuntimeError', 'message': 'failed receipt write'})
        self.assertIsInstance(physical.__cause__, RuntimeError)


@unittest.skipUnless(os.environ.get("COMP_RS_RUN_NATIVE_CPU_TESTS") == "1", "CPU framework tests await explicit resource clearance")
class TinyCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from safetensors.torch import save_file
        from experiments.maxrl_comp002_pretraining_v1 import s0_runtime, arm_build
        cls.torch, cls.save_file, cls.runtime, cls.arm_build = torch, staticmethod(save_file), s0_runtime, arm_build

    def toy(self):
        torch = self.torch
        from functools import partial
        from types import SimpleNamespace
        from torch.utils.checkpoint import checkpoint
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)
                self.gradient_checkpointing = False
            def forward(self, x):
                if self.training and self.gradient_checkpointing:
                    return torch.tanh(self._gradient_checkpointing_func(self.linear, x))
                return torch.tanh(self.linear(x))
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(32, 8)
                self.layers = torch.nn.ModuleList([Layer()])
            def forward(self, input_ids, **unused):
                hidden = self.embed_tokens(input_ids).cumsum(dim=1)
                for layer in self.layers:
                    hidden = layer(hidden)
                return SimpleNamespace(last_hidden_state=hidden)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Decoder()
                self.lm_head = torch.nn.Linear(8, 32, bias=False)
                self.config = SimpleNamespace(vocab_size=32)
                self.is_gradient_checkpointing = False
            def get_output_embeddings(self):
                return self.lm_head
            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
                self.is_gradient_checkpointing = True
                for layer in self.model.layers:
                    layer.gradient_checkpointing = True
                    layer._gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)
        torch.manual_seed(7)
        model = Toy().float().eval()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("model.layers."))
        return model

    def optimizer(self, model, value):
        opt = dict(value["optimizer"])
        del opt["class"]
        opt["betas"] = tuple(opt["betas"])
        return self.torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], **opt)

    def inputs(self, model, uniform=False):
        torch = self.torch
        value = recipe(trainable_names=sorted(n for n, p in model.named_parameters() if p.requires_grad),
                       start_policy_id=self.runtime.anchor_digest(model))
        prompts = [{"id": "p0", "prompt_ids": [7, 8]}, {"id": "p1", "prompt_ids": [11, 12, 13]}]
        groups = []
        with torch.no_grad():
            for index, prompt in enumerate(prompts):
                group = []
                for sample in range(8):
                    tokens = [3 + sample % 5, 2] if sample % 2 else [3, 4, 5, 6]
                    ids = torch.tensor([prompt["prompt_ids"] + tokens])
                    hidden = model.model(input_ids=ids).last_hidden_state
                    full = model.lm_head(hidden).float().log_softmax(-1)[0]
                    raw = [float(full[len(prompt["prompt_ids"]) - 1 + k, token]) for k, token in enumerate(tokens)]
                    group.append({"prompt_id": prompt["id"], "rollout_id": f"{index}-{sample}",
                        "policy_id": value["start_policy_id"], "completion_ids": tokens,
                        "stop_reason": "eos" if sample % 2 else "length", "raw_step_logprobs": raw,
                        "reward": 1 if uniform or (index == 0 and sample == 0) else 0})
                groups.append(group)
        return prompts, groups, value

    def test_accumulation_matches_independent_one_shift_reference(self):
        torch = self.torch
        model = self.toy()
        reference = copy.deepcopy(model)
        prompts, groups, value = self.inputs(model)
        optimizer = self.optimizer(model, value)
        ref_opt = self.optimizer(reference, value)
        advantages = [a for group in groups for a in u.group_advantages([r["reward"] for r in group], value["objective"])]
        losses, i = [], 0
        for prompt, group in zip(prompts, groups):
            for sample in group:
                ids = torch.tensor([prompt["prompt_ids"] + sample["completion_ids"]])
                full = reference.lm_head(reference.model(input_ids=ids).last_hidden_state).float().log_softmax(-1)[0]
                p = len(prompt["prompt_ids"])
                selected = torch.stack([full[p - 1 + k, token] for k, token in enumerate(sample["completion_ids"])])
                losses.append(-(advantages[i] / 16) * selected.sum())
                i += 1
        torch.stack(losses).sum().backward()
        expected = {n: p.grad.clone() for n, p in reference.named_parameters() if p.requires_grad}
        captured, events = {}, []
        original = optimizer.step
        def capture_step(*args, **kwargs):
            captured.update({n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad})
            return original(*args, **kwargs)
        with patch.object(optimizer, "step", capture_step):
            result = u.boundary_update(model, optimizer, prompts, groups, value,
                                      lambda kind, phase, **meta: events.append((kind, phase, meta)))
        self.assertEqual((result["U_before"], result["U_after"]), (0, 1))
        self.assertTrue(result["updated"])
        for name in expected:
            torch.testing.assert_close(captured[name], expected[name], atol=1e-6, rtol=1e-5)
        selected = [(kind, phase) for kind, phase, _ in events if kind in ("loss_forward", "backward")]
        self.assertEqual(selected, [(k, p) for _ in range(16) for k, p in
            (("loss_forward", "started"), ("loss_forward", "returned"), ("backward", "started"), ("backward", "returned"))])
        self.assertEqual(result["optimizer_operation"]["generation"], value["generation"])
        self.assertEqual(result["optimizer_operation"]["symbolic_snapshot"], value["symbolic_snapshot"])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_uniform_skip_preserves_lazy_and_existing_adam(self):
        model = self.toy()
        prompts, groups, value = self.inputs(model, uniform=True)
        optimizer = self.optimizer(model, value)
        before = self.runtime.anchor_digest(model)
        result = u.boundary_update(model, optimizer, prompts, groups, value, lambda *a, **k: None)
        self.assertFalse(result["updated"])
        self.assertFalse(optimizer.state)
        self.assertEqual(self.runtime.anchor_digest(model), before)
        prompts, groups, value = self.inputs(model)
        u.boundary_update(model, optimizer, prompts, groups, value, lambda *a, **k: None)
        saved = copy.deepcopy(optimizer.state_dict())
        before = self.runtime.anchor_digest(model)
        prompts, groups, value = self.inputs(model, uniform=True)
        result = u.boundary_update(model, optimizer, prompts, groups, value, lambda *a, **k: None)
        self.assertEqual((result["U_before"], result["U_after"]), (1, 1))
        self.assertEqual(self.runtime.anchor_digest(model), before)
        for parameter in saved["state"]:
            for key in saved["state"][parameter]:
                self.assertTrue(self.torch.equal(saved["state"][parameter][key], optimizer.state_dict()["state"][parameter][key]))

    def test_late_parity_failure_has_no_step_and_clears_accumulated_gradients(self):
        model = self.toy()
        prompts, groups, value = self.inputs(model)
        optimizer = self.optimizer(model, value)
        before = self.runtime.anchor_digest(model)
        groups[-1][-1]["raw_step_logprobs"][0] -= .01
        events = []
        with self.assertRaisesRegex(ValueError, "parity"):
            u.boundary_update(model, optimizer, prompts, groups, value,
                              lambda kind, phase, **meta: events.append((kind, phase)))
        self.assertNotIn(("optimizer", "started"), events)
        self.assertFalse(optimizer.state)
        self.assertEqual(self.runtime.anchor_digest(model), before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_behavior_identity_and_optimizer_recipe_fail_closed(self):
        model = self.toy()
        prompts, groups, value = self.inputs(model)
        optimizer = self.optimizer(model, value)
        value["optimizer"]["lr"] = 2e-6
        with self.assertRaisesRegex(ValueError, "recipe drift"):
            u.boundary_update(model, optimizer, prompts, groups, value, lambda *a, **k: None)
        value["optimizer"]["lr"] = 1e-6
        with self.torch.no_grad():
            next(p for p in model.parameters() if p.requires_grad).add_(.01)
        with self.assertRaisesRegex(ValueError, "behavior snapshot"):
            u.boundary_update(model, optimizer, prompts, groups, value, lambda *a, **k: None)

    def test_uniform_group_does_not_bypass_actual_vocabulary_validation(self):
        model = self.toy()
        prompts, groups, value = self.inputs(model, uniform=True)
        optimizer = self.optimizer(model, value)
        groups[0][0]["completion_ids"][0] = 32
        with self.assertRaisesRegex(ValueError, "actual head vocabulary"):
            u.boundary_update(model, optimizer, prompts, groups, value, lambda *a, **k: None)
        self.assertFalse(optimizer.state)

    def donor_case(self, directory, sequential=False):
        torch = self.torch
        class Donor(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.weight = torch.nn.Parameter(value.clone())
                self.frozen = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
        base = {"weight": torch.tensor([1.0, 1.0], dtype=torch.bfloat16), "frozen": torch.tensor([3.0], dtype=torch.bfloat16)}
        warm = {"weight": torch.tensor([2.0 ** -23, 0.25], dtype=torch.float32)}
        base_path, warm_path = directory / "base.safetensors", directory / "warm.safetensors"
        self.save_file(base, base_path)
        warm_data = self.arm_build.write_delta(warm_path, warm)
        common = Donor(base["weight"].float() + warm["weight"])
        start = copy.deepcopy(common)
        components = []
        if sequential:
            first = {"weight": torch.tensor([.125, -.125])}
            first_path = directory / "first.safetensors"
            first_data = self.arm_build.write_delta(first_path, first)
            with torch.no_grad():
                start.weight.add_(first["weight"])
            components = [{"path": str(first_path), "coefficient": 1.0, "file_sha256": u._file_sha(first_path),
                           "data_sha256": first_data, "state_sha256": self.arm_build.state_digest(first)}]
        final = copy.deepcopy(start)
        with torch.no_grad():
            final.weight.add_(torch.tensor([.25, -.25]))
        names = ["weight"]
        final_digests = self.runtime.state_digests(final)
        bindings = {"schema": u.DONOR_SCHEMA, "synthetic_cpu": True, "approved": False, "training_authorized": False,
            "base_weights_sha256": u._file_sha(base_path), "warm_delta_file_sha256": u._file_sha(warm_path),
            "warm_delta_data_sha256": warm_data, "warm_delta_state_sha256": self.arm_build.state_digest(warm),
            "common_anchor_sha256": self.runtime.anchor_digest(common), "own_start_sha256": self.runtime.anchor_digest(start),
            "own_start_kind": "sequential" if sequential else "independent", "own_start_components": components,
            "final_master_sha256": self.runtime.anchor_digest(final),
            "frozen_state_sha256": self.runtime.combine({k: v for k, v in final_digests.items() if k not in names}),
            "trainable_names_sha256": u.digest(names), "operation": lambda *a, **k: None, "gate": lambda stage: None}
        return final, common, base_path, warm_path, names, bindings

    def test_streamed_delta_uses_common_warm_reference_and_separate_sequential_start(self):
        for sequential in (False, True):
            with self.subTest(sequential=sequential), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                final, common, base, warm, names, bindings = self.donor_case(directory, sequential)
                out = directory / "donor.safetensors"
                record = u.extract_donor(final, base, warm, names, out, bindings)
                saved = self.arm_build.read_delta(out, record["data_section_sha256"], record["state_sha256"])
                self.assertTrue(self.torch.equal(saved["weight"], final.weight.detach() - common.weight.detach()))
                self.assertEqual(record["reference"], "EXACT_REBUILT_COMMON_WARM_ANCHOR")
                self.assertEqual(record["deployed_arm_sha256"], self.runtime.anchor_digest(final))
                self.assertTrue(record["roundtrip"]["proved_from_saved_delta"])
                with self.assertRaisesRegex(ValueError, "create-only"):
                    u.extract_donor(final, base, warm, names, out, bindings)

    def test_reference_and_frozen_binding_failures_publish_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            final, common, base, warm, names, bindings = self.donor_case(directory)
            for field in ("common_anchor_sha256", "own_start_sha256", "warm_delta_state_sha256", "frozen_state_sha256"):
                bad = dict(bindings, **{field: "0" * 64})
                out = directory / (field + ".safetensors")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    u.extract_donor(final, base, warm, names, out, bad)
                self.assertFalse(out.exists())

    def test_saved_roundtrip_reports_master_rounding_instead_of_requiring_equality(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            final, common, base, warm, names, bindings = self.donor_case(directory)
            with self.torch.no_grad():
                final.weight[0] = -(2.0 ** -24)
            bindings["final_master_sha256"] = self.runtime.anchor_digest(final)
            original = self.runtime.anchor_digest(final)
            out = directory / "rounded.safetensors"
            record = u.extract_donor(final, base, warm, names, out, bindings)
            self.assertFalse(record["roundtrip"]["master_equal"])
            self.assertGreater(record["roundtrip"]["residual_max_abs"], 0)
            self.assertNotEqual(record["deployed_arm_sha256"], original)
            self.assertEqual(self.runtime.anchor_digest(final), original)
            self.assertTrue(record["roundtrip"]["proved_from_saved_delta"])


if __name__ == "__main__":
    unittest.main()
