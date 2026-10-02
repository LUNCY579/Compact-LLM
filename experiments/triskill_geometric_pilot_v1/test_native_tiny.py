"""CPU tests of the native worker path with a tiny random Qwen3 (no checkpoint, no GPU).

They drive the same module-level functions the native run uses: the block-14 hook,
fixed-slot decoding (padding, both EOS ids, compute-only suffix), hook liveness,
teacher-forced readout and in-memory arm loading from safetensors files.
"""
import os
import tempfile
import types
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import geometry
import worker

EOS = (510, 511)
PAD = EOS[0]


def tiny_model(seed=0):
    config = Qwen3Config(vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=15,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512,
                         tie_word_embeddings=True)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(config).eval()


def operator(seed=1, hidden=32, rank=4, eta=geometry.ETA):
    rng = np.random.default_rng(seed)
    u = np.linalg.qr(rng.standard_normal((hidden, rank)))[0]
    v = np.linalg.qr(rng.standard_normal((hidden, rank)))[0]
    basis, rotation, _ = geometry.construct(u, v, eta)
    return basis, rotation, geometry.haar_frame(basis.shape, seed + 1)


class Unlimited(worker.Budget):
    pass


class NativeTinyTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.basis, self.rotation, self.random_basis = operator()
        self.b = torch.as_tensor(self.basis, dtype=torch.float32)
        self.state = worker.HookState(torch, self.b)
        self.handle = self.model.model.layers[geometry.LAYER].register_forward_hook(worker.make_hook(self.state))
        self.addCleanup(self.handle.remove)
        self.ids = [int(x) for x in np.random.default_rng(3).integers(0, 500, 200)]
        self.budget = Unlimited()

    def set_change(self, rotation=None, basis=None):
        basis = self.b if basis is None else torch.as_tensor(basis, dtype=torch.float32)
        change = (torch.zeros((len(self.rotation),) * 2) if rotation is None
                  else torch.as_tensor(rotation - np.eye(len(rotation)), dtype=torch.float32))
        self.state.set(basis, change)

    def decode(self, **kwargs):
        return worker.decode(self.model, torch, "cpu", self.ids, pad_id=PAD, eos_ids=EOS, state=self.state,
                             budget=self.budget, **kwargs)

    def test_identity_hook_is_bitwise_noop_and_rotation_changes_logits(self):
        self.set_change()
        hooked, _ = worker.prefill_logits(self.model, torch, "cpu", self.ids, pad_id=PAD, state=self.state, budget=self.budget)
        self.assertEqual(self.state.displacement, 0.0)
        self.handle.remove()
        state = worker.HookState(torch, self.b)
        input_ids, mask, positions = worker._padded(torch, "cpu", self.ids, PAD)
        with torch.inference_mode():
            plain = self.model(input_ids=input_ids, attention_mask=mask, position_ids=positions, use_cache=False).logits[0]
        self.assertTrue(torch.equal(hooked, plain))
        self.handle = self.model.model.layers[geometry.LAYER].register_forward_hook(worker.make_hook(self.state))
        self.set_change(self.rotation)
        rotated, _ = worker.prefill_logits(self.model, torch, "cpu", self.ids, pad_id=PAD, state=self.state, budget=self.budget)
        self.assertGreater(self.state.displacement, 0.0)
        self.assertGreater(float((rotated - plain).abs().max()), 0.0)

    def test_padding_parity_and_fixed_slot_accounting(self):
        self.set_change()
        padded = self.decode(slots=4, padded=True)
        unpadded = self.decode(slots=4, padded=False)
        self.assertEqual(padded["raw_tokens"], unpadded["raw_tokens"])
        before = self.budget.forwards
        result = self.decode()
        self.assertEqual(self.budget.forwards - before, worker.OUTPUT_SLOTS)
        self.assertEqual(len(result["raw_tokens"]), worker.OUTPUT_SLOTS)
        self.assertEqual(len(result["margins"]), worker.OUTPUT_SLOTS)
        with self.assertRaisesRegex(RuntimeError, "STOP_INPUT_TOKEN_BUDGET"):
            worker._padded(torch, "cpu", [1] * (worker.INPUT_SLOTS + 1), PAD)

    def test_second_eos_id_ends_text_and_suffix_is_compute_only(self):
        script = [5, 6, EOS[1], 7, 8] + [9] * 27
        fed = []

        class Scripted:
            def __call__(self, input_ids, attention_mask, position_ids, past_key_values=None, use_cache=True):
                fed.append(int(input_ids[0, -1]))
                step = len(fed) - 1
                logits = torch.full((1, input_ids.shape[1], 512), -10.0)
                logits[0, -1, script[step]] = 10.0
                logits[0, -1, 0] = 9.0
                return types.SimpleNamespace(logits=logits, past_key_values=object())
        result = worker.decode(Scripted(), torch, "cpu", self.ids, pad_id=PAD, eos_ids=EOS, state=self.state,
                               budget=self.budget)
        self.assertEqual(result["tokens"], [5, 6])
        self.assertEqual(result["scored_tokens"], [5, 6, EOS[1]])
        self.assertTrue(result["ended"])
        self.assertEqual(fed[3:], [EOS[0]] * (worker.OUTPUT_SLOTS - 3))
        self.assertEqual(result["margins"][0], 1.0)

    def test_hook_liveness_passes_live_and_fails_dead_hook(self):
        dev = [self.ids, self.ids[:150]]
        live = worker.hook_liveness(self.model, torch, "cpu", dev, pad_id=PAD, state=self.state, budget=self.budget,
                                    basis=self.basis, random_basis=self.random_basis, rotation=self.rotation)
        self.assertTrue(live["ok"], live)
        for record in live["rows"]:
            self.assertEqual(record["identity_displacement"], 0.0)
            for arm in worker.LIVENESS_ARMS:
                self.assertTrue(record["arms"][arm]["ok"])
        self.handle.remove()
        dead = worker.hook_liveness(self.model, torch, "cpu", dev, pad_id=PAD, state=self.state, budget=self.budget,
                                    basis=self.basis, random_basis=self.random_basis, rotation=self.rotation)
        self.assertFalse(dead["ok"])

    def _liveness_with_bug(self, bug):
        original = worker.HookState.set
        worker.HookState.set = lambda state, basis, change: original(state, *bug(basis, change))
        try:
            return worker.hook_liveness(self.model, torch, "cpu", [self.ids], pad_id=PAD, state=self.state,
                                        budget=self.budget, basis=self.basis, random_basis=self.random_basis,
                                        rotation=self.rotation)
        finally:
            worker.HookState.set = original

    def test_liveness_detects_direction_bugs_with_equal_energy(self):
        # Both bugs displace exactly the registered energy (the operator spectrum is
        # symmetric), so only the element-wise vector check can catch them.
        for name, bug in (("column-reversed basis", lambda basis, change: (basis.flip(1), change)),
                          ("transposed change", lambda basis, change: (basis, change.T))):
            result = self._liveness_with_bug(bug)
            self.assertFalse(result["ok"], name)
            arm = result["rows"][0]["arms"]["ROTATION"]
            self.assertAlmostEqual(arm["measured"], arm["analytic"], delta=1e-6 * arm["analytic"])
            self.assertGreater(arm["max_relative_vector_error"], worker.LIVENESS_RTOL)
        scaled = self._liveness_with_bug(lambda basis, change: (basis, 1.01 * change))
        self.assertFalse(scaled["ok"])

    def test_teacher_forced_readout_reproduces_greedy_path(self):
        self.set_change()
        decoded = self.decode()
        target = decoded["scored_tokens"]
        reference = worker.teacher_forced_logits(self.model, torch, "cpu", self.ids, target, pad_id=PAD,
                                                 state=self.state, budget=self.budget)
        self.assertEqual(reference.argmax(-1).tolist(), target)
        same = worker.readout_metrics(torch, reference, reference, target)
        self.assertEqual(same["argmax_agrees_with_TH_token"], len(target))
        self.assertEqual(same["max_abs_logit_change"], 0.0)
        self.assertAlmostEqual(same["mean_kl_TH_to_arm"], 0.0, places=12)
        self.set_change(self.rotation)
        rotated = worker.teacher_forced_logits(self.model, torch, "cpu", self.ids, target, pad_id=PAD,
                                               state=self.state, budget=self.budget)
        moved = worker.readout_metrics(torch, reference, rotated, target)
        self.assertGreater(moved["max_abs_logit_change"], 0.0)
        self.assertGreater(moved["mean_kl_TH_to_arm"], 0.0)
        self.assertIsNone(worker.teacher_forced_logits(self.model, torch, "cpu", self.ids, [], pad_id=PAD,
                                                       state=self.state, budget=self.budget))


class ArmLoadingTests(unittest.TestCase):
    def test_arms_load_exactly_from_safetensors_and_restore(self):
        from safetensors.torch import save_file
        model = tiny_model(0)
        names = [name for name, _ in model.named_parameters()]
        rng = np.random.default_rng(5)
        tensors = {name: torch.as_tensor(rng.standard_normal(tuple(p.shape)), dtype=torch.float32)
                   for name, p in model.named_parameters()}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            base = {name: value.to(torch.bfloat16) for name, value in tensors.items()}
            t = {name: base[name].float() + 0.01 for name in names}
            th = {name: t[name] + 0.003 for name in names}
            for label, payload in (("BASE", base), ("T", t), ("TH", th)):
                save_file({k: v.contiguous() for k, v in payload.items()}, str(folder / f"{label}.safetensors"))
            fake = types.SimpleNamespace(TensorFile=geometry.TensorFile,
                                         SOURCES={label: (folder / f"{label}.safetensors", None) for label in ("BASE", "T", "TH")})
            needed = []
            worker.load_arm_weights(model, "H", torch, fake, budget=worker.Budget(), reserve_bytes=needed.append)
            self.assertEqual(len(needed), len(names))
            for name, param in model.named_parameters():
                expected = (base[name].double() + th[name].double() - t[name].double()).float()
                self.assertTrue(torch.equal(param.data, expected), name)
            worker.load_arm_weights(model, "BASE", torch, fake)
            for name, param in model.named_parameters():
                self.assertTrue(torch.equal(param.data, base[name].float()), name)
            worker.load_arm_weights(model, "ROTATION", torch, fake)  # TH-hosted
            for name, param in model.named_parameters():
                self.assertTrue(torch.equal(param.data, th[name]), name)
            def refuse(_bytes):
                raise RuntimeError("STOP_HOST_ACTIVE_RESERVE")
            with self.assertRaisesRegex(RuntimeError, "STOP_HOST_ACTIVE_RESERVE"):
                worker.load_arm_weights(model, "T", torch, fake, reserve_bytes=refuse)


if __name__ == "__main__":
    unittest.main()
