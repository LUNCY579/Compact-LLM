from dataclasses import replace
import unittest
import torch
from torch import nn

from ..alignment import AffineFit
from ..fixtures import decoder
from ..modules import (Bridge, FFNGraft, GraftedDecoder, StitchedDecoder, fingerprint,
                       interface_parameters, masked_next_token_loss)
from ..native_qwen import from_loaded_stock_qwen3


def bridge(a, b, *, delta=False, bias_value=0.1):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(55)
        weight = torch.randn(b, a, dtype=torch.float64) / a**0.5
    return Bridge.from_fit(AffineFit(weight, torch.full((b,), bias_value, dtype=torch.float64),
                                     20, 1e-3, "centered_affine_ridge_sum_v1"), delta=delta, dtype=torch.float64)


def graft(donor, *, alpha=0.05, train_alpha=False):
    return FFNGraft(donor.blocks[2].norm2, donor.blocks[2].mlp,
                    bridge(8, 12), bridge(12, 8, delta=True), donor_identity=donor.identity,
                    donor_block_index=2, alpha=alpha, train_alpha=train_alpha)


class ModuleTests(unittest.TestCase):
    def setUp(self):
        self.small = decoder()
        self.large = decoder(12, 6, 29, heads=6, kv_heads=2, head_dim=4)
        self.ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    def test_same_source_identity_stitch_exact(self):
        b = Bridge(8, 8, dtype=torch.float64)
        with torch.no_grad():
            b.weight.copy_(torch.eye(8))
            b.bias.zero_()
        model = StitchedDecoder(self.small, self.small, 2, 2, b)
        self.assertTrue(torch.equal(self.small(self.ids), model(self.ids)))

    def test_bidirectional_stitch_shapes(self):
        for model in (StitchedDecoder(self.small, self.large, 2, 3, bridge(8, 12)),
                      StitchedDecoder(self.large, self.small, 3, 2, bridge(12, 8))):
            result = model(self.ids)
            self.assertEqual(tuple(result.shape), (1, 4, 23))
            self.assertTrue(torch.isfinite(result).all())

    def test_stitch_does_not_retain_unused_backbone(self):
        model = StitchedDecoder(self.small, self.large, 2, 3, bridge(8, 12))
        retained = {id(p) for p in model.parameters()}
        unused = {id(p) for b in self.large.blocks[:3] for p in b.parameters()}
        self.assertFalse(retained & unused)
        self.assertNotIn(id(self.large.embedding.weight), retained)
        self.assertNotIn(id(self.small.head.weight), retained)

    def test_stitch_preserves_source_heads_and_tables(self):
        model = StitchedDecoder(self.small, self.large, 2, 3, bridge(8, 12))
        self.assertIs(model.embedding, self.small.embedding)
        self.assertIs(model.head, self.large.head)
        self.assertIs(model.norm, self.large.norm)

    def test_graft_bypass_exact(self):
        model = GraftedDecoder(self.small, 1, graft(self.large), bypass=True)
        self.assertTrue(torch.equal(self.small(self.ids), model(self.ids)))

    def test_fixed_zero_alpha_exact(self):
        model = GraftedDecoder(self.small, 1, graft(self.large, alpha=0))
        self.assertTrue(torch.equal(self.small(self.ids), model(self.ids)))

    def test_fixed_zero_does_not_call_donor(self):
        g = graft(self.large, alpha=0)
        def bad(*args):
            raise AssertionError("donor called on zero path")
        g.donor_ffn.forward = bad
        model = GraftedDecoder(self.small, 1, g)
        self.assertTrue(torch.equal(self.small(self.ids), model(self.ids)))

    def test_nonzero_graft_changes_logits(self):
        model = GraftedDecoder(self.small, 1, graft(self.large))
        self.assertFalse(torch.equal(self.small(self.ids), model(self.ids)))

    def test_graft_does_not_retain_donor_attention_or_embedding(self):
        model = GraftedDecoder(self.small, 1, graft(self.large))
        retained = {id(p) for p in model.parameters()}
        self.assertNotIn(id(self.large.embedding.weight), retained)
        self.assertNotIn(id(self.large.blocks[2].q.weight), retained)
        self.assertIn(id(self.large.blocks[2].mlp.down.weight), retained)

    def test_frozen_backbone_updates_unchanged(self):
        before = (fingerprint(self.small), fingerprint(self.large))
        g = graft(self.large, train_alpha=True)
        model = GraftedDecoder(self.small, 1, g)
        allowed = [*g.input_bridge.parameters(), *g.return_bridge.parameters(), g.alpha]
        params = interface_parameters(model, allowed)
        opt = torch.optim.SGD(params, lr=1e-3)
        old_bridge = g.input_bridge.weight.detach().clone()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            mask = torch.tensor([[False, True, True, True]])
            loss = masked_next_token_loss(model(self.ids), self.ids, mask)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
            opt.step()
        self.assertEqual(before, (fingerprint(self.small), fingerprint(self.large)))
        self.assertFalse(torch.equal(old_bridge, g.input_bridge.weight))

    def test_frozen_suffix_still_propagates_bridge_gradient(self):
        b = bridge(8, 12)
        model = StitchedDecoder(self.small, self.large, 2, 3, b)
        model(self.ids).square().mean().backward()
        self.assertIsNotNone(b.weight.grad)
        self.assertGreater(float(b.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in self.large.parameters()))

    def test_train_mode_keeps_sources_eval(self):
        g = graft(self.large)
        model = GraftedDecoder(self.small, 1, g).train()
        self.assertFalse(model.blocks[1].host.training)
        self.assertFalse(g.donor_ffn.training)
        self.assertFalse(g.donor_norm.training)
        self.assertTrue(g.input_bridge.training)

    def test_trainable_backbone_refused_by_allowlist(self):
        g = graft(self.large)
        model = GraftedDecoder(self.small, 1, g)
        self.small.embedding.weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            interface_parameters(model, [*g.input_bridge.parameters(), *g.return_bridge.parameters()])

    def test_duplicate_optimizer_parameter_refused(self):
        b = bridge(8, 12)
        model = StitchedDecoder(self.small, self.large, 2, 3, b)
        with self.assertRaises(ValueError):
            interface_parameters(model, [*b.parameters(), b.weight])

    def test_missing_optimizer_parameter_refused(self):
        b = bridge(8, 12)
        model = StitchedDecoder(self.small, self.large, 2, 3, b)
        with self.assertRaises(ValueError):
            interface_parameters(model, [b.weight])

    def test_delta_bridge_has_no_bias(self):
        b = bridge(12, 8, delta=True, bias_value=100)
        self.assertIsNone(b.bias)
        self.assertTrue(torch.equal(b(torch.zeros(2, 12, dtype=torch.float64)), torch.zeros(2, 8, dtype=torch.float64)))

    def test_bad_delta_bridge_refused(self):
        with self.assertRaises(ValueError):
            FFNGraft(self.large.blocks[2].norm2, self.large.blocks[2].mlp, bridge(8, 12), bridge(12, 8),
                     donor_identity=self.large.identity, donor_block_index=2)

    def test_trainable_zero_alpha_refused(self):
        with self.assertRaises(ValueError):
            graft(self.large, alpha=0, train_alpha=True)

    def test_native_dependency_absence_fails_closed(self):
        from unittest.mock import patch
        with patch.dict("sys.modules", {"transformers": None}):
            with self.assertRaisesRegex(RuntimeError, "requires Transformers"):
                from_loaded_stock_qwen3(self.small, replace(self.small.identity, fixture=False))

    def test_capture_raw_not_final_normalized(self):
        raw = self.small.capture(self.ids, 3, "block_output_raw")
        self.assertFalse(torch.equal(raw, self.small.norm(raw)))
        torch.testing.assert_close(self.small(self.ids), self.small.head(self.small.norm(raw)))

    def test_capture_postattention_convention(self):
        h, mask, pos = self.small.layout(self.ids)
        h = self.small.blocks[0](h, mask, pos)
        expected = self.small.blocks[1].attention_residual(h, mask, pos)
        self.assertTrue(torch.equal(expected, self.small.capture(self.ids, 1, "post_attention_raw")))

    def test_causality_for_all_candidates(self):
        changed = self.ids.clone()
        changed[0, -1] = 9
        for model in (self.small, GraftedDecoder(self.small, 1, graft(self.large)),
                      StitchedDecoder(self.small, self.large, 2, 3, bridge(8, 12)),
                      StitchedDecoder(self.large, self.small, 3, 2, bridge(12, 8))):
            torch.testing.assert_close(model(self.ids)[:, :-1], model(changed)[:, :-1], rtol=0, atol=0)

    def test_fixture_rng_isolated(self):
        before = torch.get_rng_state().clone()
        decoder()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_loss_shift_includes_eos(self):
        logits = torch.zeros(1, 4, 23, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[False, False, True, True]])
        loss = masked_next_token_loss(logits, self.ids, mask)
        loss.backward()
        self.assertEqual(float(logits.grad[:, 0].abs().sum()), 0)
        self.assertGreater(float(logits.grad[:, 1:3].abs().sum()), 0)
        self.assertEqual(float(logits.grad[:, 3].abs().sum()), 0)

    def test_no_native_authority_from_fixture_identity(self):
        self.assertTrue(self.small.identity.fixture)


def invalid_input_test(kind):
    def test(self):
        with self.assertRaises(ValueError):
            if kind == "cache": self.small(self.ids, use_cache=True)
            elif kind == "past": self.small(self.ids, past_key_values=())
            elif kind == "batch": self.small(self.ids.repeat(2, 1))
            elif kind == "dtype": self.small(self.ids.int())
            elif kind == "empty": self.small(self.ids[:, :0])
            elif kind == "too_long": self.small(torch.ones(1, 65, dtype=torch.long))
            elif kind == "bad_token": self.small(torch.tensor([[999]]))
            elif kind == "negative_token": self.small(torch.tensor([[-1]]))
    return test


for kind in ("cache", "past", "batch", "dtype", "empty", "too_long", "bad_token", "negative_token"):
    setattr(ModuleTests, "test_decoder_refuses_" + kind, invalid_input_test(kind))


def invalid_cut_test(a, b):
    def test(self):
        with self.assertRaises(ValueError):
            StitchedDecoder(self.small, self.large, a, b, bridge(8, 12))
    return test


for name, a, b in (("zero_prefix", 0, 3), ("whole_prefix", 4, 3), ("zero_suffix", 2, 0),
                   ("whole_suffix", 2, 6), ("bool_prefix", True, 3), ("negative", -1, 3)):
    setattr(ModuleTests, "test_stitch_refuses_" + name, invalid_cut_test(a, b))


def invalid_loss_test(kind):
    def test(self):
        logits = torch.zeros(1, 4, 23)
        mask = torch.tensor([[False, True, True, True]])
        if kind == "empty": mask[:] = False
        elif kind == "position_zero": mask[0, 0] = True
        elif kind == "dtype": mask = mask.long()
        elif kind == "nonfinite": logits[0, 0, 0] = float("nan")
        elif kind == "shape": logits = logits[:, :3]
        with self.assertRaises(ValueError):
            masked_next_token_loss(logits, self.ids, mask)
    return test


for kind in ("empty", "position_zero", "dtype", "nonfinite", "shape"):
    setattr(ModuleTests, "test_loss_refuses_" + kind, invalid_loss_test(kind))


class OrdinaryControlTests(unittest.TestCase):
    def test_control_trainable_count_matches_bridges(self):
        from ..modules import OrdinaryAdapter
        host, donor = decoder(), decoder(12, 6, 29)
        g = graft(donor)
        control = OrdinaryAdapter(host.blocks[1].norm2, 8, 12)
        self.assertEqual(sum(p.numel() for p in g.parameters() if p.requires_grad),
                         sum(p.numel() for p in control.parameters() if p.requires_grad))

    def test_control_starts_at_identity_and_learns(self):
        from ..modules import OrdinaryAdapter
        host = decoder()
        control = OrdinaryAdapter(host.blocks[1].norm2, 8, 12)
        model = GraftedDecoder(host, 1, control)
        ids = torch.tensor([[1, 2, 3, 4]])
        self.assertTrue(torch.equal(host(ids), model(ids)))
        model(ids).square().mean().backward()
        self.assertGreater(float(control.return_bridge.weight.grad.norm()), 0)
        self.assertEqual(float(control.input_bridge.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in host.parameters()))


class CompositionIdentityTests(unittest.TestCase):
    def test_candidate_digest_changes_after_interface_update(self):
        from ..modules import composition_descriptor
        a, b = decoder(), decoder(12, 6, 29)
        g = graft(b)
        model = GraftedDecoder(a, 1, g)
        old = composition_descriptor(model)
        with torch.no_grad():
            g.input_bridge.bias.add_(1)
        new = composition_descriptor(model)
        self.assertNotEqual(old["composition_sha256"], new["composition_sha256"])
        self.assertEqual(old["details"]["donor"], new["details"]["donor"])
        self.assertFalse(new["native_qualification"])

    def test_custom_a11_policy_not_silently_dense(self):
        from ..modules import Decoder
        a = decoder()
        with self.assertRaisesRegex(ValueError, "custom Supa"):
            Decoder(replace(a.identity, logit_policy="custom_a11_unqualified"),
                    a.embedding, list(a.blocks), a.norm, a.head)
