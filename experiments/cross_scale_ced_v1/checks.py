"""Reusable mechanical gates. Active reads are tested separately from zero identity."""
from __future__ import annotations
import tempfile
from pathlib import Path
import torch
from .checkpoint import save_interface, load_interface, interface_state
from .guards import protocol
from .source import state_digest


def assert_close(a, b, *, native=False) -> float:
    tol = protocol()["tolerances"]
    prefix = "native_" if native else ""
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("nonfinite logits")
    torch.testing.assert_close(a, b, atol=tol[prefix+"atol"], rtol=tol[prefix+"rtol"])
    return float((a-b).abs().max())


def assert_exact(a, b) -> float:
    if not torch.isfinite(a).all() or not torch.equal(a, b):
        raise AssertionError("exact structural invariant failed")
    return 0.0


def cache_tensors(c):
    return [x for pair in (*c.encoder, *c.decoder, c.memory) for x in pair]+[c.positions, c.valid, c.segments]


def native_parity(reference, native) -> dict:
    ids = torch.tensor([[3, 11, 8, 7, 19, 2, 5, 13]], dtype=torch.long)
    errors = []
    with torch.inference_mode():
        for offset in (0, 257, 32760):
            pos = torch.arange(ids.shape[1])[None]+offset
            errors.append(assert_close(reference(ids, position_ids=pos).logits,
                                       native(input_ids=ids, position_ids=pos, use_cache=False).logits,
                                       native=True))
        cache, parts = None, []
        for start, stop in ((0, 3), (3, 5), (5, 8)):
            out = native(input_ids=ids[:, start:stop], past_key_values=cache, use_cache=True)
            parts.append(out.logits); cache = out.past_key_values
        errors.append(assert_close(reference(ids).logits, torch.cat(parts, 1), native=True))
    return dict(passed=True, max_abs=max(errors), positions=[0, 257, 32760], cached_native_checked=True)


def qualify_model(model, *, checkpoint_parent: Path | None = None, binding=None) -> dict:
    """Engineering only. Restores the interface, mode and flags before returning."""
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("official shared qualification arithmetic is float32")
    model.freeze_backbones()
    before = state_digest(model, backbone_only=True)
    saved = {k: v.detach().clone() for k, v in interface_state(model).items()}
    was_training, original_read = model.training, model.read_enabled
    result = {}
    ids = torch.tensor([[3, 11, 8, 7, 19, 2, 5, 13]], dtype=torch.long)
    try:
        model.eval()
        with torch.inference_mode():
            result["zero_identity_max_abs"] = assert_exact(model(ids).logits, model.decoder(ids).logits)
            # Nonzero output projections expose Q/K/V, masks, and caches.
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(713)
            for r in model.reads:
                r.o_proj.weight.normal_(0, 0.01)
        with torch.inference_mode():
            full = model(ids, diagnostics=True).logits
            changed_ids = ids.clone(); changed_ids[:, 3] = 17
            changed = model(ids, encoder_input_ids=changed_ids).logits
            result["encoder_intervention_max_abs"] = float((full-changed).abs().max())
            if result["encoder_intervention_max_abs"] <= 1e-7:
                raise AssertionError("active memory does not affect outputs")
            future = ids.clone(); future[:, 5:] = 1
            result["future_leak_max_abs"] = assert_exact(full[:, :5], model(future).logits[:, :5])
            samepos = torch.zeros_like(ids)
            result["repeated_position_causality"] = assert_exact(
                model(ids, position_ids=samepos).logits[:, :5],
                model(future, position_ids=samepos).logits[:, :5])
            segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
            changed_earlier = ids.clone(); changed_earlier[:, :4] = 2
            result["segment_isolation"] = assert_exact(
                model(ids, segment_ids=segments).logits[:, 4:],
                model(changed_earlier, segment_ids=segments).logits[:, 4:])
            pad = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]])
            changed_pad = ids.clone(); changed_pad[:, :2] = 4
            result["padding_isolation"] = assert_exact(model(ids, attention_mask=pad).logits[:, 2:],
                                                       model(changed_pad, attention_mask=pad).logits[:, 2:])
            masked = model(ids, attention_mask=torch.zeros_like(ids)).logits
            if not torch.isfinite(masked).all():
                raise AssertionError("all-masked rows are nonfinite")
            result["all_masked_finite"] = True
            errors = []
            for chunks in ((1,)*8, (3, 2, 3), (1, 7)):
                offset, cache, pieces = 0, None, []
                for n in chunks:
                    old = [] if cache is None else [t.clone() for t in cache_tensors(cache)]
                    out = model(ids[:, offset:offset+n], past=cache, use_cache=True)
                    if cache is not None:
                        for a, b in zip(old, cache_tensors(cache), strict=True):
                            if not torch.equal(a, b):
                                raise AssertionError("input cache was mutated")
                    pieces.append(out.logits); cache = out.cache; offset += n
                errors.append(assert_close(full, torch.cat(pieces, 1)))
            result["cache_schedule_max_abs"] = max(errors)
            result["cache_immutability"] = True
            result["tail_logits_max_abs"] = assert_close(full[:, -1:], model(ids, last_position_only=True).logits)
            model.read_enabled = False
            result["disabled_identity_max_abs"] = assert_exact(model(ids).logits, model.decoder(ids).logits)
            model.read_enabled = True
        # A gradient through a frozen decoder must reach every new interface tensor.
        model.train(); model.zero_grad(set_to_none=True)
        logits = model(ids).logits
        torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                          ids[:, 1:].reshape(-1)).backward()
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        for n, p in trainable:
            if p.grad is None or not torch.isfinite(p.grad).all() or not bool(p.grad.abs().sum() > 0):
                raise AssertionError(f"inactive/nonfinite interface gradient: {n}")
        result["gradient_tensors"] = len(trainable)
        result["gradient_tensor_names"] = [n for n, _ in trainable]
        model.zero_grad(set_to_none=True); model.eval()
        # Save/reload active interface rather than relying on zero-output identity.
        if checkpoint_parent is None:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"interface"
                save_interface(model, path, binding or {"scope": "synthetic"})
                with torch.no_grad():
                    next(iter(model.reads)).o_proj.weight.zero_()
                load_interface(model, path, binding or {"scope": "synthetic"})
        else:
            path = checkpoint_parent/f"{model.config.mode}_active_interface"
            save_interface(model, path, binding)
            with torch.no_grad():
                next(iter(model.reads)).o_proj.weight.zero_()
            load_interface(model, path, binding)
        with torch.inference_mode():
            result["checkpoint_max_abs"] = assert_exact(full, model(ids).logits)
        if state_digest(model, backbone_only=True) != before:
            raise AssertionError("frozen donor changed")
        result["frozen_backbones_unchanged"] = True
        result["passed"] = True
        return result
    finally:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in saved:
                    p.copy_(saved[name])
        model.zero_grad(set_to_none=True)
        model.read_enabled = original_read
        model.train(was_training)
