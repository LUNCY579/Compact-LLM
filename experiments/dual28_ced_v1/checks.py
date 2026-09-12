"""Active-path checks and bounded engineering, never a capability scorer."""
from __future__ import annotations
from contextlib import contextmanager
import gc
import hashlib
import random
import torch
from torch.nn import functional as F
from .guards import canonical, protocol, require
from .model import DualStack
from experiments.full_width_transition_v1.source import state_digest


def close(actual, expected, label, *, exact=False):
    require(actual.shape == expected.shape and bool(torch.isfinite(actual).all())
            and bool(torch.isfinite(expected).all()), f"{label}: shape or nonfinite output")
    p = protocol()["parity"]
    ok = torch.equal(actual, expected) if exact else torch.allclose(actual, expected, atol=p["atol"], rtol=p["rtol"])
    require(ok, f"{label}: elementwise parity failure")
    return float((actual.detach() - expected.detach()).abs().max())


@contextmanager
def activate(model, source):
    require(model.mode in {"H", "P"}, "activation fixture requires hybrid")
    saved = [r.o_proj.weight.detach().clone() for r in model.reads]
    try:
        with torch.no_grad():
            for r, donor in zip(model.reads, source.encoder):
                r.o_proj.weight.copy_(donor.o_proj.weight * protocol()["parity"]["active_output_scale"])
        yield
    finally:
        with torch.no_grad():
            for r, value in zip(model.reads, saved):
                r.o_proj.weight.copy_(value)


def grid(model):
    """Full/token/chunk equality, causality, padding, segments and long RoPE positions."""
    model.eval()
    ids = (torch.arange(1, 9, dtype=torch.long) % model.config.vocab_size).unsqueeze(0)
    cases = {
        "ordinary": {},
        "repeated_positions": {"position_ids": torch.full_like(ids, 5)},
        "large_positions": {"position_ids": torch.arange(model.config.max_position_embeddings-8,
                                                          model.config.max_position_embeddings)[None]},
        "left_padding": {"attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]])},
        "all_padding": {"attention_mask": torch.zeros_like(ids)},
        "packed": {"position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
                   "segment_ids": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])},
    }
    errors = {}
    with torch.inference_mode():
        for name, kwargs in cases.items():
            full = model(ids, **kwargs).logits
            require(bool(torch.isfinite(full).all()), f"{name}: nonfinite logits")
            errors[name] = 0.0
            for chunks in ([1]*8, [3, 2, 3]):
                past, outputs, start = None, [], 0
                for length in chunks:
                    end = start + length
                    part = {k: v[:, start:end] for k, v in kwargs.items()}
                    out = model(ids[:, start:end], past=past, use_cache=True, **part)
                    outputs.append(out.logits)
                    past, start = out.cache, end
                errors[name] = max(errors[name], close(torch.cat(outputs, 1), full, name))
            if name != "all_padding":
                close(model(ids, last_position_only=True, **kwargs).logits, full[:, -1:], name+"_tail")
            changed = ids.clone()
            changed[:, -2:] = (changed[:, -2:] + 3) % model.config.vocab_size
            close(model(changed, **kwargs).logits[:, :-2], full[:, :-2], name+"_future", exact=True)
            if name == "left_padding":
                changed = ids.clone(); changed[:, :2] = (changed[:, :2]+5) % model.config.vocab_size
                close(model(changed, **kwargs).logits[:, 2:], full[:, 2:], "padding_isolation", exact=True)
            if name == "packed":
                changed = ids.clone(); changed[:, :4] = (changed[:, :4]+7) % model.config.vocab_size
                close(model(changed, **kwargs).logits[:, 4:], full[:, 4:], "segment_isolation", exact=True)
    return errors


def active_path(model, source):
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]) % model.config.vocab_size
    with activate(model, source), torch.inference_mode():
        enabled = model(ids).logits
        model.read_enabled = False
        try:
            disabled = model(ids).logits
        finally:
            model.read_enabled = True
        require(not torch.equal(enabled, disabled), "active added path has no measured effect")
        result = grid(model)
    return result


def train_pair(source, rows):
    """Exactly 25 H and 25 P steps. No model-quality score and no downstream permission."""
    p = protocol()["engineering"]
    require(rows and all(isinstance(r, list) and 2 <= len(r) <= p["max_tokens"]
                        and all(type(t) is int and 0 <= t < source.config.vocab_size for t in r) for r in rows),
            "invalid engineering token rows")
    rng = random.Random(p["seed"])
    order = [rng.randrange(len(rows)) for _ in range(p["steps"])]
    data_digest = hashlib.sha256(canonical(rows)).hexdigest()
    before_source = state_digest(source)
    results = {}
    for mode in ("H", "P"):
        torch.manual_seed(p["seed"])
        model = DualStack.from_source(source, mode)
        trainable = model.freeze_backbones()
        frozen_before = state_digest(model, backbone_only=True)
        model.gradient_checkpointing = p["gradient_checkpointing"]
        model.train()
        opt = torch.optim.AdamW(trainable, lr=p["learning_rate"], weight_decay=p["weight_decay"])
        seen, losses = set(), []
        required = {n for n, value in model.named_parameters() if value.requires_grad}
        for step, index in enumerate(order):
            ids = torch.tensor([rows[index]], dtype=torch.long)
            with torch.no_grad():
                teacher = source(ids).logits[:, :-1].float()
            logits = model(ids).logits[:, :-1].float()
            target = ids[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            temp = p["temperature"]
            kl = F.kl_div(F.log_softmax(logits/temp, -1), F.softmax(teacher/temp, -1), reduction="none").sum(-1).mean()*temp**2
            loss = ce + p["kl_weight"]*kl
            require(bool(torch.isfinite(loss)), "nonfinite engineering loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for name, value in model.named_parameters():
                if value.grad is not None:
                    require(value.requires_grad and bool(torch.isfinite(value.grad).all()), "invalid/frozen gradient")
                    if bool((value.grad != 0).any()):
                        seen.add(name)
            norm = torch.nn.utils.clip_grad_norm_(trainable, p["clip_norm"], error_if_nonfinite=True)
            opt.step()
            require(all(bool(torch.isfinite(v).all()) for v in trainable), "nonfinite trained tensor")
            losses.append({"step": step+1, "row": index, "ce": float(ce.detach()), "kl": float(kl.detach()),
                           "loss": float(loss.detach()), "gradient_norm": float(norm)})
        require(seen == required, "missing upstream gradient coverage: " + repr(sorted(required-seen)))
        require(state_digest(model, backbone_only=True) == frozen_before, "encoder/decoder backbone changed")
        require(all(bool((r.o_proj.weight != 0).any()) for r in model.reads), "dormant output projection")
        model.eval()
        mechanics = grid(model)
        results[mode] = {"order": order, "data_sha256": data_digest, "steps": len(losses), "losses": losses,
                         "trainable_parameters": sum(v.numel() for v in trainable),
                         "nonzero_gradient_tensors": sorted(seen), "frozen_backbones": True,
                         "post_step_grid": mechanics, "state_sha256": state_digest(model)}
        # Engineering checkpoints are deliberately not returned as recovered/released models.
        del model, opt, trainable, logits, teacher, ce, kl, loss
        gc.collect()
    require(state_digest(source) == before_source, "source changed")
    require(results["H"]["order"] == results["P"]["order"]
            and results["H"]["trainable_parameters"] == results["P"]["trainable_parameters"], "unmatched engineering")
    return results
