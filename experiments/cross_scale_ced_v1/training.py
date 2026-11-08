"""Fixed paired engineering optimization, not recovery or a quality campaign."""
from __future__ import annotations
import gc
import json
import math
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from .checkpoint import save_interface, load_interface
from .guards import protocol, write_json_new, sha256_file, permissions
from .model import CrossScale
from .source import state_digest


def causal_loss(logits, ids, *, attention_mask=None, segment_ids=None, teacher_logits=None, teacher_kl=0.1):
    if not math.isfinite(teacher_kl) or teacher_kl < 0:
        raise ValueError("teacher KL coefficient must be finite and nonnegative")
    if logits.ndim != 3 or ids.shape != logits.shape[:2] or ids.dtype != torch.long or ids.shape[1] < 2:
        raise ValueError("aligned logits/IDs with at least two tokens required")
    if (ids < 0).any() or (ids >= logits.shape[-1]).any():
        raise ValueError("labels outside vocabulary")
    if attention_mask is not None and not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("loss mask must be binary")
    valid = torch.ones_like(ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    seg = torch.zeros_like(ids) if segment_ids is None else segment_ids
    if (valid.shape != ids.shape or seg.shape != ids.shape or seg.dtype != torch.long
            or valid.device != ids.device or seg.device != ids.device or logits.device != ids.device):
        raise ValueError("loss masks must match IDs")
    pairs = valid[:, :-1] & valid[:, 1:] & (seg[:, :-1] == seg[:, 1:])
    if not pairs.any():
        raise ValueError("no valid next-token targets")
    z = logits[:, :-1][pairs].float()
    ce = F.cross_entropy(z, ids[:, 1:][pairs])
    kl = z.new_zeros(())
    if teacher_logits is not None:
        if teacher_logits.shape != logits.shape:
            raise ValueError("teacher shape mismatch")
        target = teacher_logits[:, :-1][pairs].detach().float().softmax(-1)
        kl = F.kl_div(z.log_softmax(-1), target, reduction="batchmean")
    return ce+teacher_kl*kl, {"ce": float(ce.detach()), "kl": float(kl.detach()), "targets": int(pairs.sum())}


def paired_smoke(encoder, decoder, rows: list[torch.Tensor], output: Path, binding: dict,
                 *, synthetic=False) -> dict:
    p = protocol()
    policy = p["smoke"]
    results = {}
    inventories = []
    for mode in ("H", "P"):
        # Both modes receive exactly the same writer initialization, rows and updates.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p["seed"])
            model = CrossScale.from_sources(encoder, decoder, mode)
        params = model.freeze_backbones()
        model.gradient_checkpointing = True
        inventories.append([(n, list(t.shape)) for n, t in model.named_parameters() if t.requires_grad])
        before = state_digest(model, backbone_only=True)
        original = state_digest(model)
        optimizer = torch.optim.AdamW(params, lr=policy["learning_rate"], weight_decay=policy["weight_decay"])
        coverage = {n: False for n, t in model.named_parameters() if t.requires_grad}
        records = []
        path = output/f"{mode}_steps.jsonl"
        with path.open("x", encoding="utf-8") as log:
            model.train()
            for step in range(policy["steps_per_arm"]):
                ids = rows[step % len(rows)]
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher = decoder(ids).logits
                loss, parts = causal_loss(model(ids).logits, ids, teacher_logits=teacher,
                                          teacher_kl=policy["teacher_kl"])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite engineering loss")
                loss.backward()
                for n, t in model.named_parameters():
                    if t.requires_grad:
                        if t.grad is None or not torch.isfinite(t.grad).all():
                            raise ValueError(f"missing or nonfinite gradient: {n}")
                        coverage[n] |= bool(t.grad.abs().sum() > 0)
                    elif t.grad is not None:
                        raise ValueError("gradient attached to frozen backbone")
                norm = torch.nn.utils.clip_grad_norm_(params, policy["clip_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(t).all() for t in params):
                    raise ValueError("nonfinite trained interface")
                rec = dict(step=step+1, row=step % len(rows), loss=float(loss.detach()),
                           grad_norm=float(norm), **parts)
                records.append(rec)
                log.write(json.dumps(rec, allow_nan=False)+"\n"); log.flush(); os.fsync(log.fileno())
        if not all(coverage.values()) or state_digest(model, backbone_only=True) != before:
            raise ValueError("incomplete interface gradient coverage or frozen backbone changed")
        if state_digest(model) == original:
            raise ValueError("engineering optimizer did not change the interface")
        if records[-1]["loss"] > 2*records[0]["loss"]:
            raise ValueError("engineering loss exploded; not a model-quality comparison")
        model.eval(); model.zero_grad(set_to_none=True)
        with torch.inference_mode():
            expected = model(rows[0]).logits.clone()
        checkpoint = output/f"{mode}_interface"
        save_interface(model, checkpoint, binding)
        with torch.no_grad():
            model.reads[0].o_proj.weight.zero_()
        load_interface(model, checkpoint, binding)
        with torch.inference_mode():
            torch.testing.assert_close(expected, model(rows[0]).logits, atol=0, rtol=0)
        results[mode] = dict(passed=True, steps=len(records), gradient_tensors=len(coverage),
                             frozen_backbone_sha256=before, initial_loss=records[0]["loss"],
                             final_loss=records[-1]["loss"], log_sha256=sha256_file(path),
                             checkpoint_manifest_sha256=sha256_file(checkpoint/"manifest.json"))
        # Release the completed autograd graph BEFORE allocating the next 5.38B arm.
        del loss, teacher, expected, norm, t, model, optimizer, params
        gc.collect()
    if inventories[0] != inventories[1]:
        raise ValueError("H/P trainable tensor inventories differ")
    result = dict(schema=1, experiment="XSC-001", phase="tiny-smoke" if synthetic else "P2",
                  status="SYNTHETIC_PASS_ONLY" if synthetic else "PASS", results=results,
                  paired_inventory_exact=True, permissions=permissions(), licenses="nothing", binding=binding)
    write_json_new(output/"smoke.json", result)
    return result
