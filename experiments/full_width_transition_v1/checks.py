"""Elementwise mechanical checks and bounded objectives, not capability scores."""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F


def compare(actual, expected, *, atol=1e-4, rtol=1e-4):
    if actual.shape != expected.shape:
        return {"pass": False, "shape_mismatch": True}
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    delta = (actual - expected).abs()
    violations = int((delta > atol + rtol * expected.abs()).sum()) if finite else actual.numel()
    return {"pass": finite and violations == 0, "finite": finite,
            "violations": violations, "elements": actual.numel(),
            "max_abs": float(delta.max()) if finite else None}


@torch.inference_mode()
def active_checks(model, ids):
    """Exercise nonzero branches: masks, prefix causality, and multiple cache orders."""
    model.eval()
    n = ids.shape[1]
    if ids.shape[0] != 1 or n < 4:
        raise ValueError("active check needs a batch-one sequence of at least four tokens")
    pos = torch.arange(n).unsqueeze(0)
    mask = torch.ones_like(ids)
    mask[:, :2] = 0
    segments = torch.zeros_like(ids)
    segments[:, n // 2:] = 1
    reset = torch.cat([torch.arange(n // 2), torch.arange(n - n // 2)]).unsqueeze(0)
    cases = {"plain": {}, "left_padding": {"attention_mask": mask},
             "repeated_positions": {"position_ids": pos // 2},
             "packed_segments": {"position_ids": reset, "segment_ids": segments},
             "large_positions": {"position_ids": pos + model.config.max_position_embeddings - n}}
    rows = []
    tail = compare(model(ids, last_position_only=True).logits, model(ids).logits[:, -1:])
    for label, kwargs in cases.items():
        full = model(ids, **kwargs).logits
        for size in (1, 3):
            cache, parts = None, []
            for start in range(0, n, size):
                stop = min(n, start + size)
                chunk = {k: v[:, start:stop] for k, v in kwargs.items()}
                result = model(ids[:, start:stop], past=cache, use_cache=True, **chunk)
                cache = result.cache
                parts.append(result.logits)
            rows.append({"case": label, "chunk_size": size,
                         **compare(torch.cat(parts, dim=1), full)})
    mutated = ids.clone()
    mutated[:, -2:] = (mutated[:, -2:] + 1) % model.config.vocab_size
    future = compare(model(mutated).logits[:, :-2], model(ids).logits[:, :-2], atol=0, rtol=0)
    mutated = ids.clone()
    mutated[:, :n // 2] = (mutated[:, :n // 2] + 1) % model.config.vocab_size
    packed = cases["packed_segments"]
    isolation = compare(model(mutated, **packed).logits[:, n // 2:],
                        model(ids, **packed).logits[:, n // 2:], atol=0, rtol=0)
    return {"cache_pass": tail["pass"] and all(row["pass"] for row in rows),
            "causal_pass": future["pass"] and isolation["pass"],
            "cache_grid": rows, "tail_projection": tail, "future_prefix": future, "segment_isolation": isolation}


@torch.inference_mode()
def greedy(model, ids, count=8):
    model.eval()
    cache, output = None, []
    for _ in range(count):
        step = model(ids, past=cache, use_cache=True, last_position_only=True)
        token = step.logits[:, -1].argmax(-1, keepdim=True)
        output.append(int(token.item()))
        ids, cache = token, step.cache
    return output


def objective(student, teacher, ids, *, temperature=2.0, kl_weight=0.5, ce_weight=1.0):
    if not all(math.isfinite(x) for x in (temperature, kl_weight, ce_weight)) or temperature <= 0 or kl_weight < 0 or ce_weight <= 0:
        raise ValueError("invalid objective weights")
    if (student.ndim != 3 or student.shape != teacher.shape or ids.shape != student.shape[:2]
            or ids.shape[1] < 2 or ids.dtype != torch.long):
        raise ValueError("objective tensor shapes or token dtype do not match")
    s, t = student[:, :-1].float(), teacher[:, :-1].detach().float()
    labels = ids[:, 1:]
    ce = F.cross_entropy(s.reshape(-1, s.shape[-1]), labels.reshape(-1))
    log_t, log_s = F.log_softmax(t / temperature, -1), F.log_softmax(s / temperature, -1)
    kl = (log_t.exp() * (log_t - log_s)).sum(-1).mean() * temperature ** 2
    return ce_weight * ce + kl_weight * kl, {"ce": float(ce.detach()), "kl": float(kl.detach())}
