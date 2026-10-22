"""Bounded completion-only training, shared by fixture and optional native lanes."""
import math
import time

import torch
from torch.nn import functional as F

from .adapters import adapter_state
from .data import validate_cases
from .regularization import regularize


def bounded_prefix(rows, plan):
    """One pass, whole sequences, stable prefix; no cycling or hidden truncation."""
    validate_cases(rows, split="train")
    selected, tokens = [], 0
    for row in rows:
        size = len(row["prompt_ids"]) + len(row["target_ids"])
        if size > plan.train_context:
            raise ValueError("Training row exceeds context; truncation is forbidden")
        if tokens + size > plan.max_tokens or len(selected) == plan.max_steps * plan.accumulation:
            break
        selected.append(row)
        tokens += size
    if not selected:
        raise ValueError("Token/step budget cannot fit one complete sequence")
    return selected


def completion_loss(logits, labels):
    """Causal shift; mean over each sequence's supervised completion tokens."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.shape[1] < 2:
        raise ValueError("Invalid causal-LM shapes")
    active = labels[:, 1:].ne(-100)
    counts = active.sum(-1)
    if (counts == 0).any():
        raise ValueError("Every sequence must contain a supervised shifted target")
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels[:, 1:],
                             ignore_index=-100, reduction="none")
    return ((losses * active).sum(-1) / counts).mean()


def tensors(row, device):
    prompt, target = row["prompt_ids"], row["target_ids"]
    ids = torch.tensor([prompt + target], dtype=torch.long, device=device)
    labels = torch.tensor([[-100] * len(prompt) + target], dtype=torch.long, device=device)
    return ids, labels


def lr_multiplier(step, total, warmup_fraction):
    warmup = min(total, math.ceil(total * warmup_fraction))
    if warmup and step <= warmup:
        return step / warmup
    # Last planned optimizer update still has a positive learning rate.
    return max(0.0, (total - step + 1) / max(1, total - warmup))


def train_bounded(model, rows, plan, *, guard=None, on_checkpoint=None):
    """No loader, downloads, threads, retries, resumed state, or automatic sweep.

    Caller must qualify the graph and install Q/V adapters. Callback snapshots
    occur at step boundaries crossing 0/25/50/100% of actual selected tokens.
    Each callback receives CPU adapter tensors, never pretrained weights.
    """
    selected = bounded_prefix(rows, plan)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    expected = adapter_state(model)
    if not expected or {n for n, _ in trainable} != set(expected):
        raise ValueError("Trainable scope must be exactly this experiment's LoRA A/B tensors")
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=plan.learning_rate,
                                  betas=(plan.beta1, plan.beta2), weight_decay=plan.weight_decay)
    total_tokens = sum(len(r["prompt_ids"]) + len(r["target_ids"]) for r in selected)
    steps = math.ceil(len(selected) / plan.accumulation)
    old_modes = [(module, module.training) for module in model.modules()]
    trace, tokens, supervised, examples = [], 0, 0, 0
    started = time.perf_counter()
    pending = [0.25, 0.5, 1.0]
    def snapshot(fraction, step):
        if on_checkpoint is not None:
            on_checkpoint({"fraction": fraction, "step": step, "tokens": tokens,
                           "supervised_tokens": supervised, "examples": examples}, adapter_state(model))
    torch.manual_seed(plan.seed)
    try:
        model.train()
        snapshot(0.0, 0)
        with regularize(model, plan) as patched:
            for step, start in enumerate(range(0, len(selected), plan.accumulation), 1):
                batch = selected[start:start + plan.accumulation]
                optimizer.zero_grad(set_to_none=True)
                mean_loss = 0.0
                for row in batch:
                    if guard is not None:
                        guard()
                    ids, labels = tensors(row, device)
                    outputs = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                    loss = completion_loss(outputs.logits, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss; no retry authorized")
                    (loss / len(batch)).backward()
                    mean_loss += loss.item() / len(batch)
                    tokens += ids.numel()
                    supervised += len(row["target_ids"])
                    examples += 1
                    del outputs, loss, ids, labels
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], plan.max_grad_norm,
                                                       error_if_nonfinite=True)
                rate = plan.learning_rate * lr_multiplier(step, steps, plan.warmup_fraction)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.step()
                if any(not torch.isfinite(p).all() for _, p in trainable):
                    raise FloatingPointError("Nonfinite adapter state")
                trace.append({"step": step, "tokens": tokens, "examples": examples,
                              "supervised_tokens": supervised, "loss": mean_loss,
                              "gradient_norm": norm.item(), "learning_rate": rate})
                while pending and tokens >= total_tokens * pending[0]:
                    snapshot(pending.pop(0), step)
        return {"status": "BOUNDED_TRAINING_COMPLETE_NOT_QUALIFIED_FOR_RELEASE", "tokens": tokens,
                "supervised_tokens": supervised, "examples": examples, "steps": steps,
                "seconds": time.perf_counter() - started, "trace": trace, "patched_modules": patched,
                "trainable_parameters": sum(p.numel() for _, p in trainable)}
    finally:
        optimizer.zero_grad(set_to_none=True)
        for module, mode in old_modes:
            module.training = mode
