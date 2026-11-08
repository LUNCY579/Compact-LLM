"""Explicitly authorized native interfaces; imports alone allocate no model/GPU."""
from __future__ import annotations

import math
import random

from .core import Backend, Reply, digest, require
from .data import encode_row


def hf_backend(model, tokenizer, *, checkpoint_id: str, authorized=False,
               max_context=4096, temperature=0.7) -> Backend:
    """Wrap an ALREADY loaded, approved model. No loader, downloads or provider calls.

    Uses the exact unwrapped text/token boundary used by encode_row. A native host
    must qualify this envelope against its checkpoint; custom Rineya loaders stay upstream.
    """
    require(authorized is True, "explicit model-call grant required")
    require(type(checkpoint_id) is str and bool(checkpoint_id), "checkpoint fingerprint")
    require(type(max_context) is int and max_context > 0, "context")
    require(type(temperature) in (float, int) and math.isfinite(temperature) and temperature > 0, "temperature")
    require(type(tokenizer.eos_token_id) is int, "EOS required")
    import torch

    def generate(prompt, seed, max_new_tokens):
        ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        require(ids and len(ids) + max_new_tokens <= max_context, "native context overflow")
        device = next(model.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        was_training = model.training
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.manual_seed(seed)
                x = torch.tensor([ids], device=device)
                output = model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                                        max_new_tokens=max_new_tokens, do_sample=True,
                                        temperature=temperature, eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.eos_token_id, use_cache=True)
                require(output.ndim == 2 and output.shape[0] == 1 and output.shape[1] >= len(ids)
                        and torch.equal(output[0, :len(ids)], x[0]), "generation prefix contract")
                tail = output[0, len(ids):].tolist()
            text = tokenizer.decode(tail, skip_special_tokens=True)
            return Reply(text, len(ids), len(tail))
        finally:
            model.train(was_training)

    return Backend(checkpoint_id, "native", generate,
                   lambda text: len(tokenizer.encode(text, add_special_tokens=False)))


def train_sft(model, tokenizer, dataset: dict, *, authorized=False,
              initial_checkpoint: str, tokenizer_id: str, trainable_names: list[str],
              optimizer_steps=4, tokens_per_step=128, learning_rate=1e-5,
              max_context=4096, seed=37, retention_policy="none-mechanism-screen") -> dict:
    """Real causal-loss optimizer updates with exact supervised-token quotas.

    Uses full sequences without truncation. When a quota cuts through an action,
    the remaining target positions are supervised in the next update; future text
    cannot affect earlier labels in a causal model. This is not DPO/RL.
    Requires an explicit parameter allowlist (including pre-attached adapters).
    Host owns checkpoint rollback/persistence and resource/timeout enforcement.
    """
    require(authorized is True, "explicit training grant required")
    require(type(optimizer_steps) is int and optimizer_steps > 0, "optimizer_steps")
    require(type(tokens_per_step) is int and tokens_per_step > 0, "tokens_per_step")
    require(type(learning_rate) in (float, int) and math.isfinite(learning_rate) and learning_rate > 0, "learning_rate")
    require(initial_checkpoint and tokenizer_id, "immutable identities required")
    require(retention_policy == "none-mechanism-screen", "independent replay is not implemented in this trainer")
    require(dataset["schema"] == "rsr-sft-v1", "compiler schema")
    require(digest(dataset["rows"]) == dataset["rows_hash"], "dataset integrity")
    require(bool(trainable_names) and len(set(trainable_names)) == len(trainable_names), "parameter allowlist")
    import torch
    import torch.nn.functional as F

    parameters = dict(model.named_parameters())
    require(set(trainable_names) <= set(parameters), "unknown trainable parameter")
    require(all(p.grad is None for p in parameters.values()), "existing gradient state; use a fresh model")
    encoded = [encode_row(row, tokenizer, max_context) for row in dataset["rows"]]
    require(bool(encoded), "empty training data")
    trainable = [parameters[n] for n in sorted(trainable_names)]
    device = trainable[0].device
    require(all(p.device == device for p in parameters.values()), "single-device reference trainer")
    require(all(p.is_floating_point() for p in trainable), "trainable parameters must be floating point")
    previous_flags = {n: p.requires_grad for n, p in parameters.items()}
    was_training = model.training
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    order = list(range(len(encoded)))
    random.Random(seed).shuffle(order)
    cursor = target_offset = 0
    losses, exposures = [], [0] * len(encoded)
    try:
        for name, parameter in parameters.items():
            parameter.requires_grad_(name in trainable_names)
        model.train()
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for _ in range(optimizer_steps):
                optimizer.zero_grad(set_to_none=True)
                remaining, loss_total = tokens_per_step, 0.0
                while remaining:
                    index = order[cursor % len(order)]
                    row = encoded[index]
                    positions = [i for i, label in enumerate(row["labels"]) if label != -100]
                    take = min(remaining, len(positions) - target_offset)
                    labels = [-100] * len(row["labels"])
                    for position in positions[target_offset:target_offset + take]:
                        labels[position] = row["labels"][position]
                    inputs = torch.tensor([row["input_ids"]], device=device)
                    targets = torch.tensor([labels], device=device)
                    logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
                    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                           targets[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
                    require(bool(torch.isfinite(loss).item()), "nonfinite loss")
                    (loss / tokens_per_step).backward()
                    loss_total += float(loss.detach().cpu())
                    exposures[index] += take
                    remaining -= take
                    target_offset += take
                    if target_offset == len(positions):
                        cursor += 1
                        target_offset = 0
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in trainable),
                        "nonfinite gradients")
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(loss_total / tokens_per_step)
    finally:
        for name, parameter in parameters.items():
            parameter.requires_grad_(previous_flags[name])
            parameter.grad = None
        model.train(was_training)
    return dict(schema="rsr-training-receipt-v1", dataset_arm=dataset["arm"],
                rows_hash=dataset["rows_hash"], initial_checkpoint=initial_checkpoint,
                tokenizer_id=tokenizer_id, optimizer_steps=optimizer_steps,
                supervised_tokens=optimizer_steps * tokens_per_step, max_context=max_context,
                learning_rate=learning_rate, seed=seed, trainable_names=sorted(trainable_names),
                retention_policy=retention_policy, losses=losses, row_token_exposures=exposures,
                note="Optimizer completed; no capability/retention claim follows from training loss.")
