"""Bounded Phase 1 training loop.

Deliberately hand-written rather than delegated to ``Trainer``: the pilot needs
exact control over what is matched between Arm B (LoRA) and Arm C (NoRA-init) --
dataset, data order, seed, rank, optimizer, LR, scheduler, context length,
accumulation, step count and precision are all shared, and only the adapter
initialization differs.

Precision. The frozen base stays in float16, bit-identical to the checkpoint the
evaluation loads; adapter parameters are float32; the forward runs under
autocast with a gradient scaler. Training in the base's own dtype keeps Arm A,
B and C sharing one base.

Abstention gate. ``Qwen3AbstainForCausalLM.forward`` adds its constant offset to
the returned logits *after* the parent has already computed the loss, so the
gate never enters the objective. It also mutates the logit tensor in place,
which is an autograd hazard on a tensor that requires grad. The gate is
therefore disabled for the duration of training and restored before any
evaluation -- a no-op for the loss by construction, and asserted rather than
assumed.
"""

from __future__ import annotations

import contextlib
import math
import time
from typing import Any, Callable, Sequence

import torch


@contextlib.contextmanager
def gate_disabled(model: Any):
    """Temporarily zero the abstention logit bias, restoring it on exit."""
    config = model.config
    original = getattr(config, "abstain_logit_bias", None)
    try:
        if original:
            config.abstain_logit_bias = 0.0
        yield original
    finally:
        if original is not None:
            config.abstain_logit_bias = original


def adapter_norm(model: Any) -> float:
    """L2 norm over all trainable parameters -- how far the adapter has moved."""
    total = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total += float(p.detach().float().pow(2).sum())
    return math.sqrt(total)


def _batch(example: dict, device: str) -> dict:
    """Keep the full decoder input, projecting only causal answer positions.

    The extra preceding position predicts the first answer token. The final
    position remains ignored by the causal shift, so EOS is still supervised.
    This changes neither decoder attention nor the answer-token objective.
    """
    input_ids = example["input_ids"]
    labels = example["labels"]
    n_answer = example["n_answer_tokens"]
    if not isinstance(n_answer, int) or not 0 < n_answer <= len(input_ids):
        raise ValueError("answer-token count must be positive and fit the full input")
    if len(labels) != len(input_ids):
        raise ValueError("full labels and input must have the same length")
    if any(label != -100 for label in labels[:-n_answer]):
        raise ValueError("answer-only logits require a fully masked prompt")
    if labels[-n_answer:] != input_ids[-n_answer:]:
        raise ValueError("answer-only logits require a contiguous unmasked answer suffix")
    # min also preserves the existing objective for a synthetic no-prompt test;
    # real curriculum examples always have a nonempty masked chat prompt.
    keep = min(len(input_ids), n_answer + 1)
    return {
        "input_ids": torch.tensor([input_ids], device=device),
        "labels": torch.tensor([labels[-keep:]], device=device),
        "logits_to_keep": keep,
    }


def base_dtype(model: Any) -> torch.dtype:
    """The frozen base's storage dtype -- float16 for A1.1, bfloat16 for A1."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for p in base.parameters():
        if not p.requires_grad:
            return p.dtype
    return torch.float16


@torch.no_grad()
def evaluate_loss(model: Any, dataset: Sequence[dict], device: str,
                  amp_dtype: torch.dtype = torch.float16) -> dict:
    assert model.config.abstain_logit_bias, "gate must be on for validation evaluation"
    was_training = model.training
    total_loss = 0.0
    total_tokens = 0
    try:
        model.eval()
        for example in dataset:
            batch = _batch(example, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                out = model(**batch)
            n = example["n_answer_tokens"]
            total_loss += float(out.loss.detach()) * n
            total_tokens += n
            del out
    finally:
        model.train(was_training)
    return {
        "loss": round(total_loss / max(1, total_tokens), 6),
        "n_examples": len(dataset),
        "n_answer_tokens": total_tokens,
    }


def enable_gradient_checkpointing(model: Any) -> bool:
    """Trade ~30% step time for roughly 4x less activation memory.

    This lab shares an 8 GB laptop GPU with other research that routinely holds
    7 GB. Without checkpointing, 2560-token activations across 28 layers need
    roughly 3 GB and the run simply OOMs whenever a neighbour is busy. Being a
    good side-car is a stated requirement, so this is on by default.

    ``use_reentrant=False`` is required: the reentrant implementation sees no
    input requiring grad (the base is frozen) and silently produces no gradient
    for the adapter.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.config.use_cache = False
        # Every parameter before layer 17 is frozen, so the embedding output is
        # the first tensor in the graph that could carry grad. Marking it makes
        # the checkpointed segments differentiable regardless of implementation.
        if hasattr(base, "enable_input_require_grads") and not hasattr(base, "_require_grads_hook"):
            base.enable_input_require_grads()
    except Exception as exc:
        raise RuntimeError("required non-reentrant gradient checkpointing could not be enabled") from exc
    if not getattr(base, "is_gradient_checkpointing", False):
        raise RuntimeError("gradient checkpointing was requested but is not active")
    return True


@contextlib.contextmanager
def _training_environment(model: Any, checkpointing_requested: bool):
    """Restore mode, cache and temporary checkpointing hooks even on failure."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    was_training = model.training
    old_cache = getattr(base.config, "use_cache", None)
    old_checkpointing = bool(getattr(base, "is_gradient_checkpointing", False))
    old_input_hook = hasattr(base, "_require_grads_hook")
    try:
        model.train()
        enabled = checkpointing_requested and enable_gradient_checkpointing(model)
        yield bool(enabled)
    finally:
        if checkpointing_requested and not old_checkpointing:
            if hasattr(base, "gradient_checkpointing_disable"):
                base.gradient_checkpointing_disable()
            if not old_input_hook and hasattr(base, "_require_grads_hook"):
                base.disable_input_require_grads()
        if old_cache is not None:
            base.config.use_cache = old_cache
        model.train(was_training)


def _validation_with_gate(model, dataset, device, amp_dtype, original_bias):
    """The gate changes returned logits, not the already-computed loss."""
    assert original_bias, "the canonical evaluation gate must be present"
    training_bias = model.config.abstain_logit_bias
    try:
        model.config.abstain_logit_bias = original_bias
        return evaluate_loss(model, dataset, device, amp_dtype)
    finally:
        model.config.abstain_logit_bias = training_bias


def train(
    model: Any,
    train_set: Sequence[dict],
    *,
    device: str = "cuda",
    max_steps: int = 300,
    grad_accum: int = 8,
    lr: float = 1e-4,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    seed: int = 17,
    gradient_checkpointing: bool = True,
    eval_set: Sequence[dict] | None = None,
    eval_every: int = 0,
    log: Callable[[str], None] = print,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Run exactly ``max_steps`` attempts; separately count applied updates.

    An overflow consumes its fixed examples but never advances the LR scheduler.
    The budget is never silently extended to compensate for skipped updates.
    """
    if not train_set or max_steps < 1 or grad_accum < 1:
        raise ValueError("training requires nonempty data and positive steps/accumulation")
    assert model.config.abstain_logit_bias, "expected the canonical evaluation gate"
    torch.manual_seed(seed)

    # Autocast in the base's own dtype. bfloat16 carries float32's exponent
    # range, so it needs no loss scaling; float16 does. Using one recipe for
    # both cores would either destabilise bf16 or under-protect fp16.
    amp_dtype = base_dtype(model)
    needs_scaler = amp_dtype == torch.float16

    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named_params or any(not ({"lora_A", "lora_B"} & set(name.split("."))) for name, _ in named_params):
        raise RuntimeError("training must contain only actual LoRA A/B parameters")
    params = [p for _, p in named_params]
    for p in params:
        p.data = p.data.float()

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    warmup = max(1, int(warmup_ratio * max_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # A default init_scale of 65536 overflows for the first several steps and
    # the scaler discards them. On a 300-step budget that is real training
    # lost, so start lower and let it grow.
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=2 ** 12, enabled=(device == "cuda" and needs_scaler)
    )

    trajectory: list[dict] = []
    evals: list[dict] = []
    cursor = 0
    consumed = 0
    skipped_steps = 0
    safety_skipped_steps = 0
    applied_steps = 0
    finite_nonzero_steps = 0
    applied_examples = 0
    applied_answer_tokens = 0
    t0 = time.perf_counter()

    with _training_environment(model, bool(gradient_checkpointing)) as checkpointing, gate_disabled(model) as original_bias:
        assert model.config.abstain_logit_bias == 0.0, "gate not disabled for training"
        for step in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_tokens = 0

            for _ in range(grad_accum):
                example = train_set[cursor % len(train_set)]
                cursor += 1
                consumed += 1
                batch = _batch(example, device)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                    out = model(**batch)
                loss = out.loss / grad_accum
                scaler.scale(loss).backward()
                step_loss += float(out.loss.detach())
                step_tokens += example["n_answer_tokens"]
                del out, loss

            if not any(p.grad is not None for p in params):
                raise RuntimeError("no gradient reached any LoRA parameter")
            scaler.unscale_(optimizer)
            gradients_finite = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
            finite = math.isfinite(grad_norm)
            usable_gradient = gradients_finite and finite and grad_norm > 0.0
            finite_nonzero_steps += int(usable_gradient)
            scale_enabled = scaler.is_enabled()
            before_scale = scaler.get_scale() if scale_enabled else None
            applied_lr = optimizer.param_groups[0]["lr"]
            skipped_by_scaler = False
            if usable_gradient:
                scaler.step(optimizer)
                scaler.update()
                skipped_by_scaler = bool(scale_enabled and scaler.get_scale() < before_scale)
                applied = not skipped_by_scaler
            elif scale_enabled and not gradients_finite:
                # unscale_ recorded nonfinite gradients, so GradScaler will skip
                # AdamW itself; neither bf16 nor a disabled scaler takes this path.
                scaler.step(optimizer)
                scaler.update()
                skipped_by_scaler = True
                applied = False
            else:
                # A bf16 nonfinite gradient, zero gradient, or overflowing norm
                # must never reach an optimizer, including with no GradScaler.
                if scale_enabled:
                    scaler.update()
                applied = False
            skipped = not applied
            skipped_steps += int(skipped_by_scaler)
            safety_skipped_steps += int(skipped and not skipped_by_scaler)
            if applied:
                applied_steps += 1
                applied_examples += grad_accum
                applied_answer_tokens += step_tokens
                scheduler.step()

            record = {
                "step": step + 1,
                "attempt": step + 1,
                "optimizer_updates_applied": applied_steps,
                "optimizer_update_applied": applied,
                "examples_consumed": consumed,
                "examples_in_applied_updates": applied_examples,
                "answer_tokens_in_applied_updates": applied_answer_tokens,
                "train_loss": round(step_loss / grad_accum, 6) if math.isfinite(step_loss) else None,
                "grad_norm": round(grad_norm, 6) if finite else None,
                "gradients_finite": gradients_finite,
                "finite_nonzero_gradient": usable_gradient,
                "grad_overflow": not finite,
                "step_skipped_by_scaler": skipped_by_scaler,
                "step_skipped_by_safety_check": skipped and not skipped_by_scaler,
                "steps_skipped_by_scaler_total": skipped_steps,
                "steps_skipped_by_safety_check_total": safety_skipped_steps,
                "scaler_scale_before": before_scale,
                "scaler_scale_after": scaler.get_scale() if scale_enabled else None,
                "lr_used": round(applied_lr, 10),
                "lr": round(scheduler.get_last_lr()[0], 10),
                "adapter_norm": round(adapter_norm(model), 6),
                "answer_tokens": step_tokens,
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
            }
            trajectory.append(record)
            if progress_callback:
                progress_callback({"event": "optimizer_attempt", **record})

            if eval_every and eval_set and (step + 1) % eval_every == 0:
                evals.append({"step": step + 1,
                              **_validation_with_gate(model, eval_set, device, amp_dtype, original_bias)})
                if progress_callback:
                    progress_callback({"event": "validation", **evals[-1]})
                gn = record["grad_norm"]
                log(f"    step {step + 1}: train {record['train_loss']} "
                    f"val {evals[-1]['loss']:.4f} "
                    f"|g| {gn if gn is None else format(gn, '.3f')}"
                    f"{' SKIPPED' if skipped else ''}")
            elif (step + 1) % max(1, max_steps // 10) == 0:
                gn = record["grad_norm"]
                log(f"    step {step + 1}/{max_steps}: train {record['train_loss']} "
                    f"|g| {gn if gn is None else format(gn, '.3f')} "
                    f"|adapter| {record['adapter_norm']:.3f}")

        if eval_set and (not evals or evals[-1]["step"] != max_steps):
            evals.append({"step": max_steps,
                          **_validation_with_gate(model, eval_set, device, amp_dtype, original_bias)})
            if progress_callback:
                progress_callback({"event": "validation", **evals[-1]})

        if not finite_nonzero_steps or not applied_steps:
            raise RuntimeError("no finite nonzero gradient produced an applied adapter update")

    assert model.config.abstain_logit_bias == original_bias, "gate not restored"

    wall = time.perf_counter() - t0
    return {
        "max_steps": max_steps,
        "optimizer_attempts": max_steps,
        "grad_accum": grad_accum,
        "effective_batch": grad_accum,
        "examples_consumed": consumed,
        "steps_skipped_by_scaler": skipped_steps,
        "steps_skipped_by_safety_check": safety_skipped_steps,
        "steps_applied": applied_steps,
        "finite_nonzero_gradient_attempts": finite_nonzero_steps,
        "examples_in_applied_updates": applied_examples,
        "answer_tokens_in_applied_updates": applied_answer_tokens,
        "epochs_equivalent": round(consumed / max(1, len(train_set)), 3),
        "lr": lr,
        "warmup_steps": warmup,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "precision": (f"frozen base {str(amp_dtype).replace('torch.','')}, adapter float32, "
                      f"autocast {str(amp_dtype).replace('torch.','')}"
                      f"{' + GradScaler' if needs_scaler else ' (no scaler; bf16 range)'}"),
        "gradient_checkpointing": checkpointing,
        "gate_disabled_for_training": True,
        "gate_on_for_validation_evaluations": bool(evals),
        "gate_restored_after_training": model.config.abstain_logit_bias == original_bias,
        "wall_seconds": round(wall, 2),
        "steps_per_second": round(max_steps / wall, 4),
        "applied_updates_per_second": round(applied_steps / wall, 4),
        "final_train_loss": trajectory[-1]["train_loss"] if trajectory else None,
        "first_train_loss": trajectory[0]["train_loss"] if trajectory else None,
        "final_adapter_norm": trajectory[-1]["adapter_norm"] if trajectory else None,
        "trajectory": trajectory,
        "validation": evals,
    }
