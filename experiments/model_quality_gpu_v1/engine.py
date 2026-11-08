"""One matched arm, clean update checkpoints, frozen-backbone checks, raw dev outputs."""
from __future__ import annotations

import math
from pathlib import Path
import torch
from .checkpoints import restore_checkpoint, save_checkpoint, validate_checkpoint
from .config import ARMS, Design, rate_at
from .data import targets, validate_plan
from .evaluation import evaluate
from .integrity import digest, file_hash, write_new
from .losses import chunked_sum
from .native import hidden_states, parameter_digest, select_tail
from .optimizer import CpuAdamW


def expected_progress(plan: dict, objective: str, updates: int) -> dict:
    if type(updates) is not int or not 0 <= updates <= len(plan["groups"]):
        raise ValueError("checkpoint cursor beyond registered update groups")
    indices = [i for group in plan["groups"][:updates] for i in group]
    return {"updates": updates, "next_group": updates,
            "processed_tokens": sum(len(plan["train"][i]["ids"]) for i in indices),
            "supervised_tokens": sum(sum(t != -100 for t in targets(plan["train"][i], objective)[1:])
                                     for i in indices)}


def train_cell(model, sealed_plan: dict, arm: str, directory: Path, binding: dict, *,
               resume: Path | None = None, resource_guard=None, tokenizer=None, eos_ids=None,
               stop_after_updates: int | None = None) -> dict:
    """Synthetic fixtures may interrupt at a clean boundary. Native CLI exposes no such switch."""
    plan = validate_plan(sealed_plan)
    if arm not in ARMS:
        raise ValueError("unknown frozen arm")
    if stop_after_updates is not None and plan["kind"] != "synthetic":
        raise ValueError("test interruption hook is synthetic-only")
    d = Design(**plan["design"])
    objective, peak = ARMS[arm]
    named = select_tail(model, d.trainable_tail)
    frozen_before = parameter_digest(model, frozen_only=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = CpuAdamW(named, d)
    baseline = evaluate(model, plan["dev"], d.ce_chunk, tokenizer=tokenizer, eos_ids=eos_ids)
    progress = expected_progress(plan, objective, 0)
    if resume is not None:
        meta = validate_checkpoint(resume, named, binding)
        if meta["progress"] != expected_progress(plan, objective, meta["progress"]["updates"]):
            raise ValueError("checkpoint cursor/counts do not describe this frozen plan")
        progress = restore_checkpoint(resume, optimizer, binding)
    write_new(directory / "baseline.json", baseline)
    trace, checkpoint_path = [], None
    device = next(model.parameters()).device
    with (directory / "steps.jsonl").open("xb") as log:
        for group_index in range(progress["next_group"], len(plan["groups"])):
            resource = resource_guard() if resource_guard else {"scope": "synthetic"}
            group = plan["groups"][group_index]
            denominator = sum(sum(t != -100 for t in targets(plan["train"][i], objective)[1:])
                              for i in group)
            if denominator <= 0:
                raise ValueError("update without supervised targets")
            model.train(); optimizer.zero_grad()
            update_loss = 0.0
            for i in group:
                row = plan["train"][i]
                ids = torch.tensor([row["ids"]], dtype=torch.long, device=device)
                labels = torch.tensor([targets(row, objective)], dtype=torch.long, device=device)
                loss, _ = chunked_sum(hidden_states(model, ids), model.lm_head.weight, labels, d.ce_chunk)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                (loss / denominator).backward()
                update_loss += float(loss.detach()) / denominator
                del loss
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError("frozen parameter acquired a gradient")
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], d.clip_norm,
                                                  error_if_nonfinite=True)
            next_progress = expected_progress(plan, objective, group_index + 1)
            lr = rate_at(peak, next_progress["processed_tokens"], plan["processed_tokens"], d)
            optimizer.step(lr); optimizer.zero_grad()
            progress = next_progress
            record = {**progress, "arm": arm, "lr": lr, "objective_ce": update_loss,
                      "grad_norm": float(norm), "row_ids": [plan["train"][i]["id"] for i in group],
                      "resource_before_update": resource}
            from .integrity import canonical
            import os
            log.write(canonical(record) + b"\n"); log.flush(); os.fsync(log.fileno())
            trace.append(record)
            stopped = stop_after_updates is not None and progress["updates"] >= stop_after_updates
            if progress["updates"] % d.checkpoint_every == 0 or group_index + 1 == len(plan["groups"]) or stopped:
                checkpoint_path = directory / f"checkpoint-{progress['updates']:05d}"
                save_checkpoint(checkpoint_path, optimizer, binding, progress)
            if stopped:
                break
    if parameter_digest(model, frozen_only=True) != frozen_before:
        raise RuntimeError("frozen source weights changed")
    complete = progress["updates"] == len(plan["groups"])
    # Evaluation is AFTER the final optimizer update, including the partial last group.
    final = evaluate(model, plan["dev"], d.ce_chunk, tokenizer=tokenizer, eos_ids=eos_ids)
    write_new(directory / "final_dev.json", final)
    result = {"schema": 1, "experiment": "MQ-001", "arm": arm, "binding": binding,
              "status": "ENGINEERING_COMPLETE_QUALITY_UNDECIDED" if complete else "INTERRUPTED_FIXTURE",
              "synthetic": plan["kind"] == "synthetic", "progress": progress,
              "frozen_sha256": frozen_before, "checkpoint": checkpoint_path.name if checkpoint_path else None,
              "baseline_sha256": file_hash(directory / "baseline.json"),
              "final_dev_sha256": file_hash(directory / "final_dev.json"),
              "steps_sha256": file_hash(directory / "steps.jsonl"),
              "quality_improvement_established": False, "release_permission": False}
    write_new(directory / "result.json", result)
    return result
