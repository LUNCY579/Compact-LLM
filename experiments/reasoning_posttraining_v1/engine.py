"""Bounded microfit engine reusing MQ losses, Adam, RNG and full-state storage.

The engine is not an execution permit. The supervised native controller must
authorize and account for every reservation before using the actual checkpoint.
"""
from __future__ import annotations
from pathlib import Path
import os
import torch

from experiments.model_quality_gpu_v1.checkpoints import (
    restore_checkpoint, rng_state, restore_rng, save_checkpoint, validate_checkpoint)
from experiments.model_quality_gpu_v1.config import Design, rate_at
from experiments.model_quality_gpu_v1.data import group_updates
from experiments.model_quality_gpu_v1.integrity import canonical, file_hash, new_dir, write_new
from experiments.model_quality_gpu_v1.losses import chunked_sum
from experiments.model_quality_gpu_v1.native import hidden_states, parameter_digest, select_tail
from experiments.model_quality_gpu_v1.optimizer import CpuAdamW


def progress_at(rows, groups, count):
    if type(count) is not int or not 0 <= count <= len(groups):
        raise ValueError("invalid update cursor")
    seen = [i for group in groups[:count] for i in group]
    return {"updates": count, "next_group": count,
            "processed_tokens": sum(len(rows[i]["ids"]) for i in seen),
            "supervised_tokens": sum(sum(t != -100 for t in rows[i]["labels"][1:]) for i in seen)}


def microfit(model, rows: list[dict], design: Design, binding: dict, directory: Path, *,
             evaluate, reserve, resource_guard, resume: Path | None = None,
             synthetic=False, stop_after_updates=None):
    """Evaluate callbacks use a fixed probe set and cannot change training RNG."""
    if not callable(reserve) or not callable(resource_guard) or not callable(evaluate):
        raise ValueError("budget, resource and evaluator contracts are required")
    # The result is attributed from here. A binding without it would silently mislabel the
    # run, so refuse instead of defaulting to a literal (which is what it used to do).
    if not isinstance(binding.get("experiment"), str) or not binding["experiment"].strip():
        raise ValueError("binding must name its experiment; the result is attributed from it")
    params = sum(p.numel() for p in model.parameters())
    if synthetic:
        if params >= 1000000 or next(model.parameters()).device.type != "cpu":
            raise ValueError("synthetic hooks are limited to small CPU fixtures")
    elif params != 596049920 or design.trainable_tail != 20 or stop_after_updates is not None:
        raise ValueError("native microfit requires exact 596M tail20 inventory and registered stopping")
    if not rows or any(r["source_split"] != "train" for r in rows):
        raise ValueError("microfit only consumes declared TRAIN presentations")
    if any(len(r["ids"]) != len(r["labels"]) or r["labels"][0] != -100 for r in rows):
        raise ValueError("invalid shifted target contract")
    groups = group_updates(rows, design.update_tokens)
    total = progress_at(rows, groups, len(groups))
    if total["processed_tokens"] != design.processed_budget:
        raise ValueError("exact processed exposure differs from frozen design")
    directory = new_dir(directory)
    named = select_tail(model, design.trainable_tail)
    if not synthetic and (len(named) != 220 or sum(p.numel() for _, p in named) != 314618880):
        raise ValueError("native trainable tensor inventory differs")
    frozen_before = parameter_digest(model, frozen_only=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    resource_guard()
    optimizer = CpuAdamW(named, design)
    progress = progress_at(rows, groups, 0)
    if resume is not None:
        meta = validate_checkpoint(resume, named, binding)
        if meta["progress"] != progress_at(rows, groups, meta["progress"]["updates"]):
            raise ValueError("saved cursor does not describe this presentation plan")
        progress = restore_checkpoint(resume, optimizer, binding)
    attempt_start = dict(progress)

    def probe(label):
        resource_guard()
        original_rng, was_training = rng_state(), model.training
        try:
            model.eval()
            with torch.no_grad():
                value = evaluate(model, label, reserve)
            write_new(directory / (label + ".json"), value)
            return value
        finally:
            restore_rng(original_rng)
            model.train(was_training)

    if resume is None:
        probe("baseline")
    else:
        write_new(directory / "resume_source.json", {"checkpoint": str(resume),
                  "manifest_sha256": file_hash(resume / "manifest.json"), "progress": progress})
    device = next(model.parameters()).device
    with (directory / "steps.jsonl").open("xb") as log:
        for index in range(progress["next_group"], len(groups)):
            group = groups[index]
            processed = sum(len(rows[i]["ids"]) for i in group)
            denominator = sum(sum(x != -100 for x in rows[i]["labels"][1:]) for i in group)
            if denominator < 1:
                raise ValueError("an update has no response targets")
            reserve({"processed_training_tokens_all_attempts": processed})
            resource = resource_guard()
            model.train(); optimizer.zero_grad()
            value = 0.0
            for i in group:
                row = rows[i]
                ids = torch.tensor([row["ids"]], device=device, dtype=torch.long)
                labels = torch.tensor([row["labels"]], device=device, dtype=torch.long)
                loss, _ = chunked_sum(hidden_states(model, ids), model.lm_head.weight, labels, design.ce_chunk)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite microfit loss")
                (loss / denominator).backward()
                value += float(loss.detach()) / denominator
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError("frozen tensor acquired gradient")
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], design.clip_norm, error_if_nonfinite=True)
            progress = progress_at(rows, groups, index + 1)
            lr = rate_at(1e-5, progress["processed_tokens"], total["processed_tokens"], design)
            optimizer.step(lr); optimizer.zero_grad()
            record = {**progress, "objective_ce": value, "lr": lr, "grad_norm": float(norm),
                      "row_ids": [rows[i]["id"] for i in group], "resources": resource}
            log.write(canonical(record) + b"\n"); log.flush(); os.fsync(log.fileno())
            stopped = stop_after_updates is not None and progress["updates"] >= stop_after_updates
            if index + 1 == 2 or index + 1 == len(groups) or stopped:
                # Conservative budget reservation includes full parameters/moments and metadata.
                reserve({"new_disk_bytes": sum(p.numel() for _, p in named) * 12 + 2**20})
                resource_guard()
                save_checkpoint(directory / f"checkpoint-{index + 1:05d}", optimizer, binding, progress)
            if resume is None and index + 1 == max(1, len(groups) // 2):
                probe("midpoint")
            if stopped:
                break
    if parameter_digest(model, frozen_only=True) != frozen_before:
        raise RuntimeError("frozen source weights changed")
    final = probe("final")
    # The binding carries the experiment. Hardcoding a literal here misattributed every
    # successor arm to RRP-001 in its own result file.
    result = {"experiment": binding["experiment"],
              "status": "MICROFIT_EXECUTED_NOT_QUALITY_CONFIRMATION",
              "synthetic": synthetic, "complete": progress["updates"] == len(groups), "progress": progress,
              "attempt_updates": progress["updates"] - attempt_start["updates"],
              "attempt_processed_tokens": progress["processed_tokens"] - attempt_start["processed_tokens"],
              "binding": binding, "frozen_sha256": frozen_before, "final_state_sha256": parameter_digest(model),
              "final_evaluation_sha256": file_hash(directory / "final.json"), "steps_sha256": file_hash(directory / "steps.jsonl"),
              "quality_improvement_established": False, "release_permission": False}
    write_new(directory / "result.json", result)
    return result
