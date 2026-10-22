"""Bounded matched pilot: no automatic training, no production checkpoint writes."""
from __future__ import annotations

import gc
from pathlib import Path
import random
import time

import torch

from .audit import audit_case, metric_passes, numerical_gate
from .backends import precision_guard
from .capture import save_capture, sdpa_session
from .core import Case
from .native import forward_loss, load_model, preflight, qualification_batches
from .protocol import parameter_hash, verify_qualification, write_json

ARMS = ("native", "fp32", "cast_only", "gproj")


def prepare_output(output: Path, checkpoint: str) -> None:
    if output.resolve().is_relative_to(Path(checkpoint).resolve()):
        raise ValueError("Outputs must not be written inside the source checkpoint")
    output.mkdir(parents=True, exist_ok=False)


def qualify(config: dict, output: Path) -> dict:
    rows, ident = preflight(config)
    prepare_output(output, config["checkpoint"])
    write_json(output / "identity.json", ident)
    write_json(output / "frozen_config.json", config)
    model = None
    try:
        model = load_model(config)
        captures, model_details, events = qualification_batches(model, rows, config)
    except Exception as exc:
        write_json(output / "FAILED.json", {"status": "FAILED", "reason": str(exc)})
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    audits = []
    for index, case in enumerate(captures):
        save_capture(case, output / f"capture_{index:03d}.pt")
        payload = {key: value.to(config["device"]) if isinstance(value, torch.Tensor) else value
                   for key, value in vars(case).items()}
        audits.append(audit_case(Case(**payload), config["backend"]))
    relative, absolute = config["audit_relative_tolerance"], config["audit_absolute_tolerance"]
    passed = numerical_gate(audits, relative, absolute) and bool(events)
    # No-update full-model check catches errors that boundary tests miss.
    passed &= all(metric_passes(metric, relative, absolute)
                  for record in model_details
                  for metric in record["trainable_gradients_vs_fp32"]["gproj"].values())
    report = {"identity": ident, "status": "NATIVE_NUMERICAL_QUALIFIED" if passed else "BLOCKED",
              "capture_count": len(captures), "backend_events": events,
              "cohorts": sorted({case.metadata["purpose"] for case in captures}),
              "boundary_audits": audits, "full_model_diagnostics": model_details,
              "scope": "exact local checkpoint/runtime; short dense research arms only",
              "fused_gproj_qualified": False, "efficacy_established": False}
    write_json(output / "qualification.json", report)
    return report


def dev_losses(model, rows: list[dict], config: dict) -> dict:
    """Fixed FP32-attention evaluation. Loss proxies are NOT skill pass rates."""
    records = []
    with torch.no_grad(), precision_guard(), sdpa_session("fp32", "math", max_tokens=config["max_tokens"]):
        for row in rows:
            if row["split"] == "dev":
                loss = float(forward_loss(model, row, config))
                records.append({"id": row["id"], "group_id": row["group_id"],
                                "purpose": row["purpose"], "loss": loss,
                                "supervised_tokens": sum(t != -100 for t in row["labels"][1:])})
    return {"endpoint": "teacher_forced_loss_proxy_not_behavioral_success", "rows": records,
            "by_purpose": {purpose: sum(r["loss"] * r["supervised_tokens"] for r in records if r["purpose"] == purpose)
                           / sum(r["supervised_tokens"] for r in records if r["purpose"] == purpose)
                           for purpose in ("acquisition", "retention")}}


def train_arm(model, rows: list[dict], config: dict, arm: str, seed: int) -> tuple[dict, dict]:
    if arm not in ARMS:
        raise ValueError("Unregistered training arm")
    torch.manual_seed(seed)
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    if sum(p.numel() for _, p in parameters) > config["max_trainable_parameters"]:
        raise ValueError("Trainable budget exceeded")
    if any(p.dtype != torch.float32 for _, p in parameters):
        raise ValueError("Pilot requires FP32 master trainable parameters")
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    frozen_before = parameter_hash(frozen)
    initial = {name: p.detach().cpu().clone() for name, p in parameters}
    baseline = dev_losses(model, rows, config)
    train_rows = [row for row in rows if row["split"] == "train"]
    order = list(range(len(train_rows)))
    random.Random(seed).shuffle(order)
    scheduled = [train_rows[order[step % len(order)]] for step in range(config["max_steps"])]
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=config["learning_rate"],
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    steps = []
    if config["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with precision_guard(), sdpa_session(arm, config["backend"], max_tokens=config["max_tokens"]) as session:
        for index, row in enumerate(scheduled):
            if config["device"] == "cuda":
                free, _ = torch.cuda.mem_get_info()
                if free < config["min_free_cuda_gib"] * 1024**3:
                    raise RuntimeError("Free-memory floor crossed; abort without changing the recipe")
                torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            before = [p.detach().clone() for _, p in parameters]
            loss = forward_loss(model, row, config)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in parameters):
                raise RuntimeError("Missing/nonfinite gradient; arm stopped before optimizer step")
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], config["clip_norm"],
                                                  error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for _, p in parameters):
                raise RuntimeError("Nonfinite update; failed arm must not be promoted")
            update = sum(float((p.detach() - old).double().square().sum())
                         for (_, p), old in zip(parameters, before)) ** 0.5
            if config["device"] == "cuda":
                torch.cuda.synchronize()
            steps.append({"step": index, "row_id": row["id"], "input_tokens": len(row["input_ids"]),
                          "loss": float(loss.detach()), "preclip_gradient_norm": float(norm),
                          "actual_update_l2": update, "step_seconds": time.perf_counter() - start})
    if not session["calls"] or parameter_hash(frozen) != frozen_before:
        raise RuntimeError("No interception or frozen-weight mutation")
    final = dev_losses(model, rows, config)
    deltas = {name: p.detach().cpu() - initial[name] for name, p in parameters}
    report = {"arm": arm, "seed": seed, "steps": steps, "before": baseline, "after": final,
              "frozen_parameters_unchanged": True,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if config["device"] == "cuda" else None,
              "capability_claim": False, "timing_scope": "instrumented dense micro-pilot, not fused-kernel benchmark"}
    return report, deltas


def matched_train(config: dict, qualification: dict, output: Path,
                  execute_training: bool = False) -> dict:
    if not execute_training:
        raise PermissionError("Training disabled. Qualify first, then explicitly pass --execute-training")
    rows, ident = preflight(config)
    verify_qualification(qualification, ident)
    prepare_output(output, config["checkpoint"])
    write_json(output / "identity.json", ident)
    write_json(output / "frozen_config.json", config)
    results, starting_hash = [], None
    for seed in config["seeds"]:
        for arm in ARMS:
            torch.manual_seed(seed)
            model = None
            try:
                model = load_model(config)
                current = parameter_hash(model.named_parameters())
                if starting_hash is None:
                    starting_hash = current
                if current != starting_hash:
                    raise RuntimeError("Unmatched initial parameters across arms/seeds")
                record, delta = train_arm(model, rows, config, arm, seed)
                write_json(output / f"{seed}_{arm}.json", record)
                torch.save(delta, output / f"{seed}_{arm}_delta.pt")
                results.append(record)
            except Exception as exc:
                write_json(output / f"{seed}_{arm}_FAILED.json", {"status": "FAILED", "reason": str(exc)})
                raise
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    report = {"identity": ident, "status": "EXPLORATORY_MICROPILOT_COMPLETED",
              "initial_parameters_sha256": starting_hash, "runs": results,
              "promotion_allowed": False,
              "limitations": ["Loss endpoints are proxies, not demonstrated skill acquisition/retention.",
                              "No sealed suites read; no optimized GProj kernel or speed claim.",
                              "Inspect paired seeds without treating rows x seeds as independent replications."]}
    write_json(output / "matched_report.json", report)
    return report
