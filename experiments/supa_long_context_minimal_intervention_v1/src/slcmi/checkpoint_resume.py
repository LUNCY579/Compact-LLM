"""Isolated audit prototype: transactional latest-state checkpoints, never quality selection."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import torch

try:
    import numpy as np
except ImportError:
    np = None


FINGERPRINT_KEYS = {
    "recipe_sha256", "train_data_sha256", "eval_data_sha256",
    "base_checkpoint_sha256", "adapter_initialization_sha256",
}


def fingerprint(value):
    """Hash exact JSON content; array order is deliberately significant."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_identity(caller, train_set, eval_set, effective_recipe):
    if not isinstance(caller, dict) or set(caller) != FINGERPRINT_KEYS:
        raise ValueError("checkpoint fingerprints require exactly " + ", ".join(sorted(FINGERPRINT_KEYS)))
    for name, value in caller.items():
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    if caller["train_data_sha256"] != fingerprint(list(train_set)):
        raise ValueError("caller train data fingerprint does not match exact ordered examples")
    if caller["eval_data_sha256"] != fingerprint(list(eval_set) if eval_set is not None else None):
        raise ValueError("caller eval data fingerprint does not match exact ordered examples")
    return {"caller": dict(caller), "effective_recipe": effective_recipe,
            "effective_recipe_sha256": fingerprint(effective_recipe)}


def clone_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(v) for v in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"unsafe checkpoint value type: {type(value).__name__}")


def capture_rng(device):
    numpy_state = None
    if np is not None:
        state = np.random.get_state()
        numpy_state = [state[0], torch.tensor(state[1].astype("int64")),
                       int(state[2]), int(state[3]), float(state[4])]
    return {"torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [s.cpu().clone() for s in torch.cuda.get_rng_state_all()] if device == "cuda" else [],
            "python": random.getstate(), "numpy": numpy_state}


def restore_rng(state, device):
    if device == "cuda" and len(state["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("checkpoint CUDA RNG device topology mismatch")
    if device != "cuda" and state["torch_cuda"]:
        raise ValueError("CUDA checkpoint cannot be resumed on CPU")
    if (np is not None) != (state["numpy"] is not None):
        raise ValueError("checkpoint NumPy RNG availability mismatch")
    torch.set_rng_state(state["torch_cpu"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    if np is not None:
        s = state["numpy"]
        np.random.set_state((s[0], s[1].numpy().astype("uint32"), s[2], s[3], s[4]))


def validate_rng(state, device):
    """Validate on private generators before mutating caller parameters or RNG."""
    try:
        if not isinstance(state, dict) or set(state) != {"torch_cpu", "torch_cuda", "python", "numpy"}:
            raise ValueError("RNG schema")
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
        random.Random().setstate(state["python"])
        if (np is not None) != (state["numpy"] is not None):
            raise ValueError("NumPy availability")
        if np is not None:
            s = state["numpy"]
            np.random.RandomState().set_state((s[0], s[1].numpy().astype("uint32"), s[2], s[3], s[4]))
        expected_devices = torch.cuda.device_count() if device == "cuda" else 0
        if len(state["torch_cuda"]) != expected_devices:
            raise ValueError("CUDA topology")
        for i, saved in enumerate(state["torch_cuda"]):
            torch.Generator(device=f"cuda:{i}").set_state(saved)
    except (RuntimeError, TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:
        raise ValueError("checkpoint RNG state is invalid") from exc


def validate_progress(progress, *, max_steps, grad_accum, eval_due, train_set):
    expected_progress = {"completed_attempts", "cursor", "consumed", "skipped_steps", "safety_skipped_steps",
                         "applied_steps", "finite_nonzero_steps", "applied_examples", "applied_answer_tokens",
                         "trajectory", "evals", "pending_validation", "elapsed_seconds"}
    if not isinstance(progress, dict) or set(progress) != expected_progress:
        raise ValueError("malformed checkpoint progress fields")
    count_keys = expected_progress - {"trajectory", "evals", "pending_validation", "elapsed_seconds"}
    if any(type(progress[k]) is not int or progress[k] < 0 for k in count_keys):
        raise ValueError("checkpoint counters must be nonnegative integers")
    if not isinstance(progress["elapsed_seconds"], (int, float)) or not math.isfinite(progress["elapsed_seconds"]) or progress["elapsed_seconds"] < 0:
        raise ValueError("checkpoint elapsed duration is invalid")
    done = progress["completed_attempts"]
    if done > max_steps:
        raise ValueError("checkpoint attempt budget mismatch")
    if progress["cursor"] != done * grad_accum or progress["consumed"] != progress["cursor"]:
        raise ValueError("checkpoint data cursor does not match committed attempts")
    if not isinstance(progress["trajectory"], list) or len(progress["trajectory"]) != done:
        raise ValueError("checkpoint trajectory does not match committed attempts")
    applied = skips = safety = finite = tokens = 0
    for attempt, record in enumerate(progress["trajectory"], 1):
        flags = [record["optimizer_update_applied"], record["step_skipped_by_scaler"], record["step_skipped_by_safety_check"]]
        if any(type(flag) is not bool for flag in flags) or sum(flags) != 1 or type(record["finite_nonzero_gradient"]) is not bool:
            raise ValueError("checkpoint attempt outcome flags are invalid")
        if record["attempt"] != attempt or record["step"] != attempt or record["examples_consumed"] != attempt * grad_accum:
            raise ValueError("checkpoint trajectory cursor/attempt mismatch")
        step_tokens = sum(train_set[i % len(train_set)]["n_answer_tokens"] for i in range((attempt - 1) * grad_accum, attempt * grad_accum))
        if record["answer_tokens"] != step_tokens:
            raise ValueError("checkpoint trajectory token count does not match ordered data")
        applied += int(flags[0]); skips += int(flags[1]); safety += int(flags[2])
        finite += int(record["finite_nonzero_gradient"])
        tokens += step_tokens if flags[0] else 0
        expected = {"optimizer_updates_applied": applied, "steps_skipped_by_scaler_total": skips,
                    "steps_skipped_by_safety_check_total": safety, "examples_in_applied_updates": applied * grad_accum,
                    "answer_tokens_in_applied_updates": tokens}
        if any(record[k] != value for k, value in expected.items()):
            raise ValueError("checkpoint trajectory cumulative accounting mismatch")
    expected = {"applied_steps": applied, "skipped_steps": skips, "safety_skipped_steps": safety,
                "finite_nonzero_steps": finite, "applied_examples": applied * grad_accum, "applied_answer_tokens": tokens}
    if any(progress[k] != value for k, value in expected.items()):
        raise ValueError("checkpoint progress does not match trajectory")
    completed_evals = [r["step"] for r in progress["evals"]]
    pending = progress["pending_validation"]
    expected_evals = [s for s in range(1, done + 1) if eval_due(s)]
    if pending is not None and (type(pending) is not int or pending != done or not eval_due(done)):
        raise ValueError("checkpoint pending validation is not scheduled")
    if completed_evals + ([pending] if pending is not None else []) != expected_evals:
        raise ValueError("checkpoint validation schedule/accounting mismatch")


def validate_optimizer_states(payload, *, named_params, optimizer, scheduler, scaler):
    """Validate AdamW, fixed LR schedule and scaler before any live state load."""
    progress = payload["progress"]
    applied = progress["applied_steps"]
    expected_scheduler = scheduler.state_dict()
    expected_scheduler.update(last_epoch=applied, _step_count=applied + 1,
                              _last_lr=[base * fn(applied) for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)])
    if payload["scheduler"] != expected_scheduler:
        raise ValueError("checkpoint scheduler does not match fixed applied-update schedule")
    current_optimizer = optimizer.state_dict()
    saved_optimizer = payload["optimizer"]
    if not isinstance(saved_optimizer, dict) or set(saved_optimizer) != set(current_optimizer):
        raise ValueError("checkpoint optimizer schema mismatch")
    expected_groups = clone_to_cpu(current_optimizer["param_groups"])
    for group, expected_lr in zip(expected_groups, expected_scheduler["_last_lr"]):
        group["lr"] = expected_lr
    if saved_optimizer["param_groups"] != expected_groups:
        raise ValueError("checkpoint optimizer configuration/LR/parameter order mismatch")
    param_ids = [key for group in expected_groups for key in group["params"]]
    params_by_id = dict(zip(param_ids, [p for _, p in named_params]))
    if not isinstance(saved_optimizer["state"], dict) or not set(saved_optimizer["state"]) <= set(params_by_id):
        raise ValueError("checkpoint optimizer state parameter IDs mismatch")
    if applied and not saved_optimizer["state"]:
        raise ValueError("checkpoint applied updates require AdamW state")
    for key, state in saved_optimizer["state"].items():
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("checkpoint AdamW state fields mismatch")
        step = state["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not math.isfinite(float(step)) or float(step) != int(float(step)) or not 0 <= float(step) <= applied:
            raise ValueError("checkpoint AdamW per-parameter step is invalid")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            param = params_by_id[key]
            if not isinstance(value, torch.Tensor) or value.shape != param.shape or value.dtype != param.dtype or not torch.isfinite(value).all():
                raise ValueError("checkpoint AdamW moment shape/dtype/value is invalid")
    expected_scaler = scaler.state_dict()
    if expected_scaler:
        for record in progress["trajectory"]:
            if record["scaler_scale_before"] != expected_scaler["scale"]:
                raise ValueError("checkpoint scaler trajectory scale mismatch")
            if record["step_skipped_by_scaler"]:
                # GradScaler stores scale as float32, including underflow to zero.
                expected_scaler["scale"] = float(torch.tensor(expected_scaler["scale"], dtype=torch.float32) * expected_scaler["backoff_factor"])
                expected_scaler["_growth_tracker"] = 0
            else:
                expected_scaler["_growth_tracker"] += 1
                if expected_scaler["_growth_tracker"] == expected_scaler["growth_interval"]:
                    grown = float(torch.tensor(expected_scaler["scale"], dtype=torch.float32) * expected_scaler["growth_factor"])
                    if math.isfinite(grown):
                        expected_scaler["scale"] = grown
                    expected_scaler["_growth_tracker"] = 0
            if record["scaler_scale_after"] != expected_scaler["scale"]:
                raise ValueError("checkpoint scaler trajectory update mismatch")
    if payload["scaler"] != expected_scaler:
        raise ValueError("checkpoint scaler state does not match committed trajectory")
    count = 0
    for record in progress["trajectory"]:
        expected_used = scheduler.base_lrs[0] * scheduler.lr_lambdas[0](count)
        count += int(record["optimizer_update_applied"])
        expected_next = scheduler.base_lrs[0] * scheduler.lr_lambdas[0](count)
        if record["lr_used"] != round(expected_used, 10) or record["lr"] != round(expected_next, 10):
            raise ValueError("checkpoint trajectory LR does not match fixed schedule")


def save_latest(directory, *, identity, named_params, optimizer, scheduler, scaler, progress, device):
    """One synchronous transaction. A crash exposes the previous or new complete file."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1, "identity": identity,
        "trainable_names": [name for name, _ in named_params],
        "adapter": {name: p for name, p in named_params},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "rng": capture_rng(device),
        "progress": progress,
    }
    payload = clone_to_cpu(payload)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=directory, prefix="latest.", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, directory / "latest.pt")
        temporary = None
        # POSIX directory durability; Windows lacks a portable equivalent. Atomic
        # replace still protects against process death, not guaranteed power loss.
        if os.name != "nt":
            fd = os.open(directory, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_latest(directory, *, identity, named_params, optimizer, scheduler, scaler, max_steps, grad_accum, eval_due, train_set, device):
    payload = torch.load(Path(directory) / "latest.pt", map_location="cpu", weights_only=True)
    expected_keys = {"schema_version", "identity", "trainable_names", "adapter", "optimizer", "scheduler", "scaler", "rng", "progress"}
    if not isinstance(payload, dict) or set(payload) != expected_keys or payload["schema_version"] != 1:
        raise ValueError("unsupported or malformed checkpoint schema")
    if payload["identity"] != identity:
        raise ValueError("checkpoint recipe/data/model/runtime fingerprint mismatch")
    if payload["trainable_names"] != [name for name, _ in named_params] or set(payload["adapter"]) != {n for n, _ in named_params}:
        raise ValueError("checkpoint trainable parameter identity/order mismatch")
    for name, param in named_params:
        saved = payload["adapter"][name]
        if not isinstance(saved, torch.Tensor) or saved.shape != param.shape or saved.dtype != param.dtype:
            raise ValueError(f"checkpoint adapter shape/dtype mismatch: {name}")
    progress = payload["progress"]
    validate_progress(progress, max_steps=max_steps, grad_accum=grad_accum, eval_due=eval_due, train_set=train_set)
    validate_optimizer_states(payload, named_params=named_params, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
    validate_rng(payload["rng"], device)
    # Identity and accounting checks happen before any caller model mutation.
    with torch.no_grad():
        for name, param in named_params:
            param.copy_(payload["adapter"][name])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    return progress, payload["rng"]
