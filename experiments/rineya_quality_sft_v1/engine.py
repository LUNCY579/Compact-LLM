"""One bounded RQSFT SFT arm; no baseline, DEV selection or native authority.

Parent/pair grouping and target compilation are independently validated. Every
row is consumed once; failures and completed cleared-boundary checkpoints remain
durable. MQ numerical primitives are reused, never its permits or v1 validators.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import random
import torch
from safetensors.torch import load_file, save_file

from experiments.model_quality_gpu_v1.losses import chunked_sum
from experiments.model_quality_gpu_v1.native import hidden_states, parameter_digest, select_tail
from experiments.model_quality_gpu_v1.optimizer import CpuAdamW
from experiments.model_quality_gpu_v1.config import rate_at
from experiments.model_quality_gpu_v1.checkpoints import rng_state as audited_rng_state, restore_rng as audited_restore_rng
from . import EXPERIMENT, SOURCE_CORE, SOURCE_WEIGHT_SHA256
from .contract import ARMS, Design, external_output
from .native import buffer_digest, freeze_state_digest, math_attention, _guard


def _helpers():
    from . import integrity
    return integrity


def validate_plan(sealed):
    from .compiler import validate_plan as compiler_validate
    return compiler_validate(sealed)


def targets(row, objective):
    from .compiler import targets as compiler_targets
    return compiler_targets(row, objective)


def select_trainables(model, plan):
    named = select_tail(model, Design(**plan["design"]).trainable_tail)
    if plan["kind"] == "native_plan":
        if len(model.model.layers) != 28 or len(named) != 220 or sum(p.numel() for _, p in named) != 314618880:
            raise ValueError("Exact native blocks8..27 trainable inventory required")
    if any(p.dtype != torch.float32 for _, p in named):
        raise ValueError("Trainables must remain FP32; no precision fallback")
    return named


def verify_binding(model, sealed, plan, binding):
    if binding.get("experiment") != EXPERIMENT or binding.get("compiled_plan_sha256") != sealed["sha256"]:
        raise ValueError("RQSFT experiment/compiled plan identity differs")
    if parameter_digest(model) != binding.get("initial_parameters_sha256"):
        raise ValueError("Every cell/qualification must begin at the freshly loaded exact source")
    if freeze_state_digest(model) != binding.get("initial_frozen_state_sha256"):
        raise ValueError("Initial fixed frozen-state identity differs")
    if plan["kind"] == "native_plan":
        source = plan["source_identity"]
        if (binding.get("source_core") != SOURCE_CORE or binding.get("source_weight_sha256") != SOURCE_WEIGHT_SHA256
                or source.get("core_id") != SOURCE_CORE or source.get("weight_sha256") != SOURCE_WEIGHT_SHA256
                or source.get("parameters") != 596049920
                or binding.get("code_sha256") != plan["code_identity"]["sha256"]):
            raise ValueError("Native source/code identity does not match the reviewed RQSFT plan")
    elif binding.get("fixture_only") is not True:
        raise ValueError("Tiny fixtures require explicit nonnative evidence scope")


def expected_progress(plan, objective, updates):
    if type(updates) is not int or not 0 <= updates <= len(plan["groups"]):
        raise ValueError("Checkpoint cursor outside frozen complete groups")
    indices = [index for group in plan["groups"][:updates] for index in group]
    return {"updates": updates, "next_group": updates, "consumed_rows": len(indices),
            "consumed_row_ids": [plan["train"][index]["id"] for index in indices],
            "processed_tokens": sum(len(plan["train"][index]["ids"]) for index in indices),
            "supervised_tokens": sum(sum(label != -100 for label in targets(plan["train"][index], objective)[1:])
                                     for index in indices)}


def _rng_state(fixture):
    if not fixture:
        return audited_rng_state()
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state().tolist(), "torch_cuda": []}


def _restore_rng(state, fixture):
    if not fixture:
        return audited_restore_rng(state)
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, list) else value
    if state["torch_cuda"]:
        raise ValueError("CPU fixture cannot restore CUDA state")
    random.setstate(tuples(state["python"]))
    torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))


def save_checkpoint(directory, optimizer, binding, progress):
    """RQSFT-native manifest and streamed official parameter/moment/RNG state."""
    integrity = _helpers()
    if any(p.grad is not None for _, p in optimizer.named) or progress["updates"] != optimizer.step_count:
        raise ValueError("Checkpoint requires exact cleared optimizer/progress boundary")
    directory = integrity.new_dir(Path(directory))
    entries = []
    for index, (name, parameter) in enumerate(optimizer.named):
        mean, variance = optimizer.state[name]
        state = {"parameter": parameter.detach().cpu().contiguous(),
                 "exp_avg": mean.contiguous(), "exp_avg_sq": variance.contiguous()}
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
            raise FloatingPointError("Cannot checkpoint nonfinite/non-FP32 official state")
        filename = f"tensor-{index:04d}.safetensors"
        save_file(state, str(directory / filename))
        with (directory / filename).open("rb+") as stream:
            os.fsync(stream.fileno())
        entries.append({"name": name, "file": filename, "shape": list(parameter.shape),
                        "sha256": integrity.file_hash(directory / filename)})
        del state
    body = {"schema": 1, "experiment": EXPERIMENT, "status": "COMPLETE", "binding": binding,
            "progress": progress, "rng": _rng_state(binding.get("fixture_only") is True), "entries": entries,
            "state_schema": "selected_FP32_parameters_CPU_FP32_AdamW_moments_and_all_native_RNG"}
    sealed = integrity.seal(body)
    integrity.write_new(directory / "manifest.json", sealed)
    return sealed


def validate_checkpoint(directory, named, binding):
    integrity = _helpers()
    directory = Path(directory)
    meta = integrity.unseal(integrity.read_json(directory / "manifest.json"))
    if (meta.get("schema") != 1 or meta.get("experiment") != EXPERIMENT
            or meta.get("status") != "COMPLETE" or meta.get("binding") != binding):
        raise ValueError("Foreign, incomplete or stale RQSFT official-state checkpoint")
    entries = meta["entries"]
    if [entry["name"] for entry in entries] != [name for name, _ in named]:
        raise ValueError("Checkpoint trainable-name inventory changed")
    expected = {"manifest.json"} | {f"tensor-{index:04d}.safetensors" for index in range(len(named))}
    if {path.name for path in directory.iterdir()} != expected:
        raise ValueError("Checkpoint file inventory changed")
    for index, (entry, (_, parameter)) in enumerate(zip(entries, named, strict=True)):
        if (entry["file"] != f"tensor-{index:04d}.safetensors" or entry["shape"] != list(parameter.shape)
                or integrity.file_hash(directory / entry["file"]) != entry["sha256"]):
            raise ValueError("Checkpoint file/hash/shape mismatch")
        state = load_file(str(integrity.regular(directory / entry["file"])), device="cpu")
        if set(state) != {"parameter", "exp_avg", "exp_avg_sq"} or any(
                value.dtype != torch.float32 or value.shape != parameter.shape or not torch.isfinite(value).all()
                for value in state.values()) or bool((state["exp_avg_sq"] < 0).any()):
            raise ValueError("Invalid official parameter/moment tensors")
        if meta["progress"].get("updates") == 0 and any(bool(torch.count_nonzero(state[key])) for key in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Cursor0 checkpoint must contain exact zero moments")
    return meta


def _restore_official_state(directory, optimizer, binding):
    integrity = _helpers()
    meta = validate_checkpoint(directory, optimizer.named, binding)
    progress = meta["progress"]
    if type(progress.get("updates")) is not int or progress["updates"] < 0:
        raise ValueError("Checkpoint has invalid optimizer cursor")
    fixture = binding.get("fixture_only") is True
    original = _rng_state(fixture)
    try:
        _restore_rng(meta["rng"], fixture)
    finally:
        _restore_rng(original, fixture)
    with torch.no_grad():
        for entry, (name, parameter) in zip(meta["entries"], optimizer.named, strict=True):
            state = load_file(str(integrity.regular(Path(directory) / entry["file"])), device="cpu")
            parameter.copy_(state["parameter"])
            mean, variance = optimizer.state[name]
            mean.copy_(state["exp_avg"])
            variance.copy_(state["exp_avg_sq"])
    optimizer.step_count = progress["updates"]
    optimizer.zero_grad()
    _restore_rng(meta["rng"], fixture)
    return progress


def restore_checkpoint(directory, optimizer, binding):
    return _restore_official_state(directory, optimizer, binding)


def _official_state_digest(optimizer):
    from .native import _tensor_hash
    values = []
    for name, parameter in optimizer.named:
        mean, variance = optimizer.state[name]
        values.extend((("parameter:" + name, parameter), ("mean:" + name, mean), ("variance:" + name, variance)))
    return _tensor_hash(values)


def checkpoint_roundtrip_zero(model, named, design, directory, binding):
    """Restore deliberately perturbed mechanical sentinels; no optimizer step.

    Reversible selected-parameter/moment/cursor/RNG writes challenge restoration.
    They are not training updates or a learned candidate. A failed challenge
    remains FAILED even if independent cleanup successfully restores saved state.
    """
    integrity = _helpers()
    before, frozen, buffers = parameter_digest(model), freeze_state_digest(model), buffer_digest(model)
    fixture = binding.get("fixture_only") is True
    rng_before = _rng_state(fixture)
    optimizer = CpuAdamW(named, design)
    state_before = _official_state_digest(optimizer)
    progress = {"updates": 0, "next_group": 0, "consumed_rows": 0, "consumed_row_ids": [],
                "processed_tokens": 0, "supervised_tokens": 0}
    manifest = save_checkpoint(directory, optimizer, binding, progress)
    try:
        with torch.no_grad():
            for name, parameter in optimizer.named:
                scalar = parameter.reshape(-1)[0]
                old_value = float(scalar)
                scalar.fill_(-old_value if old_value != 0 else 1.)
                optimizer.state[name][0].reshape(-1)[0] = 1.
                optimizer.state[name][1].reshape(-1)[0] = 2.
                if float(parameter.reshape(-1)[0]) == old_value:
                    raise RuntimeError("Selected parameter sentinel did not alter the actual parameter")
        optimizer.step_count = 17
        random.random()
        torch.rand(1)
        cuda_rng_perturbed = not fixture and next(model.parameters()).device.type == "cuda"
        if cuda_rng_perturbed:
            torch.rand(1, device=next(model.parameters()).device)
        if _official_state_digest(optimizer) == state_before or integrity.digest(_rng_state(fixture)) == integrity.digest(rng_before):
            raise RuntimeError("Mechanical restoration sentinels failed to perturb official state/RNG")
        restored = restore_checkpoint(directory, optimizer, binding)
        if (restored != progress or optimizer.step_count != 0 or parameter_digest(model) != before
                or freeze_state_digest(model) != frozen or buffer_digest(model) != buffers
                or _official_state_digest(optimizer) != state_before
                or integrity.digest(_rng_state(fixture)) != integrity.digest(rng_before)
                or any(bool(torch.count_nonzero(value)) for state in optimizer.state.values() for value in state)):
            raise RuntimeError("Official0-update sentinel recovery failed to restore state/cursor/RNG")
    except BaseException as error:
        # Use the validated physical checkpoint through an independent internal
        # path; do not let a no-op public restore make cleanup vacuous too.
        try:
            _restore_official_state(directory, optimizer, binding)
            if (parameter_digest(model) != before or _official_state_digest(optimizer) != state_before
                    or integrity.digest(_rng_state(fixture)) != integrity.digest(rng_before)):
                raise RuntimeError("Sentinel cleanup did not restore initial official state")
            error.add_note("Independent saved-checkpoint cleanup restored in-memory state; qualification still FAILED")
        except BaseException as cleanup_error:
            error.add_note("Sentinel cleanup failed; discard the in-memory model: " + str(cleanup_error))
        raise
    del optimizer
    return {"recovery_passed": True, "training_updates": 0, "optimizer_step_calls": 0,
            "manifest_sha256": manifest["sha256"], "checkpoint": str(Path(directory).resolve()),
            "parameters_buffers_rng_cursor_and_zero_moments_restored": True,
            "sentinel_restore_verified": True, "selected_official_state_sha256": state_before,
            "parameter_sentinel_writes": len(named), "moment_sentinel_writes": 2 * len(named),
            "sentinel_scope": "every_selected_parameter_and_both_moment_tensors",
            "optimizer_cursor_sentinel": 17, "python_cpu_rng_sentinels": True,
            "cuda_rng_sentinel": cuda_rng_perturbed,
            "entries": manifest["entries"],
            "native_resume_authority": False}


class MeasuredCpuAdamW(CpuAdamW):
    """Audited AdamW equations with a bounded per-tensor actual-update measure.

    No full-tail before snapshot is allocated. Gradients are validated globally
    before any mutation; moments/parameter order and all numerical operations
    match CpuAdamW. A partial step invalidates the uncheckpointed model.
    """
    @torch.no_grad()
    def step(self, lr, guard=None):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("Invalid learning rate")
        for name, parameter in self.named:
            if parameter.grad is None or parameter.grad.is_sparse or not torch.isfinite(parameter.grad).all():
                raise ValueError("Missing/sparse/nonfinite gradient: " + name)
        d, step, update_squared = self.design, self.step_count + 1, 0.
        for name, parameter in self.named:
            _guard(guard, "optimizer_tensor_before", tensor_name=name,
                   largest_temporary_parameter_elements=parameter.numel())
            gradient = parameter.grad.detach().to("cpu", copy=True)
            weight = parameter.detach().to("cpu", copy=True)
            old_weight = weight.clone()
            mean, variance = self.state[name]
            weight.mul_(1.0 - lr * d.weight_decay)
            mean.lerp_(gradient, 1.0 - d.beta1)
            variance.mul_(d.beta2).addcmul_(gradient, gradient, value=1.0 - d.beta2)
            denominator = (variance.sqrt() / math.sqrt(1.0 - d.beta2 ** step)).add_(d.epsilon)
            weight.addcdiv_(mean, denominator, value=-lr / (1.0 - d.beta1 ** step))
            if not torch.isfinite(weight).all():
                raise FloatingPointError("Nonfinite AdamW update; discard uncheckpointed state")
            current, previous = weight.reshape(-1), old_weight.reshape(-1)
            delta = None
            for offset in range(0, current.numel(), 262144):
                delta = current[offset:offset + 262144].double() - previous[offset:offset + 262144].double()
                update_squared += float(delta.square().sum())
            parameter.copy_(weight)
            del gradient, weight, old_weight, denominator, current, previous, delta
        self.step_count = step
        return math.sqrt(update_squared)


def train_cell(model, sealedplan, arm, out, binding, guard=None, resume=None, tokenizer=None, eos_ids=None):
    """Train one already-authorized cell; evaluation is exclusively the root caller."""
    integrity = _helpers()
    plan = validate_plan(sealedplan)
    if plan["kind"] == "native_plan" and guard is None:
        raise PermissionError("Native training requires its independently reviewed owned resource guard")
    verify_binding(model, sealedplan, plan, binding)
    if arm not in ARMS:
        raise ValueError("Only whole_1e-5 and response_1e-5 are registered")
    d = Design(**plan["design"])
    if len(plan["groups"]) > d.max_updates or plan["processed_tokens"] > d.processed_budget:
        raise ValueError("Whole-bank update/token ceilings exceeded")
    objective, peak = ARMS[arm]
    out = Path(out)
    if plan["kind"] == "native_plan":
        out = external_output(out)
    if not out.is_dir():
        raise ValueError("Caller must provide its fresh reviewed arm directory")
    from .native import source_custody
    source_before = source_custody(model._rqsft_source) if plan["kind"] == "native_plan" else None
    named = select_trainables(model, plan)
    frozen_before, buffers_before = freeze_state_digest(model), buffer_digest(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    _guard(guard, "optimizer_before", cpu_moment_bytes=sum(p.numel() * 8 for _, p in named))
    optimizer = MeasuredCpuAdamW(named, d)
    _guard(guard, "optimizer_after")
    state_binding = dict(binding, arm=arm, objective=objective)
    progress = expected_progress(plan, objective, 0)
    if resume is not None:
        metadata = validate_checkpoint(resume, named, state_binding)
        if metadata["progress"] != expected_progress(plan, objective, metadata["progress"]["updates"]):
            raise ValueError("Resume checkpoint does not describe the exact intact parent groups")
        progress = restore_checkpoint(resume, optimizer, state_binding)
    trace, checkpoint, unsafe_optimizer_step = [], None, False
    device = next(model.parameters()).device
    raw_path, steps_path = out / "raw-steps.jsonl", out / "steps.jsonl"
    def emit(stream, record):
        stream.write(integrity.canonical(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        with raw_path.open("xb") as raw, steps_path.open("xb") as completed, math_attention():
            for index in range(progress["next_group"], len(plan["groups"])):
                group = plan["groups"][index]
                processed = sum(len(plan["train"][row_index]["ids"]) for row_index in group)
                if processed > d.update_tokens or len(group) < 1:
                    raise ValueError("Atomic parent group exceeds fixed per-update token ceiling")
                denominator = sum(sum(label != -100 for label in targets(plan["train"][row_index], objective)[1:]) for row_index in group)
                if denominator <= 0:
                    raise ValueError("Update has no captured final-answer/whole targets")
                emit(raw, {"event": "update_started", "next_group": index, "row_ids": [plan["train"][i]["id"] for i in group],
                           "processed_tokens": processed, "supervised_tokens": denominator})
                _guard(guard, "update_before")
                model.train()
                optimizer.zero_grad()
                loss_value = 0.
                for row_index in group:
                    row = plan["train"][row_index]
                    ids = torch.tensor([row["ids"]], dtype=torch.long, device=device)
                    labels = torch.tensor([targets(row, objective)], dtype=torch.long, device=device)
                    _guard(guard, "training_forward_before", input_tokens=len(row["ids"]), forward_calls=1)
                    try:
                        hidden = hidden_states(model, ids)
                    finally:
                        _guard(guard, "training_forward_after")
                    loss, count = chunked_sum(hidden, model.lm_head.weight, labels, d.ce_chunk)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite SFT objective")
                    _guard(guard, "backward_before")
                    (loss / denominator).backward()
                    _guard(guard, "backward_after")
                    loss_value += float(loss.detach()) / denominator
                    emit(raw, {"event": "row_backward_completed", "group": index, "id": row["id"],
                               "processed_tokens": len(row["ids"]), "supervised_tokens": count,
                               "loss_sum": float(loss.detach())})
                    del hidden, loss
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError("Frozen parameter acquired a gradient")
                if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in named):
                    raise RuntimeError("Missing/nonfinite selected gradient")
                preclip = torch.nn.utils.clip_grad_norm_([p for _, p in named], d.clip_norm, error_if_nonfinite=True)
                next_progress = expected_progress(plan, objective, index + 1)
                lr = rate_at(peak, next_progress["processed_tokens"], plan["processed_tokens"], d)
                _guard(guard, "optimizer_step_before")
                unsafe_optimizer_step = True
                update_l2 = optimizer.step(lr, guard=guard)
                unsafe_optimizer_step = False
                optimizer.zero_grad()
                _guard(guard, "optimizer_step_after")
                if freeze_state_digest(model) != frozen_before or buffer_digest(model) != buffers_before:
                    raise RuntimeError("Frozen parameter/buffer changed during update")
                progress = next_progress
                record = dict(progress, arm=arm, objective=objective, learning_rate=lr,
                              objective_ce=loss_value, preclip_gradient_l2=float(preclip), update_l2=update_l2,
                              row_ids=[plan["train"][i]["id"] for i in group],
                              frozen_state_unchanged=True, buffers_unchanged=True,
                              logical_forward_calls=len(group), processed_input_tokens=processed)
                emit(completed, record)
                emit(raw, {"event": "update_completed", **record})
                trace.append(record)
                if progress["updates"] % d.checkpoint_every == 0 or index + 1 == len(plan["groups"]):
                    _guard(guard, "checkpoint_before")
                    candidate = out / f"checkpoint-{progress['updates']:05d}"
                    save_checkpoint(candidate, optimizer, state_binding, progress)
                    checkpoint = candidate
                    _guard(guard, "checkpoint_after")
        if (progress["consumed_rows"] != len(plan["train"]) or progress["processed_tokens"] != plan["processed_tokens"]
                or len(set(progress["consumed_row_ids"])) != len(plan["train"])):
            raise RuntimeError("Whole admitted bank was not consumed exactly once")
        if freeze_state_digest(model) != frozen_before or buffer_digest(model) != buffers_before:
            raise RuntimeError("Frozen state changed at final boundary")
        source_after = source_custody(model._rqsft_source) if source_before is not None else None
        if source_after != source_before:
            raise RuntimeError("Immutable source custody changed during native SFT")
        model.eval()
        result = {"experiment": EXPERIMENT, "arm": arm, "binding": binding,
                  "fixture_only": plan["kind"] == "fixture", "status": "TRAINING_COMPLETE_DEV_NOT_EVALUATED",
                  "progress": progress, "trace": trace, "last_checkpoint": str(checkpoint) if checkpoint else None,
                  "initial_parameters_sha256": binding["initial_parameters_sha256"],
                  "final_parameters_sha256": parameter_digest(model), "frozen_state_sha256": frozen_before,
                  "frozen_state_unchanged": True, "buffers_unchanged": True,
                  "steps_sha256": integrity.file_hash(steps_path), "raw_steps_sha256": integrity.file_hash(raw_path),
                  "dev_evaluated": False, "efficacy_established": False, "release_permission": False}
        result["source_before"], result["source_after"] = source_before, source_after
        sealed = integrity.seal(result)
        integrity.write_new(out / "train-result.json", sealed)
        return sealed
    except BaseException as error:
        optimizer.zero_grad()
        model.eval()
        failure = {"experiment": EXPERIMENT, "status": "FAILED", "arm": arm, "binding": binding,
                   "progress_last_completed": progress, "trace": trace,
                   "last_completed_checkpoint": str(checkpoint) if checkpoint and (checkpoint / "manifest.json").exists() else None,
                   "uncheckpointed_model_invalid": True, "failure_during_optimizer_step": unsafe_optimizer_step,
                   "error_type": type(error).__name__, "reason": str(error), "automatic_retry": False}
        try:
            source_after = source_custody(model._rqsft_source) if source_before is not None else None
            failure.update(source_before=source_before, source_after=source_after,
                           source_custody_unchanged=source_before == source_after)
            if source_after != source_before:
                error.add_note("Immutable source custody changed during failed native SFT")
        except BaseException as custody_error:
            error.add_note("Failed to recheck source custody: " + str(custody_error))
        integrity.write_new(out / "ENGINE_FAILED.json", integrity.seal(failure))
        raise
    finally:
        optimizer.zero_grad()
