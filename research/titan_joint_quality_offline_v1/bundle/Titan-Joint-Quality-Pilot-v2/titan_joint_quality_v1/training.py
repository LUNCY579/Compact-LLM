"""Common-denominator loss, pure forecasts and a guarded tiny CPU fixture.

There is no native loader or dispatch path. The new data/objective/source combination
is unqualified even though the fixture adapter reuses two exact MQ primitives.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import random
import sys
import types

from .common import CONTRACT, digest

MQ_SOURCE = Path("C:/Users/clclc/BusinessProject/ChatBubble-rineya-reasoning-posttraining-v1/"
                 "experiments/model_quality_gpu_v1")
MQ_HASHES = {"optimizer.py": "ae45207132fe8d98c763f571f20e1a94bfdb521e818692e7736d7a3b2648c6d1",
             "config.py": "de8c268886978050634e0e8f18691439c024c5d8c31a6132cdc4913b3a367834"}


def _positive_integer(value, name, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a strict {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def run_native(*args, **kwargs):
    """Deliberately refuse every native/real-model request, including forged flags."""
    raise PermissionError("native training/evaluation is closed; this objective and source are unqualified")


def load_cpu_fixture_mq_adapter(*, fixture_only):
    """Execute only the hash-pinned optimizer/config bytes in a named fixture adapter.

    No MQ runner, six-cell design validator, model importer, permit or native
    authorization is imported. Files are read-only, and execution uses the same
    bytes that were hashed rather than reopening a mutable path after validation.
    """
    if fixture_only is not True:
        return run_native()
    source = {}
    for filename, expected in MQ_HASHES.items():
        data = (MQ_SOURCE / filename).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"pinned MQ fixture dependency changed: {filename}")
        source[filename] = data
    package_name = "titan_joint_quality_cpu_fixture_mq_adapter"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MQ_SOURCE)]
    sys.modules[package_name] = package
    modules = {}
    for filename in ("config.py", "optimizer.py"):
        name = package_name + "." + filename[:-3]
        module = types.ModuleType(name)
        module.__file__ = str(MQ_SOURCE / filename)
        module.__package__ = package_name
        sys.modules[name] = module
        source_text = source[filename].decode("utf-8")
        if filename == "config.py":
            tree = ast.parse(source_text, filename=module.__file__)
            # Reuse the exact parsed primitive definitions, not ARMS or the
            # inherited native six-cell admission/25k design validator.
            selected = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                        or isinstance(node, (ast.ClassDef, ast.FunctionDef))
                        and node.name in {"Design", "rate_at"}]
            tree = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
            exec(compile(tree, module.__file__, "exec"), module.__dict__)
        else:
            exec(compile(source_text, module.__file__, "exec"), module.__dict__)
        modules[filename] = module
    return {"CpuAdamW": modules["optimizer.py"].CpuAdamW,
            "Design": modules["config.py"].Design,
            "rate_at": modules["config.py"].rate_at,
            "bindings": {"class": "READ_ONLY_CPU_FIXTURE_PRIMITIVES_ONLY",
                         "source": str(MQ_SOURCE), "sha256": dict(MQ_HASHES),
                         "config_definitions_reused": ["Design", "rate_at"],
                         "legacy_native_design_or_permit_imported": False,
                         "new_objective_native_qualified": False}}


def common_denominator_loss(logits, token_ids, label_mask, planned_denominator, *, fixture_only=True):
    """Sum correctly shifted labeled CE, divided by the COMMON planned denominator.

    Labels refer to token positions, so token position i is predicted by logit
    position i-1. Non-R rows in the control use an all-false mask but retain their
    complete forward/input work and the joint arm's planned denominator.
    """
    if fixture_only is not True:
        return run_native()
    import torch
    _positive_integer(planned_denominator, "planned_denominator")
    if not isinstance(logits, torch.Tensor) or logits.device.type != "cpu" or logits.ndim != 2:
        raise ValueError("loss fixture requires a CPU [sequence,vocabulary] tensor")
    if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise ValueError("finite FP32 fixture logits required")
    if not isinstance(token_ids, (list, tuple)) or len(token_ids) != logits.shape[0]:
        raise ValueError("token IDs must bind every complete input position")
    if len(token_ids) < 3 or len(token_ids) > CONTRACT["max_row_tokens"]:
        raise ValueError("complete fixture row length is outside the contract")
    for token in token_ids:
        if type(token) is not int or not 0 <= token < logits.shape[1]:
            raise ValueError("token ID is not a strict in-vocabulary integer")
    if (not isinstance(label_mask, (list, tuple)) or len(label_mask) != len(token_ids)
            or any(type(value) is not bool for value in label_mask) or label_mask[0]):
        raise ValueError("strict boolean masks must bind positions; position zero cannot be labeled")
    if sum(label_mask) > planned_denominator:
        raise ValueError("row labels exceed the common group's planned denominator")
    targets = torch.tensor(token_ids[1:], dtype=torch.long, device="cpu")
    mask = torch.tensor(label_mask[1:], dtype=torch.bool, device="cpu")
    losses = torch.nn.functional.cross_entropy(logits[:-1], targets, reduction="none")
    return losses.masked_select(mask).sum() / planned_denominator


def validate_resource_forecast(measurements, operation_counts, *, exposure_plan=None):
    """Pure prospective full-cost check; missing measured upper bounds BLOCK.

    Five evaluation configurations, two training arms and two disposable qualifier
    loads are accounted for. Failure work, save/restore, hashes and finalization
    are charged separately. A fitting forecast never grants a native attempt.
    """
    receipt = {"status": "BLOCKED_RESOURCE_FORECAST", "native_authorized": False,
               "training_authorized": False, "source_objective_qualified": False,
               "deadline_seconds": CONTRACT["native_seconds_cap"],
               "finalization_reserve_seconds": CONTRACT["finalization_reserve_seconds"],
               "evaluation_configurations": 5, "training_arms": 2,
               "missing_or_invalid": [], "cost_components": {}}
    count_keys = ("evaluation_configurations", "evaluation_rows_per_configuration", "max_forward_calls_per_evaluation",
                  "processed_tokens_per_arm", "qualification_forward_calls",
                  "qualification_update_tokens", "failed_forward_calls", "failed_update_tokens",
                  "checkpoint_saves", "checkpoint_restores", "source_hashes", "failed_attempts")
    rate_keys = ("model_load_seconds", "max_forward_seconds", "training_seconds_per_token",
                 "failed_forward_seconds", "checkpoint_save_seconds", "checkpoint_restore_seconds",
                 "source_hash_seconds", "failed_cleanup_seconds", "artifact_finalization_seconds")
    if not isinstance(measurements, dict) or not isinstance(operation_counts, dict):
        receipt["missing_or_invalid"].append("explicit measurement and operation-count dictionaries required")
        return receipt
    for key in count_keys:
        value = operation_counts.get(key)
        if type(value) is not int or value <= 0:
            receipt["missing_or_invalid"].append(key)
    for key in rate_keys:
        value = measurements.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value <= 0):
            receipt["missing_or_invalid"].append(key)
    if measurements.get("complete_measured_upper_bounds") is not True:
        receipt["missing_or_invalid"].append("complete_measured_upper_bounds attestation")
    if receipt["missing_or_invalid"]:
        return receipt
    if (operation_counts["evaluation_configurations"] != 5 or
            operation_counts["evaluation_rows_per_configuration"] != 384 or
            operation_counts["max_forward_calls_per_evaluation"] != CONTRACT["max_new_tokens"]):
        receipt["missing_or_invalid"].append("complete evaluation requires exactly five configurations, 384 rows each and 48 forward cap")
        return receipt
    if exposure_plan is None:
        if operation_counts["processed_tokens_per_arm"] != CONTRACT["processed_tokens_cap_per_arm"]:
            receipt["missing_or_invalid"].append("an unbound training forecast must use the full 54172-token cap")
            return receipt
        receipt["training_exposure_basis"] = "CONSERVATIVE_FULL_FROZEN_TOKEN_CAP"
    else:
        try:
            if not isinstance(exposure_plan, dict):
                raise ValueError("exposure_plan dictionary required")
            unsigned = dict(exposure_plan)
            recorded = unsigned.pop("plan_sha256", None)
            if recorded != digest(unsigned) or exposure_plan.get("status") != "PROSPECTIVE_OFFLINE_EXPOSURE_PLAN":
                raise ValueError("complete exposure plan hash/status does not match")
            groups = exposure_plan.get("groups")
            if not isinstance(groups, list) or not groups:
                raise ValueError("complete exposure plan groups required")
            actual_tokens = sum(group["processed_tokens"] for group in groups)
            actual_rows = sum(len(group["rows"]) for group in groups)
            actual_updates = len(groups)
            if (exposure_plan["processed_tokens_per_arm"] != actual_tokens or
                    exposure_plan["selected_rows"] != actual_rows or exposure_plan["updates"] != actual_updates or
                    not 0 < actual_tokens <= CONTRACT["processed_tokens_cap_per_arm"]):
                raise ValueError("exposure plan row/update/position census differs")
            for key, actual in (("processed_tokens_per_arm", actual_tokens),
                                ("training_rows_per_arm", actual_rows), ("updates_per_arm", actual_updates)):
                _positive_integer(operation_counts.get(key), key)
                if operation_counts[key] != actual:
                    raise ValueError(f"declared {key} differs from exact frozen exposure plan")
            receipt["training_exposure_basis"] = "EXACT_FROZEN_EXPOSURE_PLAN"
            receipt["exposure_plan_sha256"] = recorded
            receipt["training_rows_per_arm"] = actual_rows
            receipt["updates_per_arm"] = actual_updates
        except (ValueError, TypeError, KeyError) as error:
            receipt["missing_or_invalid"].append(str(error))
            return receipt
    c, m = operation_counts, measurements
    costs = {
        "five_evaluation_two_training_two_qualification_loads": 9 * m["model_load_seconds"],
        "all_five_configuration_evaluation_forwards": 5 * c["evaluation_rows_per_configuration"]
            * c["max_forward_calls_per_evaluation"] * m["max_forward_seconds"],
        "pair_qualification_forwards": c["qualification_forward_calls"] * m["max_forward_seconds"],
        "both_training_arms_and_qualification_update_tokens":
            (2 * c["processed_tokens_per_arm"] + c["qualification_update_tokens"])
            * m["training_seconds_per_token"],
        "failed_forward_work": c["failed_forward_calls"] * m["failed_forward_seconds"],
        "failed_update_work": c["failed_update_tokens"] * m["training_seconds_per_token"],
        "checkpoint_saves": c["checkpoint_saves"] * m["checkpoint_save_seconds"],
        "checkpoint_restores": c["checkpoint_restores"] * m["checkpoint_restore_seconds"],
        "source_hashes": c["source_hashes"] * m["source_hash_seconds"],
        "failed_attempt_cleanup": c["failed_attempts"] * m["failed_cleanup_seconds"],
        "seven_configuration_training_artifact_finalizations": 7 * m["artifact_finalization_seconds"],
    }
    body = sum(costs.values())
    total = body + CONTRACT["finalization_reserve_seconds"]
    if not math.isfinite(total):
        receipt["missing_or_invalid"].append("nonfinite complete forecast")
        return receipt
    fits = total <= CONTRACT["native_seconds_cap"]
    receipt.update({"status": "PROSPECTIVE_COST_FITS_ONLY" if fits else "BLOCKED_COMPLETE_COST_EXCEEDS_CAP",
                    "cost_components": costs, "body_seconds": body, "complete_seconds": total,
                    "fits_six_hour_cap": fits, "measurements": measurements,
                    "operation_counts": operation_counts,
                    "qualification_cost_is_separate_from_training": True,
                    "unknown_partial_cost_policy": "missing bound blocks, never imputed zero"})
    return receipt


def _tensor_digest(tensors):
    result = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous().clone()
        result.update(name.encode("utf-8"))
        result.update(str(value.dtype).encode("ascii"))
        result.update(str(tuple(value.shape)).encode("ascii"))
        # NumPy is unnecessary: the small fixture's byte storage is directly readable.
        result.update(bytes(value.untyped_storage()))
    return result.hexdigest()


def _optimizer_digest(optimizer):
    tensors = {}
    for name, (first, second) in optimizer.state.items():
        tensors[name + ".first_moment"] = first
        tensors[name + ".second_moment"] = second
    return digest({"step_count": optimizer.step_count, "tensor_sha256": _tensor_digest(tensors)})


def cpu_fixture_checkpoint_digest(checkpoint):
    """Bind all full-checkpoint fields without writing a tensor or pickle file."""
    values = dict(checkpoint)
    values["model_state"] = _tensor_digest(checkpoint["model_state"])
    moments = {}
    for name, (first, second) in checkpoint["optimizer_state"].items():
        moments[name + ".first_moment"] = first
        moments[name + ".second_moment"] = second
    values["optimizer_state"] = _tensor_digest(moments)
    values["torch_cpu_rng_state"] = hashlib.sha256(
        bytes(checkpoint["torch_cpu_rng_state"].untyped_storage())).hexdigest()
    return digest(values)


def make_cpu_fixture_checkpoint(model, optimizer, *, cursor, completed_tokens, plan_sha256):
    """Full in-memory CPU fixture boundary, after gradients have been cleared."""
    import torch
    _positive_integer(cursor, "cursor", allow_zero=True)
    _positive_integer(completed_tokens, "completed_tokens", allow_zero=True)
    named = list(model.named_parameters())
    if (sum(value.numel() for value in model.state_dict().values()) >= 1_000_000 or
            any(value.device.type != "cpu" for value in model.state_dict().values())):
        raise ValueError("only a tiny CPU fixture model may checkpoint")
    if any(parameter.grad is not None for _, parameter in named):
        raise ValueError("checkpoint requires cleared gradients on every parameter")
    return {"class": "FULL_CPU_FIXTURE_CHECKPOINT_NOT_NATIVE", "cursor": cursor,
            "completed_tokens": completed_tokens, "plan_sha256": plan_sha256,
            "model_state": {name: value.detach().clone() for name, value in model.state_dict().items()},
            "optimizer_state": {name: (first.clone(), second.clone())
                                for name, (first, second) in optimizer.state.items()},
            "optimizer_step_count": optimizer.step_count,
            "optimizer_design": optimizer.design.as_dict(),
            "requires_grad": {name: value.requires_grad for name, value in named},
            "module_modes": {name: module.training for name, module in model.named_modules()},
            "python_rng_state": random.getstate(), "torch_cpu_rng_state": torch.get_rng_state().clone(),
            "gradients": "NONE", "native_authorized": False}


def restore_cpu_fixture_checkpoint(model, optimizer, checkpoint, *, expected_plan_sha256):
    """Strict full-state restore; no parameter-only resume can pass this adapter."""
    import torch
    required = {"class", "cursor", "completed_tokens", "plan_sha256", "model_state",
                "optimizer_state", "optimizer_step_count", "optimizer_design", "requires_grad",
                "module_modes", "python_rng_state", "torch_cpu_rng_state", "gradients", "native_authorized"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("full fixture checkpoint is incomplete")
    if (checkpoint["class"] != "FULL_CPU_FIXTURE_CHECKPOINT_NOT_NATIVE" or
            checkpoint["plan_sha256"] != expected_plan_sha256 or
            checkpoint["native_authorized"] is not False or checkpoint["gradients"] != "NONE"):
        raise ValueError("checkpoint scope or plan differs")
    named = dict(model.named_parameters())
    if sum(value.numel() for value in model.state_dict().values()) >= 1_000_000 or any(
            value.device.type != "cpu" for value in model.state_dict().values()):
        raise ValueError("only a tiny CPU fixture model may restore")
    if any(value.grad is not None for value in named.values()):
        raise ValueError("restore target has uncleared gradients")
    if checkpoint["requires_grad"] != {name: value.requires_grad for name, value in named.items()}:
        raise ValueError("frozen/trainable inventory changed")
    if checkpoint["optimizer_design"] != optimizer.design.as_dict():
        raise ValueError("optimizer recipe differs")
    if set(checkpoint["optimizer_state"]) != set(optimizer.state):
        raise ValueError("optimizer inventory differs")
    if set(checkpoint["module_modes"]) != {name for name, _ in model.named_modules()}:
        raise ValueError("module-mode inventory differs")
    for name, moments in checkpoint["optimizer_state"].items():
        if (not isinstance(moments, (list, tuple)) or len(moments) != 2 or
                any(value.device.type != "cpu" or value.dtype != torch.float32 or
                    value.shape != optimizer.state[name][index].shape or not torch.isfinite(value).all()
                    for index, value in enumerate(moments)) or (moments[1] < 0).any()):
            raise ValueError("invalid full optimizer moments")
    current_state = model.state_dict()
    if set(checkpoint["model_state"]) != set(current_state):
        raise ValueError("complete model/buffer state inventory differs")
    for name, value in checkpoint["model_state"].items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or
                value.dtype != current_state[name].dtype or value.shape != current_state[name].shape or
                (value.is_floating_point() and not torch.isfinite(value).all())):
            raise ValueError("invalid model parameter/buffer state")
    for mode in checkpoint["module_modes"].values():
        if type(mode) is not bool:
            raise ValueError("module modes must be strict booleans")
    _positive_integer(checkpoint["cursor"], "checkpoint cursor", allow_zero=True)
    _positive_integer(checkpoint["completed_tokens"], "checkpoint token position", allow_zero=True)
    _positive_integer(checkpoint["optimizer_step_count"], "optimizer step count", allow_zero=True)
    if checkpoint["optimizer_step_count"] != checkpoint["cursor"]:
        raise ValueError("completed exposure/Adam positions differ")
    # Validate RNG restoration before mutating model or optimizer state.
    temporary_python = random.Random()
    temporary_python.setstate(checkpoint["python_rng_state"])
    temporary_torch = torch.Generator(device="cpu")
    temporary_torch.set_state(checkpoint["torch_cpu_rng_state"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    for name, moments in checkpoint["optimizer_state"].items():
        optimizer.state[name] = (moments[0].clone(), moments[1].clone())
    optimizer.step_count = checkpoint["optimizer_step_count"]
    for name, module in model.named_modules():
        module.training = checkpoint["module_modes"][name]
    random.setstate(checkpoint["python_rng_state"])
    torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
    return {"cursor": checkpoint["cursor"], "completed_tokens": checkpoint["completed_tokens"]}


def _validate_fixture_plan(plan):
    if not isinstance(plan, dict) or plan.get("status") != "PROSPECTIVE_OFFLINE_EXPOSURE_PLAN":
        raise ValueError("a complete prospective exposure plan is required")
    expected = dict(plan)
    recorded = expected.pop("plan_sha256", None)
    if recorded != digest(expected):
        raise ValueError("exposure plan binding changed")
    groups = plan.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("nonempty whole-row update groups required")
    token_max = 1
    seen = set()
    for index, group in enumerate(groups):
        if group.get("group_index") != index or not group.get("rows"):
            raise ValueError("group identity/order changed")
        tokens = denominator = reasoning = 0
        for row in group["rows"]:
            if row["case_id"] in seen:
                raise ValueError("repeated training row")
            seen.add(row["case_id"])
            ids = row["ids"]
            if any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("fixture token IDs must be strict nonnegative integers")
            token_max = max(token_max, max(ids))
            tokens += len(ids)
            denominator += sum(row["label_mask"])
            if row["role"] == "reasoning":
                reasoning += sum(row["label_mask"])
        if (tokens != group["processed_tokens"] or tokens > CONTRACT["max_update_tokens"] or
                denominator != group["planned_denominator"] or reasoning <= 0):
            raise ValueError("common group census, denominator or R coverage changed")
    if token_max >= 128:
        raise ValueError("real tokenizer IDs are refused: the tiny fixture vocabulary is <128")
    return token_max + 1


def verify_masked_reasoning_gradient(plan, *, fixture_only=True):
    """Loss-level fixture: changing masked non-R targets cannot change R gradients."""
    if fixture_only is not True:
        return run_native()
    vocabulary = _validate_fixture_plan(plan)
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("CPU fixture refuses an initialized CUDA context")
    groups_checked = 0
    for group in plan["groups"]:
        generator = torch.Generator(device="cpu").manual_seed(CONTRACT["seed"] + group["group_index"])
        originals, modified, first_losses, second_losses = [], [], [], []
        for row in group["rows"]:
            logits = torch.randn(len(row["ids"]), vocabulary, generator=generator,
                                 dtype=torch.float32, device="cpu").requires_grad_()
            alternative = logits.detach().clone().requires_grad_()
            mask = row["label_mask"] if row["role"] == "reasoning" else [False] * len(row["ids"])
            changed_targets = list(row["ids"])
            if row["role"] != "reasoning":
                changed_targets = [(token + 1) % vocabulary for token in changed_targets]
            first_losses.append(common_denominator_loss(logits, row["ids"], mask, group["planned_denominator"]))
            second_losses.append(common_denominator_loss(alternative, changed_targets, mask,
                                                         group["planned_denominator"]))
            originals.append(logits)
            modified.append(alternative)
        first_gradients = torch.autograd.grad(sum(first_losses), originals)
        second_gradients = torch.autograd.grad(sum(second_losses), modified)
        for row, first, second in zip(group["rows"], first_gradients, second_gradients):
            if not torch.equal(first, second):
                raise ValueError("masked non-R targets changed loss-level gradients")
            if row["role"] != "reasoning" and torch.count_nonzero(first).item() != 0:
                raise ValueError("masked non-R row contributed a gradient")
        groups_checked += 1
    return {"status": "CPU_FIXTURE_LOSS_IDENTITY_ONLY", "groups_checked": groups_checked,
            "reasoning_gradients_invariant": True, "masked_nonreasoning_gradients_zero": True,
            "common_denominators": plan["planned_denominators"],
            "capability_evidence": False, "native_authorized": False}


def run_cpu_fixture(plan, arm="joint", *, fixture_only=True, checkpoint_after=None):
    """Show accounting and full resume mechanics on a newly initialized tiny model.

    This function cannot accept a model, checkpoint path, CUDA device or native
    source. Its output is a software-fixture receipt, never Titan evidence.
    """
    if fixture_only is not True:
        return run_native()
    if arm not in {"joint", "reasoning_only"}:
        raise ValueError("fixture arm must be joint or reasoning_only")
    vocabulary = _validate_fixture_plan(plan)
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("CPU fixture refuses an already initialized CUDA context")
    if checkpoint_after is not None:
        _positive_integer(checkpoint_after, "checkpoint_after")
        if checkpoint_after >= len(plan["groups"]):
            raise ValueError("resume fixture needs a nonterminal complete boundary")
    adapter = load_cpu_fixture_mq_adapter(fixture_only=True)
    caller_python, caller_torch = random.getstate(), torch.get_rng_state().clone()
    total_tokens = plan["processed_tokens_per_arm"]
    design = adapter["Design"](processed_budget=CONTRACT["processed_tokens_cap_per_arm"],
        max_sequence=CONTRACT["max_row_tokens"], update_tokens=CONTRACT["max_update_tokens"],
        beta1=CONTRACT["betas"][0], beta2=CONTRACT["betas"][1],
        epsilon=CONTRACT["epsilon"], weight_decay=CONTRACT["weight_decay"],
        clip_norm=CONTRACT["clip_norm"])

    class TinyFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocabulary, 8, dtype=torch.float32)
            self.embedding.weight.requires_grad_(False)
            self.dropout = torch.nn.Dropout(.2)
            self.readout = torch.nn.Linear(8, vocabulary, dtype=torch.float32)

        def forward(self, token_ids):
            return self.readout(self.dropout(self.embedding(token_ids)))

    def create():
        model = TinyFixture().cpu()
        optimizer = adapter["CpuAdamW"]([(name, value) for name, value in model.named_parameters()
                                         if value.requires_grad], design)
        return model, optimizer

    def advance(model, optimizer, begin, end, completed_tokens):
        losses, rates = [], []
        for group in plan["groups"][begin:end]:
            optimizer.zero_grad()
            loss_value = 0.0
            for row in group["rows"]:
                ids = torch.tensor(row["ids"], dtype=torch.long, device="cpu")
                logits = model(ids)  # Both arms forward every common whole row.
                mask = row["label_mask"] if arm == "joint" or row["role"] == "reasoning" else [False] * len(ids)
                loss = common_denominator_loss(logits, row["ids"], mask, group["planned_denominator"])
                loss.backward()
                loss_value += float(loss.detach())
            torch.nn.utils.clip_grad_norm_([value for value in model.parameters() if value.requires_grad],
                                          CONTRACT["clip_norm"], error_if_nonfinite=True)
            completed_tokens += group["processed_tokens"]
            rate = adapter["rate_at"](CONTRACT["peak_lr"], completed_tokens, total_tokens, design)
            optimizer.step(rate)
            optimizer.zero_grad()
            if any(value.grad is not None for value in model.parameters()):
                raise ValueError("fixture boundary did not clear every gradient")
            losses.append(loss_value)
            rates.append(rate)
        return completed_tokens, losses, rates

    try:
        random.seed(CONTRACT["seed"])
        torch.manual_seed(CONTRACT["seed"])
        baseline, base_optimizer = create()
        frozen_before = _tensor_digest({name: value for name, value in baseline.named_parameters()
                                        if not value.requires_grad})
        completed, losses, rates = advance(baseline, base_optimizer, 0, len(plan["groups"]), 0)
        baseline_state = _tensor_digest(baseline.state_dict())
        baseline_optimizer = _optimizer_digest(base_optimizer)
        baseline_rng = torch.get_rng_state().clone()
        resume_equal = None
        checkpoint_digest = None
        if checkpoint_after is not None:
            random.seed(CONTRACT["seed"])
            torch.manual_seed(CONTRACT["seed"])
            prefix, prefix_optimizer = create()
            position, _, _ = advance(prefix, prefix_optimizer, 0, checkpoint_after, 0)
            checkpoint = make_cpu_fixture_checkpoint(prefix, prefix_optimizer, cursor=checkpoint_after,
                completed_tokens=position, plan_sha256=plan["plan_sha256"])
            checkpoint_digest = cpu_fixture_checkpoint_digest(checkpoint)
            resumed, resumed_optimizer = create()  # Genuinely fresh parameter/Adam objects.
            restored = restore_cpu_fixture_checkpoint(resumed, resumed_optimizer, checkpoint,
                expected_plan_sha256=plan["plan_sha256"])
            final_position, _, _ = advance(resumed, resumed_optimizer, restored["cursor"],
                                           len(plan["groups"]), restored["completed_tokens"])
            resume_equal = (final_position == completed and _tensor_digest(resumed.state_dict()) == baseline_state
                            and _optimizer_digest(resumed_optimizer) == baseline_optimizer
                            and torch.equal(torch.get_rng_state(), baseline_rng))
            if not resume_equal:
                raise ValueError("full CPU fixture resume failed bitwise equivalence")
        frozen_after = _tensor_digest({name: value for name, value in baseline.named_parameters()
                                       if not value.requires_grad})
        if frozen_before != frozen_after:
            raise ValueError("frozen fixture parameters changed")
        final_checkpoint = make_cpu_fixture_checkpoint(baseline, base_optimizer,
            cursor=len(plan["groups"]), completed_tokens=completed, plan_sha256=plan["plan_sha256"])
        return {"status": "COMPLETE", "evidence_class": "CPU_FIXTURE_MECHANICS_ONLY", "fixture_only": True,
                "arm": arm, "parent_sha256": plan["parent_sha256"],
                "selected_case_ids": plan["selected_case_ids"], "sequence_digest": plan["sequence_digest"],
                "reasoning_labels_digest": plan["reasoning_labels_digest"],
                "groups": [group["case_ids"] for group in plan["groups"]],
                "planned_denominators": plan["planned_denominators"],
                "plan_digest": plan["plan_sha256"], "updates": base_optimizer.step_count,
                "gradients_cleared": True, "native_model_calls": 0,
                "final_checkpoint_sha256": cpu_fixture_checkpoint_digest(final_checkpoint),
                "masked_reasoning_gradient_proof": verify_masked_reasoning_gradient(plan),
                "native_authorized": False, "training_authorized": False,
                "capability_evidence": False, "new_objective_native_qualified": False,
                "dependency_bindings": adapter["bindings"], "plan_sha256": plan["plan_sha256"],
                "parameter_count": sum(value.numel() for value in baseline.parameters()),
                "optimizer_steps": base_optimizer.step_count, "processed_tokens": completed,
                "forward_rows": plan["selected_rows"], "common_denominators": plan["planned_denominators"],
                "planned_label_tokens": plan["joint_label_tokens"],
                "active_label_tokens": plan["joint_label_tokens"] if arm == "joint" else plan["reasoning_only_label_tokens"],
                "losses": losses, "rates": rates, "frozen_parameters_unchanged": True,
                "final_gradients": "NONE", "model_state_sha256": baseline_state,
                "optimizer_state_sha256": baseline_optimizer, "checkpoint_sha256": checkpoint_digest,
                "full_checkpoint_resume_bitwise_equal": resume_equal, "cuda_initialized": False}
    finally:
        random.setstate(caller_python)
        torch.set_rng_state(caller_torch)
