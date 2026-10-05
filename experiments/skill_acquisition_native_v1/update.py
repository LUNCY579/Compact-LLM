"""Bounded update/extraction mechanics; no runner, grant, model import or launch.

Framework imports are deferred until an admitted caller supplies a model. The
runner owns collection, resources, verifier rewards, C/U and checkpoint commits.
An exception once optimizer.step starts (including a partial step) requires
restoration; this helper cannot roll back an optimizer in place. Every operation
is observed before it begins.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import uuid

SCHEMA = "SKILL-ACQUISITION-UPDATE-RECIPE-1"
DONOR_SCHEMA = "SKILL-ACQUISITION-DONOR-BINDINGS-1"
TOKEN_TOLERANCE = 1e-4
CHUNK_BYTES = 1024 * 1024
HEX = frozenset("0123456789abcdef")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _hex(value):
    return type(value) is str and len(value) == 64 and not set(value) - HEX


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_recipe(recipe):
    """Pure admission/shape checks. No implicit recipe or scientific defaults."""
    require(type(recipe) is dict and recipe.get("schema") == SCHEMA, "update recipe schema")
    require(type(recipe.get("synthetic_cpu")) is bool, "explicit synthetic_cpu required")
    if not recipe["synthetic_cpu"]:
        require(recipe.get("approved") is True and recipe.get("training_authorized") is True,
                "native update needs an approved recipe and training authorization")
    require(type(recipe.get("B")) is int and recipe["B"] == 2
            and type(recipe.get("G")) is int and recipe["G"] == 8, "fixed B2/G8 required")
    require(type(recipe.get("boundary_id")) is int and recipe["boundary_id"] >= 0, "boundary identity")
    require(type(recipe.get("generation")) is int and recipe["generation"] >= 0, "rollback generation")
    require(_hex(recipe.get("symbolic_snapshot")) and _hex(recipe.get("start_policy_id")), "snapshot identities")
    names = recipe.get("trainable_names")
    require(type(names) is list and names and len(set(names)) == len(names)
            and all(type(n) is str and n for n in names), "exact trainable names required")
    eos = recipe.get("eos_ids")
    require(type(eos) is list and eos and len(set(eos)) == len(eos)
            and all(type(x) is int and x >= 0 for x in eos), "EOS inventory")
    require(type(recipe.get("pad_id")) is int and recipe["pad_id"] >= 0, "PAD identity")
    require(type(recipe.get("max_new_tokens")) is int and recipe["max_new_tokens"] > 0, "response cap")
    if not recipe["synthetic_cpu"]:
        require(eos == [151645, 151643] and recipe["pad_id"] == 151643
                and recipe["max_new_tokens"] == 128, "registered EOS/PAD/cap drift")
    objective = recipe.get("objective")
    require(type(objective) is dict and objective.get("method") in ("maxrl", "grpo")
            and _number(objective.get("epsilon")) and objective["epsilon"] >= 0, "explicit objective/epsilon")
    if objective["method"] == "grpo":
        require(recipe["synthetic_cpu"] and type(objective.get("ddof")) is int and objective["ddof"] == 1,
                "GRPO support is synthetic CPU only with explicit ddof1")
    elif not recipe["synthetic_cpu"]:
        require(objective["epsilon"] == 1e-6, "registered MaxRL epsilon drift")
    require(recipe.get("ratio_clipping", "missing") is None
            and recipe.get("gradient_clipping", "missing") is None, "clipping is unsupported; explicit None required")
    kl = recipe.get("anchor_KL")
    require(type(kl) is dict and kl.get("enabled") is False
            and _number(kl.get("coefficient")) and kl["coefficient"] == 0,
            "anchor KL is unsupported; explicit disabled coefficient0 required")
    require(recipe.get("gradient_checkpointing") == {"use_reentrant": False},
            "explicit non-reentrant gradient checkpointing required")
    opt = recipe.get("optimizer")
    required = {"class", "lr", "betas", "eps", "weight_decay", "foreach", "capturable"}
    require(type(opt) is dict and set(opt) == required and opt["class"] == "torch.optim.AdamW",
            "explicit AdamW recipe required")
    require(_number(opt["lr"]) and opt["lr"] > 0 and _number(opt["eps"]) and opt["eps"] > 0
            and _number(opt["weight_decay"]) and opt["weight_decay"] >= 0,
            "finite explicit optimizer scalars")
    require(type(opt["betas"]) in (tuple, list) and len(opt["betas"]) == 2
            and all(_number(b) and 0 <= b < 1 for b in opt["betas"]), "Adam betas")
    require(opt["foreach"] is False and type(opt["capturable"]) is bool, "Adam execution convention")
    return recipe


def group_advantages(rewards, objective):
    require(type(rewards) in (list, tuple) and len(rewards) == 8
            and all(type(r) is int and r in (0, 1) for r in rewards), "eight strict binary rewards")
    mean = sum(rewards) / 8
    if mean in (0.0, 1.0):
        return [0.0] * 8
    if objective["method"] == "maxrl":
        denominator = mean + objective["epsilon"]
    else:
        denominator = math.sqrt(sum((r - mean) ** 2 for r in rewards) / 7) + objective["epsilon"]
    return [(r - mean) / denominator for r in rewards]


def validate_trajectory(tokens, reason, raw, eos, cap):
    """Pure positional stop/mask contract; PAD=EOS is never inferred from IDs."""
    require(type(tokens) is list and 0 < len(tokens) <= cap
            and all(type(t) is int and t >= 0 for t in tokens), "completion token IDs")
    positions = [i for i, token in enumerate(tokens) if token in eos]
    require((reason == "eos" and positions == [len(tokens) - 1])
            or (reason == "length" and not positions and len(tokens) == cap), "genuine first EOS / exact cap")
    require(type(raw) is list and len(raw) == len(tokens)
            and all(_number(x) and x <= 0 for x in raw), "raw behavior token logprobs")


def _boundary_inputs(prompts, rollouts, recipe):
    require(type(prompts) is list and len(prompts) == 2
            and type(rollouts) is list and len(rollouts) == 2, "B2 prompts/groups")
    prompt_ids, rollout_ids, parsed, advantage = set(), set(), [], []
    for prompt, group in zip(prompts, rollouts):
        require(type(prompt) is dict and type(prompt.get("id")) is str and prompt["id"]
                and prompt["id"] not in prompt_ids, "unique prompt IDs")
        prompt_ids.add(prompt["id"])
        ids = prompt.get("prompt_ids")
        require(type(ids) is list and ids and all(type(t) is int and t >= 0 for t in ids), "prompt token IDs")
        require(type(group) is list and len(group) == 8, "G8 rollout group")
        rewards = []
        for sample in group:
            require(type(sample) is dict and type(sample.get("rollout_id")) is str and sample["rollout_id"]
                    and sample["rollout_id"] not in rollout_ids, "unique rollout IDs")
            rollout_ids.add(sample["rollout_id"])
            require(sample.get("prompt_id") == prompt["id"] and sample.get("policy_id") == recipe["start_policy_id"],
                    "one exact behavior policy/prompt per boundary")
            validate_trajectory(sample.get("completion_ids"), sample.get("stop_reason"),
                                sample.get("raw_step_logprobs"), recipe["eos_ids"], recipe["max_new_tokens"])
            require(type(sample.get("reward")) is int and sample["reward"] in (0, 1), "strict binary reward")
            rewards.append(sample["reward"])
            parsed.append((prompt, sample))
        advantage.extend(group_advantages(rewards, recipe["objective"]))
    return parsed, advantage


@contextmanager
def _span(operation, kind, metadata, device=None):
    require(callable(operation), "operation observer is required")
    identifier = uuid.uuid4().hex
    started = time.perf_counter()
    operation(kind, "started", operation_id=identifier, **metadata)
    record = {"operation_id": identifier, "kind": kind, **metadata}
    try:
        if device is not None and device.type == "cuda":
            import torch
            torch.cuda.synchronize(device)
        yield record
        if device is not None and device.type == "cuda":
            import torch
            torch.cuda.synchronize(device)
    except BaseException as error:
        try:
            operation(kind, "failed", operation_id=identifier, wall_s=time.perf_counter() - started,
                      error_type=type(error).__name__, error=str(error)[:1000], **metadata)
        except BaseException as observer_error:
            error.accounting_failure = {"type": type(observer_error).__name__, "message": str(observer_error)[:1000]}
            raise error from observer_error
        raise
    else:
        record["wall_s"] = time.perf_counter() - started
        operation(kind, "returned", **{key: value for key, value in record.items() if key != "kind"})


def _parameter_storage_check(named):
    owners = {}
    for name, parameter in named.items():
        require(parameter.device.type != "meta" and parameter.numel() > 0, "unmaterialized/empty parameter: " + name)
        storage = (str(parameter.device), parameter.untyped_storage().data_ptr())
        require(owners.setdefault(storage, id(parameter)) == id(parameter), "unsupported distinct-Parameter shared storage")


def _actual_optimizer(model, optimizer, recipe, torch):
    require(type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups) == 1, "one exact AdamW group")
    named = dict(model.named_parameters(remove_duplicate=False))
    _parameter_storage_check(named)
    actual = sorted(name for name, tensor in named.items() if tensor.requires_grad)
    require(actual == sorted(recipe["trainable_names"]), "trainable inventory drift")
    trainable = {id(named[name]): named[name] for name in actual}
    require(len(trainable) == len(actual), "donor inventory must not contain tied aliases")
    group = optimizer.param_groups[0]
    require(len(group["params"]) == len(trainable) and {id(p) for p in group["params"]} == set(trainable),
            "optimizer parameter inventory drift")
    for key in ("lr", "eps", "weight_decay", "foreach", "capturable"):
        require(group.get(key) == recipe["optimizer"][key], "optimizer recipe drift: " + key)
    require(tuple(group["betas"]) == tuple(recipe["optimizer"]["betas"]), "optimizer beta drift")
    require(group.get("amsgrad") is False and group.get("maximize") is False
            and group.get("differentiable") is False and group.get("fused") in (None, False),
            "unsupported Adam variant")
    states, steps = [], []
    for parameter in trainable.values():
        require(parameter.dtype == torch.float32 and parameter.is_contiguous(), "FP32 contiguous trainable parameter")
        state = optimizer.state.get(parameter)
        if state:
            require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "Adam state schema")
            for moment in (state["exp_avg"], state["exp_avg_sq"]):
                require(moment.dtype == torch.float32 and moment.shape == parameter.shape
                        and moment.device == parameter.device, "Adam moment dtype/shape/device")
            step = state["step"]
            require(step.numel() == 1 and step.device == parameter.device, "Adam counter device/shape")
            value = float(step.item())
            require(math.isfinite(value) and value >= 0 and value.is_integer(), "Adam counter value")
            steps.append(int(value))
        states.append(bool(state))
    require(not any(states) or all(states), "partially initialized Adam inventory")
    require(not steps or len(set(steps)) == 1, "Adam counters differ within inventory")
    return named, list(trainable.values()), (steps[0] if steps else 0)


def _gradient_proof(named, torch):
    squares = []
    for name, parameter in named.items():
        if not parameter.requires_grad:
            require(parameter.grad is None, "frozen parameter received gradient: " + name)
            continue
        require(parameter.grad is not None and parameter.grad.dtype == torch.float32 and parameter.grad.is_contiguous(),
                "missing/non-FP32 trainable gradient: " + name)
        flat = parameter.grad.detach().reshape(-1)
        for start in range(0, flat.numel(), CHUNK_BYTES // 4):
            part = flat[start:start + CHUNK_BYTES // 4]
            require(bool(torch.isfinite(part).all()), "nonfinite trainable gradient: " + name)
            squares.append(float((part.double() * part.double()).sum().item()))
    norm = math.sqrt(math.fsum(squares))
    require(math.isfinite(norm), "nonfinite gradient norm")
    return norm


def boundary_update(model, optimizer, prompts, rollouts, recipe, operation):
    """One transactional B2/G8 update. Caller commits C/U only after success.

    prompts=[{id,prompt_ids}], rollouts are two lists of eight records containing
    prompt_id, rollout_id, policy_id, completion_ids, stop_reason, reward and
    raw_step_logprobs. Observers receive (kind, phase, **metadata). A returned
    optimizer operation remains physical evidence if a later proof fails.
    """
    validate_recipe(recipe)
    parsed, advantages = _boundary_inputs(prompts, rollouts, recipe)
    require(callable(operation), "operation observer is required")
    devices = {p.device for p in model.parameters()}
    require(len(devices) == 1, "model parameters span devices")
    device = devices.pop()
    require(not recipe["synthetic_cpu"] or device.type == "cpu", "synthetic_cpu is valid only on CPU")
    require(recipe["synthetic_cpu"] or device.type == "cuda", "registered native update requires CUDA")
    require(recipe["optimizer"]["capturable"] == (device.type == "cuda"), "Adam capturable/device drift")
    vocab = model.get_output_embeddings().weight.shape[0]
    require(vocab > 0 and all(token < vocab for token in recipe["eos_ids"] + [recipe["pad_id"]])
            and all(token < vocab for prompt, sample in parsed
                    for token in prompt["prompt_ids"] + sample["completion_ids"]), "token ID outside actual head vocabulary")
    import torch
    from experiments.maxrl_comp002_pretraining_v1 import s0_generate as generate, s0_runtime as runtime
    from experiments.maxrl_comp002_pretraining_v1.mechanics import completion_labels
    metadata = {"boundary_id": recipe["boundary_id"], "generation": recipe["generation"],
                "symbolic_snapshot": recipe["symbolic_snapshot"], "start_policy_id": recipe["start_policy_id"]}
    state = dict(model.named_parameters(remove_duplicate=False))
    state.update(dict(model.named_buffers(remove_duplicate=False)))
    unique_state = {id(value): value for value in state.values()}
    state_bytes = sum(value.numel() * value.element_size() for value in unique_state.values())
    from .ce_operation_contracts import MODEL_FINGERPRINT
    with _span(operation, "fingerprint", dict(metadata, reason="pre_update_identity_and_optimizer",
                state_bytes=state_bytes, wall_contract=MODEL_FINGERPRINT), device) as identity_op:
        before = runtime.state_digests(model)
        require(runtime.combine(before) == recipe["start_policy_id"], "model is not the sampled behavior snapshot")
        named, trainable, U_before = _actual_optimizer(model, optimizer, recipe, torch)
        require(all(p.grad is None for p in named.values()), "stale gradients at optimizer boundary")
        frozen_digests={name:value for name,value in before.items() if name not in recipe["trainable_names"]}
        frozen=runtime.combine(frozen_digests) if frozen_digests else None
        identity_op.update(policy_sha256=runtime.combine(before),U=U_before,frozen_state_sha256=frozen)
    result = {**metadata, "schema": "SKILL-ACQUISITION-BOUNDARY-RESULT-1", "B": 2, "G": 8,
              "loss_denominator": 16, "advantages": advantages, "updated": False,
              "U_before": U_before, "U_after": U_before, "optimizer_operation": None,
              "identity_operation": identity_op, "rows": [], "parity_tolerance_nats": TOKEN_TOLERANCE,
              "frozen_state_sha256":frozen,
              "optimizer_recipe": json.loads(json.dumps(recipe["optimizer"])),
              "actual_optimizer_defaults": {key: list(value) if type(value) is tuple else value
                                            for key, value in optimizer.defaults.items()}}
    if not any(advantages):
        model.eval()
        result.update(skipped_reason="uniform_all_batch", parity="NOT_EVALUATED_NO_FORWARD_OR_UPDATE", loss=0.0)
        return result
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(recipe["gradient_checkpointing"]))
    require(model.is_gradient_checkpointing, "gradient checkpointing did not activate")
    optimizer.zero_grad(set_to_none=True)
    losses, old_losses, parity_bounds = [], [], []
    model.train()
    try:
        for (prompt, sample), advantage in zip(parsed, advantages):
            ids, tokens = prompt["prompt_ids"], sample["completion_ids"]
            selected, total = len(tokens), len(ids) + len(tokens)
            row_meta = dict(metadata, prompt_id=prompt["id"], rollout_id=sample["rollout_id"],
                            selected_tokens=selected, full_sequence_tokens=total)
            with _span(operation, "loss_forward", row_meta, device) as forward_op:
                values = generate.teacher_forced_logprobs(model, [ids], [tokens], recipe["pad_id"],
                          padding_side="none", checkpoint_head=True)[0]
                require(values.requires_grad and values.shape == (selected,), "differentiable suffix shape")
                sequence = torch.tensor([ids + tokens], dtype=torch.long, device=device)
                labels = completion_labels(sequence, torch.ones_like(sequence, dtype=torch.bool), [len(ids)],
                           [selected], recipe["eos_ids"], [sample["stop_reason"]], recipe["max_new_tokens"])
                require(labels[0, :len(ids)].eq(-100).all() and torch.equal(labels[0, len(ids):], sequence[0, len(ids):]),
                        "prompt/suffix label drift")
                detached = values.detach().tolist()
                gaps = [abs(a - b) for a, b in zip(detached, sample["raw_step_logprobs"])]
                require(max(gaps) <= TOKEN_TOLERANCE, "sampled/train+checkpoint token logprob parity failed")
                new_sum, old_sum = math.fsum(detached), math.fsum(sample["raw_step_logprobs"])
                rounding = 4 * math.ulp(max(abs(new_sum), abs(old_sum), 1.0))
                require(abs(new_sum - old_sum) <= selected * TOKEN_TOLERANCE + rounding, "trajectory-sum parity failed")
                loss = -(advantage / 16) * values.sum()
                scalar = float(loss.detach().item())
                expected = -(advantage / 16) * new_sum
                u = 2.0 ** -24
                gamma = selected * u / (1 - selected * u)
                round_bound = abs(advantage / 16) * gamma * math.fsum(abs(v) for v in detached) + u * abs(expected)
                require(math.isfinite(scalar) and abs(scalar - expected) <= round_bound + 4 * math.ulp(max(abs(expected), 1.0)),
                        "FP32 trajectory-summed loss reduction drift")
            with _span(operation, "backward", row_meta, device) as backward_op:
                loss.backward()
            losses.append(scalar)
            old_losses.append(-(advantage / 16) * old_sum)
            parity_bounds.append(abs(advantage) * selected * TOKEN_TOLERANCE / 16)
            result["rows"].append({"prompt_id": prompt["id"], "rollout_id": sample["rollout_id"],
                 "completion_length": selected, "stop_reason": sample["stop_reason"], "raw_trajectory_logprob": old_sum,
                 "train_trajectory_logprob": new_sum, "max_token_gap": max(gaps), "loss": scalar,
                 "forward_operation": forward_op, "backward_operation": backward_op})
            del values, loss, sequence, labels
        with _span(operation, "gradient_validation", metadata, device):
            grad_norm = _gradient_proof(named, torch)
        with _span(operation, "optimizer", dict(metadata, U_before=U_before, U_after=U_before + 1), device) as step_op:
            optimizer.step()
        result["optimizer_operation"] = step_op
        with _span(operation, "fingerprint", dict(metadata, reason="post_update_state_proof", state_bytes=state_bytes, wall_contract=MODEL_FINGERPRINT), device) as final_op:
            _, _, U_after = _actual_optimizer(model, optimizer, recipe, torch)
            require(U_after == U_before + 1, "Adam counter did not advance exactly once")
            for parameter in trainable:
                state = optimizer.state[parameter]
                require(bool(torch.isfinite(parameter).all()) and all(bool(torch.isfinite(state[key]).all())
                        for key in ("exp_avg", "exp_avg_sq")), "nonfinite post-update parameter/moment")
            after = runtime.state_digests(model)
            require(runtime.combine({name: value for name, value in after.items() if name not in recipe["trainable_names"]})
                    == frozen, "frozen parameter or buffer changed")
            final_op.update(policy_sha256=runtime.combine(after),U=U_after,frozen_state_sha256=frozen)
        result.update(updated=True, U_after=U_after, loss=math.fsum(losses), behavior_loss=math.fsum(old_losses),
                      behavior_loss_token_bound=math.fsum(parity_bounds), gradient_norm=grad_norm,
                      final_policy_id=runtime.combine(after), final_identity_operation=final_op, parity="PASS")
        return result
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.eval()


class _TensorFile:
    """Strict bounded safetensors reader. No whole-file mmap or tensor materialization."""
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open("rb")
        try:
            first = self.file.read(8)
            require(len(first) == 8, "truncated tensor header")
            length = struct.unpack("<Q", first)[0]
            require(0 < length <= CHUNK_BYTES, "tensor header budget")
            raw = self.file.read(length)
            require(len(raw) == length, "truncated tensor header")
            def pairs(items):
                result = {}
                for key, value in items:
                    require(key not in result, "duplicate tensor header key")
                    result[key] = value
                return result
            self.header = json.loads(raw, object_pairs_hook=pairs,
                                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite header")))
            require(type(self.header) is dict, "tensor header object")
            self.header.pop("__metadata__", None)
            self.start = 8 + length
            end = 0
            for key, entry in sorted(self.header.items(), key=lambda x: x[1]["data_offsets"][0]):
                require(type(key) is str and key and "\0" not in key and "\n" not in key,
                        "invalid tensor name")
                require(type(entry) is dict and set(entry) == {"dtype", "shape", "data_offsets"}
                        and entry["dtype"] in ("BF16", "F32"), "tensor entry dtype/schema")
                require(type(entry["shape"]) is list and all(type(x) is int and x >= 0 for x in entry["shape"]), "tensor shape")
                offsets = entry["data_offsets"]
                require(type(offsets) is list and len(offsets) == 2 and all(type(x) is int for x in offsets)
                        and offsets[0] == end, "tensor offset tiling")
                width = 2 if entry["dtype"] == "BF16" else 4
                require(offsets[1] - offsets[0] == width * math.prod(entry["shape"]), "tensor bytes/shape")
                end = offsets[1]
            require(self.header and self.start + end == self.path.stat().st_size, "tensor payload size")
        except BaseException:
            self.file.close()
            raise

    def raw(self, name, start, count):
        entry = self.header[name]
        width = 2 if entry["dtype"] == "BF16" else 4
        require(type(start) is int and type(count) is int and start >= 0 and count >= 0
                and start + count <= math.prod(entry["shape"]) and count * width <= CHUNK_BYTES,
                "bounded tensor slice")
        self.file.seek(self.start + entry["data_offsets"][0] + start * width)
        raw = self.file.read(count * width)
        require(len(raw) == count * width, "truncated tensor payload")
        return raw

    def part(self, name, start, count, torch):
        dtype = torch.bfloat16 if self.header[name]["dtype"] == "BF16" else torch.float32
        value = torch.frombuffer(bytearray(self.raw(name, start, count)), dtype=dtype).to(torch.float32)
        require(bool(torch.isfinite(value).all()), "nonfinite tensor payload")
        return value

    def data_digest(self):
        self.file.seek(self.start)
        result = hashlib.sha256()
        for block in iter(lambda: self.file.read(CHUNK_BYTES), b""):
            result.update(block)
        return result.hexdigest()

    def state_digest(self, combine):
        digests = {}
        for name, entry in self.header.items():
            dtype = "torch.bfloat16" if entry["dtype"] == "BF16" else "torch.float32"
            result = _tensor_header_hash(dtype, entry["shape"])
            count = math.prod(entry["shape"])
            width = 2 if entry["dtype"] == "BF16" else 4
            for start in range(0, count, CHUNK_BYTES // width):
                result.update(self.raw(name, start, min(CHUNK_BYTES // width, count - start)))
            digests[name] = result.hexdigest()
        return combine(digests)

    def close(self):
        self.file.close()


def _tensor_header_hash(dtype, shape):
    # Both arm_build.tensor_digest and s0_runtime.tensor_sha256 use compact JSON.
    header = json.dumps({"dtype": dtype, "shape": list(shape)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(header.encode("ascii") + b"\n")


def _file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            result.update(block)
    return result.hexdigest()


def extract_donor(model, baseweights, warmdelta, names, out, bindings, *, native_permit=None):
    """Stream final - exact rebuilt common warm anchor, and prove deployment.

    bindings holds explicit hashes plus operation/gate callbacks. Sequential
    own_start_components are verified and rebuilt as a separate identity; they
    are NEVER subtracted as the common reference. No trained warm master is read.
    The returned deployed digest may differ from the FP32 trained master because
    subtraction/readdition rounds; the residual is reported, not required zero.
    """
    require(type(bindings) is dict and bindings.get("schema") == DONOR_SCHEMA, "donor binding schema")
    require(type(bindings.get("synthetic_cpu")) is bool, "donor synthetic_cpu flag")
    if not bindings["synthetic_cpu"]:
        if native_permit is None:
            require(bindings.get("approved") is True and bindings.get("training_authorized") is True, "donor extraction admission")
        else:
            from .ce_native_authority import NativeExecutionPermit
            require(type(native_permit) is NativeExecutionPermit, "opaque native CE donor permit required")
            context = native_permit.context
            stage = "delta_roundtrip" if context["phase"] == "qualification" else "donor_publish_verify"
            native_permit.require(stage, static_recipe_sha256=context["static_recipe_sha256"],
                                  operation=bindings.get("operation"), gate=bindings.get("gate"))
    hashes = ("base_weights_sha256", "warm_delta_file_sha256", "warm_delta_data_sha256", "warm_delta_state_sha256",
              "common_anchor_sha256", "own_start_sha256", "final_master_sha256", "frozen_state_sha256", "trainable_names_sha256")
    require(all(_hex(bindings.get(key)) for key in hashes), "explicit donor/reference hashes")
    require(type(names) is list and names and len(set(names)) == len(names) and all(type(n) is str and n for n in names)
            and digest(sorted(names)) == bindings["trainable_names_sha256"], "donor name inventory")
    components = bindings.get("own_start_components")
    require(type(components) is list and bindings.get("own_start_kind") in ("independent", "sequential"), "own-start convention")
    require((bindings["own_start_kind"] == "independent" and not components)
            or (bindings["own_start_kind"] == "sequential" and components), "own-start components")
    operation, gate = bindings.get("operation"), bindings.get("gate")
    require(callable(operation) and callable(gate), "extraction observers/guards required")
    devices = {p.device for p in model.parameters()}
    require(len(devices) == 1, "donor model device")
    device = devices.pop()
    require(not bindings["synthetic_cpu"] or device.type == "cpu", "donor synthetic CPU only")
    import torch
    from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as runtime, warmstart_checkpoint as stream
    require(sys.byteorder == "little", "little-endian extraction required")
    target = runtime.assert_outside_checkout(out, Path(__file__).resolve().parents[2])
    require(target.parent.is_dir() and not os.path.lexists(target), "donor output directory/create-only")
    named = dict(model.named_parameters(remove_duplicate=False))
    _parameter_storage_check(named)
    all_state = dict(named)
    all_state.update(dict(model.named_buffers(remove_duplicate=False)))
    require(sorted(name for name, value in named.items() if value.requires_grad) == sorted(names), "donor trainable scope")
    require(len({id(named[name]) for name in names}) == len(names) and "lm_head.weight" not in names,
            "donor aliases are unsupported")
    require(all(named[name].dtype == torch.float32 and named[name].is_contiguous() and named[name].numel() > 0
                for name in names), "donor FP32 contiguous inventory")
    require(all(value.grad is None for value in named.values()), "extract only at a cleared optimizer boundary")
    meta = {key: bindings[key] for key in ("common_anchor_sha256", "own_start_sha256", "final_master_sha256")}
    state_bytes = sum(value.numel() * value.element_size() for value in {id(v): v for v in all_state.values()}.values())
    from .ce_operation_contracts import MODEL_FINGERPRINT
    with _span(operation, "fingerprint", dict(meta, reason="donor_final_master_and_frozen", state_bytes=state_bytes, wall_contract=MODEL_FINGERPRINT), device) as final_op:
        final_digests = runtime.state_digests(model)
        require(runtime.combine(final_digests) == bindings["final_master_sha256"], "final donor master drift")
        require(runtime.combine({k: v for k, v in final_digests.items() if k not in names}) == bindings["frozen_state_sha256"],
                "donor frozen state drift")
        final_op.update(policy_sha256=runtime.combine(final_digests),frozen_state_sha256=bindings["frozen_state_sha256"])
    readers = []
    try:
        with _span(operation, "donor_reference", meta, device):
            gate("donor_reference")
            require(_file_sha(baseweights) == bindings["base_weights_sha256"], "base weight file drift")
            require(_file_sha(warmdelta) == bindings["warm_delta_file_sha256"], "warm delta file drift")
            base = _TensorFile(baseweights)
            readers.append(base)
            warm = _TensorFile(warmdelta)
            readers.append(warm)
            require(set(warm.header) <= set(base.header) and all(x["dtype"] == "F32" for x in warm.header.values()),
                    "warm/base support or dtype")
            require(warm.data_digest() == bindings["warm_delta_data_sha256"]
                    and warm.state_digest(runtime.combine) == bindings["warm_delta_state_sha256"], "warm delta digest drift")
            own = []
            for component in components:
                require(type(component) is dict and _number(component.get("coefficient")) and component["coefficient"] == 1.0
                        and all(_hex(component.get(k)) for k in ("file_sha256", "data_sha256", "state_sha256")),
                        "bound full-strength own-start component")
                require(_file_sha(component["path"]) == component["file_sha256"], "own-start file drift")
                reader = _TensorFile(component["path"])
                readers.append(reader)
                require(set(reader.header) <= set(names) and all(x["dtype"] == "F32" for x in reader.header.values())
                        and reader.data_digest() == component["data_sha256"]
                        and reader.state_digest(runtime.combine) == component["state_sha256"], "own-start component support/digest")
                own.append(reader)
            def source_name(name):
                if name == "lm_head.weight" and name not in base.header:
                    require(named[name] is named.get("model.embed_tokens.weight"), "unqualified tied alias")
                    return "model.embed_tokens.weight"
                return name
            require(set(base.header) <= set(all_state) and set(warm.header) <= set(named),
                    "base/warm contains unknown model state")
            for name, value in all_state.items():
                source = source_name(name)
                if source in base.header:
                    require(tuple(base.header[source]["shape"]) == tuple(value.shape), "base/model shape drift")
                    require(bindings["synthetic_cpu"] or base.header[source]["dtype"] == "BF16", "native base is not BF16")
                    for reader in (warm, *own):
                        if source in reader.header:
                            require(reader.header[source]["shape"] == base.header[source]["shape"], "component shape drift")
                else:
                    require(name not in named, "parameter missing from common base")
            require(set(names) <= set(base.header), "inventory absent from base")
            def references(name):
                value, source = all_state[name], source_name(name)
                stride = CHUNK_BYTES // max(4, value.element_size())
                for start in range(0, value.numel(), stride):
                    gate("donor_chunk")
                    count = min(stride, value.numel() - start)
                    if source in base.header:
                        common = base.part(source, start, count, torch)
                        if source in warm.header:
                            piece = common.to(device)
                            piece.add_(warm.part(source, start, count, torch).to(device))
                            common = piece.to("cpu")
                    else:
                        common = value.detach().reshape(-1)[start:start + count].to("cpu").clone()
                    require(bool(torch.isfinite(common).all()), "nonfinite rebuilt common reference")
                    start_state = common.clone()
                    for reader in own:
                        if source in reader.header:
                            piece = start_state.to(device)
                            piece.add_(reader.part(source, start, count, torch).to(device))
                            start_state = piece.to("cpu")
                    yield start, common, start_state
            common_digests, own_digests = {}, {}
            for name, value in all_state.items():
                common_hash = _tensor_header_hash(str(value.dtype), value.shape)
                own_hash = _tensor_header_hash(str(value.dtype), value.shape)
                for _, common, start_state in references(name):
                    common_hash.update(memoryview(common.contiguous().view(torch.uint8).numpy()).cast("B"))
                    own_hash.update(memoryview(start_state.contiguous().view(torch.uint8).numpy()).cast("B"))
                common_digests[name], own_digests[name] = common_hash.hexdigest(), own_hash.hexdigest()
            require(runtime.combine(common_digests) == bindings["common_anchor_sha256"], "not the exact rebuilt warm common anchor")
            require(runtime.combine(own_digests) == bindings["own_start_sha256"], "own-start reconstruction drift")
        specs = {name: ("F32", list(named[name].shape)) for name in sorted(names)}
        prewrite_rebuilt = dict(common_digests)
        residual_squares, delta_squares, nonzero, maximum, changed_tensors = [], [], 0, 0.0, 0
        def chunks(name):
            nonlocal nonzero, maximum, changed_tensors
            rebuilt_hash = _tensor_header_hash("torch.float32", named[name].shape)
            had_residual = False
            final = named[name].detach().reshape(-1)
            for start, common, _ in references(name):
                count = common.numel()
                master_device = final[start:start + count]
                reference_device = common.to(device)
                delta_device = master_device - reference_device
                require(bool(torch.isfinite(delta_device).all()), "nonfinite donor subtraction")
                deployed = (reference_device + delta_device).to("cpu").contiguous()
                delta = delta_device.to("cpu").contiguous()
                master = master_device.to("cpu").contiguous()
                require(bool(torch.isfinite(deployed).all()), "nonfinite donor roundtrip")
                rebuilt_hash.update(memoryview(deployed.view(torch.uint8).numpy()).cast("B"))
                difference = deployed.double() - master.double()
                count_changed = int(torch.count_nonzero(difference).item())
                had_residual |= bool(count_changed)
                nonzero += count_changed
                maximum = max(maximum, float(difference.abs().max().item()) if count else 0.0)
                residual_squares.append(float((difference * difference).sum().item()))
                delta_squares.append(float((delta.double() * delta.double()).sum().item()))
                yield memoryview(delta.view(torch.uint8).numpy()).cast("B")
            changed_tensors += int(had_residual)
            prewrite_rebuilt[name] = rebuilt_hash.hexdigest()
        with _span(operation, "donor_write", dict(meta, output=str(target), payload_bytes=4 * sum(named[n].numel() for n in names)), device):
            gate("donor_write")
            written = stream.stream_tensors(target, specs, chunks)
        # Publication is create-only. A failed proof leaves its unpromoted artifact
        # for the runner's failure receipt; it never silently deletes/replaces it.
        with _span(operation, "donor_roundtrip", dict(meta, output=str(target)), device):
            saved = _TensorFile(target)
            readers.append(saved)
            require(saved.data_digest() == written["data_section_sha256"] and _file_sha(target) == written["file_sha256"],
                    "published donor data/file hash drift")
            delta_state = saved.state_digest(runtime.combine)
            rebuilt_digests = dict(common_digests)
            for name in names:
                rebuilt_hash = _tensor_header_hash("torch.float32", named[name].shape)
                for start, common, _ in references(name):
                    value = common.to(device)
                    value.add_(saved.part(name, start, common.numel(), torch).to(device))
                    value = value.to("cpu").contiguous()
                    require(bool(torch.isfinite(value).all()), "nonfinite saved donor rebuild")
                    rebuilt_hash.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
                rebuilt_digests[name] = rebuilt_hash.hexdigest()
            require(runtime.combine(rebuilt_digests) == runtime.combine(prewrite_rebuilt), "saved donor rebuild differs from extraction arithmetic")
            for path, expected in ((baseweights, bindings["base_weights_sha256"]), (warmdelta, bindings["warm_delta_file_sha256"])):
                require(_file_sha(path) == expected, "reference changed during extraction")
            for component in components:
                require(_file_sha(component["path"]) == component["file_sha256"], "own start changed during extraction")
        with _span(operation, "fingerprint", dict(meta, reason="donor_master_unchanged", state_bytes=state_bytes, wall_contract=MODEL_FINGERPRINT), device) as proof_op:
            require(runtime.anchor_digest(model) == bindings["final_master_sha256"], "extraction mutated donor master")
            proof_op.update(policy_sha256=bindings["final_master_sha256"])
        return {"schema": "SKILL-ACQUISITION-DONOR-DELTA-1", "path": str(target), **written,
                "state_sha256": delta_state, "reference": "EXACT_REBUILT_COMMON_WARM_ANCHOR",
                "common_anchor_sha256": bindings["common_anchor_sha256"], "own_start_sha256": bindings["own_start_sha256"],
                "own_start_kind": bindings["own_start_kind"], "final_master_sha256": bindings["final_master_sha256"],
                "deployed_arm_sha256": runtime.combine(rebuilt_digests), "trainable_names_sha256": bindings["trainable_names_sha256"],
                "tensors": len(names), "numel": sum(named[n].numel() for n in names), "delta_norm": math.sqrt(math.fsum(delta_squares)),
                "roundtrip": {"proved_from_saved_delta": True, "master_equal": nonzero == 0,
                              "arithmetic_device": str(device), "model_parameters_mutated": False,
                              "residual_nonzero_elements": nonzero, "residual_nonzero_tensors": changed_tensors,
                              "residual_max_abs": maximum, "residual_norm": math.sqrt(math.fsum(residual_squares))},
                "payload_IO_chunk_ceiling_bytes": CHUNK_BYTES, "reference_CPU_piece_ceiling_bytes": CHUNK_BYTES,
                "fingerprint_primitive": "s0_runtime.state_digests: separately priced CHUNK-element pieces",
                "trained_warm_master_used": False}
    finally:
        for reader in readers:
            reader.close()
