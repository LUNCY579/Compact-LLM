"""Injected-Torch qualification mechanics; importing this module cannot launch ML.

This component has no checkpoint loader for a base model, framework imports,
capability evaluation, or execution permission. Its caller owns source-byte,
resource, precision, output-parity and launch qualification. A CPU toy can check
the algorithm by explicitly selecting ``require_cuda=False``; that is not a
native-model result.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import random

VERSION = "synth-training-mechanics-1"
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
MAX_MILESTONE_BYTES = 64 * 1024
SEQUENCES_PER_UPDATE = 20
CE_ATOL = 2e-6
GRAD_ATOL, GRAD_RTOL = 1e-6, 1e-5


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_number(value):
    """Keep failed nonfinite comparisons representable in strict JSON."""
    number = float(value)
    return number if math.isfinite(number) else ("NaN" if math.isnan(number) else
                                                "Infinity" if number > 0 else "-Infinity")


def _preserve_fragment(preserve, name, value):
    if preserve is None:
        return

    def validate(part):
        if type(part) is dict:
            _require(all(type(key) is str for key in part), "milestone JSON keys must be strings")
            for item in part.values():
                validate(item)
        elif type(part) is list:
            for item in part:
                validate(item)
        else:
            _require(part is None or type(part) in (str, int, float, bool),
                     "milestone must contain pure JSON values")
            _require(type(part) is not float or math.isfinite(part), "nonfinite milestone JSON number")
    _require(callable(preserve), "milestone preservation callback")
    validate(value)
    raw = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    _require(len(raw) <= MAX_MILESTONE_BYTES, "mechanics milestone exceeds 64 KiB bound")
    # Give the writer detached Python data, so it cannot mutate a later gate's
    # comparison result or retain a Torch tensor through an alias.
    preserve(name, json.loads(raw))


def bitwise_equal_tensor(torch, left, right):
    """Compare values as bytes, including the sign bit of a floating zero."""
    if not (torch.is_tensor(left) and torch.is_tensor(right)
            and left.shape == right.shape and left.dtype == right.dtype):
        return False
    # Checkpoint tensors are CPU-resident. Move only the bounded adapter/state
    # operand to the independent snapshot's device for the byte comparison.
    right = right.detach().to(device=left.device)
    return bool(torch.equal(left.detach().reshape(-1).contiguous().view(torch.uint8),
                            right.reshape(-1).contiguous().view(torch.uint8)))


def equal_tree(torch, left, right):
    if torch.is_tensor(left):
        return bitwise_equal_tensor(torch, left, right)
    if type(left) is dict:
        return (type(right) is dict and set(left) == set(right)
                and all(equal_tree(torch, left[key], right[key]) for key in left))
    if type(left) in (list, tuple):
        return (type(right) is type(left) and len(left) == len(right)
                and all(equal_tree(torch, a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _clone_tree(torch, value, *, cpu=False):
    if torch.is_tensor(value):
        source = value.detach().to(device="cpu") if cpu else value.detach()
        return source.clone()
    if type(value) is dict:
        return {key: _clone_tree(torch, part, cpu=cpu) for key, part in value.items()}
    if type(value) is list:
        return [_clone_tree(torch, part, cpu=cpu) for part in value]
    if type(value) is tuple:
        return tuple(_clone_tree(torch, part, cpu=cpu) for part in value)
    _require(value is None or type(value) in (str, int, float, bool), "unsupported state value")
    return value


def _adapters(model, torch):
    entries = [(name, parameter) for name, parameter in model.named_parameters()
               if name.endswith((".lora_A", ".lora_B"))]
    _require(bool(entries) and len(entries) == len({id(p) for _, p in entries}),
             "missing or shared adapter parameter")
    _require(all(p.requires_grad and p.dtype == torch.float32 for _, p in entries),
             "adapter factors must be trainable FP32")
    _require({name for name, p in model.named_parameters() if p.requires_grad}
             == {name for name, _ in entries}, "gradient scope includes a non-adapter")
    _require(all(bool(torch.isfinite(p).all()) for _, p in entries), "nonfinite adapter factor")
    return entries


def _rng_snapshot(torch, require_cuda):
    state = {"python": random.getstate(), "cpu": torch.get_rng_state().clone()}
    if require_cuda:
        _require(torch.cuda.is_initialized(), "CUDA RNG snapshot requires a live owned CUDA context")
        state["cuda0"] = torch.cuda.get_rng_state(0).clone()
    else:
        _require(not torch.cuda.is_initialized(), "CPU toy must not initialize CUDA")
    return state


def snapshot_training_state(torch, model, optimizer, *, require_cuda=True):
    """Detach adapter, optimizer, gradient and RNG state before any save.

    Adapter/moment snapshots stay on the live device. No base tensor is copied.
    This is the independent restoration/reference witness, not serialized state.
    """
    entries = _adapters(model, torch)
    return {"schema_version": 1,
            "adapters": {name: p.detach().clone() for name, p in entries},
            "optimizer": _clone_tree(torch, optimizer.state_dict()),
            "gradients": {name: None if p.grad is None else p.grad.detach().clone()
                          for name, p in entries},
            "rng": _rng_snapshot(torch, require_cuda)}


def _validate_optimizer_state(torch, state, entries, optimizer, *, step=None):
    _require(type(state) is dict and set(state) == {"state", "param_groups"}, "optimizer state schema")
    groups = state["param_groups"]
    actual = optimizer.state_dict()["param_groups"]
    _require(type(groups) is list and len(groups) == 1 and len(actual) == 1,
             "exactly one optimizer parameter group required")
    group = groups[0]
    _require(type(group) is dict and set(group) == set(actual[0]), "actual optimizer group fields differ")
    # Use only fields present in the real Torch 2.6 AdamW. Never fabricate the
    # decoupled_weight_decay field introduced by a different runtime version.
    expected = {"lr": 1e-4, "betas": (.9, .999), "eps": 1e-8, "weight_decay": 0,
                "amsgrad": False, "maximize": False, "foreach": False, "capturable": False,
                "differentiable": False, "fused": False}
    _require("decoupled_weight_decay" not in group,
             "optimizer group schema is not the registered Torch 2.6 profile")
    _require(all(key in group and equal_tree(torch, group[key], value)
                 for key, value in expected.items()), "optimizer settings differ")
    indices = group["params"]
    _require(type(indices) is list and all(type(index) is int for index in indices)
             and indices == list(range(len(entries))), "optimizer parameter order differs")
    _require(type(state["state"]) is dict and all(type(index) is int for index in state["state"])
             and set(state["state"]) == set(indices),
             "optimizer moment inventory incomplete")
    for index, (_, parameter) in zip(indices, entries):
        item = state["state"][index]
        _require(type(item) is dict and set(item) == {"step", "exp_avg", "exp_avg_sq"},
                 "actual AdamW moment fields differ")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = item[key]
            _require(torch.is_tensor(tensor) and tensor.shape == parameter.shape
                     and tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()),
                     "invalid FP32 optimizer moment: " + key)
        _require(bool((item["exp_avg_sq"] >= 0).all()), "negative second moment")
        count = item["step"]
        _require(torch.is_tensor(count) and count.ndim == 0 and count.dtype == torch.float32
                 and count.device.type == "cpu"
                 and bool(torch.isfinite(count)) and float(count) > 0 and float(count).is_integer(),
                 "invalid AdamW step counter")
        if step is not None:
            _require(float(count) == step, "optimizer step count differs")


def _validate_state(torch, model, optimizer, state, require_cuda):
    entries = _adapters(model, torch)
    _require(type(state) is dict and set(state) in (
        {"schema_version", "adapters", "optimizer", "rng"},
        {"schema_version", "adapters", "optimizer", "gradients", "rng"}), "training state schema")
    _require(type(state["schema_version"]) is int and state["schema_version"] == 1,
             "training state version")
    _require(type(state["adapters"]) is dict and set(state["adapters"]) == {n for n, _ in entries},
             "training state adapter inventory differs")
    for name, parameter in entries:
        saved = state["adapters"][name]
        _require(torch.is_tensor(saved) and saved.shape == parameter.shape and saved.dtype == torch.float32
                 and bool(torch.isfinite(saved).all()), "invalid saved adapter: " + name)
    _validate_optimizer_state(torch, state["optimizer"], entries, optimizer)
    rng = state["rng"]
    expected_rng = {"python", "cpu", "cuda0"} if require_cuda else {"python", "cpu"}
    _require(type(rng) is dict and set(rng) == expected_rng, "RNG inventory differs")
    # Validate Python state with a private generator so the global RNG is not
    # changed if a malformed state is rejected before restoration.
    private = random.Random()
    try:
        private.setstate(rng["python"])
    except (ValueError, TypeError) as error:
        raise ValueError("invalid Python RNG state") from error
    for name in expected_rng - {"python"}:
        tensor = rng[name]
        current = torch.get_rng_state() if name == "cpu" else torch.cuda.get_rng_state(0)
        _require(torch.is_tensor(tensor) and tensor.dtype == torch.uint8 and tensor.device.type == "cpu"
                 and tensor.shape == current.shape and tensor.ndim == 1, "invalid RNG state: " + name)
        # Shape alone does not validate a generator's encoded state. Probe a
        # separate generator before touching any live parameter, moment or RNG.
        try:
            generator = torch.Generator(device="cpu" if name == "cpu" else "cuda:0")
            generator.set_state(tensor)
        except (RuntimeError, ValueError, TypeError) as error:
            raise ValueError("invalid encoded RNG state: " + name) from error
    if "gradients" in state:
        _require(type(state["gradients"]) is dict and set(state["gradients"]) == {n for n, _ in entries},
                 "gradient snapshot inventory differs")
        for name, parameter in entries:
            gradient = state["gradients"][name]
            _require(gradient is None or (torch.is_tensor(gradient) and gradient.dtype == torch.float32
                     and gradient.shape == parameter.shape and bool(torch.isfinite(gradient).all())),
                     "invalid saved gradient: " + name)
    return entries


def restore_training_state(torch, model, optimizer, state, *, require_cuda=True):
    """Validate the whole bounded state before restoring existing objects."""
    entries = _validate_state(torch, model, optimizer, state, require_cuda)
    with torch.no_grad():
        for name, parameter in entries:
            parameter.copy_(state["adapters"][name])
    optimizer.load_state_dict(_clone_tree(torch, state["optimizer"]))
    gradients = state.get("gradients", {})
    for name, parameter in entries:
        gradient = gradients.get(name)
        parameter.grad = None if gradient is None else gradient.detach().to(parameter.device).clone()
    random.setstate(state["rng"]["python"])
    torch.set_rng_state(state["rng"]["cpu"])
    if require_cuda:
        torch.cuda.set_rng_state(state["rng"]["cuda0"], device=0)


def _state_matches(torch, model, optimizer, expected, require_cuda):
    entries = _adapters(model, torch)
    return (set(expected["adapters"]) == {name for name, _ in entries}
            and all(bitwise_equal_tensor(torch, expected["adapters"][name], p) for name, p in entries)
            and equal_tree(torch, expected["optimizer"], optimizer.state_dict())
            and all(equal_tree(torch, expected["gradients"][name], p.grad) for name, p in entries)
            and equal_tree(torch, expected["rng"], _rng_snapshot(torch, require_cuda)))


def _input_contract(torch, input_ids, labels):
    _require(torch.is_tensor(input_ids) and torch.is_tensor(labels)
             and input_ids.dtype == torch.long and labels.dtype == torch.long
             and input_ids.ndim == 2 and input_ids.shape[0] == 1
             and input_ids.shape == labels.shape and input_ids.shape[1] >= 2
             and input_ids.device == labels.device, "single whole-example Long IDs/labels required")
    _require(bool((input_ids >= 0).all()), "negative input token")
    selected = (labels[0] != -100).nonzero().flatten()
    _require(selected.numel() > 0 and int(selected[0]) > 0
             and torch.equal(selected, torch.arange(int(selected[0]), labels.shape[1], device=labels.device)),
             "mask must be a nonempty prompt followed by the complete supervised suffix")
    _require(torch.equal(labels[0, selected], input_ids[0, selected]), "target labels differ from input suffix")
    _require(bool((labels[0, :int(selected[0])] == -100).all()), "prompt masking differs")
    return selected


def completion_loss(torch, logits, input_ids, labels):
    """Mean causal CE over completion plus EOS, with no prompt/padding loss."""
    positions = _input_contract(torch, input_ids, labels)
    _require(torch.is_tensor(logits) and logits.dtype == torch.float32 and logits.ndim == 3
             and logits.shape[:2] == input_ids.shape and logits.shape[-1] > 0
             and logits.device == input_ids.device, "full FP32 causal logits required")
    _require(bool((input_ids < logits.shape[-1]).all()), "input token exceeds model vocabulary")
    return torch.nn.functional.cross_entropy(logits[0, positions - 1], labels[0, positions], reduction="mean")


def _checked_loss(torch, logits, ids, labels, failed_preserve=None):
    loss = completion_loss(torch, logits, ids, labels)
    positions = (labels[0] != -100).nonzero().flatten().tolist()
    # Select each preceding row independently. Do not derive the reference from
    # cross_entropy, model.loss, or the first implementation's selected tensor.
    independent = torch.stack([-logits[0, index - 1].detach().log_softmax(-1)[ids[0, index]]
                               for index in positions]).mean()
    observed, reference = float(loss.detach()), float(independent)
    error = abs(observed - reference)
    passed = math.isfinite(observed) and math.isfinite(reference) and error <= CE_ATOL
    if not passed and failed_preserve is not None:
        failed_preserve({"ce": _json_number(observed), "selected_logsoftmax": _json_number(reference),
                         "absolute_error": _json_number(error), "atol": CE_ATOL, "passed": False,
                         "supervised_positions": len(positions)})
    _require(passed,
             "completion CE differs from independently selected log-softmax")
    return loss, {"ce": observed, "selected_logsoftmax": reference, "absolute_error": error,
                  "supervised_positions": len(positions)}


def _validate_row(row, eos_id):
    _require(type(row) is dict, "row must be a validated dictionary")
    ids, labels, mask = (row.get(name) for name in ("input_ids", "labels", "attention_mask"))
    _require(all(type(value) is list for value in (ids, labels, mask))
             and len(ids) >= 2 and len(ids) == len(labels) == len(mask), "whole row vector lengths differ")
    _require(all(type(token) is int and token >= 0 for token in ids)
             and all(type(token) is int for token in labels)
             and all(type(token) is int and token == 1 for token in mask),
             "padding or invalid token/mask value")
    prompt = next((i for i, label in enumerate(labels) if label != -100), len(labels))
    _require(0 < prompt < len(labels) and labels == [-100] * prompt + ids[prompt:],
             "row must contain a masked prompt and complete supervised suffix")
    _require(ids[-1] == eos_id and eos_id not in ids[prompt:-1], "completion must end in exactly one EOS")
    for key, expected in (("prompt_tokens", prompt), ("supervised_tokens", len(ids) - prompt),
                          ("total_tokens", len(ids))):
        if key in row:
            _require(type(row[key]) is int and row[key] == expected, "row token count differs: " + key)
    for key, expected in (("prompt_ids", ids[:prompt]), ("completion_ids", ids[prompt:])):
        if key in row:
            _require(row[key] == expected, "row rendered token segment differs: " + key)


def _base_inventory(model, torch):
    inventory = {name: (p, p.data_ptr(), tuple(p.shape), p.dtype, str(p.device))
                 for name, p in model.named_parameters(remove_duplicate=False)
                 if not name.endswith((".lora_A", ".lora_B"))}
    _require(bool(inventory) and all(not p.requires_grad and p.grad is None and dtype == torch.bfloat16
                                   for p, _, _, dtype, _ in inventory.values()),
             "base must remain frozen resident BF16 with no gradients")
    return inventory


def _verify_base(model, torch, inventory):
    actual = dict(model.named_parameters(remove_duplicate=False))
    _require({name for name in actual if not name.endswith((".lora_A", ".lora_B"))} == set(inventory),
             "base parameter inventory changed")
    for name, (parameter, pointer, shape, dtype, device) in inventory.items():
        p = actual[name]
        _require(p is parameter and p.data_ptr() == pointer and tuple(p.shape) == shape
                 and p.dtype == dtype and str(p.device) == device and not p.requires_grad and p.grad is None,
                 "base object/storage/gradient changed: " + name)


def _adam(torch, parameters):
    return torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0,
                            amsgrad=False, maximize=False, foreach=False, fused=False,
                            capturable=False, differentiable=False)


def _save_checkpoint(torch, model, optimizer, destination, require_cuda):
    payload = {"schema_version": 1,
               "adapters": {name: p.detach().cpu().clone() for name, p in _adapters(model, torch)},
               "optimizer": _clone_tree(torch, optimizer.state_dict(), cpu=True),
               "rng": _rng_snapshot(torch, require_cuda)}
    _validate_state(torch, model, optimizer, payload, require_cuda)
    stream = io.BytesIO()
    torch.save(payload, stream)
    data = stream.getvalue()
    _require(0 < len(data) < MAX_CHECKPOINT_BYTES, "adapter checkpoint exceeds the frozen 32 MiB ceiling")
    # Exclusive creation preserves earlier evidence even if the caller repeats
    # the same output directory. Loading later hashes these exact file bytes.
    with destination.open("xb") as handle:
        _require(handle.write(data) == len(data), "incomplete checkpoint write")
        handle.flush()
    return {"path": str(destination.resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "schema_version": 1}


def _load_checkpoint(torch, descriptor):
    destination = Path(descriptor["path"])
    _require(destination.stat().st_size == descriptor["bytes"]
             and 0 < descriptor["bytes"] < MAX_CHECKPOINT_BYTES, "checkpoint byte count changed")
    data = destination.read_bytes()
    _require(len(data) == descriptor["bytes"] and hashlib.sha256(data).hexdigest() == descriptor["sha256"],
             "checkpoint bytes changed before deserialization")
    # Deserialize the same immutable byte object whose digest was verified, not
    # a second path open susceptible to substitution between hash and load.
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def qualify_training(torch, model, rows40, output_dir, call, guard, config, *, preserve=None):
    """Run a fixed two-update mechanics check with one independent continuation.

    All model forwards, backwards, optimizer actions, snapshot/copy boundaries,
    checkpoint operations and controlled faults go through ``call(op, fn)``.
    ``guard(stage)`` runs before and after every such operation. Exceptions
    propagate to the owning worker's failure/pending accounting; no retry occurs.
    Optional ``preserve(name, fragment)`` receives bounded detached strict-JSON
    milestones before the next phase. Failed comparisons are preserved before
    rejection, without adding any model or optimizer operations.
    """
    _require(type(config) is dict and callable(call) and callable(guard), "mechanics configuration/callbacks")
    _require(preserve is None or callable(preserve), "milestone preservation callback")
    require_cuda = config.get("require_cuda", True)
    _require(type(require_cuda) is bool, "require_cuda must be explicit boolean")
    _require(config.get("rows_per_update", SEQUENCES_PER_UPDATE) == SEQUENCES_PER_UPDATE
             and type(config.get("rows_per_update", SEQUENCES_PER_UPDATE)) is int,
             "20 whole sequences per update is fixed")
    eos_id = config.get("eos_id", 151645)
    _require(type(eos_id) is int and eos_id >= 0, "EOS ID contract")
    _require(type(rows40) is list and len(rows40) == 40, "exactly 40 validated TRAIN rows required")
    for row in rows40:
        _validate_row(row, eos_id)
    _require(torch.is_grad_enabled() and not model.training, "autograd and eval-mode deterministic model required")
    entries = _adapters(model, torch)
    count = config.get("adapter_tensors", 112)
    expected_parameters = config.get("adapter_parameters", 1146880)
    _require(type(count) is int and count > 0 and len(entries) == count, "adapter tensor count differs")
    _require(type(expected_parameters) is int and expected_parameters > 0
             and sum(p.numel() for _, p in entries) == expected_parameters, "adapter parameter count differs")
    device = entries[0][1].device
    expected_device = "cuda:0" if require_cuda else "cpu"
    _require(str(device) == expected_device and all(p.device == device for _, p in entries),
             "adapter placement differs from the explicitly scoped device")
    _require(require_cuda or not torch.cuda.is_initialized(), "CPU toy cannot have a CUDA context")
    _require(all(bool((p.detach() == 0).all()) for name, p in entries if name.endswith(".lora_B")),
             "qualification requires initial zero B factors")
    _require(sum(name.endswith(".lora_A") for name, _ in entries)
             == sum(name.endswith(".lora_B") for name, _ in entries), "partial A/B factor inventory")
    base = _base_inventory(model, torch)
    _require(all(str(p.device) == expected_device for p, _, _, _, _ in base.values()),
             "frozen base placement differs")
    parameters = [p for _, p in entries]
    destination = Path(output_dir).resolve() / "STATE1.pt"
    _require(not destination.exists(), "write-once state1 checkpoint already exists")
    operation_counts = {}

    def milestone(name, value):
        _preserve_fragment(preserve, "mechanics_" + name, value)

    def invoke(operation, function):
        guard(operation + ":before")
        try:
            value = call(operation, function)
        except BaseException as error:
            # A failed operation still gets its post-operation resource check.
            # Keep the original failure authoritative if that guard also fails.
            try:
                guard(operation + ":after")
            except BaseException as guard_error:
                error.add_note("post-operation guard also failed: " + str(guard_error))
            raise
        operation_counts[operation] = operation_counts.get(operation, 0) + 1
        guard(operation + ":after")
        return value

    optimizer = invoke("native_optimizer_construct", lambda: _adam(torch, parameters))
    initial = invoke("initial_adapter_snapshot", lambda: {name: p.detach().clone() for name, p in entries})
    report = {"mechanics_version": VERSION, "device": str(device), "sequences_per_native_update": 20,
              "adapter_tensors": len(entries), "adapter_parameters": sum(p.numel() for p in parameters),
              "completed_native_updates": 0, "completed_reference_updates": 0,
              "native_row_loss_checks": [], "reference_row_loss_checks": []}

    def gradients_valid():
        _require(all(p.grad is not None and p.grad.dtype == torch.float32 and p.grad.device == device
                     and bool(torch.isfinite(p.grad).all()) for p in parameters), "invalid accumulated adapter gradient")
        _require(any(bool((p.grad != 0).any()) for p in parameters), "adapter gradients identically zero")
        _verify_base(model, torch, base)

    def accumulate(rows, operation_prefix, reference=False):
        accumulated = None
        losses = []
        for offset, row in enumerate(rows):
            ids, labels, mask = invoke(operation_prefix + "_row_tensor_materialize", lambda: tuple(
                torch.tensor([row[field]], dtype=torch.long, device=device)
                for field in ("input_ids", "labels", "attention_mask")))
            output = invoke(operation_prefix + "_forward", lambda: model(
                input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=0))
            logits = output if torch.is_tensor(output) else output.logits
            row_index = offset + (20 if reference or report["completed_native_updates"] else 0)
            loss, check = invoke(operation_prefix + "_completion_loss_check", lambda: _checked_loss(
                torch, logits, ids, labels, failed_preserve=lambda failed: milestone(
                    operation_prefix + "_row_%02d_ce_failure" % row_index,
                    {"row_index": row_index, **failed})))
            check["row_index"] = row_index
            report[("reference" if reference else "native") + "_row_loss_checks"].append(check)
            losses.append(check["ce"])
            if reference:
                gradients = invoke("reference_autograd_grad", lambda: torch.autograd.grad(
                    loss / SEQUENCES_PER_UPDATE, parameters, retain_graph=False, create_graph=False, allow_unused=False))
                _require(all(g.dtype == torch.float32 and g.device == device and bool(torch.isfinite(g).all())
                             for g in gradients), "invalid independent reference gradients")

                def add_gradients():
                    nonlocal accumulated
                    if accumulated is None:
                        accumulated = [g.detach().clone() for g in gradients]
                    else:
                        for total, gradient in zip(accumulated, gradients):
                            total.add_(gradient)
                invoke("reference_gradient_accumulate", add_gradients)
            else:
                invoke("native_backward", lambda: (loss / SEQUENCES_PER_UPDATE).backward())
            _verify_base(model, torch, base)
            del output, logits, ids, labels, mask, loss
            guard(operation_prefix + "_row_" + str(offset) + ":complete")
        return losses, accumulated

    def native_update(number):
        gradients_valid()
        norm = invoke("native_gradient_clip", lambda: torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, error_if_nonfinite=True, foreach=False))
        _require(math.isfinite(float(norm)), "nonfinite pre-clip gradient norm")
        invoke("native_optimizer_step", optimizer.step)
        _validate_optimizer_state(torch, optimizer.state_dict(), entries, optimizer, step=number)
        report["completed_native_updates"] = number
        _require(all(bool(torch.isfinite(p).all()) for p in parameters), "optimizer produced a nonfinite adapter")
        _verify_base(model, torch, base)
        return float(norm)

    invoke("native_gradient_reset", lambda: optimizer.zero_grad(set_to_none=True))
    losses1, _ = accumulate(rows40[:20], "native")
    gradients_valid()
    initial_A_zero = all(bool((p.grad == 0).all()) for name, p in entries if name.endswith(".lora_A"))
    milestone("initial_A_gradient_check", {"completed_native_updates": 0,
              "initial_A_gradients_exactly_zero": initial_A_zero})
    _require(initial_A_zero,
             "initial zero B must produce exact zero A gradients")
    report["initial_A_gradients_exactly_zero"] = True
    report["pre_clip_norm_update1"] = native_update(1)
    changed1 = [name for name, p in entries if not bitwise_equal_tensor(torch, initial[name], p)]
    milestone("update1", {"completed_native_updates": 1, "sequence_mean_loss": sum(losses1) / 20,
              "pre_clip_norm": report["pre_clip_norm_update1"], "changed_adapters": changed1,
              "initial_A_gradients_exactly_zero": True, "any_adapter_changed": bool(changed1),
              "only_B_changed": all(name.endswith(".lora_B") for name in changed1)})
    _require(bool(changed1), "native update 1 changed no adapter")
    _require(all(name.endswith(".lora_B") for name in changed1), "zero-A-gradient first update changed A")
    report["changed_adapters_update1"] = changed1
    del initial
    invoke("native_gradient_reset", lambda: optimizer.zero_grad(set_to_none=True))
    state1 = invoke("state1_snapshot", lambda: snapshot_training_state(
        torch, model, optimizer, require_cuda=require_cuda))
    _require(all(value is None for value in state1["gradients"].values()), "state1 boundary gradients not cleared")
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = invoke("checkpoint_save", lambda: _save_checkpoint(
        torch, model, optimizer, destination, require_cuda))

    def perturb(tensor):
        with torch.no_grad():
            tensor.reshape(-1)[0].add_(1.0)

    first_name, first_parameter = entries[0]
    invoke("restore_fault_adapter", lambda: perturb(first_parameter))
    invoke("restore_fault_moment", lambda: perturb(optimizer.state[first_parameter]["exp_avg"]))
    invoke("restore_fault_python_rng", random.random)
    invoke("restore_fault_cpu_rng", lambda: torch.rand(1, device="cpu"))
    if require_cuda:
        invoke("restore_fault_cuda0_rng", lambda: torch.rand(1, device="cuda:0"))
    fault_witnesses = {
        "adapter_changed": not bitwise_equal_tensor(torch, state1["adapters"][first_name], first_parameter),
        "moment_changed": not bitwise_equal_tensor(torch, state1["optimizer"]["state"][0]["exp_avg"],
                                                    optimizer.state[first_parameter]["exp_avg"]),
        "python_rng_changed": random.getstate() != state1["rng"]["python"],
        "cpu_rng_changed": not bitwise_equal_tensor(torch, state1["rng"]["cpu"], torch.get_rng_state())}
    if require_cuda:
        fault_witnesses["cuda0_rng_changed"] = not bitwise_equal_tensor(
            torch, state1["rng"]["cuda0"], torch.cuda.get_rng_state(0))
    milestone("restore_fault_witnesses", {"checkpoint": checkpoint, "optimizer_boundary": 1,
              "witnesses": fault_witnesses, "all_components_changed": all(fault_witnesses.values())})
    _require(all(fault_witnesses.values()), "controlled faults did not change every registered component")
    payload = invoke("checkpoint_load", lambda: _load_checkpoint(torch, checkpoint))
    invoke("checkpoint_restore", lambda: restore_training_state(
        torch, model, optimizer, payload, require_cuda=require_cuda))
    del payload
    restored = invoke("state1_restore_verify", lambda: _state_matches(
        torch, model, optimizer, state1, require_cuda))
    milestone("state1_restoration", {"optimizer_boundary": 1, "before_group2": True,
              "restored_bitwise_against_detached_pre_save_snapshot": restored,
              "checkpoint": checkpoint, "restore_fault_witnesses": fault_witnesses})
    _require(restored, "state1 restoration differs from pre-save witness bitwise")
    report["checkpoint"] = checkpoint
    report["restore_fault_witnesses"] = fault_witnesses
    report["state1_restored_bitwise_before_group2"] = True

    losses_reference, reference_gradients = accumulate(rows40[20:], "reference", reference=True)
    reference_preserved = invoke("reference_state_verify", lambda: _state_matches(
        torch, model, optimizer, state1, require_cuda))
    milestone("reference_state", {"reference_rows": 20, "completed_native_updates": 1,
              "parameters_moments_gradients_rng_preserved": reference_preserved})
    _require(reference_preserved, "reference pass changed parameters/moments/gradients/RNG")
    _require(any(bool((g != 0).any()) for g in reference_gradients), "reference gradients identically zero")
    report["reference_preserved_state_and_rng"] = True
    invoke("native_gradient_reset", lambda: optimizer.zero_grad(set_to_none=True))
    losses2, _ = accumulate(rows40[20:], "native")
    gradients_valid()

    def compare_gradients():
        comparisons = []
        for (name, parameter), reference in zip(entries, reference_gradients):
            difference = (parameter.grad - reference).abs()
            comparisons.append({"name": name, "max_abs": _json_number(difference.max()),
                                "mean_abs": _json_number(difference.mean()),
                                "finite": bool(torch.isfinite(difference).all()),
                                "passed": bool(torch.allclose(parameter.grad, reference,
                                                               atol=GRAD_ATOL, rtol=GRAD_RTOL, equal_nan=False))})
        passed = all(item["passed"] and item["finite"] for item in comparisons)
        milestone("reference_gradient_comparison", {"comparisons": comparisons, "passed": passed,
                  "atol": GRAD_ATOL, "rtol": GRAD_RTOL, "sequences_per_update": 20})
        _require(passed,
                 "20-sequence gradient accumulation differs from independent reference")
        return comparisons

    report["gradient_comparisons"] = invoke("gradient_compare", compare_gradients)
    del reference_gradients
    ref_parameters = invoke("reference_parameter_clone", lambda: [
        torch.nn.Parameter(state1["adapters"][name].detach().clone()) for name, _ in entries])
    ref_optimizer = invoke("reference_optimizer_construct", lambda: _adam(torch, ref_parameters))
    invoke("reference_optimizer_load_state", lambda: ref_optimizer.load_state_dict(
        _clone_tree(torch, state1["optimizer"])))

    def copy_actual_gradients():
        for target, original in zip(ref_parameters, parameters):
            target.grad = original.grad.detach().clone()
    invoke("reference_gradient_copy", copy_actual_gradients)
    reference_norm = invoke("reference_gradient_clip", lambda: torch.nn.utils.clip_grad_norm_(
        ref_parameters, 1.0, error_if_nonfinite=True, foreach=False))
    _require(math.isfinite(float(reference_norm)), "nonfinite reference gradient norm")
    invoke("reference_optimizer_step", ref_optimizer.step)
    report["completed_reference_updates"] = 1
    report["pre_clip_norm_update2"] = native_update(2)
    changed2 = [name for name, p in entries if not bitwise_equal_tensor(torch, state1["adapters"][name], p)]
    milestone("update2", {"completed_native_updates": 2, "completed_reference_updates": 1,
              "sequence_mean_loss": sum(losses2) / 20, "pre_clip_norm": report["pre_clip_norm_update2"],
              "changed_adapters": changed2, "any_adapter_changed": bool(changed2)})

    def verify_continuation():
        equal_parameters = all(bitwise_equal_tensor(torch, left, right)
                               for left, right in zip(ref_parameters, parameters))
        equal_gradients = all(bitwise_equal_tensor(torch, left.grad, right.grad)
                              for left, right in zip(ref_parameters, parameters))
        equal_optimizer = equal_tree(torch, ref_optimizer.state_dict(), optimizer.state_dict())
        equal_rng = equal_tree(torch, state1["rng"], _rng_snapshot(torch, require_cuda))
        milestone("optimizer_continuation", {"completed_native_updates": 2, "completed_reference_updates": 1,
                  "parameters_bitwise_equal": equal_parameters, "clipped_gradients_bitwise_equal": equal_gradients,
                  "moments_and_steps_bitwise_equal": equal_optimizer, "rng_bitwise_equal": equal_rng,
                  "passed": equal_parameters and equal_gradients and equal_optimizer and equal_rng})
        _require(equal_parameters and equal_gradients,
                 "copied-gradient clipping/AdamW parameters differ bitwise")
        _require(equal_optimizer,
                 "copied-gradient AdamW moments/steps differ bitwise")
        _require(equal_rng,
                 "group 2 or copied-gradient continuation changed RNG")
        return True

    report["copied_gradient_optimizer_continuation_bitwise"] = invoke(
        "optimizer_continuation_verify", verify_continuation)
    _require(bool(changed2), "native update 2 changed no adapter")
    report["changed_adapters_update2"] = changed2
    report["sequence_mean_losses"] = [sum(losses1) / 20, sum(losses2) / 20]
    report["reference_sequence_mean_loss"] = sum(losses_reference) / 20
    report["optimizer_group_fields"] = sorted(optimizer.param_groups[0])
    report["optimizer_settings"] = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}
    report["operation_counts"] = operation_counts
    report["native_forward_input_positions"] = sum(len(row["input_ids"]) for row in rows40)
    report["reference_forward_input_positions"] = sum(len(row["input_ids"]) for row in rows40[20:])
    report["native_supervised_positions"] = sum(sum(label != -100 for label in row["labels"]) for row in rows40)
    report["reference_supervised_positions"] = sum(sum(label != -100 for label in row["labels"])
                                                  for row in rows40[20:])
    report["base_parameter_objects_pointers_dtypes_and_grad_none_preserved"] = True
    report["status"] = "MECHANICAL_CHECKS_PASSED"
    _verify_base(model, torch, base)
    guard("training_mechanics:complete")
    return report
