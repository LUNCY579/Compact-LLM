"""Injected Torch study backend; no framework import, base loader or launch CLI.

CPU_RANDOM_TOY is restricted to a separately declared tiny random CPU model.
UNQUALIFIED_NATIVE_IMPLEMENTATION requires the pinned resident Titan inventory,
but provides neither execution permission nor native qualification. Every Torch
boundary uses the caller's primitive journal and resource guard. The future
chunk runner must bind the latest cumulative costs and publication envelope;
this component cannot establish crash recovery or authorize rollback.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re

import native_loader as loader
import native_mechanics as mechanics
import native_precision as precision

SCOPES = ("CPU_RANDOM_TOY", "UNQUALIFIED_NATIVE_IMPLEMENTATION")
KIND = "SYNTH_STUDY_TORCH_BOUNDARY_V1"
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
BINDING_FIELDS = {"completed_updates", "consumed_occurrences", "input_tokens", "supervised_tokens",
                  "consumed_order_sha256", "next_update_ordinal", "learning_rate"}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def learning_rate(ordinal):
    require(type(ordinal) is int and 1 <= ordinal <= 27, "global update ordinal must be 1..27")
    peak = Fraction(1, 10000)
    return float(peak / 2 if ordinal == 1 else peak if ordinal == 2 else peak * Fraction(27 - ordinal, 25))


def hash64(value, label):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, label + " SHA256")
    return value


class BoundedBytesIO(io.BytesIO):
    def write(self, data):
        require(self.tell() + len(data) <= MAX_CHECKPOINT_BYTES, "checkpoint exceeds 32 MiB")
        return super().write(data)


@dataclass(frozen=True)
class BoundarySnapshot:
    """Immutable tensor archive bytes with a detached small JSON descriptor."""
    data: bytes
    descriptor_json: str

    @property
    def descriptor(self):
        return json.loads(self.descriptor_json)


class StudyTorchBackend:
    """Use only with a separately authorized caller; this is not a study runner.

    The injected model must be raw resident frozen BF16 without existing compute
    wrappers or adapters. ``call(operation, function)`` must journal attempted,
    returned, failed and pending primitive calls without erasing actual returns.
    ``guard(stage)`` owns current resource admission. Native source_hashes come
    from the separately qualified loader, never from a checkpoint warmstart.
    """
    native_launch_authorized = False
    training_authorized = False
    native_resume_qualified = False

    def __init__(self, torch, model, call, guard, *, arm, seed, recipe_sha256, cohort_sha256,
                 execution_scope, source_hashes=None, toy_profile=None):
        require(execution_scope in SCOPES, "explicit CPU_RANDOM_TOY or UNQUALIFIED_NATIVE_IMPLEMENTATION scope")
        require(arm in ("M1", "M2") and type(seed) is int and seed == 101, "adapted arm and frozen seed101")
        require(callable(call) and callable(guard), "primitive journal and resource guard required")
        self.torch, self.model, self.call, self.guard = torch, model, call, guard
        self.arm, self.seed, self.execution_scope = arm, seed, execution_scope
        self.recipe_sha256 = hash64(recipe_sha256, "recipe")
        self.cohort_sha256 = hash64(cohort_sha256, "cohort")
        self.require_cuda = execution_scope == "UNQUALIFIED_NATIVE_IMPLEMENTATION"
        self.primitive_counts = {}
        self._failed = False
        self.completed_updates = self.input_tokens = self.supervised_tokens = 0
        self.order, self._phase, self._sequences = [], "fresh", 0
        self._pending = self._loss = None
        self.backend_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        require(not any(n.endswith((".lora_A", ".lora_B")) for n, _ in model.named_parameters()),
                "fresh raw base required; qualification STATE1 warmstart forbidden")
        require(not any(getattr(m, "_synth_fp32_compute", False) or getattr(m, "_synth_qv_lora", False)
                        for _, m in model.named_modules()), "fresh unwrapped base required")
        if self.require_cuda:
            require(toy_profile is None and type(source_hashes) is dict and bool(source_hashes),
                    "native requires pinned source tensor hashes and no toy profile")
            require(torch.cuda.is_initialized() and torch.cuda.current_device() == 0,
                    "native requires separately owned live CUDA0; no context is initialized here")
            self.source_hashes = deepcopy(source_hashes)
            self.expected_shapes = loader._adapter_shapes()
            self.layers, self.linear_sites, self.eos_id, self.pad_id, self.vocab_size = 28, 197, 151645, 151643, 151936
            self._invoke("native_base_inventory_validate", lambda: loader._base_inventory(torch, model, allow_adapters=False))
            self._invoke("native_config_validate", lambda: loader._validate_config(model.config))
            self.base_count = 596049920
        else:
            require(source_hashes is None and type(toy_profile) is dict, "CPU random toy needs a separate explicit profile")
            fields = {"expected_linear_sites", "expected_layers", "expected_base_parameters", "expected_adapter_shapes",
                      "eos_id", "pad_id", "vocab_size"}
            require(set(toy_profile) == fields, "exact CPU toy profile schema")
            profile = deepcopy(toy_profile)
            require(not torch.cuda.is_initialized(), "CPU random toy must not initialize CUDA")
            require(type(profile["expected_layers"]) is int and 1 <= profile["expected_layers"] <= 2
                    and type(profile["expected_linear_sites"]) is int and 0 < profile["expected_linear_sites"] <= 32,
                    "tiny CPU toy site bounds")
            require(type(profile["expected_base_parameters"]) is int and 0 < profile["expected_base_parameters"] <= 4096,
                    "tiny CPU toy base bound")
            self.layers, self.linear_sites, self.base_count = (profile[k] for k in
                ("expected_layers", "expected_linear_sites", "expected_base_parameters"))
            self.eos_id, self.pad_id, self.vocab_size = (profile[k] for k in ("eos_id", "pad_id", "vocab_size"))
            require(all(type(v) is int for v in (self.eos_id, self.pad_id, self.vocab_size))
                    and 2 <= self.vocab_size <= 32 and 0 <= self.eos_id < self.vocab_size
                    and 0 <= self.pad_id < self.vocab_size and self.eos_id != self.pad_id, "tiny CPU toy token profile")
            self.expected_shapes = profile["expected_adapter_shapes"]
            names = {f"model.layers.{i}.self_attn.{p}.lora_{factor}" for i in range(self.layers)
                     for p in ("q_proj", "v_proj") for factor in ("A", "B")}
            require(type(self.expected_shapes) is dict and set(self.expected_shapes) == names
                    and all(type(shape) is list and len(shape) == 2 and all(type(d) is int and 0 < d <= 16 for d in shape)
                            and (shape[0] == 8 if name.endswith(".lora_A") else shape[1] == 8)
                            for name, shape in self.expected_shapes.items()), "tiny rank8 q/v shape inventory")
            require(all(p.device.type == "cpu" for p in model.parameters()), "CPU random toy must reside on CPU")
            self.source_hashes = None
        self.base_inventory = self._invoke("base_storage_inventory", lambda: mechanics._base_inventory(model, torch))
        unique = {id(value[0]): value[0] for value in self.base_inventory.values()}
        require(sum(p.numel() for p in unique.values()) == self.base_count, "frozen BF16 base parameter count")
        self._toy_base = None if self.require_cuda else self._invoke("toy_base_snapshot", lambda:
            {name: item[0].detach().clone() for name, item in self.base_inventory.items()})
        self._verify_base(content=True)
        self._invoke("fresh_training_seed", self._seed)
        self.compute_profile = self._invoke("fp32_compute_install", lambda:
            precision.install_fp32_compute(model, torch, expected_linear_sites=self.linear_sites))
        self.adapter_profile = self._invoke("rank8_qv_adapter_install", lambda:
            precision.install_qv_lora(model, torch, rank=8, alpha=16, expected_layers=self.layers))
        self.entries = self._invoke("adapter_inventory_validate", lambda: mechanics._adapters(model, torch))
        inventory = self._invoke("adapter_metadata_validate", lambda: precision.adapter_inventory(model, torch))
        require({item["name"]: item["shape"] for item in inventory} == self.expected_shapes, "exact rank8 q/v shapes")
        require(len(inventory) == (112 if self.require_cuda else 4 * self.layers)
                and sum(item["parameters"] for item in inventory) == (1146880 if self.require_cuda else
                    sum(math.prod(s) for s in self.expected_shapes.values())), "full adapter tensor/parameter inventory")
        require(all(str(p.device) == ("cuda:0" if self.require_cuda else "cpu") for _, p in self.entries),
                "adapter device inventory")
        self.adapter_inventory_sha256 = digest(inventory)
        self._invoke("zero_B_validate", lambda: require(precision.zero_lora_status(model, torch)["all_B_zero"], "fresh zeroB required"))
        self._invoke("model_eval_set", model.eval)
        self.activation_hooks = self._invoke("fp32_activation_assertions_install", lambda:
            precision.install_fp32_activation_assertions(model, torch))
        self.parameters = [p for _, p in self.entries]
        self.optimizer = self._invoke("optimizer_construct", lambda: mechanics._adam(torch, self.parameters))
        require(self.optimizer.state_dict()["state"] == {}, "fresh optimizer must contain no moments")
        self._verify_base(content=True)
        self._fresh_attestation = self._invoke("fresh_start_attestation", self._attest)

    def _invoke(self, operation, function):
        require(not self._failed, "failed primitive forbids backend continuation")
        try:
            self.guard("study_before_" + operation)
        except BaseException:
            self._failed = True
            raise
        counts = self.primitive_counts.setdefault(operation, {k: 0 for k in ("attempted", "returned", "failed", "pending")})
        counts["attempted"] += 1
        counts["pending"] += 1
        executed = False
        def actual():
            nonlocal executed
            require(not executed, "journal must invoke each primitive exactly once")
            executed = True
            try:
                result = function()
            except BaseException:
                counts["pending"] -= 1
                counts["failed"] += 1
                raise
            counts["pending"] -= 1
            counts["returned"] += 1
            return result
        try:
            value = self.call(operation, actual)
            require(executed and counts["pending"] == 0, "journal must return the executed primitive")
            self.guard("study_after_" + operation)
        except BaseException:
            self._failed = True
            raise
        return value

    def primitive_operation_snapshot(self):
        return deepcopy(self.primitive_counts)

    @property
    def fresh_start_attestation(self):
        return deepcopy(self._fresh_attestation)

    def _seed(self):
        random.seed(self.seed)
        self.torch.random.default_generator.manual_seed(self.seed)
        if self.require_cuda:
            self.torch.cuda.manual_seed(self.seed)  # current owned device is CUDA0

    def _tensor_hash(self, tensor):
        return hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(self.torch.uint8).cpu().numpy().tobytes()).hexdigest()

    def _attest(self):
        result = {"kind": "FRESH_ZERO_B_SEEDED_STUDY_START_NOT_NATIVE_PARITY", "seed": self.seed,
                  "base_parameters": self.base_count, "base_dtype": "bfloat16", "rank": 8, "alpha": 16,
                  "adapters": {name: self._tensor_hash(p) for name, p in self.entries}, "optimizer_state_empty": True,
                  "qualification_STATE1_used": False, "native_output_parity_qualified": False,
                  "scope": self.execution_scope, "base_content_sha256": digest(self.source_hashes) if self.require_cuda else
                      digest({n: self._tensor_hash(p) for n, p in self._toy_base.items()})}
        result["sha256"] = digest(result)
        return result

    def _verify_base(self, *, content=False):
        self._invoke("base_storage_validate", lambda: mechanics._verify_base(self.model, self.torch, self.base_inventory))
        if content:
            if self.require_cuda:
                self._invoke("native_base_content_verify", lambda:
                    loader.verify_frozen_base(self.torch, self.model, self.source_hashes, self.call, self.guard))
            else:
                self._invoke("toy_base_content_verify", lambda: require(all(mechanics.bitwise_equal_tensor(self.torch, saved,
                    dict(self.model.named_parameters(remove_duplicate=False))[name]) for name, saved in self._toy_base.items()),
                    "frozen base content changed"))

    def _row(self, row):
        mechanics._validate_row(row, self.eos_id)
        require(type(row.get("row_id")) is str and bool(row["row_id"]), "row identity required")
        require(len(row["input_ids"]) <= 512 and self.pad_id not in row["input_ids"]
                and all(token < self.vocab_size for token in row["input_ids"]), "whole bounded nonpadding vocabulary row")
        return deepcopy(row)

    def reset_gradients(self, update_ordinal):
        require(not self._failed and self._phase in ("fresh", "complete") and update_ordinal == self.completed_updates + 1,
                "reset only at next global update boundary")
        learning_rate(update_ordinal)
        self._verify_base(content=not self.require_cuda)
        self._invoke("gradient_reset", lambda: self.optimizer.zero_grad(set_to_none=True))
        self._active_ordinal, self._sequences, self._phase = update_ordinal, 0, "forward"

    def forward(self, row):
        require(not self._failed and self._phase == "forward" and self._sequences < 20, "exact20 forward/loss/backward sequence order")
        row = self._row(row)
        require(self.input_tokens + len(row["input_ids"]) <= 131072, "nonpadding training token ceiling")
        self._verify_base(content=not self.require_cuda)
        device = "cuda:0" if self.require_cuda else "cpu"
        ids, labels, mask = self._invoke("row_tensor_materialize", lambda: tuple(self.torch.tensor([row[key]], dtype=self.torch.long,
            device=device) for key in ("input_ids", "labels", "attention_mask")))
        output = self._invoke("training_forward", lambda: self.model(input_ids=ids, attention_mask=mask,
            use_cache=False, logits_to_keep=0))
        self._verify_base(content=not self.require_cuda)
        self._pending = {"logits": output.logits, "input_ids": ids, "labels": labels, "row": row, "row_sha256": digest(row)}
        self._phase = "loss"
        return self._pending

    def completion_loss(self, output, row):
        require(not self._failed and self._phase == "loss" and output is self._pending and digest(self._row(row)) == output["row_sha256"],
                "loss must use exact current row and forward")
        self._invoke("forward_array_integrity_validate", lambda: require(
            output["input_ids"].detach().cpu().tolist() == [output["row"]["input_ids"]]
            and output["labels"].detach().cpu().tolist() == [output["row"]["labels"]],
            "forward tensor IDs/labels differ from frozen row"))
        self._loss, self.last_loss_check = self._invoke("completion_CE_selected_NLL_check", lambda:
            mechanics._checked_loss(self.torch, output["logits"], output["input_ids"], output["labels"]))
        self._phase = "backward"
        return self._loss

    def backward(self, loss, *, scale):
        require(not self._failed and self._phase == "backward" and loss is self._loss and type(scale) is float and scale == 1 / 20,
                "backward requires checked loss scaled exactly1/20")
        self._invoke("training_backward", lambda: (loss * scale).backward())
        row = self._pending["row"]
        self.order.append(row["row_id"])
        self.input_tokens += len(row["input_ids"])
        self.supervised_tokens += sum(v != -100 for v in row["labels"])
        self._pending = self._loss = None
        self._sequences += 1
        self._phase = "clip" if self._sequences == 20 else "forward"

    def finite_clip_gradients(self, *, max_norm):
        require(not self._failed and self._phase == "clip" and type(max_norm) in (int, float) and max_norm == 1.0, "clip exactly once at norm1 after20")
        def clip():
            mechanics._adapters(self.model, self.torch)
            require(all(p.grad is not None and p.grad.dtype == self.torch.float32 and p.grad.shape == p.shape
                        and bool(self.torch.isfinite(p.grad).all()) for p in self.parameters), "finite confined full FP32 adapter gradients")
            return float(self.torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=1.0, error_if_nonfinite=True, foreach=False))
        self._verify_base(content=not self.require_cuda)
        norm = self._invoke("finite_gradient_clip", clip)
        require(math.isfinite(norm) and norm >= 0, "finite preclip norm")
        self._phase = "step"
        return norm

    def optimizer_step(self, *, lr, update_ordinal):
        require(not self._failed and self._phase == "step" and type(update_ordinal) is int and update_ordinal == self._active_ordinal
                and type(lr) is float and lr == learning_rate(update_ordinal), "fixed global LR ordinal and full update required")
        self.optimizer.param_groups[0]["lr"] = lr
        self._invoke("optimizer_step", self.optimizer.step)
        self._invoke("optimizer_boundary_validate", lambda: self._validate_optimizer(self.optimizer.state_dict(), update_ordinal, lr))
        self._invoke("consumed_gradient_clear", lambda: self.optimizer.zero_grad(set_to_none=True))
        self._verify_base(content=True)
        self.completed_updates, self._phase = update_ordinal, "complete"

    def _validate_optimizer(self, state, completed, lr):
        require(type(state) is dict and set(state) == {"state", "param_groups"}
                and type(state["param_groups"]) is list and len(state["param_groups"]) == 1
                and type(state["param_groups"][0]) is dict
                and type(state["param_groups"][0].get("lr")) is float
                and state["param_groups"][0]["lr"] == lr, "saved optimizer schema/global learning rate")
        view = mechanics._clone_tree(self.torch, state)
        view["param_groups"][0]["lr"] = 1e-4  # reuse fixed-profile validator; global LR separately bound above
        mechanics._validate_optimizer_state(self.torch, view, self.entries, self.optimizer, step=completed)

    def _binding(self, supplied):
        require(type(supplied) is dict and set(supplied) == BINDING_FIELDS, "exact study cursor/token/globalLR binding schema")
        expected = {"completed_updates": self.completed_updates, "consumed_occurrences": len(self.order),
                    "input_tokens": self.input_tokens, "supervised_tokens": self.supervised_tokens,
                    "consumed_order_sha256": digest(self.order), "next_update_ordinal": self.completed_updates + 1,
                    "learning_rate": learning_rate(self.completed_updates)}
        require(supplied == expected and all(type(supplied[k]) is int for k in BINDING_FIELDS - {"consumed_order_sha256", "learning_rate"})
                and type(supplied["learning_rate"]) is float, "exact complete occurrence/token/globalLR cursor")
        return deepcopy(supplied)

    def _metadata(self, binding):
        return {"schema_version": 1, "kind": KIND, "experiment": "SYNTH-CRS-001", "scope": self.execution_scope,
                "arm": self.arm, "seed": self.seed, "recipe_sha256": self.recipe_sha256, "cohort_sha256": self.cohort_sha256,
                "backend_source_sha256": self.backend_source_sha256, "fresh_start_attestation_sha256": self._fresh_attestation["sha256"],
                "adapter_parameter_order": [name for name, _ in self.entries],
                "adapter_inventory_sha256": self.adapter_inventory_sha256,
                "binding": deepcopy(binding), "native_launch_authorized": False, "training_authorized": False,
                "native_resume_qualified": False, "qualification_STATE1_used": False}

    def _validate_state(self, state, metadata):
        binding = metadata["binding"]
        completed, lr = binding["completed_updates"], binding["learning_rate"]
        require(type(completed) is int and 1 <= completed <= 27 and type(lr) is float
                and lr == learning_rate(completed), "state update/global LR binding")
        require(not self.require_cuda or completed in (9, 18, 27), "native study state only at9/18/27")
        require(binding["consumed_occurrences"] == completed * 20 and binding["next_update_ordinal"] == completed + 1
                and type(binding["input_tokens"]) is int and 0 < binding["input_tokens"] <= 131072
                and type(binding["supervised_tokens"]) is int and 0 < binding["supervised_tokens"] <= binding["input_tokens"],
                "state complete cursor/token dose")
        hash64(binding["consumed_order_sha256"], "consumed order")
        require(type(state) is dict and set(state) == {"schema_version", "adapters", "optimizer", "gradients", "rng"}, "complete study tensor state")
        require(type(state["rng"]) is dict and type(state["rng"].get("python")) is tuple
                and len(state["rng"]["python"]) == 3, "complete Python RNG tuple schema")
        try:
            random.Random().setstate(state["rng"]["python"])
        except (ValueError, TypeError, IndexError, OverflowError) as error:
            raise ValueError("invalid encoded Python RNG state") from error
        self._validate_optimizer(state["optimizer"], completed, lr)
        view = mechanics._clone_tree(self.torch, state)
        view["optimizer"]["param_groups"][0]["lr"] = 1e-4
        mechanics._validate_state(self.torch, self.model, self.optimizer, view, self.require_cuda)
        require(all(g is None for g in state["gradients"].values()), "study checkpoint requires all gradients None")
        require(all(t.device.type == "cpu" for t in state["adapters"].values())
                and all(item[key].device.type == "cpu" for item in state["optimizer"]["state"].values()
                        for key in ("step", "exp_avg", "exp_avg_sq")), "serialized state tensors must reside on CPU")

    def snapshot_boundary(self, binding):
        require(not self._failed and self._phase == "complete" and self.completed_updates > 0 and self._pending is None and self._loss is None,
                "checkpoint requires completed update and empty accumulation")
        require(not self.require_cuda or self.completed_updates in (9, 18, 27), "native study checkpoint only at9/18/27")
        binding = self._binding(binding)
        self._verify_base(content=True)
        state = self._invoke("boundary_tensor_snapshot", lambda: mechanics._clone_tree(self.torch,
            mechanics.snapshot_training_state(self.torch, self.model, self.optimizer, require_cuda=self.require_cuda), cpu=True))
        metadata = self._metadata(binding)
        self._invoke("boundary_state_validate", lambda: self._validate_state(state, metadata))
        def serialize():
            stream = BoundedBytesIO()
            self.torch.save({"metadata": metadata, "state": state, "occurrence_order": list(self.order)}, stream)
            data = stream.getvalue()
            require(0 < len(data) <= MAX_CHECKPOINT_BYTES, "bounded nonempty checkpoint bytes")
            return data
        data = self._invoke("checkpoint_serialize", serialize)
        descriptor = {"schema_version": 1, "kind": KIND, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "metadata": metadata, "metadata_sha256": digest(metadata)}
        return BoundarySnapshot(data, canonical(descriptor))

    def save_checkpoint(self, path, binding):
        path = Path(path).resolve()
        require("state1" not in path.name.lower(), "qualification STATE1 filename forbidden")
        snapshot = self.snapshot_boundary(binding)
        def write():
            with path.open("xb") as stream:
                require(stream.write(snapshot.data) == len(snapshot.data), "incomplete checkpoint write")
                stream.flush()
                os.fsync(stream.fileno())
        self._invoke("checkpoint_write", write)
        descriptor = snapshot.descriptor
        descriptor["path"] = str(path)
        return descriptor

    def _read(self, descriptor, expected_binding):
        require(type(descriptor) is dict and set(descriptor) == {"schema_version", "kind", "path", "bytes", "sha256", "metadata", "metadata_sha256"},
                "study descriptor kind/schema; qualification STATE1 forbidden")
        require(type(descriptor["schema_version"]) is int and descriptor["schema_version"] == 1 and descriptor["kind"] == KIND,
                "study checkpoint kind; qualification STATE1 forbidden")
        path = Path(descriptor["path"])
        require(path.is_absolute() and "state1" not in path.name.lower(), "owned absolute study checkpoint; STATE1 forbidden")
        require(type(expected_binding) is dict and set(expected_binding) == BINDING_FIELDS, "expected binding schema")
        metadata = descriptor["metadata"]
        require(canonical(metadata) == canonical(self._metadata(expected_binding))
                and descriptor["metadata_sha256"] == digest(metadata),
                "checkpoint arm/seed/recipe/cohort/source/start/cursor binding")
        hash64(descriptor["sha256"], "checkpoint")
        require(type(descriptor["bytes"]) is int and 0 < descriptor["bytes"] <= MAX_CHECKPOINT_BYTES, "checkpoint byte ceiling")
        def read():
            with path.open("rb") as stream:
                require(os.fstat(stream.fileno()).st_size == descriptor["bytes"], "checkpoint byte count changed")
                data = stream.read(descriptor["bytes"] + 1)
            require(len(data) == descriptor["bytes"] and hashlib.sha256(data).hexdigest() == descriptor["sha256"],
                    "checkpoint bytes changed before deserialization")
            return data
        data = self._invoke("checkpoint_read_verified_bytes", read)
        payload = self._invoke("checkpoint_weights_only_load", lambda: self.torch.load(io.BytesIO(data), map_location="cpu", weights_only=True))
        require(type(payload) is dict and set(payload) == {"metadata", "state", "occurrence_order"}
                and canonical(payload["metadata"]) == canonical(metadata),
                "serialized study metadata differs from descriptor")
        order = payload["occurrence_order"]
        require(type(order) is list and all(type(value) is str and bool(value) for value in order)
                and len(order) == expected_binding["consumed_occurrences"]
                and digest(order) == expected_binding["consumed_order_sha256"], "serialized exact occurrence order differs")
        self._invoke("checkpoint_loaded_state_validate", lambda: self._validate_state(payload["state"], metadata))
        return payload

    def restore_checkpoint(self, descriptor, expected_binding):
        require(not self._failed and self._phase in ("fresh", "complete"), "restore cannot interrupt partial accumulation")
        require(type(expected_binding) is dict and type(expected_binding.get("completed_updates")) is int
                and expected_binding["completed_updates"] >= self.completed_updates, "older checkpoint rollback forbidden")
        if self.completed_updates:
            require(expected_binding == self._binding(expected_binding), "same backend restore requires current completed cursor")
        payload = self._read(deepcopy(descriptor), deepcopy(expected_binding))
        self._verify_base(content=True)
        state = payload["state"]
        def adapters():
            with self.torch.no_grad():
                for name, parameter in self.entries:
                    parameter.copy_(state["adapters"][name])
        self._invoke("checkpoint_adapter_restore", adapters)
        self._invoke("checkpoint_optimizer_restore", lambda: self.optimizer.load_state_dict(mechanics._clone_tree(self.torch, state["optimizer"])))
        self._invoke("checkpoint_gradient_clear", lambda: self.optimizer.zero_grad(set_to_none=True))
        self._invoke("checkpoint_python_RNG_restore", lambda: random.setstate(state["rng"]["python"]))
        self._invoke("checkpoint_CPU_RNG_restore", lambda: self.torch.set_rng_state(state["rng"]["cpu"]))
        if self.require_cuda:
            self._invoke("checkpoint_CUDA0_RNG_restore", lambda: self.torch.cuda.set_rng_state(state["rng"]["cuda0"], device=0))
        self._invoke("checkpoint_bitwise_restore_verify", lambda: require(mechanics._state_matches(
            self.torch, self.model, self.optimizer, state, self.require_cuda), "restored tensor/moment/gradient/RNG state differs bitwise"))
        self._verify_base(content=True)
        binding = payload["metadata"]["binding"]
        self.completed_updates, self.input_tokens, self.supervised_tokens = (binding[k] for k in
            ("completed_updates", "input_tokens", "supervised_tokens"))
        # Bounded row identities permit exact cumulative order hashing after a
        # cross-process continuation; tensor values never become JSON lists.
        self.order = list(payload["occurrence_order"])
        self._phase, self._sequences, self._pending, self._loss = "complete", 0, None, None
        return {"status": "RESTORED_UNQUALIFIED_BACKEND_COMPONENT", "binding": deepcopy(binding),
                "native_resume_qualified": False, "training_authorized": False}

    def close(self):
        for handle in self.activation_hooks["handles"]:
            handle.remove()
