"""Internal injected-Torch adapter backend; no framework import or launch CLI.

This implementation is unqualified until an owned physical worker measures it.
The caller owns source/runtime/loader admission and accounting around methods.
Synthetic dispatch tests do not prove tensor physics or native resume integrity.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random

from experiments.post_training_mimo_centralized_v1.scripts import e1_data, native_backend_contract as c
from experiments.post_training_mimo_centralized_v1.scripts.e1_training_accounting import E1CallAccounting

GROUP_FIELDS = {"params", "lr", "betas", "eps", "weight_decay", "foreach", "fused", "capturable",
                "differentiable", "amsgrad", "maximize", "decoupled_weight_decay"}
PRIMITIVES = ("optimizer_construct", "forward", "backward", "gradient_reset", "gradient_clip",
              "optimizer_step", "checkpoint_save", "checkpoint_load")


def require(value, message):
    if not value:
        raise ValueError(message)


def base_shapes():
    shapes = {"model.embed_tokens.weight": [151936, 1024], "model.norm.weight": [1024]}
    suffixes = {"self_attn.q_proj.weight": [2048, 1024], "self_attn.k_proj.weight": [1024, 1024],
        "self_attn.v_proj.weight": [1024, 1024], "self_attn.o_proj.weight": [1024, 2048],
        "self_attn.q_norm.weight": [128], "self_attn.k_norm.weight": [128],
        "mlp.gate_proj.weight": [3072, 1024], "mlp.up_proj.weight": [3072, 1024],
        "mlp.down_proj.weight": [1024, 3072], "input_layernorm.weight": [1024], "post_attention_layernorm.weight": [1024]}
    shapes.update({f"model.layers.{layer}.{name}": shape for layer in range(28) for name, shape in suffixes.items()})
    return shapes


class LimitedWriter:
    def __init__(self, stream): self.stream = stream
    def write(self, data):
        require(self.stream.tell() + len(data) <= c.MAX_ARTIFACT_BYTES, "adapter checkpoint exceeds32MiB")
        return self.stream.write(data)
    def flush(self): return self.stream.flush()
    def tell(self): return self.stream.tell()


class NativeTrainingBackend:
    execution_scope = "UNQUALIFIED_NATIVE_COMPONENT"

    @classmethod
    def create(cls, *args, **kwargs):
        """Owned workers must use this factory to retain constructor failures.

        The original exception is re-raised, with detached primitive evidence
        when its type permits attributes. Attachment failure never replaces the
        original error; the durable primitive event sink remains authoritative.
        """
        instance = cls.__new__(cls)
        try:
            instance.__init__(*args, **kwargs)
        except BaseException as error:
            try:
                if hasattr(instance, "_primitive_counts"):
                    snapshot = instance.primitive_operation_snapshot()
                else:
                    snapshot = {"scope": "INTERNAL_TORCH_PRIMITIVE_UNQUALIFIED", "last_call_id": 0,
                        "operations": {name: dict.fromkeys(("attempted", "returned", "failed", "pending"), 0) for name in PRIMITIVES},
                        "pending_calls": [], "native_qualified": False, "native_launch_authorized": False, "E1_authorized": False}
                error.native_backend_primitive_accounting = deepcopy(snapshot)
                error.native_backend_constructor_failed = True
            except BaseException:
                pass
            raise
        return instance

    def __init__(self, torch, model, *, helpers, guard, emit, repository_root, approved_run_relative,
                 bindings, source_base_hashes, expected_rng_shapes):
        require(callable(guard) and callable(emit) and all(callable(getattr(helpers, name, None)) for name in
                ("install_lora", "completion_loss", "tensor_digest")), "injected qualified helpers and guard required")
        self.torch, self.model, self.helpers, self.guard = torch, model, helpers, guard
        self.emit = emit
        self.completed_updates = self.accumulated = 0
        self._primitive_last = 0; self._primitive_pending = {}
        self._primitive_counts = {name: dict.fromkeys(("attempted", "returned", "failed", "pending"), 0) for name in PRIMITIVES}
        self.root, self.run = Path(repository_root).resolve(), approved_run_relative
        self.bindings, self.base_hashes, self.rng_shapes = deepcopy(bindings), deepcopy(source_base_hashes), deepcopy(expected_rng_shapes)
        c.fields(self.bindings, c.BINDING_FIELDS, "BINDINGS")
        for value in self.bindings.values(): c.sha(value)
        require(self.bindings["weight_sha256"] == c.WEIGHT_SHA256 and self.bindings["data_sha256"] == c.DATA_SHA256
                and self.bindings["plan_sha256"] == c.PLAN_SHA256, "fixed Titan/data/plan bindings required")
        probe = c.safe_artifact_path(self.root, self.run, self.run + "/ownership_probe.json")
        require(probe.parent.is_dir(), "caller-owned run directory must already exist")
        require(getattr(model.config, "_attn_implementation", None) == "eager"
                and getattr(model, "is_gradient_checkpointing", None) is False, "eager path without checkpointing required")
        model.eval()
        named = list(model.named_parameters())
        require(len(named) == 422 and len({name for name, _ in named}) == 422 and len({id(p) for _, p in named}) == 422,
                "unique310base plus112adapter parameters required")
        self.adapters = {name: p for name, p in named if p.requires_grad}
        self.base = {name: p for name, p in named if not p.requires_grad}
        require(sum(p.numel() for p in self.base.values()) == 596049920, "base parameter count differs")
        inventory = [self._parameter_meta(name, p) for name, p in self.adapters.items()]
        require(model.lm_head.weight is model.model.embed_tokens.weight, "embedding tie missing")
        for name, p in self.base.items():
            original = c.canonical_base_name(name)
            self._tensor(p, base_shapes()[original], "bfloat16", "cuda:0", finite=False)
            require(p.grad is None, "frozen base gradient present")
        summary = {"unique_parameters": 596049920, "dtype": "bfloat16", "device": "cuda:0", "requires_grad": False,
            "weight_sha256": self.bindings["weight_sha256"], "embedding_tied": True, "serialized_tensor_count": len(self.base),
            "parameter_names": list(self.base), "source_tensor_sha256": self.base_hashes,
            "tied_alias": {"alias": "lm_head.weight", "target": "model.embed_tokens.weight", "same_object": True, "serialized_separately": False}}
        inventory_sha = c.validate_model_metadata(inventory, summary, source_tensor_sha256=self.base_hashes)
        require(inventory_sha == self.bindings["inventory_sha256"], "ordered adapter inventory binding differs")
        for item, p in zip(c.expected_adapter_inventory(), self.adapters.values()):
            self._tensor(p, item["shape"], "float32", "cuda:0")
        sites = [(name, module) for name, module in model.named_modules() if name.endswith((".q_proj", ".v_proj"))]
        require(len(sites) == 56, "exact56 q/v adapter sites required")
        for name, module in sites:
            require(module.lora_A is self.adapters[name + ".lora_A"] and module.lora_B is self.adapters[name + ".lora_B"]
                    and module.base.bias is None, "q/v wrapper or bias differs")
        self._verify_base()
        self.optimizer = self._primitive("optimizer_construct", lambda: torch.optim.AdamW(list(self.adapters.values()),
            lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, foreach=None, fused=None,
            capturable=False, differentiable=False), source_operation="constructor")
        self.clipped = False
        self._reset_ready = False
        self._pending_loss = None
        self._saved, self._published = {}, {}
        self._optimizer_metadata()
        c.validate_rng_metadata(self._rng_metadata(self._rng_state()), expected_rng_shapes=self.rng_shapes)
        guard("backend_constructed_unqualified")

    def _primitive(self, operation, function, *, source_operation, on_return=None):
        require(operation in PRIMITIVES and callable(function), "registered primitive required")
        self._primitive_last += 1
        call = {"call_id": self._primitive_last, "operation": operation, "source_operation": source_operation,
                "optimizer_boundary": self.completed_updates, "sequence_in_update": self.accumulated + 1}
        counts = self._primitive_counts[operation]
        counts["attempted"] += 1; counts["pending"] += 1
        self._primitive_pending[call["call_id"]] = call
        self.emit("primitive_operation_attempt", scope="INTERNAL_TORCH_PRIMITIVE_UNQUALIFIED", **call)
        try:
            value = function()
        except BaseException as error:
            counts["pending"] -= 1; counts["failed"] += 1; self._primitive_pending.pop(call["call_id"])
            self.emit("primitive_operation_failed", scope="INTERNAL_TORCH_PRIMITIVE_UNQUALIFIED",
                      error_type=type(error).__name__, error_message=str(error)[:1024], **call)
            raise
        counts["pending"] -= 1; counts["returned"] += 1; self._primitive_pending.pop(call["call_id"])
        if on_return is not None: on_return()
        # Publication is outside the primitive exception span. A returned Torch
        # call remains returned even if its event or a later guard cannot publish.
        self.emit("primitive_operation_returned", scope="INTERNAL_TORCH_PRIMITIVE_UNQUALIFIED", **call)
        return value

    def primitive_operation_snapshot(self):
        return deepcopy({"scope": "INTERNAL_TORCH_PRIMITIVE_UNQUALIFIED", "last_call_id": self._primitive_last,
            "operations": self._primitive_counts, "pending_calls": list(self._primitive_pending.values()),
            "native_qualified": False, "native_launch_authorized": False, "E1_authorized": False})

    def _parameter_meta(self, name, p):
        return {"name": name, "shape": list(p.shape), "dtype": str(p.dtype).removeprefix("torch."),
                "device": str(p.device), "requires_grad": p.requires_grad, "numel": p.numel()}

    def _tensor(self, p, shape, dtype, device, *, finite=True):
        require(self.torch.is_tensor(p) and list(p.shape) == shape and p.numel() == math.prod(shape)
                and str(p.dtype).removeprefix("torch.") == dtype and str(p.device) == device, "tensor shape/dtype/device differs")
        if finite: require(bool(self.torch.isfinite(p).all()), "nonfinite tensor")

    def _verify_base(self):
        self.guard("before_frozen_base_hash")
        for index, (name, p) in enumerate(self.base.items()):
            require(not p.requires_grad and p.grad is None and self.helpers.tensor_digest(self.torch, p)
                    == self.base_hashes[c.canonical_base_name(name)], "frozen base bytes/gradient changed")
            if index % 16 == 0: self.guard("frozen_base_hash_" + str(index))
        require(self.model.lm_head.weight is self.model.model.embed_tokens.weight, "embedding tie changed")

    def _gradients(self):
        require(all(p.grad is None and not p.requires_grad for p in self.base.values()), "gradient leaked to base")
        for p in self.adapters.values():
            require(p.requires_grad and p.grad is not None, "missing adapter gradient")
            self._tensor(p.grad, list(p.shape), "float32", "cuda:0")

    def reset_gradients(self):
        self.guard("before_gradient_reset")
        require(self.accumulated == 0 and not self.clipped and not self._reset_ready and self._pending_loss is None,
                "reset once at an untouched optimizer boundary")
        require(all(p.grad is None for p in self.base.values()), "base gradient leak")
        self._primitive("gradient_reset", lambda: self.optimizer.zero_grad(set_to_none=True), source_operation="reset_gradients")
        self._reset_ready = True
        self.guard("after_gradient_reset")

    def forward_loss(self, row):
        self.guard("before_sequence_forward")
        require(self._reset_ready and self._pending_loss is None and self.accumulated < 20
                and not self.clipped and self.torch.is_grad_enabled(), "forward phase or autograd differs")
        component = row.get("component")
        require(type(component) is str and component in e1_data.COUNTS, "explicit scheduled component required")
        e1_data.validate_array(row, component)
        require(row["nonpadding_tokens"] <= 512 and row.get("training_eligibility") is False, "complete eligible CE512 TRAIN row required")
        require(not set(row).intersection(("position_ids", "packing", "packed", "cu_seqlens", "past_key_values", "cache_position", "inputs_embeds")), "custom forward fields forbidden")
        ids = self.torch.tensor([row["input_ids"]], dtype=self.torch.long, device="cuda:0")
        labels = self.torch.tensor([row["labels"]], dtype=self.torch.long, device="cuda:0")
        require(not self.model.training and not self.model.is_gradient_checkpointing, "qualified eval math path changed")
        output = self._primitive("forward", lambda: self.model(ids, use_cache=False, logits_to_keep=0), source_operation="forward_loss")
        logits = output.logits
        loss = self.helpers.completion_loss(self.torch, logits, ids, labels)
        require(loss.requires_grad and bool(self.torch.isfinite(loss).all()), "finite differentiable sequence loss required")
        self._pending_loss = loss
        self.guard("after_sequence_forward")
        return loss

    def loss_value(self, loss):
        value = float(loss.detach().item())
        require(math.isfinite(value) and value >= 0, "finite nonnegative loss required")
        return value

    def backward(self, loss, *, scale):
        require(type(scale) is float and scale == 1 / 20 and loss is self._pending_loss
                and self._reset_ready and self.accumulated < 20 and not self.clipped, "exact sequence-mean scale/phase required")
        self.guard("before_sequence_backward")
        def acknowledged():
            self.accumulated += 1
            self._pending_loss = None
        self._primitive("backward", lambda: (loss * scale).backward(), source_operation="backward", on_return=acknowledged)
        self._gradients()
        self.guard("after_sequence_backward")

    def clip_gradients(self, *, max_norm):
        require(type(max_norm) is float and max_norm == 1.0 and self.accumulated == 20
                and self._pending_loss is None and not self.clipped, "clip once after20 sequences")
        self._gradients(); self.guard("before_gradient_clip")
        norm = self._primitive("gradient_clip", lambda: self.torch.nn.utils.clip_grad_norm_(
            list(self.adapters.values()), max_norm, error_if_nonfinite=True), source_operation="clip_gradients")
        value = float(norm.detach().item()) if self.torch.is_tensor(norm) else float(norm)
        require(math.isfinite(value) and value >= 0, "finite clip norm required")
        self.clipped = True; self.guard("after_gradient_clip")
        return value

    def optimizer_step(self, *, lr):
        require(self.accumulated == 20 and self.clipped and self.completed_updates < 256, "one optimizer step aftercomplete clipped accumulation")
        update = self.completed_updates + 1
        expected = 1e-4 if update == 13 else (1e-4 * update / 13 if update < 13 else 1e-4 * (256 - update) / 243)
        require(type(lr) in (int, float) and math.isfinite(lr) and lr == expected, "fixed update LR differs")
        self._gradients(); self.guard("before_optimizer_step")
        self.optimizer.param_groups[0]["lr"] = lr
        def acknowledged():
            self.completed_updates += 1
            self.accumulated = 0; self.clipped = False
            self._reset_ready = False
        self._primitive("optimizer_step", self.optimizer.step, source_operation="optimizer_step", on_return=acknowledged)
        for p in self.adapters.values(): self._tensor(p, list(p.shape), "float32", "cuda:0")
        self._optimizer_metadata(); self.guard("after_optimizer_step")

    def _optimizer_metadata(self):
        groups = self.optimizer.param_groups
        require(len(groups) == 1 and len(groups[0]["params"]) == 112
                and all(a is b for a, b in zip(groups[0]["params"], self.adapters.values())), "optimizer ordered parameter group differs")
        group = groups[0]
        require(set(group) == GROUP_FIELDS and group["decoupled_weight_decay"] is True,
                "pinned Torch2.10 AdamW group fields differ")
        metadata = {"metadata_scope": c.WITNESS_SCOPE, "name": "AdamW", "betas": list(group["betas"]),
            "epsilon": group["eps"], "weight_decay": group["weight_decay"],
            **{name: group[name] for name in ("foreach", "fused", "capturable", "differentiable", "amsgrad", "maximize")},
            "parameter_groups": [{"parameter_names": list(self.adapters), "learning_rate": group["lr"]}], "state": []}
        require(set(self.optimizer.state) == (set(self.adapters.values()) if self.completed_updates else set()), "optimizer state inventory differs")
        for name, p in self.adapters.items():
            if not self.completed_updates: break
            state = self.optimizer.state[p]
            require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "Adam state fields differ")
            def witness(value):
                return {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."), "device": str(value.device),
                        "numel": value.numel(), "requires_grad": value.requires_grad, "finite": bool(self.torch.isfinite(value).all())}
            metadata["state"].append({"name": name, "exp_avg": witness(state["exp_avg"]), "exp_avg_sq": witness(state["exp_avg_sq"]),
                "step": witness(state["step"]) | {"value": float(state["step"].item())}})
        c.validate_adamw_metadata(metadata, completed_updates=self.completed_updates)
        return metadata

    def _cpu(self, value):
        if self.torch.is_tensor(value): return value.detach().cpu().clone()
        if type(value) is dict: return {key: self._cpu(item) for key, item in value.items()}
        if type(value) in (tuple, list): return type(value)(self._cpu(item) for item in value)
        require(value is None or type(value) in (str, int, float, bool), "unsupported checkpoint value")
        if type(value) is float: require(math.isfinite(value), "nonfinite checkpoint scalar")
        return value

    def _rng_state(self):
        require(self.torch.cuda.device_count() == 1, "only registered CUDA0 RNG inventory supported")
        return {"python": random.getstate(), "torch_cpu": self._cpu(self.torch.get_rng_state()),
                "cuda0": self._cpu(self.torch.cuda.get_rng_state(0))}

    def _rng_metadata(self, state):
        def byte_meta(tensor, shape):
            self._tensor(tensor, shape, "uint8", "cpu")
            require(not tensor.requires_grad, "RNG byte tensor must be detached")
            return {"shape": list(tensor.shape), "dtype": "uint8", "device": "cpu", "numel": tensor.numel(),
                    "sha256": self.helpers.tensor_digest(self.torch, tensor)}
        python = state["python"]
        require(type(python) is tuple and len(python) == 3 and python[0] == 3 and type(python[1]) is tuple and len(python[1]) == 625
                and all(type(value) is int for value in python[1]), "Python RNG state shape differs")
        random.Random().setstate(python)  # Validate semantics without mutating the training RNG.
        return {"metadata_scope": c.WITNESS_SCOPE,
            "python": {"algorithm": "MT19937", "state_version": python[0], "internal_state_words": len(python[1]),
                       "state_sha256": c.digest(python), "gaussian_cache": python[2]},
            "torch_cpu": byte_meta(state["torch_cpu"], self.rng_shapes["torch_cpu"]),
            "cuda": [{"device": "cuda:0", "state": byte_meta(state["cuda0"], self.rng_shapes["cuda:0"])}]}

    def _boundary(self, cursor, costs):
        # Component boundaries may include separately registered small mechanical
        # diagnostics. This helper grants no plan/seed/native launch admission.
        require(type(cursor) is dict and type(cursor.get("completed_optimizer_steps")) is int
                and 0 <= cursor["completed_optimizer_steps"] <= 256 and type(cursor.get("accumulated_sequences")) is int,
                "typed complete boundary cursor required")
        require(self.accumulated == 0 and not self.clipped and self._pending_loss is None
                and not self._reset_ready and cursor["completed_optimizer_steps"] == self.completed_updates
                and cursor["plan_sha256"] == self.bindings["plan_sha256"] and cursor["accumulated_sequences"] == 0,
                "complete bound optimizer boundary required")
        require(cursor["identities"]["data_sha256"] == self.bindings["data_sha256"]
                and cursor["identities"]["parent_sha256"] == self.bindings["weight_sha256"], "cursor source bindings differ")
        E1CallAccounting.from_snapshot(lambda *args, **kwargs: None, costs)
        for name, count in (("forward", 20 * self.completed_updates), ("backward", 20 * self.completed_updates),
                            ("optimizer_step", self.completed_updates), ("gradient_reset", self.completed_updates), ("gradient_clip", self.completed_updates)):
            require(costs["operations"][name] == {"attempted": count, "returned": count, "failed": 0, "pending": 0}, "boundary costs differ:" + name)

    def _artifact_pin(self, path):
        require(0 < path.stat().st_size <= c.MAX_ARTIFACT_BYTES, "bounded adapter artifact required")
        before = c._path_identity(path); h = hashlib.sha256(); size = 0
        with path.open("rb") as stream:
            require(c._stream_identity(stream) == before, "artifact opened identity differs")
            while block := stream.read(1024 * 1024):
                size += len(block); require(size <= c.MAX_ARTIFACT_BYTES, "artifact hash budget exceeded")
                h.update(block)
            require(c._stream_identity(stream) == before, "artifact changed during hash")
        require(c._path_identity(path) == before and size == before[2], "artifact path/size changed")
        return {"sha256": h.hexdigest(), "bytes": size}

    def save_checkpoint(self, cursor, pre_save_costs):
        self._boundary(cursor, pre_save_costs); self.guard("before_adapter_checkpoint_save")
        self._verify_base()
        rng = self._rng_state()
        metadata = {"parameters": {"metadata_scope": c.WITNESS_SCOPE, "inventory": [self._parameter_meta(n, p) for n, p in self.adapters.items()]},
                    "optimizer": self._optimizer_metadata(), "training_rng": self._rng_metadata(rng)}
        payload = {"schema_version": 1, "bindings": deepcopy(self.bindings), "parameter_order": list(self.adapters),
            "cursor": deepcopy(cursor), "pre_save_cost_snapshot": deepcopy(pre_save_costs),
            "parameters": {name: self._cpu(p) for name, p in self.adapters.items()},
            "optimizer": self._cpu(self.optimizer.state_dict()), "training_rng": rng}
        relative = self.run + f"/adapter_{self.completed_updates:03d}.pt"
        path = c.safe_artifact_path(self.root, self.run, relative)
        with path.open("xb") as stream:
            self._primitive("checkpoint_save", lambda: self.torch.save(payload, LimitedWriter(stream)), source_operation="save_checkpoint")
            stream.flush(); os.fsync(stream.fileno())
        del payload, rng
        artifact = {"relative_path": relative, **self._artifact_pin(path),
                    "format": "torch_save_adapter_only_v1", "stored_sections": ["parameters", "optimizer", "training_rng"],
                    "base_parameters_stored": False, "gradient_tensors_stored": False}
        state = {role: {"artifact": deepcopy(artifact), "bindings": deepcopy(self.bindings), "metadata": metadata[role]}
                 for role in ("parameters", "optimizer", "training_rng")}
        c.validate_checkpoint_descriptors(state, repository_root=self.root, approved_run_relative=self.run,
            expected_bindings=self.bindings, expected_inventory_sha256=self.bindings["inventory_sha256"],
            completed_updates=self.completed_updates, expected_rng_shapes=self.rng_shapes)
        self._saved[artifact["sha256"]] = {"state": deepcopy(state), "cursor": deepcopy(cursor), "pre_costs": deepcopy(pre_save_costs)}
        self.guard("after_adapter_checkpoint_save")
        return state

    def publish_checkpoint(self, envelope):
        self.guard("before_checkpoint_publication")
        c.fields(envelope, {"schema_version", "kind", "plan_sha256", "cursor", "cost_snapshot", "backend_state", "save_call_id",
                            "native_resume_qualified", "E1_authorized"}, "ENGINE_ENVELOPE")
        require(type(envelope["schema_version"]) is int and envelope["schema_version"] == 1
                and envelope["kind"] == "E1_CONTROLLED_BOUNDARY_COMPONENT_CHECKPOINT_V2"
                and envelope["native_resume_qualified"] is False and envelope["E1_authorized"] is False
                and envelope["plan_sha256"] == self.bindings["plan_sha256"], "unqualified bound envelope required")
        state = envelope["backend_state"]; artifact = state["parameters"]["artifact"]
        saved = self._saved.get(artifact["sha256"])
        require(saved is not None and c.canonical(state) == c.canonical(saved["state"])
                and c.canonical(envelope["cursor"]) == c.canonical(saved["cursor"]), "saved state/cursor differs")
        expected = deepcopy(saved["pre_costs"])
        expected["last_call_id"] += 1
        expected["operations"]["checkpoint_save"]["attempted"] += 1
        expected["operations"]["checkpoint_save"]["returned"] += 1
        require(c.canonical(envelope["cost_snapshot"]) == c.canonical(expected)
                and type(envelope["save_call_id"]) is int and envelope["save_call_id"] == expected["last_call_id"], "exact post-save returned costs required")
        self._boundary(envelope["cursor"], envelope["cost_snapshot"])
        relative = self.run + f"/adapter_{self.completed_updates:03d}.envelope.json"
        path = c.safe_artifact_path(self.root, self.run, relative)
        raw = c.canonical(envelope) + b"\n"; require(len(raw) <= 1024 * 1024, "bounded checkpoint envelope required")
        with path.open("xb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        receipt = {"relative_path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                   "save_call_id": envelope["save_call_id"], "native_resume_qualified": False, "E1_authorized": False}
        self._published[artifact["sha256"]] = deepcopy(receipt)
        self.guard("after_checkpoint_publication")
        return receipt

    def restore_checkpoint(self, state):
        self.guard("before_controlled_adapter_restore")
        artifact = state["parameters"]["artifact"]; published = self._published.get(artifact["sha256"])
        require(published is not None, "same-process durably published checkpoint required")
        manifest = c.safe_artifact_path(self.root, self.run, published["relative_path"])
        require(type(published["bytes"]) is int and 0 < published["bytes"] <= 1024 * 1024
                and manifest.stat().st_size == published["bytes"], "bounded published manifest required")
        manifest_identity = c._path_identity(manifest)
        with manifest.open("rb") as stream:
            require(c._stream_identity(stream) == manifest_identity, "manifest opened identity differs")
            raw_manifest = stream.read(published["bytes"] + 1)
            require(c._stream_identity(stream) == manifest_identity, "manifest changed during read")
        require(c._path_identity(manifest) == manifest_identity, "manifest path changed")
        require(len(raw_manifest) == published["bytes"] and hashlib.sha256(raw_manifest).hexdigest() == published["sha256"], "published envelope bytes changed")
        envelope = json.loads(raw_manifest)
        require(c.canonical(envelope["backend_state"]) == c.canonical(state), "published state reference differs")
        updates = envelope["cursor"]["completed_optimizer_steps"]
        require(type(updates) is int and updates == self.completed_updates and self.accumulated == 0
                and not self.clipped and not self._reset_ready and self._pending_loss is None,
                "only same-process controlled saved-boundary restore supported")
        c.validate_checkpoint_descriptors(state, repository_root=self.root, approved_run_relative=self.run,
            expected_bindings=self.bindings, expected_inventory_sha256=self.bindings["inventory_sha256"],
            completed_updates=updates, expected_rng_shapes=self.rng_shapes)
        path = c.safe_artifact_path(self.root, self.run, artifact["relative_path"])
        before = c._path_identity(path); buffer = io.BytesIO(); h = hashlib.sha256()
        with path.open("rb") as stream:
            require(c._stream_identity(stream) == before, "restore opened identity differs")
            while block := stream.read(1024 * 1024):
                require(buffer.tell() + len(block) <= c.MAX_ARTIFACT_BYTES, "restore exceeds boundedbytes")
                buffer.write(block); h.update(block)
            require(c._stream_identity(stream) == before, "restore artifact changed during read")
        require(c._path_identity(path) == before and buffer.tell() == artifact["bytes"] and h.hexdigest() == artifact["sha256"], "restore verified bytes differ")
        buffer.seek(0)
        try:
            payload = self._primitive("checkpoint_load", lambda: self.torch.load(buffer, map_location="cpu", weights_only=True), source_operation="restore_checkpoint")
        finally:
            buffer.close()
        self._validate_payload(payload, state, envelope)
        with self.torch.no_grad():
            for name, p in self.adapters.items(): p.copy_(payload["parameters"][name])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.completed_updates = updates; self.accumulated = 0; self.clipped = False
        self._pending_loss = None; self._reset_ready = False
        self._optimizer_metadata()
        random.setstate(payload["training_rng"]["python"])
        self.torch.set_rng_state(payload["training_rng"]["torch_cpu"])
        self.torch.cuda.set_rng_state(payload["training_rng"]["cuda0"], 0)
        self._primitive("gradient_reset", lambda: self.optimizer.zero_grad(set_to_none=True), source_operation="restore_checkpoint")
        self._verify_base(); self.guard("after_controlled_adapter_restore")

    def _validate_payload(self, payload, state, envelope):
        require(type(payload) is dict and set(payload) == {"schema_version", "bindings", "parameter_order", "cursor", "pre_save_cost_snapshot",
                                                         "parameters", "optimizer", "training_rng"}, "adapter-only payload fields required")
        saved = self._saved[state["parameters"]["artifact"]["sha256"]]
        require(type(payload["schema_version"]) is int and payload["schema_version"] == 1
                and c.canonical(payload["bindings"]) == c.canonical(self.bindings)
                and payload["parameter_order"] == list(self.adapters)
                and c.canonical(payload["cursor"]) == c.canonical(envelope["cursor"])
                and c.canonical(payload["pre_save_cost_snapshot"]) == c.canonical(saved["pre_costs"]), "payload source/cursor/cost bindings differ")
        require(type(payload["parameters"]) is dict and list(payload["parameters"]) == list(self.adapters), "physical adapter inventory differs")
        for name, value in payload["parameters"].items():
            self._tensor(value, list(self.adapters[name].shape), "float32", "cpu")
            require(not value.requires_grad, "saved parameter must be detached")
        optimizer = payload["optimizer"]
        require(type(optimizer) is dict and set(optimizer) == {"state", "param_groups"} and len(optimizer["param_groups"]) == 1,
                "physical optimizer envelope differs")
        group = optimizer["param_groups"][0]
        require(type(group) is dict and set(group) == GROUP_FIELDS
                and type(group["params"]) is list and all(type(value) is int for value in group["params"])
                and group["params"] == list(range(112)) and group["decoupled_weight_decay"] is True,
                "physical optimizer parameter order/schema differs")
        for field, value in {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0, "foreach": None, "fused": None,
                             "capturable": False, "differentiable": False, "amsgrad": False, "maximize": False}.items():
            require(c.canonical(group[field]) == c.canonical(value), "physical optimizer setting differs:" + field)
        updates = envelope["cursor"]["completed_optimizer_steps"]
        require(set(optimizer["state"]) == (set(range(112)) if updates else set()), "physical optimizer state inventory differs")
        for index, item in optimizer["state"].items():
            require(type(index) is int and set(item) == {"step", "exp_avg", "exp_avg_sq"}, "physical Adam state fields differ")
            shape = list(list(self.adapters.values())[index].shape)
            for key in ("exp_avg", "exp_avg_sq"):
                self._tensor(item[key], shape, "float32", "cpu"); require(not item[key].requires_grad, "detached Adam moments required")
            self._tensor(item["step"], [], "float32", "cpu")
            require(float(item["step"].item()) == updates and not item["step"].requires_grad, "physical Adam step differs")
        require(type(group["lr"]) in (int, float) and math.isfinite(group["lr"])
                and group["lr"] == state["optimizer"]["metadata"]["parameter_groups"][0]["learning_rate"], "physical saved LR differs")
        c.fields(payload["training_rng"], {"python", "torch_cpu", "cuda0"}, "PHYSICAL_RNG_FIELDS")
        require(c.canonical(self._rng_metadata(payload["training_rng"])) == c.canonical(state["training_rng"]["metadata"]), "physical RNG byte identity differs")
