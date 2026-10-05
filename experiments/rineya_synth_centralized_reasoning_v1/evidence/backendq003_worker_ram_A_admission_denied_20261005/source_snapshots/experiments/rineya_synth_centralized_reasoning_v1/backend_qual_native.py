"""Staged injected native qualification adapter; no framework import or CLI.

The caller owns fresh admission, the live owned CUDA0 context, source/runtime
pins, the model-prehook dose meter and durable evidence. This component grants
no native launch, study warmstart, training, evaluation or scoring authority.
Nothing in this module executes without an explicitly injected backend/Torch.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from types import MethodType

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1] if HERE.name == "backend_qual_work" and HERE.parent.name == "artifacts" else HERE
sys.path.insert(0, str(PACKAGE))
import native_mechanics as mechanics
import native_precision as precision
import study_torch_backend as study

MAX_HASH_CHUNK_BYTES = 64 * 1024**2
SCOPE = "UNQUALIFIED_NATIVE_BACKEND_QUALIFICATION_ADAPTER"
WITNESS_KIND = "SYNTH_BACKEND_QUAL_WITNESS_NOT_STUDY_CHECKPOINT"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Reference:
    """Opaque detached FP32 gradients; tensor values never become JSON lists."""
    parameter_order: tuple
    gradients: dict
    row_checks: tuple
    input_positions: int
    preserved_state_sha256: str


class NativeAdapter:
    scope = SCOPE
    expected_gradient_count = 112
    native_launch_authorized = False
    study_warmstart_authorized = False

    def __init__(self, torch, backend, call, guard, dose):
        require(callable(call) and callable(guard), "shared primitive journal/resource guard")
        require(backend.execution_scope == "UNQUALIFIED_NATIVE_IMPLEMENTATION" and backend.require_cuda is True,
                "native CUDA0 implementation required; no CPU/native fallback")
        require(backend.torch is torch and backend.call is call and backend.guard is guard,
                "same injected backend/framework/journal/guard")
        self.torch, self.backend, self.call, self.guard, self.dose = torch, backend, call, guard, dose
        self.model, self.optimizer = backend.model, backend.optimizer
        self.entries = list(backend.entries)
        self.parameters = [p for _, p in self.entries]
        self._invoke("qualification_adapter_inventory", self._inventory)

    def _invoke(self, operation, function):
        self.guard("qualification_before_" + operation)
        value = self.call(operation, function)
        self.guard("qualification_after_" + operation)
        return value

    def _inventory(self):
        torch = self.torch
        require(torch.cuda.is_initialized() and torch.cuda.current_device() == 0,
                "separately owned live CUDA0 context required")
        observed = mechanics._adapters(self.model, torch)
        require(len(observed) == 112 and sum(p.numel() for _, p in observed) == 1146880
                and [n for n, _ in observed] == [n for n, _ in self.entries]
                and all(a is b for (_, a), (_, b) in zip(observed, self.entries))
                and all(str(p.device) == "cuda:0" for _, p in observed),
                "exact112 FP32 rank8 adapter inventory/parameter order")
        require(all(p.grad is None for p in self.parameters), "complete boundary requires gradients None")
        return True

    def _hash_tensor(self, tensor):
        """Hash actual bytes with dtype-aware chunks, including full FP32 tensors."""
        torch = self.torch
        require(torch.is_tensor(tensor) and tensor.is_contiguous(), "contiguous tensor required for bounded hash")
        width = tensor.element_size()
        require(type(width) is int and 0 < width <= MAX_HASH_CHUNK_BYTES, "typed tensor byte width")
        stride = max(1, MAX_HASH_CHUNK_BYTES // width)
        flat, value = tensor.detach().reshape(-1), hashlib.sha256()
        for offset in range(0, flat.numel(), stride):
            self.guard("qualification_tensor_hash_chunk_before")
            part = flat[offset:offset + stride].contiguous().view(torch.uint8).cpu().numpy()
            view = memoryview(part)
            require(view.c_contiguous and view.nbytes <= MAX_HASH_CHUNK_BYTES, "bounded actual tensor hash buffer")
            value.update(view)
            del view, part
            self.guard("qualification_tensor_hash_chunk_after")
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "bytes": tensor.numel() * width,
                "sha256": value.hexdigest()}

    def _state(self):
        self._inventory()
        state = mechanics.snapshot_training_state(self.torch, self.model, self.optimizer, require_cuda=True)
        completed = self.backend.completed_updates
        require(type(completed) is int and 1 <= completed <= 27, "qualification witness requires completed optimizer state")
        self.backend._validate_optimizer(state["optimizer"], completed, study.learning_rate(completed))
        require(all(v is None for v in state["gradients"].values()) and set(state["rng"]) == {"python", "cpu", "cuda0"},
                "all gradients None and exact Python/CPU/CUDA0 RNG inventory")
        return state

    def _state_witness(self, state, binding):
        completed = self.backend.completed_updates
        require(type(binding) is dict and set(binding) == study.BINDING_FIELDS
                and binding == self.backend._binding(binding), "exact current complete cursor/token/order/LR binding")
        require(binding["completed_updates"] == completed, "witness global ordinal")
        moments = {}
        for index, item in state["optimizer"]["state"].items():
            name = self.entries[index][0]
            moments[name] = {key: self._hash_tensor(item[key]) for key in ("step", "exp_avg", "exp_avg_sq")}
        groups = deepcopy(state["optimizer"]["param_groups"])
        result = {"schema_version": 1, "kind": WITNESS_KIND, "scope": SCOPE, "arm": self.backend.arm, "seed": self.backend.seed,
                  "recipe_sha256": self.backend.recipe_sha256, "cohort_sha256": self.backend.cohort_sha256,
                  "backend_source_sha256": self.backend.backend_source_sha256,
                  "binding": deepcopy(binding), "parameter_order": [n for n, _ in self.entries],
                  "adapters": {name: self._hash_tensor(p) for name, p in state["adapters"].items()},
                  "optimizer_moments_and_steps": moments, "optimizer_groups": groups,
                  "gradients_all_None": True,
                  "rng": {"python_sha256": digest(state["rng"]["python"]),
                          "cpu": self._hash_tensor(state["rng"]["cpu"]),
                          "cuda0": self._hash_tensor(state["rng"]["cuda0"])},
                  "occurrence_order_sha256": digest(self.backend.order),
                  "base_source_tensor_hash_map_sha256": digest(self.backend.source_hashes),
                  "base_frozen_source_tensors": 310, "base_parameters": 596049920,
                  "maximum_tensor_hash_chunk_bytes": MAX_HASH_CHUNK_BYTES,
                  "native_launch_authorized": False, "study_warmstart_authorized": False}
        result["sha256"] = digest(result)
        return result

    def witness(self, binding):
        # Backend._validate_state restricts native checkpoints to9/18/27. The
        # detached qualification witness at10 must not weaken that restriction.
        self.backend._verify_base(content=True)
        state = self._invoke("qualification_state_snapshot", self._state)
        return self._invoke("qualification_state_hash_witness", lambda: self._state_witness(state, binding))

    def _row_tensors(self, row):
        row = self.backend._row(deepcopy(row))
        return row, tuple(self.torch.tensor([row[key]], dtype=self.torch.long, device="cuda:0")
                          for key in ("input_ids", "labels", "attention_mask"))

    def _logits(self, output, length, *, requires_grad):
        torch = self.torch
        value = output.logits
        require(torch.is_tensor(value) and value.shape == (1, length, 151936)
                and value.dtype == torch.float32 and str(value.device) == "cuda:0"
                and value.requires_grad is requires_grad and value.is_contiguous(), "full FP32 CUDA0 logits contract")
        flat = value.detach().reshape(-1)
        stride = MAX_HASH_CHUNK_BYTES // value.element_size()
        for offset in range(0, flat.numel(), stride):
            self.guard("qualification_logit_finite_chunk")
            require(bool(torch.isfinite(flat[offset:offset + stride]).all()), "nonfinite full-logit chunk")
        return value

    def zero_parity(self, row):
        require(self.backend.completed_updates == 0 and self.backend._phase == "fresh",
                "physical zero parity requires a fresh qualification backend")
        self._invoke("qualification_parity_inventory", self._inventory)
        zero = self._invoke("qualification_zero_B_query", lambda: precision.zero_lora_status(self.model, self.torch))
        require(zero["all_B_zero"] is True and zero["qv_site_count"] == 56, "all56 exact zero B factors required")
        before = self._invoke("qualification_parity_state_snapshot", lambda:
            mechanics.snapshot_training_state(self.torch, self.model, self.optimizer, require_cuda=True))
        parsed, (ids, labels, mask) = self._invoke("qualification_parity_row_materialize", lambda: self._row_tensors(row))
        sites = [(name, module) for name, module in self.model.named_modules() if getattr(module, "_synth_qv_lora", False)]
        expected = {f"model.layers.{i}.self_attn.{p}" for i in range(28) for p in ("q_proj", "v_proj")}
        require(len(sites) == 56 and {n for n, _ in sites} == expected, "exact56 qualified q/v module sites")
        saved = [(module, "forward" in module.__dict__, module.__dict__.get("forward"), module.forward)
                 for _, module in sites]
        def dense(module, inputs):
            return precision._linear(self.torch, inputs, module.weight, module.bias)
        def bypass():
            for module, _, _, _ in saved:
                module.forward = MethodType(dense, module)
        def restore():
            for module, had, value, _ in saved:
                if had:
                    module.forward = value
                elif "forward" in module.__dict__:
                    del module.forward
            for module, had, value, original in saved:
                require(("forward" in module.__dict__) is had and
                        (module.__dict__.get("forward") is value if had else
                         module.forward.__func__ is original.__func__ and module.forward.__self__ is original.__self__),
                        "exact prior forward attribute restoration")
        try:
            self._invoke("qualification_qv_physical_bypass", bypass)
            with self.torch.no_grad():
                output = self._invoke("zero_parity_base_forward", lambda:
                    self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True))
                baseline = self._invoke("qualification_parity_base_logits_validate", lambda:
                    self._logits(output, len(parsed["input_ids"]), requires_grad=False))
                del output
        finally:
            # Restoration happens even when a journal/guard fails after the
            # baseline returns. The idempotent final cleanup changes only Python
            # method attributes and runs even if a failed guard/sink prevents
            # the journaled cleanup callback; no extra tensor/model call occurs.
            try:
                self._invoke("qualification_qv_forward_restore", restore)
            finally:
                restore()
        with self.torch.no_grad():
            output = self._invoke("zero_parity_adapter_forward", lambda:
                self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True))
            adapted = self._invoke("qualification_parity_adapter_logits_validate", lambda:
                self._logits(output, len(parsed["input_ids"]), requires_grad=False))
            del output
        def compare():
            equal = bool(self.torch.equal(baseline, adapted))
            maximum = 0.0
            if not equal:
                left, right = baseline.reshape(-1), adapted.reshape(-1)
                stride = MAX_HASH_CHUNK_BYTES // baseline.element_size()
                for offset in range(0, left.numel(), stride):
                    self.guard("qualification_parity_compare_chunk")
                    maximum = max(maximum, float((left[offset:offset+stride] - right[offset:offset+stride]).abs().max()))
            return {"equal": equal, "max_error": maximum, "all_B_zero": True, "qv_sites": 56,
                    "prior_forward_attributes_restored": True, "native_launch_authorized": False,
                    "study_warmstart_authorized": False}
        result = self._invoke("qualification_zero_output_compare", compare)
        self._invoke("qualification_parity_state_preserved", lambda: require(mechanics._state_matches(
            self.torch, self.model, self.optimizer, before, True), "physical parity changed parameters/moments/gradients/RNG"))
        self.backend._verify_base(content=True)
        require(result["equal"] and result["max_error"] == 0.0, "physical zero-B parity failed")
        return result

    def prepare_reference(self, rows):
        require(type(rows) is list and len(rows) == 20 and self.backend._phase == "complete",
                "reference requires exactly20 whole rows at a complete boundary")
        parsed = [self.backend._row(deepcopy(row)) for row in rows]
        before = self._invoke("qualification_reference_state_snapshot", self._state)
        binding = {"completed_updates": self.backend.completed_updates, "consumed_occurrences": len(self.backend.order),
                   "input_tokens": self.backend.input_tokens, "supervised_tokens": self.backend.supervised_tokens,
                   "consumed_order_sha256": study.digest(self.backend.order),
                   "next_update_ordinal": self.backend.completed_updates + 1,
                   "learning_rate": study.learning_rate(self.backend.completed_updates)}
        preserved = self._invoke("qualification_reference_before_hash", lambda: self._state_witness(before, binding))
        gradients = self._invoke("reference_gradient_construct", lambda:
            {name: self.torch.zeros_like(p, dtype=self.torch.float32, device="cuda:0") for name, p in self.entries})
        checks, positions = [], 0
        try:
            for row in parsed:
                row, (ids, labels, mask) = self._invoke("reference_row_tensor_materialize", lambda row=row: self._row_tensors(row))
                output = self._invoke("reference_forward", lambda:
                    self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True))
                logits = self._invoke("qualification_reference_logits_validate", lambda:
                    self._logits(output, len(row["input_ids"]), requires_grad=True))
                del output
                loss, check = self._invoke("reference_completion_loss_check", lambda:
                    mechanics._checked_loss(self.torch, logits, ids, labels))
                check = {"row_id": row["row_id"], **check}
                checks.append(check); positions += len(row["input_ids"])
                pieces = self._invoke("reference_autograd_grad", lambda: self.torch.autograd.grad(
                    loss * .05, self.parameters, retain_graph=False, create_graph=False, allow_unused=False))
                def accumulate():
                    require(len(pieces) == 112, "all112 independent reference gradients required")
                    with self.torch.no_grad():
                        for (name, parameter), gradient in zip(self.entries, pieces):
                            require(gradient.dtype == self.torch.float32 and gradient.shape == parameter.shape
                                    and str(gradient.device) == "cuda:0" and bool(self.torch.isfinite(gradient).all()),
                                    "finite complete FP32 independent gradient")
                            gradients[name].add_(gradient.detach())
                self._invoke("reference_gradient_accumulate", accumulate)
                del logits, loss, pieces, ids, labels, mask
        finally:
            self._invoke("qualification_reference_state_preserved", lambda: require(mechanics._state_matches(
                self.torch, self.model, self.optimizer, before, True), "reference changed parameters/moments/gradients/RNG"))
            self.backend._verify_base(content=True)
        return Reference(tuple(n for n, _ in self.entries), gradients, tuple(checks), positions, preserved["sha256"])

    def compare_reference(self, reference):
        require(type(reference) is Reference and reference.parameter_order == tuple(n for n, _ in self.entries)
                and set(reference.gradients) == set(reference.parameter_order), "exact opaque reference parameter order")
        require(self.backend._phase == "clip" and self.backend._sequences == 20,
                "compare independent reference only against complete native preclip gradients")
        def compare():
            errors = []
            for name, parameter in self.entries:
                actual, expected = parameter.grad, reference.gradients[name]
                require(self.torch.is_tensor(actual) and actual.dtype == self.torch.float32 and
                        actual.shape == parameter.shape == expected.shape and str(actual.device) == str(expected.device) == "cuda:0",
                        "native/reference complete FP32 gradient inventory")
                finite = bool(self.torch.isfinite(actual).all()) and bool(self.torch.isfinite(expected).all())
                passed = finite and bool(self.torch.allclose(actual, expected, atol=mechanics.GRAD_ATOL, rtol=mechanics.GRAD_RTOL))
                error = float((actual.detach() - expected).abs().max()) if finite else "NONFINITE"
                errors.append({"name": name, "shape": list(parameter.shape), "finite": finite,
                               "passed": passed, "maximum_absolute_error": error})
            return {"passed": all(item["passed"] for item in errors), "count": len(errors), "errors": errors,
                    "atol": mechanics.GRAD_ATOL, "rtol": mechanics.GRAD_RTOL,
                    "reference_rows": len(reference.row_checks), "reference_input_positions": reference.input_positions,
                    "reference_state_preserved_sha256": reference.preserved_state_sha256,
                    "scope": SCOPE, "native_launch_authorized": False, "study_warmstart_authorized": False}
        return self._invoke("qualification_gradient_compare", compare)

    def close(self):
        self.backend.close()
