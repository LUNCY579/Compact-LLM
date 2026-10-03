"""Scoped L12-L13 bridge for the tensor-returning, cache-free 28-layer Supa host.

No checkpoint loading here. Qualification/provenance is the runner's job. The
bridge is also tested against a CPU shape fixture, never described as Rineya.
"""
from __future__ import annotations

import inspect
import math
from typing import Any
import torch
from .core import InjectionInterface, finite_tensor
from .diagnostics import trajectory_metrics


class HostError(RuntimeError):
    pass


def _arguments(forward, args, kwargs):
    bound = inspect.signature(forward).bind_partial(*args, **kwargs).arguments
    values = dict(bound)
    for value in bound.values():
        if isinstance(value, dict):
            values.update(value)
    for name in ("past_key_values", "past_key_value", "cache"):
        if values.get(name) is not None:
            raise HostError("cache objects forbidden even with use_cache=False")
    if values.get("use_cache", False):
        raise HostError("cache mode forbidden")
    return values


def _replace_hidden(args, kwargs, hidden):
    return ((hidden, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": hidden})


class FixedPointHost:
    """One ordinary application, at most ONE extra; no deeper native switch.

    The original leading/last call signatures are retained separately, preserving
    each layer's masks and positions. Methods and ownership markers are restored
    even when installation or a forward fails. Not thread-safe or re-entrant.
    """
    def __init__(self, model: Any, *, applications: int = 2, alpha: float = 1.0,
                 interface: InjectionInterface | None = None, training_interface: bool = False,
                 max_tokens: int = 4096):
        if type(applications) is not int or applications not in (1, 2):
            raise ValueError("only one or two TOTAL native applications are qualified for preparation")
        if not math.isfinite(alpha) or not 0 <= alpha <= 2:
            raise ValueError("invalid update scale")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError("invalid token cap")
        if len(model.model.layers) != 28 or type(model).__name__ != "Qwen3AbstainForCausalLM":
            raise HostError("not the audited 28-layer Supa class")
        if any(model.config.layer_types[i] != "full_attention" for i in (12, 13)):
            raise HostError("full_attention required")
        if training_interface and (interface is None or applications != 2):
            raise HostError("interface training requires one extra application")
        self.model, self.applications, self.alpha = model, applications, alpha
        self.interface, self.training_interface, self.max_tokens = interface, training_interface, max_tokens
        self.records, self.calls, self.capture = [], 0, True
        self.min_rms_ratio, self.max_rms_ratio = 1.0, 1.0
        self.active, self._saved, self._handles = False, [], []
        self._entry = self._leading_call = self._mask = None

    @property
    def extra_layer_applications(self):
        return 2 * (self.applications - 1)

    def _guard(self, _model, args, kwargs):
        self._entry = self._leading_call = self._mask = None
        values = _arguments(self.model.forward, args, kwargs)
        if values.get("use_cache") is not False:
            raise HostError("explicit use_cache=False required")
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise HostError("canonical model must already be eval and frozen")
        if getattr(self.model, "is_gradient_checkpointing", False) or hasattr(self.model, "_orig_mod"):
            raise HostError("checkpointing/compile unsupported in scoped native bridge")
        if torch.is_grad_enabled() != self.training_interface:
            raise HostError("use no_grad/inference for diagnosis; enable_grad only for interface training")
        ids = values.get("input_ids")
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= self.max_tokens:
            raise HostError("bounded batch-one token IDs required")
        for p in self.model.parameters():
            if p.device.type != "cpu" or p.dtype != torch.float32:
                raise HostError("this native lane remains CPU FP32")
        mask = values.get("attention_mask")
        if mask is not None:
            if mask.shape != ids.shape or not torch.all((mask == 0) | (mask == 1)):
                raise HostError("binary input attention mask required")
            self._mask = mask.bool()

    def __enter__(self):
        if self.active:
            raise HostError("wrapper already active")
        layers = [self.model.model.layers[i] for i in (12, 13)]
        for block in layers:
            if any(key.endswith("owner") for key in block.__dict__ if key.startswith("_")):
                raise HostError("overlapping wrappers forbidden")
        self.active = True
        try:
            for block in layers:
                self._saved.append((block, block.forward, "forward" in block.__dict__, block.__dict__.get("forward")))
                block.__dict__["_fp_owner"] = self
            self._handles.append(self.model.register_forward_pre_hook(self._guard, with_kwargs=True, prepend=True))
            layers[0].forward, layers[1].forward = self._first, self._last
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _first(self, *args, **kwargs):
        forward = self._saved[0][1]
        values = _arguments(forward, args, kwargs)
        self._entry = values.get("hidden_states", args[0] if args else None)
        finite_tensor(self._entry, "block anchor")
        self._leading_call = (args, kwargs.copy())
        out = forward(*args, **kwargs)
        finite_tensor(out, "leading output")
        self.calls += 1
        return out

    def _last(self, *args, **kwargs):
        forward = self._saved[1][1]
        _arguments(forward, args, kwargs)
        if self._entry is None or self._leading_call is None:
            raise HostError("missing first-layer call context")
        normal = forward(*args, **kwargs)
        self.calls += 1
        finite_tensor(normal, "ordinary block output")
        if self.applications == 1:
            self._entry = self._leading_call = None
            return normal
        anchor = self._entry
        injected = normal if self.interface is None else self.interface(normal, anchor)
        a, k = _replace_hidden(*self._leading_call, injected)
        middle = self._saved[0][1](*a, **k)
        finite_tensor(middle, "repeated leading output")
        a, k = _replace_hidden(args, kwargs, middle)
        proposal = forward(*a, **k)
        finite_tensor(proposal, "proposal")
        self.calls += 2
        # Exact baseline at alpha=0, without changing the measured core-call budget.
        after = normal if self.alpha == 0 else normal + self.alpha * (proposal - normal)
        rec = trajectory_metrics(normal, injected, proposal, after, anchor, normal, self._mask)
        rec.update(application=2, alpha=self.alpha,
                   injection="none" if self.interface is None else self.interface.mode)
        ratio = rec["rms_ratio_to_normal"]
        self.min_rms_ratio, self.max_rms_ratio = min(self.min_rms_ratio, ratio), max(self.max_rms_ratio, ratio)
        if not 0.05 <= ratio <= 10.0:
            raise HostError("inherited emergency hidden-RMS bound")
        if self.capture:
            self.records.append(rec)
        self._entry = self._leading_call = None
        return after

    def __exit__(self, *_exc):
        for handle in self._handles:
            handle.remove()
        for block, _forward, had_attribute, attribute in reversed(self._saved):
            if had_attribute:
                block.forward = attribute
            elif "forward" in block.__dict__:
                del block.forward
            block.__dict__.pop("_fp_owner", None)
        self._saved.clear()
        self._handles.clear()
        self._entry = self._leading_call = self._mask = None
        self.active = False
