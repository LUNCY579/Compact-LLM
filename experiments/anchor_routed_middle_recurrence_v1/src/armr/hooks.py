"""Scoped bound-forward wrappers; no canonical layer replacement or cache writes."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

from .metrics import hidden_metrics


class SafetyError(RuntimeError):
    pass


class Reapplication:
    def __init__(self, model: Any, layer: int | None = None, applications: int = 1,
                 *, skip: bool = False, enabled: bool = True,
                 emergency_bounds: tuple[float, float] = (.05, 10.0)):
        if type(applications) is not int or applications not in (1, 2, 3):
            raise ValueError("applications must be 1, 2, or 3")
        self.model, self.layer, self.applications = model, layer, applications
        self.skip, self.enabled, self.bounds = skip, enabled, emergency_bounds
        if layer is not None and (type(layer) is not int or not 0 <= layer < len(model.model.layers)):
            raise ValueError("invalid layer")
        if layer is not None and layer < 3 and applications > 1:
            raise ValueError("anchor recurrence forbidden")
        if skip and (layer is None or applications != 1):
            raise ValueError("skip is a separate one-pass diagnostic")
        self.calls = 0
        self.capture = True
        self.records: list[dict[str, Any]] = []
        self.min_rms_ratio = 1.0
        self.max_rms_ratio = 1.0
        self.active = False

    def _guard(self, _module: Any, args: tuple, kwargs: dict) -> None:
        if not self.enabled:
            return
        if len(args) > 1:
            raise SafetyError("only input_ids may be positional")
        if kwargs.get("use_cache") is not False:
            raise SafetyError("explicit use_cache=False required")
        if any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value", "cache")):
            raise SafetyError("cache objects forbidden, including use_cache=False")
        if self.model.training or torch.is_grad_enabled():
            raise SafetyError("T0 requires eval and inference/no_grad mode")
        if getattr(self.model, "is_gradient_checkpointing", False) or hasattr(self.model, "_orig_mod"):
            raise SafetyError("checkpointing/compile unsupported")

    def __enter__(self) -> Reapplication:
        if self.active:
            raise SafetyError("wrapper already active")
        self.active = True
        self.handle = self.model.register_forward_pre_hook(self._guard, with_kwargs=True, prepend=True)
        if self.layer is None or not self.enabled:
            return self
        block = self.model.model.layers[self.layer]
        if "_armr_owner" in block.__dict__:
            self.handle.remove()
            self.active = False
            raise SafetyError("overlapping wrappers forbidden")
        self.block, self.original = block, block.forward
        self.had_forward = "forward" in block.__dict__
        self.prior_attribute = block.__dict__.get("forward")
        block.__dict__["_armr_owner"] = self
        block.forward = self._forward
        return self

    def _forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value")):
            raise SafetyError("decoder received a cache")
        if kwargs.get("use_cache", False):
            raise SafetyError("decoder cache mode forbidden")
        hidden = args[0] if args else kwargs["hidden_states"]
        out = self.original(*args, **kwargs)
        self.calls += 1
        if not isinstance(out, torch.Tensor):
            raise SafetyError("only audited tensor-returning Qwen3 decoder supported")
        normal, prev, prev_delta = out, hidden, None
        rec, prev_delta = hidden_metrics(out, normal, prev, None, 1)
        if self.capture:
            self.records.append(rec)
        if self.skip:
            return hidden
        for application in range(2, self.applications + 1):
            prev = out
            if args:
                out = self.original(out, *args[1:], **kwargs)
            else:
                out = self.original(**{**kwargs, "hidden_states": out})
            self.calls += 1
            rec, prev_delta = hidden_metrics(out, normal, prev, prev_delta, application)
            self.min_rms_ratio = min(self.min_rms_ratio, rec["rms_ratio_to_normal"])
            self.max_rms_ratio = max(self.max_rms_ratio, rec["rms_ratio_to_normal"])
            if self.capture:
                self.records.append(rec)
            if not self.bounds[0] <= rec["rms_ratio_to_normal"] <= self.bounds[1]:
                raise SafetyError("hidden norm emergency bound")
        return out

    def __exit__(self, *_exc: Any) -> None:
        self.handle.remove()
        if hasattr(self, "block"):
            if self.had_forward:
                self.block.forward = self.prior_attribute
            else:
                del self.block.forward
            del self.block.__dict__["_armr_owner"]
        self.active = False


@contextmanager
def frozen_inference(model: Any) -> Iterator[None]:
    """Do not change training flags or requires_grad inside this context."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise SafetyError("model must already be frozen and in eval mode")
    with torch.inference_mode():
        yield
