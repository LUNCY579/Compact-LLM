"""Scoped cache-free ARMR adaptation. No canonical module or weight replacement."""
from __future__ import annotations
import time
import torch
from .scheduler import TAPSScheduler, TAPSConfig, apply_update, scalar_events


class BlockRecurrence:
    """Preserve the ordinary block pass; scale only the extra block application.

    Native boundary: one or two contiguous middle layers, at most ONE extra pass.
    Supports the audited tensor-returning Qwen3 decoder contract only. State resets
    at each complete model forward, including each autoregressive token prefix.
    """
    def __init__(self, model, layers=(12, 13), extra_loops=1, *, config=None,
                 enabled=True, sink=None, sample_id="sample", guard=None):
        if type(extra_loops) is not int or extra_loops not in (0, 1):
            raise ValueError("native initial qualification permits zero or one extra pass only")
        if type(enabled) is not bool:
            raise ValueError("enabled must be bool")
        self.model, self.layers = model, tuple(layers)
        n = len(model.model.layers)
        if len(self.layers) not in (1, 2) or any(type(i) is not int or i < 3 or i >= min(25, n) for i in self.layers):
            raise ValueError("one or two anchored middle layers required")
        if len(self.layers) == 2 and self.layers[1] != self.layers[0]+1:
            raise ValueError("middle block must be contiguous")
        if any(model.config.layer_types[i] != "full_attention" for i in self.layers):
            raise ValueError("only audited full-attention layers supported")
        self.extra_loops, self.enabled = extra_loops, enabled
        self.config, self.sink = config or TAPSConfig(), sink
        self.sample_id, self.guard = sample_id, guard
        self.capture, self.records, self.calls = True, [], 0
        self.min_rms_ratio = self.max_rms_ratio = 1.
        self.forward_count = self.token_positions = 0
        self.controller_seconds = self.core_seconds = self.logging_seconds = 0.
        self.active = False
        self.handles, self.saved = [], []

    @property
    def extra_layer_applications(self):
        return len(self.layers)*self.extra_loops if self.enabled else 0

    def _entry(self, module, args, kwargs):
        if self.guard:
            self.guard()
        if len(args) > 1 or kwargs.get("use_cache") is not False:
            raise RuntimeError("explicit cache-free keyword invocation required")
        if any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value", "cache")):
            raise RuntimeError("cache objects forbidden even with use_cache=False")
        if self.model.training or torch.is_grad_enabled() or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("frozen eval model and inference/no_grad required")
        if getattr(self.model, "is_gradient_checkpointing", False) or hasattr(self.model, "_orig_mod"):
            raise RuntimeError("compiled/checkpointed host is unqualified")
        ids = args[0] if args else kwargs.get("input_ids")
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] > 4096:
            raise RuntimeError("native lane requires batch one and at most 4096 prefix tokens")
        mask = kwargs.get("attention_mask")
        if mask is not None and (mask.shape != ids.shape or not bool((mask == 1).all())):
            raise RuntimeError("padded native prefixes are not qualified")
        self.forward_count += 1
        self.token_positions += ids.shape[1]
        self.entry_hidden = None
        self.scheduler = TAPSScheduler(self.config)

    def __enter__(self):
        if self.active:
            raise RuntimeError("wrapper already active")
        blocks = [self.model.model.layers[i] for i in self.layers]
        if any("_taps_owner" in b.__dict__ or "_armr_owner" in b.__dict__ for b in blocks):
            raise RuntimeError("overlapping recurrence wrappers forbidden")
        self.active = True
        try:
            self.handles.append(self.model.register_forward_pre_hook(self._entry, with_kwargs=True, prepend=True))
            for b in blocks:
                b.__dict__["_taps_owner"] = self
            last = blocks[-1]
            self.originals = [b.forward for b in blocks]
            self.saved.append((last, "forward" in last.__dict__, last.__dict__.get("forward")))
            def capture_entry(module, args, kwargs):
                if self.entry_hidden is None:
                    self.entry_hidden = (args[0] if args else kwargs["hidden_states"]).detach()
            self.handles.append(blocks[0].register_forward_pre_hook(capture_entry, with_kwargs=True))
            last.forward = self._forward
            return self
        except BaseException:
            self.__exit__()
            raise

    def _record(self, old, new, event, normal, core_seconds, controller_seconds):
        e = scalar_events(event)[0]
        numerator = new.float().square().mean().sqrt()
        denominator = normal.float().square().mean().sqrt()
        ratio = 1. if float(numerator) == 0. and float(denominator) == 0. else float(numerator/denominator.clamp_min(1e-12))
        self.min_rms_ratio, self.max_rms_ratio = min(self.min_rms_ratio, ratio), max(self.max_rms_ratio, ratio)
        e.update(sample_id=self.sample_id, forward_index=self.forward_count,
                 state_norm=float(old.float().norm()), movement=float((new.float()-old.float()).norm()),
                 rms_ratio_to_normal=ratio, stopping_iteration=self.extra_loops if self.enabled else 0,
                 loop_core_seconds=core_seconds, loop_controller_seconds=controller_seconds,
                 logit_movement=None, loss_proxy=None, evidence_kind="native_trajectory",
                 safety_violation=not .25 <= ratio <= 4.,
                 note="first-pass core timing covers last layer only; logits/NLL use full-depth evaluator")
        if self.capture:
            self.records.append(e)
        if self.sink:
            t = time.perf_counter(); self.sink(e); self.logging_seconds += time.perf_counter()-t
        if e["safety_violation"]:
            raise FloatingPointError("unchanged ARMR smoke norm bounds violated")

    def _forward(self, *args, **kwargs):
        if any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value", "cache")) or kwargs.get("use_cache", False):
            raise RuntimeError("decoder cache forbidden")
        old = self.entry_hidden
        if old is None:
            raise RuntimeError("missing first block input")
        t = time.perf_counter()
        normal = self.originals[-1](*args, **kwargs)
        self.calls += 1
        elapsed = time.perf_counter()-t
        if not isinstance(normal, torch.Tensor) or normal.shape != old.shape:
            raise RuntimeError("tensor-returning, shape-preserving Qwen3 layer required")
        t = time.perf_counter()
        _, event = self.scheduler.observe(normal.float()-old.float(), (self.sample_id,), force_unit=True)
        control = time.perf_counter()-t
        self.controller_seconds += control; self.core_seconds += elapsed
        self._record(old, normal, event, normal, elapsed, control)
        out = normal
        if self.enabled:
            for _ in range(self.extra_loops):
                t = time.perf_counter(); proposal = out
                for forward in self.originals:
                    proposal = forward(proposal, *args[1:], **kwargs) if args else forward(**{**kwargs, "hidden_states": proposal})
                    self.calls += 1
                elapsed = time.perf_counter()-t
                t = time.perf_counter()
                alpha, event = self.scheduler.observe(proposal.float()-out.float(), (self.sample_id,))
                new = apply_update(out, proposal, alpha)
                control = time.perf_counter()-t
                self.core_seconds += elapsed; self.controller_seconds += control
                self._record(out, new, event, normal, elapsed, control)
                out = new
        self.entry_hidden = None  # release full-prefix reference promptly
        return out

    def __exit__(self, *_):
        for h in self.handles:
            h.remove()
        for block, had, prior in self.saved:
            if had: block.forward = prior
            elif "forward" in block.__dict__: del block.forward
        for i in self.layers:
            b = self.model.model.layers[i]
            if b.__dict__.get("_taps_owner") is self: del b.__dict__["_taps_owner"]
        self.handles, self.saved = [], []
        self.active = False
        self.entry_hidden = None
        self.scheduler = None
