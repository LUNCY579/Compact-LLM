"""Diagnostic-only state gradients on a fixed-alpha frozen L12-L13 operator.

No loader, assets, datasets, training, generation pool or native entrypoint.
Inference wrappers remain unchanged. This apparatus needs native qualification.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as functional


def finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("nonfinite diagnostic state/logits/gradient")


def moment_energies(first, second, beta=.8):
    """Exact two-observation formula, per sequence, with FP32 arithmetic."""
    if first.shape != second.shape or first.ndim < 2 or not 0 <= beta < 1:
        raise ValueError("two-observation shape/beta contract")
    a, b = first.detach().float(), second.detach().float()
    finite(a, b)
    mean = (beta * a + b) / (1 + beta)
    progress = mean.square().flatten(1).mean(1)
    fluctuation = beta / (1 + beta)**2 * (b - a).square().flatten(1).mean(1)
    balance = (progress - 1.5 * fluctuation) / (progress + 1.5 * fluctuation + 1e-12)
    finite(progress, fluctuation, balance)
    return progress, fluctuation, balance


class DiagnosticBlock:
    """One frozen full-prefix forward; only its post-block state is a grad leaf."""
    def __init__(self, model, *, alpha=1., extra_loops=1, device="cpu", guard=None):
        if type(extra_loops) is not int or extra_loops not in (0, 1):
            raise ValueError("diagnostic permits zero/one extra L12-L13 application")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not .8 <= alpha <= 1.2:
            raise ValueError("fixed diagnostic alpha must be finite within[.8,1.2]")
        if len(model.model.layers) <= 13 or any(model.config.layer_types[i] != "full_attention" for i in (12, 13)):
            raise ValueError("exact full-attention L12-L13 host required")
        self.model, self.alpha, self.extra_loops = model, float(alpha), extra_loops
        self.device, self.guard = torch.device(device), guard
        if self.device.type not in ("cpu", "cuda") or self.device.type == "cuda" and self.device.index != 0:
            raise ValueError("explicit CPU fixture or registered cuda:0 device required")
        if self.device.type == "cuda" and not callable(guard):
            raise RuntimeError("native diagnostic requires an explicit resource guard")
        self.handles, self.active = [], False
        self.leaf = self.d1 = self.d2 = self.proposal = self.entry = None
        self.anchors, self.calls, self.forwards = [0, 0, 0], 0, 0
        self.conditioning = None

    def _tensor(self, value):
        if not isinstance(value, torch.Tensor) or value.device != self.device or value.dtype != torch.float32:
            raise RuntimeError("diagnostic tensor must be FP32 on the explicit device")
        if self.entry is not None and value.shape != self.entry.shape:
            raise RuntimeError("diagnostic block proposal shape changed")
        finite(value)

    def _entry(self, _module, args, kwargs):
        if self.guard:
            self.guard()
        if self.model.training or not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise RuntimeError("frozen eval diagnostic requires grad mode, not inference mode")
        if getattr(self.model, "is_gradient_checkpointing", False) or hasattr(self.model, "_orig_mod"):
            raise RuntimeError("compiled/checkpointed diagnostic host unqualified")
        if any(p.requires_grad or p.grad is not None or p.dtype != torch.float32 or p.device != self.device for p in self.model.parameters()):
            raise RuntimeError("diagnostic parameters must remain frozen FP32, gradNone")
        if any(b.device != self.device or b.is_floating_point() and b.dtype != torch.float32 for b in self.model.buffers()):
            raise RuntimeError("diagnostic buffers require explicit device and FP32 floating dtype")
        if kwargs.get("use_cache") is not False or any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value", "cache")):
            raise RuntimeError("diagnostic explicit cache-free invocation required")
        ids = args[0] if args else kwargs.get("input_ids")
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 256 or ids.dtype != torch.long or ids.device != self.device:
            raise RuntimeError("diagnostic batch1 long prefix<=256 on explicit device")
        mask = kwargs.get("attention_mask")
        if mask is None or mask.shape != ids.shape or mask.device != self.device or not bool((mask == 1).all()):
            raise RuntimeError("diagnostic unpadded all-ones input mask required")
        self.anchors[:] = [0, 0, 0]
        self.forwards += 1
        self.entry = self.conditioning = None

    def _capture(self, _module, args, kwargs):
        self.entry = (args[0] if args else kwargs["hidden_states"]).detach()
        if self.entry.ndim != 3 or self.entry.shape[0] != 1:
            raise RuntimeError("diagnostic hidden state must be batch1 token/channel tensor")
        self._tensor(self.entry)
        pos = kwargs.get("position_ids")
        if pos is None or pos.device != self.device or pos.dtype != torch.long or tuple(pos.shape) != (1, self.entry.shape[1]) or not torch.equal(pos, torch.arange(self.entry.shape[1], device=self.device).unsqueeze(0)):
            raise RuntimeError("diagnostic positions must be exact full-prefix arange")
        mask = kwargs.get("attention_mask")
        if mask is None or mask.device != self.device or mask.dtype != torch.float32 or mask.shape[-2:] != (self.entry.shape[1], self.entry.shape[1]):
            raise RuntimeError("diagnostic causal-mask device/shape")
        self.conditioning = {key: kwargs.get(key) for key in ("position_ids", "position_embeddings", "attention_mask", "cache_position")}

    def _forward(self, *args, **kwargs):
        if self.entry is None or self.conditioning is None:
            raise RuntimeError("diagnostic whole-block entry missing")
        if kwargs.get("use_cache", False) or any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value", "cache")):
            raise RuntimeError("diagnostic decoder cache forbidden")
        if any(kwargs.get(key) is not value for key, value in self.conditioning.items()):
            raise RuntimeError("diagnostic block conditioning changed")
        with torch.no_grad():
            normal = self.originals[1](*args, **kwargs)
            self.calls += 1
            self._tensor(normal)
            self.d1 = (normal - self.entry).detach()
            proposal = normal
            if self.extra_loops:
                for original in self.originals:
                    proposal = original(proposal, *args[1:], **kwargs) if args else original(**{**kwargs, "hidden_states": proposal})
                    self.calls += 1
                    self._tensor(proposal)
                self.d2 = (proposal - normal).detach()
                controlled = proposal if self.alpha == 1 else normal + self.alpha * self.d2
            else:
                self.d2 = None
                controlled = normal
            self._tensor(controlled)
            self.proposal = proposal
        # Value-exact unit proposal, then a distinct non-inference state leaf.
        self.leaf = controlled.detach().clone().requires_grad_(True)
        return self.leaf

    def _finish(self, _module, _args, _kwargs, output):
        if self.anchors != [1, 1, 1]:
            raise RuntimeError("diagnostic L0-L2 must remain one-shot")
        return output

    def __enter__(self):
        if self.active or any(any(name in layer.__dict__ for name in ("_taps_owner", "_armr_owner", "_sensitivity_owner")) for layer in self.model.model.layers[12:14]):
            raise RuntimeError("overlapping diagnostic/inference wrappers forbidden")
        self.active = True
        first, last = self.model.model.layers[12:14]
        self.originals = [first.forward, last.forward]
        self.had_forward, self.prior_forward = "forward" in last.__dict__, last.__dict__.get("forward")
        try:
            first.__dict__["_sensitivity_owner"] = last.__dict__["_sensitivity_owner"] = self
            self.handles.append(self.model.register_forward_pre_hook(self._entry, with_kwargs=True, prepend=True))
            self.handles.append(first.register_forward_pre_hook(self._capture, with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self._finish, with_kwargs=True))
            for index in range(3):
                def visit(_module, _args, i=index):
                    self.anchors[i] += 1
                self.handles.append(self.model.model.layers[index].register_forward_pre_hook(visit))
            last.forward = self._forward
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        last = self.model.model.layers[13]
        if self.had_forward:
            last.forward = self.prior_forward
        else:
            last.__dict__.pop("forward", None)
        for layer in self.model.model.layers[12:14]:
            if layer.__dict__.get("_sensitivity_owner") is self:
                del layer.__dict__["_sensitivity_owner"]
        self.leaf = self.d1 = self.d2 = self.proposal = self.entry = self.conditioning = None
        self.active = False


def fixed_rollout(model, input_ids, target_token, *, alpha=1., device="cpu", extra_loops=1, guard=None, synchronize=None, compute_gradient=True):
    """Whole model recomputed; target enters only the subsequent CE calculation."""
    if torch.device(device).type == "cuda" and not callable(synchronize):
        raise RuntimeError("native diagnostic requires explicit CUDA synchronization")
    with torch.enable_grad(), DiagnosticBlock(model, alpha=alpha, device=device, extra_loops=extra_loops, guard=guard) as block:
        if synchronize:
            synchronize()
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False, logits_to_keep=1)
        if getattr(output, "past_key_values", None) is not None:
            raise RuntimeError("diagnostic model returned a forbidden cache")
        logits = output.logits[0, -1].float()
        finite(logits)
        if type(target_token) is not int or not 0 <= target_token < logits.numel():
            raise ValueError("offline target token outside actual custom readout vocabulary")
        target = torch.tensor([target_token], dtype=torch.long, device=logits.device)
        loss = functional.cross_entropy(logits.unsqueeze(0), target)
        finite(loss)
        gradient = derivative = None
        if compute_gradient:
            gradient, = torch.autograd.grad(loss, block.leaf)
            finite(gradient)
            derivative = float((gradient.float() * block.d2.float()).sum()) if block.d2 is not None else 0.
        if block.d2 is not None:
            progress, fluctuation, balance = (float(x[0]) for x in moment_energies(block.d1, block.d2))
        else:
            progress = fluctuation = balance = None  # no invented second observation
        if synchronize:
            synchronize()
        if guard:
            guard()  # resource/deadline gate after suffix/gradient work
        result = {"alpha": alpha, "loss": float(loss.detach()), "sensitivity": derivative,
                  "progress": progress, "fluctuation": fluctuation, "balance": balance,
                  "state_gradient_norm": float(gradient.norm()) if gradient is not None else None,
                  "raw_update_norm": float(block.d2.norm()) if block.d2 is not None else None,
                  "state_gradient_evaluations": int(compute_gradient),
                  "anchor_counts": list(block.anchors), "full_model_calls": 1,
                  "extra_block_applications": extra_loops, "block_local_calls": block.calls,
                  "evidence_kind": "OFFLINE_CANONICAL_FIRST_TOKEN_SURROGATE"}
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("diagnostic left trainable parameters/parameter gradients")
    return result


def diagnose(model, input_ids, target_token, *, alpha=1., step=.01, device="cpu", guard=None, synchronize=None):
    if target_token is None:
        return {"status": "NOT_APPLICABLE_MISSING_CANONICAL_TARGET", "full_model_calls": 0}
    if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0 or not .8 <= alpha - step < alpha + step <= 1.2:
        raise ValueError("central difference must remain within fixed safe scale bounds")
    kwargs = {"device": device, "guard": guard, "synchronize": synchronize}
    nominal = fixed_rollout(model, input_ids, target_token, alpha=alpha, **kwargs)
    plus = fixed_rollout(model, input_ids, target_token, alpha=alpha + step, compute_gradient=False, **kwargs)
    minus = fixed_rollout(model, input_ids, target_token, alpha=alpha - step, compute_gradient=False, **kwargs)
    difference = (plus["loss"] - minus["loss"]) / (2 * step)
    return {"status": "DIAGNOSTIC_APPARATUS_NOT_NATIVE_QUALIFIED", "nominal": nominal,
            "plus": plus, "minus": minus, "finite_difference": difference,
            "absolute_error": abs(difference - nominal["sensitivity"]),
            "full_model_calls": 3, "state_gradient_evaluations": 1,
            "trajectory_reuse": False, "gold_enters_forward_input": False,
            "generation_pool_access": False}
