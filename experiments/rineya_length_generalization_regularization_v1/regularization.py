# SPDX-License-Identifier: GPL-3.0-only
"""Training-only RMSNorm interventions with reversible, explicit placement.

VPAD equations/numerical policy adapted from deep-spin/len-gen-regularization,
commit d9749d54347f585abfd471d9a71015ac25562780, dropout/vpad_core.py.
See NOTICE.md. No guarantee of full-distribution or covariance preservation.
"""
from contextlib import contextmanager
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


def vpad(x, probability, eps, *, detach_beta=True, detach_alpha=False):
    """Return a perturbed RMS-normalized tensor, before learned norm weights.

    Moments are computed over the feature dimension in FP32. Preserves the
    upstream clamp/epsilon and stop-gradient choices. Zero-p bypass is exact
    at the enclosing module, which delegates to its original forward.
    """
    if not 0 <= probability < 1 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid VPAD probability or epsilon")
    if not x.is_floating_point() or x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("VPAD requires nonempty floating-point features")
    work = x.float()
    second = work.square().mean(-1, keepdim=True)
    inverse_rms = torch.rsqrt(second + eps)
    normalized = work * inverse_rms
    if probability == 0:
        return normalized
    keep = 1.0 - probability
    mean = work.mean(-1, keepdim=True)
    variance = (second - mean.square()).clamp_min(eps)
    scale = (variance / (keep * (second - keep * mean.square()) + eps)).sqrt()
    if detach_alpha:
        scale = scale.detach()
    shift = mean * inverse_rms * (1.0 - keep * scale)
    if detach_beta:
        shift = shift.detach()
    mask = torch.empty_like(work).bernoulli_(keep)
    return torch.addcmul(shift, normalized * mask, scale)


def norm_targets(model, expected_layers):
    """Whitelist only Qwen-style block-entry norms, never q_norm/k_norm/head."""
    if type(expected_layers) is not int or expected_layers <= 0:
        raise ValueError("An explicit positive layer count is required")
    names = [f"model.layers.{i}.{suffix}" for i in range(expected_layers)
             for suffix in ("input_layernorm", "post_attention_layernorm")]
    modules = dict(model.named_modules())
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) != expected_layers:
        raise ValueError("Unexpected decoder layout/layer count; qualify a new binding")
    targets = []
    for name in names:
        mod = modules.get(name)
        if (mod is None or "RMSNorm" not in type(mod).__name__ or
                "Gated" in type(mod).__name__ or not isinstance(getattr(mod, "weight", None), nn.Parameter)
                or mod.weight.ndim != 1 or getattr(mod, "bias", None) is not None):
            raise ValueError(f"Unsupported norm: {name}")
        eps = getattr(mod, "variance_epsilon", getattr(mod, "eps", None))
        if not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"Unsupported norm epsilon: {name}")
        if "forward" in mod.__dict__:
            raise ValueError(f"Norm already patched: {name}")
        targets.append((name, mod, float(eps)))
    if len({id(m) for _, m, _ in targets}) != 2 * expected_layers:
        raise ValueError("Aliased norm modules are not supported")
    return targets


def _forward(original, plan, eps):
    def forward(module, hidden_states):
        if not module.training or plan.probability == 0 or plan.mode == "none":
            return original(hidden_states)
        if plan.mode == "pre_linear":
            return F.dropout(original(hidden_states), plan.probability, training=True)
        if plan.mode == "pre_norm_control":
            return original(F.dropout(hidden_states, plan.probability, training=True))
        values = vpad(hidden_states, plan.probability, eps,
                      detach_beta=plan.detach_beta, detach_alpha=plan.detach_alpha)
        return module.weight * values.to(hidden_states.dtype)
    return forward


@contextmanager
def regularize(model, plan):
    """Patch only explicit norms; keep Parameter objects and state_dict keys.

    One mask is shared by downstream Q/K/V, another by gate/up. Call BEFORE
    torch.compile; compiled/fused custom execution requires new qualification.
    Restoration happens even after a failed forward/backward.
    """
    targets = norm_targets(model, plan.expected_layers)
    patched = []
    try:
        for name, mod, eps in targets:
            mod.forward = MethodType(_forward(mod.forward, plan, eps), mod)
            patched.append(mod)
        yield [name for name, _, _ in targets]
    finally:
        for mod in reversed(patched):
            del mod.forward


@contextmanager
def activation_probe(model, plan, maximum_calls=64):
    """Bounded detached statistics; never stores hidden states or model graphs."""
    if maximum_calls <= 0:
        raise ValueError("maximum_calls must be positive")
    records, handles = [], []
    def hook(name):
        def record(module, args, output):
            if len(records) < maximum_calls:
                value = output.detach().float()
                records.append({"module": name, "training": module.training,
                                "mean": value.mean().item(),
                                "variance": value.var(unbiased=False).item(),
                                "finite": bool(torch.isfinite(value).all())})
        return record
    try:
        for name, mod, _ in norm_targets(model, plan.expected_layers):
            handles.append(mod.register_forward_hook(hook(name)))
        yield records
    finally:
        for handle in handles:
            handle.remove()
