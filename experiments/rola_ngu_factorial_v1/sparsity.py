"""Separate post-training delta transform. No in-place edits or dtype conversion."""
from __future__ import annotations
import hashlib
import math
from collections.abc import Mapping
import torch
from torch import Tensor


def prune_delta(base: Mapping[str, Tensor], trained: Mapping[str, Tensor],
                names: tuple[str, ...], keep: float = .25) -> tuple[dict[str, Tensor], dict]:
    if not math.isfinite(keep) or not 0 <= keep <= 1:
        raise ValueError('keep must be in [0,1]')
    if not names or len(set(names)) != len(names) or set(base) - set(trained):
        raise ValueError('explicit unique common-backbone allowlist required')
    if any(n not in base or n not in trained for n in names):
        raise ValueError('new RoLA parameters cannot be included in backbone pruning')
    for n, t in trained.items():
        if t.dtype != torch.float32 or t.device.type != 'cpu' or t.layout != torch.strided:
            raise ValueError('FP32 CPU dense tensors required; no silent conversion')
        if not bool(torch.isfinite(t).all()):
            raise ValueError('nonfinite trained tensor')
        if n in base:
            b = base[n]
            if b.dtype != torch.float32 or b.device.type != 'cpu' or b.shape != t.shape:
                raise ValueError('source dtype/device/shape mismatch')
            if not bool(torch.isfinite(b).all()):
                raise ValueError('nonfinite source tensor')
            if n not in names and not torch.equal(b, t):
                raise ValueError('unregistered backbone update')
    out = {n: t.detach().clone() for n, t in trained.items()}
    details = {}
    for n in sorted(names):
        b, t = base[n].detach().contiguous(), trained[n].detach().contiguous()
        delta = (t - b).flatten()
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError('nonfinite FP32 delta')
        count = math.floor(delta.numel() * keep)
        order = torch.argsort(delta.abs(), descending=True, stable=True)
        mask = torch.zeros(delta.numel(), dtype=torch.bool)
        mask[order[:count]] = True
        masked = torch.where(mask, delta, torch.zeros_like(delta))
        out[n] = (b.flatten() + masked).reshape(t.shape).contiguous()
        if not bool(torch.isfinite(out[n]).all()):
            raise FloatingPointError('nonfinite reconstructed tensor')
        details[n] = {'elements': delta.numel(), 'selected': count,
                      'nonzero_selected_delta': int((masked != 0).sum()),
                      'mask_sha256': hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                      'delta_l2': float(delta.double().norm()),
                      'selected_delta_l2': float(masked.double().norm())}
    return out, {'keep': keep, 'rounding': 'floor', 'tie_break': 'ascending_flat_index',
                 'dtype': 'float32', 'new_parameters': sorted(set(trained) - set(base)),
                 'new_parameters_pruned': False, 'tensors': details,
                 'model_calls': 0, 'training_updates': 0, 'efficiency_claim': False}
