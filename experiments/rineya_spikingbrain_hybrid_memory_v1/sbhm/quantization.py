"""Deferred-stage numerical diagnostic only: dense fake activation quantization.

No event-driven execution, weight compression, energy estimate or speed claim.
"""
import torch
import math


def adaptive_fake_quantize(x, k=3.0):
    if not x.is_floating_point() or x.numel() == 0 or not torch.isfinite(x).all() or not math.isfinite(k) or k <= 0:
        raise ValueError('finite nonempty floating activations and k > 0 required')
    threshold = (x.float().abs().mean(-1, keepdim=True)/k).clamp(1e-5, 1e4)
    counts = (x.float()/threshold).round()
    reconstructed = (counts*threshold).to(x.dtype)
    return reconstructed, {'activation_zero_fraction': float((counts == 0).float().mean()),
                           'mean_squared_error': float((x.float()-reconstructed.float()).square().mean()),
                           'event_driven': False, 'measured_energy': None}
