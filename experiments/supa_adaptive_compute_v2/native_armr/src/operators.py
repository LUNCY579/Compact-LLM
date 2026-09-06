"""Reuse audited native recurrence; add a registered generic compute control.

The control remains a wrapper-local module, never part of the canonical model.
Neither backbone weights nor the historical ARMR implementation are modified.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn

HISTORICAL_SOURCE = Path(__file__).resolve().parents[3] / "anchor_routed_middle_recurrence_v1/src"
if str(HISTORICAL_SOURCE) not in sys.path:
    sys.path.insert(0, str(HISTORICAL_SOURCE))

from armr.hooks import Reapplication, SafetyError
from armr.metrics import hidden_metrics
from armr.recurrence import FixedBlock


REGISTERED_ARMS = ("A0", "A1", "A3", "A16", "AM")


def layer_matmul_flops(config: Any) -> int:
    """Historical multiply-plus-add convention; excludes attention and norms."""
    d = config.hidden_size
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    return int(2 * (d*q + 2*d*kv + q*d + 3*d*config.intermediate_size))


def matched_control_width(config: Any) -> int:
    """SwiGLU width matching the dominant linear work of two native layers."""
    work = 2 * layer_matmul_flops(config)
    divisor = 6 * config.hidden_size
    if work % divisor:
        raise ValueError("decoder geometry cannot exactly match the registered SwiGLU work")
    return work // divisor


class FrozenSwiGLU(nn.Module):
    """Independent normal initialization, unit RMS normalization, no training."""

    def __init__(self, hidden_size: int, intermediate_size: int, *, seed: int,
                 std: float, rms_epsilon: float):
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0 or std <= 0 or rms_epsilon <= 0:
            raise ValueError("invalid control dimensions or initialization")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.rms_epsilon = rms_epsilon
        self.gate = nn.Parameter(torch.randn(intermediate_size, hidden_size,
                                             generator=generator, device="cpu",
                                             dtype=torch.float32) * std, requires_grad=False)
        self.up = nn.Parameter(torch.randn(intermediate_size, hidden_size,
                                           generator=generator, device="cpu",
                                           dtype=torch.float32) * std, requires_grad=False)
        self.down = nn.Parameter(torch.randn(hidden_size, intermediate_size,
                                             generator=generator, device="cpu",
                                             dtype=torch.float32) * std, requires_grad=False)
        self.eval()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.training or any(p.requires_grad or p.grad is not None for p in self.parameters()):
            raise SafetyError("compute control must stay frozen with no gradients")
        h = hidden * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + self.rms_epsilon)
        gate = torch.nn.functional.linear(h, self.gate)
        up = torch.nn.functional.linear(h, self.up)
        return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, self.down)


class NativeComputeControl(Reapplication):
    """h + alpha M(h) after an ordinary layer, with A3-matched matrix work.

`calls` counts intercepted native-layer calls, `control_calls` counts added
SwiGLU calls. `extra_layer_applications` is a work-equivalent count for the
shared legacy dense-matmul estimator, not a claim that AM repeats layers.
"""

    def __init__(self, model: Any, *, layer: int = 13, seed: int = 7301,
                 std: float = .02, alpha: float = 1.0,
                 intermediate_size: int | None = None, enabled: bool = True,
                 emergency_bounds: tuple[float, float] = (.25, 4.0)):
        if layer < 3:
            raise ValueError("compute control cannot alter the anchored stem")
        if not torch.isfinite(torch.tensor(alpha)):
            raise ValueError("finite control alpha required")
        super().__init__(model, layer=layer, applications=1, enabled=enabled,
                         emergency_bounds=emergency_bounds)
        width = matched_control_width(model.config)
        if intermediate_size is not None and intermediate_size != width:
            raise ValueError("control width must match two native layers' matrix work")
        self.alpha, self.seed, self.std = alpha, seed, std
        self.transform = FrozenSwiGLU(model.config.hidden_size, width, seed=seed,
                                     std=std, rms_epsilon=model.config.rms_norm_eps)
        exemplar = next(model.parameters())
        # Draw the identical reference tensors on CPU, then transfer them. A
        # CUDA generator would define a different AM control even with seed 7301.
        self.transform.to(device=exemplar.device, dtype=exemplar.dtype)
        self.control_calls = 0
        self.added_parameter_count = sum(p.numel() for p in self.transform.parameters())
        self.added_active_parameters = self.added_parameter_count
        self.extra_matmul_flops_per_token = 2 * self.added_parameter_count

    @property
    def extra_layer_applications(self) -> int:
        return 2 if self.enabled else 0

    def _forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value", "cache")):
            raise SafetyError("decoder received a cache")
        if kwargs.get("use_cache", False):
            raise SafetyError("decoder cache mode forbidden")
        hidden = args[0] if args else kwargs["hidden_states"]
        normal = self.original(*args, **kwargs)
        self.calls += 1
        if not isinstance(normal, torch.Tensor):
            raise SafetyError("only audited tensor-returning Qwen3 decoder supported")
        first, delta = hidden_metrics(normal, normal, hidden, None, 1)
        if self.capture:
            self.records.append(first)
        out = normal + self.alpha * self.transform(normal)
        self.control_calls += 1
        record, _ = hidden_metrics(out, normal, normal, delta, 2)
        self.min_rms_ratio = min(self.min_rms_ratio, record["rms_ratio_to_normal"])
        self.max_rms_ratio = max(self.max_rms_ratio, record["rms_ratio_to_normal"])
        if self.capture:
            self.records.append(record)
        if not self.bounds[0] <= record["rms_ratio_to_normal"] <= self.bounds[1]:
            raise SafetyError("compute-control hidden RMS outside registered guard")
        return out


def operator_factory(model: Any, arm: str, config: dict) -> Reapplication:
    """One context-managed operator API, with no result-dependent selection."""
    if arm not in REGISTERED_ARMS or arm not in config["arms"]:
        raise ValueError(f"unregistered native arm: {arm}")
    bounds = tuple(config["hidden_rms_ratio_bounds"])
    if len(bounds) != 2 or not 0 < bounds[0] < bounds[1]:
        raise ValueError("invalid registered hidden RMS bounds")
    if arm == "A0":
        wrapper = Reapplication(model, emergency_bounds=bounds)
    elif arm == "A1":
        wrapper = Reapplication(model, layer=12, applications=2, emergency_bounds=bounds)
    elif arm == "A3":
        wrapper = FixedBlock(model, layers=[12, 13], applications=2, emergency_bounds=bounds)
    elif arm == "A16":
        wrapper = Reapplication(model, layer=16, applications=2, emergency_bounds=bounds)
    else:
        control = config["control"]
        from .protocol import (CALIBRATED_PROTOCOL_VERSION, LEARNED_PROTOCOL_VERSION,
                               verify_control_calibration, verify_learned_control_calibration)
        calibrated = config.get("protocol_version") == CALIBRATED_PROTOCOL_VERSION
        learned = config.get("protocol_version") == LEARNED_PROTOCOL_VERSION
        if learned:
            from .learned_control import NativeFFNControl
            calibration = verify_learned_control_calibration(config)
            wrapper = NativeFFNControl(model, alpha=calibration["alpha"], emergency_bounds=bounds)
            transform = wrapper.transform
            if (transform.branch_provenance != calibration["branch_provenance"] or
                    transform.source_hashes != calibration["source_hashes"] or
                    transform.copy_hashes != calibration["copy_hashes"] or
                    transform.parameter_inventory != calibration["parameter_inventory"]):
                raise ValueError("pretrained control branch source/copy provenance differs from calibration")
            if (wrapper.added_parameter_count != control["additional_parameters"] or
                    wrapper.added_active_parameters != control["additional_active_parameters"] or
                    wrapper.extra_matmul_flops_per_token != control["additional_matmul_flops_per_token"]):
                raise ValueError("pretrained control actual copied parameters or work differ from registration")
        else:
            calibration = verify_control_calibration(config) if calibrated else None
            alpha = calibration["alpha"] if calibrated else 1.0
            if (control["insertion_layer"], control["seed"], control["std"], control["alpha"]) != (13, 7301, .02, alpha):
                raise ValueError("unregistered compute-control settings")
            wrapper = NativeComputeControl(model, layer=13, seed=7301, std=.02, alpha=alpha,
                                           intermediate_size=control["intermediate_size"],
                                           emergency_bounds=bounds)
        if calibration is not None:
            from shared.integrity import state_hash
            if state_hash(wrapper.transform) != calibration["control_state_sha256"]:
                raise ValueError("compute-control tensors differ from the calibrated frozen transform")
    wrapper.arm = arm
    if arm != "AM":
        wrapper.added_parameter_count = 0
        wrapper.added_active_parameters = 0
        wrapper.extra_matmul_flops_per_token = wrapper.extra_layer_applications * layer_matmul_flops(model.config)
    return wrapper
