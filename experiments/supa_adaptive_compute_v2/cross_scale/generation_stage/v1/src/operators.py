"""Low-level frozen graft mechanics; no stage registration or execution license.

These interfaces do not load checkpoints or select bridges/doses. A future
stage-owned caller must verify complete upstream GO, exported artifacts, its own
committed protocol and every required control before invoking a real model.
"""
from __future__ import annotations

import hashlib
import math

import torch

from shared.integrity import frozen_inventory, state_hash
from shared.legacy import legacy_graft
from armr.hooks import Reapplication

BRIDGE_KEYS = ("up", "up_bias", "down", "down_bias")
GRAFT_ARMS = ("XG", "XS", "XR")
GATE_GRID = (.03, .10, .20, .40)
INSERTION_SITES = (8, 12, 16, 19)
DONOR_SITES = (12, 16, 20, 24)


def validate_bridge_state(state, small_width, donor_width):
    if not isinstance(state, dict) or set(state) != set(BRIDGE_KEYS):
        raise ValueError("the exact four upstream bridge tensors are required")
    shapes = ((small_width, donor_width), (donor_width,),
              (donor_width, small_width), (small_width,))
    for name, shape in zip(BRIDGE_KEYS, shapes):
        value = state[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.device.type != "cpu"
                or value.requires_grad or value.grad is not None
                or not torch.isfinite(value).all()):
            raise ValueError(f"invalid frozen CPU FP32 bridge tensor: {name}")


def buffer_hashes(module):
    """Include nonpersistent rotary buffers omitted by state_dict()."""
    result = {}
    for name, value in module.named_buffers():
        if value.requires_grad or value.grad is not None:
            raise RuntimeError("expert buffers must be frozen and gradient-free")
        digest = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
        flat = value.detach().contiguous().view(-1)
        step = max(1, (8 * 1024 * 1024) // value.element_size())
        for start in range(0, flat.numel(), step):
            block = flat[start:start + step].cpu().numpy()
            digest.update(memoryview(block).cast("B"))
        result[name] = digest.hexdigest()
    return result


class FrozenGraft(legacy_graft().Graft):
    """Historical hook lifecycle with explicit work and expert integrity checks."""

    def __init__(self, model, expert, *, layer, arm, emergency_bounds, enabled):
        super().__init__(model, expert, layer=layer, emergency_bounds=emergency_bounds, enabled=enabled)
        self.arm = arm
        self.donor_parameter_count = sum(p.numel() for p in expert.layer.parameters())
        self.bridge_coefficient_count = sum(value.numel() for value in expert.bridges.buffers())
        self.added_parameter_count = self.donor_parameter_count + self.bridge_coefficient_count
        self.additional_matrix_parameters = (sum(p.numel() for p in expert.layer.parameters() if p.ndim == 2)
                                             + expert.bridges.up.numel() + expert.bridges.down.numel())
        self.integrity_receipt = None
        self._configured_gate = expert.gate
        self._configured_treatment = self._treatment()

    @property
    def extra_matmul_flops_per_token(self):
        return legacy_graft().extra_flops(self.expert) if self.enabled else 0

    @property
    def added_active_parameters(self):
        if not self.enabled or self.expert.gate == 0:
            return 0
        # The affine downward bias is retained for artifact identity but cancels.
        return self.added_parameter_count - self.expert.bridges.down_bias.numel()

    def __enter__(self):
        if self.active:
            raise RuntimeError("wrapper already active; integrity baseline cannot be replaced")
        if self.expert.identity is not False or type(self.enabled) is not bool:
            raise RuntimeError("graft must use the actual donor and an explicit enabled flag")
        if self.expert.gate != self._configured_gate or self._treatment() != self._configured_treatment:
            raise RuntimeError("configured graft treatment changed before entry")
        frozen_inventory(self.model)
        frozen_inventory(self.expert)
        self._before_state = state_hash(self.expert)
        self._before_buffers = buffer_hashes(self.expert)
        self._before_gate = self.expert.gate
        self._before_treatment = self._treatment()
        return super().__enter__()

    def _treatment(self):
        return {"identity": self.expert.identity, "enabled": self.enabled,
                "insertion_layer": self.layer,
                "donor_layer": self.expert.layer.self_attn.layer_idx,
                "bounds": tuple(self.bounds)}

    def __exit__(self, exc_type, exc, traceback):
        # Always restore the canonical hook before reporting an integrity failure.
        super().__exit__(exc_type, exc, traceback)
        try:
            frozen_inventory(self.model)
            inventory = frozen_inventory(self.expert)
            after_state = state_hash(self.expert)
            after_buffers = buffer_hashes(self.expert)
            self.integrity_receipt = {
                **inventory, "state_sha256_before": self._before_state,
                "state_sha256_after": after_state, "buffers_before": self._before_buffers,
                "buffers_after": after_buffers, "gate_before": self._before_gate,
                "gate_after": self.expert.gate,
                "treatment_before": self._before_treatment,
                "treatment_after": self._treatment(),
            }
            if (self._before_state != after_state or self._before_buffers != after_buffers
                    or self._before_gate != self.expert.gate
                    or self._before_treatment != self._treatment()):
                raise RuntimeError("frozen donor, bridge, rotary buffer or treatment changed")
        except BaseException as integrity_error:
            if exc is not None:
                raise BaseExceptionGroup("graft execution and integrity failed", [exc, integrity_error])
            raise
        return False


def make_graft_wrapper(model, donor_layer, donor_config, bridge_state, *,
                       arm, insertion_layer, donor_index, gate, emergency_bounds, enabled=True):
    """Assemble already verified tensors; does not establish scientific admission."""
    if arm not in GRAFT_ARMS:
        raise ValueError("only selected and same-pair shuffled/random graft arms are supported")
    if type(enabled) is not bool:
        raise ValueError("enabled must be explicit boolean")
    if insertion_layer not in INSERTION_SITES or donor_index not in DONOR_SITES:
        raise ValueError("depth is outside the frozen Stage 1 matrix")
    if isinstance(gate, bool) or not isinstance(gate, (int, float)) or not math.isfinite(gate) or gate not in (0., *GATE_GRID):
        raise ValueError("only the prospective four doses or mechanical zero gate are supported")
    if (len(emergency_bounds) != 2 or any(not math.isfinite(v) for v in emergency_bounds)
            or not 0 < emergency_bounds[0] < emergency_bounds[1]):
        raise ValueError("explicit finite positive ordered hidden-RMS bounds required")
    frozen_inventory(model)
    frozen_inventory(donor_layer)
    exemplar = next(model.parameters())
    if exemplar.dtype != torch.float32 or exemplar.device.type not in ("cpu", "cuda"):
        raise ValueError("reference operators require FP32 on one CPU/CUDA device")
    if any(p.device != exemplar.device or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Supa tensors must share one FP32 device")
    if (getattr(donor_layer.self_attn, "layer_idx", None) != donor_index
            or donor_layer.self_attn.config != donor_config
            or any(p.device != exemplar.device or p.dtype != torch.float32 for p in donor_layer.parameters())):
        raise ValueError("donor layer identity, configuration or device mismatch")
    validate_bridge_state(bridge_state, model.config.hidden_size, donor_config.hidden_size)
    legacy = legacy_graft()
    bridges = legacy.Bridges(*(bridge_state[key] for key in BRIDGE_KEYS)).to(exemplar.device)
    expert = legacy.ResidualExpert(donor_layer, donor_config, bridges, float(gate)).to(exemplar.device)
    frozen_inventory(expert)
    return FrozenGraft(model, expert, layer=insertion_layer, arm=arm,
                       emergency_bounds=tuple(emergency_bounds), enabled=enabled)


def make_intact_wrapper(model):
    frozen_inventory(model)
    wrapper = Reapplication(model)
    wrapper.arm = "X0"
    wrapper.added_parameter_count = 0
    wrapper.added_active_parameters = 0
    wrapper.extra_matmul_flops_per_token = 0
    return wrapper
