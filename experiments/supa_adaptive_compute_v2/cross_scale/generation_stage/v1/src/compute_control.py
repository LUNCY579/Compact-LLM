"""Track B's independent frozen, matrix-work-matched residual control.

Only FrozenSwiGLU's mathematical implementation is shared with Track A. The
seed, geometry, fitted scalar, calibration data and evidence belong to Track B.
The transform is wrapper-local and is never attached to the canonical backbone.
"""
from __future__ import annotations

import math
import re

import torch

from native_armr.src.operators import FrozenSwiGLU
from shared.integrity import frozen_inventory, state_hash
from armr.hooks import Reapplication, SafetyError
from armr.metrics import hidden_metrics
from .operators import buffer_hashes


CONTROL = {"kind": "independent_frozen_swiglu", "hidden_size": 1024,
           "intermediate_size": 34560, "seed": 2702, "weight_std": .02,
           "unit_rms_epsilon": 1e-6, "insertion_layer": 19}
PARAMETERS = 106168320
MATMUL_FLOPS_PER_TOKEN = 212336640
GATE_GRID = (.03, .10, .20, .40)


def new_transform():
    """Draw reference tensors with a private CPU generator, even for CUDA use."""
    return FrozenSwiGLU(CONTROL["hidden_size"], CONTROL["intermediate_size"],
                       seed=CONTROL["seed"], std=CONTROL["weight_std"],
                       rms_epsilon=CONTROL["unit_rms_epsilon"])


def _scalar(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (not positive or value > 0))


class _FrozenComputeWrapper(Reapplication):
    """Low-level hook mechanics, also usable with small deterministic fixtures.

    Real campaigns must construct CrossScaleComputeControl, which enforces the
    production geometry and committed-artifact tensor identity supplied by the
    stage's admission caller. This low-level class does not license inference.
    """

    def __init__(self, model, transform, *, alpha, gate, expected_state_sha256,
                 layer, enabled, emergency_bounds):
        if not _scalar(alpha, positive=True):
            raise ValueError("finite positive calibration alpha required")
        if not _scalar(gate) or gate not in (0., *GATE_GRID):
            raise ValueError("only registered gate doses or mechanical zero allowed")
        if type(enabled) is not bool or type(layer) is not int or layer < 3:
            raise ValueError("explicit enabled flag and non-stem insertion required")
        if (len(emergency_bounds) != 2 or any(not _scalar(v, positive=True) for v in emergency_bounds)
                or emergency_bounds[0] >= emergency_bounds[1]):
            raise ValueError("finite ordered positive RMS bounds required")
        if not isinstance(expected_state_sha256, str) or not re.fullmatch("[0-9a-f]{64}", expected_state_sha256):
            raise ValueError("a calibrated control state digest is required")
        frozen_inventory(model)
        frozen_inventory(transform)
        exemplar = next(model.parameters())
        if exemplar.dtype != torch.float32 or exemplar.device.type not in ("cpu", "cuda"):
            raise ValueError("control requires one FP32 CPU/CUDA device")
        if any(p.dtype != torch.float32 or p.device != exemplar.device for p in model.parameters()):
            raise ValueError("canonical model tensors must share one FP32 device")
        super().__init__(model, layer=layer, applications=1, enabled=enabled,
                         emergency_bounds=tuple(emergency_bounds))
        self.transform = transform.to(device=exemplar.device, dtype=torch.float32)
        if state_hash(self.transform) != expected_state_sha256:
            raise RuntimeError("XM initialization differs from its calibrated frozen state")
        self.alpha, self.gate = float(alpha), float(gate)
        self.expected_state_sha256 = expected_state_sha256
        self.arm = "XM"
        self.control_calls = 0
        self.added_parameter_count = sum(p.numel() for p in transform.parameters())
        self.additional_matrix_parameters = sum(p.numel() for p in transform.parameters() if p.ndim == 2)
        self.integrity_receipt = None
        self._configured = self._treatment()

    @property
    def added_active_parameters(self):
        return self.added_parameter_count if self.enabled and self.gate != 0 else 0

    @property
    def extra_matmul_flops_per_token(self):
        return 2 * self.additional_matrix_parameters if self.enabled and self.gate != 0 else 0

    def _treatment(self):
        return {"alpha": self.alpha, "gate": self.gate, "enabled": self.enabled,
                "layer": self.layer, "applications": self.applications, "skip": self.skip,
                "bounds": tuple(self.bounds), "rms_epsilon": self.transform.rms_epsilon,
                "expected_state_sha256": self.expected_state_sha256}

    def _check_treatment(self):
        if (self._treatment() != self._configured or type(self.enabled) is not bool
                or type(self.skip) is not bool or type(self.layer) is not int
                or type(self.applications) is not int or not _scalar(self.alpha, positive=True)
                or not _scalar(self.gate)):
            raise RuntimeError("registered XM scalar, gate or treatment changed")
        frozen_inventory(self.transform)

    def __enter__(self):
        if self.active:
            raise RuntimeError("wrapper already active; integrity baseline cannot be replaced")
        self._check_treatment()
        frozen_inventory(self.model)
        self._before_state = state_hash(self.transform)
        self._before_buffers = buffer_hashes(self.transform)
        if self._before_state != self.expected_state_sha256:
            raise RuntimeError("XM state changed since calibration")
        return super().__enter__()

    def _guard(self, module, args, kwargs):
        self._check_treatment()
        return super()._guard(module, args, kwargs)

    def _forward(self, *args, **kwargs):
        self._check_treatment()
        if any(kwargs.get(k) is not None for k in ("past_key_values", "past_key_value", "cache")):
            raise SafetyError("decoder received a cache")
        if kwargs.get("use_cache", False):
            raise SafetyError("decoder cache mode forbidden")
        incoming = args[0] if args else kwargs["hidden_states"]
        normal = self.original(*args, **kwargs)
        self.calls += 1
        if not isinstance(normal, torch.Tensor):
            raise SafetyError("only audited tensor-returning Qwen3 decoder supported")
        first, delta = hidden_metrics(normal, normal, incoming, None, 1)
        if self.capture:
            self.records.append(first)
        if self.gate == 0:
            return normal
        refined = normal + (self.gate * self.alpha) * self.transform(normal)
        self.control_calls += 1
        record, _ = hidden_metrics(refined, normal, normal, delta, 2)
        self.min_rms_ratio = min(self.min_rms_ratio, record["rms_ratio_to_normal"])
        self.max_rms_ratio = max(self.max_rms_ratio, record["rms_ratio_to_normal"])
        if self.capture:
            self.records.append(record)
        if not self.bounds[0] <= record["rms_ratio_to_normal"] <= self.bounds[1]:
            raise SafetyError("XM hidden RMS outside registered guard")
        return refined

    def __exit__(self, exc_type, exc, traceback):
        super().__exit__(exc_type, exc, traceback)
        try:
            frozen_inventory(self.model)
            inventory = frozen_inventory(self.transform)
            after = state_hash(self.transform)
            buffers = buffer_hashes(self.transform)
            self.integrity_receipt = {
                **inventory, "state_sha256_before": self._before_state,
                "state_sha256_after": after, "buffers_before": self._before_buffers,
                "buffers_after": buffers, "treatment_before": self._configured,
                "treatment_after": self._treatment(),
            }
            self._check_treatment()
            if after != self._before_state or buffers != self._before_buffers:
                raise RuntimeError("frozen XM state or buffer changed")
        except BaseException as integrity_error:
            if exc is not None:
                raise BaseExceptionGroup("XM execution and integrity failed", [exc, integrity_error])
            raise
        return False


class CrossScaleComputeControl(_FrozenComputeWrapper):
    """Exactly 106,168,320 independent matrices' coefficients, with B-only alpha."""

    def __init__(self, model, *, alpha, gate, expected_state_sha256, layer=19,
                 enabled=True, emergency_bounds=(.25, 4.)):
        if (type(layer) is not int or layer != CONTROL["insertion_layer"]
                or model.config.hidden_size != CONTROL["hidden_size"]
                or model.config.rms_norm_eps != CONTROL["unit_rms_epsilon"]):
            raise ValueError("XM requires Supa width1024/epsilon1e-6 at post-L19")
        # Validate scalars before allocating the 405 MiB independent transform.
        if not _scalar(alpha, positive=True) or not _scalar(gate) or gate not in (0., *GATE_GRID):
            raise ValueError("unregistered XM alpha or gate")
        super().__init__(model, new_transform(), alpha=alpha, gate=gate,
                         expected_state_sha256=expected_state_sha256, layer=layer,
                         enabled=enabled, emergency_bounds=emergency_bounds)
        if (self.added_parameter_count != PARAMETERS
                or 2 * self.additional_matrix_parameters != MATMUL_FLOPS_PER_TOKEN):
            raise RuntimeError("XM actual parameter/FLOP count differs from registration")
