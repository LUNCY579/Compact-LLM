"""Frozen, intact pretrained FFNs in one parallel post-L13 residual control.

Copies are wrapper-local: the canonical model never acquires new modules or
state-dict entries. This module defines mechanics, not production geometry or
calibration admission; the separately frozen protocol supplies those checks.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

import torch
from torch import nn
from transformers.activations import SiLUActivation
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3MLP, Qwen3RMSNorm,
)

from shared.integrity import state_hash
from .operators import Reapplication, SafetyError, hidden_metrics


SOURCE_LAYERS = (12, 13, 14)
INSERTION_LAYER = 13


def _frozen(module: nn.Module) -> None:
    if any(child.training for child in module.modules()):
        raise SafetyError("parallel FFN modules must remain in eval mode")
    if any(p.requires_grad or p.grad is not None for p in module.parameters()):
        raise SafetyError("parallel FFN parameters must stay frozen without gradients")


def _intact(mlp: nn.Module, norm: nn.Module) -> tuple[int, int, float]:
    """Require actual complete installed Qwen3 operations, without tensor surgery."""
    if type(mlp) is not Qwen3MLP or type(norm) is not Qwen3RMSNorm:
        raise SafetyError("intact installed Qwen3 MLP and RMSNorm required")
    if set(mlp.state_dict()) != {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}:
        raise SafetyError("unexpected MLP parameters or buffers")
    if set(norm.state_dict()) != {"weight"}:
        raise SafetyError("unexpected RMSNorm parameters or buffers")
    d, f = mlp.hidden_size, mlp.intermediate_size
    if type(d) is not int or type(f) is not int or min(d, f) <= 0:
        raise SafetyError("invalid whole-FFN dimensions")
    if (mlp.config.hidden_size, mlp.config.intermediate_size, mlp.config.hidden_act) != (d, f, "silu"):
        raise SafetyError("MLP geometry or canonical SiLU changed")
    if type(mlp.act_fn) is not SiLUActivation:
        raise SafetyError("canonical installed SiLUActivation required")
    for name, shape in (("gate_proj", (f, d)), ("up_proj", (f, d)), ("down_proj", (d, f))):
        projection = getattr(mlp, name)
        if (type(projection) is not nn.Linear or projection.bias is not None or
                tuple(projection.weight.shape) != shape or
                (projection.out_features, projection.in_features) != shape):
            raise SafetyError("complete, unmodified bias-free FFN matrices required")
    epsilon = norm.variance_epsilon
    if (tuple(norm.weight.shape) != (d,) or isinstance(epsilon, bool) or
            not isinstance(epsilon, (float, int)) or not math.isfinite(epsilon) or epsilon <= 0):
        raise SafetyError("intact learned RMSNorm and positive epsilon required")
    for module in (mlp, norm):
        _frozen(module)
        for child in module.modules():
            if ("forward" in child.__dict__ or hasattr(child, "_orig_mod") or
                    child._forward_hooks or child._forward_pre_hooks):
                raise SafetyError("overridden, compiled or hooked FFN operations unsupported")
        if any(p.dtype != torch.float32 or p.device.type not in {"cpu", "cuda"}
               for p in module.parameters()):
            raise SafetyError("parallel FFN requires CPU/CUDA FP32 parameters")
    return d, f, float(epsilon)


class _FrozenFFNBranch(nn.Module):
    def __init__(self, mlp: Qwen3MLP, norm: Qwen3RMSNorm):
        super().__init__()
        # Keep ordinary version-tracked parameters even when the caller creates
        # the wrapper within an inference-mode scope.
        with torch.inference_mode(False), torch.no_grad():
            self.post_attention_layernorm = deepcopy(norm)
            self.mlp = deepcopy(mlp)
        self.eval().requires_grad_(False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.post_attention_layernorm(hidden))


class FrozenParallelFFN(nn.Module):
    """M(h) = (F12(h) + F13(h)) + F14(h), preserving FP32 sum order.

Use ``from_layers`` with frozen canonical decoder modules. Tiny actual Qwen3
modules are accepted for mechanics fixtures; production dimensions are a
protocol/caller responsibility. ``state_sha256`` is device-independent and
includes every copied matrix and learned norm coefficient.
    """

    @classmethod
    def from_layers(cls, layers: Mapping[int, Qwen3DecoderLayer]) -> FrozenParallelFFN:
        return cls(layers)

    def __init__(self, layers: Mapping[int, Qwen3DecoderLayer]):
        super().__init__()
        if (not isinstance(layers, Mapping) or set(layers) != set(SOURCE_LAYERS) or
                any(type(index) is not int for index in layers)):
            raise ValueError("exactly canonical source layers 12, 13, 14 required")
        branches, provenance = {}, {}
        geometries = set()
        for index in SOURCE_LAYERS:
            layer = layers[index]
            if type(layer) is not Qwen3DecoderLayer or layer.self_attn.layer_idx != index:
                raise SafetyError("source must be the identified actual Qwen3 decoder layer")
            mlp, norm = layer.mlp, layer.post_attention_layernorm
            d, f, epsilon = _intact(mlp, norm)
            geometries.add((d, f))
            original_mlp, original_norm = state_hash(mlp), state_hash(norm)
            branch = _FrozenFFNBranch(mlp, norm)
            copied_mlp, copied_norm = state_hash(branch.mlp), state_hash(branch.post_attention_layernorm)
            if ((original_mlp, original_norm) != (copied_mlp, copied_norm) or
                    original_mlp != state_hash(mlp) or original_norm != state_hash(norm)):
                raise SafetyError("canonical FFN source/copy identity mismatch")
            for source, copied in ((mlp, branch.mlp), (norm, branch.post_attention_layernorm)):
                for left, right in zip(source.parameters(), copied.parameters(), strict=True):
                    if left is right or left.data_ptr() == right.data_ptr():
                        raise SafetyError("FFN storage must be a wrapper-local deep copy")
            branches[str(index)] = branch
            provenance[str(index)] = {
                "source_layer": index, "source_mlp_sha256": original_mlp,
                "copy_mlp_sha256": copied_mlp, "source_norm_sha256": original_norm,
                "copy_norm_sha256": copied_norm, "norm_epsilon": epsilon,
                "hidden_size": d, "intermediate_size": f,
                "mlp_class": type(mlp).__name__, "norm_class": type(norm).__name__,
                "activation_class": type(mlp.act_fn).__name__,
            }
        if len(geometries) != 1:
            raise SafetyError("parallel FFNs must share intact hidden/intermediate geometry")
        self.branches = nn.ModuleDict(branches)
        self.hidden_size, self.intermediate_size = geometries.pop()
        self._branch_provenance = provenance
        self.eval().requires_grad_(False)
        if any(not bool(torch.isfinite(p).all()) for p in self.parameters()):
            raise FloatingPointError("nonfinite pretrained FFN coefficients")
        self._state_sha256 = state_hash(self)
        self._behavior = self._behavior_signature()
        self._parameter_versions = self._parameter_signature()

    @property
    def branch_provenance(self) -> dict:
        return deepcopy(self._branch_provenance)

    @property
    def source_hashes(self) -> dict:
        return {index: {"mlp": value["source_mlp_sha256"],
                        "post_attention_layernorm": value["source_norm_sha256"]}
                for index, value in self._branch_provenance.items()}

    @property
    def copy_hashes(self) -> dict:
        return {index: {"mlp": value["copy_mlp_sha256"],
                        "post_attention_layernorm": value["copy_norm_sha256"]}
                for index, value in self._branch_provenance.items()}

    @property
    def state_sha256(self) -> str:
        return self._state_sha256

    @property
    def parameter_inventory(self) -> dict:
        matrix = 9 * self.hidden_size * self.intermediate_size
        norms = 3 * self.hidden_size
        return {"matrix_parameters": matrix, "norm_parameters": norms,
                "resident_copied_parameters": matrix + norms,
                "active_copied_parameters": matrix + norms,
                "resident_bytes": sum(p.numel() * p.element_size() for p in self.parameters()),
                "newly_learned_parameters": 0,
                "extra_matmul_flops_per_token": 2 * matrix}

    def _behavior_signature(self) -> tuple:
        if tuple(self.branches) != tuple(str(i) for i in SOURCE_LAYERS):
            raise SafetyError("parallel branch ordering changed")
        signature = []
        for index, branch in self.branches.items():
            if type(branch) is not _FrozenFFNBranch:
                raise SafetyError("parallel branch implementation changed")
            signature.append((index, id(branch), *_intact(branch.mlp, branch.post_attention_layernorm)))
        for module in (self, self.branches, *self.branches.values()):
            if ("forward" in module.__dict__ or module._forward_hooks or
                    module._forward_pre_hooks or hasattr(module, "_orig_mod")):
                raise SafetyError("parallel transform overrides or hooks unsupported")
        return tuple(signature)

    def _parameter_signature(self) -> tuple:
        return tuple((name, id(p), p.data_ptr(), p._version, str(p.device), p.dtype, tuple(p.shape))
                     for name, p in self.named_parameters())

    def _check_behavior(self) -> None:
        _frozen(self)
        if self._behavior_signature() != self._behavior:
            raise SafetyError("parallel FFN treatment changed")

    def assert_unchanged(self) -> str:
        """Full byte check at lifecycle boundaries, including ``.data`` changes."""
        self._check_behavior()
        current = state_hash(self)
        if current != self.state_sha256:
            raise SafetyError("parallel FFN copied parameter state changed")
        # A device transfer before activation is legitimate only if exact bytes
        # survived. Cheap per-forward checks then bind this particular storage.
        self._parameter_versions = self._parameter_signature()
        return current

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self._check_behavior()
        if self._parameter_signature() != self._parameter_versions:
            raise SafetyError("parallel FFN parameter storage/version changed")
        if torch.is_grad_enabled():
            raise SafetyError("parallel FFN requires inference/no_grad mode")
        exemplar = next(self.parameters())
        if (hidden.dtype != torch.float32 or hidden.device != exemplar.device or
                hidden.ndim < 2 or hidden.shape[-1] != self.hidden_size or
                torch.is_autocast_enabled(hidden.device.type)):
            raise SafetyError("parallel FFN requires matching FP32 hidden states without autocast")
        if not bool(torch.isfinite(hidden).all()):
            raise FloatingPointError("nonfinite parallel FFN input")
        # Every complete branch consumes exactly the same ordinary post-L13 h.
        first = self.branches["12"](hidden)
        second = self.branches["13"](hidden)
        third = self.branches["14"](hidden)
        result = (first + second) + third
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("nonfinite parallel FFN residual")
        return result


class NativeFFNControl(Reapplication):
    """Single residual injection after ordinary L13, using frozen local copies."""

    _ACTIVE_TREATMENT = frozenset({"alpha", "enabled", "bounds", "layer", "applications",
                                   "skip", "model", "transform"})

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "active", False) and name in self._ACTIVE_TREATMENT:
            raise SafetyError(f"active parallel FFN treatment cannot change: {name}")
        super().__setattr__(name, value)

    def __init__(self, model: Any, alpha: float, *, enabled: bool = True,
                 emergency_bounds: tuple[float, float] = (.25, 4.0)):
        if (isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or
                not math.isfinite(alpha) or alpha < 0):
            raise ValueError("finite nonnegative control alpha required")
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if (len(emergency_bounds) != 2 or not all(math.isfinite(v) for v in emergency_bounds) or
                not 0 < emergency_bounds[0] < emergency_bounds[1]):
            raise ValueError("invalid registered hidden RMS bounds")
        super().__init__(model, layer=INSERTION_LAYER, applications=1,
                         enabled=enabled, emergency_bounds=tuple(emergency_bounds))
        if len(model.model.layers) <= max(SOURCE_LAYERS):
            raise ValueError("canonical source layers 12, 13, 14 required")
        self.alpha = float(alpha)
        self.transform = FrozenParallelFFN.from_layers({i: model.model.layers[i] for i in SOURCE_LAYERS})
        self._bound_treatment = (self.alpha, self.bounds, id(model), id(self.transform))
        inventory = self.transform.parameter_inventory
        self.parameter_inventory = inventory
        self.added_parameter_count = inventory["resident_copied_parameters"]
        self._active_parameter_count = inventory["active_copied_parameters"]
        self._extra_matmul_flops = inventory["extra_matmul_flops_per_token"]
        self.newly_learned_parameters = 0
        self.control_calls = 0

    @property
    def extra_layer_applications(self) -> int:
        # Three FFNs are not extra full-decoder calls. The shared estimator uses
        # extra_matmul_flops_per_token for this operator's actual matrix work.
        return 0

    @property
    def added_active_parameters(self) -> int:
        return self._active_parameter_count if self.enabled else 0

    @property
    def extra_matmul_flops_per_token(self) -> int:
        # Zero alpha still computes M for the identity check; disabled mode
        # installs no decoder wrapper and performs no control matrix work.
        return self._extra_matmul_flops if self.enabled else 0

    def _check_treatment(self) -> None:
        if ((self.alpha, self.bounds, id(self.model), id(self.transform)) != self._bound_treatment or
                self.layer != INSERTION_LAYER or self.applications != 1 or self.skip or
                type(self.enabled) is not bool):
            raise SafetyError("parallel FFN treatment differs from its construction")

    def __enter__(self) -> NativeFFNControl:
        self._check_treatment()
        self.transform.assert_unchanged()
        return super().__enter__()

    def _guard(self, module: Any, args: tuple, kwargs: dict) -> None:
        self._check_treatment()
        super()._guard(module, args, kwargs)

    def _forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_treatment()
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
        residual = self.transform(normal)
        self.control_calls += 1
        output = normal + self.alpha * residual
        record, _ = hidden_metrics(output, normal, normal, delta, 2)
        self.min_rms_ratio = min(self.min_rms_ratio, record["rms_ratio_to_normal"])
        self.max_rms_ratio = max(self.max_rms_ratio, record["rms_ratio_to_normal"])
        if self.capture:
            self.records.append(record)
        if not self.bounds[0] <= record["rms_ratio_to_normal"] <= self.bounds[1]:
            raise SafetyError("parallel FFN hidden RMS outside registered guard")
        return output

    def __exit__(self, *exc: Any) -> None:
        # Always restore the canonical forward before checking final copy bytes,
        # including exceptions, zero-alpha and disabled-mode identity runs.
        super().__exit__(*exc)
        self._check_treatment()
        self.transform.assert_unchanged()
