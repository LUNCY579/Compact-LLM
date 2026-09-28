"""Model-agnostic module injection for the existing Transformers/PEFT runtime.

This layer does not load a checkpoint or choose a dataset. It replaces only
leaf ``nn.Linear`` modules whose names end in the preregistered q_proj/v_proj
targets, so the repository's existing model loader and evaluator can remain
the source of model/tokenizer behavior.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .jump_lora import JumpLoRALinear, JumpSchedule, LoRALinear, build_adapter, deterministic_topk_mask


TARGET_SUFFIXES = ("q_proj", "v_proj")


def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    if "." not in name:
        return model, name
    parent_name, leaf = name.rsplit(".", 1)
    return model.get_submodule(parent_name), leaf


@dataclass
class AdapterBinding:
    name: str
    adapter: LoRALinear
    original: nn.Linear


class InjectedArm:
    """Lifecycle and accounting handle for one injected arm."""

    def __init__(self, model: nn.Module, bindings: list[AdapterBinding], arm: str):
        self.model = model
        self.bindings = bindings
        self.arm = str(arm).upper()
        self._restored = False

    @property
    def adapters(self) -> tuple[LoRALinear, ...]:
        return tuple(binding.adapter for binding in self.bindings)

    def set_step(self, step: int) -> None:
        for adapter in self.adapters:
            if isinstance(adapter, JumpLoRALinear):
                adapter.set_step(step)

    def paper_initial_support_counts(self) -> dict[str, int]:
        """Algorithm-1 support budget for each adapted weight matrix.

        JumpLoRA learns a threshold for each layer/update matrix. Therefore the
        initialization cardinality is derived independently for every injected
        q/v module, not from one global cross-layer threshold.
        """
        result: dict[str, int] = {}
        for binding in self.bindings:
            adapter = binding.adapter
            result[binding.name] = min(
                adapter.raw_delta().numel(),
                adapter.lora_A.numel() + adapter.lora_B.numel(),
            )
        return result

    def initialize_jump_thresholds(self,
                                   active_counts: Mapping[str, int] | int | None = None) -> dict[str, int]:
        """Initialize each J3 module's own threshold from its unscaled ``B @ A``.

        A single integer is accepted only for a one-module fixture. Multi-module
        native use must bind explicit per-module counts so a global threshold
        cannot silently change the layerwise sparsity profile.
        """
        jumps = [(binding.name, binding.adapter) for binding in self.bindings
                 if isinstance(binding.adapter, JumpLoRALinear)]
        if not jumps:
            return {}
        if active_counts is None:
            counts = self.paper_initial_support_counts()
        elif isinstance(active_counts, int):
            if len(jumps) != 1:
                raise ValueError("multi-module J3 threshold initialization requires per-module counts")
            counts = {jumps[0][0]: int(active_counts)}
        else:
            counts = {str(k): int(v) for k, v in active_counts.items()}
            names = {name for name, _ in jumps}
            if set(counts) != names:
                raise ValueError("J3 threshold count module set mismatch")
        realized: dict[str, int] = {}
        for name, adapter in jumps:
            count = counts[name]
            if not 0 <= count <= adapter.raw_delta().numel():
                raise ValueError(f"support count out of range for {name}")
            adapter.initialize_threshold(count)
            realized[name] = int(adapter.support_mask().sum().item())
            if realized[name] != count:
                raise RuntimeError(f"threshold cardinality mismatch for {name}")
        return realized

    def freeze_matched_support_counts(self, active_counts: Mapping[str, int]) -> dict[str, int]:
        """Freeze J1/J2 deterministic top-k masks to J3's per-module counts."""
        counts = {str(k): int(v) for k, v in active_counts.items()}
        names = {binding.name for binding in self.bindings}
        if set(counts) != names:
            raise ValueError("matched-density module set mismatch")
        realized: dict[str, int] = {}
        for binding in sorted(self.bindings, key=lambda item: item.name):
            adapter = binding.adapter
            if not hasattr(adapter, "set_frozen_mask"):
                raise ValueError(f"arm {self.arm} does not support frozen masks")
            count = counts[binding.name]
            if not 0 <= count <= adapter.raw_delta().numel():
                raise ValueError(f"support count out of range for {binding.name}")
            mask = deterministic_topk_mask(adapter.raw_delta(), count)
            adapter.set_frozen_mask(mask)
            realized[binding.name] = int(mask.sum().item())
        return realized

    def freeze_global_topk(self, active_count: int) -> dict[str, int]:
        """Exploratory global-budget helper; not the registered J1/J2 control.

        The registered matched-density comparison uses
        :meth:`freeze_matched_support_counts` so layer/module allocation cannot
        confound support-selection mechanism with support-budget relocation.
        """
        supports = {binding.name: binding.adapter.raw_delta() for binding in self.bindings}
        names = sorted(supports)
        values = torch.cat([supports[name].detach().abs().reshape(-1) for name in names])
        global_mask = deterministic_topk_mask(values, int(active_count)).reshape(-1)
        offset = 0
        counts: dict[str, int] = {}
        for binding in sorted(self.bindings, key=lambda item: item.name):
            size = binding.adapter.raw_delta().numel()
            local = global_mask[offset:offset + size].reshape(binding.adapter.base.weight.shape)
            if not hasattr(binding.adapter, "set_frozen_mask"):
                raise ValueError(f"arm {self.arm} does not support frozen masks")
            binding.adapter.set_frozen_mask(local)
            counts[binding.name] = int(local.sum().item())
            offset += size
        return counts

    def final_support_count(self) -> int:
        return sum(int(adapter.support_mask().sum().item()) for adapter in self.adapters)

    def final_support_counts(self) -> dict[str, int]:
        return {binding.name: int(binding.adapter.support_mask().sum().item())
                for binding in self.bindings}

    def parameter_counts(self) -> dict[str, int]:
        return {
            "base_parameters": sum(p.numel() for binding in self.bindings for p in binding.original.parameters()),
            "trainable_adapter_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "adapter_parameters": sum(adapter.lora_A.numel() + adapter.lora_B.numel()
                                       + (adapter.threshold.numel() if isinstance(adapter, JumpLoRALinear) else 0)
                                       for adapter in self.adapters),
        }

    def support_masks(self) -> dict[str, torch.Tensor]:
        return {binding.name: binding.adapter.support_mask().detach().clone()
                for binding in self.bindings}

    def merge(self, step: int | None = None) -> None:
        for adapter in self.adapters:
            adapter.merge(step)

    def unmerge(self) -> None:
        for adapter in reversed(self.adapters):
            adapter.unmerge()

    def restore(self) -> None:
        if self._restored:
            return
        for binding in reversed(self.bindings):
            parent, leaf = _parent(self.model, binding.name)
            if parent.get_submodule(leaf) is not binding.adapter:
                raise ValueError(f"module changed after injection: {binding.name}")
            setattr(parent, leaf, binding.original)
        self._restored = True


def inject_arm(model: nn.Module, arm: str, *, rank: int = 8, alpha: float = 32.0,
               schedule: JumpSchedule | None = None,
               target_suffixes: Iterable[str] = TARGET_SUFFIXES) -> InjectedArm:
    """Inject one registered arm into q/v leaf linears of an existing model."""
    suffixes = tuple(str(value) for value in target_suffixes)
    candidates = [(name, module) for name, module in model.named_modules()
                  if type(module) is nn.Linear and name.rsplit(".", 1)[-1] in suffixes]
    if not candidates:
        raise ValueError(f"no target linear modules found for suffixes {suffixes}")
    bindings: list[AdapterBinding] = []
    for name, original in sorted(candidates):
        parent, leaf = _parent(model, name)
        adapter = build_adapter(original, arm, rank=rank, alpha=alpha, schedule=schedule)
        setattr(parent, leaf, adapter)
        bindings.append(AdapterBinding(name=name, adapter=adapter, original=original))
    return InjectedArm(model, bindings, arm)
