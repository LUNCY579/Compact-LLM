"""Model-agnostic module injection for the existing Transformers/PEFT runtime.

This layer does not load a checkpoint or choose a dataset. It replaces only
leaf ``nn.Linear`` modules whose names end in the preregistered q_proj/v_proj
targets, so the repository's existing model loader and evaluator can remain
the source of model/tokenizer behavior.
"""
from __future__ import annotations

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

    def initialize_jump_thresholds(self, active_count: int | None = None) -> None:
        jumps = [adapter for adapter in self.adapters if isinstance(adapter, JumpLoRALinear)]
        if not jumps:
            return
        if active_count is None:
            active_count = min(sum(adapter.dense_delta().numel() for adapter in jumps),
                               sum(adapter.lora_A.numel() + adapter.lora_B.numel() for adapter in jumps))
        active_count = int(active_count)
        values = torch.cat([adapter.dense_delta().detach().abs().reshape(-1) for adapter in jumps])
        global_tau = values.sort(descending=True).values
        if active_count <= 0:
            threshold = global_tau[0] + torch.finfo(values.dtype).eps
        elif active_count >= values.numel():
            threshold = global_tau[-1] - torch.finfo(values.dtype).eps
        else:
            threshold = (global_tau[active_count - 1] + global_tau[active_count]) / 2.0
        for adapter in jumps:
            with torch.no_grad():
                adapter.threshold.copy_(threshold.to(adapter.threshold))

    def freeze_global_topk(self, active_count: int) -> dict[str, int]:
        """Freeze J1/J2 support using one deterministic global budget."""
        supports = {binding.name: binding.adapter.dense_delta() for binding in self.bindings}
        names = sorted(supports)
        values = torch.cat([supports[name].detach().abs().reshape(-1) for name in names])
        global_mask = deterministic_topk_mask(values, int(active_count)).reshape(-1)
        offset = 0
        counts: dict[str, int] = {}
        for binding in sorted(self.bindings, key=lambda item: item.name):
            size = binding.adapter.dense_delta().numel()
            local = global_mask[offset:offset + size].reshape(binding.adapter.base.weight.shape)
            if not hasattr(binding.adapter, "set_frozen_mask"):
                raise ValueError(f"arm {self.arm} does not support frozen masks")
            binding.adapter.set_frozen_mask(local)
            counts[binding.name] = int(local.sum().item())
            offset += size
        return counts

    def final_support_count(self) -> int:
        return sum(int(adapter.support_mask().sum().item()) for adapter in self.adapters)

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
