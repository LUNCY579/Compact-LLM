"""Matched experimental arms A/B/C/D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rineya_fwa.falcon2 import FWAConfig
from rineya_fwa.wrapper import InstalledFWA, install_fwa

ARM_SPECS = {
    "A": {"install": False, "recurrent": False, "significance_gate": False},
    "B": {"install": True, "recurrent": False, "significance_gate": False},
    "C": {"install": True, "recurrent": True, "significance_gate": False},
    "D": {"install": True, "recurrent": True, "significance_gate": True},
}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    install: bool
    recurrent: bool
    significance_gate: bool


def arm_spec(name: str) -> ArmSpec:
    key = name.strip().upper()
    if key not in ARM_SPECS:
        raise KeyError(f"unknown arm {name!r}; expected A/B/C/D")
    payload = ARM_SPECS[key]
    return ArmSpec(name=key, **payload)


def apply_arm(
    model: Any,
    name: str,
    *,
    layers: list[int],
    rank: int = 16,
    eps: float = 1e-6,
    init_alpha: float = 0.0,
    beta_raw_init: float = 0.0,
    lambda_raw_init: float = -2.0,
) -> InstalledFWA | None:
    spec = arm_spec(name)
    if not spec.install:
        return None
    cfg = FWAConfig(
        hidden_size=int(model.config.hidden_size),
        n_kv_heads=int(model.config.num_key_value_heads),
        rank=rank,
        eps=eps,
        recurrent=spec.recurrent,
        significance_gate=spec.significance_gate,
        init_alpha=init_alpha,
        beta_raw_init=beta_raw_init,
        lambda_raw_init=lambda_raw_init,
    )
    return install_fwa(model, cfg, layers)


def count_trainable(handle: InstalledFWA | None) -> int:
    if handle is None:
        return 0
    return sum(p.numel() for p in handle.trainable_parameters())


def count_fwa_tensors(handle: InstalledFWA | None) -> int:
    if handle is None:
        return 0
    return sum(p.numel() for block in handle.blocks() for p in block.parameters())
