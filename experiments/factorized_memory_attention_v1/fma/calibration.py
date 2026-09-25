"""Explicit, opt-in frozen-backbone calibration for factorized memory.

This module only accepts caller-provided local teacher batches. It has no dataset
loader and no network path, so a calibration run cannot silently read a sealed
or confirmation set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from .memory import FactorizedMemoryBank, calibration_loss, freeze_backbone_for_calibration


@dataclass(frozen=True)
class CalibrationResult:
    steps: int
    initial_loss: float
    final_loss: float
    data_ids: tuple[str, ...]
    backbone_trainable: int
    memory_trainable: int

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "data_ids": list(self.data_ids),
            "backbone_trainable": self.backbone_trainable,
            "memory_trainable": self.memory_trainable,
        }


def calibrate_frozen_backbone(
    backbone: nn.Module,
    bank: FactorizedMemoryBank,
    batches: Iterable[tuple[Tensor, int, Tensor, Tensor]],
    *,
    data_ids: Iterable[str],
    steps: int = 1,
    learning_rate: float = 1e-3,
) -> CalibrationResult:
    """Fit memory parameters to caller-supplied teacher residuals.

    Each batch is `(token_ids, layer_idx, standard_value, key_pre_rope)`.
    ``data_ids`` is required as an audit boundary; this helper does not know how
    to obtain data and therefore cannot access sealed data on its own.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    ids = tuple(str(item) for item in data_ids)
    if not ids:
        raise ValueError("calibration requires explicit local data_ids")

    trainability = freeze_backbone_for_calibration(backbone, bank)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in bank.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    cached = list(batches)
    if not cached:
        raise ValueError("calibration requires at least one teacher batch")

    initial_loss = float("nan")
    final_loss = float("nan")
    completed = 0
    bank.train()
    for step in range(steps):
        token_ids, layer_idx, standard_value, key_pre_rope = cached[step % len(cached)]
        loss = calibration_loss(bank, token_ids, layer_idx, standard_value, key_pre_rope)
        if step == 0:
            initial_loss = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("calibration produced a non-finite loss")
        final_loss = float(loss.detach().cpu())
        completed += 1

    return CalibrationResult(
        steps=completed,
        initial_loss=initial_loss,
        final_loss=final_loss,
        data_ids=ids,
        backbone_trainable=trainability["backbone_trainable"],
        memory_trainable=trainability["memory_trainable"],
    )
