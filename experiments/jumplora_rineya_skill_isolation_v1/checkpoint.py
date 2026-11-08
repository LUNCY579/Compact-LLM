"""Small checkpoint and mutation helpers used by qualification tests."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
from typing import Any

import torch
from torch import nn


def state_digest(module: nn.Module) -> str:
    """Hash a CPU state_dict without mutating or serializing through a file."""
    buffer = io.BytesIO()
    state = {name: value.detach().cpu().contiguous()
             for name, value in sorted(module.state_dict().items())}
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone()
            for name, value in module.state_dict().items()}


def assert_state_equal(module: nn.Module, expected: dict[str, torch.Tensor]) -> None:
    actual = module.state_dict()
    if set(actual) != set(expected):
        raise AssertionError("state_dict keys changed")
    for name in expected:
        if not torch.equal(actual[name].detach().cpu(), expected[name]):
            raise AssertionError(f"state changed for {name}")


def roundtrip_state(module: nn.Module) -> nn.Module:
    """Create an independent same-class copy through a state_dict round trip."""
    copy = deepcopy(module)
    copy.load_state_dict(module.state_dict())
    return copy
