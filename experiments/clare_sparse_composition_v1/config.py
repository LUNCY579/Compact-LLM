"""Frozen, validated study controls. No automatic sweep or native launch."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

ARMS = ("dense_sequential", "independent_add", "magnitude_sparse", "random_disjoint", "learned_disjoint")


@dataclass(frozen=True)
class Config:
    arm: str = "learned_disjoint"
    order: tuple[str, ...] = ("R", "U", "S", "P")
    seed: int = 17
    rho: float = 0.8
    discovery_steps: int = 4
    train_steps: int = 8
    discovery_l1: float = 0.0001
    lr: float = 0.001
    weight_decay: float = 0.0
    optimizer: str = "AdamW"
    max_coordinates: int = 8_000_000

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError("unsupported arm; shared/private is design-only")
        if not isinstance(self.order, tuple) or not self.order or any(type(s) is not str or s not in ("R", "U", "S", "P") for s in self.order) or len(set(self.order)) != len(self.order):
            raise ValueError("order must contain distinct R/U/S/P skills")
        for name in ("seed", "discovery_steps", "train_steps", "max_coordinates"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"invalid {name}")
        for name in ("rho", "discovery_l1", "lr", "weight_decay"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"invalid {name}")
        if not 0 < self.rho < 1 or self.discovery_l1 < 0 or self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("invalid numerical controls")
        if self.optimizer != "AdamW":
            raise ValueError("v1 qualifies AdamW only")

    def payload(self):
        return asdict(self)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["order"] = tuple(value.get("order", ("R", "U", "S", "P")))
        return cls(**value)
