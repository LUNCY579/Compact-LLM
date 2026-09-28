"""Configuration contracts for the factorized Memory Attention qualification lab."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ModelShape:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    model_type: str = "qwen3"

    def __post_init__(self) -> None:
        ints = (
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim",
        )
        for name in ints:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")

    @property
    def value_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    def to_dict(self) -> dict:
        data = asdict(self)
        data["value_width"] = self.value_width
        data["query_width"] = self.query_width
        return data


@dataclass(frozen=True)
class FMAConfig:
    rank: int = 32
    layers: tuple[int, ...] | None = None
    gate: float = 0.0
    init_std: float = 0.02
    affine_head_rmsnorm: bool = True

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not 0.0 <= float(self.gate) <= 1.0:
            raise ValueError("gate must be in [0, 1]")
        if self.init_std <= 0:
            raise ValueError("init_std must be positive")
        if self.layers is not None:
            if not self.layers:
                raise ValueError("layers cannot be empty when provided")
            if any(type(i) is not int or i < 0 for i in self.layers):
                raise ValueError("layer indices must be non-negative integers")
            if len(set(self.layers)) != len(self.layers):
                raise ValueError("layer indices must be unique")

    def resolved_layers(self, shape: ModelShape) -> tuple[int, ...]:
        layers = self.layers if self.layers is not None else tuple(range(shape.num_hidden_layers))
        if any(i >= shape.num_hidden_layers for i in layers):
            raise ValueError(f"configured layer outside 0..{shape.num_hidden_layers - 1}")
        return tuple(sorted(layers))


def shape_from_dict(raw: dict) -> ModelShape:
    required = {
        "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "rms_norm_eps",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise KeyError(f"missing model config keys: {missing}")
    return ModelShape(
        vocab_size=int(raw["vocab_size"]),
        hidden_size=int(raw["hidden_size"]),
        num_hidden_layers=int(raw["num_hidden_layers"]),
        num_attention_heads=int(raw["num_attention_heads"]),
        num_key_value_heads=int(raw["num_key_value_heads"]),
        head_dim=int(raw["head_dim"]),
        rms_norm_eps=float(raw["rms_norm_eps"]),
        model_type=str(raw.get("model_type", "unknown")),
    )


def load_shape(path: str | Path) -> ModelShape:
    return shape_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def normalize_layers(layers: Iterable[int], total: int) -> tuple[int, ...]:
    out = tuple(sorted(set(int(i) for i in layers)))
    if not out or out[0] < 0 or out[-1] >= total:
        raise ValueError("invalid layer set")
    return out
