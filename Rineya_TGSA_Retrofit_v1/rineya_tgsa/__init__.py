"""Pretrained Rineya/Qwen trajectory-guided sequential attention retrofit."""

from .attention import (
    NativeReferenceAttention,
    RineyaTGSAAttention,
    RotaryEmbedding,
    TrajectoryEncoder,
    TrajectoryTelemetry,
    apply_rotary,
    rotate_half,
)
from .config import RineyaTGSAConfig

__all__ = [
    "NativeReferenceAttention",
    "RineyaTGSAAttention",
    "RineyaTGSAConfig",
    "RotaryEmbedding",
    "TrajectoryEncoder",
    "TrajectoryTelemetry",
    "apply_rotary",
    "rotate_half",
]
