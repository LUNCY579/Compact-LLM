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
from .promotion import (
    LEDGER_SCHEMA_VERSION,
    MANDATORY_PROMOTION_GATES,
    promotion_failures,
    validate_candidate,
    validate_ledger,
)

__all__ = [
    "NativeReferenceAttention",
    "RineyaTGSAAttention",
    "RineyaTGSAConfig",
    "LEDGER_SCHEMA_VERSION",
    "MANDATORY_PROMOTION_GATES",
    "RotaryEmbedding",
    "TrajectoryEncoder",
    "TrajectoryTelemetry",
    "apply_rotary",
    "rotate_half",
    "promotion_failures",
    "validate_candidate",
    "validate_ledger",
]
