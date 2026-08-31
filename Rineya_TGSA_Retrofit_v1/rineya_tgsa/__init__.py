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
from .trial_queue import (
    TRIAL_QUEUE_SCHEMA_VERSION,
    validate_trial,
    validate_trial_queue,
)
from .historical_soup import (
    ARTIFACT_SCHEMA_VERSION,
    PREFLIGHT_SCHEMA_VERSION,
    TRIAL_SCHEMA_VERSION,
    build_candidate,
    preflight,
    verify_candidate,
)
from .historical_dose import (
    DOSE_ARTIFACT_SCHEMA_VERSION,
    DOSE_TRIAL_SCHEMA_VERSION,
    build_dose_trial,
    compare_tensors_exact,
    materialize_dose,
    preflight_dose,
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
    "TRIAL_QUEUE_SCHEMA_VERSION",
    "TRIAL_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_VERSION",
    "ARTIFACT_SCHEMA_VERSION",
    "DOSE_ARTIFACT_SCHEMA_VERSION",
    "DOSE_TRIAL_SCHEMA_VERSION",
    "apply_rotary",
    "rotate_half",
    "promotion_failures",
    "validate_candidate",
    "validate_ledger",
    "validate_trial",
    "validate_trial_queue",
    "build_candidate",
    "preflight",
    "verify_candidate",
    "build_dose_trial",
    "compare_tensors_exact",
    "materialize_dose",
    "preflight_dose",
]
