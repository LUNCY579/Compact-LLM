"""Trajectory-Guided Sequential Attention research reference package."""

from .config import AblationArm, ControlSource, TGSAConfig, TelemetryLevel, TrajectoryMode
from .metrics import (
    PairedBootstrapResult,
    count_parameters,
    paired_bootstrap_mean_difference,
    telemetry_summary,
)
from .module import ParallelCausalSelfAttention, TrajectoryGuidedSequentialAttention
from .mtp import (
    MTPConfig,
    MTPMode,
    SequentialMTPHead,
    TGSAWithMTPOutput,
    TinyTGSAWithMTP,
    build_tgsa_mtp_model,
)
from .telemetry import TGSATelemetry

__all__ = [
    "AblationArm",
    "ControlSource",
    "MTPConfig",
    "MTPMode",
    "ParallelCausalSelfAttention",
    "PairedBootstrapResult",
    "SequentialMTPHead",
    "TGSAConfig",
    "TGSATelemetry",
    "TGSAWithMTPOutput",
    "TelemetryLevel",
    "TinyTGSAWithMTP",
    "TrajectoryGuidedSequentialAttention",
    "TrajectoryMode",
    "build_tgsa_mtp_model",
    "count_parameters",
    "paired_bootstrap_mean_difference",
    "telemetry_summary",
]
