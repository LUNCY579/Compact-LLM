"""Trajectory-Guided Sequential Attention research reference package."""

from .config import AblationArm, ControlSource, TGSAConfig, TelemetryLevel, TrajectoryMode
from .metrics import (
    PairedBootstrapResult,
    count_parameters,
    paired_bootstrap_mean_difference,
    telemetry_summary,
)
from .module import ParallelCausalSelfAttention, TrajectoryGuidedSequentialAttention
from .telemetry import TGSATelemetry

__all__ = [
    "AblationArm",
    "ControlSource",
    "ParallelCausalSelfAttention",
    "PairedBootstrapResult",
    "TGSAConfig",
    "TGSATelemetry",
    "TelemetryLevel",
    "TrajectoryGuidedSequentialAttention",
    "TrajectoryMode",
    "count_parameters",
    "paired_bootstrap_mean_difference",
    "telemetry_summary",
]
