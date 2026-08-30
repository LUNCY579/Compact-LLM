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
    JointCausalLoss,
    MTPConfig,
    MTPLoss,
    MTPResidualHead,
    MultiTokenPredictionHeads,
    build_offset_targets,
    causal_next_token_loss,
    compute_mtp_loss,
    joint_causal_mtp_loss,
)
from .phase2_decode import (
    DecodeResult,
    DecodeStats,
    assert_exact_greedy_equivalence,
    greedy_generate,
    verified_mtp_generate,
)
from .telemetry import TGSATelemetry

__all__ = [
    "AblationArm",
    "ControlSource",
    "DecodeResult",
    "DecodeStats",
    "JointCausalLoss",
    "MTPConfig",
    "MTPLoss",
    "MTPResidualHead",
    "MultiTokenPredictionHeads",
    "ParallelCausalSelfAttention",
    "PairedBootstrapResult",
    "TGSAConfig",
    "TGSATelemetry",
    "TelemetryLevel",
    "TrajectoryGuidedSequentialAttention",
    "TrajectoryMode",
    "assert_exact_greedy_equivalence",
    "build_offset_targets",
    "causal_next_token_loss",
    "compute_mtp_loss",
    "count_parameters",
    "greedy_generate",
    "joint_causal_mtp_loss",
    "paired_bootstrap_mean_difference",
    "telemetry_summary",
    "verified_mtp_generate",
]
