"""Configuration objects for trajectory-guided sequential attention."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

TrajectoryMode = Literal["none", "raw", "learned"]
ControlSource = Literal["velocity", "history_mean"]
TelemetryLevel = Literal["none", "summary", "control_only", "full"]
AblationArm = Literal[
    "cascade",
    "history_control",
    "raw_delta",
    "learned_delta",
    "qkv_conditioned",
    "predictor_corrector",
    "full",
]


@dataclass(frozen=True, slots=True)
class TGSAConfig:
    """Configuration for :class:`TrajectoryGuidedSequentialAttention`.

    The configuration intentionally exposes the mechanism boundaries required by
    the preregistered ablation ladder. A disabled mechanism must not silently
    remain active through another path.

    ``control_only`` telemetry retains the exact control tensor consumed by each
    trajectory encoder while declining all quadratic attention-map retention.
    It is the required telemetry level for long-context trajectory readouts.
    """

    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    bias: bool = False
    trajectory_rank: int = 16
    trajectory_mode: TrajectoryMode = "learned"
    control_source: ControlSource = "velocity"
    use_state_cascade: bool = True
    use_control_injection: bool = True
    use_predictor: bool = True
    condition_q: bool = True
    condition_k: bool = True
    condition_v: bool = True
    use_importance_bias: bool = True
    normalize_carrier: bool = True
    trajectory_scale: float = 0.5
    max_predictor_step: float = 0.5
    max_correction_step: float = 1.0
    importance_floor: float = 1.0e-4
    importance_logit_scale: float = 1.0
    telemetry_level: TelemetryLevel = "none"
    detach_telemetry: bool = True

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.trajectory_rank <= 0:
            raise ValueError("trajectory_rank must be positive")
        if self.trajectory_mode not in {"none", "raw", "learned"}:
            raise ValueError(
                f"unsupported trajectory_mode: {self.trajectory_mode}"
            )
        if self.control_source not in {"velocity", "history_mean"}:
            raise ValueError(
                f"unsupported control_source: {self.control_source}"
            )
        if self.telemetry_level not in {
            "none",
            "summary",
            "control_only",
            "full",
        }:
            raise ValueError(
                f"unsupported telemetry_level: {self.telemetry_level}"
            )
        if self.trajectory_scale < 0.0:
            raise ValueError("trajectory_scale must be non-negative")
        if self.max_predictor_step < 0.0:
            raise ValueError("max_predictor_step must be non-negative")
        if self.max_correction_step <= 0.0:
            raise ValueError("max_correction_step must be positive")
        if not 0.0 < self.importance_floor <= 1.0:
            raise ValueError("importance_floor must be in (0, 1]")
        if self.importance_logit_scale < 0.0:
            raise ValueError(
                "importance_logit_scale must be non-negative"
            )

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @classmethod
    def for_ablation(
        cls,
        *,
        embed_dim: int,
        num_heads: int,
        arm: AblationArm,
        **overrides: object,
    ) -> "TGSAConfig":
        """Construct one of the mechanism-preserving preregistered ablations.

        ``history_control`` and ``learned_delta`` expose the same current carrier
        and one same-shaped learned control encoder. The former encodes the mean
        of the two preceding carrier states while the latter encodes their finite
        difference. This gives them exactly the same parameterization and, given
        the current carrier, equivalent access to both states. Their comparison
        therefore tests the trajectory inductive bias rather than extra history.
        """

        common = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            trajectory_mode="none",
            control_source="velocity",
            use_state_cascade=True,
            use_control_injection=False,
            use_predictor=False,
            condition_q=False,
            condition_k=False,
            condition_v=False,
            use_importance_bias=False,
        )
        arm_updates: dict[AblationArm, dict[str, object]] = {
            "cascade": {},
            "history_control": {
                "trajectory_mode": "learned",
                "control_source": "history_mean",
                "use_control_injection": True,
            },
            "raw_delta": {
                "trajectory_mode": "raw",
                "control_source": "velocity",
                "use_control_injection": True,
            },
            "learned_delta": {
                "trajectory_mode": "learned",
                "control_source": "velocity",
                "use_control_injection": True,
            },
            "qkv_conditioned": {
                "trajectory_mode": "learned",
                "control_source": "velocity",
                "use_control_injection": True,
                "condition_q": True,
                "condition_k": True,
                "condition_v": True,
            },
            "predictor_corrector": {
                "trajectory_mode": "learned",
                "control_source": "velocity",
                "use_control_injection": True,
                "use_predictor": True,
                "condition_q": True,
                "condition_k": True,
                "condition_v": True,
            },
            "full": {
                "trajectory_mode": "learned",
                "control_source": "velocity",
                "use_control_injection": True,
                "use_predictor": True,
                "condition_q": True,
                "condition_k": True,
                "condition_v": True,
                "use_importance_bias": True,
            },
        }
        if arm not in arm_updates:
            raise ValueError(f"unsupported ablation arm: {arm}")
        common.update(arm_updates[arm])
        common.update(overrides)
        return cls(**common)

    def with_updates(self, **updates: object) -> "TGSAConfig":
        """Return a validated copy with selected fields replaced."""

        return replace(self, **updates)
