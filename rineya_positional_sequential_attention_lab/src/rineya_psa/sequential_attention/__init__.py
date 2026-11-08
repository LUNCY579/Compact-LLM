"""Sequential attention: brief sections 22 (arms), 23 (orderings) and 42 (stability).

Public surface::

    from rineya_psa.sequential_attention import (
        SequentialArm, all_arms, arm_grid, build_arm,
        SequentialQwen3Attention, install_sequential, sequential_session,
        HeadOrder, assign_roles, all_orders,
        StabilityConfig, profile_arm, run_arm_with_stability,
    )

Importing this package pulls in torch and transformers, which SPEC section 2 permits
under ``sequential_attention/``. ``ordering`` itself has no torch import — it is pure
bookkeeping over numbers another phase measured — but reaching it through this package
still executes these re-exports, so a genuinely torch-free caller must import
``rineya_psa.sequential_attention.ordering`` in an interpreter where the package's
``__init__`` has been kept out of the way.
"""

from __future__ import annotations

from rineya_psa.sequential_attention.arms import (
    ALPHA_MODES,
    ARM_NAMES,
    MANDATED_ALPHA_MODES,
    SequentialArm,
    all_arms,
    alphas_for,
    arm_four_stage,
    arm_fully_sequential,
    arm_grid,
    arm_parallel,
    arm_query_refinement,
    arm_two_stage,
    build_arm,
    partition_heads,
    stage_significance_from_head_scores,
)
from rineya_psa.sequential_attention.ordering import (
    DEFAULT_ROLE_THRESHOLDS,
    ROLE_SEQUENCE,
    HeadOrder,
    HeadStats,
    RoleAssignment,
    RoleThresholds,
    all_orders,
    assign_roles,
    head_scores_from_significance,
    original_order,
    random_orders,
    role_guided_order,
    role_reverse_order,
    significance_descending_order,
)
from rineya_psa.sequential_attention.stability import (
    DEFAULT_STABILITY_CONFIG,
    ActivationProfile,
    ArmRunResult,
    LayerStats,
    StabilityAbort,
    StabilityConfig,
    StabilityReport,
    compare_profiles,
    profile_arm,
    profile_forward,
    run_arm_with_stability,
)
from rineya_psa.sequential_attention.wrapper import (
    InstalledSequential,
    SequentialQwen3Attention,
    decoder_layers,
    install_sequential,
    resolve_layers,
    sequential_session,
)

__all__ = [
    "ALPHA_MODES",
    "ARM_NAMES",
    "DEFAULT_ROLE_THRESHOLDS",
    "DEFAULT_STABILITY_CONFIG",
    "MANDATED_ALPHA_MODES",
    "ROLE_SEQUENCE",
    "ActivationProfile",
    "ArmRunResult",
    "HeadOrder",
    "HeadStats",
    "InstalledSequential",
    "LayerStats",
    "RoleAssignment",
    "RoleThresholds",
    "SequentialArm",
    "SequentialQwen3Attention",
    "StabilityAbort",
    "StabilityConfig",
    "StabilityReport",
    "all_arms",
    "all_orders",
    "alphas_for",
    "arm_four_stage",
    "arm_fully_sequential",
    "arm_grid",
    "arm_parallel",
    "arm_query_refinement",
    "arm_two_stage",
    "assign_roles",
    "build_arm",
    "compare_profiles",
    "decoder_layers",
    "head_scores_from_significance",
    "install_sequential",
    "original_order",
    "partition_heads",
    "profile_arm",
    "profile_forward",
    "random_orders",
    "resolve_layers",
    "role_guided_order",
    "role_reverse_order",
    "run_arm_with_stability",
    "sequential_session",
    "significance_descending_order",
    "stage_significance_from_head_scores",
]
