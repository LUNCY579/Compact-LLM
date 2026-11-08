"""Interventions: PSA, SGSA, the cheap corrections, their controls, and their cost.

Brief sections 24-27 (residual scaling and stage schedules), 31-33 (SGSA, PSA, cheap
corrections), 36-37 (negative controls and compute accounting).

The four families in one sentence each:

``psa``
    A significance matrix injected into *ordinary parallel* attention through SPEC 5.3's
    three hook modes. The mandatory control for the sequential arm: if this alone
    recovers the effect, sequentiality contributed nothing.
``sgsa``
    ``X_{s+1} = X_s + alpha_s * G_s(X_s ; S[l,s,h,p])`` — a
    ``sequential_attention.SequentialArm`` with per-stage significance and a named
    residual schedule.
``cheap``
    Four reversible inference-time corrections that need no matrix at all: a positional
    logit bias, Q/K mean centering, a low-rank positional projection, and layerwise RoPE
    scaling schedules.
``controls`` / ``compute``
    The negative controls every claim above has to survive, and the FLOPs / wall-clock /
    VRAM accounting that stops "it helped" meaning "it was bigger".

Importing this package pulls in **no torch**: every runtime import lives inside a
function, per SPEC section 2. Building a factorial, a control set or a cost estimate
therefore runs on a torch-free interpreter; only entering a session needs the GPU stack.
"""

from __future__ import annotations

from rineya_psa.interventions.cheap import (
    CHEAP_LAMBDAS,
    LOW_RANK_RANKS,
    ROPE_SCHEDULES,
    PositionChannelMeans,
    estimate_position_channel_means,
    layerwise_rope_scaling,
    low_rank_positional_projection,
    position_logit_correction,
    position_mean_centering,
    positional_subspace,
    rope_scale_values,
    rope_scaling_schedule,
)
from rineya_psa.interventions.compute import (
    COMPUTE_METRICS_COLUMNS,
    COMPUTE_METRICS_PATH,
    ComputeConfig,
    ComputeMeasurement,
    ComputeProbe,
    FlopsBreakdown,
    arm_projection_counts,
    compute_metrics_row,
    estimate_flops,
    intervention_flops,
    measure_compute,
    psutil_available,
    write_compute_metrics,
)
from rineya_psa.interventions.controls import (
    CONTROL_KINDS,
    DEFAULT_CONTROL_SEED,
    DEFAULT_CONTROL_STAGES,
    MANDATORY_CONTROLS,
    Control,
    all_controls_for,
    assert_controls_complete,
    frozen_gates,
    random_gates,
    random_head_assignment,
    random_position_matrix,
    random_sequential,
    repeat_parallel_stage,
    reversed_functional_ordering,
    sign_flipped,
    uniform_middle_boost,
)
from rineya_psa.interventions.psa import (
    LAYER_BANDS,
    PSA_LAMBDA_GRID,
    PSA_MODES,
    PSAVariant,
    SignificanceTable,
    broadcast_over_heads,
    build_psa_context,
    canonical_mode,
    coerce_significance,
    head_gamma_from_matrix,
    layer_band,
    psa_lambda_sweep,
    psa_mode_sweep,
    psa_sweep,
    standardize_significance,
)
from rineya_psa.interventions.sgsa import (
    ALPHA_SCHEDULES,
    DEFAULT_STAGE_PROFILE,
    MATRIX_CONDITIONS,
    SGSA_LAMBDA_GRID,
    STAGE_PROFILES,
    SGSAConfig,
    alpha_schedule,
    build_sgsa,
    build_sgsa_arm,
    install_sgsa,
    sgsa_factorial,
    sgsa_session,
    sgsa_value_tables,
    significance_bin_order,
    significance_head_order,
    stage_gates,
)

__all__ = [
    "ALPHA_SCHEDULES",
    "CHEAP_LAMBDAS",
    "COMPUTE_METRICS_COLUMNS",
    "COMPUTE_METRICS_PATH",
    "CONTROL_KINDS",
    "DEFAULT_CONTROL_SEED",
    "DEFAULT_CONTROL_STAGES",
    "DEFAULT_STAGE_PROFILE",
    "LAYER_BANDS",
    "LOW_RANK_RANKS",
    "MANDATORY_CONTROLS",
    "MATRIX_CONDITIONS",
    "PSA_LAMBDA_GRID",
    "PSA_MODES",
    "ROPE_SCHEDULES",
    "SGSA_LAMBDA_GRID",
    "STAGE_PROFILES",
    "ComputeConfig",
    "ComputeMeasurement",
    "ComputeProbe",
    "Control",
    "FlopsBreakdown",
    "PSAVariant",
    "PositionChannelMeans",
    "SGSAConfig",
    "SignificanceTable",
    "all_controls_for",
    "alpha_schedule",
    "arm_projection_counts",
    "assert_controls_complete",
    "broadcast_over_heads",
    "build_psa_context",
    "build_sgsa",
    "build_sgsa_arm",
    "canonical_mode",
    "coerce_significance",
    "compute_metrics_row",
    "estimate_flops",
    "estimate_position_channel_means",
    "frozen_gates",
    "head_gamma_from_matrix",
    "install_sgsa",
    "intervention_flops",
    "layer_band",
    "layerwise_rope_scaling",
    "low_rank_positional_projection",
    "measure_compute",
    "position_logit_correction",
    "position_mean_centering",
    "positional_subspace",
    "psa_lambda_sweep",
    "psa_mode_sweep",
    "psa_sweep",
    "psutil_available",
    "random_gates",
    "random_head_assignment",
    "random_position_matrix",
    "random_sequential",
    "repeat_parallel_stage",
    "reversed_functional_ordering",
    "rope_scale_values",
    "rope_scaling_schedule",
    "sgsa_factorial",
    "sgsa_session",
    "sgsa_value_tables",
    "sign_flipped",
    "significance_bin_order",
    "significance_head_order",
    "stage_gates",
    "standardize_significance",
    "uniform_middle_boost",
    "write_compute_metrics",
]
