"""Figure builders for brief section 47, and the shared style they all obey.

``plots`` holds one function per figure; ``style`` holds the palette, the save policy
(200 dpi PNG plus vector PDF) and the no-truncated-axis guard that every value axis is
checked against before a file is written.

Importing this package imports matplotlib, which SPEC section 2 permits for ``figures/``
but which is not free -- analysis modules that only need numbers should not import it.
"""

from __future__ import annotations

from rineya_psa.figures.plots import (
    FIGURE_ORDER,
    FIGURES,
    absolute_vs_distance_effects,
    accuracy_vs_latency,
    accuracy_vs_position,
    architecture_comparison,
    edge_middle_gap_by_model,
    kv_head_position_heatmap,
    layer_kv_head_significance,
    layer_position_heatmap,
    low_rank_explained_variance,
    nll_vs_position,
    significance_sequentiality_interaction,
    weight_delta_by_layer,
    worst_vs_mean_accuracy,
)
from rineya_psa.figures.style import (
    DPI,
    MODEL_COLORS,
    PALETTE,
    TruncatedAxisError,
    apply_style,
    assert_no_truncated_axis,
    assign_colors,
    figure_style,
    save_figure,
    set_accuracy_ylim,
    set_zero_based_ylim,
    style_manifest,
)

__all__ = [
    "DPI",
    "FIGURES",
    "FIGURE_ORDER",
    "MODEL_COLORS",
    "PALETTE",
    "TruncatedAxisError",
    "absolute_vs_distance_effects",
    "accuracy_vs_latency",
    "accuracy_vs_position",
    "apply_style",
    "architecture_comparison",
    "assert_no_truncated_axis",
    "assign_colors",
    "edge_middle_gap_by_model",
    "figure_style",
    "kv_head_position_heatmap",
    "layer_kv_head_significance",
    "layer_position_heatmap",
    "low_rank_explained_variance",
    "nll_vs_position",
    "save_figure",
    "set_accuracy_ylim",
    "set_zero_based_ylim",
    "significance_sequentiality_interaction",
    "style_manifest",
    "weight_delta_by_layer",
    "worst_vs_mean_accuracy",
]
