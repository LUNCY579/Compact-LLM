"""Selection metrics and cluster-aware statistical summaries."""

from .adaptive_budget_screen import (
    BoundaryBudgetScreen,
    screen_adaptive_budget,
    selected_resident_positions,
    sink_recent_protection,
    state_from_count_boundary,
)
from .attention_prior_quality import (
    ATTENTION_PRIOR_QUALITY_CONDITION_IDS,
    AttentionPriorQualityCondition,
    expected_cell_order,
    recover_prediction_prefix,
    retained_set_diagnostics,
    specificity_decision,
    validate_attention_prior_conditions,
)
from .boundary_evidence import BoundaryEvidenceScore, score_observed_boundary_evidence
from .clustered import (
    ClusteredPairedResult,
    cluster_bootstrap_interval,
    cluster_differences,
    clustered_paired_analysis,
    exact_or_sampled_sign_flip,
    normal_approximation_mde,
    validate_cluster_splits,
)
from .evidence_endpoints import (
    EVIDENCE_RETENTION_ENDPOINT_VERSION,
    PRIMARY_ENDPOINT_NAME,
    EvidenceEndpointError,
    EvidenceRetentionEndpointScore,
    FactTokenRecall,
    LayerFactRecall,
    score_evidence_retention_endpoint,
    score_layer_fact_retention,
    score_serialized_evidence_retention,
)
from .failure_taxonomy import compressed_retention_failure_labels
from .metrics import evidence_recall, selection_metrics
from .statistics import exact_mcnemar, paired_bootstrap_difference, summarize_paired_binary

__all__ = [
    "ATTENTION_PRIOR_QUALITY_CONDITION_IDS",
    "EVIDENCE_RETENTION_ENDPOINT_VERSION",
    "PRIMARY_ENDPOINT_NAME",
    "AttentionPriorQualityCondition",
    "BoundaryBudgetScreen",
    "BoundaryEvidenceScore",
    "ClusteredPairedResult",
    "EvidenceEndpointError",
    "EvidenceRetentionEndpointScore",
    "FactTokenRecall",
    "LayerFactRecall",
    "cluster_bootstrap_interval",
    "cluster_differences",
    "clustered_paired_analysis",
    "compressed_retention_failure_labels",
    "evidence_recall",
    "exact_mcnemar",
    "exact_or_sampled_sign_flip",
    "expected_cell_order",
    "normal_approximation_mde",
    "paired_bootstrap_difference",
    "recover_prediction_prefix",
    "retained_set_diagnostics",
    "score_evidence_retention_endpoint",
    "score_layer_fact_retention",
    "score_observed_boundary_evidence",
    "score_serialized_evidence_retention",
    "screen_adaptive_budget",
    "selected_resident_positions",
    "selection_metrics",
    "sink_recent_protection",
    "specificity_decision",
    "state_from_count_boundary",
    "summarize_paired_binary",
    "validate_attention_prior_conditions",
    "validate_cluster_splits",
]
