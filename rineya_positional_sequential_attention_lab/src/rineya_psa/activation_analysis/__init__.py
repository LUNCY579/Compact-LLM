"""Phase 6 activation analysis: collection, probes, patching, head statistics.

Brief sections 16-18. The three questions this package answers, and the order they must
be read in:

``collect``  what the model computed  -- module forward hooks, float16 on CPU
``probes``   what is *decodable*      -- correlational; nominates regions
``patching`` what is *used*           -- causal; the only claim about mechanism here
``head_stats`` which heads look positional, with the causal evidence joined on

Importing this package pulls in torch (SPEC section 2 permits it under
``activation_analysis/``). Torch-free analysis should import ``rineya_psa.types``.
"""

from __future__ import annotations

from rineya_psa.activation_analysis.collect import (
    DEFAULT_POLICY,
    ROLES,
    SITES,
    ActivationCollector,
    ActivationDataset,
    ActivationRow,
    PositionPolicy,
    PositionSelection,
    collect_activations,
    collect_item,
    dataset_from_arrays,
    decoder_layers,
    estimate_bytes,
    forced_sdpa_backends,
    load_dataset_npz,
    select_positions,
    site_hooks,
)
from rineya_psa.activation_analysis.head_stats import (
    HEAD_METRICS,
    HeadStatRow,
    HeadStatsConfig,
    HeadStatsTable,
    KVHeadStatRow,
    capture_head_stats,
    head_stats_to_dataframe,
    write_head_stats,
)
from rineya_psa.activation_analysis.patching import (
    ALIGNMENTS,
    DEFAULT_ALIGNMENTS,
    DEFAULT_SITES,
    Alignment,
    PatchPair,
    PatchRecord,
    build_alignment,
    build_alignments,
    find_matched_pairs,
    gold_logprob,
    patch_cost_estimate,
    patch_records_to_dataframe,
    run_patch_pair,
    run_patch_pairs,
    site_label,
    write_patch_records,
)
from rineya_psa.activation_analysis.probes import (
    BACKEND,
    DEFAULT_TASKS,
    ProbeResult,
    ProbeTask,
    logistic_probe,
    probe_results_to_dataframe,
    ridge_probe,
    run_probe_suite,
    scenario_split,
)

__all__ = [
    "ALIGNMENTS",
    "BACKEND",
    "DEFAULT_ALIGNMENTS",
    "DEFAULT_POLICY",
    "DEFAULT_SITES",
    "DEFAULT_TASKS",
    "HEAD_METRICS",
    "ROLES",
    "SITES",
    "ActivationCollector",
    "ActivationDataset",
    "ActivationRow",
    "Alignment",
    "HeadStatRow",
    "HeadStatsConfig",
    "HeadStatsTable",
    "KVHeadStatRow",
    "PatchPair",
    "PatchRecord",
    "PositionPolicy",
    "PositionSelection",
    "ProbeResult",
    "ProbeTask",
    "build_alignment",
    "build_alignments",
    "capture_head_stats",
    "collect_activations",
    "collect_item",
    "dataset_from_arrays",
    "decoder_layers",
    "estimate_bytes",
    "find_matched_pairs",
    "forced_sdpa_backends",
    "gold_logprob",
    "head_stats_to_dataframe",
    "load_dataset_npz",
    "logistic_probe",
    "patch_cost_estimate",
    "patch_records_to_dataframe",
    "probe_results_to_dataframe",
    "ridge_probe",
    "run_patch_pair",
    "run_patch_pairs",
    "run_probe_suite",
    "scenario_split",
    "select_positions",
    "site_hooks",
    "site_label",
    "write_head_stats",
    "write_patch_records",
]
