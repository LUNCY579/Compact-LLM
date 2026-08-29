"""Phase 8 checkpoint arithmetic: blockwise deltas (section 34) and position-aware soups (35).

Two entry points matter and the rest is machinery around them:

- :func:`~rineya_psa.merging.arithmetic.apply_blockwise_delta` -- in-memory
  ``theta' = theta_base + alpha_block * Delta``, which is how every screening sweep runs.
  :func:`~rineya_psa.merging.arithmetic.materialise_merge` writes a variant to
  ``experiments/merges/<id>/`` and is only worth its 1.2 GB when another *process* has to
  load it.
- :func:`~rineya_psa.merging.soup.run_soup_pipeline` -- fingerprint, complementarity,
  global alpha, layer-wise alpha, greedy soup, with the acceptance rule as an inspectable
  object rather than an inline condition.

Neither module imports torch at module scope (SPEC section 2 does not whitelist
``merging/``); the tensor-touching functions import it lazily, and the on-disk path is
numpy-only by construction.
"""

from __future__ import annotations

from rineya_psa.merging.arithmetic import (
    ALPHA_GRID,
    GLOBAL_BLOCK,
    AppliedMerge,
    BlockAlphas,
    LayerBlock,
    MergeArtifact,
    apply_blockwise_delta,
    block_alphas_grid,
    default_blocks,
    deltas_from_states,
    materialise_merge,
    merge_workspace,
    single_block_sweep,
    snapshot_state,
    uniform_alpha_sweep,
)
from rineya_psa.merging.soup import (
    AcceptanceDecision,
    AcceptanceRule,
    AppliedSoup,
    ComplementarityReport,
    PositionalFingerprint,
    SoupCandidate,
    SoupObjective,
    SoupPipelineResult,
    SoupRecipe,
    SoupSearchResult,
    apply_soup_recipe,
    complementarity,
    complementarity_matrix,
    complementarity_table,
    default_acceptance_rule,
    fingerprint_from_curve,
    fingerprint_from_records,
    global_alpha_search,
    greedy_position_aware_soup,
    layerwise_alpha_search,
    run_soup_pipeline,
)

__all__ = [
    "ALPHA_GRID",
    "GLOBAL_BLOCK",
    "AcceptanceDecision",
    "AcceptanceRule",
    "AppliedMerge",
    "AppliedSoup",
    "BlockAlphas",
    "ComplementarityReport",
    "LayerBlock",
    "MergeArtifact",
    "PositionalFingerprint",
    "SoupCandidate",
    "SoupObjective",
    "SoupPipelineResult",
    "SoupRecipe",
    "SoupSearchResult",
    "apply_blockwise_delta",
    "apply_soup_recipe",
    "block_alphas_grid",
    "complementarity",
    "complementarity_matrix",
    "complementarity_table",
    "default_acceptance_rule",
    "default_blocks",
    "deltas_from_states",
    "fingerprint_from_curve",
    "fingerprint_from_records",
    "global_alpha_search",
    "greedy_position_aware_soup",
    "layerwise_alpha_search",
    "materialise_merge",
    "merge_workspace",
    "run_soup_pipeline",
    "single_block_sweep",
    "snapshot_state",
    "uniform_alpha_sweep",
]
