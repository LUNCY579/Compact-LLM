"""Phase 2 - the controlled positional benchmark.

``filler`` makes inert prose, ``families`` makes the four question types, ``assemble``
places evidence and renders the four formats, ``tokenize`` measures where the evidence
actually landed in token space, and ``generate`` turns a config into the bank plus its
split manifest.

Nothing here imports torch at module scope; ``tokenize`` pulls the loader in lazily.
"""

from __future__ import annotations

from rineya_psa.benchmark.assemble import (
    SEPARATOR,
    SYSTEM_PROMPT,
    Layout,
    Placement,
    ScenarioBank,
    build_bank,
    build_user_prompt,
    fit_bank,
    measure_layout,
    place_evidence,
    render_chunk,
    render_context,
    render_variant,
    reorder_for_targets,
)
from rineya_psa.benchmark.families import (
    MULTI_HOP_POSITION_PAIRS,
    FamilyExample,
    build_exact_retrieval,
    build_family,
    build_multi_hop,
    build_semantic_single_hop,
    build_state_constraint,
    entity_tokens,
    multi_hop_targets,
)
from rineya_psa.benchmark.filler import (
    DOMAINS,
    Chunk,
    chunk_is_inert,
    filler_fingerprint,
    filler_vocabulary,
    generate_chunks,
    insert_sentence,
    reserved_tokens,
)
from rineya_psa.benchmark.generate import (
    BuildReport,
    build,
    load_config,
    position_sets_for_example,
)
from rineya_psa.benchmark.tokenize import (
    build_chat_prompt,
    count_prompt_tokens,
    count_tokens,
    get_tokenizer,
    offsets_available,
    tokenize_item,
)

__all__ = [
    "BuildReport",
    "Chunk",
    "DOMAINS",
    "FamilyExample",
    "Layout",
    "MULTI_HOP_POSITION_PAIRS",
    "Placement",
    "SEPARATOR",
    "SYSTEM_PROMPT",
    "ScenarioBank",
    "build",
    "build_bank",
    "build_chat_prompt",
    "build_exact_retrieval",
    "build_family",
    "build_multi_hop",
    "build_semantic_single_hop",
    "build_state_constraint",
    "build_user_prompt",
    "chunk_is_inert",
    "count_prompt_tokens",
    "count_tokens",
    "entity_tokens",
    "filler_fingerprint",
    "filler_vocabulary",
    "fit_bank",
    "generate_chunks",
    "get_tokenizer",
    "insert_sentence",
    "load_config",
    "measure_layout",
    "multi_hop_targets",
    "offsets_available",
    "place_evidence",
    "position_sets_for_example",
    "render_chunk",
    "render_context",
    "render_variant",
    "reorder_for_targets",
    "reserved_tokens",
    "tokenize_item",
]
