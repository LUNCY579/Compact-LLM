"""Phase 1: CPU-side structural audit of the four-checkpoint panel.

Reads every byte of ``Qwen3-0.6B/`` and ``Rineya_Published_Cores/*`` once, records
what is in them, and measures how far each published core sits from the base model
in weight space -- per tensor, per (layer, family), and per attention head.

Nothing here instantiates a model or touches the GPU: safetensors payloads are
memory-mapped and decoded with numpy, so the audit can run while another process
owns the card. SPEC.md §2 also forbids a module-scope torch import outside a short
whitelist that does not include this package; the only torch-adjacent imports are
lazy and guarded (``loading`` for the dtype policy, ``transformers`` for the
attention-implementation probe), and both degrade to a recorded "unavailable"
rather than failing the run.

Entry point::

    PYTHONPATH=src python -m rineya_psa.model_audit.run_audit

Python callers use ``from rineya_psa.model_audit.run_audit import run_audit``, or the
``run_model_audit`` alias re-exported here (the plain name is left bound to the
submodule so ``rineya_psa.model_audit.run_audit`` keeps meaning the module).

DEVIATION: SPEC.md §9 names ``statistics/provenance.py:stamp()`` as the source of
provenance stamps. That module does not exist yet and is owned by another phase, so
``run_audit`` carries an equivalent local stamp (git commit, config hash, seeds,
model SHA-256s, package versions) and imports the canonical ``stamp()`` opportunistically,
merging it in the moment it becomes available. ``stamp_source`` in every
``*.provenance.json`` records which one produced the file.

DEVIATION: the brief's family vocabulary has 13 entries and does not include
``lm_head``. Qwen3-0.6B and Nimbus materialise the tied ``lm_head.weight`` (311
tensors) while Halo and Supa-A1.1 omit it (310), so the audit adds ``lm_head`` as a
14th family rather than folding a duplicate of ``embed_tokens`` into the embedding
row or dropping it. ``classify_pair`` treats that presence difference as the
"tied lm_head convention" and does not let it demote a pair below
``architecture_compatible``.
"""

from __future__ import annotations

from rineya_psa.model_audit.distances import (
    ALL_FAMILIES,
    DEFAULT_REL_TOL,
    EXTRA_FAMILIES,
    FAMILIES,
    HEAD_KINDS,
    NO_LAYER,
    DeltaAccum,
    HeadLayout,
    MemmapPool,
    OrientationReport,
    PairDelta,
    TensorDelta,
    TensorKey,
    head_layout,
    pair_deltas,
    parse_tensor_name,
    resolve_orientation,
    tensor_delta,
)
from rineya_psa.model_audit.hashing import (
    CHUNK_BYTES,
    TensorLocation,
    canonical_json,
    collect_tensor_locations,
    decode_to_float64,
    directory_fingerprint,
    discover_model_files,
    hash_directory,
    hash_safetensors_file,
    json_diff_paths,
    normalize_tokenizer_config,
    read_safetensors_header,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_tensor,
    tensor_locations,
    tokenizer_identity,
    tokenizer_json_structure,
)
from rineya_psa.model_audit.inspect_ckpt import (
    AUDIT_SCHEMA_VERSION,
    DELTA_COMPARABLE_CLASSES,
    PAIR_CLASSES,
    CheckpointAudit,
    PairClassification,
    TensorRecord,
    audit_checkpoint,
    classify_pair,
    normalize_config,
)

# The runner is resolved lazily for two reasons: importing it eagerly would shadow
# ``rineya_psa.model_audit.run_audit`` (module) with ``run_audit`` (function), and it
# would put the module in ``sys.modules`` before ``python -m`` executes it, which
# makes runpy emit a RuntimeWarning on every CLI invocation.
_LAZY: dict[str, str] = {
    "main": "main",
    "pair_order": "pair_order",
    "provenance_path": "provenance_path",
    "run_model_audit": "run_audit",
}


def __getattr__(name: str) -> object:
    attr = _LAZY.get(name)
    if attr is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from rineya_psa.model_audit import run_audit as _runner

    return getattr(_runner, attr)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})

__all__ = [
    # lazily resolved (see __getattr__)
    "main",
    "pair_order",
    "provenance_path",
    "run_model_audit",
    "ALL_FAMILIES",
    "AUDIT_SCHEMA_VERSION",
    "CHUNK_BYTES",
    "DEFAULT_REL_TOL",
    "DELTA_COMPARABLE_CLASSES",
    "EXTRA_FAMILIES",
    "FAMILIES",
    "HEAD_KINDS",
    "NO_LAYER",
    "PAIR_CLASSES",
    "CheckpointAudit",
    "DeltaAccum",
    "HeadLayout",
    "MemmapPool",
    "OrientationReport",
    "PairClassification",
    "PairDelta",
    "TensorDelta",
    "TensorKey",
    "TensorLocation",
    "TensorRecord",
    "audit_checkpoint",
    "canonical_json",
    "classify_pair",
    "collect_tensor_locations",
    "decode_to_float64",
    "directory_fingerprint",
    "discover_model_files",
    "hash_directory",
    "hash_safetensors_file",
    "head_layout",
    "json_diff_paths",
    "normalize_config",
    "normalize_tokenizer_config",
    "pair_deltas",
    "parse_tensor_name",
    "read_safetensors_header",
    "resolve_orientation",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "sha256_tensor",
    "tensor_delta",
    "tensor_locations",
    "tokenizer_identity",
    "tokenizer_json_structure",
]
