# Phase 11 production data boundary

Production knowledge, procedures, character lore, and user-memory exports should be supplied as versioned JSONL inputs to `build_knowledge_snapshot.py`. Do not commit private user-memory databases or generated SQLite snapshots.

Knowledge records use stable `document_id`, `text`, source/freshness metadata, and optional `claim_key`/`claim_value`. Procedure records use stable `procedure_id` and must come from trusted curated sources. Conflicting duplicate IDs fail snapshot creation.


For Phase 9B-aware snapshots, records may also include:

```json
{
  "source_state": {"product_version": "0.9"},
  "validity_conditions": {"product_version": "0.9"},
  "memory_type": "factual"
}
```

`build_knowledge_snapshot.py` preserves these fields inside the existing metadata column. Retrieval still ranks relevance; the optional Phase 9B model pass judges current applicability.
