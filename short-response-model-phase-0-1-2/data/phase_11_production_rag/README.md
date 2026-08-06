# Phase 11 production data boundary

Production knowledge, procedures, character lore, and user-memory exports should be supplied as versioned JSONL inputs to `build_knowledge_snapshot.py`. Do not commit private user-memory databases or generated SQLite snapshots.

Knowledge records use stable `document_id`, `text`, source/freshness metadata, and optional `claim_key`/`claim_value`. Procedure records use stable `procedure_id` and must come from trusted curated sources. Conflicting duplicate IDs fail snapshot creation.
