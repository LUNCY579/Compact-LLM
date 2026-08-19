# Tool Contracts

## Model-visible tools

The model receives exactly six high-level tools. A small, stable catalog is deliberate: preprocessing, network policy, state commits, and verification remain inside the harness.

### `get_state`

Reads compact versioned active state. No side effect.

### `search_web`

Searches one explicit information gap through the configured provider. The harness performs provider access, safe fetching, extraction, deduplication, ranking, and evidence persistence. Instructions inside pages are data, not policy.

### `inspect_evidence`

Returns immutable evidence records by stable evidence ID. No side effect.

### `update_memory`

Updates or deep-merges an existing record using an expected state revision and idempotency key.

A model may update only:

```text
summary
payload
source_ids
dependencies
valid_from
expires_at
confidence
```

A model may not change identity, kind, status, temperature, pin state, ownership, lifecycle metadata, or global invariants.

### `delete_memory`

Evicts or archives memory by stable ID. The model schema exposes only:

```text
active_context
archive
```

`persistent_purge` is intentionally absent from the model-visible schema. Purge is an administrative operation with separate policy, permission, archive-first, and confirmation controls; it is also rejected by the local user API.

Deletion proposals carry an expected revision, reason code, replacement IDs, and idempotency key. The harness—not the model—checks pins, dependencies, provenance, active-evidence replacement, and lifecycle legality before commit.

### `finish`

Proposes one of:

```text
answer
search_more
clarify
abstain
```

The workflow still verifies legality, evidence support, unresolved conflicts, freshness, and citation linkage.

## Tool design rules

- Inputs MUST validate against versioned JSON Schema.
- Tools MUST return bounded structured results with stable error codes.
- Mutating tools MUST carry an expected revision and semantic idempotency key.
- Retrying the same key with a different request body MUST fail rather than replay a mismatched result.
- Read-only and mutating tools MUST be distinguishable.
- The model may propose; code MUST validate and commit.
- No tool may silently broaden permissions, trust web instructions, or bypass answerability gates.
- Irreversible actions MUST remain outside the model-visible catalog unless a future version introduces an independently reviewed administrative boundary.
