# State and Memory Architecture

## Storage layers

| Layer | Purpose | Mutability |
|---|---|---|
| Run record | Goal, mode, status, revision | Versioned |
| Materialized memory | Current and inactive state records | Transactional |
| State events | Audit trail for transitions | Append-only |
| Evidence records | Exact retrieved passages and provenance | Immutable by convention |
| Claims | Atomic support relationships | Replaceable by ID |
| Checkpoints | Recoverable snapshot | Immutable |
| Trace JSONL | Connected execution narrative | Append-only |

## Dependency graph

A memory record may list other memory IDs as dependencies. An active record cannot depend on a missing or inactive record. This prevents deletion of a premise while a retained conclusion still requires it.

## Supersession

Supersession is stronger than ordinary eviction. It means a newer record is the authoritative current representation. A superseded record records `superseded_by` and leaves the active prompt.

## State hashes

Every committed patch returns a SHA-256 hash over the canonical full memory state. Hashes support:

- regression comparison;
- export integrity;
- checkpoint verification;
- idempotency debugging;
- reproducibility.
