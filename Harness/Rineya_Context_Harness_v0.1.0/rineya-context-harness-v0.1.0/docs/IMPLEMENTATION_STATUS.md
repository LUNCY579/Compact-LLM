# Implementation Status — v0.1.0

## Implemented and executable

### Contracts and small-model boundary

- Strict Pydantic v2 contracts for runs, conversation turns, memory, evidence, claims, patches, tools, contexts, and answers.
- A six-tool model surface: `get_state`, `search_web`, `inspect_evidence`, `update_memory`, `delete_memory`, and `finish`.
- Model actors cannot create pinned/global-invariant memory, pin or unpin records, purge records, change protected lifecycle fields, or mutate a global invariant.
- Recent conversation remains in its original `user`/`assistant` role. It is never serialized as trusted system policy.
- Hidden chain-of-thought is neither requested nor persisted.

### State, memory, and deletion

- SQLite state store with WAL, foreign keys, ordered packaged migrations, transactions, optimistic revisions, semantic idempotency hashes, events, and checkpoints.
- Memory reducer for `ADD`, `UPDATE`, `MERGE`, `SUPERSEDE`, `ARCHIVE`, `EVICT`, `RESTORE`, `PURGE`, `PIN`, and `UNPIN`.
- Deterministic invariant validation for pin safety, dependency integrity, evidence provenance, active-item limits, actor permissions, lifecycle legality, and purge policy.
- Explicit reversible `delete_memory` tool. The model-visible schema exposes only active-context eviction and archive.
- Persistent purge is admin-only, archive-first, disabled by default, and rejected by the local user API.
- Replacement-before-delete enforcement for active evidence: a compact active `resolved_evidence` replacement must exist before raw active evidence may leave context.
- Restore is limited to archived/evicted records; active or superseded records cannot be silently “restored.”
- Context rebuilding after state commits, turning logical deletion into actual prompt omission.

### Retrieval, evidence, and grounded answering

- Immutable evidence ledger with content hashes and collision checks, separated from mutable active memory.
- Bounded search/fetch/extract pipeline with duplicate suppression and per-source passage caps to reduce single-source flooding.
- SearXNG JSON adapter plus fixture search/fetch implementations for offline execution.
- Safe-fetch controls for schemes, hosts, ports, redirects, response bytes, content types, timeouts, and private-network targets.
- Prompt-injection scanning and explicit untrusted-evidence delimiters, including emergency context compaction.
- Deterministic answerability gate and claim-linked citation verification. A grounded answer must cite evidence supporting a required claim rather than any unrelated retrieved record.

### Runtime, operations, and assurance

- Bounded search/reconstruction/forgetting/verification workflow with explicit counters and termination paths.
- Local OpenAI-compatible and Ollama adapters, with Qwen thinking disabled by default.
- Fixture policy/model/search/fetch components for deterministic offline tests and demos.
- JSONL traces with configurable payload redaction, run export, checkpoints, and rollback.
- Typer CLI and optional FastAPI API for health, tools, run lifecycle, memory, checkpoints, trace, and export.
- Restaurant, Hanoi, and offline research demos.
- CPU-safe unit/integration tests, migration tests, packaging tests, and RSF metrics.
- Generated JSON Schemas and OpenAPI with schema-drift validation.
- Deterministic release packaging order/timestamps and exclusion of local secrets, runtime databases, caches, coverage files, prior archives, and symlinks.

## Deliberately not claimed complete

- No trained Rineya RSF policy is included.
- No empirical claim is made that RSF improves Rineya yet.
- The deterministic fixture policy is a test oracle, not a capable semantic agent.
- Browser JavaScript rendering is not implemented.
- Search source authority, temporal freshness, and independent corroboration remain minimal configurable gates.
- Exact Qwen tokenization is not wired into the default estimator.
- KV-cache invalidation is represented as a mandatory dynamic-context rebuild, but runtime-specific cache APIs are not controlled directly.
- OpenAI-compatible structured-output support varies by local server; controlled repair/fallback qualification remains necessary.
- API authentication, TLS, rate limiting, and multi-user tenancy are not part of the local development profile.
- Persistent personal memory is not enabled.
- Permanent purge remains disabled by default.
- No live tool with external side effects exists.
- The offline validator does not exercise a public Rineya checkpoint or an external search service.

## Qualification required before a public harness release

1. Pin the exact Rineya checkpoint, tokenizer, runtime, quantization, and chat template.
2. Run structured-output/tool-call compliance tests against that exact runtime.
3. Run frozen full-history, sliding-window, one-shot-summary, structured-state, and RSF baselines.
4. Measure deletion regret, future-utility recall, citation coverage, unsupported-answer rate, context tokens, latency, and recovery behavior.
5. Test Windows 11 startup, cancellation, crash recovery, migrations, backup/restore, and path handling.
6. Test prompt injection, DNS rebinding/SSRF behavior, source flooding, and oversized responses in an isolated network profile.
7. Human-review safety, empathy, communication naturalness, and abstention behavior.
8. Freeze one candidate before sealed evaluation and publish limitations beside every performance claim.
