# Changelog

All notable changes to this project are documented here.

## 0.1.0 - 2026-08-17

### Added

- Standalone, provider-neutral Rineya Context Harness architecture and executable repository skeleton.
- Context-heavy/output-light workflow for direct replies and bounded grounded research.
- Versioned SQLite state store with WAL, foreign keys, ordered packaged migrations, optimistic concurrency, semantic idempotency hashes, append-only events, checkpoints, restore, and export.
- Explicit recursive-state-forgetting lifecycle: pin, update, merge, supersede, archive, evict, restore, and guarded administrative purge.
- Model-visible `delete_memory` with only reversible eviction/archive scopes; persistent purge is admin-only, archive-first, disabled by default, and rejected by the local user API.
- Replacement-before-delete and dependency-aware safety rules, including required resolved-evidence replacement before raw active evidence leaves context.
- Role-isolated context assembly that preserves `user`/`assistant` authority, protects untrusted-data delimiters during emergency compaction, and fails rather than silently truncating a mandatory user request.
- Immutable evidence ledger, content collision checks, source-diversified passage selection, claim proposals, deterministic answerability, and claim-linked citation verification.
- Bounded `search → fetch → extract → reconstruct → forget → verify → answer` orchestration with explicit budgets and termination states.
- Fixture, OpenAI-compatible, and Ollama model adapters; fixture and SearXNG search adapters; fixture and safe HTTP fetchers.
- SSRF-oriented URL/network controls, response bounds, prompt-injection marking, and read-only external capability profile.
- Six-tool small-model interface, strict generated JSON Schemas, and generated OpenAPI.
- JSONL tracing with optional payload redaction, Typer CLI, optional FastAPI boundary, Docker/local configuration, Windows/Linux bootstrap scripts, CI/release workflows, and governance/security documentation.
- Restaurant-route, Tower-of-Hanoi, and offline research examples.
- RSF ablation protocol and metrics for task success, context reduction, future-utility recall, deletion regret, invariant retention, and restore frequency.
- CPU-safe unit/integration, migration, schema, security, trace, API, packaging, and workflow tests with an 85% coverage gate.
