# Rineya Context Harness v0.1.0

A **small-model-first**, local-first runtime for bounded web retrieval, evidence-grounded answers, and **recursive-state forgetting (RSF)**.

The harness is designed around one central rule:

> Preserve the smallest state sufficient for the next correct decision; keep global invariants, unresolved dependencies, and immutable provenance outside the model-visible working set.

Rineya may propose semantic changes, but ordinary software owns permissions, budgets, state revisions, deletion safety, evidence persistence, verification, and termination.

## What this release contains

- A versioned SQLite state engine with WAL, ordered packaged migrations, optimistic concurrency, semantic idempotency hashes, append-only events, and checkpoints.
- Explicit `delete_memory` semantics:
  - `active_context`: reversible eviction from model-visible state;
  - `archive`: reversible inactive retention;
  - `persistent_purge`: admin-only, separately gated, disabled by default, archive-first, and intentionally absent from the model-visible tool schema.
- Atomic replacement-before-deletion transactions.
- Pinned global invariants and dependency-aware deletion checks.
- Immutable evidence records separated from mutable active memory, with exact collision checks and claim-linked provenance.
- A bounded and source-diversified `search → fetch → extract → reconstruct → forget → verify → answer` workflow.
- Local OpenAI-compatible and Ollama model adapters, with Qwen thinking disabled by default.
- Fixture search/fetch/model components for completely offline tests and demos.
- SSRF-oriented outbound URL checks and prompt-injection marking for untrusted web text.
- Context rebuilding after state changes, so logical deletion becomes physical prompt omission. Recent conversation retains `user`/`assistant` roles and is never promoted into the system-policy channel.
- Claim-linked citation verification: a grounded final answer must cite evidence that supports a required claim, not merely any retrieved source.
- JSONL traces with optional payload redaction, checkpoints, rollback, run export, CLI, and optional FastAPI API.
- A CPU-safe test suite and an RSF evaluation module.

## Status

This is a **research-grade architecture specification and executable repository skeleton**, not a claim of production safety or benchmark superiority. The deterministic state engine, deletion guards, offline workflow, and tests are implemented. Learned RSF policies, Rineya checkpoint evaluation, browser-grade rendering, exact tokenizer integration, and runtime-specific KV-cache invalidation remain explicit follow-on work; see [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md).

## Architecture at a glance

```text
User request
    │
    ▼
Deterministic mode gate
    ├── DIRECT ─────────────────────────────────────────────┐
    ├── CLARIFY                                             │
    └── SEARCH                                              │
          │                                                 │
          ▼                                                 │
  Search provider → safe fetch → readable extraction        │
          │                                                 │
          ▼                                                 │
  Immutable evidence ledger                                 │
          │                                                 │
          ▼                                                 │
  Rineya proposes reconstruction and deletion               │
          │                                                 │
          ▼                                                 │
  Deterministic validator                                   │
  - revision / idempotency                                  │
  - pins / dependencies                                     │
  - source provenance                                       │
  - budgets / permissions                                   │
          │                                                 │
          ▼                                                 │
  Atomic state commit: ADD + UPDATE + EVICT                 │
          │                                                 │
          ▼                                                 │
  Fresh context assembled from ACTIVE state only            │
          │                                                 │
          ▼                                                 │
  Code-driven answerability gate                            │
          ├── SEARCH_MORE                                    │
          ├── ABSTAIN                                        │
          └── ANSWER ────────────────────────────────────────┘
```

The complete normative design is in [`ARCHITECTURE_SPECIFICATION.md`](ARCHITECTURE_SPECIFICATION.md).

## Quick start

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[api,dev]"

rineya-harness doctor --config .\configs\default.yaml
rineya-harness demo restaurant
rineya-harness demo hanoi
rineya-harness demo research
pytest
```

Or run:

```powershell
.\scripts\bootstrap.ps1
.\scripts\check.ps1
```

### Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[api,dev]'

rineya-harness doctor --config configs/default.yaml
rineya-harness demo research
pytest
```

## Connect a local Rineya checkpoint

Copy the example configuration and point it at an OpenAI-compatible local endpoint such as a llama.cpp server:

```powershell
Copy-Item .\configs\local-rineya.example.yaml .\configs\local-rineya.yaml
rineya-harness run "Search for the latest supported answer." `
  --mode grounded `
  --config .\configs\local-rineya.yaml
```

Key defaults:

```yaml
model:
  provider: openai_compatible
  model: rineya-1-454m
  base_url: http://127.0.0.1:8080/v1
  temperature: 0.0
  enable_thinking: false
```

The harness does not assume that every OpenAI-compatible server supports strict JSON schema. The model-facing policy is designed for strict JSON; deployment qualification must test the exact runtime/model pair and add a controlled repair/fallback path where required.

## Explicit deletion example

```powershell
rineya-harness delete-memory RUN_ID route.checkpoint.A `
  --scope active_context `
  --reason-code completed_no_future_dependency `
  --expected-revision 7 `
  --replacement-state-id route.remaining_checkpoints `
  --replacement-state-id route.next_checkpoint
```

Default deletion is reversible. The model-visible `delete_memory` tool exposes only reversible eviction and archive. Permanent purge is available only through an approved administrative code path and is disabled unless all of the following are true:

1. repository policy enables purge;
2. the caller explicitly requests `persistent_purge`;
3. archive-first requirements are satisfied;
4. any configured confirmation token matches;
5. the target is not pinned and no active dependency requires it.

## CLI surface

```text
rineya-harness doctor
rineya-harness run
rineya-harness state
rineya-harness delete-memory
rineya-harness restore-memory
rineya-harness checkpoint
rineya-harness restore-checkpoint
rineya-harness export-run
rineya-harness trace
rineya-harness evaluate
rineya-harness demo
rineya-harness serve
```

## API surface

With the `api` extra installed:

```powershell
rineya-harness serve --config .\configs\default.yaml
```

Default binding is `127.0.0.1:8765`. Main routes:

```text
GET  /healthz
GET  /v1/tools
GET  /v1/runs
POST /v1/runs
GET  /v1/runs/{run_id}
GET  /v1/runs/{run_id}/memory
POST /v1/runs/{run_id}/memory/delete
POST /v1/runs/{run_id}/memory/restore
GET  /v1/runs/{run_id}/checkpoints
POST /v1/runs/{run_id}/checkpoints
POST /v1/runs/{run_id}/checkpoints/{checkpoint_id}/restore
GET  /v1/runs/{run_id}/trace
GET  /v1/runs/{run_id}/export
```

The local user API rejects persistent purge. A public or multi-user deployment still requires authentication, TLS, request limits, tenancy isolation, and a dedicated admin boundary.

## Tests and validation

```bash
pytest
pytest --cov=rineya_harness --cov-report=term-missing
python scripts/export_schemas.py --check
python scripts/validate_repository.py --with-tests
```

The packaged release includes generated JSON Schemas/OpenAPI, an offline validation report, and a SHA-256 repository manifest under `schemas/generated/` and `artifacts/release/`. Release packaging excludes runtime databases, local secrets/configuration, caches, coverage data, prior archives, and symlinks.

## Repository map

```text
src/rineya_harness/
  api/              optional HTTP boundary
  context/          context budgets, assembly, physical prompt compaction
  domain/           strict versioned contracts
  evidence/         passage ranking, immutable ledger, answerability
  evals/            RSF metrics and evaluation runner
  memory/           explicit update/delete/restore services
  models/           local model adapters
  orchestration/    bounded workflow and model-facing policy
  search/           provider abstraction and SearXNG adapter
  security/         untrusted-data and outbound-network policy
  state/            SQLite store, reducer, invariants, checkpoints
  tools/            six-tool small-model interface
  tracing/          connected JSONL traces
  web/              safe fetch and readable-text extraction
```

## Design boundaries

Version 0.1 intentionally does **not** include autonomous browser clicking, shell execution, email/calendar actions, purchases, persistent user profiling, unrestricted MCP discovery, self-modifying prompts, or multi-agent delegation. The first public harness should remain read-only and auditable:

```text
Search → Fetch → Extract → Reconstruct → Forget → Verify → Answer
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
