# Rineya Context Harness v0.1 Architecture Specification

**Version:** 0.1.0  
**Status:** Research architecture and executable skeleton  
**Primary target:** Rineya-family small language models, initially approximately 344M–454M parameters  
**Deployment posture:** Local-first, read-only external retrieval, bounded workflows  
**Normative language:** **MUST**, **SHOULD**, and **MAY** have their ordinary RFC-style meanings.

---

## 1. Purpose

Rineya is a context-heavy, output-light model family: the runtime may need to inspect substantial context while the final answer is often a few words or one compact sentence. The harness exists to prevent the model’s limited context window from becoming a continually growing transcript database.

The architecture turns a model into a controlled system that can:

1. retrieve current information;
2. preserve immutable source evidence;
3. reconstruct the current task state;
4. explicitly remove obsolete model-visible memory;
5. verify that required facts and invariants survived;
6. rebuild a compact prompt;
7. answer, clarify, search again, or abstain.

The core hypothesis is testable:

> A small language model can complete longer, more dynamic tasks when the harness continually replaces historical trajectories with a minimal sufficient current state, while externally preserving provenance and recoverability.

## 2. Scope and non-goals

### 2.1 Version 0.1 scope

The first version MUST provide:

- one bounded workflow rather than unrestricted autonomy;
- a small semantic tool vocabulary;
- explicit structured state;
- immutable evidence records;
- reversible active-context deletion;
- deterministic deletion authorization;
- context and workflow budgets;
- local persistence and checkpoints;
- traceable state transitions;
- answerability verification;
- offline fixtures and tests.

### 2.2 Non-goals

Version 0.1 is not:

- a universal autonomous agent;
- an operating-system control layer;
- a browser-clicking robot;
- a multi-agent system;
- a replacement for a full-text archive;
- a claim that the learned RSF policy already outperforms baselines;
- a mechanism for deleting model weights or internal neural representations;
- a guarantee that logical deletion can mutate a transformer’s already-computed KV cache in place.

## 3. Design principles

### 3.1 Deterministic shell around a probabilistic core

The model MAY interpret evidence, propose a query, propose compact state, and generate language. Code MUST own:

- permissions;
- budgets;
- state revisions;
- idempotency;
- schema validation;
- legal workflow transitions;
- dependency checks;
- pin protection;
- evidence provenance;
- purge policy;
- answerability gates;
- completion status.

### 3.2 Current state is not conversation history

Natural-language history MUST NOT be the only state database. A task run uses explicit records for goals, current facts, open gaps, evidence, conflicts, and completed checkpoints.

### 3.3 Forget model-visible state; preserve recoverable evidence

The harness distinguishes:

- **active context**: what Rineya can see now;
- **state store**: versioned current and inactive records;
- **evidence ledger**: immutable attributable source passages;
- **trace**: what happened during execution;
- **checkpoint**: a recoverable snapshot.

An item can leave active context without being destroyed.

### 3.4 Replacement before deletion

Any state transition that supersedes a current fact MUST construct and validate its replacement before the old fact becomes inactive. The commit SHOULD be atomic.

### 3.5 Small tool surface

The model-facing tool set SHOULD remain approximately six high-level operations:

```text
get_state
search_web
inspect_evidence
update_memory
delete_memory
finish
```

Internal components MAY be more numerous, but tool granularity MUST not force a small model to reason about HTML parsers, redirects, chunkers, indexes, or transaction mechanics.

### 3.6 Evidence, not confidence, determines completion

A model saying “I am confident” is not proof. The harness MUST evaluate whether required claims have valid evidence links, whether critical conflicts remain, whether freshness requirements are satisfied, and whether open information gaps remain.

## 4. System context

```text
┌────────────────────────────────────────────────────────────────────┐
│                         RINEYA CONTEXT HARNESS                     │
│                                                                    │
│  Request / policy boundary                                         │
│          │                                                         │
│          ▼                                                         │
│  Deterministic route + budget controller                           │
│          │                                                         │
│          ├──────── DIRECT ────────────────────────────────┐         │
│          │                                                │         │
│          ▼                                                │         │
│  Query planning → search → safe fetch → extraction        │         │
│          │                                                │         │
│          ▼                                                │         │
│  Immutable evidence ledger                                │         │
│          │                                                │         │
│          ▼                                                │         │
│  Reconstruction proposal (Rineya)                         │         │
│          │                                                │         │
│          ▼                                                │         │
│  Invariant/dependency/provenance validator                │         │
│          │                                                │         │
│          ▼                                                │         │
│  Transactional state reducer                              │         │
│  ADD / UPDATE / MERGE / SUPERSEDE / ARCHIVE / EVICT      │         │
│          │                                                │         │
│          ▼                                                │         │
│  Fresh context assembly from ACTIVE state only            │         │
│          │                                                │         │
│          ▼                                                │         │
│  Answerability gate → answer / search more / abstain ◀────┘         │
│                                                                    │
│  Surrounding services: SQLite, checkpoints, JSONL traces, evals    │
└────────────────────────────────────────────────────────────────────┘
```

## 5. Trust boundaries

### 5.1 Trusted control plane

The following are trusted configuration or code:

- static system policy;
- state schemas;
- memory retention policy;
- workflow graph;
- tool registry;
- URL policy;
- answerability rules;
- token and call budgets;
- purge policy.

### 5.2 Untrusted data plane

The following MUST be treated as untrusted data:

- search snippets;
- fetched webpages;
- retrieved document text;
- user-provided URLs and recent user/assistant messages;
- model-proposed state patches;
- model-proposed citations;
- imported memory records.

Untrusted text MUST be clearly delimited before it enters a model prompt. Instructions found inside evidence MUST NOT change tool permissions or workflow policy. Recent conversation MUST preserve its original `user`/`assistant` role and MUST NOT be promoted into a trusted `system` message during compaction.

## 6. Memory model

### 6.1 Memory kinds

| Kind | Meaning | Default active behavior |
|---|---|---|
| `global_invariant` | Goal, rules, immutable constraints | Pinned |
| `current_state` | What is currently true | Replace or update |
| `open_goal` | Unresolved subgoal or information gap | Retain until resolved |
| `active_evidence` | Raw model-visible source passage | Temporary |
| `resolved_evidence` | Compact claim with provenance | Retain while useful |
| `transient_observation` | Intermediate result | Evict after causal use |
| `completed_checkpoint` | Finished route/task checkpoint | Archive or evict |
| `superseded_state` | Older version of current state | Inactive |
| `conflict` | Unresolved contradiction | Retain |
| `user_preference` | Explicit user preference | Scoped and correctable |
| `event_history` | Historical event | Usually external/inactive |

### 6.2 Temperatures

```text
PINNED → HOT → WARM → COLD
```

- `PINNED`: mandatory active invariant;
- `HOT`: required immediately;
- `WARM`: likely future utility;
- `COLD`: inactive or low-priority but recoverable.

`PURGED` is a status, not a temperature.

### 6.3 Statuses

```text
ACTIVE
ARCHIVED
EVICTED
SUPERSEDED
PURGED
```

Only `ACTIVE` records enter ordinary context assembly.

### 6.4 Stable identity

Every memory record MUST have a stable identifier. Deletion by free-form description is forbidden. IDs enable:

- deterministic targeting;
- dependency checks;
- audit and rollback;
- idempotent retries;
- precise evaluation.

## 7. Explicit deletion contract

### 7.1 Tool name

```text
delete_memory
```

### 7.2 Request

```json
{
  "run_id": "run_...",
  "memory_ids": ["route.checkpoint.A"],
  "scope": "active_context",
  "reason_code": "completed_no_future_dependency",
  "reason": "Light A has been passed.",
  "replacement_state_ids": [
    "route.remaining_checkpoints",
    "route.next_checkpoint"
  ],
  "expected_state_revision": 7,
  "idempotency_key": "delete-route-A-r7"
}
```

### 7.3 Scopes

| Scope | Effect | Reversible | Default permission |
|---|---|---:|---|
| `active_context` | Status becomes `EVICTED`; excluded from next prompt | Yes | Model may propose |
| `archive` | Status becomes `ARCHIVED`; retained for inspection | Yes | Model/user may propose |
| `persistent_purge` | Content replaced with purge tombstone | No/limited | Admin-only policy path; absent from model tool schema |

### 7.4 Preconditions

Deletion MUST fail if any of the following is true:

- target ID is unknown;
- target is pinned;
- an active record depends on the target;
- replacement IDs are missing, inactive, or themselves deleted;
- active evidence is removed without an active `resolved_evidence` replacement;
- the expected revision is stale;
- purge is disabled;
- archive-first policy is unmet;
- a required purge confirmation token is invalid.

### 7.5 Postconditions

A successful deletion MUST return:

- new revision;
- changed IDs;
- active-context removals;
- archive/purge IDs;
- state hash;
- event ID;
- `prompt_rebuild_required=true`.

## 8. Logical versus physical deletion

A model cannot generally erase tokens already encoded in a current transformer KV cache. Therefore deletion has two stages:

1. **Logical commit:** state status changes to inactive.
2. **Physical omission:** the next model call is built from the new active state and does not include the old record.

Runtime adapters MAY preserve a stable static-prefix cache. They MUST invalidate or rebuild dynamic prompt/KV state when active memory changes.

## 9. Transaction semantics

State updates use optimistic concurrency:

```text
read revision r
construct candidate state
validate candidate state
commit only if current revision == r
increment to r + 1
```

Every patch MUST include an idempotency key. Repeating the same key for the same run and the same canonical request MUST return the original result without applying the mutation twice. Reusing a key with different canonical request content MUST fail as an idempotency conflict.

A typical RSF transaction is:

```text
ADD new current state
ADD compact supported claim
SUPERSEDE old current state
EVICT completed checkpoint
EVICT raw evidence reduced to claim
COMMIT atomically
```

## 10. Restaurant example

### 10.1 Before

```json
{
  "global": {
    "goal": "Reach Restaurant R",
    "mode": "walking"
  },
  "current": {
    "eta_minutes": 20,
    "remaining_lights": ["A", "B", "C"],
    "next_light": "A"
  }
}
```

### 10.2 Observation

```json
{
  "elapsed_minutes": 5,
  "passed_light": "A"
}
```

### 10.3 Transition

```text
ETA 20 → 15
remaining [A,B,C] → [B,C]
next A → B
archive/evict A
```

### 10.4 After

```json
{
  "global": {
    "goal": "Reach Restaurant R",
    "mode": "walking"
  },
  "current": {
    "eta_minutes": 15,
    "remaining_lights": ["B", "C"],
    "next_light": "B"
  }
}
```

The active prompt no longer needs the narrative of approaching and crossing A. The external event/archive remains available for audit or route reversal.

## 11. Tower of Hanoi example

Pinned state:

```text
disk count
disk ordering rule
peg names
target peg
```

Mutable sufficient state:

```text
A=[3,2,1], B=[], C=[]
```

After moving disk 1 from A to C:

```text
A=[3,2], B=[], C=[1]
```

The previous board is superseded. Full move history is unnecessary unless a future policy depends on it. A compact external visited-state hash MAY be retained to prevent cycles without replaying the whole trajectory.

## 12. Workflow graph

Legal high-level transitions:

```text
INTAKE → ROUTE
ROUTE → DIRECT | CLARIFY | SEARCH | ABSTAIN
SEARCH → FETCH → EXTRACT → RECONSTRUCT
RECONSTRUCT → COMPACT → VERIFY
VERIFY → SEARCH | ANSWER | ABSTAIN
ANSWER | CLARIFY | ABSTAIN → DONE
ANY → ERROR
```

The graph is bounded by:

- maximum rounds;
- maximum model calls;
- maximum search calls;
- maximum fetched sources;
- maximum active records;
- maximum wall-clock time;
- context token budget.

## 13. Search and evidence pipeline

### 13.1 Search provider

The built-in live provider is a SearXNG JSON adapter. Provider APIs are abstracted behind `SearchProvider`.

### 13.2 Safe fetch

The fetcher applies:

- `http`/`https` scheme restriction;
- localhost and private-address blocking by default;
- port allowlisting;
- bounded redirects;
- bounded response bytes;
- content-type allowlisting;
- timeouts;
- readable-text extraction;
- prompt-injection heuristics.

Application-level SSRF checks are defense in depth. A public deployment SHOULD also use isolated, credential-free network egress and DNS/connect pinning.

### 13.3 Immutable evidence

Each evidence record includes:

- source URI;
- title;
- retrieval query;
- exact passage;
- content hash;
- retrieval time;
- passage offsets/rank;
- injection flags;
- source document hash.

Evidence records are not deleted merely because their active-memory projections are evicted.

## 14. Reconstruction policy

The model-facing reconstruction schema permits:

- atomic claim proposals;
- compact memory additions;
- memory update proposals;
- deletion proposals;
- unresolved information gaps.

The policy MUST NOT directly commit state. The harness validates:

- evidence IDs exist;
- supported claims cite evidence;
- dependency IDs exist;
- update targets exist;
- pin protection is not weakened;
- model updates touch only semantic/provenance fields;
- a model does not create or modify a global invariant;
- a model does not pin, unpin, purge, or directly alter lifecycle status;
- deletion targets exist and are not pinned;
- active evidence has a compact, active, source-linked replacement before removal.

## 15. Context assembly

The assembler builds bounded sections:

```text
static policy
goals and invariants
current state
evidence ledger
recent conversation
current user message
output reserve
```

Priority is determined by memory temperature, kind, and recency. Untrusted evidence is wrapped in explicit data delimiters, and emergency compaction MUST preserve those delimiters. Recent conversation remains role-separated `user`/`assistant` content rather than trusted system text. If the current user message plus mandatory policy/output reserve cannot fit, assembly fails explicitly instead of silently truncating the request. The exact Rineya tokenizer SHOULD replace the conservative estimator for release measurements.

## 16. Answerability gate

A grounded answer is permitted only when:

- at least one required claim exists;
- every required claim is supported;
- each supported claim maps to existing evidence IDs;
- no critical conflict remains;
- freshness requirements pass;
- no required information gap remains;
- configured source-diversity conditions pass;
- each final citation used for a grounded answer maps to evidence supporting at least one required claim.

Otherwise the gate returns:

```text
SEARCH_MORE
ABSTAIN
```

A model cannot override this gate with confidence language.

## 17. Small-model-first protocol

### 17.1 Structured output

Model-controlled workflow stages use strict JSON schemas. Hidden chain-of-thought is not requested or stored. A deployment runtime SHOULD support constrained decoding or grammar/schema enforcement.

### 17.2 One unified model

Version 0.1 uses one configured Rineya checkpoint for query planning, reconstruction, and answering. Separate prompts are workflow roles, not independent agents.

### 17.3 Backend abstraction

Supported adapters:

- deterministic fixture;
- OpenAI-compatible local chat endpoint;
- Ollama chat endpoint.

Future adapters MAY add llama.cpp native bindings, Transformers, ONNX, or other runtimes without changing the workflow contracts.

## 18. Persistence and recovery

SQLite is the default local state store because it provides:

- transactions;
- WAL mode;
- foreign keys;
- simple backup/export;
- Windows compatibility;
- no external service requirement.

The store applies ordered packaged SQL migrations before serving a run. Repository and packaged migration copies are release-validated byte-for-byte; released migrations are forward-only artifacts.

The store maintains:

- materialized current memory;
- append-only state events;
- immutable evidence;
- claims;
- checkpoints;
- idempotency records.

Checkpoints restore memory, evidence, and claims into a new revision. Purged content cannot be recovered from a checkpoint created after purge.

## 19. Observability

Every run receives a trace ID. JSONL trace events include:

- run start/finish;
- node entry/exit;
- model and tool boundaries;
- state patch commits;
- context rebuilds;
- answerability decisions;
- security rejections;
- errors.

Payloads are bounded. Configurable payload redaction is implemented; production deployments SHOULD enable an appropriate redaction profile, redact secrets, and minimize raw personal data.

## 20. Security posture

### 20.1 Least privilege

Version 0.1 is read-only with respect to external systems. It does not send email, execute shell commands, alter websites, make purchases, or modify user files.

### 20.2 Prompt injection

Fetched text is evidence, never policy. Prompt-injection heuristics mark suspicious passages; markers alone are not a complete defense. The system relies primarily on trust-boundary separation, tool permissions, and deterministic control.

### 20.3 Local API

The default API binds to `127.0.0.1`. Public exposure requires explicit authentication, TLS termination, rate limiting, request-size limits, and a stronger deployment threat review.

### 20.4 Purge

Permanent purge is disabled by default, absent from the model-visible tool schema, and rejected by the local user API. Enabling an administrative purge path requires an explicit policy decision and end-to-end test coverage for privacy/reset semantics, including traces, exports, derived records, and backups.

## 21. Evaluation

### 21.1 Required baselines

A research evaluation SHOULD compare:

| Condition | State strategy |
|---|---|
| A | Full history append |
| B | Sliding window |
| C | One-shot summary |
| D | Structured state without deletion |
| E | Structured state + explicit deletion |
| F | Deletion + archive + restore |
| G | Oracle RSF |

### 21.2 Core metrics

- task success;
- next-action correctness;
- required-fact retention;
- invariant retention;
- obsolete-eviction precision and recall;
- future-utility recall;
- deletion regret;
- restore frequency;
- contradiction survival/resolution;
- active-context tokens;
- prefill latency;
- total model calls;
- citation precision and coverage;
- unsupported-answer rate.

### 21.3 Deletion regret

```text
prematurely deleted items later needed
──────────────────────────────────────
all deleted items
```

### 21.4 Future-utility recall

```text
future-needed items correctly retained
──────────────────────────────────────
all items later shown to be needed
```

### 21.5 Release evidence

No public capability claim should rely only on fixture tests. The exact Rineya checkpoint/runtime pair must be evaluated on frozen DEV and then one explicitly selected sealed candidate.

## 22. Operational requirements

A qualified release SHOULD record:

- repository commit/tag;
- configuration hash;
- model and tokenizer revision;
- runtime version;
- database schema version;
- tool schemas;
- eval-set hashes;
- hardware and OS;
- exact command;
- trace and report hashes;
- known limitations;
- rollback path.

## 23. Extensibility

Versioned interfaces exist for:

```text
ChatModel
SearchProvider
DocumentFetcher
TokenEstimator
ResearchPolicy
StateStore (future abstraction)
PassageRanker (future abstraction)
TraceRecorder (future abstraction)
```

A future MCP adapter MAY expose selected high-level tools. MCP MUST remain an integration boundary rather than forcing the model to inspect an unrestricted tool catalog.

## 24. Future adapter specialization

Workflow nodes may eventually accept:

```yaml
model_id: rineya-1.1-454m
adapter_id: rsf-reducer
adapter_scale: 0.90
context_budget: 1536
output_contract: reconstruction-v1
```

This supports future experiments on node-specific capability activation thresholds. Version 0.1 does not choose or tune adapter scales.

## 25. Version 0.1 acceptance criteria

The architecture skeleton is acceptable when:

- all schemas generate successfully;
- all tests pass on a CPU-only environment;
- coverage meets the repository threshold;
- restaurant and Hanoi demos preserve pinned invariants and replace current state;
- fixture research retrieves, source-diversifies, reconstructs, replaces and evicts raw active evidence, rebuilds context, verifies required claims, and answers with claim-linked evidence IDs;
- recent user/assistant turns retain their authority roles through context assembly and compaction;
- the model tool schema contains no persistent-purge scope;
- model attempts to create pins/global invariants or alter protected lifecycle fields fail;
- pinned deletion fails;
- dependency-breaking deletion fails;
- stale revisions fail;
- repeated idempotency keys do not duplicate state and mismatched reuse is rejected;
- ordered migrations initialize and upgrade deterministically;
- purge remains disabled by default;
- repository validation and ZIP integrity pass.

## 26. Roadmap

### v0.1.x

- exact Qwen tokenizer adapter;
- structured-output repair and deterministic fallback;
- richer source-quality/freshness policies;
- Windows runtime qualification matrix;
- cancellation propagation;
- migration interruption/recovery and N-1 compatibility fixtures.

### v0.2

- local PDFs/documents;
- hybrid lexical/vector retrieval;
- session-level event/state APIs;
- learned RSF policy training data;
- full-history/sliding-window/summary/RSF ablation runner;
- deletion-regret replay tests.

### v0.3

- optional MCP boundary;
- node-specific adapters and scales;
- model escalation only after deterministic failure evidence;
- persistent memory with explicit user controls.

### v1.0

- stable public SDK;
- authenticated service profile;
- migration compatibility guarantees;
- complete deployment hardening;
- published benchmark and reproducible research artifacts.
