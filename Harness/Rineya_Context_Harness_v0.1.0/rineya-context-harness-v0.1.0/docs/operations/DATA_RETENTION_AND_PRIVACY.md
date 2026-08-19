# Data Retention and Privacy Profile

## Scope

This document distinguishes model-visible forgetting from storage retention. Recursive-state forgetting reduces active context; it is not automatically a legal or privacy deletion mechanism.

## Data classes

| Class | Examples | Default handling |
|---|---|---|
| Active state | current goal, constraints, next checkpoint | Included only while active and causally useful |
| Archived state | completed checkpoint, superseded current state | Excluded from prompts; retained for recovery/audit |
| Immutable evidence | source URI, passage, retrieval query, hashes | Retained independently of active-memory eviction |
| Claims | compact supported facts and source IDs | Retained while the run is retained |
| Trace | model/tool boundaries, state events, gate decisions | JSONL; payload redaction configurable |
| Configuration secrets | API keys, purge token | Environment/secret store only; never packaged |
| Runtime database | local SQLite state | Excluded from release archives |

## Default profile

- Local-first, single-user research runtime.
- Persistent purge disabled.
- Model-visible deletion is reversible eviction/archive.
- Trace payload redaction may be enabled; production deployments SHOULD default it on.
- Raw web evidence is treated as untrusted and should not contain credentials.
- Release packaging excludes `.env`, local non-example configuration, runtime databases, traces, caches, and coverage artifacts.

## User deletion versus RSF deletion

```text
RSF delete_memory(active_context/archive)
    = remove from active model context; preserve recoverability

administrative persistent purge
    = replace protected content with a tombstone after policy checks
```

A privacy request must not be reported as complete merely because a memory item was evicted from context. An operator must identify all applicable stores: materialized memory, evidence, claims, traces, exports, backups, and external search-provider logs.

## Purge constraints

Persistent purge is:

- absent from the model-visible tool schema;
- rejected by the local user API;
- admin-only in code;
- disabled by default;
- archive-first when configured;
- optionally protected by a confirmation token;
- blocked for pinned records and active dependencies.

A production privacy workflow requires separate end-to-end qualification, including backups and derived artifacts. Version 0.1 does not claim such qualification.

## Recommended deployment controls

- Encrypt the runtime volume where appropriate.
- Restrict filesystem permissions to the harness process/user.
- Bind the local API to loopback unless an authenticated gateway is present.
- Define trace and evidence retention periods.
- Redact or hash identifiers where exact content is not required.
- Never place provider secrets in prompts, state payloads, or trace fields.
- Document which external search/fetch providers receive queries and URLs.
- Test export and deletion behavior using synthetic sensitive records before live data.

## Incident response

On suspected leakage:

1. Stop the service and revoke affected credentials.
2. Preserve a minimal forensic copy under controlled access.
3. Identify run IDs, trace files, exports, database rows, and external providers involved.
4. Apply the approved deletion/notification policy.
5. Add a regression test without reintroducing sensitive content.
