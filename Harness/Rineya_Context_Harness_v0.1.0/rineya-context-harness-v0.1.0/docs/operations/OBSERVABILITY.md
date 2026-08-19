# Observability and Trace Contract

## Goals

A trace must answer:

```text
Which configuration and model ran?
What state was active?
Which tools were requested and executed?
Which evidence supported each required claim?
What was updated, archived, or evicted?
Why did the answerability gate answer, search again, clarify, or abstain?
What budget or error ended the run?
```

## Event model

Each JSONL event belongs to one `run_id` and records a connected execution story. Events include:

- run start and finish;
- workflow node entry/exit;
- model boundary;
- search/fetch/extract boundary;
- evidence persistence and reuse;
- state-patch proposal/commit/rejection;
- context rebuild and token estimate;
- answerability decision;
- security rejection;
- checkpoint/restore;
- error and terminal outcome.

Payloads are bounded. Optional trace redaction removes configured sensitive fields before serialization.

## Required correlation fields

Where applicable, events SHOULD carry:

```text
run_id
trace_id
event_id
parent_event_id
state_revision
idempotency_key
workflow_node
tool_name
model_backend
evidence_ids
claim_ids
error_code
```

## Metrics derived from traces

Core operational measurements:

- model/search/fetch/tool call counts;
- search rounds and termination reason;
- active-context token estimate by section;
- evidence records considered/retained/reused;
- state mutations by operation;
- context rebuild count;
- answerability outcomes;
- error/retry counts;
- deletion and restore counts;
- unsupported-answer and citation-linkage failures in evaluation.

Core RSF research measurements:

- obsolete-eviction precision/recall;
- future-utility recall;
- deletion regret;
- invariant retention;
- restore frequency;
- task success versus context reduction.

## Privacy and security

- Do not trace secrets or authorization headers.
- Treat user messages, evidence passages, and model payloads as potentially sensitive.
- Prefer IDs, hashes, counts, and bounded summaries when raw content is unnecessary.
- Do not expose trace endpoints publicly without authentication and tenant isolation.
- Prompt-injection text remains untrusted even when copied into a trace.

## Debugging procedure

1. Reproduce from the run export and configuration hash.
2. Inspect the first incorrect state transition, not only the final answer.
3. Separate model proposal errors from validator, store, retrieval, and orchestration failures.
4. Verify actual postconditions in SQLite and evidence records.
5. Convert the failure into a deterministic or frozen regression case.
6. Confirm the fix under the same model/runtime and budgets.

## Release evidence

A public benchmark report SHOULD publish aggregate trace-derived metrics and hashes, not private raw traces. Representative redacted traces may be included only after review.
