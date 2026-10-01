# Proposed system design

This is a design contract, not implemented functionality or measured performance.
The organizing hypothesis is explained in the [charter](../../RINEYA_CENTRALIZED_REASONING_SYSTEM.md).
Literature motivation is bounded by [SOURCES.md](SOURCES.md).

## 1. Centralize interfaces and evidence, not every model's context

A shared service should manage source access, task state, executable operations,
procedure versions, and receipts. A model receives only the currently necessary
context and structured decisions. Begin with one checkpoint used sequentially
for all model-dependent roles. "Centralized" does not mean a giant prompt,
parallel deployment of the entire family, a shared latent space, or automatic
sharing of KV caches between different checkpoints.

Proposed flow:

`source -> evidence store -> reader -> working state -> controller -> executor`

`executor -> verification -> answer with provenance OR further bounded reading`

`development traces -> candidate procedures -> counterexamples -> frozen library`

Evaluation instances may read the frozen library but must never write to it.

## 2. Evidence store: immutable source and revisable interpretations

Keep original source bytes, a document/version hash, stable passage IDs, and
unambiguous offsets. Specify the offset unit; UTF-8 byte offsets and character
offsets are not interchangeable. Token offsets also require a tokenizer hash.
An extracted event should retain its source span, entity references, event time,
negation, modality, confirmation/cancellation status, extractor identity, and
an explicit verification state.

For example, "Morgan proposed transferring parcel A to Riley" must not become
"Riley owns parcel A." Store the proposition and its status separately from
any inferred ownership. Distinguish confirmed source statements, extraction
candidates, derived claims, unresolved references, and contradictions.

Record derived claims as a dependency graph: input evidence IDs, operation,
procedure version, and intermediate results. Reopening the original passage
must remain possible. A corrected extraction invalidates dependent derived
claims and prevents stale answers from being reused. Summaries are aids, never
the sole authority for a fact that can change the answer.

The first implementation can use simple indexed records; no particular vector
database or embedding service is required. A source corpus and its index are
instance-specific and must not accidentally mix unrelated users or tasks.

## 3. Working state: a bounded, auditable task workspace

Maintain the question, relevant source references, entities, extracted events,
open dependencies, intermediate values, coverage state, contradictions, and
remaining budget. Keep the raw corpus externally accessible. Compact or spill
state by explicit policy, retaining source pointers and unresolved dependencies.
Measure external state growth rather than claiming constant total memory.

Use at least three-valued handling of evidence: supported, refuted, and unknown;
contradictory support needs its own unresolved status. "No confirmation found"
is not "no confirmation exists" unless a justified coverage condition holds.
Event time, document order, and assertion time are separate concepts.

An example working question is: who owns parcel A at 15:00? The state may know
that Morgan owned it initially and a transfer was proposed at 14:05 while still
requiring confirmation and later-cancellation evidence. The next step is to
resolve those dependencies, not guess an owner from the most recent name.

## 4. Restricted operation interface

| Operation | Contract and important failure boundary |
| --- | --- |
| `LOOKUP` | Return evidence references and explicit scope/coverage information; a top-k miss is not proof of absence. |
| `JOIN` | Connect records through validated entity/relation keys; ambiguous aliases stay unresolved. |
| `FILTER` | Apply an explicit predicate preserving time, exceptions, negation, and unknown values. |
| `TRACE_STATE` | Apply ordered, valid events under the task's supplied rules; cancellations and equal-time ambiguity need explicit semantics. |
| `AGGREGATE` | Compute over a declared population with deduplication and a coverage receipt. |
| `CHECK` | Test a candidate against available evidence and constraints; return verified, rejected, or insufficient, not a fabricated guarantee. |

These are proposed names, not existing Python functions. A later implementation
should define typed request/result schemas, deterministic error behavior,
operation/version IDs, and reproducible receipts. Use a bounded whitelist rather
than arbitrary model-generated code for v1. Reject unknown operations, malformed
arguments, invalid references, and budget overruns without silently repairing
answers from hidden labels.

Deterministic arithmetic can verify its own computation but cannot guarantee
that a model interpreted the natural-language source correctly. Report extraction
and execution validity separately. An independent reference oracle used for
scoring must be inaccessible to the model and runtime verifier.

## 5. Controller and coverage

Begin with a transparent state machine. It chooses whether to read another span,
resolve an alias, run an operation, inspect counterevidence, revisit a source,
answer, or abstain. Persist transitions and stop reasons. Impose total model-call,
input-token, output-token, operation, wall-time, and external-state limits.
Cycles and repeated uninformative retrieval must lead to a recorded stop.

Global counts, "all," and absence queries need coverage over the relevant
population. Permit an exhaustive scan or a justified complete index query;
retrieving a few promising passages cannot establish a global total. Charge
all parsing, scanning, indexing, and model-mediated extraction to cost. Do not
hide a full-document model pass behind an apparently cheap final prompt.

Readers and verifiers see source data as data. Instructions embedded in a
retrieved document cannot modify system permissions, procedure policy, split
membership, or scoring. No network access, arbitrary shell execution, secrets,
or cross-task writes are necessary for the proposed v1 executor.

## 6. Procedure library: the distinct research mechanism

A procedure is a reusable operation sequence or constrained program, not a
memorized answer. Each entry should record its ID/version, origin, typed inputs,
applicability conditions, algorithm, source/development provenance, supporting
cases, counterexamples, failure conditions, validation scope, cost, and status.

Distinguish origins explicitly: `HUMAN_SUPPLIED`, `MODEL_PROPOSED`,
`MODEL_REVISED`, and `EMPIRICALLY_SELECTED`. Human-authored primitives and a
hand-written ownership solver are supplied scaffolding, not discoveries.
Parameter learning is not required for discovering a procedure in external
memory, but external memory growth is not evidence of weight learning.

Proposed lifecycle:

`candidate -> development-tested -> counterexample-tested -> frozen for evaluation`

Candidates may instead be rejected, narrowed, or quarantined. Passing tests means
"verified under the recorded checks," not universal correctness. Failures remain
in a versioned ledger; do not overwrite an old procedure to conceal a regression.
An owner-level decision about the track is separate from rejecting an individual
procedure. No single successful answer promotes a general rule.

For example, "latest transfer determines ownership" fails on proposed but
unconfirmed transfers. Adding a confirmation condition may still fail when a
later cancellation reverses the event. Such counterexamples should change the
procedure's semantics or applicability, with the earlier failure preserved.
Task rules come from the supplied problem: never assume that every domain uses
the same cancellation or transfer convention.

## 7. Family interface and later distillation

A model adapter should bind model/config/tokenizer hashes, inference settings,
context qualification, expected request/result format, and role-specific test
receipts. Exchange source references and structured messages rather than assuming
hidden states are compatible. Initially, family membership is not evidence that
a checkpoint is a qualified reader, planner, verifier, or abstention module.

After a reproducible system gain, separately investigate training on verified
extraction decisions, evidence selection, operation choices, invalid-inference
rejection, and calibrated stopping. Preserve the central program's acquisition
and retention requirements. Test the distilled model with external procedures
disabled or deliberately restricted, plus new tasks and entities.

Do not distill unsupported explanations, test-bank answers, or unvalidated
self-generated traces. A stronger outside model used for supervision or
verification changes the experiment and requires disclosure, permission, and
full cost accounting.

## 8. Resource and failure expectations

One resident model, sequential requests, batch size one, and a proposed 2K-4K
active input/output envelope are starting design choices, not a qualified fit
claim. The source corpus can be longer, but repeated reading, external records,
indexes, verification, and procedure search all have costs. Keep GPU and host
memory separate. Preserve existing dynamic admission and numerical guards.

Diagnose failures at retrieval, extraction, reference resolution, procedure
selection, applicability checking, execution, coverage, verification, and final
answering. A precise negative at one stage is more useful than claiming that
"memory does not work" or "small models cannot reason."
