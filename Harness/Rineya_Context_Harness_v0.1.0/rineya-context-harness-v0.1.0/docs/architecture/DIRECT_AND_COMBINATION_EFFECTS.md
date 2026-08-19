# Direct and Combination Effects

This document defines the effects the v0.1 architecture is intended to isolate. These are design hypotheses until measured on frozen Rineya checkpoints.

## Direct effects

| Component | Intended direct effect | Primary verification |
|---|---|---|
| Mode gate | Avoid unnecessary search and tool calls | routing accuracy, unnecessary-search rate |
| Search provider | Access information outside model parameters | answer-bearing result recall |
| Safe fetcher | Retrieve source content without granting source instructions | fetch success, SSRF rejection, injection isolation |
| Extractor/ranker | Reduce page noise before model ingestion | passage precision/recall, selected-token count |
| Evidence ledger | Preserve attributable immutable passages | content-hash stability, claim-to-source coverage |
| Structured current state | Represent what is true now without transcript replay | state consistency, next-action correctness |
| `update_memory` | Replace superseded values instead of appending versions | stale-state rate, update correctness |
| `delete_memory` | Remove obsolete records from the next model-visible context | eviction precision/recall, token reduction |
| Archive/restore | Make active forgetting reversible | restore success, deletion-regret recovery |
| Invariant/dependency validator | Prevent premature or destructive deletion | invariant-loss rate, rejected-invalid-patch rate |
| Prompt rebuild barrier | Turn logical deletion into physical prompt omission | post-delete prompt inspection, token delta |
| Answerability gate | Prevent unsupported completion | unsupported-answer and abstention correctness |
| Reply controller | Preserve Rineya’s concise product behavior | median/P90 tokens, adequacy, completeness |
| Trace and event log | Make failures attributable and reproducible | trace completeness, replay/debug success |

## Combination effects

### Search × extraction × ranking

Broad retrieval becomes a small set of answer-bearing passages rather than raw webpages. Expected effect: higher evidence density and lower prefill cost.

### Reconstruction × update

A new observation produces a new current state rather than another historical paragraph. Expected effect: lower stale-state interference and more reliable next actions.

### Update × delete

The replacement is committed while the superseded state leaves active context. Expected effect: bounded state size without losing the present truth.

### Delete × archive

Obsolete material is absent from the prompt but remains recoverable. Expected effect: context savings with lower irreversible-error cost.

### Delete × dependency verification

The model proposes what appears obsolete; deterministic code checks whether any open goal, claim, citation, or active state still requires it. Expected effect: lower deletion regret and invariant loss.

### Evidence ledger × deletion

Raw passages may leave active memory after they are reduced to compact supported claims, while immutable evidence remains available by ID. Expected effect: smaller prompts without citation loss.

### Deletion × prompt/KV rebuild

Logical eviction triggers reconstruction of the dynamic prompt. Expected effect: actual context-window and prefill savings rather than a database-only status change.

### Structured state × short-response generation

Complex work is captured in state transitions; the final model call receives a small sufficient state. Expected effect: concise answers without shallow reasoning.

### Answerability × abstention

Unsupported or conflicted states route to search-more, clarification, or abstention instead of relying on model confidence. Expected effect: improved real-world abstention and groundedness.

### Tracing × evaluation × training data

A failed query, bad deletion, lost invariant, or unsupported answer becomes a reproducible regression case and later SFT/DPO example. Expected effect: a measurable improvement loop across harness and model releases.

## Required isolation order

The recommended ablation sequence is:

```text
Direct model
→ raw search/RAG
→ search + ranking
→ one-shot summary
→ structured reconstruction
→ reconstruction + explicit deletion
→ deletion + answerability
→ deletion + archive/restore
```

Each condition must use the same frozen model, decoding settings, source snapshot, task set, and tool budget. This prevents “combo” gains from being misattributed to new weights, more searches, or larger prompts.
