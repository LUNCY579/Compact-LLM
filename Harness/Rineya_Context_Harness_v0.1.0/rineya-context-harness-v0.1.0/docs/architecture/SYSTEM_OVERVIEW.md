# System Overview

## Architectural stance

The Rineya Context Harness is a **hybrid deterministic workflow**. It does not ask a small model to control every system rule. The model is used where language understanding is valuable; code is used where the correct rule is already known.

| Concern | Owner |
|---|---|
| Query semantics | Model policy |
| Evidence interpretation | Model policy |
| Compact state proposal | Model policy |
| Final natural language | Model policy |
| State revision | Code |
| Transaction commit | Code |
| Memory IDs | Code |
| Pin/dependency protection | Code |
| Purge permission | Code |
| Network policy | Code |
| Search/model budgets | Code |
| Answerability | Code |
| Trace/checkpoint | Code |

## Main components

- **Request layer** validates the input and selects a run mode.
- **Workflow** owns the bounded graph and termination conditions.
- **Search service** normalizes provider results.
- **Fetcher** retrieves only allowed read-only text resources.
- **Passage ranker** selects a compact candidate evidence set.
- **Evidence ledger** preserves attributable passages outside active context.
- **Research policy** proposes claims and current-state reconstruction.
- **State validator** rejects unsafe or inconsistent patches.
- **State store** commits one revision atomically.
- **Context assembler** rebuilds a new model prompt from active state.
- **Answerability gate** permits answer/search/abstain.
- **Trace recorder** makes execution inspectable.

## Why a single model

Version 0.1 uses one model with stage-specific contracts. This avoids:

- multi-agent coordination cost;
- context loss during handoffs;
- extra latency;
- attribution ambiguity;
- correlated state merge failures.

Specialists should only be introduced after ablations show measurable value.
