# Phase 5 — Context and Memory System

**Blueprint reference:** pages 8–10 and 23–25.

Phase 5 reduces the amount of context the selected model must process. It owns retrieval, per-tier token budgets, contradiction/staleness handling, and stable-prefix cache metadata.

## Prerequisite

Phase 4B must have:

1. an explicit human `PASS` in `reports/PHASE_4B_STOP_GATE.md`;
2. a finalized `artifacts/references/selected_inference_reference.json`.

The runner validates both before evaluating the context system.

## Step-by-step execution

1. Resolve and print the Phase 4B-selected reference.
2. Prepare structured long-term memories, recent turns, summaries, and current messages.
3. Retrieve relevant memories using relevance plus recency/staleness/contradiction checks.
4. Enforce total and per-tier token budgets.
5. Deduplicate facts and preserve a retrieval trace.
6. Measure selected-token count, relevant-memory recall, and context-assembly latency.
7. Validate stable-prefix cache keys against exact model/tokenizer/context settings.
8. Review incorrect retrievals and missing memories manually.

```powershell
.\scripts\phase_5_context_system\run_phase_5.ps1
```

## Files

- `context_system.py`: memory store, retrieval scoring, bounded context assembly, and prefix-key construction.
- `evaluate_context_system.py`: pilot evaluation and evidence output.
- `run_phase_5.ps1`: gate/reference validation and phase execution.
- `EXECUTION_MANUAL.md`: operating and recovery procedure.

## Gate

Phase 5 passes only when relevant-memory recall and context latency/token usage improve without introducing stale or contradictory facts.
