# Phase 5 execution manual

1. Approve Phase 4B and finalize `selected_inference_reference.json`.
2. Run `.\scripts\phase_5_context_system\run_phase_5.ps1`.
3. Inspect `retrieval_traces.jsonl` for missing required facts, stale/contradicted selections, duplicates, and overly broad memories.
4. Inspect compact/full raw replies; token reduction cannot compensate for lost correctness or context consistency.
5. Confirm prefix-cache logits are numerically consistent before trusting timing.
6. Adjust TopK and tier budgets only on development/context cases, not sealed data.
7. Approve `PHASE_5_STOP_GATE.md` only after retrieval and generation review.
