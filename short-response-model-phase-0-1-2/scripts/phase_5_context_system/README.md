# Phase 5 — Context retrieval and prefix reuse

Phase 5 runs real retrieval traces, bounded prompt assembly, compact-versus-full model generation, and exact stable-prefix KV-cache timing.

```powershell
.\scripts\phase_5_context_system\run_phase_5.ps1
```

Default cases are in `data/context/context_eval.jsonl`. Memory rows may include auditable retrieval-only `keywords`, `aliases`, or `topics`; only the memory `text` is placed in the prompt. Outputs include `evaluation.json`, `retrieval_traces.jsonl`, raw compact/full replies, `prefix_cache.json`, `CONTEXT_SYSTEM_REPORT.md`, and `PHASE_5_STOP_GATE.md`. See `FILE_RESPONSIBILITIES.md`.
