# Phase 5 — Context retrieval and prefix reuse

Phase 5 runs real retrieval traces, bounded prompt assembly, compact-versus-full model generation, and exact stable-prefix KV-cache timing.

```powershell
.\scripts\phase_5_context_system\run_phase_5.ps1
```

Default cases are in `data/context/context_eval.jsonl`. Memory rows may include auditable retrieval-only `keywords`, `aliases`, or `topics`; only the memory `text` is placed in the prompt. Outputs include `evaluation.json`, `retrieval_traces.jsonl`, raw compact/full replies, `prefix_cache.json`, `CONTEXT_SYSTEM_REPORT.md`, and `PHASE_5_STOP_GATE.md`. See `FILE_RESPONSIBILITIES.md`.

Compact and full-context generations use the same `one_sentence` response-control mode by default. This keeps the context comparison fair while evaluating the deployable output boundary; use `-ResponseControl eos_only` only for an explicitly labeled behavior-only control.

## Quality evaluator v2.0

This phase consumes or produces evaluator schema `2.0`. The main score is not a length/completion average: hard intelligibility collapse receives zero, and phase gates separately enforce tail quality, category behavior, semantic and completion floors, reference alignment, and critical/major issue limits. Evidence created by the legacy evaluator must be regenerated before this phase can automatically pass. See the repository-root `QUALITY_EVALUATION_V2_AUDIT.md` and `QUALITY_EVALUATION_V2_MIGRATION.md`.

