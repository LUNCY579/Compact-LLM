# Phase 5 — Context retrieval and prefix reuse

Phase 5 runs real retrieval traces, bounded prompt assembly, compact-versus-full model generation, and exact stable-prefix KV-cache timing.

```powershell
.\scripts\phase_5_context_system\run_phase_5.ps1
```

Default cases are in `data/context/context_eval.jsonl`. Memory rows may include auditable retrieval-only `keywords`, `aliases`, or `topics`; only the memory `text` is placed in the prompt. Outputs include `evaluation.json`, `retrieval_traces.jsonl`, raw compact/full replies, `prefix_cache.json`, `CONTEXT_SYSTEM_REPORT.md`, and `PHASE_5_STOP_GATE.md`. See `FILE_RESPONSIBILITIES.md`.

Compact and full-context generations use the same `one_sentence` response-control mode by default. This keeps the context comparison fair while evaluating the deployable output boundary; use `-ResponseControl eos_only` only for an explicitly labeled behavior-only control.

## Quality evaluator v3.0

This phase uses Quality Evaluator `3.0` and the shared relative-opportunity engine. Informal or mildly ungrammatical language is permitted when understandable; references are examples rather than unique answers; and behavioral restrictions are enforced only when explicitly present in the response contract. Promotion never depends on an absolute score, pass-rate, completion, category, NLL, or issue-count floor. The declared proportional resource benefit must be strictly greater than combined proportional quality opportunity loss. See `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md` and `EVALUATION_V3_MIGRATION.md`.

