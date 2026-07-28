# Phase 8 — Release Candidate

**Blueprint reference:** pages 22–26.

Phase 8 evaluates and packages the exact checkpoint selected in Phase 4B, together with the context and optional sparse runtime accepted in later phases.

## Prerequisite

Phase 7 must have an explicit human `PASS`. The runner resolves the selected model from:

```text
artifacts/references/selected_inference_reference.json
```

## Step-by-step execution

1. Confirm human gates for Phases 1, 2, 3, 4, 4B, 5, 6, and 7.
2. Run the required A–I ablation matrix, including dense-vs-structural Phase 4B controls.
3. Use sealed evaluation only now and preserve immutable outputs.
4. Conduct blinded randomized human review for correctness, naturalness, usefulness, and preferred brevity.
5. Enforce category floors and zero critical safety regressions.
6. Hash the selected model, tokenizer, reference manifest, adapters/soup recipe, routers, and runtime configuration.
7. Record hardware, software, seeds, commands, latency, VRAM/RAM, and fallback rates.
8. Confirm rollback to the Phase 4 dense reference remains possible.
9. Choose dense, structurally reduced, or sparse-feature-flag deployment based on measured value.
10. Record a separate human release decision.

```powershell
.\scripts\phase_8_release_candidate\run_phase_8.ps1
```

## Files

- `build_release_audit.py`: hashes the exact local checkpoint, checks the entire gate chain including 4B, and generates the audit.
- `run_phase_8.ps1`: resolves the selected reference and invokes the audit.
- `EXECUTION_MANUAL.md`: execution, review, and recovery discipline.

No script self-approves commercial release.
