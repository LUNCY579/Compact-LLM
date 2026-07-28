# Migration from the experimental folder to Phase 4B

The former `scripts/experimental_architecture_reduction/` folder has been removed. Its functionality now lives at:

```text
scripts/phase_4b_structural_reduction/
```

## Path changes

| Old | New |
|---|---|
| `scripts/experimental_architecture_reduction/` | `scripts/phase_4b_structural_reduction/` |
| `artifacts/architecture_reduction/` | `artifacts/phase_4b_structural_reduction/` |
| `data/architecture_reduction/` | `data/phase_4b_structural_reduction/` |
| `configs/architecture_reduction_pilot.yaml` | `configs/phase_4b_structural_reduction.yaml` |

## Functional changes

- Phase 4B now requires the Phase 4 human gate.
- All candidates are evaluated through the existing Phase 1 evaluator.
- Candidate footprints and evaluations are registered in one manifest.
- A recommendation script applies practical quality floors but cannot self-approve.
- A separate finalization command writes the exact checkpoint consumed by Phases 5–8.
- Phase 5 now requires Phase 4B approval instead of advancing directly from Phase 4.

Existing local artifacts under the old path are not migrated automatically. Move or regenerate them only when their model, code revision, benchmark, and settings match the new run.
