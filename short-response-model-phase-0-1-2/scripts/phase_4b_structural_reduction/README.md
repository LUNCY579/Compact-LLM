# Phase 4B — Post-alignment structural reduction and reference selection

Phase 4B tests permanent MLP pruning, optional grouped-query-attention pruning, and depth reduction after Phase 4 has finalized aligned model weights. It always keeps the untouched Phase 4 reference as a candidate.

Phase 2B may already have selected a shallower architecture. In that case Phase 4B is a **revalidation/refinement stage**: it may keep that architecture, prune components further, or reject additional reduction. It must not discard the earlier Phase 2B lineage or compare models from incompatible profile branches.

## Normal command

```powershell
.\scripts\phase_4b_structural_reduction\run_phase_4b.ps1 -Mode Pilot
```

After review, finalize one candidate with `finalize_reference_selection.py`. This writes `artifacts/references/selected_inference_reference.json`, which Phases 5–8 consume automatically.


## Named Phase 2B branch

```powershell
.\scripts\phase_4b_structural_reduction\run_phase_4b.ps1 `
  -BranchName balanced `
  -Mode Pilot
```

The branch name automatically resolves the matching Phase 4 stop gate and
reference manifest and writes an isolated Phase 4B artifact/report tree.
