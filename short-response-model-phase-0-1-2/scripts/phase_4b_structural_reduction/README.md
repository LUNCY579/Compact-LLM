# Phase 4B — Post-alignment structural reduction and reference selection

Phase 4B tests permanent MLP pruning, optional grouped-query-attention pruning, and depth reduction after Phase 4 has finalized aligned model weights. It always keeps the untouched Phase 4 reference as a candidate.

Phase 2B may already have selected a shallower architecture, and Phase 2C may have added a targeted SFT/DPO path on that exact architecture. In that case Phase 4B is a **revalidation/refinement stage**: it may keep that architecture, prune components further, or reject additional reduction. It must not discard the earlier Phase 2B lineage or compare models from incompatible profile branches.

## Normal command

```powershell
.\scripts\phase_4b_structural_reduction\run_phase_4b.ps1 -Mode Pilot
```

After review, finalize one candidate with `finalize_reference_selection.py`. This writes `artifacts/references/selected_inference_reference.json`, which Phases 5–8 consume automatically.


## Named Phase 2B or Phase 2C branch

```powershell
.\scripts\phase_4b_structural_reduction\run_phase_4b.ps1 `
  -BranchName balanced `
  -Mode Pilot
```

The branch name automatically resolves the matching Phase 4 stop gate and
reference manifest and writes an isolated Phase 4B artifact/report tree. For a
Phase 2C branch, use a name such as `phase2c_balanced_training`.

## Quality evaluator v3.1

This phase uses Quality Evaluator `3.1` and the shared relative-opportunity engine. Informal or mildly ungrammatical language is permitted when understandable; references are examples rather than unique answers; and behavioral restrictions are enforced only when explicitly present in the response contract. Promotion never depends on an absolute score, pass-rate, completion, category, NLL, or issue-count floor. The declared proportional resource benefit must be strictly greater than combined proportional quality opportunity loss. See `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md` and `EVALUATION_V3_MIGRATION.md`.

