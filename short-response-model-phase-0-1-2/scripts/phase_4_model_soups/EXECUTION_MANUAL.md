# Phase 4 Execution Manual

1. Choose exactly one Phase 3 branch and confirm its branch-specific stop-gate report has a human PASS.
2. Use a matching branch name:
   - Phase 2B: `balanced` or `min_memory`;
   - Phase 2C: `phase2c_maximum_quality`, `phase2c_balanced_training`, or `phase2c_minimum_training_memory`.
3. The runner derives the matching Phase 3 adapter, gate, model ID, predecessor SFT adapter, and isolated output root. Explicit overrides remain available but must describe the same branch.
4. Never mix architectures, LoRA placements, ranks, target modules, or another branch's gate/report.
5. Run `run_phase_4.ps1`; additional seed adapters use the same initialization as the primary branch.
6. Inspect every ingredient and trial output, especially safety, malformed replies, logic, and the weakest category.
7. Accept the best individual when no soup produces measurable improvement; a saturated score does not justify inventing a difference.
8. Record the human decision before Phase 4B.
9. Keep the accepted model immutable. Phase 4B resolves it through the branch-specific reference manifest.

Phase 2C example:

```powershell
.\scripts\phase_4_model_soups\run_phase_4.ps1 `
  -BranchName phase2c_balanced_training
```
