# Phase 4 execution manual

1. Choose one Phase 3 profile branch and confirm its exact stop-gate report has a human PASS.
2. Use `-BranchName balanced` or `-BranchName min_memory`; the runner derives the matching Phase 3 adapter, gate, and isolated output root. Explicit path overrides remain available.
3. Never mix profile architectures or reuse another branch's gate/report.
4. Run `run_phase_4.ps1`; additional seed adapters are initialized exactly like the primary branch.
5. Inspect every ingredient and trial output, especially safety and low-score categories.
6. Accept the best individual when no soup produces measurable improvement; a saturated 1.0 score does not justify inventing a difference.
7. Record the human decision before Phase 4B.
8. Keep the accepted model immutable; Phase 4B resolves it through the branch's reference manifest/output configuration.
