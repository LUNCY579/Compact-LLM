# Phase 2C Manual Update Checklist

1. Back up the repository and completed `artifacts/phase_2_progressive_reduction` folder.
2. Extract the hotfix ZIP over the repository root, preserving paths.
3. Do **not** replace or delete Phase 2B artifacts.
4. Confirm `reports/PHASE_2B_STOP_GATE.md` says `Human decision: PASS` and includes `balanced` or `min_memory` in `Approved profiles`.
5. Run `python -m compileall -q scripts tests`.
6. Run `python -m unittest discover -s tests -v`.
7. Run Phase 2C using `scripts/phase_2c_targeted_recovery/EXECUTION_MANUAL.md`.
8. After interruption, use the identical command with `-Resume`; changed settings require a new output root.
9. Run the Phase 2C artifact auditor after completion.
10. Review base and repeated-seed outputs and edit only the human fields in `reports/PHASE_2C_STOP_GATE.md`.
11. Continue Phase 3 with `-Phase2CProfile <approved-name>`.
12. For Phase 4, use branch name `phase2c_<approved-name>` so only compatible Phase 3 artifacts are merged.

No manual source-code edit is required when using the full ZIP or applying the hotfix at the repository root.
