# Phase 4 execution manual

1. Confirm the Phase 3 report contains the exact line `- Human decision: PASS` (backticks around `PASS` are also accepted).
2. Verify the Phase 2/SFT adapter and accepted Phase 3 DPO adapter paths.
3. Run `run_phase_4.ps1`; do not launch each ingredient manually unless debugging.
4. On interruption, rerun with `-ReuseExisting`; the runner reuses summaries/trials only when expected files exist.
5. Inspect every ingredient and trial `generations.jsonl`, especially safety and low-score categories.
6. Review `MODEL_SOUP_REPORT.md`. The selected best individual is acceptable when merging does not help.
7. Change only the human-decision line in `PHASE_4_STOP_GATE.md` after review.
8. Keep `accepted_model` immutable; Phase 4B resolves it through the manifest.
