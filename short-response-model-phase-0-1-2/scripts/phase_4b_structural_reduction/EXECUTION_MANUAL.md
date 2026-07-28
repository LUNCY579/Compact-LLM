# Phase 4B execution manual

1. Approve Phase 4 and keep its accepted checkpoint immutable.
2. Start with `.\scripts\phase_4b_structural_reduction\run_phase_4b.ps1 -Mode Pilot`.
3. The runner generates matched short/long calibration data if needed, profiles components, creates guided/random MLP and depth controls, reload-tests checkpoints, evaluates all candidates, and writes a recommendation.
4. Use `-ReuseExisting` only when model/calibration/code provenance has not changed.
5. Inspect reload smoke tests, parameter/byte reduction, per-category quality, critical failures, and raw generations.
6. Expand with `-Mode Full` only when the pilot is useful. Enable attention pruning separately because GQA layouts are architecture-sensitive.
7. Approve `PHASE_4B_STOP_GATE.md`, then run `finalize_reference_selection.py`.
8. Never alter the selected checkpoint after Phase 6 profiling begins; doing so invalidates activation/router calibration.
