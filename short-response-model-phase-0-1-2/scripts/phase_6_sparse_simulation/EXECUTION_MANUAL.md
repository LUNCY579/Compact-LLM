# Phase 6 execution manual

1. Approve Phase 5.
2. Run `.\scripts\phase_6_sparse_simulation\run_phase_6.ps1`. The calibration corpus is created automatically from non-sealed train/dev/benchmark sources.
3. Verify `router_dataset.pt` fingerprint matches the selected model.
4. Inspect per-layer sample counts and oracle contribution retention.
5. Review dense-versus-masked generations at every ratio and category.
6. Approve only a ratio that stays inside score, semantic, completeness, and critical-failure floors.
7. Record a negative result honestly. Approve the `DENSE_ONLY` path when no ratio passes; do not claim runtime gains from Phase 6.
