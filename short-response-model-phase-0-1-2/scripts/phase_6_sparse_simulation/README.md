# Phase 6 — Activation profiling and masked quality frontier

Phase 6 automatically builds non-sealed calibration data, records real decoder-layer activations/oracle block labels, and runs dense-compute masked generations at 100%, 80%, 65%, 50%, and 35% active ratios.

```powershell
.\scripts\phase_6_sparse_simulation\run_phase_6.ps1
```

This phase measures quality tolerance only; it must not claim speed or VRAM savings because gate/up projections are still computed densely. When no sparse ratio passes, the gate records `DENSE_ONLY` rather than blocking the commercial pipeline. Outputs include `router_dataset.pt`, human-readable trace metadata, `frontier_with_generations.json`, raw replies, `SPARSITY_CALIBRATION_REPORT.md`, and `PHASE_6_STOP_GATE.md`. See `FILE_RESPONSIBILITIES.md`.
