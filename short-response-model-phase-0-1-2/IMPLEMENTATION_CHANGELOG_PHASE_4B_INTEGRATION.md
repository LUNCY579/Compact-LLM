# Phase 4B integration changelog

## Repository organization

- Replaced `scripts/experimental_architecture_reduction/` with `scripts/phase_4b_structural_reduction/`.
- Renamed related config, data, report, artifact, and test paths to use the Phase 4B vocabulary.
- Added a Phase 4B stop-gate template and migration guide.

## Production workflow integration

- Added a Pilot/Full Phase 4B PowerShell runner.
- Added common candidate evaluation through the existing ShortChatBench evaluator.
- Added real checkpoint-footprint measurement and a candidate registry.
- Added practical threshold-based recommendation with dense fallback.
- Added human-gated reference finalization.
- Added a canonical selected-reference manifest consumed by Phases 5–8.
- Updated the common gate checker to support the `4B` label.
- Updated Phase 8 to include Phase 4B in its gate and ablation audit.

## Documentation and maintainability

- Added per-file responsibility documentation.
- Added module docstrings, operational comments, and PowerShell help/comments.
- Added a complete Phase 4B execution manual and explicit rollback rules.
- Preserved the separation between selected-layer training, permanent pruning, sparse simulation, and routed runtime.

## Verification boundary

The ZIP includes code and unit/static verification only. No model-quality, speed, VRAM, or commercial-readiness result is claimed until the user runs the pipeline with the accepted checkpoints and target hardware.
