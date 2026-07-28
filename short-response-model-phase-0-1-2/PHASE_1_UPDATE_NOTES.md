# Phase 1 Package Update

## Changes

- Organized executable scripts into `phase_0_data`, `phase_1_dense_sft`, and
  `phase_2_layer_selection`.
- Added the complete Phase 1 PowerShell workflow.
- Added a formal automated Stop Gate 1 comparison report.
- Added a hard Phase 2 guard requiring an explicit human PASS.
- Updated nested scripts to resolve project-root paths correctly.
- Preserved the original root-level filenames as compatibility wrappers.
- Updated the project and scripts READMEs.
- Verified Python syntax with `compileall`.
- Re-ran Phase 0 schema validation, leakage checks, and audit-packet generation.

## Not performed

- No model download, CUDA inference, training, or benchmark run was performed in
  this packaging environment.
- No artifacts or model checkpoints are included.
